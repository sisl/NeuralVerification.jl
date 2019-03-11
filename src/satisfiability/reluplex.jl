"""
    Reluplex(optimizer, eager::Bool)

Reluplex uses binary tree search to find an activation pattern that maps a feasible input to an infeasible output.

# Problem requirement
1. Network: any depth, ReLU activation
2. Input: hyperrectangle
3. Output: PolytopeComplement

# Return
`CounterExampleResult`

# Method
Binary search of activations (0/1) and pruning by optimization.

# Property
Sound and complete.

# Reference
[G. Katz, C. Barrett, D. L. Dill, K. Julian, and M. J. Kochenderfer,
"Reluplex: An Efficient SMT Solver for Verifying Deep Neural Networks," in
*International Conference on Computer Aided Verification*, 2017.](https://arxiv.org/abs/1702.01135)
"""
@with_kw struct Reluplex
    optimizer = GLPK.Optimizer
end

function solve(solver::Reluplex, problem::Problem)
    initial_model = Model(solver)
    bs, fs = encode(solver, initial_model, problem)
    layers = problem.network.layers
    initial_status = [zeros(Int, n) for n in n_nodes.(layers)]
    insert!(initial_status, 1, zeros(Int, dim(problem.input)))

    return reluplex_step(solver, problem, initial_model, bs, fs, initial_status)
end

function find_relu_to_fix(ẑ, z)
    for i in 1:length(z), j in 1:length(z[i])
        ẑᵢⱼ = value(ẑ[i][j])
        zᵢⱼ = value(z[i][j])

        if type_one_broken(ẑᵢⱼ, zᵢⱼ) ||
           type_two_broken(ẑᵢⱼ, zᵢⱼ)
            return (i, j)
        end
    end
    return (0, 0)
end

type_one_broken(ẑᵢⱼ, zᵢⱼ) = (zᵢⱼ > 0.0)  && (zᵢⱼ != ẑᵢⱼ)  # TODO consider renaming to `inactive_broken` and `active_broken`
type_two_broken(ẑᵢⱼ, zᵢⱼ) = (zᵢⱼ == 0.0) && (ẑᵢⱼ > 0.0)

# Corresponds to a ReLU that shouldn't be active but is
function type_one_repair!(model, ẑᵢⱼ, zᵢⱼ)
    @constraint(model, ẑᵢⱼ == zᵢⱼ)
    @constraint(model, ẑᵢⱼ >= 0.0)
end
# Corresponds to a ReLU that should be active but isn't
function type_two_repair!(model, ẑᵢⱼ, zᵢⱼ)
    @constraint(model, ẑᵢⱼ <= 0.0)
    @constraint(model, zᵢⱼ == 0.0)
end

function activation_constraint!(model, ẑᵢ, zᵢ, act::ReLU)
    # ReLU ensures that the variable after activation is always
    # greater than before activation and also ≥0
    @constraint(model, zᵢ .>= ẑᵢ)
    @constraint(model, zᵢ .>= 0.0)
end

function activation_constraint!(model, ẑᵢ, zᵢ, act::Id)
    @constraint(model, zᵢ .== ẑᵢ)
end

function encode(solver::Reluplex, model::Model,  problem::Problem)
    layers = problem.network.layers
    ẑ = init_neurons(model, layers) # before activation
    z = init_neurons(model, layers) # after activation

    # Each layer has an input set constraint associated with it based on the bounds.
    # Additionally, consective variables zᵢ, ẑᵢ₊₁ are related by a constraint given
    # by the affine map encoded in the layer Lᵢ.
    # Finally, the before-activation-variables and after-activation-variables are
    # related by the activation function. Since the input layer has no activation,
    # its variables are related implicitly by identity.
    activation_constraint!(model, ẑ[1], z[1], Id())
    bounds = get_bounds(problem)
    for (i, L) in enumerate(layers)
        @constraint(model, affine_map(L, z[i]) .== ẑ[i+1])
        add_set_constraint!(model, bounds[i], ẑ[i])
        activation_constraint!(model, ẑ[i+1], z[i+1], L.activation)
    end
    add_complementary_set_constraint!(model, problem.output, last(z))
    feasibility_problem!(model)
    return ẑ, z
end

function enforce_repairs!(model::Model, ẑ, z, relu_status)
    # Need to decide what to do with last layer, this assumes there is no ReLU.
    for i in 1:length(relu_status), j in 1:length(relu_status[i])
        ẑᵢⱼ = ẑ[i][j]
        zᵢⱼ = z[i][j]
        if relu_status[i][j] == 1
            type_one_repair!(model, ẑᵢⱼ, zᵢⱼ)
        elseif relu_status[i][j] == 2
            type_two_repair!(model, ẑᵢⱼ, zᵢⱼ)
        end
    end
end

function reluplex_step(solver::Reluplex,
                       problem::Problem,
                       model::Model,
                       ẑ::Vector{Vector{VariableRef}},
                       z::Vector{Vector{VariableRef}},
                       relu_status::Vector{Vector{Int}})

    optimize!(model)

    # If the problem is optimally solved, this could potentially be a counterexample.
    # Branch by repair type (inactive or active) and redetermine if this is a valid
    # counterexample. If the problem is infeasible or unbounded. The property holds.
    if termination_status(model) == OPTIMAL
        i, j = find_relu_to_fix(ẑ, z)

        # In case no broken relus could be found, return the "input" as a counterexample
        i == 0 && return CounterExampleResult(:violated, value.(first(ẑ)))

        for repair_type in 1:2
            # Set the relu status to the current fix.
            relu_status[i][j] = repair_type
            new_m  = Model(solver)
            bs, fs = encode(solver, new_m, problem)
            enforce_repairs!(new_m, bs, fs, relu_status)

            result = reluplex_step(solver, problem, new_m, bs, fs, relu_status)

            # Reset the relu when we're done with it.
            relu_status[i][j] = 0

            result.status == :violated && return result
        end
    end
    return CounterExampleResult(:holds)
end