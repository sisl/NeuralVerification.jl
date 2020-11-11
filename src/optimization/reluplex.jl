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
@with_kw struct Reluplex <: Solver
    optimizer = GLPK.Optimizer
end

function solve(solver::Reluplex, problem::Problem)
    model = Model(solver)
    encode(solver, model, problem)
    return reluplex_step(solver, model)
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

 # TODO consider renaming to `inactive_broken` and `active_broken`
type_one_broken(ẑᵢⱼ, zᵢⱼ) = (zᵢⱼ > 0.0 + TOL[])  && (!(-TOL[] < ẑᵢⱼ - zᵢⱼ < TOL[])) # (zᵢⱼ > 0) && (ẑᵢⱼ != zᵢⱼ)
type_two_broken(ẑᵢⱼ, zᵢⱼ) = (-TOL[] < zᵢⱼ < TOL[]) && (ẑᵢⱼ > 0.0 + TOL[]) # (zᵢⱼ == 0) && (ẑᵢⱼ > 0)

# Corresponds to a ReLU that shouldn't be active but is
function type_one_repair!(model, ẑᵢⱼ, zᵢⱼ)
    con_one = @constraint(model, ẑᵢⱼ == zᵢⱼ)
    con_two = @constraint(model, ẑᵢⱼ >= 0.0)
    return con_one, con_two
end
# Corresponds to a ReLU that should be active but isn't
function type_two_repair!(model, ẑᵢⱼ, zᵢⱼ)
    con_one = @constraint(model, ẑᵢⱼ <= 0.0)
    con_two = @constraint(model, zᵢⱼ == 0.0)
    return con_one, con_two
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
    ẑ = init_vars(model, layers, :ẑ, with_input=true) # before activation
    z = init_vars(model, layers, :z, with_input=true) # after activation

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
        add_set_constraint!(model, bounds[i], z[i])
        activation_constraint!(model, ẑ[i+1], z[i+1], L.activation)
    end
    # Add the bounds on your output layer
    add_set_constraint!(model, last(bounds), last(z))
    # Add the complementary set defind as part of the problem
    add_complementary_set_constraint!(model, problem.output, last(z))
    feasibility_problem!(model)

    return nothing
end

function reluplex_step(solver::Reluplex, model::Model)

    optimize!(model)

    ẑ, z = model[:ẑ], model[:z]
    # If the problem is optimally solved, this could potentially be a counterexample.
    # Branch by repair type (inactive or active) and redetermine if this is a valid
    # counterexample. If the problem is infeasible or unbounded. The property holds.
    if termination_status(model) == OPTIMAL
        i, j = find_relu_to_fix(ẑ, z)

        # In case no broken relus could be found, return the input as a counterexample
        i == 0 && return CounterExampleResult(:violated, value(first(ẑ)))

        for repair! in (type_one_repair!, type_two_repair!)
            # Add the constraints associated with the ReLU being fixed
            new_constraints = repair!(model, ẑ[i][j], z[i][j])

            # Recurse with the ReLU i, j fixed to active or inactive
            result = reluplex_step(solver, model)

            # Return (all the way to top level) if violated.
            result.status == :violated && return result

            # Reset the model when we're done with this ReLU
            delete.(model, new_constraints)
        end
    end
    return CounterExampleResult(:holds)
end
