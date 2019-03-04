"""
    Reluplex(optimizer, eager::Bool)

Reluplex uses binary tree search to find an activation pattern that maps a feasible input to an infeasible output.

# Problem requirement
1. Network: any depth, ReLU activation
2. Input: hyperrectangle
3. Output: halfspace

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
@with_kw struct Reluplex{O<:AbstractMathProgSolver}
    optimizer::O = GLPKSolverLP(method = :Exact)
end

function solve(solver::Reluplex, problem::Problem)
    initial_model = new_model(solver)
    bs, fs = encode(solver, initial_model, problem)
    layers = problem.network.layers
    initial_status = [zeros(Int, n) for n in n_nodes.(layers)]
    insert!(initial_status, 1, zeros(Int, dim(problem.input)))

    return reluplex_step(solver, problem, initial_model, bs, fs, initial_status)
end

function find_relu_to_fix(bs, fs)
    for i in 1:length(fs), j in 1:length(fs[i])
        b = getvalue(bs[i][j])
        f = getvalue(fs[i][j])

        if type_one_broken(b, f) ||
           type_two_broken(b, f)
            return (i, j)
        end
    end
    return (0, 0)
end

type_one_broken(b, f) = (f > 0.0)  && (f != b)  # TODO consider renaming to `inactive_broken` and `active_broken`
type_two_broken(b, f) = (f == 0.0) && (b > 0.0)

# Corresponds to a ReLU that shouldn't be active but is
function type_one_repair!(model, b, f)
    @constraint(model, b == f)
    @constraint(model, b >= 0.0)
end
# Corresponds to a ReLU that should be active but isn't
function type_two_repair!(model, b, f)
    @constraint(model, b <= 0.0)
    @constraint(model, f == 0.0)
end

function activation_constraint!(model, bs, fs, act::ReLU)
    # ReLU ensures that the variable after activation is always
    # greater than before activation and also ≥0
    @constraint(model, fs .>= bs)
    @constraint(model, fs .>= 0.0)
end

function activation_constraint!(model, bs, fs, act::Id)
    @constraint(model, fs .== bs)
end

function encode(solver::Reluplex, model::Model,  problem::Problem)
    layers = problem.network.layers
    bs = init_neurons(model, layers) # before activation
    fs = init_neurons(model, layers) # after activation

    # Each layer has an input set constraint associated with it based on the bounds.
    # Additionally, consective variables fsᵢ, bsᵢ₊₁ are related by a constraint given
    # by the affine map encoded in the layer Lᵢ.
    # Finally, the before-activation-variables and after-activation-variables are
    # related by the activation function. Since the input layer has no activation,
    # its variables are related implicitly by identity.
    activation_constraint!(model, bs[1], fs[1], Id())
    bounds = get_bounds(problem)
    for (i, L) in enumerate(layers)
        @constraint(model, affine_map(L, fs[i]) .== bs[i+1])
        add_set_constraint!(model, bounds[i], bs[i])
        activation_constraint!(model, bs[i+1], fs[i+1], L.activation)
    end
    add_complementary_set_constraint!(model, problem.output, last(fs))
    zero_objective!(model)
    return bs, fs
end

function enforce_repairs!(model::Model, bs, fs, relu_status)
    # Need to decide what to do with last layer, this assumes there is no ReLU.
    for i in 1:length(relu_status), j in 1:length(relu_status[i])
        b = bs[i][j]
        f = fs[i][j]
        if relu_status[i][j] == 1
            type_one_repair!(model, b, f)
        elseif relu_status[i][j] == 2
            type_two_repair!(model, b, f)
        end
    end
end

function reluplex_step(solver::Reluplex,
                       problem::Problem,
                       model::Model,
                       b_vars::Vector{Vector{Variable}},
                       f_vars::Vector{Vector{Variable}},
                       relu_status::Vector{Vector{Int}})
    status = solve(model, suppress_warnings = true)
    if status == :Infeasible
        return CounterExampleResult(:holds)

    elseif status == :Optimal
        i, j = find_relu_to_fix(b_vars, f_vars)

        # In case no broken relus could be found, return the "input" as a counterexample
        i == 0 && return CounterExampleResult(:violated, getvalue.(first(b_vars)))

        for repair_type in 1:2
            # Set the relu status to the current fix.
            relu_status[i][j] = repair_type
            new_m  = new_model(solver)
            bs, fs = encode(solver, new_m, problem)
            enforce_repairs!(new_m, bs, fs, relu_status)

            result = reluplex_step(solver, problem, new_m, bs, fs, relu_status)

            # Reset the relu when we're done with it.
            relu_status[i][j] = 0

            result.status == :violated && return result
        end
        return CounterExampleResult(:holds)
    else
        error("unexpected status $status") # are there alternatives to the if and elseif?
    end
end

new_model(solver::Reluplex) = Model(solver = solver.optimizer)
