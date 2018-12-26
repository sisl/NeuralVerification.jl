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
G. Katz, C. Barrett, D. L. Dill, K. Julian, and M. J. Kochenderfer,
"Reluplex: An Efficient SMT Solver for Verifying Deep Neural Networks," in
*International Conference on Computer Aided Verification*, 2017.
"""
struct Reluplex end

function solve(solver::Reluplex, problem::Problem)
    basic_model = new_model(solver)
    bs, fs = encode(solver, basic_model, problem)
    layers = problem.network.layers
    initial_status = [zeros(Int, n) for n in n_nodes.(layers)]

    return reluplex_step(solver, basic_model, bs, fs, initial_status)
end

function find_relu_to_fix(b_vars, f_vars)
    for i in 1:length(f_vars), j in 1:length(f_vars[i])
        b = getvalue(b_vars[i+1][j])
        f = getvalue(f_vars[i][j])

        if type_one_broken(b, f) ||
           type_two_broken(b, f)
            (i, j)
        end
    end
    return (0, 0)
end

type_one_broken(b::Real, f::Real) = (f >= 0.0) && (f != b)
type_two_broken(b::Real, f::Real) = (f == 0.0) && (b > 0.0)

# NOTE that the b that is passed in should be: bs[i+1] relative to fs[i]
function type_one_repair!(m::Model, b::Variable, f::Variable)
    @constraint(m, b == f)
    @constraint(m, b >= 0.0)
    return nothing
end
function type_two_repair!(m::Model, b::Variable, f::Variable)
    @constraint(m, b <= 0.0)
    @constraint(m, f == 0.0)
    return nothing
end

function encode(solver::Reluplex, model::Model,  problem::Problem)
    layers = problem.network.layers
    bs = init_neurons(model, layers)
    fs = init_forward_facing_vars(model, layers)

    # Positivity contraint for forward-facing variables
    for i in 1:length(fs)
        @constraint(model, fs[i] .>= 0.0)
    end

    # All layers (input and hidden) get an "input" constraint
    # In addition, each set of back-facing and forward-facing
    # variables are related to each other by a constraint.
    bounds = get_bounds(problem)
    for (i, L) in enumerate(layers)
        ## layerwise input constraint
        add_set_constraint!(model, bounds[i], bs[i])

        ## b<——>f[next] constraint
        # first layer technically has only vars which are forward facing,
        # but they behave like b-vars, so they are treated as such.
        W, bias = L.weights, L.bias
        if (i == 1)
            @constraint(model, -bs[i+1] .+ W*bs[i] .== -bias)
        else
            @constraint(model, -bs[i+1] .+ W*fs[i-1] .== -bias)
        end

        ## f >= b always, by definition
        if i <= length(fs)
            @constraint(model, fs[i] .>= bs[i+1])
        end
    end
    add_set_constraint!(model, problem.output, last(bs))

    zero_objective!(model)

    return bs, fs
end

function enforce_repairs!(model::Model, bs, fs, relu_status)
    for i in 1:length(relu_status), j in 1:length(relu_status[i])
        b = bs[i+1][j]
        f = fs[i][j]
        if relu_status[i][j] == 1
            type_one_repair(m, b, f)
        elseif relu_status[i][j] == 2
            type_two_repair(m, b, f)
        end
    end
end

function reluplex_step(solver::Reluplex,
                       model::Model,
                       b_vars::Vector{Vector{Variable}},
                       f_vars::Vector{Vector{Variable}},
                       relu_status::Vector{Vector{Int}})

    status = solve(model)

    if status == :Infeasible
        return CounterExampleResult(:SAT)

    elseif status == :Optimal
        i, j = find_relu_to_fix(b_vars, f_vars)

        # in case no broken relus could be found, return the "input" as a countereexample
        i == 0 && return CounterExampleResult(:UNSAT, getvalue.(first(b_vars)))

        for repair_type in 1:2
            relu_status[i][j] = repair_type

            new_m  = new_model(solver)
            bs, fs = encode(solver, new_m, problem)
            enforce_repairs!(new_m, bs, fs, relu_status)

            result = reluplex_step(solver, new_m, bs, fs, relu_status)

            relu_status[i][j] = 0
            result.status == :UNSAT && return result
        end
    else
        error("unexpected status $status") # are there alternatives to the if and elseif?
    end
end

# for convenience:
new_model(::Reluplex) = Model(solver = GLPKSolverLP(method = :Exact))


# doesn't do what it should! TODO open feature request issue on JuMP
# function extract_bs_fs(m::Model)
#     vars = collect(keys(m.varData))
#     L1, L2 = map(length, vars)
#     # "b_vars" is the longer of the two, and should be returned first
#     if L1 < L2
#         reverse!(vars)
#     end
#     return vars
# end
