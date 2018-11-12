# Reluplex
# Minimal implementation of Reluplex

struct Reluplex end

function find_relu_to_fix(b_vars, f_vars)
    for i in 1:length(f_vars), j in 1:length(f_vars[i])
        b, f = b_vars[i+1][j], f_vars[i][j]
        if  type_one_broken(b, f) ||
            type_two_broken(b, f)

            return (i, j)
        end
    end
    return (0, 0)
end

type_one_broken(b::Real, f::Real) = (f == 0.0) && (b > 0.0) # NOTE should this be >= ?
type_two_broken(b::Real, f::Real) = (f >= 0.0) && (f != b)  # NOTE changed to >=

function type_one_repair!(m::Model, i::Int, j::Int)
    bs, fs = extract_bs_fs(m)
    type_one_repair!(m, bs[i+1][j], fs[i][j])
end
function type_two_repair!(m::Model, i::Int, j::Int)
    bs, fs = extract_bs_fs(m)
    type_two_repair!(m, bs[i+1][j], fs[i][j])
end
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
    fs = init_neurons(model, layers)  # alias can be init_forward_facing_vars
    bs = init_back_facing_vars(model, layers)

    # each hidden layer get an input constraint
    bounds = get_bounds(problem)
    for i in 1:length(bs)
        add_input_constraint(model,  bounds[i+1], bs[i])  # TODO: Chris confirm [i+1]
    end

    for (i, L) in enumerate(layers)
        (W, b, act) = (L.weights, L.bias, L.activation)

        vars = (i == 1) ? bs[i] : fs[i-1] # ternary is uglier than if?
        @constraint(model, -bs[i+1] .+  W'*vars .== -b) ## NOTE added transpose of W to make dims work.
    end

    # positivity contraint for f variables
    for i in 1:length(fs)
        @constraint(model, fs[i] .>= 0.0)
    end

    zero_objective(model)
    return nothing
end


function reluplex_step(model)
    status = solve(model)

    if status == :Infeasible
        return AdversarialResult(:UNSAT)

    elseif status == :Optimal
        b_vars, f_vars = extract_bs_fs(model)
        i, j = find_relu_to_fix(b_vars, f_vars)

        i == 0 && return AdversarialResult(:SAT, getvalue.(first(b_vars)))  # NOTE: isn't this backwards? In the SAT case we don't return values, no?

        for repair! in (type_one_repair!, type_two_repair!)
            new_m = deepcopy(model)
            repair!(new_m, i, j)
            result = reluplex_step(new_m)
            result.status == :SAT && return result
        end
    end
    # are there alternatives to the if and elseif?
end

function solve(solver::Reluplex, problem::Problem)
    basic_model = Model(solver = GLPKSolverLP(method = :Exact))
    encode(solver, basic_model, problem)

    return reluplex_step(basic_model)
end

function extract_bs_fs(m::Model)

    vars = collect(keys(m.varData))
    L1, L2 = map(length, vars)
    # "b_vars" is the longer of the two, and should be returned first
    if L1 < L2
        reverse!(vars)
    end
    return vars
end
