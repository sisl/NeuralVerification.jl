"""
    Sherlock(optimizer, ϵ::Float64)

Sherlock combines local and global search to estimate the range of the output node.

# Problem requirement
1. Network: any depth, ReLU activation, single output
2. Input: hpolytope and hyperrectangle
3. Output: hyperrectangle (1d interval)

# Return
`CounterExampleResult` or `ReachabilityResult`

# Method
Local search: solve a linear program to find local optima on a line segment of the piece-wise linear network.
Global search: solve a feasibilty problem using MILP encoding (default calling NSVerify).
- `optimizer` default `GLPKSolverMIP()`
- `ϵ` is the margin for global search, default `0.1`.

# Property
Sound but not complete.

# Reference
[S. Dutta, S. Jha, S. Sanakaranarayanan, and A. Tiwari,
"Output Range Analysis for Deep Neural Networks,"
*ArXiv Preprint ArXiv:1709.09130*, 2017.](https://arxiv.org/abs/1709.09130)

[https://github.com/souradeep-111/sherlock](https://github.com/souradeep-111/sherlock)
"""
@with_kw struct Sherlock <: Solver
    optimizer = GLPK.Optimizer
    ϵ::Float64 = 0.1
end

function solve(solver::Sherlock, problem::Problem)
    isbounded(problem.input) || UnboundedInputError("Temprarily, Sherlock cannot accept unbounded inputs")

    (x_u, u) = output_bound(solver, problem, :max)
    (x_l, l) = output_bound(solver, problem, :min)
    println("bounds: [", l, ", ", u, "]")
    bound = Hyperrectangle(low = [l], high = [u])
    reach = Hyperrectangle(low = [l - solver.ϵ], high = [u + solver.ϵ])

    output = problem.output
    if reach ⊆ output
        return ReachabilityResult(:holds, [reach])
    end
    high(bound) > high(output) && return CounterExampleResult(:violated, x_u)
    low(bound)  < low(output)  && return CounterExampleResult(:violated, x_l)
    return ReachabilityResult(:unknown, [reach])
end

function output_bound(solver::Sherlock, problem::Problem, type::Symbol)
    opt = solver.optimizer
    x = an_element(problem.input)
    while true
        (x, bound) = local_search(problem, x, opt, type)
        bound_ϵ = bound + ifelse(type == :max, solver.ϵ, -solver.ϵ)
        (x_new, bound_new, feasible) = global_search(problem, bound_ϵ, opt, type)
        feasible || return (x, bound)
        (x, bound) = (x_new, bound_new)
    end
end


function local_search(problem::Problem, x::Vector{Float64}, optimizer, type::Symbol)
    nnet = problem.network

    model = Model(optimizer)
    model[:δ] = get_activation(nnet, x)
    z = init_vars(model, nnet, :z, with_input=true)
    add_set_constraint!(model, problem.input, first(z))
    encode_network!(model, nnet, StandardLP())

    # the gradient wrt to the input is a `1xn` matrix,
    # because the output is 1-d. Therefore o[1] is the
    # only entry in g*z₀
    gradient = get_gradient(nnet, x)
    o = gradient * first(z)
    index = ifelse(type == :max, 1, -1)
    @objective(model, Max, index * o[1])
    optimize!(model)

    x_new = value(first(z))
    bound_new = compute_output(nnet, x_new)
    return (x_new, bound_new[1])
end

function global_search(problem::Problem, bound::Float64, optimizer, type::Symbol)
    index = ifelse(type == :max, 1.0, -1.0)
    h = HalfSpace([index], index * bound)
    output_set = HPolytope([h])
    problem_new = Problem(problem.network, problem.input, output_set)
    solver  = NSVerify(optimizer = optimizer)
    result  = solve(solver, problem_new)
    if result.status == :violated
        x = result.counter_example
        bound = compute_output(problem.network, x)
        return (x, bound[1], true)
    else
        return ([], 0.0, false)
    end
end
