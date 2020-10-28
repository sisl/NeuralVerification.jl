"""
    NSVerify(optimizer, m::Float64)

NSVerify finds counter examples using mixed integer linear programming.

# Problem requirement
1. Network: any depth, ReLU activation
2. Input: hyperrectangle or hpolytope
3. Output: PolytopeComplement

# Return
`CounterExampleResult`

# Method
MILP encoding (using `m`). No presolve.
Default `optimizer` is `GLPKSolverMIP()`. Default `m` is `1000.0` (should be large enough to avoid approximation error).

# Property
Sound and complete.

# Reference
[A. Lomuscio and L. Maganti,
"An Approach to Reachability Analysis for Feed-Forward Relu Neural Networks,"
*ArXiv Preprint ArXiv:1706.07351*, 2017.](https://arxiv.org/abs/1706.07351)
"""
@with_kw struct NSVerify <: Solver
    optimizer = GLPK.Optimizer
    m = nothing
end

function solve(solver::NSVerify, problem::Problem)
    # Set up and solve the problem
    model = Model(solver)
    z = init_vars(model, problem.network, :z, with_input=true)
    δ = init_vars(model, problem.network, :δ, binary=true)
    # Set M automatically if not already set.
    if isnothing(solver.m)
        model[:M] = set_automatic_M(problem)
    else
        @warn "M should be chosen carefully. An M which is too small will cause
        the problem to be solved incorrectly. Not setting the `m` keyword, or setting
        it equal to `nothing` will cause a safe value to be calculated automatically.
        E.g. NSVerify()." maxlog = 1
        model[:M] = solver.m
    end

    add_set_constraint!(model, problem.input, first(z))
    add_complementary_set_constraint!(model, problem.output, last(z))
    encode_network!(model, problem.network, MixedIntegerLP())
    feasibility_problem!(model)
    optimize!(model)

    if termination_status(model) == OPTIMAL
        return CounterExampleResult(:violated, value(first(z)))
    elseif termination_status(model) == INFEASIBLE
        return CounterExampleResult(:holds)
    else
        return CounterExampleResult(:unknown)
    end
end

function set_automatic_M(problem)
    # Compute the largest pre-activation bound absolute value.
    # M must be larger than any value a variable in the problem
    # can take.
    bounds = get_bounds(problem; before_act = true)
    M = maximum(abs, Iterators.flatten(abs.(hr.center) + hr.radius for hr in bounds))
end
