struct NSVerify{O<:AbstractMathProgSolver}
    optimizer::O
    m::Float64 # The big M in the linearization
end

NSVerify(x) = NSVerify(x, 1000.0)

function solve(solver::NSVerify, problem::Problem)
    model = JuMP.Model(solver = solver.optimizer)
    neurons = init_neurons(model, problem.network)
    deltas = init_deltas(model, problem.network)
    add_set_constraint!(model, problem.input, first(neurons))
    add_complementary_output_constraint!(model, problem.output, last(neurons))
    encode_mip_constraint!(model, problem.network, solver.m, neurons, deltas)
    zero_objective!(model)
    status = solve(model)
    if status == :Optimal
        return CounterExampleResult(:UNSAT, getvalue(first(neurons)))
    end
    if status == :Infeasible
        return CounterExampleResult(:SAT)
    end
    return CounterExampleResult(:Unknown)
end

"""
    NSVerify(optimizer, m::Float64)

NSVerify finds counter examples using mixed integer linear programming.

# Problem requirement
1. Network: any depth, ReLU activation
2. Input: hyperrectangle or hpolytope
3. Output: halfspace

# Return
`CounterExampleResult`

# Method
MILP encoding (using `m`). No presolve.
Default `optimizer` is `GLPKSolverMIP()`. Default `m` is `1000.0` (should be large enough to avoid approximation error).

# Property
Sound and complete.
"""
NSVerify