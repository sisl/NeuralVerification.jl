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
@with_kw struct NSVerify
    optimizer = GLPK.Optimizer
    m::Float64 = 1000.0  # The big M in the linearization
end

function solve(solver::NSVerify, problem::Problem)
    model = Model(solver)
    neurons = init_neurons(model, problem.network)
    deltas = init_deltas(model, problem.network)
    add_set_constraint!(model, problem.input, first(neurons))
    add_complementary_set_constraint!(model, problem.output, last(neurons))
    encode_network!(model, problem.network, neurons, deltas, MixedIntegerLP(solver.m))
    feasibility_problem!(model)
    optimize!(model)
    if termination_status(model) == OPTIMAL
        return CounterExampleResult(:violated, value.(first(neurons)))
    end
    if termination_status(model) == INFEASIBLE
        return CounterExampleResult(:holds)
    end
    return CounterExampleResult(:unknown)
end
