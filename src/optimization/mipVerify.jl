"""
    MIPVerify(optimizer)

MIPVerify computes maximum allowable disturbance using mixed integer linear programming.

# Problem requirement
1. Network: any depth, ReLU activation
2. Input: hyperrectangle
3. Output: PolytopeComplement

# Return
`AdversarialResult`

# Method
MILP encoding. Use presolve to compute a tight node-wise bounds first.
Default `optimizer` is `GLPKSolverMIP()`.

# Property
Sound and complete.

# Reference

V. Tjeng, K. Xiao, and R. Tedrake,
["Evaluating Robustness of Neural Networks with Mixed Integer Programming,"
*ArXiv Preprint ArXiv:1711.07356*, 2017.](https://arxiv.org/abs/1711.07356)

[https://github.com/vtjeng/MIPVerify.jl](https://github.com/vtjeng/MIPVerify.jl)
"""
@with_kw struct MIPVerify
    optimizer = GLPK.Optimizer
end

function solve(solver::MIPVerify, problem::Problem)
    model = Model(solver)
    neurons = init_neurons(model, problem.network)
    deltas = init_deltas(model, problem.network)
    add_complementary_set_constraint!(model, problem.output, last(neurons))
    bounds = get_bounds(problem)
    encode_network!(model, problem.network, neurons, deltas, bounds, BoundedMixedIntegerLP())
    o = max_disturbance!(model, first(neurons) - problem.input.center)
    optimize!(model)
    if termination_status(model) == INFEASIBLE
        return AdversarialResult(:holds)
    end
    if value(o) >= maximum(problem.input.radius)
        return AdversarialResult(:holds)
    else
        return AdversarialResult(:violated, value(o))
    end
end