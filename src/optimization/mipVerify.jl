"""
    MIPVerify(optimizer)

MIPVerify computes maximum allowable disturbance using mixed integer linear programming.

# Problem requirement
1. Network: any depth, ReLU activation
2. Input: hyperrectangle
3. Output: halfspace

# Return
`AdversarialResult`

# Method
MILP encoding. Use presolve to compute a tight node-wise bounds first.
Default `optimizer` is `GLPKSolverMIP()`.

# Property
Sound and complete.

# Reference

V. Tjeng, K. Xiao, and R. Tedrake,
"Evaluating Robustness of Neural Networks with Mixed Integer Programming,"
*ArXiv Preprint ArXiv:1711.07356*, 2017.
"""
@with_kw struct MIPVerify{O<:AbstractMathProgSolver}
    optimizer::O
end

function solve(solver::MIPVerify, problem::Problem)
    model = Model(solver = solver.optimizer)
    neurons = init_neurons(model, problem.network)
    deltas = init_deltas(model, problem.network)
    add_complementary_set_constraint!(model, problem.output, last(neurons))
    bounds = get_bounds(problem)
    encode_mip_constraint!(model, problem.network, bounds, neurons, deltas)
    J = max_disturbance!(model, first(neurons) - problem.input.center)
    status = solve(model, suppress_warnings = true)
    if status == :Infeasible
        return AdversarialResult(:SAT)
    end
    if getvalue(J) >= minimum(problem.input.radius)
        return AdversarialResult(:SAT)
    else
        return AdversarialResult(:UNSAT, getvalue(J))
    end
end