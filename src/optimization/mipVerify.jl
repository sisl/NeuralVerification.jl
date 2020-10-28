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
@with_kw struct MIPVerify <: Solver
    optimizer = GLPK.Optimizer
end

function solve(solver::MIPVerify, problem::Problem)
    model = Model(solver)
    z = init_vars(model, problem.network, :z, with_input=true)
    δ = init_vars(model, problem.network, :δ, binary=true)
    # get the pre-activation bounds:
    model[:bounds] = get_bounds(problem, before_act=true)
    model[:before_act] = true

    add_set_constraint!(model, problem.input, first(z))
    add_complementary_set_constraint!(model, problem.output, last(z))
    encode_network!(model, problem.network, BoundedMixedIntegerLP())
    o = max_disturbance!(model, first(z) - problem.input.center)
    optimize!(model)
    if termination_status(model) == OPTIMAL
        return AdversarialResult(:violated, value(o))
    end
    return AdversarialResult(:holds)
end
