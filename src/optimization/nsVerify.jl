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
    # Set M automatically if not already set.
    M = solver.m == nothing ? set_automatic_M(problem) : solver.m

    @show M

    # Set up and solve the problem
    model = Model(solver)
    neurons = init_neurons(model, problem.network)
    deltas = init_deltas(model, problem.network)
    add_set_constraint!(model, problem.input, first(neurons))
    add_complementary_set_constraint!(model, problem.output, last(neurons))
    encode_network!(model, problem.network, neurons, deltas, MixedIntegerLP(M))
    # encode_network!(model, problem.network, neurons, deltas, get_bounds(problem), BoundedMixedIntegerLP())
    feasibility_problem!(model)
    optimize!(model)


   #  @show model

   #  for T in list_of_constraint_types(model)
   #     cs = all_constraints(model, T...)
   #     println(T)
   #     println.("\t", cs)
   # end


    if termination_status(model) == OPTIMAL
        # Issue a warning if M is not sufficiently large.
        # M must be larger than any other variable in the problem.
        # @show value.(neurons)
        @show max_z = mapreduce(x->norm(x, Inf), max, value.(neurons))
        @show M
        if max_z >= M
            @warn "M not sufficiently large. Problem may have been solved incorrectly.
            The minimum viable value of M was found to be $max_z"
        end

        return CounterExampleResult(:violated, value.(first(neurons)))

    elseif termination_status(model) == INFEASIBLE
        return CounterExampleResult(:holds)
    else
        return CounterExampleResult(:unknown)
    end
end

function set_automatic_M(problem)
    # new_nnet = Network([Layer(L.weights, L.bias, Id()) for L in problem.network.layers])
    # new_problem = Problem(new_nnet, problem.input, problem.output)

    flat = Iterators.flatten
    # bounds = get_bounds(new_problem)
    bounds = get_bounds(problem)
    M = maximum(abs, flat(hr.center + hr.radius for hr in bounds))
    # add 1% margin?
    # M *= 1.1
end