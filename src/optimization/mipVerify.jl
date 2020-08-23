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

function solve(solver::MIPVerify, problem::Problem; bounds=get_bounds(problem))
    model = Model(solver)
    neurons = init_neurons(model, problem.network)
    deltas = init_deltas(model, problem.network)
    add_set_constraint!(model, problem.input, first(neurons))
    add_complementary_set_constraint!(model, problem.output, last(neurons))
    encode_network!(model, problem.network, neurons, deltas, bounds, BoundedMixedIntegerLP())
    o = max_disturbance!(model, first(neurons) - problem.input.center)
    optimize!(model)
    if termination_status(model) == INFEASIBLE
        return AdversarialResult(:holds)
    end
    return AdversarialResult(:violated, value(o))
end

function get_bounds_for_node(solver::MIPVerify, network::Network, input::Hyperrectangle, layer_index::Int, node_index::Int; pre_activation::Bool = true, bounds=get_bounds(network, input))
    model = Model(solver)
    # Truncate the network. we just encode the earliest layer_index layers
    # layer_index = 1 corresponds to the input layer.
    network = Network(network.layers[1:layer_index-1])
    neurons = init_neurons(model, network)
    deltas = init_deltas(model, network)
    add_set_constraint!(model, input, first(neurons))
    encode_network!(model, network, neurons, deltas, bounds, BoundedMixedIntegerLP())

    # Define the objective
    if (pre_activation)
        objective = dot(network.layers[layer_index-1].weights[node_index, :], neurons[layer_index-1]) + network.layers[layer_index-1].bias[node_index]
    else
        objective = neurons[layer_index][node_index]
    end
    # Find the lower bound
    @objective(model, Min, objective)
    optimize!(model)
    lower = value(objective)

    # Find the upper bound
    @objective(model, Max, objective)
    optimize!(model)
    upper = value(objective)

    return lower, upper
end

# Return a list of Hyperrectangles corresponding to the true bounds on
# each node
function get_bounds(solver::MIPVerify, network::Network, input::Hyperrectangle; pre_activation::Bool = false)
    println("UNIMPLEMENTED")
end
