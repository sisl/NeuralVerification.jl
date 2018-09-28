# Sherlock
# Input constraint HPolytope
# Output: 1D Hyperrectangle
struct Sherlock
    global_solver::AbstractMathProgSolver
    delta::Float64
end

function solve(solver::Sherlock, problem::Problem)
    (x_u, u) = output_bound(solver, problem, true) # true for upper bound, false for lower bound
    (x_l, l) = output_bound(solver, problem, false)

    uh = u - high(problem.output)[1]
    ll = l - low(problem.output)[1]

    uh <= 0 && ll >= 0     && return Result(:SAT)
    uh > solver.delta      && return Result(:UNSAT, x_u)
    ll < solver.delta      && return Result(:UNSAT, x_l)

    return Result(:UNKNOWN)
end

function output_bound(solver::Sherlock, problem::Problem, upper::Bool)
    x = sample(problem.input)
    while true
        (x, bound) = local_search(problem.network, x, problem.input, solver.global_solver, upper)
        bound += ifelse(upper, solver.delta, -solver.delta)
        (x_new, bound_new, feasibile) = global_search(problem.network, bound, problem.input, solver.global_solver, upper)
        if feasibile
            (x, bound) = (x_new, bound_new)
        else
            return (x, bound)
        end
    end
end

# Choose the first vertex
function sample(set::AbstractPolytope)
    x = vertices_list(set)
    return x[1]
end

function local_search(nnet::Network, x::Vector{Float64}, inputSet::AbstractPolytope, optimizer::AbstractMathProgSolver, upper::Bool)
    act_pattern = get_activation(nnet, x)
    gradient = get_gradient(nnet, x)

    model = JuMP.Model(solver = optimizer)

    neurons = init_nnet_vars(solver, model, problem.network)
    add_input_constraint(model, problem.input, first(neurons))

    for (i, layer) in enumerate(problem.network.layers)
        (W, b, act) = (layer.weights, layer.bias, layer.activation)
        before_act = W * neurons[i] + b
        for j in 1:length(layer.bias) # For evey node
            if act_pattern[i][j]
                @constraint(model, before_act[j] >= 0.0)
                @constraint(model, neurons[i+1][j] == before_act[j])
            else
                @constraint(model, before_act[j] <= 0.0)
                @constraint(model, neurons[i+1][j] == 0.0)
            end
        end
    end

    J = gradient * neurons[1]
    if upper
        @objective(model, Max, J[1])
    else
        @objective(model, Min, J[1])
    end

    JuMP.solve(model)

    x_new = getvalue(neurons[1])
    bound_new = compute_output(nnet, x_new)
    return (x_new, bound_new[1])
end

function global_search(nnet::Network, bound::Float64, inputSet::AbstractPolytope, optimizer::AbstractMathProgSolver, upper::Bool)
    # Call Reverify for global search
    if (upper)    h = HalfSpace([1.0], bound)
    else          h = HalfSpace([-1.0], -bound)
    end
    outputSet = HPolytope([h])

    problem = Problem(nnet, inputSet, outputSet)
    solver  = Reverify(optimizer)
    result  = solve(solver, problem)
    if result.status == :SAT
        x = result.counter_example
        bound = compute_output(nnet, x)
        return (x, bound[1], true)
    else
        return ([], 0.0, false)
    end
end

function init_nnet_vars(solver::Sherlock, model::Model, network::Network)
    layers = network.layers
    neurons = Vector{Vector{Variable}}(length(layers) + 1) # +1 for input layer
    # input layer is treated differently from other layers
    input_layer_n = size(first(layers).weights, 2)
    all_layers_n  = n_nodes.(layers)
    prepend!(all_layers_n, input_layer_n)
    for (i, n) in enumerate(all_layers_n)
        neurons[i] = @variable(model, [1:n])
    end
    return neurons
end