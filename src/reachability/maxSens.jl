import LazySets.Hyperrectangle

struct MaxSens
    resolution::Float64
end

# This is the main function
function solve(solver::MaxSens, problem::Problem)
    inputs = partition(problem.input, solver.resolution)
    outputs = Vector{Hyperrectangle}(length(inputs))
    for i in 1:length(inputs)
        outputs[i] = forward_network(solver, problem.network, inputs[i])
    end
    return check_inclusion(outputs, problem.output)
end

# This function is called by forward_network
function forward_layer(solver::MaxSens, layer::Layer, input::Hyperrectangle)
    gamma = Vector{Float64}(size(layer.weights, 1))
    for j in 1:size(layer.weights, 1)
        node = Node(layer.weights[j,:], layer.bias[j], layer.activation)
        gamma[j] = forward_node(solver, node, input)
    end
    output = layer.activation(layer.weights * input.center + layer.bias)

    # Here we do not require the Hyperrectangle to have equal sides
    output = Hyperrectangle(output, gamma)
    return output
end

function forward_node(solver::MaxSens, node::Node, input::Hyperrectangle)
    output = node.w' * input.center + node.b
    deviation = sum(abs(node.w[i])*input.radius[i] for i in 1:length(input.center))
    betaMax = output + deviation
    betaMin = output - deviation
    return max(abs(node.act(betaMax) - node.act(output)), abs(node.act(betaMin) - node.act(output)))
end

# This function needs to be improved
# Ad hoc implementation for now
# Assuming the constraint only contains lower and upper bounds
# [I; -I] x <= [Upper; Lower]
function partition(input::HPolytope, delta::Float64)
    n_dim = dim(input)
    hyperrectangle_list = Vector{Int64}(n_dim)
    n_hyperrectangle = 1

    # This part is ad hoc
    inputA, inputb = tosimplehrep(input)
    upper = inputb[1:n_dim]
    lower = -inputb[(n_dim+1):(2*n_dim)]

    for i in 1:n_dim
        hyperrectangle_list[i] = n_hyperrectangle
        n_hyperrectangle *= ceil((upper[i] - lower[i])/delta)
    end
    n_hyperrectangle = trunc(Int, n_hyperrectangle)

    hyperrectangles = Vector{Hyperrectangle}(n_hyperrectangle)
    for k in 1:n_hyperrectangle
        number = k
        center = Vector{Float64}(n_dim)
        radius = Vector{Float64}(n_dim)
        for i in n_dim:-1:1
            id = div(number-1, hyperrectangle_list[i])
            number = mod(number-1, hyperrectangle_list[i])+1
            center[i] = lower[i] + delta/2 + delta * id;
            radius[i] = delta;
        end
        hyperrectangles[k] = Hyperrectangle(center, radius)
    end
    return hyperrectangles
end