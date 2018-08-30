include("utils/reachability.jl")

import LazySets.Hyperrectangle

struct MaxSens <: Reachability
    resolution::Float64
end

# This is the main function
function solve(solver::MaxSens, problem::Problem)
    inputs = partition(problem.input, solver.resolution)
    outputs = Vector{Hyperrectangle}(length(inputs))
    for i in 1:length(inputs)
        outputs[i] = forward_network(solver, problem.network, inputs[i])
    end
    return checkInclusion(outputs, problem.output)
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
    #println("forwardNode ", node, " ", output, " ", betaMax, " ", betaMin)
    return max(abs(node.act(betaMax) - node.act(output)), abs(node.act(betaMin) - node.act(output)))
end

function partition(input::Constraints, delta::Float64)
    n_dim = length(input.upper)
    hyperrectangle_list = Vector{Int64}(n_dim)
    n_hyperrectangle = 1
    for i in 1:n_dim
        hyperrectangle_list[i] = n_hyperrectangle
        n_hyperrectangle *= ceil((input.upper[i]-input.lower[i])/delta)
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
            center[i] = input.lower[i] + delta/2 + delta * id;
            radius[i] = delta;
        end
        hyperrectangles[k] = Hyperrectangle(center, radius)
    end
    return hyperrectangles
end

function checkInclusion(hyperrectangles::Vector{Hyperrectangle}, set::Constraints)
    n_vertex = 2^(length(hyperrectangles[1].center))
    for i in 1:length(hyperrectangles)
        x = hyperrectangles[i].center
        delta = hyperrectangles[i].radius 
        for k in 1:n_vertex
            binary = bin(k - 1, length(hyperrectangles[i].center))
            for j in 1:length(hyperrectangles[i].center)
                x[j] += ifelse(binary[j] == '1', 1, -1) * delta[j]
            end
            if maximum(set.A * x - set.b)>0 || maximum(x - set.upper) > 0 || maximum(x - set.lower) < 0
                return false
            end
        end
    end
    return true
end