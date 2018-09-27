#=
Read in layer from nnet file and return a Layer struct containing its weights/biases
=#
function init_layer(model::Model, i::Int64, layerSizes::Array{Int64}, f::IOStream)
     bias = Vector{Float64}(layerSizes[i+1])
     weights = Matrix{Float64}(layerSizes[i+1], layerSizes[i])
     # first read in weights
     for r = 1: layerSizes[i+1]
        line = readline(f)
        record = split(line, ",")
        token = record[1]
        c = 1
        for c = 1: layerSizes[i]
            weights[r, c] = parse(Float64, token)
            token = record[c]
        end
     end
     # now read in bias
     for r = 1: layerSizes[i+1]
        line = readline(f)
        record = split(line, ",")
        bias[r] = parse(Float64, record[1])
     end

     # activation function is set to ReLU as default
     return Layer(weights, bias, ReLU())
end

#=
Read in neural net from file and return Network struct
=#
function read_nnet(fname::String)
    f = open(fname)
    line = readline(f)
    while contains(line, "//") #skip comments
        line = readline(f)
    end

    # read in layer sizes and populate array
    record = split(line, ",")
    nLayers = parse(Int64, record[1])
    record = split(readline(f), ",")
    layerSizes = Vector{Int64}(nLayers + 1)
    for i = 1: nLayers + 1
        layerSizes[i] = parse(Int64, record[i])
    end

    # read past additonal information
    for i = 1: 5
        line = readline(f)
    end

    # initialize layers
    model = Model(solver=GLPKSolverMIP())
    layers = Vector{Layer}(nLayers)
    for i = 1:nLayers
        curr_layer = init_layer(model, i, layerSizes, f)
        layers[i] = curr_layer
    end

    return Network(layers)
end

#=
Compute output of an nnet for a given input vector
=#
function compute_output(nnet::Network, input::Vector{Float64})
    curr_value = input
    layers = nnet.layers
    for i = 1:length(layers) # layers does not include input layer (which has no weights/biases)
        curr_value = (layers[i].weights * curr_value) + layers[i].bias
        curr_value = layers[i].activation(curr_value)
    end
    return curr_value # would another name be better?
end

#=
Returns ouput of neuron j in layer i for a given input. NOTE: The was necessary
for the sampling methods, but now might not be.
=#
function compute_output(nnet::Network, input::Vector{Float64}, I, j)
    layers = nnet.layers
    @assert 0 <= i <= length(layers)         "number of layers in nnet is $(length(layers)). Got $i for number of layers to compute"
    @assert 0 <= j <= length(layers[i].bias) "number of neurons in layer is $(length(layers[i].bias)). Got $j for neuron index"
    curr_value = input
    for i = 1:I
        curr_value = (layers[i].weights * curr_value) + layers[i].bias
        curr_value = layers[i].activation(curr_value)
    end
    return curr_value[j]
end

# Given a network, find the activation pattern of all neurons at a given point x
# Assume ReLU
# return Vector{Vector{Bool}}
function get_activation(nnet::Network, x::Vector{Float64})
    act_pattern = Vector{Vector{Bool}}(length(nnet.layers))
    curr_value = x
    for (i, layer) in enumerate(nnet.layers)
        curr_value = layer.weights * curr_value + layer.bias
        act_pattern[i] = curr_value .>= 0.0
        curr_value = layer.activation(curr_value)
    end
    return act_pattern
end

# Given a network, find the activation pattern of all neurons for a set
# Assume ReLU
# 1: activated
# 0: undertermined
# -1: not activated
function get_activation(nnet::Network, input::Hyperrectangle)
    bounds = get_bounds(nnet, input)
    return get_activation(nnet, bounds)
end

function get_activation(nnet::Network, bounds::Vector{Hyperrectangle})
    act_pattern = Vector{Vector{Int}}(length(nnet.layers))
    for (i, layer) in enumerate(nnet.layers)
        before_act_bound = linear_transformation(layer, bounds[i])
        lower = low(before_act_bound)
        upper = high(before_act_bound)
        act_pattern[i] = fill(0, length(layer.bias))
        for j in 1:length(layer.bias) # For evey node
            if lower[j] > 0.0
                act_pattern[i][j] = 1
            elseif upper[j] < 0.0 
                act_pattern[i][j] = -1
            end
        end
    end
    return act_pattern
end
# Given a network, find the gradient at the input x
# Assume ReLU
function get_gradient(nnet::Network, x::Vector{Float64})
    curr_value = x
    gradient = eye(length(x))
    for (i, layer) in enumerate(nnet.layers)
        curr_value = layer.weights * curr_value + layer.bias
        act_pattern = curr_value .>= 0.0
        gradient = diagm(Vector{Float64}(act_pattern)) * layer.weights * gradient
        curr_value = layer.activation(curr_value)
    end
    return gradient
end

# Presolve to determine the bounds of variables
# This function calls maxSens to compute the bounds
# Bounds are computed AFTER activation function
# Return Vector{Hyperrectangle}
function get_bounds(nnet::Network, input::Hyperrectangle)
    solver = MaxSens()
    bounds = Vector{Hyperrectangle}(length(nnet.layers) + 1)
    bounds[1] = input
    for (i, layer) in enumerate(nnet.layers)
        bounds[i+1] = forward_layer(solver, layer, bounds[i])
    end
    return bounds
end

get_bounds(problem::Problem) = get_bounds(problem.nnet, problem.input)

function linear_transformation(layer::Layer, input::Hyperrectangle)
    (W, b, act) = (layer.weights, layer.bias, layer.activation)
    before_act_center = W * bounds[i].center + b
    before_act_radius = zeros(size(W,1))
    for j in 1:size(W, 1)
        before_act_radius[j] = sum(abs.(W[j, :]) .* bounds[i].radius)
    end
    return Hyperrectangle(before_act_center, before_act_radius)
end
