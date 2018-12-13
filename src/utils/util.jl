"""
    read_nnet(fname::String)

Read in neural net from file and return Network struct
"""
function read_nnet(fname::String)
    f = open(fname)
    line = readline(f)
    while occursin("//", line) #skip comments
        line = readline(f)
    end
    # read in layer sizes and populate array
    record = split(line, ",")
    nLayers = parse(Int64, record[1])
    record = split(readline(f), ",")
    layerSizes = Vector{Int64}(undef, nLayers + 1)
    for i = 1: nLayers + 1
        layerSizes[i] = parse(Int64, record[i])
    end
    # read past additonal information
    for i = 1: 5
        line = readline(f)
    end
    # initialize layers
    layers = Vector{Layer}(undef, nLayers)
    for i = 1:nLayers
        curr_layer = init_layer(i, layerSizes, f)
        layers[i] = curr_layer
    end

    return Network(layers)
end

"""
    init_layer(i::Int64, layerSizes::Array{Int64}, f::IOStream)

Read in layer from nnet file and return a Layer struct containing its weights/biases
"""
function init_layer(i::Int64, layerSizes::Array{Int64}, f::IOStream)
     bias = Vector{Float64}(undef, layerSizes[i+1])
     weights = Matrix{Float64}(undef, layerSizes[i+1], layerSizes[i])
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

"""
    compute_output(nnet::Network, input::Vector{Float64})

Compute output of an nnet for a given input vector
"""
function compute_output(nnet::Network, input::Vector{Float64})
    curr_value = input
    layers = nnet.layers
    for i = 1:length(layers) # layers does not include input layer (which has no weights/biases)
        curr_value = (layers[i].weights * curr_value) + layers[i].bias
        curr_value = layers[i].activation(curr_value)
    end
    return curr_value # would another name be better?
end

"""
    compute_output(nnet::Network, input::Vector{Float64}, i, j)

Returns ouput of neuron j in layer i for a given input. 
NOTE: The was necessary for the sampling methods, but now might not be.
"""
function compute_output(nnet::Network, input::Vector{Float64}, i, j)
    layers = nnet.layers
    @assert 0 <= i <= length(layers)         "number of layers in nnet is $(length(layers)). Got $i for number of layers to compute"
    @assert 0 <= j <= length(layers[i].bias) "number of neurons in layer is $(length(layers[i].bias)). Got $j for neuron index"
    curr_value = input
    for m = 1:i
        curr_value = (layers[m].weights * curr_value) + layers[m].bias
        curr_value = layers[m].activation(curr_value)
    end
    return curr_value[j]
end

"""
    get_activation(nnet::Network, x::Vector{Float64})

Given a network, find the activation pattern of all neurons at a given point x.
Assume ReLU.
return Vector{Vector{Bool}}.
"""
function get_activation(nnet::Network, x::Vector{Float64})
    act_pattern = Vector{Vector{Bool}}(undef, length(nnet.layers))
    curr_value = x
    for (i, layer) in enumerate(nnet.layers)
        curr_value = layer.weights * curr_value + layer.bias
        act_pattern[i] = curr_value .>= 0.0
        curr_value = layer.activation(curr_value)
    end
    return act_pattern
end

"""
    get_activation(nnet::Network, input::Hyperrectangle)

Given a network, find the activation pattern of all neurons for a given input set.
Assume ReLU.
return Vector{Vector{Int64}}.
- 1: activated
- 0: undertermined
- -1: not activated
"""
function get_activation(nnet::Network, input::Hyperrectangle)
    bounds = get_bounds(nnet, input)
    return get_activation(nnet, bounds)
end

"""
    get_activation(nnet::Network, bounds::Vector{Hyperrectangle})

Given a network, find the activation pattern of all neurons given the node-wise bounds.
Assume ReLU.
return Vector{Vector{Int64}}.
- 1: activated
- 0: undertermined
- -1: not activated
"""
function get_activation(nnet::Network, bounds::Vector{Hyperrectangle})
    act_pattern = Vector{Vector{Int}}(undef, length(nnet.layers))
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

"""
    get_gradient(nnet::Network, x::Vector{N}) where N

Given a network, find the gradient at the input x
"""
function get_gradient(nnet::Network, x::Vector{N}) where N
    z = x
    gradient = Matrix(1.0I, length(x), length(x))
    for (i, layer) in enumerate(nnet.layers)
        z_hat = layer.weights * z + layer.bias
        σ_gradient = act_gradient(layer.activation, z_hat)
        gradient = Diagonal(σ_gradient) * layer.weights * gradient
        z = layer.activation(z_hat)
    end
    return gradient
end

"""
    act_gradient(act::ReLU, z_hat::Vector{N}) where N

Computing the gradient of an activation function at point z_hat.
Currently only support ReLU.
"""
function act_gradient(act::ReLU, z_hat::Vector{N}) where N
    return z_hat .>= 0.0
end

"""
    get_gradient(nnet::Network, input::AbstractPolytope)

Get lower and upper bounds on network gradient for a given input set.
Return:
- `LG::Vector{Matrix}`: lower bounds
- `UG::Vector{Matrix}`: upper bounds 
"""
function get_gradient(nnet::Network, input::AbstractPolytope)
    LΛ, UΛ = act_gradient_bounds(nnet, input)
    return get_gradient(nnet, LΛ, UΛ)
end

"""
    act_gradient_bounds(nnet::Network, input::AbstractPolytope)

Computing the bounds on the gradient of all activation functions given an input set.
Currently only support ReLU.
Return:
- `LΛ::Vector{Matrix}`: lower bounds
- `UΛ::Vector{Matrix}`: upper bounds 
"""
function act_gradient_bounds(nnet::Network, input::AbstractPolytope)
    bounds = get_bounds(nnet, input)
    LΛ = Vector{Matrix}(undef, 0) 
    UΛ = Vector{Matrix}(undef, 0)
    for (i, layer) in enumerate(nnet.layers)
        before_act_bound = linear_transformation(layer, bounds[i])
        lower = low(before_act_bound)
        upper = high(before_act_bound)
        l = act_gradient(layer.activation, lower)
        u = act_gradient(layer.activation, upper)
        push!(LΛ, Diagonal(l))
        push!(UΛ, Diagonal(u))
    end
    return (LΛ, UΛ)
end

"""
    get_gradient(nnet::Network, LΛ::Vector{Matrix}, UΛ::Vector{Matrix})

Get lower and upper bounds on network gradient for given gradient bounds on activations
Inputs:
- `LΛ::Vector{Matrix}`: lower bounds on activation gradients
- `UΛ::Vector{Matrix}`: upper bounds on activation gradients
Return:
- `LG::Vector{Matrix}`: lower bounds
- `UG::Vector{Matrix}`: upper bounds 
"""
function get_gradient(nnet::Network, LΛ::Vector{Matrix}, UΛ::Vector{Matrix})
    n_input = size(nnet.layers[1].weights, 2)
    LG = Matrix(1.0I, n_input, n_input)
    UG = Matrix(1.0I, n_input, n_input)
    for (i, layer) in enumerate(nnet.layers)
        LG_hat, UG_hat = interval_map(layer.weights, LG, UG)
        LG = LΛ[i] * max.(LG_hat, 0) + UΛ[i] * min.(LG_hat, 0)
        UG = LΛ[i] * min.(UG_hat, 0) + UΛ[i] * max.(UG_hat, 0)
    end
    return (LG, UG)
end

"""
    get_gradient(nnet::Network, LΛ::Vector{Vector{N}}, UΛ::Vector{Vector{N}}) where N

Get lower and upper bounds on network gradient for given gradient bounds on activations
Inputs:
- `LΛ::Vector{Vector{N}}`: lower bounds on activation gradients
- `UΛ::Vector{Vector{N}}`: upper bounds on activation gradients
Return:
- `LG::Vector{Matrix}`: lower bounds
- `UG::Vector{Matrix}`: upper bounds 
"""
function get_gradient(nnet::Network, LΛ::Vector{Vector{N}}, UΛ::Vector{Vector{N}}) where N
    n_input = size(nnet.layers[1].weights, 2)
    LG = Matrix(1.0I, n_input, n_input)
    UG = Matrix(1.0I, n_input, n_input)
    for (i, layer) in enumerate(nnet.layers)
        LG_hat, UG_hat = interval_map(layer.weights, LG, UG)
        LG = Diagonal(LΛ[i]) * max.(LG_hat, 0) + Diagonal(UΛ[i]) * min.(LG_hat, 0)
        UG = Diagonal(LΛ[i]) * min.(UG_hat, 0) + Diagonal(UΛ[i]) * max.(UG_hat, 0)
    end
    return (LG, UG)
end

"""
    interval_map(W::Matrix{N}, l::Vector{N}, u::Vector{N}) where N

Simple linear mapping on intervals
Inputs:
- `W::Matrix{N}`: the linear mapping
- `l::Vector{N}`: the lower bound
- `u::Vector{N}`: the upper bound
Outputs:
- `l_new::Vector{N}`: the lower bound after mapping
- `u_new::Vector{N}`: the upper bound after mapping
"""
function interval_map(W::Matrix{N}, l::Vector{N}, u::Vector{N}) where N
    l_new = max.(W, 0) * l + min.(W, 0) * u
    u_new = max.(W, 0) * u + min.(W, 0) * l
    return (l_new, u_new)
end

"""
    interval_map(W::Matrix{N}, l::Vector{N}, u::Vector{N}) where N

Simple linear mapping on intervals
Inputs:
- `W::Matrix{N}`: the linear mapping
- `l::Matrix{N}`: the lower bound
- `u::Matrix{N}`: the upper bound
Outputs:
- `l_new::Matrix{N}`: the lower bound after mapping
- `u_new::Matrix{N}`: the upper bound after mapping
"""
function interval_map(W::Matrix{N}, l::Matrix{N}, u::Matrix{N}) where N
    l_new = max.(W, 0) * l + min.(W, 0) * u
    u_new = max.(W, 0) * u + min.(W, 0) * l
    return (l_new, u_new)
end

"""
    get_bounds(nnet::Network, input::Hyperrectangle)

This function calls maxSens to compute node-wise bounds given a input set.

Return: 
- `bounds::Vector{Hyperrectangle}`: bounds for all nodes AFTER activation. `bounds[1]` is the input set.
"""
function get_bounds(nnet::Network, input::Hyperrectangle) # NOTE there is another function by the same name in convDual. Should reconsider dispatch
    solver = MaxSens(0.0, true)
    bounds = Vector{Hyperrectangle}(undef, length(nnet.layers) + 1)
    bounds[1] = input
    for (i, layer) in enumerate(nnet.layers)
        bounds[i+1] = forward_layer(solver, layer, bounds[i])
    end
    return bounds
end

"""
    get_bounds(problem::Problem)

This function calls maxSens to compute node-wise bounds given a problem.

Return: 
- `bounds::Vector{Hyperrectangle}`: bounds for all nodes AFTER activation. `bounds[1]` is the input set.
"""
get_bounds(problem::Problem) = get_bounds(problem.network, problem.input)

"""
    get_bounds(nnet::Network, input::Hyperrectangle, act::Bool)

Compute bounds before or after activation by interval arithmetic. To be implemented.

Inputs:
- `nnet::Network`: network
- `input::Hyperrectangle`: input set
- `act::Bool`: `true` for after activation bound; `false` for before activation bound
Return: 
- `bounds::Vector{Hyperrectangle}`: bounds for all nodes AFTER activation. `bounds[1]` is the input set.
"""
function get_bounds(nnet::Network, input::Hyperrectangle, act::Bool)
    if act
        return get_bounds(nnet, input)
    else
        error("before activation bounds not supported yet.")
    end
end

"""
    linear_transformation(layer::Layer, input::Hyperrectangle)

Transformation of a set considering linear mappings in a layer.

Inputs:
- `layer::Layer`: a layer in a network
- `input::Hyperrectangle`: input set
Return: 
- `output::Hyperrectangle`: set after transformation.
"""
function linear_transformation(layer::Layer, input::Hyperrectangle)
    (W, b, act) = (layer.weights, layer.bias, layer.activation)
    before_act_center = W * input.center + b
    before_act_radius = abs.(W) * input.radius
    return Hyperrectangle(before_act_center, before_act_radius)
end

"""
    split_interval(dom::Hyperrectangle, index::Int64)

Split a set into two at the given index.

Inputs:
- `dom::Hyperrectangle`: the set to be split
- `index`: the index to split at
Return: 
- `(left, right)::Tuple{Hyperrectangle, Hyperrectangle}`: two sets after split
"""
function split_interval(dom::Hyperrectangle, index::Int64)
    input_lower, input_upper = low(dom), high(dom)

    input_upper[index] = dom.center[index]
    input_split_left = Hyperrectangle(low = input_lower, high = input_upper)

    input_lower[index] = dom.center[index]
    input_upper[index] = dom.center[index] + dom.radius[index]
    input_split_right = Hyperrectangle(low = input_lower, high = input_upper)
    return (input_split_left, input_split_right)
end