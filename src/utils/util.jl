"""
    read_nnet(fname::String; last_layer_activation = Id())

Read in neural net from a `.nnet` file and return Network struct.
The `.nnet` format is borrowed from [NNet](https://github.com/sisl/NNet).
The format assumes all hidden layers have ReLU activation.
Keyword argument `last_layer_activation` sets the activation of the last
layer, and defaults to `Id()`, (i.e. a linear output layer).
"""
function read_nnet(fname::String; last_layer_activation = Id())
    f = open(fname)
    line = readline(f)
    while occursin("//", line) #skip comments
        line = readline(f)
    end
    # number of layers
    nlayers = parse(Int64, split(line, ",")[1])
    # read in layer sizes
    layer_sizes = parse.(Int64, split(readline(f), ",")[1:nlayers+1])
    # read past additonal information
    for i in 1:5
        line = readline(f)
    end
    # i=1 corresponds to the input dimension, so it's ignored
    layers = Layer[read_layer(dim, f) for dim in layer_sizes[2:end-1]]
    push!(layers, read_layer(last(layer_sizes), f, last_layer_activation))

    return Network(layers)
end

"""
    read_layer(output_dim::Int, f::IOStream, [act = ReLU()])

Read in layer from nnet file and return a `Layer` containing its weights/biases.
Optional argument `act` sets the activation function for the layer.
"""
function read_layer(output_dim::Int64, f::IOStream, act = ReLU())

    rowparse(splitrow) = parse.(Float64, splitrow[findall(!isempty, splitrow)])
     # first read in weights
     W_str_vec = [rowparse(split(readline(f), ",")) for i in 1:output_dim]
     weights = vcat(W_str_vec'...)
     # now read in bias
     bias_string = [split(readline(f), ",")[1] for j in 1:output_dim]
     bias = rowparse(bias_string)
     # activation function is set to ReLU as default
     return Layer(weights, bias, act)
end

"""
    compute_output(nnet::Network, input::Vector{Float64})

Propagate a given vector through a nnet and compute the output.
"""
function compute_output(nnet::Network, input)
    curr_value = input
    for layer in nnet.layers # layers does not include input layer (which has no weights/biases)
        curr_value = layer.activation(affine_map(layer, curr_value))
    end
    return curr_value # would another name be better?
end

"""
    get_activation(L, x::Vector)
Finds the activation pattern of a vector `x` subject to the activation function given by the layer `L`.
Returns a Vector{Bool} where `true` denotes the node is "active". In the sense of ReLU, this would be `x[i] >= 0`.
"""
get_activation(L::Layer{ReLU}, x::Vector) = x .>= 0.0
get_activation(L::Layer{Id}, args...) = trues(n_nodes(L))

"""
    get_activation(nnet::Network, x::Vector)

Given a network, find the activation pattern of all neurons at a given point x.
Returns Vector{Vector{Bool}}. Each Vector{Bool} refers to the activation pattern of a particular layer.
"""
function get_activation(nnet::Network, x::Vector{Float64})
    act_pattern = Vector{Vector{Bool}}(undef, length(nnet.layers))
    curr_value = x
    for (i, layer) in enumerate(nnet.layers)
        curr_value = affine_map(layer, curr_value)
        act_pattern[i] = get_activation(layer, curr_value)
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
- 0: undetermined
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
- 0: undetermined
- -1: not activated
"""
function get_activation(nnet::Network, bounds::Vector{Hyperrectangle})
    act_pattern = Vector{Vector{Int}}(undef, length(nnet.layers))
    for (i, layer) in enumerate(nnet.layers)
        act_pattern[i] = get_activation(layer, bounds[i])
    end
    return act_pattern
end

function get_activation(L::Layer{ReLU}, bounds::Hyperrectangle)
    before_act_bound = approximate_affine_map(L, bounds)
    lower = low(before_act_bound)
    upper = high(before_act_bound)
    act_pattern = zeros(n_nodes(L))
    for j in 1:n_nodes(L) # For evey node
        if lower[j] > 0.0
            act_pattern[j] = 1
        elseif upper[j] < 0.0
            act_pattern[j] = -1
        end
    end
    return act_pattern
end

"""
    get_gradient(nnet::Network, x::Vector)

Given a network, find the gradient at the input x
"""
function get_gradient(nnet::Network, x::Vector)
    z = x
    gradient = Matrix(1.0I, length(x), length(x))
    for (i, layer) in enumerate(nnet.layers)
        z_hat = affine_map(layer, z)
        σ_gradient = act_gradient(layer.activation, z_hat)
        gradient = Diagonal(σ_gradient) * layer.weights * gradient
        z = layer.activation(z_hat)
    end
    return gradient
end

"""
    act_gradient(act::ReLU, z_hat::Vector{N}) where N

Computing the gradient of an activation function at point z_hat.
Currently only support ReLU and Id.
"""
act_gradient(act::ReLU, z_hat::Vector) = z_hat .>= 0.0
act_gradient(act::Id,   z_hat::Vector) = trues(length(z_hat))

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
        before_act_bound = approximate_affine_map(layer, bounds[i])
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
- `(LG, UG)` lower and upper bounds
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
    interval_map(W::Matrix, l, u)

Simple linear mapping on intervals
Inputs:
- `W::Matrix{N}`: linear mapping
- `l::Vector{N}`: lower bound
- `u::Vector{N}`: upper bound
Outputs:
- `(lbound, ubound)` (after the mapping)
"""
function interval_map(W::Matrix{N}, l::AbstractVecOrMat, u::AbstractVecOrMat) where N
    l_new = max.(W, zero(N)) * l + min.(W, zero(N)) * u
    u_new = max.(W, zero(N)) * u + min.(W, zero(N)) * l
    return (l_new, u_new)
end

"""
    get_bounds(problem::Problem)
    get_bounds(nnet::Network, input::Hyperrectangle)

This function calls maxSens to compute node-wise bounds given a input set.

Return:
- `Vector{Hyperrectangle}`: bounds for all nodes **after** activation. `bounds[1]` is the input set.
"""
function get_bounds(nnet::Network, input::Hyperrectangle, act::Bool = true) # NOTE there is another function by the same name in convDual. Should reconsider dispatch
    if act
        solver = MaxSens(0.0, true)
        bounds = Vector{Hyperrectangle}(undef, length(nnet.layers) + 1)
        bounds[1] = input
        for (i, layer) in enumerate(nnet.layers)
            bounds[i+1] = forward_layer(solver, layer, bounds[i])
        end
        return bounds
    else
       error("before activation bounds not supported yet.")
    end
end
get_bounds(problem::Problem) = get_bounds(problem.network, problem.input)

"""
    affine_map(layer, input::AbstractPolytope)

Affine transformation of a set using the weights and bias of a layer.

Inputs:
- `layer`: Layer
- `input`: input set (Hyperrectangle, HPolytope)
Return:
- `output`: set after transformation.


    affine_map(layer, input)

Inputs:
- `layer`: Layer
- `input`: Vector
Return:
- `output`: Vector after mapping
"""
affine_map(layer::Layer, input) = layer.weights*input + layer.bias
function affine_map(layer::Layer, input::AbstractPolytope)
    W, b = layer.weights, layer.bias
    return translate(b, linear_map(W, input))
end

"""
   approximate_affine_map(layer, input::Hyperrectangle)

Returns a Hyperrectangle overapproximation of the affine map of the input.
"""
function approximate_affine_map(layer::Layer, input::Hyperrectangle)
    c = affine_map(layer, input.center)
    r = abs.(layer.weights) * input.radius
    return Hyperrectangle(c, r)
end

function translate(v::Vector, H::HPolytope)
    # translate each halfpsace according to:
    # a⋅(x-v) ≤ b  ⟶  a⋅x ≤ b+a⋅v
    C, d = tosimplehrep(H)
    return HPolytope(C, d+C*v)
end
# translate(v::Vector, H::Hyperrectangle)   = Hyperrectangle(H.center .+ v, H.radius)
translate(v::Vector, V::AbstractPolytope) = tohrep(VPolytope([x+v for x in vertices_list(V)]))

"""
    split_interval(dom, i)

Split a set into two at the given index.

Inputs:
- `dom::Hyperrectangle`: the set to be split
- `i`: the index to split at
Return:
- `(left, right)::Tuple{Hyperrectangle, Hyperrectangle}`: two sets after split
"""
function split_interval(dom::Hyperrectangle, i::Int64)
    input_lower, input_upper = low(dom), high(dom)

    input_upper[i] = dom.center[i]
    input_split_left = Hyperrectangle(low = input_lower, high = input_upper)

    input_lower[i] = dom.center[i]
    input_upper[i] = dom.center[i] + dom.radius[i]
    input_split_right = Hyperrectangle(low = input_lower, high = input_upper)
    return (input_split_left, input_split_right)
end