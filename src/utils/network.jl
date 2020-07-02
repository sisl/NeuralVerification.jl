"""
    Node{N, F}

A single node in a layer.
### Fields
 - `w::Vector{N}`
 - `b::N`
 - `activation::F`
"""
struct Node{F<:ActivationFunction, N<:Number}
    w::Vector{N}
    b::N
    act::F
end

"""
    Layer{F, N}

Consists of `weights` and `bias` for linear mapping, and `activation` for nonlinear mapping.
### Fields
 - `weights::Matrix{N}`
 - `bias::Vector{N}`
 - `activation::F`

See also: [`Network`](@ref)
"""
struct Layer{F<:ActivationFunction, N<:Number}
    weights::Matrix{N}
    bias::Vector{N}
    activation::F
end
Base.:(==)(x::Layer, y::Layer) = x.weights == y.weights && x.bias == y.bias && x.activation == y.activation


"""
A Vector of layers.

    Network([layer1, layer2, layer3, ...])

See also: [`Layer`](@ref)
"""
struct Network
    layers::Vector{Layer} # layers includes output layer
end
Base.:(==)(x::Network, y::Network) = all(x.layers .== y.layers)


"""
    n_nodes(L::Layer)

Returns the number of neurons in a layer.
"""
n_nodes(L::Layer) = length(L.bias)
