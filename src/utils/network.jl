abstract type AbstractNetwork end

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

"""
A Vector of layers.

    Network([layer1, layer2, layer3, ...])

See also: [`Layer`](@ref)
"""
struct Network <: AbstractNetwork
    layers::Vector{Layer} # layers includes output layer
end

"""
    n_nodes(L::Layer)

Returns the number of neurons in a layer.
"""
n_nodes(L::Layer) = length(L.bias)

