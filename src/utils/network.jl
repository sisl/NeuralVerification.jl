"""
    Node(w::Vector{Float64}, b::Float64, activation::ActivationFunction)

Node consists of `w` and `b` for linear mapping, and `activation` for nonlinear mapping.
"""
struct Node
    w::Vector{Float64}
    b::Float64
    act::ActivationFunction
end

"""
    Layer(weights::Matrix{Float64}, bias::Vector{Float64}, activation::ActivationFunction)

Layer consists of `weights` and `bias` for linear mapping, and `activation` for nonlinear mapping.
See also: [`Network`]@ref
"""
struct Layer
    weights::Matrix{Float64}
    bias::Vector{Float64}
    activation::ActivationFunction
end

"""
    Network(layers::Vector{Layer})

Network consists of a Vector of layers.
See also: [`Layer`]@ref
"""
struct Network
    layers::Vector{Layer} # layers includes output layer
end

"""
    n_node(L::Layer)

Returns the number of neurons in a layer.
"""
n_nodes(L::Layer) = length(L.bias)

