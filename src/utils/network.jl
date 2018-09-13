
struct Node
    w::Vector{Float64}
    b::Float64
    act::ActivationFunction
end

struct Layer
    weights::Matrix{Float64}
    bias::Vector{Float64}
    activation::ActivationFunction
end

struct Network
    layers::Vector{Layer} # layers includes output layer
end
