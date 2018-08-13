abstract type ActivationFunction end

struct GeneralAct <: ActivationFunction
	activation::ActivationFunction
	GeneralAct() = new()
end

struct ReLU <: ActivationFunction
	ReLU() = new()
	ReLU(x) = max.(0,x)
end

struct Max <: ActivationFunction
	Max() = new()
	Max(x) = max(maximum(x),0)
end

struct Layer
    weights::Matrix{Float64}
    bias::Vector{Float64}
    activation::ActivationFunction
end

struct Network
    layers::Vector{Layer} # layers includes output layer
end
