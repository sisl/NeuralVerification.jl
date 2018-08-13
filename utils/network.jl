abstract type ActivationFunction end

struct GeneralAct <: ActivationFunction
	activation::ActivationFunction
	GeneralAct() = new()
end

struct ReLU <: ActivationFunction
	function ReLU(x::Float64)
		return max(0,x)
	end
	ReLU() = new()
end

function f(x::Float64, activation::ReLU)
	return max(0,x)
end

struct Layer
    weights::Matrix{Float64}
    bias::Vector{Float64}
    activation::ActivationFunction
end

struct Network
    layers::Vector{Layer} # layers includes output layer
end
