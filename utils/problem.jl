include("network.jl")
using JuMP

type Constraints
	A::Matrix{Float64}
	b::Vector{Float64}
	upper::Vector{Float64}
	lower::Vector{Float64}
end

type Problem
	network::Network
	input::Constraints
	output::Constraints
end
