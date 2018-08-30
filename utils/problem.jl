using JuMP
using LazySets
include("network.jl")

abstract type Problem end

struct Constraints
	A::Matrix{Float64}
	b::Vector{Float64}
	upper::Vector{Float64}
	lower::Vector{Float64}
end

struct AdversarialProblem <: Problem
	network::Network
	input::Vector{Float64}
	targets::Vector{Int64}
end

struct FeasibilityProblem <: Problem
	network::Network
	input::Constraints
	output::Constraints
end

struct ReachabilityProblem <: Problem
	network::Network
	input::AbstractPolytope
	output::AbstractPolytope
end
	
#=
Add constraints from Constraint struct to a variable
=#
function add_constraints(model::Model, x::Array{Variable}, constraints::Constraints)
	@constraint(model, constraints.A *x .== constraints.b)
	@constraint(model, x .<= constraints.upper)
	@constraint(model, x .>= constraints.lower)
end