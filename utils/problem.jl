using JuMP
include("network.jl")

struct Constraints
	A::Matrix{Float64}
	b::Vector{Float64}
	upper::Vector{Float64}
	lower::Vector{Float64}
end

struct Problem
	network::Network
	input::Constraints
	output::Constraints
end

#=
Add constraints from Constraint struct to a variable
=#
function add_constraints(model::Model, x::Array{Variable}, constraints::Constraints)
	@constraint(model, constraints.A *x .== constraints.b)
	@constraint(model, x .<= constraints.upper)
	@constraint(model, x .>= constraints.lower)
end