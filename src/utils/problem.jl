using JuMP
using LazySets
include("network.jl")

# struct Constraints
# 	A::Matrix{Float64}
# 	b::Vector{Float64}
# 	upper::Vector{Float64}
# 	lower::Vector{Float64}
# end

struct Problem{P<:AbstractPolytope}
	network::Network
	input::P
	output::P
end

struct Result
	status::Int64
	counter_example::Vector{Float64}
end

#=
Add constraints from Polytope to a variable
=#
# function add_constraints(model::Model, x::Array{Variable}, constraints::HPolytope)
#     A, b = tosimplehrep(constraints)
#     @constraint(model, A*x .<= b)
# end
