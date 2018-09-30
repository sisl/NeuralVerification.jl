
# struct Constraints
#   A::Matrix{Float64}
#   b::Vector{Float64}
#   upper::Vector{Float64}
#   lower::Vector{Float64}
# end

struct Problem{P<:AbstractPolytope, Q<:AbstractPolytope}
    network::Network
    input::P
    output::Q
end

# Abstract type Result
# CounterExampleResult
# TrueFalseResult
# AdversarialResult
# function status(result::Result)
abstract type Result end

# This is the basic result type which only specifies
# :SAT when the input-output constraint is always satisfied
# :UNSAT when the input-output constraint is not satisfied
# :Unknown
struct BasicResult <: Result
	status::Symbol
end

# In addition to basic status
# We also output the counter example
# It is a point in the input set such that after the NN, it lies outside the output set
struct CounterExampleResult <: Result
    status::Symbol
    counter_example::Vector{Float64}
end

# Given the output constraint,
# What is the maximum allowable disturbance in the input side
struct AdversarialResult <: Result
	status::Symbol
	max_disturbance::Float64
end

# Given the input constraint,
# What is the output reachable set
struct ReachabilityResult <: Result
	status::Symbol
	reachable::Vector{<:AbstractPolytope}
end

Result(x) = BasicResult(x)
Result(x, y::Vector{Float64}) = CounterExampleResult(x, y)
Result(x, y::Float64) = AdversarialResult(x, y)
Result(x, y::AbstractPolytope) = ReachabilityResult(x, [y])
Result(x, y::Vector{<:AbstractPolytope}) = ReachabilityResult(x, y)

CounterExampleResult(x) = CounterExampleResult(x, [])
AdversarialResult(x) = AdversarialResult(x, -1.0)
ReachabilityResult(x) = ReachabilityResult(x, [])

function status(result::Result)
	return result.status
end

#=
Add constraints from Polytope to a variable
=#
# function add_constraints(model::Model, x::Array{Variable}, constraints::HPolytope)
#     A, b = tosimplehrep(constraints)
#     @constraint(model, A*x .<= b)
# end
