
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

struct TrueFalseResult <: Result
	status::Symbol
end

struct CounterExampleResult <: Result
    status::Symbol
    counter_example::Vector{Float64}
end

struct AdversarialResult <: Result
	status::Symbol
	max_disturbance::Float64
end

Result(x) = TrueFalseResult(x)
Result(x, y::Vector{Float64}) = CounterExampleResult(x, y)
Result(x, y::Float64) = AdversarialResult(x, y)
CounterExampleResult(x) = CounterExampleResult(x, [])
AdversarialResult(x) = AdversarialResult(x, -1.0)

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
