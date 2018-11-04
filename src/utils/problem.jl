
struct Problem{P<:AbstractPolytope, Q<:AbstractPolytope}
    network::Network
    input::P
    output::Q
end


abstract type Result end

validate_status(st::Symbol) = st ∈ (:SAT, :UNSAT, :Unknown) ? st : error("unexpected status code: `$st`.\nOnly (:SAT, :UNSAT, :Unknown) are accepted")

struct BasicResult <: Result
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

struct ReachabilityResult <: Result
	status::Symbol
	reachable::Vector{<:AbstractPolytope}
end

# Additional constructors:
CounterExampleResult(s, ce) = CounterExampleResult(validate_status(s), ce)
AdversarialResult(s, md)    = AdversarialResult(validate_status(s), md)
ReachabilityResult(s, r)    = ReachabilityResult(validate_status(s), r)

CounterExampleResult(s)     = CounterExampleResult(s, Float[])
AdversarialResult(s)        = AdversarialResult(s, -1.0)
ReachabilityResult(s)       = ReachabilityResult(s, [])

function status(result::Result)
	return result.status
end


# TODO: Adversarial and Reachability need clarification
# RESULT TYPE DOCUMENTATION:
"""
    BasicResult(status::Symbol)

Result type that captures whether the input-output constraint is satisfied.
Possible status values:\n
    :SAT (io constraint is satisfied always)\n
    :UNSAT (io constraint is violated)\n
    :Unknown (could not be determined)
"""
BasicResult

"""
    CounterExampleResult(status, counter_example)

Like `BasicResult`, but also returns a `counter_example` if one is found (if :UNSAT).
The `counter_example` is a point in the input set that, after the NN, lies outside the output set.
"""
CounterExampleResult

"""
    AdversarialResult(status, max_disturbance)

Like `BasicResult`, but also returns a the maximum allowable disturbance in the input (if :UNSAT).
"""
AdversarialResult

"""
    ReachabilityResult(status, reachable)

Like `BasicResult`, but also returns the output reachable set given the input constraint (if :UNSAT).
"""
ReachabilityResult
