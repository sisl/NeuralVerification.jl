"""
    Problem(network, input, output)

Problem definition for neural verification.
- `network` is defined by type `Network`
- `input` belongs to `AbstractPolytope` in `LazySets.jl`
- `output` belongs to `AbstractPolytope` in `LazySets.jl`

We need to verify: for all input points in the input set, the corresponding output of the network belongs to the output set. 
"""
struct Problem{P<:AbstractPolytope, Q<:AbstractPolytope}
    network::Network
    input::P
    output::Q
end


abstract type Result end

validate_status(st::Symbol) = st ∈ (:SAT, :UNSAT, :Unknown) ? st : error("unexpected status code: `:$st`.\nOnly (:SAT, :UNSAT, :Unknown) are accepted")

struct BasicResult <: Result
    status::Symbol
end

struct CounterExampleResult <: Result
    status::Symbol
    counter_example::Vector{Float64}
    CounterExampleResult(s, ce) = new(validate_status(s), ce)
end

struct AdversarialResult <: Result
	status::Symbol
	max_disturbance::Float64
    AdversarialResult(s, md) = new(validate_status(s), md)
end

struct ReachabilityResult <: Result
	status::Symbol
	reachable::Vector{<:AbstractPolytope}
    ReachabilityResult(s, r) = new(validate_status(s), r)
end

# Additional constructors:
CounterExampleResult(s) = CounterExampleResult(s, Float64[])
AdversarialResult(s)    = AdversarialResult(s, -1.0)
ReachabilityResult(s)   = ReachabilityResult(s, AbstractPolytope[])

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

Like `BasicResult`, but also returns the maximum allowable disturbance in the input (if :UNSAT).
"""
AdversarialResult

"""
    ReachabilityResult(status, reachable)

Like `BasicResult`, but also returns the output reachable set given the input constraint (if :UNSAT).
"""
ReachabilityResult
