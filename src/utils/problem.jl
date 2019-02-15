
"""
    PolytopeComplement

The complementary set to a given `AbstractPolytope`. Note that with the exception
 of Halfspaces, the complement of a polytope is not itself a polytope.

### Examples
```julia
julia> H = Hyperrectangle([0,0], [1,1])
Hyperrectangle{Int64}([0, 0], [1, 1])

julia> PC = PolytopeComplement(H)
PolytopeComplement of:
  Hyperrectangle{Int64}([0, 0], [1, 1])

julia> center(H) ∈ PC
false

julia> high(H).+[1,1] ∈ PC
true
```
"""
struct PolytopeComplement{Q<:AbstractPolytope}
    P::Q
end

Base.show(io::IO, PC::PolytopeComplement) = (println(io, "PolytopeComplement of:"), println(io, "  ", PC.P))
LazySets.issubset(s, PC::PolytopeComplement) = LazySets.in_intersection_empty(s, PC.P)
LazySets.is_intersection_empty(s, PC::PolytopeComplement) = LazySets.issubset(s, PC.P)
complement(PC::AbstractPolytope) = PolytopeComplement(PC)
complement(PC::PolytopeComplement) = PC.P
Base.in(pt, PC::PolytopeComplement) = pt ∉ PC.P
# etc.


"""
    Problem{P, Q}(network::Network, input::P, output::Q)

Problem definition for neural verification.

The verification problem consists of: for all  points in the input set,
the corresponding output of the network must belong to the output set.
"""
struct Problem{P, Q}
    network::Network
    input::P
    output::Q
end

"""
    Result
Supertype of all result types.

See also: [`BasicResult`](@ref), [`CounterExampleResult`](@ref), [`AdversarialResult`](@ref), [`ReachabilityResult`](@ref)
"""
abstract type Result end

status(result::Result) = result.status

function validate_status(st::Symbol)
    @assert st ∈ (:holds, :violated, :Unknown) "unexpected status code: `:$st`.\nOnly (:holds, :violated, :Unknown) are accepted"
    return st
end

"""
    BasicResult(status::Symbol)

Result type that captures whether the input-output constraint is satisfied.
Possible status values:\n
    :holds (io constraint is satisfied always)\n
    :violated (io constraint is violated)\n
    :Unknown (could not be determined)
"""
struct BasicResult <: Result
    status::Symbol
end

"""
    CounterExampleResult(status, counter_example)

Like `BasicResult`, but also returns a `counter_example` if one is found (if status = :violated).
The `counter_example` is a point in the input set that, after the NN, lies outside the output set.
"""
struct CounterExampleResult <: Result
    status::Symbol
    counter_example::Vector{Float64}
    CounterExampleResult(s, ce) = new(validate_status(s), ce)
end

"""
    AdversarialResult(status, max_disturbance)

Like `BasicResult`, but also returns the maximum allowable disturbance in the input (if status = :violated).
"""
struct AdversarialResult <: Result
	status::Symbol
	max_disturbance::Float64
    AdversarialResult(s, md) = new(validate_status(s), md)
end

"""
    ReachabilityResult(status, reachable)

Like `BasicResult`, but also returns the output reachable set given the input constraint (if status = :violated).
"""
struct ReachabilityResult <: Result
	status::Symbol
	reachable::Vector{<:AbstractPolytope}
    ReachabilityResult(s, r) = new(validate_status(s), r)
end

# Additional constructors:
CounterExampleResult(s) = CounterExampleResult(s, Float64[])
AdversarialResult(s)    = AdversarialResult(s, -1.0)
ReachabilityResult(s)   = ReachabilityResult(s, AbstractPolytope[])
