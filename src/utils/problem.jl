
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

struct Result
    status::Symbol
    counter_example::Vector{Float64}
    max_disturbance::Float64 # For adversarial problems
end

Result(x) = Result(x, [], -1.0)
Result(x, y::Vector{Float64}) = Result(x, y, -1.0)
Result(x, y::Float64) = Result(x, [], y)

#=
Add constraints from Polytope to a variable
=#
# function add_constraints(model::Model, x::Array{Variable}, constraints::HPolytope)
#     A, b = tosimplehrep(constraints)
#     @constraint(model, A*x .<= b)
# end
