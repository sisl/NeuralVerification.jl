"""
    NNV

# Problem requirement
1. Network: any depth, ReLU activation
2. Input: HPolytope
3. Output: Any (AbstractPolytope and PolytopeComplement)

# Return
`ReachabilityResult`

# Method
Exact reachability analysis.

# Property
Sound and complete.

# Reference

"""
struct NNV <: Solver end

function solve(solver::NNV, problem::Problem)
    reach = forward_network(solver, problem.network, problem.input)
    if problem.output isa AbstractPolytope
        return check_inclusion(reach, problem.output)
    else
        @assert False "Non polytope output sets not yet supported"
    end
end

# Forward linear for a single polytope and a list of polytopes converts to Stars
forward_linear(solver::NNV, L::Layer, input::AbstractPolytope) = forward_linear(solver, L, convert(Star, input))
forward_linear(solver::NNV, L::Layer, input::Vector{<:AbstractPolytope}) = forward_linear(solver, L, [convert(Star, set) for set in input])

# Forward linear for a single Star and a list of Stars
forward_linear(solver::NNV, L::Layer, input::Star) = affine_map(L, input)
forward_linear(solver::NNV, L::Layer, input::Vector{Star}) = [affine_map(L, set) for set in input]

# Forward act for a single Star and a list of Stars
forward_act(solver::NNV, L::Layer, input::Star) = forward_partition(L.activation, input)
function forward_act(solver::NNV, L::Layer, input::Vector{<:Star})
    output = Vector{Star}(undef, 0)
    for i in 1:length(input)
        append!(output, forward_partition(L.activation, input[i]))
    end
    return output
end
