"""
    Ai2

Ai2 performs over-approximated reachability analysis to compute the over-approximated output reachable set for a network.

# Problem requirement
1. Network: any depth, ReLU activation (more activations to be supported in the future)
2. Input: HPolytope
3. Output: AbstractPolytope

# Return
`ReachabilityResult`

# Method
Reachability analysis using split and join.

# Property
Sound but not complete.

# Reference
T. Gehr, M. Mirman, D. Drashsler-Cohen, P. Tsankov, S. Chaudhuri, and M. Vechev,
"Ai2: Safety and Robustness Certification of Neural Networks with Abstract Interpretation,"
in *2018 IEEE Symposium on Security and Privacy (SP)*, 2018.
"""
struct Ai2 end

function solve(solver::Ai2, problem::Problem)
    reach = forward_network(solver, problem.network, problem.input)
    return check_inclusion(reach, problem.output)
end

forward_layer(solver::Ai2, layer::Layer, inputs::Vector{<:AbstractPolytope}) = forward_layer.(solver, layer, inputs)

function forward_layer(solver::Ai2, layer::Layer, input::AbstractPolytope)
    outlinear = affine_map(layer, input)
    relued_subsets = forward_partition(layer.activation, outlinear) # defined in ExactReach
    return convex_hull(relued_subsets)
end

# extend lazysets convex_hull to a vector of polytopes
function LazySets.convex_hull(sets::Vector{<:AbstractPolytope}; backend = CDDLib.Library())
    hull = first(sets)
    for P in sets
        hull = convex_hull(hull, P, backend = backend)
    end
    return hull
end