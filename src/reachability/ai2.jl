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
struct Ai2{T<:Union{Hyperrectangle, Zonotope, HPolytope}} <: Solver end

Ai2() = Ai2{HPolytope}()
Ai2z() = Ai2{Zonotope}()
Box() = Ai2{Hyperrectangle}()

function solve(solver::Ai2, problem::Problem)
    reach = forward_network(solver, problem.network, problem.input)
    return check_inclusion(reach, problem.output)
end

forward_layer(solver::Ai2, L::Layer, inputs::Vector) = forward_layer.(solver, L, inputs)

function forward_layer(solver::Ai2{HPolytope}, L::Layer, input::AbstractPolytope)
    Ẑ = affine_map(L, input)
    relued_subsets = forward_partition(L.activation, Ẑ) # defined in reachability.jl
    return convex_hull(UnionSetArray(relued_subsets))
end

# method for Zonotope and Hyperrectangle, if the input set isn't a Zonotpe
function forward_layer(solver::Ai2, L::Layer, input::AbstractPolytope)
    X = overapproximate(input, Hyperrectangle)
    return forward_layer(solver, L, X)
end

function forward_layer(solver::Ai2{T}, L::Layer, input::AbstractZonotope) where T
    Ẑ = affine_map(L, input)
    return overapproximate(Rectification(Ẑ), T)
end
