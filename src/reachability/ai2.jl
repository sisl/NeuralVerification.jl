"""
    Ai2{T}

`Ai2` performs over-approximated reachability analysis to compute the over-approximated
output reachable set for a network. `T` can be `Hyperrectangle`, `Zonotope`, or
`HPolytope`, and determines the amount of over-approximation (and hence also performance
tradeoff). The original implementation (from [1]) uses Zonotopes, so we consider this
the "benchmark" case. The `HPolytope` case is more precise, but slower, and the opposite
is true of the `Hyperrectangle` case.

Note that initializing `Ai2()` defaults to `Ai2{Zonotope}`.
The following aliases also exist for convenience:

```julia
const Ai2h = Ai2{HPolytope}
const Ai2z = Ai2{Zonotope}
const Box = Ai2{Hyperrectangle}
```

# Problem requirement
1. Network: any depth, ReLU activation (more activations to be supported in the future)
2. Input: AbstractPolytope
3. Output: AbstractPolytope

# Return
`ReachabilityResult`

# Method
Reachability analysis using split and join.

# Property
Sound but not complete.

# Reference
[1] T. Gehr, M. Mirman, D. Drashsler-Cohen, P. Tsankov, S. Chaudhuri, and M. Vechev,
"Ai2: Safety and Robustness Certification of Neural Networks with Abstract Interpretation,"
in *2018 IEEE Symposium on Security and Privacy (SP)*, 2018.

## Note
Efficient over-approximation of intersections and unions involving zonotopes relies on Theorem 3.1 of

[2] Singh, G., Gehr, T., Mirman, M., Püschel, M., & Vechev, M. (2018). Fast
and effective robustness certification. In Advances in Neural Information
Processing Systems (pp. 10802-10813).
"""
struct Ai2{T<:Union{Hyperrectangle, Zonotope, HPolytope}} <: Solver end

Ai2() = Ai2{Zonotope}()
const Ai2h = Ai2{HPolytope}
const Ai2z = Ai2{Zonotope}
const Box = Ai2{Hyperrectangle}

function solve(solver::Ai2, problem::Problem)
    reach = forward_network(solver, problem.network, problem.input)
    return check_inclusion(reach, problem.output)
end

forward_layer(solver::Ai2, L::Layer, inputs::Vector) = forward_layer.(solver, L, inputs)

function forward_layer(solver::Ai2h, L::Layer, input::AbstractPolytope)
    Ẑ = affine_map(L, input)
    relued_subsets = forward_partition(L.activation, Ẑ) # defined in reachability.jl
    return convex_hull(relued_subsets)
end

# method for Zonotope and Hyperrectangle, if the input set isn't a Zonotope
function forward_layer(solver::Ai2, L::Layer, input::AbstractPolytope)
    X = overapproximate(input, Hyperrectangle)
    return forward_layer(solver, L, X)
end

function forward_layer(solver::Ai2{T}, L::Layer, input::AbstractZonotope) where T<:Union{Hyperrectangle, Zonotope}
    Ẑ = affine_map(L, input)
    return overapproximate(Rectification(Ẑ), T)
end


# extend lazysets convex_hull to a vector of polytopes
function LazySets.convex_hull(sets::Vector{<:AbstractPolytope}; backend = CDDLib.Library())
    hull = first(sets)
    for P in sets
        hull = convex_hull(hull, P, backend = backend)
    end
    return hull
end