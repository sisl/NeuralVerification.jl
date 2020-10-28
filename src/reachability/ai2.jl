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

# Ai2h and Ai2z use affine_map
# Box uses approximate_affine_map for the linear region if it is propagating a zonotope
forward_linear(solver::Ai2h, L::Layer{ReLU}, input::AbstractPolytope) = affine_map(L, input)
forward_linear(solver::Ai2z, L::Layer{ReLU}, input::AbstractZonotope) = affine_map(L, input)
forward_linear(solver::Box, L::Layer{ReLU}, input::AbstractZonotope) = approximate_affine_map(L, input)
# method for Zonotope and Hyperrectangle, if the input set isn't a Zonotope overapproximate
forward_linear(solver::Union{Ai2z, Box}, L::Layer{ReLU}, input::AbstractPolytope) = forward_linear(solver, L, overapproximate(input, Hyperrectangle))

# Forward_act is different for Ai2h, Ai2z and Box
forward_act(solver::Ai2h, L::Layer{ReLU}, Ẑ::AbstractPolytope) = convex_hull(UnionSetArray(forward_partition(L.activation, Ẑ)))
forward_act(solver::Ai2z, L::Layer{ReLU}, Ẑ::AbstractZonotope) = overapproximate(Rectification(Ẑ), Zonotope)
forward_act(slver::Box, L::Layer{ReLU}, Ẑ::AbstractPolytope) = rectify(Ẑ)

# For ID activation do an affine map for all methods
forward_linear(solver::Ai2, L::Layer{Id}, input::AbstractPolytope) = affine_map(L, input)
forward_act(solver::Ai2, L::Layer{Id}, input::AbstractPolytope) = input

function convex_hull(U::UnionSetArray{<:Any, <:HPolytope})
    tohrep(VPolytope(LazySets.convex_hull(U)))
end
