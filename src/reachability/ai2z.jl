"""
    Ai2z <: AbstractSolver

Ai2 performs over-approximated reachability analysis to compute the over-approximated output reachable set for a network.

# Problem requirement
1. Network: any depth, ReLU activation (more activations to be supported in the future)
2. Input: Zonotope
3. Output: Zonotope

# Return
`ReachabilityResult`

# Method
Reachability analysis using split and join using Zonotopes as proposed on [1].

# Property
Sound but not complete.

# Reference
T. Gehr, M. Mirman, D. Drashsler-Cohen, P. Tsankov, S. Chaudhuri, and M. Vechev,
"Ai2: Safety and Robustness Certification of Neural Networks with Abstract Interpretation,"
in *2018 IEEE Symposium on Security and Privacy (SP)*, 2018.
"""
struct Ai2z <: Solver end

function solve(solver::Ai2z, problem::Problem)
    if isa(problem.input, LazySet)
        input = [problem.input]
    else
        input = problem.input
    end
    f_n(x) = forward_network(solver, problem.network, x)
    reach = map(f_n, input)
    return check_inclusion(reach, problem.output)
end

forward_layer(solver::Ai2z, layer::Layer, inputs::Vector{<:LazySet}) = forward_layer.(solver, layer, inputs)

function forward_layer(solver::Ai2z, layer::Layer, input::AbstractPolytope)
    return forward_layer(solver, layer, overapproximate(input, Hyperrectangle))
end

function forward_layer(solver::Ai2z, layer::Layer, input::AbstractZonotope)
    outlinear = affine_map(layer, input)
    relued_subsets = forward_partition(layer.activation, outlinear)
    return relued_subsets
end
