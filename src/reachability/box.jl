"""
    Box <: AbstractSolver
Box performs over-approximated reachability analysis to compute the over-approximated output reachable set for a network.
# Problem requirement
1. Network: any depth, ReLU activation (more activations to be supported in the future)
2. Input: Hyperrectangle
3. Output: Hyperrectangle
# Return
`ReachabilityResult`
# Method
Reachability analysis using using boxes.
# Property
Sound but not complete?.
"""
struct Box <: Solver end

function solve(solver::Box, problem::Problem)
    reach = forward_network(solver, problem.network, problem.input)
    return check_inclusion(reach, problem.output)
end

forward_layer(solver::Box, layer::Layer, inputs::Vector{<:LazySet}) = forward_layer.(solver, layer, inputs)

function forward_layer(solver::Box, layer::Layer, input::AbstractPolytope)
    return forward_layer(solver, layer, overapproximate(input, Hyperrectangle))
end

function forward_layer(solver::Box, layer::Layer, input::Hyperrectangle)
    outlinear = overapproximate(AffineMap(layer.weights, input, layer.bias), Hyperrectangle)
    relued_subsets = forward_partition(layer.activation, outlinear)
    return relued_subsets
end
