"""
    ExactReach

ExactReach performs exact reachability analysis to compute the output reachable set for a network.

# Problem requirement
1. Network: any depth, ReLU activation
2. Input: HPolytope
3. Output: AbstractPolytope

# Return
`ReachabilityResult`

# Method
Exact reachability analysis.

# Property
Sound and complete.

# Reference
[W. Xiang, H.-D. Tran, and T. T. Johnson,
"Reachable Set Computation and Safety Verification for Neural Networks with ReLU Activations,"
*ArXiv Preprint ArXiv:1712.08163*, 2017.](https://arxiv.org/abs/1712.08163)
"""
struct ExactReach end

function solve(solver::ExactReach, problem::Problem)
    reach = forward_network(solver, problem.network, problem.input)
    return check_inclusion(reach, problem.output)
end

function forward_layer(solver::ExactReach, layer::Layer, input::Vector{HPolytope})
    output = Vector{HPolytope}(undef, 0)
    for i in 1:length(input)
        input[i] = affine_map(layer, input[i])
        append!(output, forward_partition(layer.activation, input[i]))
    end
    return output
end

function forward_layer(solver::ExactReach, layer::Layer, input::HPolytope)
    input = affine_map(layer, input)
    return forward_partition(layer.activation, input)
end