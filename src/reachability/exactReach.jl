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
struct ExactReach <: Solver end

function solve(solver::ExactReach, problem::Problem)
    reach = forward_network(solver, problem.network, problem.input)
    return check_inclusion(reach, problem.output)
end

# Forward_linear that coverts the input to HPolytope
forward_linear(solver::ExactReach, L::Layer, input) = forward_linear(solver, L, convert(HPolytope, input))

# Forward linear for a single polytope and a list of polytopes
forward_linear(solver::ExactReach, L::Layer, input::HPolytope) = affine_map(L, input)
forward_linear(solver::ExactReach, L::Layer, input::Vector{<:HPolytope}) = [affine_map(L, set) for set in input]

# Forward act for a single polytope and a list of polytopes
forward_act(solver::ExactReach, L::Layer, input::HPolytope) = forward_partition(L.activation, input)
function forward_act(solver::ExactReach, L::Layer, input::Vector{<:HPolytope})
    output = Vector{HPolytope}(undef, 0)
    for i in 1:length(input)
        append!(output, forward_partition(L.activation, input[i]))
    end
    return output
end
