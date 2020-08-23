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
@with_kw struct ExactReach <: Solver
    return_bounds::Bool = false
end

function solve(solver::ExactReach, problem::Problem)
    reach = forward_network(solver, problem.network, problem.input)
    return check_inclusion(reach, problem.output)
end

forward_layer(solver::ExactReach, layer::Layer, input) = forward_layer(solver, layer, convert(HPolytope, input))

function forward_layer(solver::ExactReach, layer::Layer, input::Vector{<:HPolytope})
    output = Vector{HPolytope}(undef, 0)
    for i in 1:length(input)
        input[i] = affine_map(layer, input[i])
        append!(output, forward_partition(layer.activation, input[i]))
    end
    return output
end

function forward_layer(solver::ExactReach, layer::Layer, input::Vector{<:HPolytope}, bounds::Vector{Hyperrectangle})
    output = Vector{HPolytope}(undef, 0)

    num_bounds = size(layer.weights, 1)
    lower_bounds = Vector{Float64}(undef, num_bounds)
    upper_bounds = Vector{Float64}(undef, num_bounds)

    for i in 1:length(input)
        input[i] = affine_map(layer, input[i])

        # Keep track of the pre-activation upper and lower bounds for the
        # next layer. This will
        cur_lower_bounds, cur_upper_bounds = hpolytope_to_bounds(input[i])
        if (i == 1)
            lower_bounds, upper_bounds = cur_lower_bounds, cur_upper_bounds
        else
            lower_bounds, upper_bounds = min.(cur_lower_bounds, lower_bounds), max.(cur_upper_bounds, upper_bounds)
        end

        # Project with the ReLU activation function
        append!(output, forward_partition(layer.activation, input[i]))
    end

    push!(bounds, Hyperrectangle(low=lower_bounds, high=upper_bounds))

    return output, bounds
end

function forward_layer(solver::ExactReach, layer::Layer, input::HPolytope)
    input = affine_map(layer, input)
    return forward_partition(layer.activation, input)
end

function forward_layer(solver::ExactReach, layer::Layer, input::HPolytope, bounds::Vector{Hyperrectangle})
    input = affine_map(layer, input)

    # Add to your list of bounds for the set of nodes output from this layer.
    bound_length = size(layer.weights, 1) # number of outputs of this layer gives us the number of bounds we want
    new_lower_bounds, new_upper_bounds = hpolytope_to_bounds(input)
    push!(bounds, Hyperrectangle(low=new_lower_bounds, high=new_upper_bounds))

    return forward_partition(layer.activation, input), bounds
end

# Returns the lower and upper bound on each dimension for a hpolytope
function hpolytope_to_bounds(set::HPolytope)
    hyperrectangle = overapproximate(set, Hyperrectangle)
    return low(hyperrectangle), high(hyperrectangle)
end
