"""
    ReluVal(max_iter::Int64, tree_search::Symbol)

ReluVal combines symbolic reachability analysis with iterative interval refinement to minimize over-approximation of the reachable set.

# Problem requirement
1. Network: any depth, ReLU activation
2. Input: hyperrectangle
3. Output: AbstractPolytope

# Return
`CounterExampleResult` or `ReachabilityResult`

# Method
Symbolic reachability analysis and iterative interval refinement (search).
- `max_iter` default `10`.
- `tree_search` default `:DFS` - depth first search.

# Property
Sound but not complete.

# Reference
[S. Wang, K. Pei, J. Whitehouse, J. Yang, and S. Jana,
"Formal Security Analysis of Neural Networks Using Symbolic Intervals,"
*CoRR*, vol. abs/1804.10829, 2018. arXiv: 1804.10829.](https://arxiv.org/abs/1804.10829)

[https://github.com/tcwangshiqi-columbia/ReluVal](https://github.com/tcwangshiqi-columbia/ReluVal)
"""
@with_kw struct ReluVal
    max_iter::Int64     = 10
    tree_search::Symbol = :DFS # only :DFS/:BFS allowed? If so, we should assert this.
end

struct SymbolicInterval
    Low::Matrix{Float64}
    Up::Matrix{Float64}
    interval::Hyperrectangle
end

SymbolicInterval(x::Matrix{Float64}, y::Matrix{Float64}) = SymbolicInterval(x, y, Hyperrectangle([0],[0]))

# Data to be passed during forward_layer
struct SymbolicIntervalMask
    sym::SymbolicInterval
    LΛ::Vector{Vector{Int64}}
    UΛ::Vector{Vector{Int64}}
end

function solve(solver::ReluVal, problem::Problem)
    reach = forward_network(solver, problem.network, problem.input)
    result = check_inclusion(reach.sym, problem.output, problem.network)
    result.status == :unknown || return result
    reach_list = SymbolicIntervalMask[reach]
    for i in 2:solver.max_iter
        length(reach_list) > 0 || return BasicResult(:holds)
        reach = pick_out!(reach_list, solver.tree_search)
        LG, UG = get_gradient(problem.network, reach.LΛ, reach.UΛ)
        feature = get_smear_index(problem.network, reach.sym.interval, LG, UG)
        intervals = split_interval(reach.sym.interval, feature)
        for interval in intervals
            reach = forward_network(solver, problem.network, interval)
            result = check_inclusion(reach.sym, problem.output, problem.network)
            result.status == :violated && return result
            result.status == :holds || (push!(reach_list, reach))
        end
    end
    return BasicResult(:unknown)
end

function pick_out!(reach_list, tree_search)
    if tree_search == :BFS
        reach = reach_list[1]
        deleteat!(reach_list, 1)
    else
        n = length(reach_list)
        reach = reach_list[n]
        deleteat!(reach_list, n)
    end
    return reach
end

function symbol_to_concrete(reach::SymbolicInterval)
    n_output = size(reach.Low, 1)
    upper = zeros(n_output)
    lower = zeros(n_output)
    for i in 1:n_output
        lower[i] = lower_bound(reach.Low[i, :], reach.interval)
        upper[i] = upper_bound(reach.Up[i, :], reach.interval)
    end
    return Hyperrectangle(low = lower, high = upper)
end

function check_inclusion(reach::SymbolicInterval, output::AbstractPolytope, nnet::Network)
    reachable = symbol_to_concrete(reach)
    issubset(reachable, output) && return BasicResult(:holds)
    # is_intersection_empty(reachable, output) && return BasicResult(:violated)

    # Sample the middle point
    middle_point = (high(reach.interval) + low(reach.interval))./2
    y = compute_output(nnet, middle_point)
    ∈(y, output) || return CounterExampleResult(:violated, middle_point)

    return BasicResult(:unknown)
end

function forward_layer(solver::ReluVal, layer::Layer, input::Union{SymbolicIntervalMask, Hyperrectangle})
    return forward_act(forward_linear(input, layer), layer)
end

# Symbolic forward_linear for the first layer
function forward_linear(input::Hyperrectangle, layer::Layer)
    (W, b) = (layer.weights, layer.bias)
    sym = SymbolicInterval(hcat(W, b), hcat(W, b), input)
    LΛ = Vector{Vector{Int64}}(undef, 0)
    UΛ = Vector{Vector{Int64}}(undef, 0)
    return SymbolicIntervalMask(sym, LΛ, UΛ)
end

# Symbolic forward_linear
function forward_linear(input::SymbolicIntervalMask, layer::Layer)
    (W, b) = (layer.weights, layer.bias)
    output_Up = max.(W, 0) * input.sym.Up + min.(W, 0) * input.sym.Low
    output_Low = max.(W, 0) * input.sym.Low + min.(W, 0) * input.sym.Up
    output_Up[:, end] += b
    output_Low[:, end] += b
    sym = SymbolicInterval(output_Low, output_Up, input.sym.interval)
    return SymbolicIntervalMask(sym, input.LΛ, input.UΛ)
end

# Symbolic forward_act
function forward_act(input::SymbolicIntervalMask, layer::Layer{ReLU})
    n_node, n_input = size(input.sym.Up)
    output_Low, output_Up = input.sym.Low[:, :], input.sym.Up[:, :]
    mask_lower, mask_upper = zeros(Int, n_node), ones(Int, n_node)
    for i in 1:n_node
        if upper_bound(input.sym.Up[i, :], input.sym.interval) <= 0.0
            # Update to zero
            mask_lower[i], mask_upper[i] = 0, 0
            output_Up[i, :] = zeros(n_input)
            output_Low[i, :] = zeros(n_input)
        elseif lower_bound(input.sym.Low[i, :], input.sym.interval) >= 0
            # Keep dependency
            mask_lower[i], mask_upper[i] = 1, 1
        else
            # Concretization
            mask_lower[i], mask_upper[i] = 0, 1
            output_Low[i, :] = zeros(n_input)
            if lower_bound(input.sym.Up[i, :], input.sym.interval) < 0
                output_Up[i, :] = zeros(n_input)
                output_Up[i, end] = upper_bound(input.sym.Up[i, :], input.sym.interval)
            end
        end
    end
    sym = SymbolicInterval(output_Low, output_Up, input.sym.interval)
    LΛ = push!(input.LΛ, mask_lower)
    UΛ = push!(input.UΛ, mask_upper)
    return SymbolicIntervalMask(sym, LΛ, UΛ)
end

# Symbolic forward_act
function forward_act(input::SymbolicIntervalMask, layer::Layer{Id})
    sym = input.sym
    n_node = size(input.sym.Up, 1)
    LΛ = push!(input.LΛ, ones(Int, n_node))
    UΛ = push!(input.UΛ, ones(Int, n_node))
    return SymbolicIntervalMask(sym, LΛ, UΛ)
end

# Return the splited intervals
function get_smear_index(nnet::Network, input::Hyperrectangle, LG::Matrix, UG::Matrix)
    largest_smear = - Inf
    feature = 0
    r = input.radius
    for i in 1:dim(input)
        smear = sum(max.(abs.(UG[:, i]), abs.(LG[:, i]))) * r[i]
        if smear > largest_smear
            largest_smear = smear
            feature = i
        end
    end
    return feature
end

# Get upper bound in concretization
function upper_bound(map::Vector{Float64}, input::Hyperrectangle)
    bound = map[dim(input)+1]
    input_upper = high(input)
    input_lower = low(input)
    for i in 1:dim(input)
        bound += ifelse( map[i]>0, map[i]*input_upper[i], map[i]*input_lower[i])
    end
    return bound
end

# Get lower bound in concretization
function lower_bound(map::Vector{Float64}, input::Hyperrectangle)
    bound = map[dim(input)+1]
    input_upper = high(input)
    input_lower = low(input)
    for i in 1:dim(input)
        bound += ifelse(map[i]>0, map[i]*input_lower[i], map[i]*input_upper[i])
    end
    return bound
end

# Concrete forward_linear
# function forward_linear_concrete(input::Hyperrectangle, W::Matrix{Float64}, b::Vector{Float64})
#     n_output = size(W, 1)
#     output_upper = zeros(n_output)
#     output_lower = zeros(n_output)
#     for j in 1:n_output
#         output_upper[j] = upper_bound(W[j, :], input) + b[j]
#         output_lower[j] = lower_bound(W[j, :], input) + b[j]
#     end
#     output = Hyperrectangle(low = output_lower, high = output_upper)
#     return output
# end

# Concrete forward_act
# function forward_act(input::Hyperrectangle)
#     input_upper = high(input)
#     input_lower = low(input)
#     output_upper = zeros(dim(input))
#     output_lower = zeros(dim(input))
#     mask_upper = ones(Int, dim(input))
#     mask_lower = zeros(Int, dim(input))
#     for i in 1:dim(input)
#         if input_upper[i] <= 0
#             mask_upper[i] = mask_lower[i] = 0
#             output_upper[i] = output_lower[i] = 0.0
#         elseif input_lower[i] >= 0
#             mask_upper[i] = mask_lower[i] = 1
#         else
#             output_lower[i] = 0.0
#         end
#     end
#     return (output_lower, output_upper, mask_lower, mask_upper)
# end

# This function is similar to forward_linear
# function backward_linear(Low::Matrix{Float64}, Up::Matrix{Float64}, W::Matrix{Float64})
#     n_output, n_input = size(W)
#     n_symbol = size(Low, 2) - 1

#     output_Low = zeros(n_output, n_symbol + 1)
#     output_Up = zeros(n_output, n_symbol + 1)
#     for k in 1:n_symbol + 1
#         for j in 1:n_output
#             for i in 1:n_input
#                 output_Up[j, k] += ifelse(W[j, i]>0, W[j, i] * Up[i, k], W[j, i] * Low[i, k])
#                 output_Low[j, k] += ifelse(W[j, i]>0, W[j, i] * Low[i, k], W[j, i] * Up[i, k])
#             end
#         end
#     end
#     return (output_Low, output_Up)
# end