struct ReluVal
    max_iter::Int64
    tree_search::Symbol
end

ReluVal() = ReluVal(10, :DFS)
ReluVal(x::Int64) = ReluVal(x, :DFS)

struct SymbolicInterval
    Low::Matrix{Float64}
    Up::Matrix{Float64}
    interval::Hyperrectangle
end

SymbolicInterval(x::Matrix{Float64}, y::Matrix{Float64}) = SymbolicInterval(x, y, Hyperrectangle([0],[0]))

# Gradient mask for a single layer
# struct GradientMask
#     lower::Vector{Int64}
#     upper::Vector{Int64}
# end

# Data to be passed during forward_layer
struct SymbolicIntervalMask
    sym::SymbolicInterval
    LΛ::Vector{Vector{Int64}}
    UΛ::Vector{Vector{Int64}}
end

function solve(solver::ReluVal, problem::Problem)
    # Compute the reachable set without splitting the interval
    reach = forward_network(solver, problem.network, problem.input)
    result = check_inclusion(reach.sym, problem.output, problem.network)
    if result.status != :Unknown
        return result
    end

    # If undertermined, split the interval
    # Bisection tree. Defult DFS.
    reach_list = SymbolicIntervalMask[reach]
    for i in 2:solver.max_iter
        if length(reach_list) == 0
            return BasicResult(:SAT)
        end
        if solver.tree_search == :BFS
            reach = reach_list[1]
            deleteat!(reach_list, 1)
        else
            n = length(reach_list)
            reach = reach_list[n]
            deleteat!(reach_list, n)
        end
        LG, UG = get_gradient(problem.network, reach.LΛ, reach.UΛ)
        feature = get_smear_index(problem.network, reach.sym.interval, LG, UG)
        intervals = split_interval(reach.sym.interval, feature)
        for interval in intervals
            reach = forward_network(solver, problem.network, interval)
            result = check_inclusion(reach.sym, problem.output, problem.network)
            if result.status == :UNSAT # If counter_example found
                return result
            elseif result.status == :Unknown # If undertermined, need to split
                reach_list = vcat(reach_list, reach)
            end
        end
    end
    return BasicResult(:Unknown) # undetermined
end

# This overwrites check_inclusion in utils/reachability.jl
function check_inclusion(reach::SymbolicInterval, output::AbstractPolytope, nnet::Network)
    n_output = dim(output)
    n_input = dim(reach.interval)
    upper = fill(0.0, n_output)
    lower = fill(0.0, n_output)

    for i in 1:n_output
        lower[i] = lower_bound(reach.Low[i, :], reach.interval)
        upper[i] = upper_bound(reach.Low[i, :], reach.interval)
    end
    reachable = Hyperrectangle(low = lower, high = upper)

    if issubset(reachable, output)
        return BasicResult(:SAT)
    end
    if is_intersection_empty(reachable, output)
        return BasicResult(:UNSAT)
    end
    # Sample the middle point
    middle_point = (high(reach.interval) + low(reach.interval))./2
    if ~∈(compute_output(nnet, middle_point), output)
        return CounterExampleResult(:UNSAT, middle_point)
    end

    return BasicResult(:Unknown)
end

function forward_layer(solver::ReluVal, layer::Layer, input::Union{SymbolicIntervalMask, Hyperrectangle})
    return forward_act(forward_linear(input, layer.weights, layer.bias))
end

# Symbolic forward_linear for the first layer
function forward_linear(input::Hyperrectangle, W::Matrix{Float64}, b::Vector{Float64})
    sym = SymbolicInterval(hcat(W, b), hcat(W, b), input)
    LΛ = Vector{Vector{Int64}}(undef, 0)
    UΛ = Vector{Vector{Int64}}(undef, 0)
    return SymbolicIntervalMask(sym, LΛ, UΛ)
end

# Symbolic forward_linear
function forward_linear(input::SymbolicIntervalMask, W::Matrix{Float64}, b::Vector{Float64})
    n_output, n_input = size(W)
    n_symbol = size(input.sym.Low, 2) - 1
    output_Low = zeros(n_output, n_symbol + 1)
    output_Up = zeros(n_output, n_symbol + 1)
    for k in 1:n_symbol + 1
        for j in 1:n_output
            for i in 1:n_input
                output_Up[j, k] += ifelse(W[j, i]>0, W[j, i] * input.sym.Up[i, k], W[j, i] * input.sym.Low[i, k])
                output_Low[j, k] += ifelse(W[j, i]>0, W[j, i] * input.sym.Low[i, k], W[j, i] * input.sym.Up[i, k])
            end
            if k > n_symbol
                output_Up[j, k] += b[j]
                output_Low[j, k] += b[j]
            end
        end
    end
    sym = SymbolicInterval(output_Low, output_Up, input.sym.interval)
    return SymbolicIntervalMask(sym, input.LΛ, input.UΛ)
end

# Symbolic forward_act
function forward_act(input::SymbolicIntervalMask)
    n_output, n_input = size(input.sym.Up)
    input_upper = high(input.sym.interval)
    input_lower = low(input.sym.interval)
    output_Up = input.sym.Up[:, :]
    output_Low = input.sym.Low[:, :]
    mask_upper = fill(1, n_output)
    mask_lower = fill(0, n_output)
    for i in 1:n_output
        if upper_bound(input.sym.Up[i, :], input.sym.interval) <= 0.0
            # Update to zero
            mask_upper[i] = 0
            mask_lower[i] = 0
            output_Up[i, :] = fill(0.0, n_input)
            output_Low[i, :] = fill(0.0, n_input)
        elseif lower_bound(input.sym.Low[i, :], input.sym.interval) >= 0
            # Keep dependency
            mask_upper[i] = 1
            mask_lower[i] = 1
        else
            # Concretization
            mask_upper[i] = 1
            mask_lower[i] = 0
            output_Low[i, :] = zeros(1, n_input)
            if lower_bound(input.sym.Up[i, :], input.sym.interval) <= 0
                output_Up[i, :] = hcat(zeros(1, n_input - 1), upper_bound(input.sym.Up[i, :], input.sym.interval))
            end
        end
    end
    sym = SymbolicInterval(output_Low, output_Up, input.sym.interval)
    LΛ = push!(input.LΛ, mask_lower)
    UΛ = push!(input.UΛ, mask_upper)
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
#     output_upper = fill(0.0, n_output)
#     output_lower = fill(0.0, n_output)
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
#     output_upper = fill(0.0, dim(input))
#     output_lower = fill(0.0, dim(input))
#     mask_upper = fill(1, dim(input))
#     mask_lower = fill(0, dim(input))
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