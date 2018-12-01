
struct ReluVal
    max_iter::Int64
    tree_search::Symbol
end

ReluVal(x::Int64) = ReluVal(x, :DFS)
ReluVal() = ReluVal(10, :DFS)

struct SymbolicInterval
    L::Matrix{Float64}
    U::Matrix{Float64}
    interval::Hyperrectangle
end

SymbolicInterval(x::Matrix{Float64}, y::Matrix{Float64}) = SymbolicInterval(x, y, Hyperrectangle([0],[0]))

# Gradient mask for a single layer
struct GradientMask
    lower::Vector{Int64}
    upper::Vector{Int64}
end

# Data to be passed during forward_layer
struct SymbolicIntervalMask
    sym::SymbolicInterval
    mask::Vector{GradientMask}
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
            return CounterExampleResult(:SAT)
        end
        if solver.tree_search == :BFS
            reach = first(reach_list)
            popfirst!(reach_list)
        else
            reach = last(reach_list)
            pop!(reach_list)
        end
        gradient = back_prop(problem.network, reach.mask)
        intervals = split_input(problem.network, reach.sym.interval, gradient)
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
    return CounterExampleResult(:Unknown) # undetermined
end

# This overwrites check_inclusion in utils/reachability.jl
function check_inclusion(reach::SymbolicInterval, output::AbstractPolytope, nnet::Network)
    n_output = dim(output)
    n_input = dim(reach.interval)
    upper = zeros(n_output)
    lower = zeros(n_output)

    for i in 1:n_output
        lower[i] = lower_bound(reach.L[i, :], reach.interval)
        upper[i] = upper_bound(reach.L[i, :], reach.interval) # TODO both bounds are on L?
    end
    reachable = Hyperrectangle(low = lower, high = upper)

    if issubset(reachable, output)
        return CounterExampleResult(:SAT)
    end
    if is_intersection_empty(reachable, output)
        return CounterExampleResult(:UNSAT)
    end
    # Sample the middle point
    middle_point = (high(reach.interval) + low(reach.interval))./2
    if compute_output(nnet, middle_point) ∉ output
        return CounterExampleResult(:UNSAT, middle_point)
    end

    return CounterExampleResult(:Unknown)
end

function forward_layer(solver::ReluVal, layer::Layer, input::Union{SymbolicIntervalMask, Hyperrectangle})
    return forward_act(forward_linear(input, layer.weights, layer.bias))
end

# Concrete forward_linear
function forward_linear_concrete(input::Hyperrectangle, W::Matrix{Float64}, b::Vector{Float64})
    n_output = size(W, 1)
    output_upper = fill(0.0, n_output)
    output_lower = fill(0.0, n_output)
    for j in 1:n_output
        output_upper[j] = upper_bound(W[j, :], input) + b[j]
        output_lower[j] = lower_bound(W[j, :], input) + b[j]
    end
    output = Hyperrectangle(low = output_lower, high = output_upper)
    return output
end

# Symbolic forward_linear for the first layer
function forward_linear(input::Hyperrectangle, W::Matrix{Float64}, b::Vector{Float64})
    sym = SymbolicInterval(hcat(W, b), hcat(W, b), input)
    mask = GradientMask[]
    return SymbolicIntervalMask(sym, mask)
end

# Symbolic forward_linear
function forward_linear(input::SymbolicIntervalMask, W::Matrix{Float64}, b::Vector{Float64})
    n_output, n_input = size(W)
    n_symbol = size(input.sym.L, 2) - 1

    output_L = zeros(n_output, n_symbol + 1)
    output_U = zeros(n_output, n_symbol + 1)
    for k in 1:n_symbol + 1
        for j in 1:n_output
            for i in 1:n_input
                if W[j, i]>0
                    output_U[j, k] = W[j, i] * input.sym.U[i, k]
                    output_L[j, k] = W[j, i] * input.sym.L[i, k]
                else
                    output_U[j, k] = W[j, i] * input.sym.L[i, k]
                    output_L[j, k] = W[j, i] * input.sym.U[i, k]
                end
            end
            if k > n_symbol
                output_U[j, k] += b[j]
                output_L[j, k] += b[j]
            end
        end
    end
    sym = SymbolicInterval(output_L, output_U, input.sym.interval)
    mask = input.mask
    return SymbolicIntervalMask(sym, mask)
end

# Concrete forward_act
function forward_act(input::Hyperrectangle)
    input_upper = high(input)
    input_lower = low(input)
    output_upper = zeros(dim(input))
    output_lower = zeros(dim(input))
    mask_upper = fill(1, dim(input))
    mask_lower = fill(0, dim(input))
    for i in 1:dim(input)
        if input_upper[i] <= 0
            mask_upper[i] = mask_lower[i] = 0
            output_upper[i] = output_lower[i] = 0.0
        elseif input_lower[i] >= 0
            mask_upper[i] = mask_lower[i] = 1
        else
            output_lower[i] = 0.0
        end
    end
    return (output_lower, output_upper, mask_lower, mask_upper)
end

# Symbolic forward_act
function forward_act(input::SymbolicIntervalMask)
    n_output, n_input = size(input.sym.U)

    input_upper = high(input.sym.interval)
    input_lower = low(input.sym.interval)

    output_U = deepcopy(input.sym.U)
    output_L = deepcopy(input.sym.L)

    mask_upper = fill(1, n_output)
    mask_lower = fill(0, n_output)

    for i in 1:n_output
        if upper_bound(input.sym.U[i, :], input.sym.interval) <= 0.0
            # Update to zero
            mask_upper[i] = 0
            mask_lower[i] = 0
            output_U[i, :] = zeros(n_input)
            output_L[i, :] = zeros(n_input)
        elseif lower_bound(input.sym.L[i, :], input.sym.interval) >= 0
            # Keep dependency
            mask_upper[i] = 1
            mask_lower[i] = 1
        else
            # Concretization
            mask_upper[i] = 1
            mask_lower[i] = 0
            output_L[i, :] = zeros(n_input)
            if lower_bound(input.sym.U[i, :], input.sym.interval) <= 0
                output_U[i, :] = [zeros(n_input - 1);  upper_bound(input.sym.U[i, :];  input.sym.interval)] #concatination
            end
        end
    end
    sym = SymbolicInterval(output_L, output_U, input.sym.interval)
    mask = vcat(input.mask, GradientMask(mask_lower, mask_upper))
    return SymbolicIntervalMask(sym, mask)
end

# To be tested
function back_prop(nnet::Network, R::Vector{GradientMask})
    n_layer = length(nnet.layers)
    n_output = n_nodes(last(nnet.layers))
    # For now, assume the last layer is identity
    U = Matrix(1.0*I, n_output, n_output)
    L = Matrix(1.0*I, n_output, n_output)

    for k in n_layer:-1:1
        # back through activation function using the gradient mask
        n_node = n_nodes(nnet.layers[k])
        output_U = zeros(n_node, n_output)
        output_L = zeros(n_node, n_output)
        for i in 1:n_node

            output_U[i, :]  = ifelse(R[k].upper[i] > 0, U[i, :],  zeros(1, size(U, 2)))
            output_L[i, :] = ifelse(R[k].lower[i] > 0, L[i, :], zeros(1, size(L,2)))
        end
        # back through weight matrix
        (L, U) = backward_linear(output_L, output_U, pinv(nnet.layers[k].weights))
    end

    return SymbolicInterval(L, U)
end

# This function is similar to forward_linear
function backward_linear(L::Matrix{Float64}, U::Matrix{Float64}, W::Matrix{Float64})
    n_output, n_input = size(W)
    n_symbol = size(L, 2) - 1

    output_L = zeros(n_output, n_symbol + 1)
    output_U = zeros(n_output, n_symbol + 1)
    for k in 1:n_symbol + 1
        for j in 1:n_output
            for i in 1:n_input
                if W[j, i]>0
                    output_U[j,k]  += W[j,i] * U[i,k]
                    output_L[j,k] += W[j,i] * L[i,k]
                else
                    output_U[j,k]  += W[j,i] * L[i,k]
                    output_L[j,k] += W[j,i] * U[i,k]
                end
            end
        end
    end
    return (output_L, output_U)
end

# Return the split intervals
function split_input(nnet::Network, input::Hyperrectangle{T}, g::SymbolicInterval) where T
    largest_smear = -Inf
    feature = 0
    r = radius(input) .* 2
    for i in 1:dim(input)
        smear = get_smear(g.U[i, :], g.L[i, :], r[i])
        if smear > largest_smear
            largest_smear = smear
            feature = i
        end
    end

    input_split_left  = input_split(input, feature, false) # false for left
    input_split_right = input_split(input, feature, true)

    return (input_split_left, input_split_right)
end

function get_smear(up::Vector, low::Vector, rad::Float64)
    smear = 0.0
    for j in 1:size(g.U, 2)
        if up[j] > low[j]
            smear += up[j] * rad
        else
            smear -= low[j] * rad
        end
    end
    return smear
end

function input_split(input::Hyperrectangle, i::Int, right::Bool)
    hi = high(input)
    lo = low(input)
    hi[i] = input.center[i]
    if right
        lo[i] = input.center[i]
        hi[i] += input.radius[i]
    end
    return Hyperrectangle(low = lo, high = hi)
end

# Get upper bound in concretization
function upper_bound(mapping::Vector{Float64}, input::Hyperrectangle)
    bound = mapping[dim(input)+1]
    hi = high(input)
    lo = low(input)
    for i in 1:dim(input)
        a = (mapping[i]>0) ? hi[i] : lo[i]
        bound += a*mapping[i]
    end
    return bound
end

# Get lower bound in concretization
function lower_bound(mapping::Vector{Float64}, input::Hyperrectangle)
    bound = mapping[dim(input)+1]
    hi = high(input)
    lo = low(input)
    for i in 1:dim(input)
        a = (mapping[i]<0) ? hi[i] : lo[i]
        bound += a*mapping[i]
    end
    return bound
end
