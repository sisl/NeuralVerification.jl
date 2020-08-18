"""
    Neurify(max_iter::Int64, tree_search::Symbol)

Neurify combines symbolic reachability analysis with constraint refinement to minimize over-approximation of the reachable set.

# Problem requirement
1. Network: any depth, ReLU activation
2. Input: AbstractPolytope
3. Output: AbstractPolytope

# Return
`CounterExampleResult` or `ReachabilityResult`

# Method
Symbolic reachability analysis and iterative interval refinement (search).
- `max_iter` default `10`.

# Property
Sound but not complete.

# Reference
[S. Wang, K. Pei, J. Whitehouse, J. Yang, and S. Jana,
"Efficient Formal Safety Analysis of Neural Networks,"
*CoRR*, vol. abs/1809.08098, 2018. arXiv: 1809.08098.](https://arxiv.org/pdf/1809.08098.pdf)
[https://github.com/tcwangshiqi-columbia/Neurify](https://github.com/tcwangshiqi-columbia/Neurify)
"""
@with_kw struct Neurify
    max_iter::Int64     = 10
    tree_search::Symbol = :DFS # only :DFS/:BFS allowed? If so, we should assert this.
end

struct SymbolicInterval{F<:AbstractPolytope}
    Low::Matrix{Float64}
    Up::Matrix{Float64}
    interval::F
end

SymbolicInterval(x::Matrix{Float64}, y::Matrix{Float64}) = SymbolicInterval(x, y, Hyperrectangle([0],[0]))

# Data to be passed during forward_layer
struct SymbolicIntervalGradient
    sym::SymbolicInterval
    LΛ::Vector{Vector{Float64}}
    UΛ::Vector{Vector{Float64}}
end

function solve(solver::Neurify, problem::Problem)
    reach = forward_network(solver, problem.network, problem.input)
    result = check_inclusion(reach.sym, problem.output, problem.network)
    result.status == :unknown || return result
    reach_list = SymbolicIntervalGradient[reach]
    for i in 2:solver.max_iter
        length(reach_list) > 0 || return BasicResult(:holds)
        reach = pick_out!(reach_list, solver.tree_search)
        intervals = constraint_refinement(solver, problem.network, reach)
        for interval in intervals
            reach = forward_network(solver, problem.network, interval)
            result = check_inclusion(reach.sym, problem.output, problem.network)
            result.status == :violated && return result
            result.status == :holds || (push!(reach_list, reach))
        end
    end
    return BasicResult(:unknown)
end

function symbol_to_concrete(reach::SymbolicInterval{HPolytope{N}}) where N
    n_output = size(reach.Low, 1)
    vertices = tovrep(reach.interval).vertices
    new_v_up = vertices
    new_v_low = vertices
    for (i, v) in enumerate(vertices)
        new_v_up[i] = reach.Low[:, 1:end-1] * v + reach.Low[:, end]
        new_v_low[i] = reach.Up[:, 1:end-1] * v + reach.Up[:, end]
    end
    V_up = VPolytope(new_v_up)
    V_low = VPolytope(new_v_low)
    return convex_hull([V_up, V_low])
end

function check_inclusion(reach::SymbolicInterval{HPolytope{N}}, output::AbstractPolytope, nnet::Network) where N
    reachable = symbol_to_concrete(reach)
    flag = true
    for v in reachable.vertices
        if ~∈(v, output)
            flag = false
            y = compute_output(nnet, v)
            ∈(y, output) || return CounterExampleResult(:violated, v)
        end
    end
    return ifelse(flag, BasicResult(:holds), BasicResult(:unknown))
end

function constraint_refinement(solver::Neurify, nnet::Network, reach::SymbolicIntervalGradient)
    i, j, gradient = get_nodewise_gradient(nnet, reach.LΛ, reach.UΛ)
    # We can generate four more constraints
    # Symbolic representation of node i j is Low[i][j,:] and Up[i][j,:]
    nnet_new = Network(nnet.layers[1:i])
    reach_new = forward_network(solver, nnet_new, reach.sym.interval)
    C, d = tosimplehrep(reach.sym.interval)
    l_sym = reach_new.sym.Low[j, 1:end-1]
    l_off = reach_new.sym.Low[j, end]
    u_sym = reach_new.sym.Up[j, 1:end-1]
    u_off = reach_new.sym.Up[j, end]
    intervals = Vector{HPolytope{Float64}}(undef, 4)
    intervals[1] = HPolytope([C; l_sym; u_sym], [d; -l_off; -u_off])
    intervals[2] = HPolytope([C; l_sym; -u_sym], [d; -l_off; u_off])
    intervals[3] = HPolytope([C; -l_sym; u_sym], [d; l_off; -u_off])
    intervals[4] = HPolytope([C; -l_sym; -u_sym], [d; l_off; u_off])
    return intervals
end

function get_nodewise_gradient(nnet::Network, LΛ::Vector{Vector{Float64}}, UΛ::Vector{Vector{Float64}})
    n_output = size(nnet.layers[end].weights, 1)
    n_length = length(nnet.layers)
    LG = Matrix(1.0I, n_output, n_output)
    UG = Matrix(1.0I, n_output, n_output)
    max_tuple = (0, 0, 0.0)
    for (k, layer) in enumerate(reverse(nnet.layers))
        i = n_length - k + 1
        for j in size(layer.bias)
            if LΛ[i][j] ∈ (0.0, 1.0) && UΛ[i][j] ∈ (0.0, 1.0)
                max_gradient = max(abs(LG[j]), abs(UG[j]))
                if max_gradient > max_tuple[3]
                    max_tuple = (i, j, max_gradient)
                end
            end
        end
        i >= 1 || break
        LG_hat = max.(LG, 0.0) * Diagonal(LΛ[i]) + min.(LG, 0.0) * Diagonal(UΛ[i])
        UG_hat = min.(UG, 0.0) * Diagonal(LΛ[i]) + max.(UG, 0.0) * Diagonal(UΛ[i])
        LG, UG = interval_map_right(layer.weights, LG_hat, UG_hat)
    end
    return max_tuple
end

function forward_layer(solver::Neurify, layer::Layer, input)
    return forward_act(forward_linear(solver, input, layer), layer)
end

# Symbolic forward_linear for the first layer
function forward_linear(solver::Neurify, input::AbstractPolytope, layer::Layer)
    (W, b) = (layer.weights, layer.bias)
    sym = SymbolicInterval(hcat(W, b), hcat(W, b), input)
    LΛ = Vector{Vector{Int64}}(undef, 0)
    UΛ = Vector{Vector{Int64}}(undef, 0)
    return SymbolicIntervalGradient(sym, LΛ, UΛ)
end

# Symbolic forward_linear
function forward_linear(solver::Neurify, input::SymbolicIntervalGradient, layer::Layer)
    (W, b) = (layer.weights, layer.bias)
    output_Up = max.(W, 0) * input.sym.Up + min.(W, 0) * input.sym.Low
    output_Low = max.(W, 0) * input.sym.Low + min.(W, 0) * input.sym.Up
    output_Up[:, end] += b
    output_Low[:, end] += b
    sym = SymbolicInterval(output_Low, output_Up, input.sym.interval)
    return SymbolicIntervalGradient(sym, input.LΛ, input.UΛ)
end

# Symbolic forward_act
function forward_act(input::SymbolicIntervalGradient, layer::Layer{ReLU})
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
            # Symbolic linear relaxation
            # This is different from ReluVal
            up_up = upper_bound(input.sym.Up[i, :], input.sym.interval)
            up_low = lower_bound(input.sym.Up[i, :], input.sym.interval)
            low_up = upper_bound(input.sym.Low[i, :], input.sym.interval)
            low_low = lower_bound(input.sym.Low[i, :], input.sym.interval)
            up_slop = up_up / (up_up - up_low)
            low_slop = low_up / (low_up - low_low)
            output_Low[i, :] =  low_slop * output_Low[i, :]
            output_Up[i, end] -= up_low
            output_Up[i, :] = up_slop * output_Up[i, :]
            mask_lower[i], mask_upper[i] = low_slop, up_slop
        end
    end
    sym = SymbolicInterval(output_Low, output_Up, input.sym.interval)
    LΛ = push!(input.LΛ, mask_lower)
    UΛ = push!(input.UΛ, mask_upper)
    return SymbolicIntervalGradient(sym, LΛ, UΛ)
end

function forward_act(input::SymbolicIntervalGradient, layer::Layer{Id})
    sym = input.sym
    n_node = size(input.sym.Up, 1)
    LΛ = push!(input.LΛ, ones(Float64, n_node))
    UΛ = push!(input.UΛ, ones(Float64, n_node))
    return SymbolicIntervalGradient(sym, LΛ, UΛ)
end

function upper_bound(map::Vector{Float64}, input::HPolytope)
    vertices = tovrep(input).vertices
    up = -Inf
    for v in vertices
        up = max(up, map'*[v;1.0])
    end
    return up
end

function lower_bound(map::Vector{Float64}, input::HPolytope)
    vertices = tovrep(input).vertices
    low = Inf
    for v in vertices
        low = min(low, map'*[v;1.0])
    end
    return low
end
