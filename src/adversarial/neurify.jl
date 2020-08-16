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

@with_kw struct Neurify <: Solver
    max_iter::Int64     = 100
    tree_search::Symbol = :DFS # only :DFS/:BFS allowed? If so, we should assert this.
    optimizer = GLPK.Optimizer
end

struct SymbolicInterval{F<:AbstractPolytope}
    Low::Matrix{Float64}
    Up::Matrix{Float64}
    interval::F
end

# Data to be passed during forward_layer
struct SymbolicIntervalGradient
    sym::SymbolicInterval
    LΛ::Vector{Vector{Float64}} # mask for computing gradient.
    UΛ::Vector{Vector{Float64}}
    r::Vector{Vector{Float64}} # range of a input interval (upper_bound - lower_bound)
end

function solve(solver::Neurify, problem::Problem)

    # TODO get rid of
    function to_float_hrep(set)
        VF = Vector{HalfSpace{Float64, Vector{Float64}}}
        HPolytope(VF(constraints_list(set)))
    end
    problem = Problem(problem.network, to_float_hrep(problem.input), to_float_hrep(problem.output))


    model = Model(solver)
    set_silent(model)

    x = @variable(model, [1:dim(problem.input)])
    add_set_constraint!(model, problem.input, x)

    reach = forward_network(solver, problem.network, problem.input)
    result, max_violation_con = check_inclusion(solver, reach.sym, problem.output, problem.network) # This calls the check_inclusion function in ReluVal, because the constraints are Hyperrectangle
    result.status == :unknown || return result

    reach_list = [(reach, max_violation_con, Set())]

    # Becuase of over-approximation, a split may not bisect the input set.
    # Therefore, the gradient remains unchanged (since input didn't change).
    # And this node will be chosen to split forever.
    # To prevent this, we split each node only once if the gradient of this node hasn't change.
    # Each element in splits is a tuple (gradient_of_the_node, layer_index, node_index).
    # splits = Set() # To prevent infinity loop.

    for i in 2:solver.max_iter
        isempty(reach_list) && return BasicResult(:holds)

        reach, max_violation_con, splits = select!(reach_list, solver.tree_search)

        intervals = constraint_refinement(solver, problem.network, reach, max_violation_con, splits)

        for interval in intervals
            reach = forward_network(solver, problem.network, interval)
            result, max_violation_con = check_inclusion(solver, reach.sym, problem.output, problem.network)
            if result.status == :violated
                return result
            elseif result.status == :unknown
                push!(reach_list, (reach, max_violation_con, copy(splits)))
            end
        end
    end
    return BasicResult(:unknown)
end

function check_inclusion(solver::Neurify, reach::SymbolicInterval{<:HPolytope},
                         output::AbstractPolytope, nnet::Network)
    # The output constraint is in the form A*x < b
    # We try to maximize output constraint to find a violated case, or to verify the inclusion.
    # Suppose the output is [1, 0, -1] * x < 2, Then we are maximizing reach.Up[1] * 1 + reach.Low[3] * (-1)

    model = Model(solver)
    set_silent(model)

    x = @variable(model, [1:dim(reach.interval)])
    add_set_constraint!(model, reach.interval, x)

    output_constraints = constraints_list(output)

    max_violation = 0.0
    max_con_ind = 0
    # max_violation_con = nothing
    for (i, cons) in enumerate(output_constraints)
        # NOTE can be taken out of the loop, but maybe there's no advantage
        # NOTE max.(M, 0) * U  + ... is a common operation, and maybe should get a name. It's also an "interval map".
        a, b = cons.a, cons.b
        obj = max.(a, 0)'*reach.Up + min.(a, 0)'*reach.Low

        @objective(model, Max, obj * [x; 1] - b)
        optimize!(model)

        if termination_status(model) == OPTIMAL
            if compute_output(nnet, value(x)) ∉ output
                return CounterExampleResult(:violated, value(x)), nothing
            end

            viol = objective_value(model)
            if viol > max_violation
                max_violation = viol
                max_con_ind = i
            end

        # NOTE This entire else branch should be eliminated for the paper version
        else
            # NOTE Is this even valid if the problem isn't solved optimally?
            if value(x) ∈ reach.interval
                error("Not OPTIMAL, but x in the input set.\n
                This is usually caused by open input set.\n
                Please check your input constraints.")
            end
            # TODO can we be more descriptive?
            error("No solution, please check the problem definition.")
        end

    end

    if max_violation > 0.0
        return CounterExampleResult(:unknown), output_constraints[max_con_ind].a
    else
        return CounterExampleResult(:holds),   nothing
    end
end

function constraint_refinement(solver::Neurify,
                               nnet::Network,
                               reach::SymbolicIntervalGradient,
                               max_violation_con::AbstractVector{Float64},
                               splits)

    i, j, influence = get_max_nodewise_influence(solver, nnet, reach, max_violation_con, splits)
    # We can generate three more constraints
    # Symbolic representation of node i j is Low[i][j,:] and Up[i][j,:]
    reach_new = forward_network(solver, Network(nnet.layers[1:i]), reach.sym.interval)

    aL, bL = reach_new.sym.Low[j, 1:end-1], reach_new.sym.Low[j, end]
    aU, bU = reach_new.sym.Up[j, 1:end-1], reach_new.sym.Up[j, end]

    # custom intersection function that doesn't do constraint pruning
    ∩ = (set, lc) -> HPolytope([constraints_list(set); lc])

    subsets = [reach.sym.interval]

    # If either of the normal vectors is the 0-vector, we must skip it.
    # It cannot be used to create a halfspace constraint.
    # NOTE: how can this come about, and does it mean anything?
    if !iszero(aL)
        subsets = subsets .∩ [HalfSpace(aL, -bL), HalfSpace(aL, -bL), HalfSpace(-aL, bL)]
    end
    if !iszero(aU)
        subsets = subsets .∩ [HalfSpace(aU, -bU), HalfSpace(-aU, bU), HalfSpace(-aU, bU)]
    end
    return filter(!isempty, subsets)
end


function get_max_nodewise_influence(solver::Neurify,
                                    nnet::Network,
                                    reach::SymbolicIntervalGradient,
                                    max_violation_con::AbstractVector{Float64},
                                    splits)

    LΛ, UΛ, radii = reach.LΛ, reach.UΛ, reach.r
    is_ambiguous_activation(i, j) = (0 < LΛ[i][j] < 1) || (0 < UΛ[i][j] < 1)

    # We want to find the node with the largest influence
    # Influence is defined as gradient * interval width
    # The gradient is with respect to a loss defined by the most violated constraint.
    LG = UG = max_violation_con
    i_max, j_max, influence_max = 0, 0, -Inf

    # Backpropagation to calculate the node-wise gradient
    for i in reverse(1:length(nnet.layers))
        layer = nnet.layers[i]
        if layer.activation isa ReLU
            for j in 1:n_nodes(layer)
                if is_ambiguous_activation(i, j)
                    # taking `influence = max_gradient * reach.r[i][j]*k` would be
                    # different from original paper, but can improve the split efficiency.
                    # where `k = n-i+1`, i.e. counts up from 1 as you go back in layers.
                    influence = max(abs(LG[j]), abs(UG[j])) * radii[i][j]
                    if influence >= influence_max && (i, j, influence) ∉ splits
                        i_max, j_max, influence_max = i, j, influence
                    end
                end
            end
        end

        LG_hat = max.(LG, 0.0) .* LΛ[i] .+ min.(LG, 0.0) .* UΛ[i]
        UG_hat = min.(UG, 0.0) .* LΛ[i] .+ max.(UG, 0.0) .* UΛ[i]

        LG, UG = interval_map(layer.weights', LG_hat, UG_hat)
    end

    # NOTE can omit this line in the paper version
    (i_max == 0 || j_max == 0) && error("Can not find valid node to split")

    push!(splits, (i_max, j_max, influence_max))

    return (i_max, j_max, influence_max)
end

function forward_layer(solver::Neurify, layer::Layer, input)
    return forward_act(solver, forward_linear(solver, input, layer), layer)
end

# Symbolic forward_linear for the first layer
function forward_linear(solver::Neurify, input::AbstractPolytope, layer::Layer)
    (W, b) = (layer.weights, layer.bias)
    sym = SymbolicInterval(hcat(W, b), hcat(W, b), input)
    LΛ = Vector{Vector{Int64}}(undef, 0)
    UΛ = Vector{Vector{Int64}}(undef, 0)
    r = Vector{Vector{Int64}}(undef, 0)
    return SymbolicIntervalGradient(sym, LΛ, UΛ, r)
end

# Symbolic forward_linear
function forward_linear(solver::Neurify, input::SymbolicIntervalGradient, layer::Layer)
    (W, b) = (layer.weights, layer.bias)
    output_Up = max.(W, 0) * input.sym.Up + min.(W, 0) * input.sym.Low
    output_Low = max.(W, 0) * input.sym.Low + min.(W, 0) * input.sym.Up
    output_Up[:, end] += b
    output_Low[:, end] += b
    sym = SymbolicInterval(output_Low, output_Up, input.sym.interval)
    return SymbolicIntervalGradient(sym, input.LΛ, input.UΛ, input.r)
end

# Symbolic forward_act
function forward_act(solver::Neurify, input::SymbolicIntervalGradient, layer::Layer{ReLU})
    n_node, n_input = size(input.sym.Up)
    output_Low, output_Up = copy(input.sym.Low), copy(input.sym.Up)
    mask_lower, mask_upper = zeros(Float64, n_node), ones(Float64, n_node)
    interval_width = zeros(Float64, n_node)
    for i in 1:n_node
        # Symbolic linear relaxation
        # This is different from ReluVal

        up_low, up_up = bounds(input.sym.Up[i, :], input.sym.interval)
        low_low, low_up = bounds(input.sym.Low[i, :], input.sym.interval)

        interval_width[i] = up_up - low_low

        up_slope = act_gradient(up_low, up_up)
        low_slope = act_gradient(low_low, low_up)

        output_Up[i, :] = up_slope * output_Up[i, :]
        output_Up[i, end] += up_slope * max(-up_low, 0)

        output_Low[i, :] = low_slope * output_Low[i, :]

        mask_lower[i], mask_upper[i] = low_slope, up_slope
    end
    sym = SymbolicInterval(output_Low, output_Up, input.sym.interval)
    LΛ = push!(input.LΛ, mask_lower)
    UΛ = push!(input.UΛ, mask_upper)
    r = push!(input.r, interval_width)
    return SymbolicIntervalGradient(sym, LΛ, UΛ, r)
end

function forward_act(solver::Neurify, input::SymbolicIntervalGradient, layer::Layer{Id})
    sym = input.sym
    n_node = size(input.sym.Up, 1)
    LΛ = push!(input.LΛ, ones(Float64, n_node))
    UΛ = push!(input.UΛ, ones(Float64, n_node))
    r = push!(input.r, ones(Float64, n_node))
    return SymbolicIntervalGradient(sym, LΛ, UΛ, r)
end
