"""
    ReluVal(max_iter::Int64, tree_search::Symbol)

ReluVal combines symbolic reachability analysis with iterative interval refinement to
minimize over-approximation of the reachable set.

# Problem requirement
1. Network: any depth, ReLU activation
2. Input: Hyperrectangle
3. Output: LazySet

# Return
`CounterExampleResult`

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
@with_kw struct ReluVal <: Solver
    max_iter::Int64     = 10
    tree_search::Symbol = :DFS # only :DFS/:BFS allowed? If so, we should assert this.
end

function solve(solver::ReluVal, problem::Problem)
    isbounded(problem.input) || throw(UnboundedInputError("ReluVal can only handle bounded input sets."))

    reach_list = []
    interval = problem.input
    for i in 1:solver.max_iter
        if i > 1
            interval = select!(reach_list, solver.tree_search)
        end
        reach = forward_network(solver, problem.network, init_symbolic_mask(interval))
        result = check_inclusion(reach.sym, problem.output, problem.network)

        if result.status === :violated
            return result
        elseif result.status === :unknown
            intervals = bisect_interval_by_max_smear(problem.network, reach)
            append!(reach_list, intervals)
        end
        isempty(reach_list) && return CounterExampleResult(:holds)
    end
    return CounterExampleResult(:unknown)
end

function bisect_interval_by_max_smear(nnet::Network, reach::SymbolicIntervalMask)
    LG, UG = get_gradient_bounds(nnet, reach.LΛ, reach.UΛ)
    feature, monotone = get_max_smear_index(nnet, reach.sym.domain, LG, UG) #monotonicity not used in this implementation.
    return collect(split_interval(reach.sym.domain, feature))
end

function select!(reach_list, tree_search)
    if tree_search == :BFS
        reach = popfirst!(reach_list)
    elseif tree_search == :DFS
        reach = pop!(reach_list)
    else
        throw(ArgumentError(":$tree_search is not a valid tree search strategy"))
    end
    return reach
end

function check_inclusion(reach::SymbolicInterval{<:Hyperrectangle}, output, nnet::Network)
    reachable = Hyperrectangle(low = low(reach), high = high(reach))

    issubset(reachable, output) && return CounterExampleResult(:holds)

    # Sample the middle point
    middle_point = center(domain(reach))
    y = compute_output(nnet, middle_point)
    y ∈ output || return CounterExampleResult(:violated, middle_point)

    return CounterExampleResult(:unknown)
end

# Symbolic forward_linear
function forward_linear(solver::ReluVal, L::Layer, input::SymbolicIntervalMask)
    output_Low, output_Up = interval_map(L.weights, input.sym.Low, input.sym.Up)
    output_Up[:, end] += L.bias
    output_Low[:, end] += L.bias
    sym = SymbolicInterval(output_Low, output_Up, domain(input))
    return SymbolicIntervalGradient(sym, input.LΛ, input.UΛ)
end

# Symbolic forward_act
function forward_act(::ReluVal, L::Layer{ReLU},  input::SymbolicIntervalMask)
    output_Low, output_Up = copy(input.sym.Low), copy(input.sym.Up)
    n_node = n_nodes(L)
    LΛᵢ, UΛᵢ = falses(n_node), trues(n_node)

    for j in 1:n_node
        # If the upper bound of the upper bound is negative, set
        # the generators and centers of both bounds to 0, and
        # the gradient mask to 0
        if upper_bound(upper(input), j) <= 0
            LΛᵢ[j], UΛᵢ[j] = 0, 0
            output_Low[j, :] .= 0
            output_Up[j, :] .= 0

        # If the lower bound of the lower bound is positive,
        # the gradient mask should be 1
        elseif lower_bound(lower(input), j) >= 0
            LΛᵢ[j], UΛᵢ[j] = 1, 1

        # if the bounds overlap 0, concretize by setting
        # the generators to 0, and setting the new upper bound
        # center to be the current upper-upper bound.
        else
            LΛᵢ[j], UΛᵢ[j] = 0, 1
            output_Low[j, :] .= 0
            if lower_bound(upper(input), j) < 0
                output_Up[j, :] .= 0
                output_Up[j, end] = upper_bound(upper(input), j)
            end
        end
    end

    sym = SymbolicInterval(output_Low, output_Up, domain(input))
    LΛ = push!(input.LΛ, LΛᵢ)
    UΛ = push!(input.UΛ, UΛᵢ)
    return SymbolicIntervalGradient(sym, LΛ, UΛ)
end

# Symbolic forward_act
function forward_act(::ReluVal, L::Layer{Id}, input::SymbolicIntervalMask)
    n_node = size(input.sym.Up, 1)
    LΛ = push!(input.LΛ, trues(n_node))
    UΛ = push!(input.UΛ, trues(n_node))
    return SymbolicIntervalGradient(input.sym, LΛ, UΛ)
end

function get_max_smear_index(nnet::Network, input::Hyperrectangle, LG::Matrix, UG::Matrix)

    smear(lg, ug, r) = sum(max.(abs.(lg), abs.(ug))) * r

    ind = argmax(smear.(eachcol(LG), eachcol(UG), input.radius))
    monotone = all(>(0), LG[:, ind] .* UG[:, ind]) # NOTE should it be >= 0 instead?

    return ind, monotone
end
