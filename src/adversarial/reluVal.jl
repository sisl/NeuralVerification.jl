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
@with_kw struct ReluVal <: Solver
    max_iter::Int64     = 10
    tree_search::Symbol = :DFS # only :DFS/:BFS allowed? If so, we should assert this.
end


struct SymbolicInterval{F<:AbstractPolytope}
    Low::Matrix{Float64}
    Up::Matrix{Float64}
    domain::F
end
# Data to be passed during forward_layer
struct SymbolicIntervalGradient{F<:AbstractPolytope, N<:Real}
    sym::SymbolicInterval{F}
    LΛ::Vector{Vector{N}}
    UΛ::Vector{Vector{N}}
end
# Data to be passed during forward_layer
const SymbolicIntervalMask = SymbolicIntervalGradient{<:Hyperrectangle, Bool}

function _init_symbolic_grad_general(domain, N)
    n = dim(domain)
    I = Matrix{Float64}(LinearAlgebra.I(n))
    Z = zeros(n)
    symbolic_input = SymbolicInterval([I Z], [I Z], domain)
    symbolic_mask = SymbolicIntervalGradient(symbolic_input,
                                             Vector{Vector{N}}(),
                                             Vector{Vector{N}}())
end
function init_symbolic_grad(domain)
    VF = Vector{HalfSpace{Float64, Vector{Float64}}}
    domain = HPolytope(VF(constraints_list(domain)))
    _init_symbolic_grad_general(domain, Float64)
end
function init_symbolic_mask(interval)
    _init_symbolic_grad_general(interval, Bool)
end

function solve(solver::ReluVal, problem::Problem)
    isbounded(problem.input) || throw(UnboundedInputError("ReluVal can only handle bounded input sets."))

    reach_list = SymbolicIntervalMask[]
    for i in 1:solver.max_iter
        if i == 1
            intervals = [problem.input]
        else
            reach = select!(reach_list, solver.tree_search)
            intervals = bisect_interval_by_max_smear(problem.network, reach)
        end

        for interval in intervals
            reach = forward_network(solver, problem.network, init_symbolic_mask(interval))
            result = check_inclusion(reach.sym, problem.output, problem.network)

            if result.status === :violated
                return result
            elseif result.status === :unknown
                push!(reach_list, reach)
            end
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

function symbol_to_concrete(reach::SymbolicInterval{<:Hyperrectangle})
    lower = [lower_bound(l, reach.domain) for l in eachrow(reach.Low)]
    upper = [upper_bound(u, reach.domain) for u in eachrow(reach.Up)]
    return Hyperrectangle(low = lower, high = upper)
end

function check_inclusion(reach::SymbolicInterval{<:Hyperrectangle}, output, nnet::Network)
    reachable = symbol_to_concrete(reach)

    issubset(reachable, output) && return CounterExampleResult(:holds)

    # Sample the middle point
    middle_point = center(reach.domain)
    y = compute_output(nnet, middle_point)
    y ∈ output || return CounterExampleResult(:violated, middle_point)

    return CounterExampleResult(:unknown)
end

function forward_layer(solver::ReluVal, layer::Layer, input)
    return forward_act(solver, forward_linear(solver, input, layer), layer)
end

# Symbolic forward_linear
function forward_linear(solver::ReluVal, input::SymbolicIntervalMask, layer::Layer)
    (W, b) = (layer.weights, layer.bias)
    output_Low, output_Up = interval_map(W, input.sym.Low, input.sym.Up)
    output_Up[:, end] += b
    output_Low[:, end] += b
    sym = SymbolicInterval(output_Low, output_Up, input.sym.domain)
    return SymbolicIntervalGradient(sym, input.LΛ, input.UΛ)
end

# Symbolic forward_act
function forward_act(::ReluVal, input::SymbolicIntervalMask, layer::Layer{ReLU})

    interval = input.sym.domain
    Low, Up = input.sym.Low, input.sym.Up

    n_node = n_nodes(layer)
    output_Low, output_Up = copy(Low), copy(Up)
    LΛᵢ, UΛᵢ = falses(n_node), trues(n_node)

    for j in 1:n_node
        # Loop-local variable bindings for notational convenience.
        # These are direct views into the rows of the parent arrays.
        lowᵢⱼ, upᵢⱼ, out_lowᵢⱼ, out_upᵢⱼ = @views Low[j, :], Up[j, :], output_Low[j, :], output_Up[j, :]

        # If the upper bound of the upper bound is negative, set
        # the generators and centers of both bounds to 0, and
        # the gradient mask to 0
        if upper_bound(upᵢⱼ, interval) <= 0
            LΛᵢ[j], UΛᵢ[j] = 0, 0
            out_lowᵢⱼ .= 0
            out_upᵢⱼ .= 0

        # If the lower bound of the lower bound is positive,
        # the gradient mask should be 1
        elseif lower_bound(lowᵢⱼ, interval) >= 0
            LΛᵢ[j], UΛᵢ[j] = 1, 1

        # if the bounds overlap 0, concretize by setting
        # the generators to 0, and setting the new upper bound
        # center to be the current upper-upper bound.
        else
            LΛᵢ[j], UΛᵢ[j] = 0, 1
            out_lowᵢⱼ .= 0
            if lower_bound(upᵢⱼ, interval) < 0
                out_upᵢⱼ .= 0
                out_upᵢⱼ[end] = upper_bound(upᵢⱼ, interval)
            end
        end
    end

    sym = SymbolicInterval(output_Low, output_Up, interval)
    LΛ = push!(input.LΛ, LΛᵢ)
    UΛ = push!(input.UΛ, UΛᵢ)
    return SymbolicIntervalGradient(sym, LΛ, UΛ)
end

# Symbolic forward_act
function forward_act(::ReluVal, input::SymbolicIntervalMask, layer::Layer{Id})
    sym = input.sym
    n_node = size(input.sym.Up, 1)
    LΛ = push!(input.LΛ, trues(n_node))
    UΛ = push!(input.UΛ, trues(n_node))
    return SymbolicIntervalGradient(sym, LΛ, UΛ)
end

function get_max_smear_index(nnet::Network, input::Hyperrectangle, LG::Matrix, UG::Matrix)

    smear(lg, ug, r) = sum(max.(abs.(lg), abs.(ug))) * r

    ind = argmax(smear.(eachcol(LG), eachcol(UG), input.radius))
    monotone = all(>(0), LG[:, ind] .* UG[:, ind]) # NOTE should it be >= 0 instead?

    return ind, monotone
end

upper_bound(v, domain) = ρ(v[1:end-1], domain) + v[end]
lower_bound(v, domain) = -ρ(-v[1:end-1], domain) + v[end]
bounds(v, domain) = (lower_bound(v, domain), upper_bound(v, domain))
# radius of the symbolic interval in the direction of the
# jth generating vector. This is not the axis aligned radius,
# or the bounding radius, but rather a radius with respect to
# a node in the network. Equivalent to the upper-upper
# bound minus the lower-lower bound
function radius(sym::SymbolicInterval, j::Integer)
    upper_bound(@view(sym.Up[j, :]), sym.domain) -
    lower_bound(@view(sym.Low[j, :]), sym.domain)
end
