"""
    BaB(optimizer, ϵ::Float64)

BaB uses branch and bound to estimate the range of the output node.

# Problem requirement
1. Network: any depth, ReLU activation, single output
2. Input: hyperrectangle
3. Output: hyperrectangle (1d interval)

# Return
`CounterExampleResult` or `ReachabilityResult`

# Method
Branch and bound.
For branch, it uses iterative interval refinement.
For bound, it computes concrete bounds by sampling, approximated bound by optimization.
- `optimizer` default `GLPKSolverMIP()`
- `ϵ` is the desired accurancy for termination, default `0.1`.

# Property
Sound and complete.

# Reference
[R. Bunel, I. Turkaslan, P. H. Torr, P. Kohli, and M. P. Kumar,
"A Unified View of Piecewise Linear Neural Network Verification,"
*ArXiv Preprint ArXiv:1711.00455*, 2017.](https://arxiv.org/abs/1711.00455)
"""
@with_kw struct BaB
    optimizer = GLPK.Optimizer
    ϵ::Float64 = 0.1
end

function solve(solver::BaB, problem::Problem)
    (u_approx, u, x_u) = output_bound(solver, problem, :max)
    (l_approx, l, x_l) = output_bound(solver, problem, :min)
    bound = Hyperrectangle(low = [l], high = [u])
    reach = Hyperrectangle(low = [l_approx], high = [u_approx])
    return interpret_result(reach, bound, problem.output, x_l, x_u)
end

function interpret_result(reach, bound, output, x_l, x_u)
    if high(reach) < high(output) && low(reach) > low(output)
        return ReachabilityResult(:holds, [reach])
    end
    high(bound) > high(output)    && return CounterExampleResult(:violated, x_u)
    low(bound)  < low(output)     && return CounterExampleResult(:violated, x_l)
    return ReachabilityResult(:unknown, reach)
end

function output_bound(solver::BaB, problem::Problem, type::Symbol)
    nnet = problem.network
    global_concrete, x_star = concrete_bound(nnet, problem.input, type)
    global_approx = approx_bound(nnet, problem.input, solver.optimizer, type)
    doms = Tuple{Float64, Hyperrectangle}[(global_approx, problem.input)]
    index = ifelse(type == :max, 1, -1)
    while index * (global_approx - global_concrete) > solver.ϵ
        dom = pick_out(doms) # pick_out implements the search strategy
        subdoms = split_dom(dom[2]) # split implements the branching rule
        for i in 1:length(subdoms)
            dom_concrete, x = concrete_bound(nnet, subdoms[i], type) # Sample
            dom_approx = approx_bound(nnet, subdoms[i], solver.optimizer, type)
            if index * (dom_concrete - global_concrete) > 0
                (global_concrete, x_star) = (dom_concrete, x)
            end
            if index * (dom_approx - global_concrete) > 0
                add_domain!(doms, (dom_approx, subdoms[i]), type)
            end
        end
        global_approx = doms[1][1]
    end
    return (global_approx, global_concrete, x_star)
end

# Always pick the first dom
function pick_out(doms)
    return doms[1]
end

function add_domain!(doms::Vector{Tuple{Float64, Hyperrectangle}}, new::Tuple{Float64, Hyperrectangle}, type::Symbol)
    rank = length(doms) + 1
    index = ifelse(type == :max, 1, -1)
    for i in 1:length(doms)
        if index * (new[1] - doms[i][1]) >= 0
            rank = i
            break
        end
    end
    insert!(doms, rank, new)
end

# Always split the longest input dimension
function split_dom(dom::Hyperrectangle)
    max_value, index_to_split = findmax(dom.radius)
    return split_interval(dom, index_to_split)
end

# For simplicity
function concrete_bound(nnet::Network, subdom::Hyperrectangle, type::Symbol)
    points = [subdom.center, low(subdom), high(subdom)]
    values = Vector{Float64}(undef, 0)
    for p in points
        push!(values, sum(compute_output(nnet, p)))
    end
    value, index = ifelse(type == :min, findmin(values), findmax(values))
    return (value, points[index])
end


function approx_bound(nnet::Network, dom::Hyperrectangle, optimizer, type::Symbol)
    bounds = get_bounds(nnet, dom)
    model = Model(with_optimizer(optimizer))
    neurons = init_neurons(model, nnet)
    add_set_constraint!(model, dom, first(neurons))
    encode_network!(model, nnet, neurons, bounds, TriangularRelaxedLP())
    index = ifelse(type == :max, 1, -1)
    o = sum(last(neurons))
    @objective(model, Max, index * o)
    optimize!(model)
    termination_status(model) == OPTIMAL && return value(o)
    error("Could not find bound for dom: ", dom)
end
