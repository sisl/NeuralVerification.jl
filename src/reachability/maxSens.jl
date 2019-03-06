"""
    MaxSens(resolution::Float64, tight::Bool)

MaxSens performs over-approximated reachability analysis to compute the over-approximated output reachable set for a network.

# Problem requirement
1. Network: any depth, any activation that is monotone
2. Input: `Hyperrectangle` or `HPolytope`
3. Output: `AbstractPolytope`

# Return
`ReachabilityResult`

# Method
First partition the input space into small grid cells according to `resolution`.
Then use interval arithmetic to compute the reachable set for each cell.
Two versions of interval arithmetic is implemented with indicator `tight`.
Default `resolution` is `1.0`. Default `tight = false`.

# Property
Sound but not complete.

# Reference
[W. Xiang, H.-D. Tran, and T. T. Johnson,
"Output Reachable Set Estimation and Verification for Multi-Layer Neural Networks,"
*ArXiv Preprint ArXiv:1708.03322*, 2017.](https://arxiv.org/abs/1708.03322)
"""
@with_kw struct MaxSens
    resolution::Float64 = 1.0
    tight::Bool         = false
end

# This is the main function
function solve(solver::MaxSens, problem::Problem)
    inputs = partition(problem.input, solver.resolution)
    f_n(x) = forward_network(solver, problem.network, x)
    outputs = map(f_n, inputs)
    return check_inclusion(outputs, problem.output)
end

# This function is called by forward_network
function forward_layer(solver::MaxSens, L::Layer, input::Hyperrectangle)
    (W, b, act) = (L.weights, L.bias, L.activation)
    center = zeros(size(W, 1))
    gamma  = zeros(size(W, 1))
    for j in 1:size(W, 1)
        node = Node(W[j,:], b[j], act)
        center[j], gamma[j] = forward_node(solver, node, input)
    end
    return Hyperrectangle(center, gamma)
end

function forward_node(solver::MaxSens, node::Node, input::Hyperrectangle)
    output    = node.w' * input.center + node.b
    deviation = sum(abs.(node.w) .* input.radius)
    β    = node.act(output)  # TODO expert suggestion for variable name. beta? β? O? x?
    βmax = node.act(output + deviation)
    βmin = node.act(output - deviation)
    if solver.tight
        return ((βmax + βmin)/2, (βmax - βmin)/2)
    else
        return (β, max(abs(βmax - β), abs(βmin - β)))
    end
end

function partition(input::Hyperrectangle, delta::Float64)
    n_dim = dim(input)
    lower, upper = low(input), high(input)
    radius = fill(delta/2, n_dim)

    hyperrectangle_list = [1; ceil.(Int, (upper - lower)/delta)]
    n_hyperrectangle = prod(hyperrectangle_list)

    hyperrectangles = Vector{Hyperrectangle}(undef, n_hyperrectangle)
    for k in 1:n_hyperrectangle
        n = k
        center = Vector{Float64}(undef, n_dim)
        for i in n_dim:-1:1
            id = div(n-1, hyperrectangle_list[i])
            n = mod(n-1, hyperrectangle_list[i])+1
            center[i] = lower[i] + delta * (id + 0.5);
        end
        hyperrectangles[k] = Hyperrectangle(center, radius)
    end
    return hyperrectangles
end

function partition(input::HPolytope, delta::Float64)
    @info "MaxSens overapproximates HPolytope input sets as Hyperrectangles."
    partition(overapproximate(input), delta)
end