"""
    ExactReach

ExactReach performs exact reachability analysis to compute the output reachable set for a network.

# Problem requirement
1. Network: any depth, ReLU activation
2. Input: HPolytope
3. Output: HPolytope

# Return
`ReachabilityResult`

# Method
Exact reachability analysis.

# Property
Sound and complete.

# Reference
W. Xiang, H.-D. Tran, and T. T. Johnson,
"Reachable Set Computation and Safety Verification for Neural Networks with ReLU Activations,"
*ArXiv Preprint ArXiv:1712.08163*, 2017.
"""
struct ExactReach end

function solve(solver::ExactReach, problem::Problem)
    reach = forward_network(solver, problem.network, problem.input)
    return check_inclusion(reach, problem.output)
end

function forward_layer(solver::ExactReach, layer::Layer, input::Vector{HPolytope})
    output = Vector{HPolytope}(undef, 0)
    for i in 1:length(input)
        input[i] = linear_transformation(layer, input[i])
        append!(output, forward_partition(layer.activation, input[i]))
    end
    return output
end

function forward_layer(solver::ExactReach, layer::Layer, input::HPolytope)
    input = linear_transformation(layer, input)
    return forward_partition(layer.activation, input)
end

function forward_partition(act::ReLU, input::HPolytope)
    n = dim(input)
    output = Vector{HPolytope}(undef, 0)
    C, d = tosimplehrep(input)
    dh = [d; zeros(n)]
    for h in 0:2^n-1
        P = getP(h, n)
        Ch = [C; I - 2P]
        input_h = HPolytope(Ch, dh)
        if !isempty(input_h)
            push!(output, linear_transformation(Matrix(P), input_h))
        end
    end
    return output
end

function getP(h::Int64, n::Int64)
    str = string(h-1, pad = n, base = 2)
    vec = Vector{Int64}(undef, n)
    for i in 1:n
        vec[i] = ifelse(str[i] == '1', 1, 0)
    end
    return Diagonal(vec)
end