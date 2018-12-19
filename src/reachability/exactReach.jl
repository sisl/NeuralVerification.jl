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
        push!(output, forward_positive(layer.activation, input[i]))
        push!(output, forward_negative(layer.activation, input[i]))
        append!(output, forward_undetermined(layer.activation, input[i]))
    end
    return output
end

function forward_layer(solver::ExactReach, layer::Layer, input::HPolytope)
    input = linear_transformation(layer, input)
    output = Vector{HPolytope}(undef, 0)
    push!(output, forward_positive(layer.activation, input))
    push!(output, forward_negative(layer.activation, input))
    append!(output, forward_undetermined(layer.activation, input))
    return output
end

function forward_positive(act::ReLU, input::HPolytope)
    C, d = tosimplehrep(input)
    n = dim(input)
    C = vcat(C, -Matrix(1.0I, n, n))
    d = vcat(d, zeros(n))
    return HPolytope(C, d)
end

function forward_negative(act::ReLU, input::HPolytope)
    n = dim(input)
    eye = Matrix(1.0I, n, n)
    if !HPolytope_intersection_empty(input, HPolytope(eye, zeros(n)))
        return HPolytope(vcat(eye, -eye), zeros(2*n))
    end
end

function forward_undetermined(act::ReLU, input::HPolytope)
    n = dim(input)
    output = Vector{HPolytope}(undef, 0)
    C, d = tosimplehrep(input)
    eye = Matrix(1.0I, n, n)
    for h in 1:2^n
        p = getP(h, n)
        Ch = vcat(C, eye - 2p)
        dh = vcat(d, zeros(n))
        set = HPolytope(Ch, dh)
        if !isempty(set)
            push!(output, linear_transformation(Matrix(p), set))
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

# This function is called in forward_negative
# NOTE: renamed function to avoid type piracy with LazySets. TODO: submit this function to them.
function HPolytope_intersection_empty(set_a::HPolytope, set_b::HPolytope)
    aVrep = tovrep(set_a)
    for v in vertices_list(aVrep)
        v in set_b || return false
    end
    bVrep = tovrep(set_b)
    for v in vertices_list(bVrep)
        v in set_a || return false
    end
    return true
end
