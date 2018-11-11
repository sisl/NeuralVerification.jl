
struct ExactReach end

function solve(solver::ExactReach, problem::Problem)
    reach = forward_network(solver, problem.network, problem.input)
    #println(reach)
    return check_inclusion(reach, problem.output)
end

function forward_layer(solver::ExactReach, layer::Layer, input::Vector{HPolytope})
    output = Vector{HPolytope}(undef, 0)
    for i in 1:length(input)
        append!(output, forward_positive(layer, input[i]))
        append!(output, forward_negative(layer, input[i]))
        append!(output, forward_undetermined(layer, input[i]))
    end
    return output
end

function forward_layer(solver::ExactReach, layer::Layer, input::HPolytope)
    output = Vector{HPolytope}(undef, 0)
    append!(output, forward_positive(layer, input))
    append!(output, forward_negative(layer, input))
    append!(output, forward_undetermined(layer, input))
    return output
end

function forward_positive(layer::Layer, input::HPolytope)
    inputA, inputb = tosimplehrep(input)
    n = n_nodes(layer)
    A = vcat(inputA*pinv(layer.weights), -Matrix(1.0I, n, n))
    b = vcat(inputb - inputA*pinv(layer.weights) * layer.bias, zeros(n))
    return HPolytope[HPolytope(A, b)]
end

function forward_negative(layer::Layer, input::HPolytope)
    # if is_intersection_empty(input, HPolytope(layer.weights, -layer.bias))
    if HPolytope_intersection_empty(input, HPolytope(layer.weights, -layer.bias))
        return HPolytope[]
    else
        n = length(layer.bias)
        eye = Matrix(1.0I, n, n)
        H = HPolytope(vcat(eye, -eye), zeros(2*n))
        return HPolytope[H]
    end
end

function forward_undetermined(layer::Layer, input::HPolytope)
    n = size(layer.weights, 1)
    output = Vector{HPolytope}(undef, 2^n)
    inputA, inputb = tosimplehrep(input)
    for h in 1:2^n
        IxAinv = inputA * pinv(layer.weights)
        p = getP(h, n)
        # A = vcat(IxAinv, Matrix(1.0I, n, n)-p, -p)
        # b = vcat(inputb - IxAinv * layer.bias, zeros(2*n))
        A = [IxAinv ; Matrix(1.0I, n, n)-p ; -p]
        b = [inputb - IxAinv * layer.bias ; zeros(2*n)]
        output[h] = HPolytope(A, b)
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

