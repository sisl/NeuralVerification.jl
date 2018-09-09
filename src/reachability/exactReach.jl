import LazySets.HPolytope # Constraints representation
import LazySets.LinearConstraint
import LazySets.is_intersection_empty

struct ExactReach end

function solve(solver::ExactReach, problem::Problem)
    reach = forward_network(solver, problem.network, problem.input)
    #println(reach)
    return check_inclusion(reach, problem.output)
end

function forward_layer(solver::ExactReach, layer::Layer, input::Vector{HPolytope})
    output = Vector{HPolytope}(0)
    for i in 1:length(input)
        append!(output, forward_positive(layer, input[i]))
        append!(output, forward_negative(layer, input[i]))
        append!(output, forward_undertermined(layer, input[i]))
    end
    return output
end

function forward_layer(solver::ExactReach, layer::Layer, input::HPolytope)
    output = Vector{HPolytope}(0)
    append!(output, forward_positive(layer, input))
    append!(output, forward_negative(layer, input))
    append!(output, forward_undertermined(layer, input))
    return output
end

function forward_positive(layer::Layer, input::HPolytope)
    inputA, inputb = tosimplehrep(input)
    A = vcat(inputA * pinv(layer.weights), -eye(length(layer.bias)))
    b = vcat(inputb - inputA * pinv(layer.weights) * layer.bias, fill(0.0, length(layer.bias)))
    return HPolytope[HPolytope(A, b)]
end

function forward_negative(layer::Layer, input::HPolytope)
    if is_intersection_empty(input, HPolytope(layer.weights, -layer.bias))
        return HPolytope[]
    else
        n = length(layer.bias)
        return HPolytope[HPolytope(Matrix(vcat(eye(n),-eye(n))),Vector{Float64}(2*n))]
    end
end

function forward_undertermined(layer::Layer, input::HPolytope)
    n = size(layer.weights, 1)
    output = Vector{HPolytope}(2^n)
    inputA, inputb = tosimplehrep(input)
    for h in 1:2^n
        p = getP(h, n)
        A = vcat(inputA * pinv(layer.weights), eye(n) - p, -p)
        b = vcat(inputb - inputA * pinv(layer.weights) * layer.bias, fill(0.0, 2*n))
        output[h] = HPolytope(A, b)
    end
    return output
end

function getP(h::Int64, n::Int64)
    string = bin(h-1, n)
    vec = Vector{Int64}(n)
    for i in 1:n
        vec[i] = ifelse(string[i] == '1', 1, 0)
    end
    return diagm(vec)
end

# This function is called in forward_negative
function is_intersection_empty(set_a::HPolytope, set_b::HPolytope)
    inter = intersect(set_a, set_b)
    return dim(inter) == -1
end