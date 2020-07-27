# General structure for reachability methods

# Performs layer-by-layer propagation
# It is called by all solvers under reachability
# TODO: also called by ReluVal and FastLin, so move to general utils (or to network.jl)
function forward_network(solver, nnet::Network, input::AbstractPolytope)
    reach = input
    for layer in nnet.layers
        reach = forward_layer(solver, layer, reach)
    end
    return reach
end

# Checks whether the reachable set belongs to the output constraint
# It is called by all solvers under reachability
# Note vertices_list is not defined for HPolytope: to be defined
function check_inclusion(reach::Vector{<:LazySet}, output)
    for poly in reach
        issubset(poly, output) || return ReachabilityResult(:violated, reach)
    end
    return ReachabilityResult(:holds, similar(reach, 0))
end

function check_inclusion(reach::P, output) where P<:LazySet
    if issubset(reach, output)
        return ReachabilityResult(:holds, P[])
    end
    return ReachabilityResult(:violated, [reach])
end

# return a vector so that append! is consistent with the relu forward_partition
forward_partition(act::Id, input::AbstractPolytope) = [input]

forward_partition(act::Id, input::Zonotope) = input

function forward_partition(act::ReLU, input::HPolytope)
    n = dim(input)
    output = Vector{HPolytope}(undef, 0)
    C, d = tosimplehrep(input)
    dh = [d; zeros(n)]
    for h in 0:(2^n)-1
        P = getP(h, n)
        Ch = [C; I - 2P]
        input_h = HPolytope(Ch, dh)
        if !isempty(input_h)
            push!(output, linear_map(Matrix{Float64}(P), input_h))
        end
    end
    return output
end

function getP(h::Int64, n::Int64)
    str = string(h, pad = n, base = 2)
    vec = Vector{Int64}(undef, n)
    for i in 1:n
        vec[i] = ifelse(str[i] == '1', 1, 0)
    end
    return Diagonal(vec)
end

# forward_partition for Zonotopes
function forward_partition(act::ReLU, input::Zonotope)
    return overapproximate(Rectification(input), Zonotope)
end

# for Hyperrectangles
function forward_partition(act::ReLU, input::Hyperrectangle)
    return rectify(input)
end
