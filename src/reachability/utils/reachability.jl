# General structure for reachability methods

# Performs layer-by-layer propagation
# It is called by all solvers under reachability
# TODO: also called by ReluVal and FastLin, so move to general utils (or to network.jl)
function forward_network(solver, nnet::Network, input; get_bounds=false)
    if (get_bounds)
        reach = input
        # Add a hyperrectangle corresponding to the input as the first set of bounds
        if input isa AbstractPolytope
            bounds = Vector{Hyperrectangle}([overapproximate(input, Hyperrectangle)])
        elseif input isa SymbolicIntervalMask
            bounds = Vector{Hyperrectangle}([overapproximate(symbol_to_concrete(input.sym), Hyperrectangle)])
        else
            @assert false "Unsupported input type for bounds"
        end
        for layer in nnet.layers
            reach, bounds = forward_layer(solver, layer, reach, bounds)
        end
        return reach, bounds
    else
        reach = input
        for layer in nnet.layers
            reach = forward_layer(solver, layer, reach)
        end
        return reach
    end
end

# Checks whether the reachable set belongs to the output constraint
# It is called by all solvers under reachability
# Note vertices_list is not defined for HPolytope: to be defined
function check_inclusion(reach::Vector{<:LazySet}, output)
    for poly in reach
        issubset(poly, output) || return ReachabilityResult(:violated, reach)
    end
    return ReachabilityResult(:holds, reach)
end

function check_inclusion(reach::P, output) where P<:LazySet
    return ReachabilityResult(issubset(reach, output) ? :holds : :violated, [reach])
end

# return a vector so that append! is consistent with the relu forward_partition
forward_partition(act::Id, input) = [input]

function forward_partition(act::ReLU, input)
    N = dim(input)
    N > 30 && @warn "Got dim(X) == $N in `forward_partition`. Expecting 2ᴺ = $(2^big(N)) output sets."

    output = HPolytope{Float64}[]

    for h in 0:(big"2"^N)-1
        P = Diagonal(1.0.*digits(h, base = 2, pad = N))
        orthant = HPolytope(Matrix(I - 2.0P), zeros(N))
        S = intersection(input, orthant)
        if !isempty(S)
            push!(output, linear_map(P, S))
        end
    end
    return output
end
