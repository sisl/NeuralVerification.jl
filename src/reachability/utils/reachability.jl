# General structure for reachability methods

# This function performs layer-by-layer propagation
# It is called by all solvers under reachability
# TODO: also called by ReluVal and FastLin, so move to general utils (or to network.jl)
function forward_network(solver, nnet::Network, input::AbstractPolytope)
    reach = input
    for layer in nnet.layers
        reach = forward_layer(solver, layer, reach)
    end
    return reach
end

# This function checks whether the reachable set belongs to the output constraint
# It is called by all solvers under reachability
# Note vertices_list is not defined for HPolytope: to be defined
function check_inclusion(reach::Vector{<:AbstractPolytope}, output::AbstractPolytope)
    for poly in reach
        issubset(poly, output) || return ReachabilityResult(:UNSAT, reach)
    end
    return ReachabilityResult(:SAT, reach)
end
function check_inclusion(reach::AbstractPolytope, output::AbstractPolytope)
    if issubset(reach, output)
        return ReachabilityResult(:SAT, reach)
    end
    return ReachabilityResult(:UNSAT, reach)
end
