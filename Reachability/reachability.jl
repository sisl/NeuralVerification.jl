# General structure for reachability methods

# new datatype
# abstract type GeometricObj end

using LazySets

abstract type Reachability end

# This function performs layer-by-layer propagation
# It is called by all solvers under reachability
function forward_network(solver::Reachability, nnet::Network, input::AbstractPolytope)
    reach = input
    for layer in nnet.layers
        reach = forward_layer(solver, layer, reach)
    end
    return reach
end