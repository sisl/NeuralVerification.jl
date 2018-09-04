# General structure for reachability methods

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

# This function checks whether the reachable set belongs to the output constraint
# It is called by all solvers under reachability
# Note vertices_list is not defined for HPolytope: to be defined
function check_inclusion(reach::Vector{<:AbstractPolytope}, output::AbstractPolytope)
	for i in 1:length(reach)
		vertices = vertices_list(reach[i])
		for vertex in vertices
			if ~∈(vertex, output)
				return false
			end
		end
	end
	return true
end