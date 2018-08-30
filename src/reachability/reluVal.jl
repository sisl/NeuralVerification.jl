include("utils/reachability.jl")

import LazySets.Zonotope
import LazySets.EmptySet

struct ReluVal <: Reachability end

function solve(solver::ReluVal, problem::Problem)
	reach = forward_network(solver, problem.network, problem.input)
	println(reach)
    return check_inclusion(reach, problem.output)
end

function forward_layer(solver::ReluVal, layer::Layer, inputs::Vector{Zonotope})
	output = Vector{Zonotope}(0)
	for input in inputs
		outLinear = forward_linear(input, layer.weights, layer.bias)
		append!(output, forward_act(layer.activation, outLinear))
	end
	return output
end

function forward_layer(solver::ReluVal, layer::Layer, input::Zonotope)
	output = Vector{Zonotope}(0)
	outLinear = forward_linear(input, layer.weights, layer.bias)
	append!(output, forward_act(layer.activation, outLinear))
	return output
end