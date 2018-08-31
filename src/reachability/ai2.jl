include("utils/reachability.jl")

import LazySets.Zonotope
import LazySets.EmptySet

abstract type Case end
# three cases:
# x_i >= 0 is represented as pos(i)
# x_i >= x_j is represented as greater(i,j)
# 0 > x_i is represented as neg(i)

struct pos <: Case
	i::Int64
end

struct neg <: Case
	i::Int64
end

struct greater <: Case
	i::Int64
	j::Int64
end

struct Ai2 <: Reachability end

function solve(solver::Ai2, problem::Problem)
	reach = forward_network(solver, problem.network, problem.input)
	println(reach)
    return check_inclusion(reach, problem.output)
end

function forward_layer(solver::Ai2, layer::Layer, inputs::Vector{Zonotope})
	output = Vector{Zonotope}(0)
	for input in inputs
		outLinear = forward_linear(input, layer.weights, layer.bias)
		append!(output, forward_act(layer.activation, outLinear))
	end
	return output
end

function forward_layer(solver::Ai2, layer::Layer, input::Zonotope)
	output = Vector{Zonotope}(0)
	outLinear = forward_linear(input, layer.weights, layer.bias)
	append!(output, forward_act(layer.activation, outLinear))
	return output
end

function forward_linear(input::Zonotope, W::Matrix{Float64}, b::Vector{Float64})
	rotation = linear_map(W, input)
	output = Zonotope(rotation.center + b, rotation.generators)
	return output
end

function forward_act(act, input::Zonotope) 
	return error("not supported yet")
end

function forward_act(act::ReLU, input::Zonotope)
	output = Zonotope[input]
	for i in 1:dim(input)
		newPoly = Vector{Zonotope}(0)
		for j = 1:length(output)
			zonos = Vector{Zonotope}(0)
			# Positive case
			meet_pos = meet(pos(i), output[j])
			if dim(meet_pos) > -1
				append!(zonos, Zonotope[meet_pos])
			end
			# Negative case
			meet_neg = meet(neg(i), output[j])
			if dim(meet_neg) > -1
				append!(zonos, Zonotope[linear_map(getI(dim(input), i), meet_neg)])
			end
			println(zonos)
			append!(newPoly, simplify(zonos))
		end
		output = newPoly
	end
	return output
end

# To be implemented
# Adjust the zonotope by applying the conditions
function meet(case::Case, zono::Zonotope)
	return zono
end

# Combine zonotopes
function simplify(zonos::Vector{Zonotope})
	n = length(zonos)
	vertices = Vector{Vector{Vector{Float64}}}(n)
	flag = Vector{Int64}(n)
	for i in 1:n
		vertices[i] = vertices_list(zonos[i])
	end

	# check inclusion
	for i in 1:n
		if flag[i] > -1
			for j in i+1:n
				if ⊆(zonos[i], zonos[j])
					flag[i] = -1
					break
				end
				if ⊆(zonos[j], zonos[i])
					flag[j] = -1
				end
			end
		end
	end

	new_zonos = Vector{Zonotope}(0)
	for i in 1:n
		if flag[i] > -1
			append!(new_zonos, Zonotope[zonos[i]])
		end
	end
	return new_zonos
end

function getI(n::Int64, id::Int64)
	vec = Vector{Int64}(n)
	for i in 1:n
		vec[i] = ifelse(i == id, 0, 1)
	end
	return diagm(vec)
end

# Check wheather all vertices satisfy to the output constraint
function check_inclusion(reach::Vector{Zonotope}, out::AbstractPolytope)
	for i in 1:length(reach)
		vertices = vertices_list(reach[i])
		for vertex in vertices
			if ~∈(vertex, out)
				return false
			end
		end
	end
	return true
end