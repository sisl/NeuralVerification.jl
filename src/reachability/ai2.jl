import LazySets.Zonotope
import LazySets.EmptySet

abstract type Case end
# three cases:
# x_i >= 0 is represented as pos(i)
# x_i >= x_j is represented as greater(i,j)
# 0 > x_i is represented as neg(i)

struct Pos <: Case
	i::Int64
end

struct Neg <: Case
	i::Int64
end

struct Greater <: Case
	i::Int64
	j::Int64
end

struct Ai2 end

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
			meet_pos = meet(Pos(i), output[j])
			if dim(meet_pos) > -1
				append!(zonos, Zonotope[meet_pos])
			end
			# Negative case
			meet_neg = meet(Neg(i), output[j])
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
	flag = fill(true, length(zonos))
	for i in 1:length(zonos)
		if flag[i]
			for j in i+1:length(zonos)
				if ⊆(zonos[i], zonos[j])
					flag[i] = false
					break
				end
				if ⊆(zonos[j], zonos[i])
					flag[j] = false
				end
			end
		end
	end
	return zonos[flag]
end

function getI(n::Int64, id::Int64)
	vec = Vector{Int64}(n)
	for i in 1:n
		vec[i] = ifelse(i == id, 0, 1)
	end
	return diagm(vec)
end