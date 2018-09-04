include("utils/reachability.jl")

import LazySets.Hyperrectangle
import LazySets.EmptySet

struct ReluVal <: Reachability end

struct Symbolic_Interval
	Low::Matrix{Float64}
	Up::Matrix{Float64}
	input_range::Hyperrectangle
end

function solve(solver::ReluVal, problem::Problem)
	reach = forward_network(solver, problem.network, problem.input)
	println(reach)
    return check_inclusion(reach, problem.output)
end

function forward_layer(solver::ReluVal, layer::Layer, input::Union{Symbolic_Interval, Hyperrectangle})
	return forward_act(forward_linear(input, layer.weights, layer.bias))
end

# Concrete forward_linear
function forward_linear_concrete(input::Hyperrectangle, W::Matrix{Float64}, b::Vector{Float64})
	n_output = size(W, 1)
	output_upper = fill(0.0, n_output)
	output_lower = fill(0.0, n_output)
	for j in 1:n_output
		output_upper[j] = upper_bound(W[j, :], input) + b[j]
		output_lower[j] = lower_bound(W[j, :], input) + b[j]
	end
	output = high_dim_interval(output_lower, output_upper)
	return output
end

# Symbolic forward_linear for the first layer
function forward_linear(input::Hyperrectangle, W::Matrix{Float64}, b::Vector{Float64})
	return Symbolic_Interval(hcat(W, b), hcat(W, b), input)
end

# Symbolic forward_linear
function forward_linear(input::Symbolic_Interval, W::Matrix{Float64}, b::Vector{Float64})
	n_output, n_input = size(W)
	n_symbol = size(Low, 2) - 1
	output_Low = zeros(n_output, n_symbol + 1)
	output_Up = zeros(n_output, n_symbol + 1)
	for k in 1:n_symbol + 1
		for j in 1:n_output
			for i in 1:n_input
				output_Up[j, k] += ifelse(W[j, i]>0, W[j, i] * input.Up[i, k], W[j, i] * input.Low[i, k])
				output_Low[j, k] += ifelse(W[j, i]>0, W[j, i] * input.Low[i, k], W[j, i] * input.Up[i, k])
			end
			if k > n_symbol
				output_Up[j, k] += b[j]
				output_Low[j, k] += b[j]
			end
		end
	end
	return Symbolic_Interval(output_Low, output_Up, input.input_range)
end

# Concrete forward_act
function forward_act(input::Hyperrectangle)
	input_upper = high(input)
	input_lower = low(input)
	output_upper = fill(0.0, dim(input))
	output_lower = fill(0.0, dim(input))
	symbolic_upper = fill(1, dim(input))
	symbolic_lower = fill(0, dim(input))
	for i in 1:dim(input)
		if input_upper[i] <= 0
			symbolic_upper[i] = symbolic_lower[i] = 0
			output_upper[i] = output_lower[i] = 0.0
		elseif input_lower[i] >= 0
			symbolic_upper[i] = symbolic_lower[i] = 1
		else
			output_lower[i] = 0.0
		end
	end
	return (output_lower, output_upper, symbolic_lower, symbolic_upper)
end

# Symbolic forward_act
# Low: lower bounds (last entry is the concretized value)
# Up: upper bounds (last entry is the concretized value)
# input: input constraint
function forward_act(Low::Matrix{Float64}, Up::Matrix{Float64}, input::Hyperrectangle)
	n_output, n_input = size(Low)

	input_upper = high(input)
	input_lower = low(input)

	output_Up = Up[:, :]
	output_Low = Low[:, :]

	symbolic_upper = fill(1, dim(input))
	symbolic_lower = fill(0, dim(input))

	for i in 1:n_output
		if upper_bound(Up[i, :], input) <= 0.0
			# Update to zero
			symbolic_upper[i] = 0
			symbolic_lower[i] = 0
			output_Up[i, :] = fill(0.0, n_input)
			output_Low[i, :] = fill(0.0, n_input)
		elseif lower_bound(Low[i, :], input) >= 0
			# Keep dependency
			symbolic_upper[i] = symbolic_lower[i] = 1
		else
			# Concretization
			symbolic_upper[i] = 1
			symbolic_lower[i] = 0
			output_Low[i, :] = zeros(1, n_input)
			if lower_bound(Up[i, :], input) <= 0
				output_Up[i, :] = hcat(zeros(1, n_input - 1), upper_bound(Up[i, :], input))
			end
		end
	end
	return (output_Low, output_Up, symbolic_lower, symbolic_upper)
end

# To be implemented
function back_prop(nnet::Network, gradient::Vector{Vector{Float64}})
	return true
end

# To be implemented
function split_input(nnet::Network, g::Hyperrectangle, input::Hyperrectangle)
	return true
end

# Get upper bound in concretization
function upper_bound(map::Vector{Float64}, input::Hyperrectangle)
	bound = map[dim(input)+1]
	input_upper = high(input)
	input_lower = low(input)
	for i in 1:dim(input)
		bound += ifelse( map[i]>0, map[i]*input_upper[i], map[i]*input_lower[i])
	end
	return bound
end

# Get lower bound in concretization
function lower_bound(map::Vector{Float64}, input::Hyperrectangle)
	bound = map[dim(input)+1]
	input_upper = high(input)
	input_lower = low(input)
	for i in 1:dim(input)
		bound += ifelse( map[i]>0, map[i]*input_lower[i], map[i]*input_upper[i])
	end
	return bound
end

# Turn lower and upper bounds in high dimension into Hyperrectangle
function high_dim_interval(lower::Vector{Float64}, upper::Vector{Float64})
	return Hyperrectangle((upper + lower)./2, (upper - lower)./2)
end