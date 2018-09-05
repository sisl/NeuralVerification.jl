include("utils/reachability.jl")

import LazySets.Hyperrectangle
import LazySets.EmptySet

struct ReluVal <: Reachability 
	max_iter::Int64
end

struct SymbolicInterval
	Low::Matrix{Float64}
	Up::Matrix{Float64}
	interval::Hyperrectangle
end

# Gradient mask for a single layer
struct GradientMask
	lower::Vector{Int64}
	upper::Vector{Int64}
end

# Data to be passed during forward_layer
struct SymbolicInterval_Mask
	sym::SymbolicInterval
	mask::Vector{GradientMask}
end

function solve(solver::ReluVal, problem::Problem)
	# Compute the reachable set without splitting the interval
	reach = forward_network(solver, problem.network, problem.input)
	if check_inclusion(reach, problem.output) > 0
		return check_inclusion(reach, problem.output)
	end

	# If undertermined, split the interval
	# Bisection tree. Unsure how to explore, BFS or DFS?
	for i in 2:solver.max_iter
		gradient = back_prop(problem.network, reach.mask)
		intervals = split_input(problem.network, reach.sym.interval, gradient)
		for interval in intervals
			reach = forward_network(solver, problem.network, interval)
			if check_inclusion(reach, problem.output)
				return check_inclusion(reach, problem.output)
			end
		end
	end
    return "undertermined"
end

# To be implemented
function check_inclusion(reach::SymbolicInterval_Mask, output::AbstractPolytope)
	return 0
end

function forward_layer(solver::ReluVal, layer::Layer, input::Union{SymbolicInterval_Mask, Hyperrectangle})
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
	sym = SymbolicInterval(hcat(W, b), hcat(W, b), input)
	mask = GradientMask[]
	return SymbolicInterval_Mask(sym, mask)
end

# Symbolic forward_linear
function forward_linear(input::SymbolicInterval_Mask, W::Matrix{Float64}, b::Vector{Float64})
	n_output, n_input = size(W)
	println(input.sym.Low)
	n_symbol = size(input.sym.Low, 2) - 1
	println(n_output, " ", n_input, " ", n_symbol)

	output_Low = zeros(n_output, n_symbol + 1)
	output_Up = zeros(n_output, n_symbol + 1)
	for k in 1:n_symbol + 1
		for j in 1:n_output
			for i in 1:n_input
				output_Up[j, k] += ifelse(W[j, i]>0, W[j, i] * input.sym.Up[i, k], W[j, i] * input.sym.Low[i, k])
				output_Low[j, k] += ifelse(W[j, i]>0, W[j, i] * input.sym.Low[i, k], W[j, i] * input.sym.Up[i, k])
			end
			if k > n_symbol
				output_Up[j, k] += b[j]
				output_Low[j, k] += b[j]
			end
		end
	end
	sym = SymbolicInterval(output_Low, output_Up, input.sym.interval)
	mask = input.mask
	return SymbolicInterval_Mask(sym, mask)
end

# Concrete forward_act
function forward_act(input::Hyperrectangle)
	input_upper = high(input)
	input_lower = low(input)
	output_upper = fill(0.0, dim(input))
	output_lower = fill(0.0, dim(input))
	mask_upper = fill(1, dim(input))
	mask_lower = fill(0, dim(input))
	for i in 1:dim(input)
		if input_upper[i] <= 0
			mask_upper[i] = mask_lower[i] = 0
			output_upper[i] = output_lower[i] = 0.0
		elseif input_lower[i] >= 0
			mask_upper[i] = mask_lower[i] = 1
		else
			output_lower[i] = 0.0
		end
	end
	return (output_lower, output_upper, mask_lower, mask_upper)
end

# Symbolic forward_act
function forward_act(input::SymbolicInterval_Mask)
	n_output, n_input = size(input.sym.Up)

	input_upper = high(input.sym.interval)
	input_lower = low(input.sym.interval)

	output_Up = input.sym.Up[:, :]
	output_Low = input.sym.Low[:, :]

	mask_upper = fill(1, n_output)
	mask_lower = fill(0, n_output)

	for i in 1:n_output
		if upper_bound(input.sym.Up[i, :], input.sym.interval) <= 0.0
			# Update to zero
			mask_upper[i] = 0
			mask_lower[i] = 0
			output_Up[i, :] = fill(0.0, n_input)
			output_Low[i, :] = fill(0.0, n_input)
		elseif lower_bound(input.sym.Low[i, :], input.sym.interval) >= 0
			# Keep dependency
			mask_upper[i] = 1
			mask_lower[i] = 1
		else
			# Concretization
			mask_upper[i] = 1
			mask_lower[i] = 0
			output_Low[i, :] = zeros(1, n_input)
			if lower_bound(input.sym.Up[i, :], input.sym.interval) <= 0
				output_Up[i, :] = hcat(zeros(1, n_input - 1), upper_bound(input.sym.Up[i, :], input.sym.interval))
			end
		end
	end
	sym = SymbolicInterval(output_Low, output_Up, input.sym.interval)
	mask = vcat(input.mask, GradientMask(mask_lower, mask_upper))
	return SymbolicInterval_Mask(sym, mask)
end

# To be implemented
function back_prop(nnet::Network, R::Vector{GradientMask})
	n_layer = length(nnet.layers)
	# For now, assume the last layer is identity
	Up = eye(length(nnet.layers[n_layer].bias))
	Low = eye(length(nnet.layers[n_layer].bias))

	for k in n_layer:-1:1
		# back through activation function using the gradient mask
		for i in 1:length(nnet.layers[k].bias)
			output_Up[i, :] = ifelse(R[k].upper[i], Up[i, :], zeros(1, size(Up,2)))
			output_Low[i, :] = ifelse(R[k].lower[i], Low[i, :], zeros(1, size(Low,2)))	
		end
		output = SymbolicInterval(output_Low, output_Up, Hyperrectangle())
		# back through weight matrix
		output = forward_linear(output, inv(nnet.layers[k].weights), zeros(1, length(nnet.layers[n_layer].bias)))
		Up = output.Up[:, :]
		Low = output.Low[:, :]
	end

	return output
end

# Return the splited intervals
function split_input(nnet::Network, input::Hyperrectangle, g::SymbolicInterval)
	largest_smear = - Inf
	feature = 0
	r = input.radius .* 2
	for i in 1:dim(input)
		smear = sum(ifelse(g.Up[i, j] - g.Low[i, j], g.Up[i, j] * r[i], -g.Low[i, j] * r[i]) for j in 1:size(g.Up, 2))
		if smear > largest_smear
			largest_smear = smear
			feature = i
		end
	end
	input_upper = high(input)
	input_lower = low(input)
	input_upper[feature] = input.center[feature]
	input_split_left = high_dim_interval(input_lower, input_upper)

	input_lower[feature] = input.center[feature]
	input_upper[feature] = input.center[feature] + input.radius[feature]
	input_split_right = high_dim_interval(input_lower, input_upper)
	return (input_split_left, input_split_right)
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
		bound += ifelse(map[i]>0, map[i]*input_lower[i], map[i]*input_upper[i])
	end
	return bound
end

# Turn lower and upper bounds in high dimension into Hyperrectangle
function high_dim_interval(lower::Vector{Float64}, upper::Vector{Float64})
	return Hyperrectangle((upper + lower)./2, (upper - lower)./2)
end