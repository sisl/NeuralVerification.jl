struct ConvDual
	optimizer::Int64
	ConvDual(model) = new(model)
end

function solve(solver::ConvDual, problem::Problem)
	l, u = get_bounds(problem.network, problem.input.center, problem.input.radius[1])
end

# This step is similar to reachability method
function get_bounds(nnet::Network, input::Vector{Float64}, epsilon::Float64)
	n_layer = length(nnet.layers)
	l = Vector{Vector{Float64}}(n_layer)
	u = Vector{Vector{Float64}}(n_layer)
	gamma = Vector{Vector{Float64}}(n_layer)
	mu = Vector{Vector{Vector{Float64}}}(n_layer)

	v1 = -nnet.layers[1].weights'
	gamma[1] = -nnet.layers[1].bias'

	# Bounds for the first layer
	output = nnet.layers[1].weights * input + nnet.layers[1].bias
	n_output = length(output)
	l[1] = fill(0.0, n_output)
	u[1] = fill(0.0, n_output)
	for i in 1:length(n_output)
	    l[1][i] = output(i) - epsilon * sum(abs.(nnet.layers[1].weights[i, :])
	    u[1][i] = output(i) + epsilon * sum(abs.(nnet.layers[1].weights[i, :])
	end

	for i in 2:n_layer
		act_pattern, D = get_activaton(l[i-1], u[i-1])

		# Initialize new terms
		gamma[i] = -nnet.layers[i].bias'

		n_output = length(act_pattern)
		mu[i] = Vector{Vector{Float64}}(n_output)
		for j in 1:n_output
			#mu[i][j] = ifelse(act_pattern[j] == 0, nnet.layers[i].weights * D[:, j], zeros(n_output))
			mu[i][j] = nnet.layers[i].weights * D[:, j]
		end

		# Propagate existiing terms
		for j in 1:i-1
			for k in 1:length(mu[j])
				mu[j][k] = nnet.layers[i].weights * D * mu[j][k]
			end
			gamma[j] = nnet.layers[i].weights * D * gamma[j]
		end
		v1 = nnet.layers[i].weights * D * v1

		# Compute bounds
		phi = v1 * input + sum.(gamma[:])
		for i in 1:length(n_output)
			neg = fill(0.0, i-1)
			pos = fill(0.0, i-1)
			for j in 1:i-1
				neg[j] = sum(ifelse(act_pattern[k] == 0 && mu[j][k][i] < 0, l[j][k]*(-mu[j][k][i]), 0) for k in 1:length(mu[j]))
				pos[j] = sum(ifelse(act_pattern[k] == 0 && mu[j][k][i] > 0, u[j][k]*(mu[j][k][i]), 0) for k in 1:length(mu[j]))
			end
		    l[1][i] = phi(i) - epsilon * sum(abs.(v1[:, i]) + sum(neg)
		    u[1][i] = phi(i) + epsilon * sum(abs.(v1[:, i]) - sum(pos)
		end
	end

	return (l, u)
end

function get_activation(l::Vector{Float74}, u::Vector{Float64})
	n = length(l)
	act_pattern = fill(0, n)
	D = zeros(0.0, n, n)
	for i in 1:n
		if u(i) < 0
			act_pattern(i) = -1
			D[i, i] = 1
		elseif l(i) > 0
			act_pattern(i) = 1
			D[i, i] = 1
		else
			D[i, i] = u(i) / (u(i) - l(i))
		end
	end
	return (act_pattern, D)
end

function dual(solver::ConvDual, problem::Problem)
	

end