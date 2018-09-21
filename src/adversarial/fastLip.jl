struct FastLip
    maxIter::Int64
    ϵ0::Float64
    accuracy::Float64
end

function solve(solver::FastLip, problem::Problem)
	# Call FastLin or Call get_bounds in convDual
	# Need bounds and activation patterns for all layers
	bounds, act_pattern = get_bounds()
	result = solve(FastLin(), problem)
	ϵ_fastLin = result.max_disturbance
	
	C = problem.network.layers[1].weights
	L = zeros(size(C))
	U = zeros(size(C))

	for l in 2:length(problem.network.layers)
		C, L, U = bound_layer_grad(C, L, U, problem.network.layers[l].weights, act_pattern[l])
	end

	v = max.(abs.(C+L), abs.(C+U))
	# To do: find out how to compute g
	ϵ = min(g(problem.input.center)/maximum(abs.(v)), ϵ_fastLin)

	return ifelse(ϵ > minimum(problem.input.radius), Result(:True, ϵ), Result(:False, ϵ))
end

function bound_layer_grad(C::Matrix, L::Matrix, U::Matrix, W::Matrix, D::Vector{Float64})
	n_input = size(C)
	n2, n1 = size(W)
	new_C = zeros(n2, n_input)
	new_L = zeros(n2, n_input)
	new_U = zeros(n2, n_input)
	for k in 1:n_input
		for j in 1:n2
			for i in 1:n1
				new_C[j,k] += ifelse(D[i] == 1, W[j,i]*C[i,k], 0)
				new_U[j,k] += ifelse(D[i] == 1, W[j,i]*ifelse(W[j,i] > 0, U[i,k], L[i,k]), 0)
				new_U[j,k] += ifelse(D[i] == 0 && W[j,i]*(C[i,k]+U[i,k])>0, W[j,i]*(C[i,k]+U[i,k]), 0)
				new_L[j,k] += ifelse(D[i] == 1, W[j,i]*ifelse(W[j,i] > 0, L[i,k], U[i,k]), 0)
				new_L[j,k] += ifelse(D[i] == 0 && W[j,i]*(C[i,k]+L[i,k])>0, W[j,i]*(C[i,k]+L[i,k]), 0)
			end
		end
	end
	return (new_C, new_L, new_U)
end