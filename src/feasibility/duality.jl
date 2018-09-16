# This method only works for half space output constraint
# c y <= d
# For this implementation, limit the input constraint to Hyperrectangle
struct Duality{O<:AbstractMathProgSolver} <: Feasibility
    optimizer::O
end

# False if J > 0, True if J <= 0
# To be implemented
function interpret_result(solver::Duality, status)
    return Result(:Undertermined)
end

function encode(solver::Duality, model::Model, problem::Problem)
	# Need to implement get_bounds (not included in the paper)
    bounds = get_bounds(problem)
    lambda, mu = init_nnet_vars(model, problem.network)

    c, d = tosimplehrep(problem.output)
    all_layers_n  = [length(l.bias) for l in problem.network.layers]
	
	# Input constraint
    J = d - mu[1]' * (problem.network.layers[1].weights * problem.input.center + problem.network.layers[1].bias)
    # w = problem.network.layers[1].weights
    # J += sum(problem.input.radius[j] * ifelse(mu[1][j] * w[:,j] > 0, mu[1][j] * w[:,j], -mu[1][j] * w[:,j]) for j in 1:all_layers_n[1])
    
    # How to include absolute value in objective function?
    # problem.input.radius' * abs.(mu[1]' * (problem.network.layers[1].weights))

    @objective(model, Min, J[1])
end

# This function calls maxSens to compute the bounds
function get_bounds(problem::Problem)
    solver = MaxSens()
    bounds = Vector{Hyperrectangle}(length(problem.network.layers))
    reach = problem.input
    for (i, layer) in enumerate(problem.network.layers)
        reach = forward_layer(solver, layer, reach)
        bounds[i] = reach
    end
    return bounds
end

function init_nnet_vars(model::Model, network::Network)
    layers = network.layers
    lambda = Vector{Vector{Variable}}(length(layers)) 
    mu = Vector{Vector{Variable}}(length(layers))

    all_layers_n  = [length(l.bias) for l in layers]

    for (i, n) in enumerate(all_layers_n)
        lambda[i] = @variable(model, [1:n])
        mu[i]  = @variable(model, [1:n], Bin)
    end

    return lambda, mu
end
