# This method only works for half space output constraint
# c y <= d
# For this implementation, limit the input constraint to Hyperrectangle

import Base.abs
import JuMP: GenericAffExpr

struct Duality{O<:AbstractMathProgSolver} <: Feasibility
    optimizer::O
end

# False if J > 0, True if J <= 0
# To be implemented
function interpret_result(solver::Duality, status)
    return Result(:Undertermined)
end

function encode(solver::Duality, model::Model, problem::Problem)
    bounds = get_bounds(problem)

    lambda, mu = init_nnet_vars(solver, model, problem.network)

    n_layer = length(problem.network.layers)

    c, d = tosimplehrep(problem.output)
    all_layers_n  = [length(l.bias) for l in problem.network.layers]
	
	# Input constraint
    # max { - mu[1]' * (W[1] * input + b[1]) }
    # input belongs to a Hyperrectangle
    J = d - mu[1]' * (problem.network.layers[1].weights * problem.input.center + problem.network.layers[1].bias)
    for i in 1:size(problem.network.layers[1].weights, 2)
        J += abs(mu[1]' * problem.network.layers[1].weights[:, i]) * problem.input.radius[i]
    end

    # Cost for all linear layers
    # For all layer l
    # max { lambda[l]' * x[l] - mu[l]' * (W[l] * x[l] + b[l]) }
    # x[i] belongs to a Hyperrectangle
    for l in 2:n_layer
        (W, b) = (problem.network.layers[l].weights, problem.network.layers[l].bias)
        J += lambda[l-1]' * bounds[l].center - mu[l]' * (W * bounds[l].center + b)
        for i in 1:all_layers_n[l-1]
            J += abs(lambda[l-1][i] - mu[l]' * W[:, i]) * bounds[l].radius[i] 
        end
    end

    # Cost for all activations functions
    # For all layer l and node k
    # max { mu[l][k] * z - lambda[l][k] * act(z) }
    # z = W * bounds[l] + b
    for l in 1:n_layer
        (W, b, act) = (problem.network.layers[l].weights, problem.network.layers[l].bias, problem.network.layers[l].activation)
        for k in 1:all_layers_n[l]
            z = W[k, :] * bounds[l].center + b[k]
            r = sum(abs.(W[k, :]) .* bounds[l].radius)
            J += mu[l][k] * z + abs(mu[l][k]) * r
            J += ifelse(lambda[l][k] > 0, -lambda[l][k] * act(z-r), -lambda[l][k] * act(z+r))
        end
    end

    @constraints(model, lambda[l] == -c)
    @objective(model, Min, J[1])
end

function abs(v::Variable)
    @variable(v.m, aux >= 0)
    @addConstraint(v.m, aux >= v)
    @addConstraint(v.m, aux >= -v)
    return aux
end

function abs{V<:GenericAffExpr}(v::V)
    m = first(v).m
    @variable(m, aux >= 0)
    @addConstraint(m, aux >= v)
    @addConstraint(m, aux >= -v)
    return aux
end

function abs{V<:GenericAffExpr}(v::Array{V})
    m = first(first(v).vars).m
    @variable(m, aux[1:length(v)] >= 0)
    @addConstraint(m, aux .>= v)
    @addConstraint(m, aux .>= -v)
    return aux
end

# The variables in Duality are Lagrangian Multipliers
function init_nnet_vars(solver::Duality, model::Model, network::Network)
    layers = network.layers
    lambda = Vector{Vector{Variable}}(length(layers)) 
    mu = Vector{Vector{Variable}}(length(layers))

    all_layers_n  = [length(l.bias) for l in layers]

    for (i, n) in enumerate(all_layers_n)
        lambda[i] = @variable(model, [1:n])
        mu[i]  = @variable(model, [1:n])
    end

    return lambda, mu
end
