"""
    Duality(optimizer)

Duality uses Lagrangian relaxation to compute over-approximated bounds for a network

# Problem requirement
1. Network: any depth, any activation function that is monotone
2. Input: hyperrectangle
3. Output: halfspace

# Return
`BasicResult`

# Method
Lagrangian relaxation.
Default `optimizer` is `GLPKSolverMIP()`.

# Property
Sound but not complete.

# Reference
K. Dvijotham, R. Stanforth, S. Gowal, T. Mann, and P. Kohli,
"A Dual Approach to Scalable Verification of Deep Networks,"
*ArXiv Preprint ArXiv:1803.06567*, 2018.
"""
struct Duality{O<:AbstractMathProgSolver}
    optimizer::O
end

Duality() =  Duality(GLPKSolverMIP())

function solve(solver::Duality, problem::Problem)
    model = JuMP.Model(solver = solver.optimizer)
    bounds = get_bounds(problem)
    c, d = tosimplehrep(problem.output)
    λ = init_multipliers(model, problem.network)
    μ = init_multipliers(model, problem.network)
    J = dual_cost(solver, model, problem.network, bounds, λ, μ)
    @constraint(model, last(λ) .== -c)
    status = solve(model)
    return interpret_result(solver, status, J - d[1])
end

# False if J > 0, True if J <= 0
function interpret_result(solver::Duality, status, J)
    status != :Optimal && return BasicResult(:Unknown)
    getvalue(J) <= 0.0 && return BasicResult(:SAT)
    return BasicResult(:UNSAT)
end

# For each layer l and node k
# max { mu[l][k] * z - lambda[l][k] * act(z) }
function activation_cost(layer, μ, λ, bound)
    J = zero(typeof(first(μ)))
    # (W, b, act) = (layer.weights, layer.bias, layer.activation)
    b_hat = linear_transformation(layer, bound)
    l_hat, u_hat = low(b_hat), high(b_hat)
    l, u = layer.activation(l_hat), layer.activation(u_hat)
    for k in 1:length(l)
        J += symbolic_max(μ[k]*l_hat[k], μ[k]*u_hat[k])
        J += symbolic_max(λ[k]*l[k], λ[k]*u[k])
        # z = W[k, :]' * bound.center + b[k]
        # r = sum(abs.(W[k, :]) .* bound.radius)
        # high = μ[k] * (z + r) - λ[k] * act(z + r)
        # low = μ[k] * (z - r) - λ[k] * act(z - r)
        # J += symbolic_max(high, low)
    end
    return J
end

# For all layer l
# max { λ[l-1]' * x[l] - μ[l]' * (W[l] * x[l] + b[l]) }
# x[i] belongs to a Hyperrectangle
# TODO consider bringing in μᵀ instead of μ
function layer_cost(layer, μ, λ, bound)
    (W, b) = (layer.weights, layer.bias)
    J = λ' * bound.center - μ' * (W * bound.center + b)
    # instead of for-loop:
    J += sum(symbolic_abs(λ - W' * μ) .* bound.radius) # TODO check that this is equivalent to before
    return J
end

# Input constraint
# max { - mu[1]' * (W[1] * input + b[1]) }
# input belongs to a Hyperrectangle
function input_layer_cost(layer, μ, input)
    W, b = layer.weights, layer.bias
    J = - μ' * (W * input.center .+ b)
    J += sum(symbolic_abs.(μ' * W) .* input.radius)   # TODO check that this is equivalent to before
    return J
end

function dual_cost(solver::Duality, model::Model, nnet::Network, bounds::Vector{Hyperrectangle}, λ, μ)
    layers = nnet.layers
    # input layer
    J = input_layer_cost(layers[1], μ[1], bounds[1])
    J += activation_cost(layers[1], μ[1], λ[1], bounds[1])
    # other layers
    for i in 2:length(layers)
        J += layer_cost(layers[i], μ[i], λ[i-1], bounds[i])
        J += activation_cost(layers[i], μ[i], λ[i], bounds[i])
    end
    @objective(model, Min, J)
    return J
end