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
    layers = problem.network.layers
    bounds = get_bounds(problem)

    λ, μ = init_nnet_vars(solver, model, problem.network)

    n_layer = length(layers)

    c, d = tosimplehrep(problem.output)
    all_layers_n  = map(l -> length(l.bias), layers)  # this shows up a lot. Consider making it a utility function

    # Input constraint
    J = input_layer_cost(layers[1], μ[1], d, problem.input)

    # Cost for all linear layers
    for l in 2:n_layer
        J += layer_cost(layers[l], μ[l], λ[l-1], bounds[l])
    end

    J += activation_cost.(layers, μ, λ, bounds) |> sum

    # output constraint
    @constraint(model, λ[n_layer] .== -c)
    @objective(model, Min, J[1])
end

# For each layer l and node k
# max { mu[l][k] * z - lambda[l][k] * act(z) }
function activation_cost(layer, μ, λ, bound)
    J = zero(typeof(first(μ)))
    (W, b, act) = (layer.weights, layer.bias, layer.activation)
    for k in 1:length(b)
        z = W[k, :]' * bound.center + b[k]
        r = sum(abs.(W[k, :]) .* bound.radius)
        J += μ[k] * z + symbolic_abs(μ[k]) * r
        # TODO make this pretty:
        # NOTE 1-2 = -1. Use this if "if" doesn't work for Variables
        if λ[k] > 0   J += -λ[k] * act(z-r)
        else          J += -λ[k] * act(z+r)
        end
    end
    return J
end

# For all layer l
# max { lambda[l]' * x[l] - mu[l]' * (W[l] * x[l] + b[l]) }
# x[i] belongs to a Hyperrectangle
# TODO consider bringing in μᵀ instead of μ
function layer_cost(layer, μ, λ, bound)
    (W, b) = (layer.weights, layer.bias)
    J = λ' * bound.center - μ' * (W' * bound.center + b)
    # instead of for-loop:
    J += sum(symbolic_abs(λ .- W'*μ) .* bound.radius) # TODO check that this is equivalent to before
    return J
end
# Input constraint
# max { - mu[1]' * (W[1] * input + b[1]) }
# input belongs to a Hyperrectangle
function input_layer_cost(layer, μ, d, input)
    W, b = layer.weights, layer.bias
    J = d - μ' * (W' * input.center .+ b)
    J += sum(symbolic_abs.(μ' * W) .* input.radius)   # TODO check that this is equivalent to before
    return J
end
# NOTE renamed to symbolic_abs to avoid type piracy
function symbolic_abs(m::Model, v)
    aux = @variable(m) #get an anonymous variable
    @constraint(m, aux >= 0)
    @constraint(m, aux >= v)
    @constraint(m, aux >= -v)
    return aux
end
symbolic_abs(v::Variable)                     = symbolic_abs(v.m, v)
symbolic_abs(v::JuMP.GenericAffExpr)          = symbolic_abs(first(v.vars).m, v)
symbolic_abs(v::Array{<:JuMP.GenericAffExpr}) = symbolic_abs.(first(first(v).vars).m, v)
#=
    this definition not necessary if using broadcast in the array case
    (see array method above). It might be less efficient in the solver though(?)
    so maybe this method with internal broadcast is desired
=#
# function symbolic_abs(m::Model, v::Array)
#     @variable(m, aux >= 0)
#     @addConstraint(m, aux .>= v)
#     @addConstraint(m, aux .>= -v)
#     return aux
# end


# The variables in Duality are Lagrangian Multipliers
function init_nnet_vars(solver::Duality, model::Model, network::Network)
    layers = network.layers
    λ = Vector{Vector{Variable}}(length(layers))
    μ = Vector{Vector{Variable}}(length(layers))

    all_layers_n = map(l->length(l.bias), layers)

    for (i, n) in enumerate(all_layers_n)
        λ[i] = @variable(model, [1:n])
        μ[i] = @variable(model, [1:n])
    end

    return λ, μ
end
