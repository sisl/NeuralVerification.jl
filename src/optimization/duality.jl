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
[K. Dvijotham, R. Stanforth, S. Gowal, T. Mann, and P. Kohli,
"A Dual Approach to Scalable Verification of Deep Networks,"
*ArXiv Preprint ArXiv:1803.06567*, 2018.](https://arxiv.org/abs/1803.06567)
"""
@with_kw struct Duality
    optimizer::AbstractMathProgSolver = GLPKSolverMIP()
end

# can pass keyword args to optimizer
# Duality(optimizer::DataType = GLPKSolverMIP; kwargs...) =  Duality(optimizer(kwargs...))

function solve(solver::Duality, problem::Problem)
    model = Model(solver = solver.optimizer)
    c, d = tosimplehrep(problem.output)
    λ = init_multipliers(model, problem.network)
    μ = init_multipliers(model, problem.network)
    o = dual_value(solver, problem, model, λ, μ)
    @constraint(model, last(λ) .== -c)
    status = solve(model, suppress_warnings = true)
    return interpret_result(solver, status, o - d[1])
end

# False if o > 0, True if o <= 0
function interpret_result(solver::Duality, status, o)
    status != :Optimal && return BasicResult(:unknown)
    getvalue(o) <= 0.0 && return BasicResult(:holds)
    return BasicResult(:violated)
end

function dual_value(solver::Duality,
                    problem::Problem,
                    model::Model,
                    λ::Vector{Vector{VariableRef}},
                    μ::Vector{Vector{VariableRef}})
    bounds = get_bounds(problem)
    layers = problem.network.layers
    # input layer
    o = input_layer_value(layers[1], μ[1], bounds[1])
    o += activation_value(layers[1], μ[1], λ[1], bounds[1])
    # other layers
    for i in 2:length(layers)
        o += layer_value(layers[i], μ[i], λ[i-1], bounds[i])
        o += activation_value(layers[i], μ[i], λ[i], bounds[i])
    end
    @objective(model, Min, o)
    return o
end

function activation_value(layer::Layer,
                          μᵢ::Vector{VariableRef},
                          λᵢ::Vector{VariableRef},
                          bound::Hyperrectangle)
    o = zero(eltype(μᵢ))
    b_hat = approximate_affine_map(layer, bound)
    l_hat, u_hat = low(b_hat), high(b_hat)
    l, u = layer.activation(l_hat), layer.activation(u_hat)

    o += sum(symbolic_max.(μᵢ.*l_hat, μᵢ.*u_hat))
    o += sum(symbolic_max.(λᵢ.*l, λᵢ.*u))
    return o
end

function layer_value(layer::Layer,
                     μᵢ::Vector{VariableRef},
                     λᵢ::Vector{VariableRef},
                     bound::Hyperrectangle)
    (W, b) = (layer.weights, layer.bias)
    o = λᵢ' * bound.center - μᵢ' * (W * bound.center + b)
    # instead of for-loop:
    o += sum(symbolic_abs.(λᵢ .- W'*μᵢ) .* bound.radius) # TODO check that this is equivalent to before
    return o
end

function input_layer_value(layer::Layer,
                           μᵢ::Vector{VariableRef},
                           input::Hyperrectangle)
    W, b = layer.weights, layer.bias
    o = -μᵢ' * (W*input.center .+ b)
    o += sum(symbolic_abs.(μᵢ'*W) .* input.radius)   # TODO check that this is equivalent to before
    return o
end
