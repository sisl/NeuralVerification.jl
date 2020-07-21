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
@with_kw struct Duality <: Solver
    optimizer = GLPK.Optimizer
end

# can pass keyword args to optimizer
# Duality(optimizer::DataType = GLPKSolverMIP; kwargs...) =  Duality(optimizer(kwargs...))

function solve(solver::Duality, problem::Problem)
    model = Model(solver)
    c, d = tosimplehrep(problem.output)
    λ = init_multipliers(model, problem.network, "λ")
    μ = init_multipliers(model, problem.network, "μ")
    o = dual_value(solver, problem, model, λ, μ)
    @constraint(model, last(λ) .== -c)
    optimize!(model)
    return interpret_result(solver, termination_status(model), o - d[1])
end

# False if o > 0, True if o <= 0
function interpret_result(solver::Duality, status, o)
    status != OPTIMAL && return BasicResult(:unknown)
    value(o) <= 0.0 && return BasicResult(:holds)
    return BasicResult(:violated) # This violation may not be true violation
end

function dual_value(solver::Duality,
                    problem::Problem,
                    model::Model,
                    λ::Vector{Vector{VariableRef}},
                    μ::Vector{Vector{VariableRef}})
    bounds = get_bounds(problem)
    layers = problem.network.layers
    # input layer
    o = layer_value(layers[1], μ[1], 0, bounds[1])
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
    B = approximate_affine_map(layer, bound)
    l̂, û = low(B), high(B)
    o = zero(eltype(μᵢ))

    for (μᵢⱼ, λᵢⱼ, l̂ᵢⱼ, ûᵢⱼ) in zip(μᵢ, λᵢ, l̂, û)

        gᵢⱼ(ẑᵢⱼ) = μᵢⱼ*ẑᵢⱼ - λᵢⱼ*L.activation(ẑᵢⱼ)

        if l̂ᵢⱼ < 0 < ûᵢⱼ && layer.activation isa ReLU
            o += symbolic_max(gᵢⱼ(l̂ᵢⱼ), gᵢⱼ(ûᵢⱼ), 0)
        else
            o += symbolic_max(gᵢⱼ(l̂ᵢⱼ), gᵢⱼ(ûᵢⱼ))
        end
    end
    o
end


function layer_value(layer::Layer, μᵢ, λᵢ, bound::Hyperrectangle)
    W, b = layer.weights, layer.bias
    c, r = bound.center, bound.radius

    return λᵢ'c - μᵢ'(W*c + b) + sum(symbolic_abs.(λᵢ .- W'*μᵢ) .* r)
end
