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

function solve(solver::Duality, problem::Problem)

    @assert is_halfspace_equivalent(problem.output) "Duality only accepts HalfSpace output sets. Got a $(length(d))-d polytope."

    model = Model(solver)
    c, d = tosimplehrep(problem.output)
    d = d[1]

    λ = init_vars(model, problem.network, :λ)
    μ = init_vars(model, problem.network, :μ)
    o = dual_value(solver, problem, model, λ, μ)
    @constraint(model, last(λ) .== -c)
    optimize!(model)

    # Interpret result:
    if termination_status(model) != OPTIMAL
        return BasicResult(:unknown)
    elseif value(o) - d <= 0.0
        return BasicResult(:holds)
    else
        return BasicResult(:violated)
    end
end

function dual_value(solver::Duality,
                    problem::Problem,
                    model::Model,
                    λ::Vector{Vector{VariableRef}},
                    μ::Vector{Vector{VariableRef}})
    bounds = get_bounds(problem)
    layers = problem.network.layers

    # prepend an array of 0s to λ so the λᵢ
    # term cancels in the input layer
    λ = [zeros(dim(bounds[1])), λ...]

    o = 0
    for i in 1:length(layers)
        Lᵢ, Bᵢ = layers[i], bounds[i]
        W, b = Lᵢ.weights, Lᵢ.bias
        c, r = Bᵢ.center, Bᵢ.radius
        B̂ᵢ₊₁ = approximate_affine_map(Lᵢ, Bᵢ)

        # layer value
        o += λ[i]'*c - μ[i]'*(W*c + b) + sum(symbolic_abs.(λ[i] .- W'*μ[i]) .* r)

        # activation value
        o += activation_value(Lᵢ.activation, μ[i], λ[i+1], low(B̂ᵢ₊₁), high(B̂ᵢ₊₁))
    end

    @objective(model, Min, o)
    return o
end

function activation_value(σ::ReLU, μᵢ, λᵢ, l̂ᵢ, ûᵢ)

    gᵢl̂ᵢ = @. μᵢ*l̂ᵢ - λᵢ*σ(l̂ᵢ)
    gᵢûᵢ = @. μᵢ*ûᵢ - λᵢ*σ(ûᵢ)

    max = symbolic_max

    # NOTE: ifelse evaluates all of its arguments in advance.
    # This may seem like a bug since symbolic_max mutates the model
    # and introduces new variables, but it is fine. Only those
    # variables that are part of the objective will affect the problem,
    # and those will only the the ones summed *after* taking the ifelse
    return sum(@. ifelse(l̂ᵢ < 0 < ûᵢ,
                         max(gᵢl̂ᵢ, gᵢûᵢ, 0),
                         max(gᵢl̂ᵢ, gᵢûᵢ)))
end

function activation_value(σ::Id, μᵢ, λᵢ, l̂ᵢ, ûᵢ)
    max = symbolic_max
    sum(@. max(μᵢ*l̂ᵢ - λᵢ*l̂ᵢ, μᵢ*ûᵢ - λᵢ*ûᵢ))
end

function activation_value(σ::Any, μᵢ, λᵢ, l̂ᵢ, ûᵢ)
    max = symbolic_max
    sum(@. max(μᵢ*l̂ᵢ, μᵢ*ûᵢ) + max(-λᵢ*σ(l̂ᵢ), -λᵢ*σ(ûᵢ)))
end