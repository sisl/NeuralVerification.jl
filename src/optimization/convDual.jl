"""
    ConvDual

ConvDual uses convex relaxation to compute over-approximated bounds for a network

# Problem requirement
1. Network: any depth, ReLU activation
2. Input: hypercube
3. Output: halfspace

# Return
`BasicResult`

# Method
Convex relaxation with duality.

# Property
Sound but not complete.

# Reference
[E. Wong and J. Z. Kolter, "Provable Defenses against Adversarial Examples via the Convex Outer Adversarial Polytope,"
*ArXiv Preprint ArXiv:1711.00851*, 2017.](https://arxiv.org/abs/1711.00851)

[https://github.com/locuslab/convex_adversarial](https://github.com/locuslab/convex_adversarial)
"""
struct ConvDual end

function solve(solver::ConvDual, problem::Problem)
    o = dual_value(solver, problem.network, problem.input, problem.output)
    # Check if the lower bound satisfies the constraint
    if o >= 0.0
        return BasicResult(:holds)
    end
    return BasicResult(:unknown)
end

# compute lower bound of the dual problem.
function dual_value(solver::ConvDual, network::Network, input::Hyperrectangle{N}, output::HPolytope{N}) where N

    @assert all(iszero.(input.radius .- input.radius[1])) "input.radius must be uniform. Got $(input.radius)"

    layers = network.layers
    L, U  = get_bounds(network, input.center, input.radius[1])
    v0, d = tosimplehrep(output)

    v = vec(v0)
    o = d[1]

    for i in reverse(1:length(layers))
        o -= v'*layers[i].bias
        v = layers[i].weights'*v
        if i>1
            o += backprop!(v, U[i-1], L[i-1])
        end
    end
    o -= input.center' * v + input.radius[1] * sum(abs.(v))
    return o
end

#=
modifies v and returns o
=#
function backprop!(v::Vector{Float64}, u::Vector{Float64}, l::Vector{Float64})
    o = 0.0
    for j in 1:length(v)
        val = relaxed_ReLU(l[j], u[j])
        if val < 1.0 # if val is 1, it means ReLU result is identity so do not update (NOTE is that the right reasoning?)
            v[j] = v[j] * val
            o += v[j] * l[j]
        end
    end
    return o
end

# Forward_network and forward_layer:
# This step is similar to reachability method
function get_bounds(nnet::Network, input::Vector{Float64}, ϵ::Float64)
    layers  = nnet.layers
    n_layer = length(layers)

    l = Vector{Vector{Float64}}()
    u = Vector{Vector{Float64}}()
    γ = Vector{Vector{Float64}}()
    μ = Vector{Vector{Vector{Float64}}}()

    v1 = layers[1].weights'
    push!(γ, layers[1].bias)
    # Bounds for the first layer
    l1, u1 = input_layer_bounds(layers[1], input, ϵ)
    push!(l, l1)
    push!(u, u1)

    for i in 2:n_layer
        n_input  = length(layers[i-1].bias)
        n_output = length(layers[i].bias)

        input_ReLU = relaxed_ReLU.(last(l), last(u))
        D = Diagonal(input_ReLU)   # a matrix whose diagonal values are the relaxed_ReLU values (maybe should be sparse?)

        # Propagate existing terms
        WD = layers[i].weights*D
        v1 = v1 * WD' # TODO CHECK
        map!(g -> WD*g,   γ, γ)
        for M in μ
            map!(m -> WD*m,   M, M)
        end
        # New terms
        push!(γ, layers[i].bias)
        push!(μ, new_μ(n_input, n_output, input_ReLU, WD))

        # Compute bounds
        ψ = v1' * input + sum(γ)
        eps_v1_sum = ϵ * vec(sum(abs, v1, dims = 1))
        neg, pos = all_neg_pos_sums(input_ReLU, l, μ, n_output)
        push!(l,  ψ - eps_v1_sum + neg )
        push!(u,  ψ + eps_v1_sum - pos )
    end

    return l, u
end

# TODO rename function and inputs
function all_neg_pos_sums(slopes, l, μ, n_output)
    # n_output = length(last(l))
    neg = zeros(n_output)
    pos = zeros(n_output)
    # Need to debug
    for (i, ℓ) in enumerate(l)                # ℓ::Vector{Float64}
        for (j, M) in enumerate(μ[i])         # M::Vector{Float64}
            if 0 < slopes[j] < 1              # if in the triangle region of relaxed ReLU
                #posind = M .> 0
                neg .+= ℓ[j] * min.(M, 0) #-M .* !posind  # multiply by boolean to set the undesired values to 0.0
                pos .+= ℓ[j] * max.(M, 0) #M .* posind
            end
        end
    end
    return neg, pos
end

function input_layer_bounds(input_layer, input, ϵ)
    W, b = input_layer.weights, input_layer.bias

    out1 = vec(W * input + b)
    Δ    = ϵ * vec(sum(abs, W, dims = 2))

    l = out1 - Δ
    u = out1 + Δ
    return l, u
end


function new_μ(n_input, n_output, input_ReLU, WD)
    sub_μ = Vector{Vector{Float64}}(undef, n_input)
    for j in 1:n_input
        if 0 < input_ReLU[j] < 1 # negative region  ## TODO CONFIRM. Previously input_ReLU[j] == 0
            sub_μ[j] = WD[:, j] # TODO CONFIRM
        else
            sub_μ[j] = zeros(n_output)
        end
    end
    return sub_μ
end

function relaxed_ReLU(l::Float64, u::Float64)
    u <= 0.0 && return 0.0
    l >= 0.0 && return 1.0
    return u / (u - l)
end
