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
struct ConvDual <: Solver end

function solve(solver::ConvDual, problem::Problem)
    o = dual_value(solver, problem.network, problem.input, problem.output)
    # Check if the lower bound satisfies the constraint
    if o >= 0.0
        return BasicResult(:holds)
    end
    return BasicResult(:violated)
end

# compute lower bound of the dual problem.
function dual_value(solver::ConvDual, network::Network, input::Hyperrectangle, output)

    @assert is_hypercube(input) "ConvDual only accepts hypercube input constraints."
    @assert is_halfspace_equivalent(output) "ConvDual only accepts HalfSpace output contraints"

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
        val = relaxed_relu_gradient(l[j], u[j])
        if val < 1.0 # if val is 1, it means ReLU result is identity so do not update (NOTE is that the right reasoning?)
            v[j] = v[j] * val
            o += v[j] * l[j]
        end
    end
    return o
end

function get_bounds(solver::ConvDual, nnet::Network, input::Hyperrectangle; before_act::Bool=true)
    # only supports pre-activation bounds for now
    @assert before_act == true "ConvDual get_bounds only currently supports pre-activation bounds"
    # only supports hypercubes in this implementation
    @assert all(input.radius[1] .== input.radius)
    L, U = get_bounds(nnet, input.center, input.radius[1])
    # add the bounds from the input set
    pushfirst!(L, low(input))
    pushfirst!(U, high(input))
    # convert convdual's bounds into hyperrectangles
    return [Hyperrectangle(low=L[i], high=U[i]) for i = 1:length(L)]
end

# Forward_network and forward_layer:
# This step is similar to reachability method
function get_bounds(nnet::Network, input::Vector{Float64}, ϵ::Float64)
    layers  = nnet.layers

    l = Vector{Vector{Float64}}() # Lower bound
    u = Vector{Vector{Float64}}() # Upper bound
    b = Vector{Vector{Float64}}() # bias
    μ = Vector{Vector{Vector{Float64}}}() # Dual variables
    input_ReLU = Vector{Vector{Float64}}()

    v1 = layers[1].weights'
    push!(b, layers[1].bias)
    # Bounds for the first layer
    l1, u1 = input_layer_bounds(layers[1], input, ϵ)
    push!(l, l1)
    push!(u, u1)

    for i in 2:length(layers)
        n_input  = length(layers[i-1].bias)
        n_output = length(layers[i].bias)


        last_input_ReLU = relaxed_relu_gradient.(last(l), last(u))
        push!(input_ReLU, last_input_ReLU)
        D = Diagonal(last_input_ReLU)   # a matrix whose diagonal values are the relaxed_ReLU values (maybe should be sparse?)

        # Propagate existing terms by right multiplication of D*W' or left multiplication of W*D
        WD = layers[i].weights*D
        v1 = v1 * WD' # propagate V_1^{i-1} to V_1^{i}
        map!(g -> WD*g,   b, b) # propagate bias
        for V in μ
            map!(m -> WD*m,   V, V) # Updating ν_j for all previous layers
        end

        # New terms
        push!(b, layers[i].bias)
        push!(μ, new_μ(n_input, n_output, last_input_ReLU, WD))

        # Compute bounds
        ψ = v1' * input + sum(b)
        eps_v1_sum = ϵ * vec(sum(abs, v1, dims = 1))
        neg, pos = residual(input_ReLU, l, μ, n_output)
        push!(l,  ψ - eps_v1_sum - neg )
        push!(u,  ψ + eps_v1_sum - pos )
    end

    return l, u
end

# TODO rename function and inputs
function residual(slopes, l, μ, n_output)
    # n_output = length(last(l))
    neg = zeros(n_output)
    pos = zeros(n_output)
    # Need to debug
    for (i, ℓ) in enumerate(l)                # ℓ::Vector{Float64}
        for (j, V) in enumerate(μ[i])         # M::Vector{Float64}
            if 0 < slopes[i][j] < 1              # if in the triangle region of relaxed ReLU
                #posind = M .> 0
                neg .+= ℓ[j] * min.(V, 0) #-M .* !posind  # multiply by boolean to set the undesired values to 0.0
                pos .+= ℓ[j] * max.(V, 0) #M .* posind
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
