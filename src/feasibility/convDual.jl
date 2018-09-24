# This method only works for half space output constraint
# c y >= b
# Input constraint needs to be a hyperrectangle with uniform radius
struct ConvDual
end

function solve(solver::ConvDual, problem::Problem)
    J = dual_cost(solver, problem.network, problem.input, problem.output)
    # Check if the lower bound satisfies the constraint
    if J[1] >= 0.0
        return Result(:True)
    end
    return Result(:Undetermined)
end

# compute lower bound of the dual problem.
function dual_cost(solver::ConvDual, network::Network, input::Hyperrectangle{N}, output::HPolytope{N}) where N

    @assert iszero(input.radius - input.radius[1]) "input.radius must be uniform. Got $(input.radius)"

    layers = network.layers
    L, U = get_bounds(network, input.center, input.radius[1])
    v, J = tosimplehrep(output)

    for i in reverse(1:length(layers))
        J -= v'*layers[i].bias
        v = layers[i].weights'*v
        if i>1
            J += backprop!(v, U[i-1], L[i-1])
        end
    end
    J -= input.center * v + input.radius[1] * sum(abs.(v))
    return J
end

#=
modifies v and returns J
=#
function backprop!(v, u, l)
    J = 0.0
    for j in 1:length(v)
        val = relaxed_ReLU(l[j], u[j])
        if val < 1.0 # if val is 1, it means ReLU result is identity so do not update (NOTE is that the right reasoning?)
            v[j] = abs(v[j]) * val
            J += v[j] * l[j]
        end
    end
    return J
end

# Forward_network and forward_layer:
# This step is similar to reachability method
function get_bounds(nnet::Network, input::Vector{Float64}, ϵ::Float64)
    layers  = nnet.layers
    n_layer = length(layers)

    l = Vector{Vector{Float64}}(0)              ; sizehint!(l, n_layer)
    u = Vector{Vector{Float64}}(0)              ; sizehint!(u, n_layer)
    γ = Vector{Vector{Float64}}(0)              ; sizehint!(γ, n_layer)
    μ = Vector{Vector{Vector{Float64}}}(0)      ; sizehint!(μ, n_layer)

    v1 = layers[1].weights'
    push!(γ, layers[1].bias)
    # Bounds for the first layer
    l1, u1 = input_layer_bounds(layers[1], input, ϵ)
    push!(l, l1)
    push!(u, u1)

    for i in 2:n_layer
        # NOTE: no good way of mutating v1 inside
        v1, new_l, new_u = bounds_forward_layer!(layers[i], l, u, μ, v1, γ, ϵ)
        push!(l, new_l)
        push!(u, new_u)
    end

    return l, u
end

function bounds_forward_layer!(layer, l, u, μ, v1, γ, ϵ)

    W, b = layer.weights, layer.bias
    n_in  = length(last(l))
    n_out = length(b)

    input_ReLU = relaxed_ReLU.(last(l), last(u))
    D = diagm(input_ReLU)   # a matrix whose diagonal values are the relaxed_ReLU values (maybe should be sparse?)

    # Propagate existing terms
    DW = D*W'
    v1 = v1 * DW
    map!(g -> g*DW,   γ, γ)
    for M in μ
        map!(m -> m*DW,   M, M)
    end
    # New terms
    push!(γ, b)
    push!(μ, new_μ(n_in, n_out, input_ReLU))

    # Compute bounds
    ψ = input' * v1 + sum(γ)
    colsums_v1 = vec(sum(abs, v1, 1)) # sum down the columns after abs
    colsums_v1 *= ϵ
    neg, pos = all_neg_pos_sums(input_ReLU, l, μ)
    new_l = ψ - colsums_v1 + neg
    new_u = ψ + colsums_v1 - pos

    return v1, new_l, new_u
end

# TODO rename function and inputs
function all_neg_pos_sums(slopes, l, mu)
    n_output = length(first(l))
    neg = zeros(n_output)
    pos = zeros(n_output)
    # Need to debug
    for (j, ℓ) in enumerate(l)          # ℓ::Vector{Float64}
        for (k, M) in enumerate(mu[j])  # M::Vector{Float64}
            if 0 < slopes[k] < 1  # if in the triangle region of relaxed ReLU
                posind = M .> 0

                neg .+= ℓ[k] * -M .* !posind  # multiply by boolean to set the undesired values to 0.0
                pos .+= ℓ[k] *  M .* posind
            end
        end
    end
    return neg, pos
end

function input_layer_bounds(input_layer, input, ϵ)
    W, b = input_layer.weights, input_layer.bias
    out1 = W * input + b
    Δ    = ϵ * sum(abs, W, 2)  #TODO check sum(, 1) vs sum(, 2)
    l = out1 - Δ
    u = out1 + Δ
    return l, u
end


function new_μ(n_input, n_output, input_ReLU)
    μ = Vector{Vector{Float64}}(n_input)
    for j in 1:n_input
        if input_ReLU[j] == 0 # negative part
            μ[j] = W * D[:, j]
        else
            μ[j] = zeros(n_output)
        end
    end
    return μ
end

function relaxed_ReLU(l::Float64, u::Float64)
    u <= 0.0 && return 0.0
    l >= 0.0 && return 1.0
    return u / (u - l)
end
