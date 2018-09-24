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
        val = relaxed_ReLU_slope(l[j], u[j])
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

    l = Vector{Vector{Float64}}(n_layer)
    u = Vector{Vector{Float64}}(n_layer)
    γ = Vector{Vector{Float64}}(n_layer)
    μ = Vector{Vector{Vector{Float64}}}(n_layer)

    W, b = layers[1].weights, layers[1].bias
    v1   = W'
    γ[1] = b

    # Bounds for the first layer
    out1 = W * input + b
    Δ    = ϵ * sum(abs.(W), 2)  #TODO check sum(, 1) vs sum(, 2)
    l[1] = out1 - Δ
    u[1] = out1 + Δ

    for i in 2:n_layer

        W, b = layers[i].weights, layers[i].bias
        n_input  = length(l[i-1])
        n_output = length(b)

        # Form Ι⁻ᵢ, Ι⁺ᵢ, Ιᵢ; Form D
        prev_layer_ReLU_slopes = relaxed_ReLU_slope.(l[i-1], u[i-1])
        D = diagm(prev_layer_ReLU_slopes)   # a matrix whose diagonal values are the relaxed_ReLU values (maybe should be sparse?)

        # Initialize new terms
        μ[i] = [zeros(n_output) for j in 1:n_input]
        for j in 1:n_input
            if prev_layer_ReLU_slopes[j] == 0 # negative part
                μ[i][j] = W * D[:, j]
            end
        end
        γ[i] = b

        # Propagate existing terms
        DW = D*W'
        γ = [γ[j]*DW              for j in 1:i-1]
        μ = [[m*DW for m in μ[j]] for j in 2:i-1]

        v1 = v1 * DW

        # Compute bounds
        ψ = input' * v1 + sum(γ[j] for j in 1:i)

        colsums_v1 = vec(sum(abs.(v1), 1)) # sum down the columns after abs
        neg, pos = all_neg_pos_sums(i, n_output, prev_layer_ReLU_slopes, l, μ)
        l[i] = ψ - ϵ*colsums_v1 + neg
        u[i] = ψ + ϵ*colsums_v1 - pos
    end

    return l, u
end

# TODO rename function and inputs
function all_neg_pos_sums(i, n_output, slopes, l, mu)
    neg = zeros(n_output)
    pos = zeros(n_output)
    # Need to debug
    for j in 1:i-1
        for k in 1:length(mu[j+1])
            if 0 < slopes[k] < 1  # if in the triangle region
                posind = mu[j+1][k] .> 0

                neg .+= l[j][k] .* -mu[j+1][k] .* !posind  # multiply by boolean to set the undesired values to 0.0
                pos .+= l[j][k] .*  mu[j+1][k] .* posind
            end
        end
    end
    return neg, pos
end


function get_activation(reluslope::Float64)
    reluslope == 0.0 && return (:neg)
    reluslope == 1.0 && return (:pos)
    return (:tri)
end
function get_activation(l::Float64, u::Float64)
    u <= 0.0 && return (:neg)
    l >= 0.0 && return (:pos)
    return (:tri)
end
function relaxed_ReLU_slope(l::Float64, u::Float64)
    u <= 0.0 && return 0.0
    l >= 0.0 && return 1.0
    return u / (u - l)
end
