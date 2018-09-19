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
    L, U, act_pattern = get_bounds(network, input.center, input.radius[1])
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


# This step is similar to reachability method
function get_bounds(nnet::Network, input::Vector{Float64}, epsilon::Float64)
    n_layer = length(nnet.layers)
    l = Vector{Vector{Float64}}(n_layer)
    u = Vector{Vector{Float64}}(n_layer)
    act_pattern = Vector{Vector{Float64}}(n_layer)

    gamma = Vector{Vector{Float64}}(n_layer)
    mu = Vector{Vector{Vector{Float64}}}(n_layer)

    v1 = nnet.layers[1].weights'
    gamma[1] = nnet.layers[1].bias

    # Bounds for the first layer
    output = nnet.layers[1].weights * input + nnet.layers[1].bias
    n_output = length(output)
    l[1] = zeros(n_output)
    u[1] = zeros(n_output)
    for i in 1:n_output
        l[1][i] = output[i] - epsilon * sum(abs.(nnet.layers[1].weights[i, :])) #TODO check [i, :] vs [:, i]
        u[1][i] = output[i] + epsilon * sum(abs.(nnet.layers[1].weights[i, :]))
    end

    for i in 2:n_layer
        act_pattern[i-1], D = get_activation(l[i-1], u[i-1])

        # Initialize new terms
        gamma[i] = nnet.layers[i].bias

        n_input = length(act_pattern[i-1])
        n_output = length(nnet.layers[i].bias)

        mu[i] = Vector{Vector{Float64}}(n_input)
        for j in 1:n_input
            if act_pattern[i-1][j] == 0
            #if relaxed_ReLU(l[i-1], u[i-1]) == -1 # equivalent
                mu[i][j] = nnet.layers[i].weights * D[:, j]
            else
                mu[i][j] = zeros(n_output)
            end
        end

        # Propagate existiing terms
        for j in 1:i-1
            ReLUed_weights = nnet.layers[i].weights * D
            if j > 1
                mu[j] = [ReLUed_weights * m for m in mu[j]]
            end
            gamma[j] = ReLUed_weights * gamma[j]
        end
        # (ABC')' = CB'A'
        v1 = v1 * D' * nnet.layers[i].weights'

        # Compute bounds
        phi = v1' * input + sum(gamma[j] for j in 1:i)

        vertical_sums_on_v1 = vec(sum(abs.(v1), 1)) # sum down the columns after abs
        neg, pos = all_neg_pos_sums(i, n_output, act_pattern, l, mu)
        l[i] = phi .- epsilon * vertical_sums_on_v1 + neg
        u[i] = phi .+ epsilon * vertical_sums_on_v1 - pos
    end

    act_pattern[n_layer], D = get_activation(l[n_layer], u[n_layer])

    return (l, u, act_pattern)
end

# TODO rename function and inputs
function all_neg_pos_sums(i, n_output, act_pattern, l, mu)
    neg = zeros(n_output)
    pos = zeros(n_output)
    # Need to debug
    for j in 1:i-1
        for k in 1:length(mu[j+1])
            if act_pattern[i-1][k] == 0  # relaxed_ReLU(l, u) !∈ (0.0, 1.0)
                muvals = mu[j+1][k]
                for (ii, val) in enumerate(muvals)
                    if     val < 0    neg[ii] += l[j][k] * -val
                    elseif val > 0    pos[ii] += l[j][k] *  val
                    end
                end
            end
        end
    end
    return neg, pos
end




function get_activation(l::Vector{Float64}, u::Vector{Float64})
    act_pattern = get_activation.(l, u)
    D  = spdiagm(relaxed_ReLU.(l, u))  # a sparse matrix whose diagonal values are the relaxed_ReLU values
    return (act_pattern, D)
end
function get_activation(l::Float64, u::Float64)
    u <= 0.0 && return -1
    l >= 0.0 && return 1
    return 0
end
function relaxed_ReLU(l::Float64, u::Float64)
    u <= 0.0 && return 0.0
    l >= 0.0 && return 1.0
    return u / (u - l)
end
