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
        if val < 1.0 # if val is 1, it means ReLU result is identity so do not update (NOTE is the the right reasoning?)
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
    l[1] = fill(0.0, n_output)
    u[1] = fill(0.0, n_output)
    for i in 1:n_output
        l[1][i] = output[i] - epsilon * sum(abs.(nnet.layers[1].weights[i, :]))
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
            mu[i][j] = ifelse(act_pattern[i-1][j] == 0, nnet.layers[i].weights * D[:, j], zeros(n_output))
        end

        # Propagate existiing terms
        for j in 1:i-1
            if j > 1
                for k in 1:length(mu[j])
                    mu[j][k] = nnet.layers[i].weights * D * mu[j][k]
                end
            end
            gamma[j] = nnet.layers[i].weights * D * gamma[j]
        end
        v1 = nnet.layers[i].weights * D * v1'
        v1 = v1'

        # Compute bounds
        l[i] = fill(0.0, n_output)
        u[i] = fill(0.0, n_output)
        phi = v1' * input + sum(gamma[j] for j in 1:i)

        for ii in 1:n_output
            neg = fill(0.0, i-1)
            pos = fill(0.0, i-1)
            # Need to debug
            for j in 1:i-1
                for k in 1:length(mu[j+1])
                    if act_pattern[i-1][k] == 0
                        if mu[j+1][k][ii] < 0
                            neg[j] = l[j][k] * (-mu[j+1][k][ii])
                        elseif mu[j+1][k][ii] > 0
                            pos[j] = l[j][k] * (mu[j+1][k][ii])
                        end
                    end
                end
            end
            l[i][ii] = phi[ii] - epsilon * sum(abs.(v1[:, ii])) + sum(neg)
            u[i][ii] = phi[ii] + epsilon * sum(abs.(v1[:, ii])) - sum(pos)
        end
    end

    act_pattern[n_layer], D = get_activation(l[n_layer], u[n_layer])

    return (l, u, act_pattern)
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
