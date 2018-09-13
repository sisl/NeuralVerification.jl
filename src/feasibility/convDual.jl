# This method only works for half space output constraint
# c y >= b
# Input constraint needs to be a hyperrectangle with uniform radius
struct ConvDual
end

function solve(solver::ConvDual, problem::Problem)
    check_compactability(solver, problem)
    J = dual_cost(solver, problem)
    # Check if the lower bound satisfies the constraint
    # println("Dual cost: ", J)
    return ifelse(J[1] >= 0.0, Result(:True), Result(:Undetermined))
end

function check_compactability(solver::ConvDual, problem::Problem)
    # Input should be a hyperrectangle with uniform radius
    if typeof(problem.input) != LazySets.Hyperrectangle{Float64}
        return error("Input constraint needs to be a Hyperrectangle.")
    end
    if typeof(problem.output) != LazySets.HPolytope{Float64}
        return error("Output constraint needs to be a HPolytope.")
    end
end

# compute lower bound of the dual problem.
function dual_cost(solver::ConvDual, problem::Problem)
    l, u, act_pattern = get_bounds(problem.network, problem.input.center, problem.input.radius[1])
    n_layer = length(problem.network.layers)
    v, J = tosimplehrep(problem.output)

    # println("lower bound: ", l)
    # println("upper bound: ", u)
    # println("activation pattern: ", act_pattern)

    for i in n_layer:-1:1
        J -= v'*problem.network.layers[i].bias
        v = problem.network.layers[i].weights'*v
        if i > 1
            for j in 1:length(v)
                if act_pattern[i-1][j] < 1
                    v[j] = ifelse(act_pattern[i-1][j] == 0, u[i-1][j] * abs(v[j])/(u[i-1][j] - l[i-1][j]), 0)
                end
            end
            J += sum(ifelse(act_pattern[i-1][j] == 0 && v[j] > 0, l[i-1][j] * v[j], 0) for j in 1:length(v))
        end
    end
    J -= problem.input.center * v + problem.input.radius[1] * sum(abs.(v))
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
                neg[j] = sum(ifelse(act_pattern[i-1][k] == 0 && mu[j+1][k][ii] < 0, l[j][k]*(-mu[j+1][k][ii]), 0) for k in 1:length(mu[j+1]))
                pos[j] = sum(ifelse(act_pattern[i-1][k] == 0 && mu[j+1][k][ii] > 0, l[j][k]*(mu[j+1][k][ii]), 0) for k in 1:length(mu[j+1]))
            end
            l[i][ii] = phi[ii] - epsilon * sum(abs.(v1[:, ii])) + sum(neg)
            u[i][ii] = phi[ii] + epsilon * sum(abs.(v1[:, ii])) - sum(pos)
        end
    end

    act_pattern[n_layer], D = get_activation(l[n_layer], u[n_layer])

    return (l, u, act_pattern)
end

function get_activation(l::Vector{Float64}, u::Vector{Float64})
    n = length(l)
    act_pattern = fill(0, n)
    D = zeros(n, n)
    for i in 1:n
        if u[i] <= 0
            act_pattern[i] = -1
            D[i, i] = 0.0
        elseif l[i] >= 0
            act_pattern[i] = 1
            D[i, i] = 1.0
        else
            D[i, i] = u[i] / (u[i] - l[i])
        end
    end
    return (act_pattern, D)
end