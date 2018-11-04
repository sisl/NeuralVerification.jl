# This contains several objectives

##
# TODO make objectives have ! names
##
# Objective: L∞ norm of the disturbance
function max_disturbance(model::Model, var)
    J = maximum(symbolic_abs(var))
    @objective(model, Min, J)
    return J
end

function min_sum_all(model::Model, var)
    J = 0.0
    for i in 1:length(var)
        for j in 1:length(var[i])
            J += var[i][j]
        end
    end
    @objective(model, Min, J)
    return J
end

function max_sum_all(model::Model, var)
    J = 0.0
    for i in 1:length(var)
        for j in 1:length(var[i])
            J += var[i][j]
        end
    end
    @objective(model, Max, J)
    return J
end

function zero_objective(model::Model)
    @objective(model, Max, 0.0)
end

# NOTE: if this is only used in duality, it should probably be  defined there.
# particularly since dual_cost for convDual is defined differently
function dual_cost(model::Model, nnet::Network, bounds::Vector{Hyperrectangle}, λ, μ)
    J = input_layer_cost(nnet.layers[1], μ[1], bounds[1])
    for i in 2:length(nnet.layers)
        J += layer_cost(nnet.layers[i], μ[i], λ[i-1], bounds[i])
    end
    for i in 1:length(nnet.layers)
        J += activation_cost(nnet.layers[i], μ[i], λ[i], bounds[i])
    end
    @objective(model, Min, J)
    return J
end