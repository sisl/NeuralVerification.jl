# This contains several objectives

# Objective: L∞ norm of the disturbance
function max_disturbance(model::Model, var)
    J = maximum(symbolic_abs(var))
    @objective(model, Min, J)
    return J
end

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