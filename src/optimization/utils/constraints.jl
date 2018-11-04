# This file is for different constraints
# Default activation: ReLU

# Encode constraint as LP according to the activation pattern
# this is used in Sherlock
function encode_lp(model::Model, nnet::Network, act_pattern::Vector{Vector{Bool}}, neurons)
    for (i, layer) in enumerate(nnet.layers)
        before_act = layer.weights * neurons[i] + layer.bias
        for j in 1:length(layer.bias)
            if act_pattern[i][j]
                @constraint(model, before_act[j] >= 0.0)
                @constraint(model, neurons[i+1][j] == before_act[j])
            else
                @constraint(model, before_act[j] <= 0.0)
                @constraint(model, neurons[i+1][j] == 0.0)
            end
        end
    end
    return nothing
end

# This function is called in iLP
function encode_relaxed_lp(model::Model, nnet::Network, act_pattern::Vector{Vector{Bool}}, neurons)
    for (i, layer) in enumerate(nnet.layers)
        before_act = layer.weights * neurons[i] + layer.bias
        for j in 1:length(layer.bias)
            if act_pattern[i][j]
                @constraint(model, neurons[i+1][j] == before_act[j])
            else
                @constraint(model, neurons[i+1][j] == 0.0)
            end
        end
    end
    return nothing
end

# Encode constraint as LP according to the Δ relaxation of ReLU
# This function is called in planet and bab
function encode_Δ_lp(model::Model, nnet::Network, bounds::Vector{Hyperrectangle}, neurons)
    for (i, layer) in enumerate(nnet.layers)
        (W, b, act) = (layer.weights, layer.bias, layer.activation)
        before_act = W * neurons[i] + b
        before_act_rectangle = linear_transformation(layer, bounds[i])
        lower, upper = low(before_act_rectangle), high(before_act_rectangle)
        # For now assume only ReLU activation
        for j in 1:length(layer.bias) # For evey node
            if lower[j] > 0.0
                @constraint(model, neurons[i+1][j] == before_act[j])
            elseif upper[j] < 0.0
                @constraint(model, neurons[i+1][j] == 0.0)
            else # Here use triangle relaxation
                @constraints(model, begin
                                    neurons[i+1][j] >= before_act[j]
                                    neurons[i+1][j] <= upper[j] / (upper[j] - lower[j]) * (before_act[j] - lower[j])
                                    neurons[i+1][j] >= 0.0
                                end)
            end
        end
    end
    return nothing
end

function encode_slack_lp(model::Model, nnet::Network, p::Vector{Vector{Int64}}, neurons)
    slack = Vector{Vector{Variable}}(length(nnet.layers))
    for (i, layer) in enumerate(nnet.layers)
        before_act = layer.weights * neurons[i] + layer.bias
        slack[i] = @variable(model, [1:length(layer.bias)])
        for j in 1:length(layer.bias)
            if p[i][j] == 1
                @constraint(model, neurons[i+1][j] == before_act[j] + slack[i][j])
                @constraint(model, before_act[j] + slack[i][j] >= 0.0)
            else
                @constraint(model, neurons[i+1][j] == 0.0)
                @constraint(model, 0.0 >= before_act[j] - slack[i][j])
            end
        end
    end
    return slack
end

# Encode constraint as MIP without bounds
# This function is called in Reverify
function encode_mip_constraint(model::Model, nnet::Network, M::Float64, neurons, deltas)
    for (i, layer) in enumerate(nnet.layers)
        lbounds = layer.weights * neurons[i] + layer.bias
        dy = M*(deltas[i])  # TODO rename variable
        ubounds = lbounds + dy
        for j in 1:length(layer.bias)
            @constraints(model, begin
                                    neurons[i+1][j] >= lbounds[j]
                                    neurons[i+1][j] <= ubounds[j]
                                    neurons[i+1][j] >= 0.0
                                    neurons[i+1][j] <= M-dy[j]
                                end)
        end
    end
    return nothing
end

# Encode constraint as MIP with bounds
# This function is called in MIPVerify
function encode_mip_constraint(model::Model, nnet::Network, bounds::Vector{Hyperrectangle}, neurons, deltas)
    for (i, layer) in enumerate(nnet.layers)
        (W, b, act) = (layer.weights, layer.bias, layer.activation)
        before_act = W * neurons[i] + b
        before_act_rectangle = linear_transformation(layer, bounds[i])
        lower, upper = low(before_act_rectangle), high(before_act_rectangle)

        for j in 1:length(layer.bias) # For evey node
            if lower[j] >= 0.0
                @constraint(model, neurons[i+1][j] == before_act[j])
            elseif upper[j] <= 0.0
                @constraint(model, neurons[i+1][j] == 0.0)
            else
                @constraints(model, begin
                                    neurons[i+1][j] >= before_act[j]
                                    neurons[i+1][j] <= upper[j] * deltas[i][j]
                                    neurons[i+1][j] >= 0.0
                                    neurons[i+1][j] <= before_act[j] - lower[j] * (1 - deltas[i][j])
                                end)
            end
        end
    end
    return nothing
end


#=
Add input/output constraints to model
=#
function add_complementary_output_constraint(model::Model, output::AbstractPolytope, neuron_vars::Vector{Variable})
    out_A, out_b = tosimplehrep(output)
    # Needs to take the complementary of output constraint
    n = length(out_b)
    if n == 1
        # Here the output constraint is a half space
        # So the complementary is just out_A * y .> out_b
        @constraint(model, -out_A * neuron_vars .<= -out_b)
    else
        # Here the complementary is a union of different constraints
        # We use binary variable to encode the union of constraints
        out_deltas = @variable(model, [1:n], Bin)
        @constraint(model, sum(out_deltas) == 1)
        for i in 1:n
            @constraint(model, -out_A[i, :]' * neuron_vars * out_deltas[i] <= -out_b[i] * out_deltas[i])
        end
    end
    return nothing
end

function add_input_constraint(model::Model, input::HPolytope, neuron_vars::Vector{Variable})
    in_A,  in_b  = tosimplehrep(input)
    @constraint(model,  in_A * neuron_vars .<= in_b)
    return nothing
end

function add_input_constraint(model::Model, input::Hyperrectangle, neuron_vars::Vector{Variable})
    @constraint(model,  neuron_vars .<= high(input))
    @constraint(model,  neuron_vars .>= low(input))
    return nothing
end

function add_output_constraint(model::Model, output::HPolytope, neuron_vars::Vector{Variable})
    out_A, out_b = tosimplehrep(output)
    @constraint(model, out_A * neuron_vars .<= out_b)
    return nothing
end