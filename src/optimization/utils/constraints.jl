# This file is for different constraints
# Default activation: ReLU

# Encode constraint as LP according to the activation pattern
# this is used in Sherlock
function encode_lp!(model::Model, nnet::Network, δ::Vector{Vector{Bool}}, z)
    for (i, layer) in enumerate(nnet.layers)
        ẑ = layer.weights * z[i] + layer.bias
        for j in 1:length(layer.bias)
            if δ[i][j]
                @constraint(model, ẑ[j] >= 0.0)
                @constraint(model, z[i+1][j] == ẑ[j])
            else
                @constraint(model, ẑ[j] <= 0.0)
                @constraint(model, z[i+1][j] == 0.0)
            end
        end
    end
    return nothing
end

# This function is called in iLP
function encode_relaxed_lp!(model::Model, nnet::Network, δ::Vector{Vector{Bool}}, z)
    for (i, layer) in enumerate(nnet.layers)
        ẑ = layer.weights * z[i] + layer.bias
        for j in 1:length(layer.bias)
            if δ[i][j]
                @constraint(model, z[i+1][j] == ẑ[j])
            else
                @constraint(model, z[i+1][j] == 0.0)
            end
        end
    end
    return nothing
end

# Encode constraint as LP according to the Δ relaxation of ReLU
# This function is called in planet and bab
function encode_Δ_lp!(model::Model, nnet::Network, bounds::Vector{Hyperrectangle}, z)
    for (i, layer) in enumerate(nnet.layers)
        ẑ = layer.weights * z[i] + layer.bias
        ẑ_bound = linear_transformation(layer, bounds[i])
        l̂, û = low(ẑ_bound), high(ẑ_bound)
        for j in 1:length(layer.bias)
            if l̂[j] > 0.0
                @constraint(model, z[i+1][j] == ẑ[j])
            elseif û[j] < 0.0
                @constraint(model, z[i+1][j] == 0.0)
            else
                @constraints(model, begin
                                    z[i+1][j] >= ẑ[j]
                                    z[i+1][j] <= û[j] / (û[j] - l̂[j]) * (ẑ[j] - l̂[j])
                                    z[i+1][j] >= 0.0
                                end)
            end
        end
    end
    return nothing
end

function encode_slack_lp!(model::Model, nnet::Network, δ::Vector{Vector{Bool}}, z)
    slack = Vector{Vector{Variable}}(undef, length(nnet.layers))
    for (i, layer) in enumerate(nnet.layers)
        ẑ = layer.weights * z[i] + layer.bias
        slack[i] = @variable(model, [1:length(layer.bias)])
        for j in 1:length(layer.bias)
            if δ[i][j]
                @constraint(model, z[i+1][j] == ẑ[j] + slack[i][j])
                @constraint(model, ẑ[j] + slack[i][j] >= 0.0)
            else
                @constraint(model, z[i+1][j] == 0.0)
                @constraint(model, 0.0 >= ẑ[j] - slack[i][j])
            end
        end
    end
    return slack
end

# Encode constraint as MIP without bounds
# This function is called in Reverify
function encode_mip_constraint!(model::Model, nnet::Network, m::Float64, z, δ)
    for (i, layer) in enumerate(nnet.layers)
        ẑ = layer.weights * z[i] + layer.bias
        for j in 1:length(layer.bias)
            @constraints(model, begin
                                    z[i+1][j] >= ẑ[j]
                                    z[i+1][j] >= 0.0
                                    z[i+1][j] <= ẑ[j] + m * δ[i][j]
                                    z[i+1][j] <= m - m * δ[i][j]
                                end)
        end
    end
    return nothing
end

# Encode constraint as MIP with bounds
# This function is called in MIPVerify
function encode_mip_constraint!(model::Model, nnet::Network, bounds::Vector{Hyperrectangle}, z, δ)
    for (i, layer) in enumerate(nnet.layers)
        ẑ = layer.weights * z[i] + layer.bias
        ẑ_bound = linear_transformation(layer, bounds[i])
        l̂, û = low(ẑ_bound), high(ẑ_bound)

        for j in 1:length(layer.bias) # For evey node
            if l̂[j] >= 0.0
                @constraint(model, z[i+1][j] == ẑ[j])
            elseif û[j] <= 0.0
                @constraint(model, z[i+1][j] == 0.0)
            else
                @constraints(model, begin
                                        z[i+1][j] >= ẑ[j]
                                        z[i+1][j] >= 0.0
                                        z[i+1][j] <= û[j] * δ[i][j]
                                        z[i+1][j] <= ẑ[j] - l̂[j] * (1 - δ[i][j])
                                    end)
            end
        end
    end
    return nothing
end


#=
Add input/output constraints to model
=#
function add_complementary_output_constraint!(model::Model, output::HPolytope, neuron_vars::Vector{Variable})
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

function add_complementary_output_constraint!(model::Model, output::Hyperrectangle, neuron_vars::Vector{Variable})
    @constraint(model, neuron_vars .>= -high(output))
    @constraint(model, neuron_vars .<= -low(output))
    return nothing
end

function add_set_constraint!(model::Model, set::HPolytope, neuron_vars::Vector{Variable})
    A, b = tosimplehrep(set)
    @constraint(model,  A * neuron_vars .<= b)
    return nothing
end

function add_set_constraint!(model::Model, set::Hyperrectangle, neuron_vars::Vector{Variable})
    @constraint(model,  neuron_vars .<= high(set))
    @constraint(model,  neuron_vars .>= low(set))
    return nothing
end
