# This file is for different constraints

abstract type AbstractLinearProgram end
struct StandardLP            <: AbstractLinearProgram end
struct LinearRelaxedLP       <: AbstractLinearProgram end
struct TriangularRelaxedLP   <: AbstractLinearProgram end
struct BoundedMixedIntegerLP <: AbstractLinearProgram end
struct SlackLP <: AbstractLinearProgram
    slack::Vector{Vector{Variable}}
end
SlackLP() = SlackLP([])
struct MixedIntegerLP <: AbstractLinearProgram
    m::Float64
end

# Any encoding passes through here first:
function encode_network!(model::Model,
                         network::Network,
                         zs::Vector{Vector{Variable}},
                         δs::Vector,
                         encoding::AbstractLinearProgram)

    for (i, layer) in enumerate(network.layers)
        encode_layer!(encoding, model, layer, zs[i], zs[i+1], δs[i])
    end
    return encoding # only matters for SlackLP
end

# TODO: find a way to eliminate the two methods below.
# i.e. make BoundedMixedIntegerLP(bounds) and TriangularRelaxedLP(bounds) or something
function encode_network!(model::Model,
                         network::Network,
                         zs::Vector{Vector{Variable}},
                         δs::Vector,
                         bounds::Vector{Hyperrectangle},
                         encoding::AbstractLinearProgram)

    for (i, layer) in enumerate(network.layers)
        encode_layer!(encoding, model, layer, zs[i], zs[i+1], δs[i], bounds[i])
    end
    return encoding # only matters for SlackLP
end

function encode_network!(model::Model,
                         network::Network,
                         zs::Vector{Vector{Variable}},
                         bounds::Vector{Hyperrectangle},
                         encoding::AbstractLinearProgram)

    for (i, layer) in enumerate(network.layers)
        encode_layer!(encoding, model, layer, zs[i], zs[i+1], bounds[i])
    end
    return encoding # only matters for SlackLP
end


# For an Id Layer, any encoding type defaults to this:
function encode_layer!(::AbstractLinearProgram,
                       model::Model,
                       layer::Layer{Id},
                       z_current::Vector{Variable},
                       z_next::Vector{Variable},
                       args...)
    @constraint(model, z_next .== layer.weights*z_current + layer.bias)
end

# SlackLP is slightly different, because we need to keep track of the slack variables
function encode_layer!(SLP::SlackLP,
                       model::Model,
                       layer::Layer{Id},
                       z1::Array{Variable,1},
                       z2::Array{Variable,1},
                       δ...)

    encode_layer!(StandardLP(), model, layer, z1, z2)
    # We need identity layer slack variables so that the algorithm doesn't
    # "get confused", but they are set to 0 because they're not relevant
    slack_vars = @variable(model, [1:n_nodes(layer)])
    @constraint(model, slack_vars .== 0.0)
    push!(SLP.slack, slack_vars)
    return nothing
end

# alternative signature:
# encode_layer!(encoding, model::Model, current_layer::VarLayer{ReLU},  next_layer::VarLayer)
function encode_layer!(::StandardLP,
                       model::Model,
                       layer::Layer{ReLU},
                       z_current::Vector{Variable},
                       z_next::Vector{Variable},
                       δ::Vector{Bool})

    # The jth ReLU is forced to be active or inactive,
    # depending on the activation pattern given by δᵢ.
    # δᵢⱼ == true denotes ẑ >=0 (i.e. an *inactive* ReLU)

    ẑ = layer.weights * z_current + layer.bias
    for j in 1:length(layer.bias)
        if δ[j]
            @constraint(model, ẑ[j] >= 0.0)
            @constraint(model, z_next[j] == ẑ[j])
        else
            @constraint(model, ẑ[j] <= 0.0)
            @constraint(model, z_next[j] == 0.0)
        end
    end
end

function encode_layer!(SLP::SlackLP,
                       model::Model,
                       layer::Layer{ReLU},
                       z_current::Vector{Variable},
                       z_next::Vector{Variable},
                       δ::Vector{Bool})

    ẑ = layer.weights * z_current + layer.bias
    slack_vars = @variable(model, [1:length(layer.bias)])
    for j in 1:length(layer.bias)
        if δ[j]
            @constraint(model, z_next[j] == ẑ[j] + slack_vars[j])
            @constraint(model, ẑ[j] + slack_vars[j] >= 0.0)
        else
            @constraint(model, z_next[j] == slack_vars[j])
            @constraint(model, 0.0 >= ẑ[j] - slack_vars[j])
        end
    end
    push!(SLP.slack, slack_vars)
    return nothing
end

function encode_layer!(::LinearRelaxedLP,
                       model::Model,
                       layer::Layer{ReLU},
                       z_current::Vector{Variable},
                       z_next::Vector{Variable},
                       δ::Vector{Bool})

    ẑ = layer.weights * z_current + layer.bias
    for j in 1:length(layer.bias)
        if δ[j]
            @constraint(model, z_next[j] == ẑ[j])
        else
            @constraint(model, z_next[j] == 0.0)
        end
    end
end


function encode_layer!(::TriangularRelaxedLP,
                       model::Model,
                       layer::Layer{ReLU},
                       z_current::Vector{Variable},
                       z_next::Vector{Variable},
                       bounds::Hyperrectangle)

    ẑ = layer.weights * z_current + layer.bias
    ẑ_bound = approximate_affine_map(layer, bounds)
    l̂, û = low(ẑ_bound), high(ẑ_bound)
    for j in 1:length(layer.bias)
        if l̂[j] > 0.0
            @constraint(model, z_next[j] == ẑ[j])
        elseif û[j] < 0.0
            @constraint(model, z_next[j] == 0.0)
        else
            @constraints(model, begin
                                    z_next[j] >= ẑ[j]
                                    z_next[j] <= û[j] / (û[j] - l̂[j]) * (ẑ[j] - l̂[j])
                                    z_next[j] >= 0.0
                                end)
        end
    end
end

function encode_layer!(MIP::MixedIntegerLP,
                       model::Model,
                       layer::Layer{ReLU},
                       z_current::Vector{Variable},
                       z_next::Vector{Variable},
                       δ::Vector{Variable})
    m = MIP.m

    ẑ = layer.weights * z_current + layer.bias
    for j in 1:length(layer.bias)
        @constraints(model, begin
                                z_next[j] >= ẑ[j]
                                z_next[j] >= 0.0
                                z_next[j] <= ẑ[j] + m * δ[j]
                                z_next[j] <= m - m * δ[j]
                            end)
    end
end

function encode_layer!(::BoundedMixedIntegerLP,
                       model::Model,
                       layer::Layer{ReLU},
                       z_current::Vector{Variable},
                       z_next::Vector{Variable},
                       δ::Vector,
                       bounds::Hyperrectangle)

    ẑ = layer.weights * z_current + layer.bias
    ẑ_bound = approximate_affine_map(layer, bounds)
    l̂, û = low(ẑ_bound), high(ẑ_bound)

    for j in 1:length(layer.bias) # For evey node
        if l̂[j] >= 0.0
            @constraint(model, z_next[j] == ẑ[j])
        elseif û[j] <= 0.0
            @constraint(model, z_next[j] == 0.0)
        else
            @constraints(model, begin
                                    z_next[j] >= ẑ[j]
                                    z_next[j] >= 0.0
                                    z_next[j] <= û[j] * δ[j]
                                    z_next[j] <= ẑ[j] - l̂[j] * (1 - δ[j])
                                end)
        end
    end
end


#=
Add input/output constraints to model
=#
function add_complementary_set_constraint!(model::Model, output::HPolytope, neuron_vars::Vector{Variable})
    out_A, out_b = tosimplehrep(output)
    # Needs to take the complementary of output constraint
    n = length(out_b)
    if n == 1
        # Here the output constraint is a half space
        # So the complementary is just out_A * y .> out_b
        @constraint(model, -out_A * neuron_vars .<= -out_b)
    else
        LC = length(constraints_list(output))
        @assert LC == 1 "Quadratic constraints are not yet supported. Please make sure that the
        output constraint is a HalfSpace (an HPolytope with a single constraint). Got $LC constraints."
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

function add_complementary_set_constraint!(model::Model, output::Hyperrectangle, neuron_vars::Vector{Variable})
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
