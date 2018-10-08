import JuMP: GenericAffExpr

function init_neurons(model::Model, network::Network)
    layers = network.layers
    neurons = Vector{Vector{Variable}}(length(layers) + 1)

    input_layer_n = size(first(layers).weights, 2)
    all_layers_n  = [length(l.bias) for l in layers]
    insert!(all_layers_n, 1, input_layer_n)

    for (i, n) in enumerate(all_layers_n)
        neurons[i] = @variable(model, [1:n])
    end

    return neurons
end

function init_deltas(model::Model, network::Network)
    layers = network.layers
    deltas = Vector{Vector{Variable}}(length(layers) + 1)

    input_layer_n = size(first(layers).weights, 2)
    all_layers_n  = [length(l.bias) for l in layers]
    insert!(all_layers_n, 1, input_layer_n)

    for (i, n) in enumerate(all_layers_n)
        deltas[i] = @variable(model, [1:n], Bin)
    end

    return deltas
end

# Lagrangian Multipliers
function init_multipliers(model::Model, network::Network)
    layers = network.layers
    λ = Vector{Vector{Variable}}(length(layers))

    all_layers_n = map(l->length(l.bias), layers)

    for (i, n) in enumerate(all_layers_n)
        λ[i] = @variable(model, [1:n])
    end

    return λ
end

function symbolic_max(m::Model, a, b)
    aux = @variable(m)
    @constraint(m, aux >= a)
    @constraint(m, aux >= b)
    return aux
end
symbolic_max(a::Variable, b::Variable)                                         = symbolic_max(a.m, a, b)
symbolic_max(a::JuMP.GenericAffExpr, b::JuMP.GenericAffExpr)                   = symbolic_max(first(a.vars).m, a, b)
symbolic_max(a::Array{<:JuMP.GenericAffExpr}, b::Array{<:JuMP.GenericAffExpr}) = symbolic_max.(first(first(a).vars).m, a, b)


# NOTE renamed to symbolic_abs to avoid type piracy
function symbolic_abs(m::Model, v)
    aux = @variable(m) #get an anonymous variable
    @constraint(m, aux >= 0)
    @constraint(m, aux >= v)
    @constraint(m, aux >= -v)
    return aux
end
symbolic_abs(v::Variable)                     = symbolic_abs(v.m, v)
symbolic_abs(v::JuMP.GenericAffExpr)          = symbolic_abs(first(v.vars).m, v)
symbolic_abs(v::Array{<:JuMP.GenericAffExpr}) = symbolic_abs.(first(first(v).vars).m, v)