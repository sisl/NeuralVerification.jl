import JuMP: GenericAffExpr

init_neurons(model::Model, network::Network) = init_variables(model, network, :Cont)
init_deltas(model::Model, network::Network)  = init_variables(model, network, :Bin)

function init_variables(model::Model, network::Network, vartype::Symbol)
    layers = network.layers
    vars = Vector{Vector{Variable}}(length(layers) + 1)

    input_layer_n = size(first(layers).weights, 2)
    all_layers_n  = n_nodes.(layers)
    prepend!(all_layers_n, input_layer_n)

    for (i, n) in enumerate(all_layers_n)
        vars[i] = @variable(model, [1:n], category = vartype)
    end
    return vars
end

# Lagrangian Multipliers
function init_multipliers(model::Model, network::Network)
    layers = network.layers
    λ = Vector{Vector{Variable}}(length(layers))

    all_layers_n = n_nodes.(layers)

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
symbolic_max(a::Variable, b::Variable)                               = symbolic_max(a.m, a, b)
symbolic_max(a::GenericAffExpr, b::GenericAffExpr)                   = symbolic_max(first(a.vars).m, a, b)
symbolic_max(a::Array{<:GenericAffExpr}, b::Array{<:GenericAffExpr}) = symbolic_max.(first(first(a).vars).m, a, b)

function symbolic_abs(m::Model, v)
    aux = @variable(m) #get an anonymous variable
    @constraint(m, aux >= 0)
    @constraint(m, aux >= v)
    @constraint(m, aux >= -v)
    return aux
end
symbolic_abs(v::Variable)                = symbolic_abs(v.m, v)
symbolic_abs(v::GenericAffExpr)          = symbolic_abs(first(v.vars).m, v)
symbolic_abs(v::Array{<:GenericAffExpr}) = symbolic_abs.(first(first(v).vars).m, v)