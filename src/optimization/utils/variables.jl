import JuMP: GenericAffExpr

init_neurons(model::Model, network::Network)     = init_variables(model, network, :Cont, true)
init_deltas(model::Model, network::Network)      = init_variables(model, network, :Bin, true)
init_multipliers(model::Model, network::Network) = init_variables(model, network, :Cont, false)

function init_variables(model::Model, network::Network, vartype::Symbol, input_is_special::Bool)
    layers = network.layers
    vars = Depth2Vec{Variable}(length(layers) )
    all_layers_n  = n_nodes.(layers)

    if input_is_special
        input_layer_n = size(first(layers).weights, 2)
        prepend!(all_layers_n, input_layer_n)  # input layer gets special treatment
        push!(vars, Vector{Variable}())        # expand vars by one also
    end

    for (i, n) in enumerate(all_layers_n)
        vars[i] = @variable(model, [1:n], category = vartype)
    end
    return vars
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