import JuMP: GenericAffExpr

init_neurons(model::Model, layers::Vector{Layer})     = init_variables(model, layers, include_input = true)
init_deltas(model::Model, layers::Vector{Layer})      = init_variables(model, layers, binary = true, name = "δ")
init_multipliers(model::Model, layers::Vector{Layer}, name::String) = init_variables(model, layers, name = name)
# Allow ::Network input also (NOTE for legacy purposes mostly...)
init_neurons(m,     network::Network) = init_neurons(m, network.layers)
init_deltas(m,      network::Network) = init_deltas(m,  network.layers)
init_multipliers(m, network::Network, name::String) = init_multipliers(m, network.layers, name)

function init_variables(model::Model, layers::Vector{Layer}; binary = false, include_input = false, name = "z")
    # TODO: only neurons get offset array
    vars = Vector{Vector{VariableRef}}(undef, length(layers))
    all_layers_n = n_nodes.(layers)

    if include_input
        # input to the first layer also gets variables
        # essentially an input constraint
        input_layer_n = size(first(layers).weights, 2)
        prepend!(all_layers_n, input_layer_n)
        push!(vars, Vector{VariableRef}())        # expand vars by one to account
    end

    for (i, n) in enumerate(all_layers_n)
        vars[i] = @variable(model, [1:n], binary = binary, base_name = string(name, "$i"))
    end
    return vars
end

function symbolic_max(m::Model, a, b)
    aux = @variable(m, base_name = "aux_max")
    @constraint(m, aux >= a)
    @constraint(m, aux >= b)
    return aux
end
symbolic_max(a::VariableRef, b::VariableRef)                         = symbolic_max(a.model, a, b)
symbolic_max(a::GenericAffExpr, b::GenericAffExpr)                   = symbolic_max(first(first(a.terms)).model, a, b)
symbolic_max(a::Array{<:GenericAffExpr}, b::Array{<:GenericAffExpr}) = symbolic_max.(a, b)

function symbolic_abs(m::Model, v)
    aux = @variable(m, base_name = "aux_abs") #get an anonymous variable
    @constraint(m, aux >= 0)
    @constraint(m, aux >= v)
    @constraint(m, aux >= -v)
    return aux
end
symbolic_abs(v::VariableRef)             = symbolic_abs(v.m, v)
symbolic_abs(v::GenericAffExpr)          = symbolic_abs(first(first(v.terms)).model, v)
symbolic_abs(v::Array{<:GenericAffExpr}) = symbolic_abs.(v)

function symbolic_infty_norm(m::Model, v::Array{<:GenericAffExpr})
    aux = @variable(m, base_name = "aux_inf")
    @constraint(m, aux >= 0)
    @constraint(m, aux .>= v)
    @constraint(m, aux .>= -v)
    return aux
end
# # in general, default to symbolic_abs behavior:
# symbolic_infty_norm(v) = symbolic_abs(v)
# only Array{<:GenericAffExpr} is needed
symbolic_infty_norm(v::Array{<:GenericAffExpr}) = symbolic_infty_norm(first(first(first(v).terms)).model, v)
