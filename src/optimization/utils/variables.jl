import JuMP: GenericAffExpr

init_neurons(model::Model, layers::Vector{Layer})     = init_variables(model, layers, :Cont, input_is_special = true)
init_deltas(model::Model, layers::Vector{Layer})      = init_variables(model, layers, :Bin,  input_is_special = true)
init_multipliers(model::Model, layers::Vector{Layer}) = init_variables(model, layers, :Cont)

# Allow ::Network input also (NOTE for legacy purposes mostly...)
init_neurons(m,     network::Network) = init_neurons(m, network.layers)
init_deltas(m,      network::Network) = init_deltas(m,  network.layers)
init_multipliers(m, network::Network) = init_multipliers(m, network.layers)

function init_variables(model::Model, layers::Vector{Layer}, vartype::Symbol; input_is_special::Bool = false)
    vars = Vector{Vector{Variable}}(length(layers) )
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