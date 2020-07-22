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

# For the aux_* variables, we want unique names counting up from 1.
# E.g. aux_abs1. To this end, we store each type of aux variable in the
# model in a vector accessible as e.g. `m[:aux_abs]`. To create a new one,
# we count the length of the associated vector and add 1.
# I.e. if aux_abs1 exists in the model, this will return aux_abs2
function new_named_var(m::Model, base_symbol::Symbol)
    aux_vars = get!(m.obj_dict, base_symbol, VariableRef[])
    aux = @variable(m, base_name = string(base_symbol)*string(length(aux_vars)+1))
    push!(aux_vars, aux)
    aux
end


_model(a::Number) = nothing
_model(a::VariableRef) = a.model
_model(a::Array) = _model(first(a))
_model(a::GenericAffExpr) = isempty(a.terms) ? nothing : _model(first(keys(a.terms)))
_model(as...) = something(_model.(as)..., missing)

# These should only be hit if we're comparing 0 with 0 or if we somehow
# hit a branch that is numbers only (no variables or expressions).
# We don't have a way to get the model in that case. This is a problem since
# we do sometimes end up with (0, 0). It shouldn't happen in any other case.
symbolic_max(m::Missing, a, b) = (iszero(a) && iszero(b)) ? (return zero(promote_type(typeof(a), typeof(b)))) : ArgumentError("Cannot get model from ($a, $b)")


function symbolic_max(m::Model, a, b)
    aux = new_named_var(m, :aux_max)
    @constraint(m, aux >= a)
    @constraint(m, aux >= b)
    return aux
end
symbolic_max(a, b) = symbolic_max(_model(a, b), a, b)

function symbolic_max(m::Model, a...)
    aux = new_named_var(m, :aux_max)
    for z in a
        @constraint(m, aux >= z)
    end
    aux
end
symbolic_max(a...) = symbolic_max(_model(a...), a...)

function symbolic_abs(m::Model, v)
    aux = new_named_var(m, :aux_abs)
    @constraint(m, aux >= 0)
    @constraint(m, aux >= v)
    @constraint(m, aux >= -v)
    return aux
end
symbolic_abs(v) = symbolic_abs(_model(v), v)


function symbolic_infty_norm(m::Model, v::Array{<:GenericAffExpr})
    aux = new_named_var(m, :aux_inf)
    @constraint(m, aux >= 0)
    @constraint(m, aux .>= v)
    @constraint(m, aux .>= -v)
    return aux
end
symbolic_infty_norm(v::Array{<:GenericAffExpr}) = symbolic_infty_norm(_model(v), v)
