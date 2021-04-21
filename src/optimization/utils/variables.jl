import JuMP: GenericAffExpr

init_vars(model::Model, net::Network, args...; kwargs...) = init_vars(model, net.layers, args...; kwargs...)
function init_vars(model::Model, layers::Vector{Layer}, name=nothing; binary=false, with_input = false)
    N = length(layers)
    vars = Vector{Vector{VariableRef}}(undef, N + with_input)

    if with_input
        n = size(layers[1].weights, 2)
        vars[1] = @variable(model, [1:n], binary = binary)
    end

    for i in 1:N
        n = n_nodes(layers[i])
        vars[i+with_input] = @variable(model, [1:n], binary = binary)
    end

    if !isnothing(name)
        for i in 1:length(vars), j in 1:length(vars[i])
            set_name(vars[i][j], string(name, i-with_input, '[', j, ']'))
        end
        model[name] = vars
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
