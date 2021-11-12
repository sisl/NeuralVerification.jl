# This file is for different constraints

abstract type AbstractLinearProgram end
struct StandardLP            <: AbstractLinearProgram end
struct LinearRelaxedLP       <: AbstractLinearProgram end
struct TriangularRelaxedLP   <: AbstractLinearProgram end
struct BoundedMixedIntegerLP <: AbstractLinearProgram end
struct SlackLP               <: AbstractLinearProgram end
struct MixedIntegerLP        <: AbstractLinearProgram end

Base.Broadcast.broadcastable(LP::AbstractLinearProgram) = Ref(LP)

function init_model_vars(model::Model,
                         problem::Problem,
                         encoding::AbstractLinearProgram
                         ;
                         kwargs...)
    model[:network] = problem.network
    model[:input]   = problem.input
    model[:output]  = problem.output

    init_vars(model, problem.network, :z, with_input=true)
    for (k, v) in kwargs
        model[k] = v
    end
    _init_unique(model, problem, encoding)
    model
end

_init_unique(m::Model, prob::Problem, encoding::AbstractLinearProgram) = nothing
_init_unique(m::Model, prob::Problem, encoding::SlackLP) = init_vars(m, prob.network, :slack)
_init_unique(m::Model, prob::Problem, encoding::MixedIntegerLP) = init_vars(m, prob.network, :δ, binary=true)
_init_unique(m::Model, prob::Problem, encoding::TriangularRelaxedLP) = _insert_bounds(m, prob, encoding)
function _init_unique(m::Model, prob::Problem, encoding::BoundedMixedIntegerLP)
    init_vars(m, prob.network, :δ, binary=true)
    _insert_bounds(m, prob, encoding)
end

function _insert_bounds(m::Model, prob::Problem, encoding::Union{TriangularRelaxedLP, BoundedMixedIntegerLP})
    if !haskey(object_dictionary(m), :bounds)
        before_act = get!(object_dictionary(m), :before_act, true)
        m[:bounds] = get_bounds(prob, !before_act)
    end
end

model_params(LP::BoundedMixedIntegerLP, m::Model, i::Integer) = (_ẑᵢ₊₁(m, i), m[:z][i+1], m[:δ][i], _ẑᵢ₊₁_bound(m, i)...)
model_params(LP::TriangularRelaxedLP,   m::Model, i::Integer) = (_ẑᵢ₊₁(m, i), m[:z][i+1], _ẑᵢ₊₁_bound(m, i)...)
model_params(LP::StandardLP,            m::Model, i::Integer) = (_ẑᵢ₊₁(m, i), m[:z][i+1], m[:δ][i])
model_params(LP::LinearRelaxedLP,       m::Model, i::Integer) = (_ẑᵢ₊₁(m, i), m[:z][i+1], m[:δ][i])
model_params(LP::MixedIntegerLP,        m::Model, i::Integer) = (_ẑᵢ₊₁(m, i), m[:z][i+1], m[:δ][i], -m[:M], m[:M])
model_params(LP::SlackLP,               m::Model, i::Integer) = (_ẑᵢ₊₁(m, i), m[:z][i+1], m[:δ][i], m[:slack][i])

# helper function to get the bounds relevant for the i-th layer constraint.
# If before_act=true, then the model is storing the pre-activation
# bounds, and we return the i+1th bound (goes with ẑᵢ₊₁). Otherwise
# NOTE we forward propagate (affine map) the ith bound. We assume the model
# stores post-activation bounds since this is the current default for get_bounds
function _ẑᵢ₊₁_bound(m, i)
    layer = m[:network].layers[i]
    # let's assume we're post-activation if the parameter isn't set,
    # since that's the default for get_bounds
    if get(object_dictionary(m), :before_act, false)
        ẑ_bound = m[:bounds][i+1]
    else
        ẑ_bound = approximate_affine_map(layer, m[:bounds][i])
    end
    low(ẑ_bound), high(ẑ_bound)
end
_ẑᵢ₊₁(m, i) = affine_map(m[:network].layers[i], m[:z][i])


# Any encoding passes through here first:
function encode_network!(model::Model, network::Network, encoding::AbstractLinearProgram)
    object_dictionary(model)[:network] =  network # stores model[:network]
    for (i, layer) in enumerate(network.layers)
        encode_layer!(encoding, model, layer, model_params(encoding, model, i)...)
    end
end

function encode_layer!(LP::AbstractLinearProgram, model::Model, i::Integer)
    L = model[:network].layers[i]
    encode_layer!(LP, model, L, model_params(LP, model, i)...)
    nothing
end

# For an Id Layer, any encoding type defaults to this:
function encode_layer!(::AbstractLinearProgram, model::Model, layer::Layer{Id}, ẑᵢ, zᵢ, args...)
    @constraint(model, zᵢ .== ẑᵢ)
    nothing
end

# All ReLU layers pass through this
function encode_layer!(LP::AbstractLinearProgram, model::Model, layer::Layer{ReLU}, args...)
    encode_relu.(LP, model, args...)
    nothing
end

# SlackLP is slightly different, because we need to keep track of the slack variables
function encode_layer!(SLP::SlackLP, model::Model, layer::Layer{Id}, ẑᵢ, zᵢ, δᵢⱼ, sᵢ)
    @constraint(model, zᵢ .== ẑᵢ)
    # We need identity layer slack variables so that the algorithm doesn't
    # "get confused", but they are set to 0 because they're not relevant
    @constraint(model, sᵢ .== 0.0)
    return nothing
end

# need to fix δᵢⱼ for BoundedMixedIntegerLP and possibly other types 
function encode_layer!(::BoundedMixedIntegerLP, model::Model, layer::Layer{Id}, ẑᵢ, zᵢ, δᵢ, args...)
    @constraint(model, zᵢ .== ẑᵢ)
    @constraint(model, δᵢ .== 1)
    return nothing
end

function encode_ij(LP, model, i, j)
    # where is this function used? Needs documentation.
    L = model[:network].layers[i]
    params = model_params(LP, model, i)
    if L.activation isa Id
        ẑᵢⱼ = _ẑᵢ₊₁(model, i)[j]
        zᵢⱼ = model[:z][i+1][j]
        @constraint(model, ẑᵢⱼ == zᵢⱼ)
    elseif L.activation isa ReLU
        # ridiculous hack so it works with scalar params (like :M)
        _getindex_hack(x, j) = x isa Number ? x : x[j]
        encode_relu(LP, model, _getindex_hack.(params, j)...)
    end
end



function encode_relu(::SlackLP, model, ẑᵢⱼ, zᵢⱼ, δᵢⱼ, sᵢⱼ)
    if δᵢⱼ
        @constraint(model, zᵢⱼ == ẑᵢⱼ + sᵢⱼ)
        @constraint(model, ẑᵢⱼ + sᵢⱼ >= 0.0)
    else
        @constraint(model, zᵢⱼ == sᵢⱼ)
        @constraint(model, ẑᵢⱼ <= sᵢⱼ)
    end
end

function encode_relu(::BoundedMixedIntegerLP, model, ẑᵢⱼ, zᵢⱼ, δᵢⱼ, l̂ᵢⱼ, ûᵢⱼ)
    if l̂ᵢⱼ >= 0.0
        @constraint(model, zᵢⱼ == ẑᵢⱼ)
        @constraint(model, δᵢⱼ == 1)
    elseif ûᵢⱼ <= 0.0
        @constraint(model, zᵢⱼ == 0.0)
        @constraint(model, δᵢⱼ == 0)
    else
        @constraints(model, begin
                                zᵢⱼ >= 0.0
                                zᵢⱼ >= ẑᵢⱼ
                                zᵢⱼ <= ûᵢⱼ * δᵢⱼ
                                zᵢⱼ <= ẑᵢⱼ - l̂ᵢⱼ * (1 - δᵢⱼ)
                            end)
    end
end

function encode_relu(::MixedIntegerLP, args...)
    encode_relu(BoundedMixedIntegerLP(), args...)
end

function encode_relu(::TriangularRelaxedLP, model, ẑᵢⱼ, zᵢⱼ, l̂ᵢⱼ, ûᵢⱼ)
    if l̂ᵢⱼ > 0.0
        @constraint(model, zᵢⱼ == ẑᵢⱼ)
    elseif ûᵢⱼ < 0.0
        @constraint(model, zᵢⱼ == 0.0)
    else
        @constraints(model, begin
                                zᵢⱼ >= 0.0
                                zᵢⱼ >= ẑᵢⱼ
                                zᵢⱼ <= (ẑᵢⱼ - l̂ᵢⱼ) * ûᵢⱼ / (ûᵢⱼ - l̂ᵢⱼ)
                            end)
    end
end

function encode_relu(::LinearRelaxedLP, model, ẑᵢⱼ, zᵢⱼ, δᵢⱼ)
    @constraint(model, zᵢⱼ == (δᵢⱼ ? ẑᵢⱼ : 0.0)) # in LinearRelaxedLP δᵢⱼ is a constant not a variable
end

function encode_relu(::StandardLP, model, ẑᵢⱼ, zᵢⱼ, δᵢⱼ)
    if δᵢⱼ # in StandardLP δᵢⱼ is a constant, not a variable
        @constraint(model, ẑᵢⱼ >= 0.0)
        @constraint(model, zᵢⱼ == ẑᵢⱼ)
    else
        @constraint(model, ẑᵢⱼ <= 0.0)
        @constraint(model, zᵢⱼ == 0.0)
    end
end








#=
Add input/output constraints to model
=#
function add_complementary_set_constraint!(model::Model, output::HPolytope, z::Vector{VariableRef})
    out_A, out_b = tosimplehrep(output)
    # Needs to take the complementary of output constraint
    n = length(constraints_list(output))
    if n == 1
        # Here the output constraint is a half space
        halfspace = first(constraints_list(output))
        add_complementary_set_constraint!(model, halfspace, z)
    else
        error("Non-convex constraints are not supported. Please make sure that the
            output set is a HalfSpace (or an HPolytope with a single constraint) so that the
            complement of the output is convex. Got $n constraints.")
    end
    return nothing
end

function add_complementary_set_constraint!(m::Model, H::HalfSpace, z::Vector{VariableRef})
    a, b = tosimplehrep(H)
    @constraint(m, a * z .>= b)
    return nothing
end
function add_complementary_set_constraint!(m::Model, PC::Complement, z::Vector{VariableRef})
    add_set_constraint!(m, PC.X, z)
    return nothing
end

function add_set_constraint!(m::Model, set::Union{HPolytope, HalfSpace}, z::Vector{VariableRef})
    A, b = tosimplehrep(set)
    @constraint(m, A * z .<= b)
    return nothing
end

function add_set_constraint!(m::Model, set::Hyperrectangle, z::Vector{VariableRef})
    @constraint(m, z .<= high(set))
    @constraint(m, z .>= low(set))
    return nothing
end

function add_set_constraint!(m::Model, PC::Complement, z::Vector{VariableRef})
    add_complementary_set_constraint!(m, PC.X, z)
    return nothing
end
