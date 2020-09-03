# This file is for different constraints

abstract type AbstractLinearProgram end
struct StandardLP            <: AbstractLinearProgram end
struct LinearRelaxedLP       <: AbstractLinearProgram end
struct TriangularRelaxedLP   <: AbstractLinearProgram end
struct BoundedMixedIntegerLP <: AbstractLinearProgram end
struct SlackLP               <: AbstractLinearProgram end
struct MixedIntegerLP        <: AbstractLinearProgram end

Base.Broadcast.broadcastable(LP::AbstractLinearProgram) = Ref(LP)

function init_model_vars(model, problem, encoding::AbstractLinearProgram; kwargs...)
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
    if !haskey(object_dictionary(m, :bounds))
        before_act = get!(object_dictionary(m, :before_act, true))
        m[:bounds] = get_bounds(prob, !before_act)
    end
end


# Any encoding passes through here first:
function encode_network!(model::Model, network::Network, encoding::AbstractLinearProgram)
    for (i, layer) in enumerate(network.layers)
        encode_layer!(encoding, model, layer, model_params(encoding, model, layer, i)...)
    end
end

function model_params(LP::BoundedMixedIntegerLP, m, layer, i)
    # let's assume we're post-activation if the parameter isn't set,
    # since that's the default for get_bounds
    if get(object_dictionary(m), :before_act, false)
        ẑ_bound = m[:bounds][i+1]
    else
        ẑ_bound = approximate_affine_map(layer, m[:bounds][i])
    end
    affine_map(layer, m[:z][i]), m[:z][i+1], m[:δ][i], low(ẑ_bound), high(ẑ_bound)
end

function model_params(LP::TriangularRelaxedLP, m, layer, i)
    # let's assume we're post-activation if the parameter isn't set,
    # since that's the default for get_bounds
    if get(object_dictionary(m), :before_act, false)
        ẑ_bound = m[:bounds][i+1]
    else
        ẑ_bound = approximate_affine_map(layer, m[:bounds][i])
    end
    affine_map(layer, m[:z][i]), m[:z][i+1], low(ẑ_bound), high(ẑ_bound)
end

model_params(LP::StandardLP,      m, layer, i) = (affine_map(layer, m[:z][i]), m[:z][i+1], m[:δ][i])
model_params(LP::LinearRelaxedLP, m, layer, i) = (affine_map(layer, m[:z][i]), m[:z][i+1], m[:δ][i])
model_params(LP::MixedIntegerLP,  m, layer, i) = (affine_map(layer, m[:z][i]), m[:z][i+1], m[:δ][i], -m[:M], m[:M])
model_params(LP::SlackLP,         m, layer, i) = (affine_map(layer, m[:z][i]), m[:z][i+1], m[:δ][i], m[:slack][i])

# For an Id Layer, any encoding type defaults to this:
function encode_layer!(::AbstractLinearProgram, model::Model, layer::Layer{Id}, ẑᵢ, zᵢ, args...)
    @constraint(model, zᵢ .== ẑᵢ)
    nothing
end

function encode_layer!(LP::AbstractLinearProgram, model::Model, layer::Layer{ReLU}, args...)
    encode_relu.(LP, model, args...)
    nothing
end

# TODO not needed I think. But test without first!
# SlackLP is slightly different, because we need to keep track of the slack variables
function encode_layer!(SLP::SlackLP, model::Model, layer::Layer{Id}, ẑᵢ, zᵢ, δᵢⱼ, sᵢ)
    @constraint(model, zᵢ .== ẑᵢ)
    # We need identity layer slack variables so that the algorithm doesn't
    # "get confused", but they are set to 0 because they're not relevant
    @constraint(model, sᵢ .== 0.0)
    return nothing
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
    elseif ûᵢⱼ <= 0.0
        @constraint(model, zᵢⱼ == 0.0)
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
    @constraint(model, zᵢⱼ == (δᵢⱼ ? ẑᵢⱼ : 0.0))
end

function encode_relu(::StandardLP, model, ẑᵢⱼ, zᵢⱼ, δᵢⱼ)
    if δᵢⱼ
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
function add_complementary_set_constraint!(m::Model, PC::PolytopeComplement, z::Vector{VariableRef})
    add_set_constraint!(m, PC.P, z)
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

function add_set_constraint!(m::Model, PC::PolytopeComplement, z::Vector{VariableRef})
    add_complementary_set_constraint!(m, PC.P, z)
    return nothing
end
