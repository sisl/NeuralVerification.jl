abstract type ActivationFunction end

struct ReLU <: ActivationFunction end
struct Max <: ActivationFunction end
struct Id <: ActivationFunction end

(f::ReLU)(x) = max.(x, zero(eltype(x)))  # these type stable definitions probably don't need to go in the paper as-is
(f::Max)(x) = max(maximum(x), zero(eltype(x)))
(f::Id)(x) = x

struct GeneralAct <: ActivationFunction
    f::Function
end
(G::GeneralAct)(x) = G.f(x)

#=
While brief, this isn't exactly pretty.
TODO: consider writing our own interpolation scheme. Should only require three functions or so.
Also NOTE: inherently not type stable unless parameterized.
=#
struct PiecewiseLinear <: ActivationFunction
    etp::Interpolations.Extrapolation
end

# default extrapolation is Line(). Can also do Flat() or constant,
function PiecewiseLinear(knots_x::AbstractVector,
                         knots_y::AbstractVector,
                         extrapolation = Interpolations.Line())
    PiecewiseLinear(extrapolate(interpolate(tuple(knots_x), knots_y, Gridded(Linear())), extrapolation))
end

(f::PiecewiseLinear)(x) = f.etp(x)


# # rather than rely on Interpolations.jl, can do like:
# struct PiecewiseLinear <: ActivationFunction
#     x::Vector{Float64}
#     y::Vector{Float64}
# end

# function (f::PiecewiseLinear)(x)
#     ind = findfirst(knot -> knot>x, f.x)

#     if ind == nothing # x is bigger than all knots
#     elseif ind == 1 # all knots are bigger than x
#     else

#     end
# end