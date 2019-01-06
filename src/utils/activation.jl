abstract type ActivationFunction end

"""
    ReLU <: ActivationFunction

    (ReLU())(x) -> max.(x, 0)
"""
struct ReLU <: ActivationFunction end

"""
    Max <: ActivationFunction

    (Max())(x) -> max(maximum(x), 0)
"""
struct Max <: ActivationFunction end

"""
    Id <: ActivationFunction
Identity operator

    (Id())(x) -> x
"""
struct Id <: ActivationFunction end

"""
    GeneralAct <: ActivationFunction
Wrapper type for a general activation function.

### Usage
```julia
act = GeneralAct(tanh)

act(0) == tanh(0)           # true
act(10.0) == tanh(10.0)     # true
```
```julia
act = GeneralAct(x->tanh.(x))

julia> act(-2:2)
5-element Array{Float64,1}:
 -0.9640275800758169
 -0.7615941559557649
  0.0
  0.7615941559557649
  0.9640275800758169
```
"""
struct GeneralAct <: ActivationFunction
    f::Function
end

#=
TODO: consider writing our own interpolation scheme to avoid a dependency for this one thing.
Should only require a handful of functions.
Also NOTE: inherently not type stable unless parameterized.
=#
"""
    PiecewiseLinear <: ActivationFunction
Activation function that uses linear interpolation between supplied `knots`.
An extrapolation condition can be set for values outside the set of knots. Default is `Linear`.

    PiecewiseLinear(knots_x, knots_y, [extrapolation = Line()])

### Usage
```julia
kx = [0.0, 1.2, 1.7, 3.1]
ky = [0.0, 0.5, 1.0, 1.5]
act = PiecewiseLinear(kx, ky)

act(first(kx)) == first(ky) == 0.0
act(last(kx))  == last(ky)  == 1.5

act(1.0)    # 0.4166666666666667
act(-102)   # -42.5
```
```julia
act = PiecewiseLinear(kx, ky, Flat())

act(-102)   # 0.0
act(Inf)    # 1.5
```

### Extrapolations
- Flat()
- Line()
- constant (supply a number as the argument)
- Throw() (throws bounds error)

`PiecewiseLinear` uses [Interpolations.jl](http://juliamath.github.io/Interpolations.jl/latest/).
"""
struct PiecewiseLinear <: ActivationFunction
    f::Interpolations.Extrapolation
end

# default extrapolation is Line(). Can also do Flat() or supply a constant,
function PiecewiseLinear(knots_x::AbstractVector,
                         knots_y::AbstractVector,
                         extrapolation = Interpolations.Line())
    PiecewiseLinear(LinearInterpolation(knots_x, knots_y, extrapolation_bc = extrapolation))
end

# the type stable definitions probably don't need to go in the paper as-is
(f::ReLU)(x) = max.(x, zero(eltype(x)))
(f::Max)(x) = max(maximum(x), zero(eltype(x)))
(f::Id)(x) = x
(G::GeneralAct)(x) = G.f(x)
(PL::PiecewiseLinear)(x) = PL.f(x)
