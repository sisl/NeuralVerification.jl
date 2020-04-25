using .Flux

# Network -> Flux

activation(x) = GeneralAct(x)

activation(::typeof(identity)) = Id()
activation(::typeof(relu)) = ReLU()

layer(x) = error("Can't use $x as a layer.")

layer(d::Dense) = Layer(data(d.W), data(d.b), activation(d.σ))

network(c::Chain) = Network([layer.(c.layers)...])

Problem(c::Chain, input::AbstractPolytope, output::AbstractPolytope) =
  Problem(network(c), input, output)

# Flux -> Network

_flux(::ReLU) = relu
_flux(::Id) = identity
_flux(f::GeneralAct) = f.f

_flux(m::Layer) = Dense(m.weights, m.bias, _flux(m.activation))
_flux(m::Network) = Chain(_flux.(m.layers)...)

Flux.Chain(m::Network) = _flux(m)
