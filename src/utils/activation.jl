abstract type ActivationFunction end

struct GeneralAct <: ActivationFunction end
struct ReLU <: ActivationFunction end
struct Max <: ActivationFunction end
struct Id <: ActivationFunction end

(f::GeneralAct)(x) = f(x)
(f::ReLU)(x) = max.(x, zero(eltype(x)))  # these type stable definitions probably don't need to go in the paper as-is
(f::Max)(x) = max(maximum(x), zero(eltype(x)))
(f::Id)(x) = x
