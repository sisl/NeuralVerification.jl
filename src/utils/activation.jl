abstract type ActivationFunction end

struct GeneralAct <: ActivationFunction end
struct ReLU <: ActivationFunction end
struct Max <: ActivationFunction end
struct Id <: ActivationFunction end

(f::GeneralAct)(x) = f(x)
(f::ReLU)(x) = max.(x,0)
(f::Max)(x) = max(maximum(x),0)
(f::Id)(x) = x
