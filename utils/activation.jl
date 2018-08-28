abstract type ActivationFunction end

struct GeneralAct <: ActivationFunction end
struct ReLU <: ActivationFunction end
struct Max <: ActivationFunction end

(f::GeneralAct)(x::Float64) = f(x)
(f::ReLU)(x::Float64) = max(x,0.0)



