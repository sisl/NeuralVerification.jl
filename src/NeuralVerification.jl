module NeuralVerification

using JuMP
using PicoSAT # needed for Planet
using GLPK, SCS # SCS only needed for Certify

using LazySets, LazySets.Approximations
using Polyhedra, CDDLib

using LinearAlgebra
using Parameters
using Interpolations # only for PiecewiseLinear

import LazySets: dim, HalfSpace # necessary to avoid conflict with Polyhedra

using Requires

# abstract type Solver end # no longer needed

# For optimization methods:
import JuMP.MOI.OPTIMAL, JuMP.MOI.INFEASIBLE
JuMP.Model(solver) = Model(with_optimizer(solver.optimizer))
JuMP.value(vars::Vector{VariableRef}) = value.(vars)

include("utils/activation.jl")
include("utils/network.jl")
include("utils/problem.jl")
include("utils/util.jl")

function __init__()
  @require Flux="587475ba-b771-5e3f-ad9e-33799f191a9c" include("utils/flux.jl")
end

export
    Solver,
    Network,
    AbstractActivation,
    PolytopeComplement,
    complement,
    # NOTE: not sure if exporting these is a good idea as far as namespace conflicts go:
    # ReLU,
    # Max,
    # Id,
    GeneralAct,
    PiecewiseLinear,
    Problem,
    Result,
    BasicResult,
    CounterExampleResult,
    AdversarialResult,
    ReachabilityResult,
    read_nnet,
    solve,
    forward_network,
    check_inclusion

solve(m::Model; kwargs...) = JuMP.solve(m; kwargs...)
export solve

# TODO: consider creating sub-modules for each of these.
include("optimization/utils/constraints.jl")
include("optimization/utils/objectives.jl")
include("optimization/utils/variables.jl")
include("optimization/nsVerify.jl")
include("optimization/convDual.jl")
include("optimization/duality.jl")
include("optimization/certify.jl")
include("optimization/iLP.jl")
include("optimization/mipVerify.jl")
export NSVerify, ConvDual, Duality, Certify, ILP, MIPVerify

include("reachability/utils/reachability.jl")
include("reachability/exactReach.jl")
include("reachability/maxSens.jl")
include("reachability/ai2.jl")
export ExactReach, MaxSens, Ai2

include("satisfiability/bab.jl")
include("satisfiability/planet.jl")
include("satisfiability/sherlock.jl")
include("satisfiability/reluplex.jl")
export BaB, Planet, Sherlock, Reluplex

include("adversarial/reluVal.jl")
include("adversarial/fastLin.jl")
include("adversarial/fastLip.jl")
include("adversarial/dlv.jl")
export ReluVal, FastLin, FastLip, DLV

end
