module NeuralVerification

using JuMP

using GLPK, SCS # SCS only needed for Certify
using PicoSAT # needed for Planet
using LazySets, LazySets.Approximations
using Polyhedra, CDDLib

using LinearAlgebra
using Parameters
using Interpolations # only for PiecewiseLinear

import LazySets: dim, HalfSpace # necessary to avoid conflict with Polyhedra

using Requires

abstract type Solver end

# For optimization methods:
import JuMP.MOI.OPTIMAL, JuMP.MOI.INFEASIBLE
JuMP.Model(solver::Solver) = Model(solver.optimizer)
# define a `value` function that recurses so that value(vector) and
# value(VecOfVec) works cleanly. This is only so the code looks nice.
value(var::JuMP.AbstractJuMPScalar) = JuMP.value(var)
value(vars::Vector) = value.(vars)
value(val) = val


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
    # NOTE: not sure if exporting these is a good idea as far as namespace conflicts go:
    # ReLU,
    # Max,
    # Id,
    # Sigmoid,
    # Tanh,
    GeneralAct,
    PiecewiseLinear,
    Problem,
    Result,
    BasicResult,
    CounterExampleResult,
    AdversarialResult,
    ReachabilityResult,
    read_nnet,
    write_nnet,
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
include("optimization/bab.jl")
include("optimization/sherlock.jl")
include("optimization/reluplex.jl")
include("optimization/planet.jl")
export NSVerify, ConvDual, Duality, Certify, ILP, MIPVerify,
       BaB, Sherlock, Reluplex, Planet

include("reachability/utils/reachability.jl")
include("reachability/exactReach.jl")
include("reachability/maxSens.jl")
include("reachability/ai2.jl")
include("reachability/reluVal.jl")
include("reachability/neurify.jl")
include("reachability/fastLin.jl")
include("reachability/fastLip.jl")
include("reachability/dlv.jl")
export ExactReach, MaxSens, Ai2, Ai2h, Ai2z, Box,
       ReluVal, Neurify, FastLin, FastLip, DLV

const TOL = Ref(sqrt(eps()))
set_tolerance(x::Real) = (TOL[] = x)
export set_tolerance

end
