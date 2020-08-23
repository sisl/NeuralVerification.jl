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
using Gurobi

abstract type Solver end
const GRB_ENV = Gurobi.Env()

# For optimization methods:
import JuMP.MOI.OPTIMAL, JuMP.MOI.INFEASIBLE
function model_creator(solver)
    if (solver.optimizer == Gurobi.Optimizer)
        println("Creating Gurobi model")
        #return JuMP.direct_model(Gurobi.Optimizer(OutputFlag=0))
        return JuMP.Model(with_optimizer(solver.optimizer, OutputFlag=0))
    else
        println("Creating optimizer not Gurobi")
        return JuMP.Model(with_optimizer(solver.optimizer))
    end
end
JuMP.Model(solver::Solver) = model_creator(solver)


JuMP.value(vars::Vector{VariableRef}) = value.(vars)

include("utils/activation.jl")
include("utils/network.jl")
include("utils/problem.jl")
include("utils/util.jl")
include("utils/testing_utils.jl")


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
export NSVerify, ConvDual, Duality, Certify, ILP, MIPVerify

include("reachability/utils/reachability.jl")
include("reachability/exactReach.jl")
include("reachability/maxSens.jl")
include("reachability/ai2.jl")
export ExactReach, MaxSens, Ai2, Ai2h, Ai2z, Box

include("satisfiability/bab.jl")
include("satisfiability/sherlock.jl")
include("satisfiability/reluplex.jl")
export BaB, Sherlock, Reluplex

include("satisfiability/planet.jl")
export Planet

include("adversarial/reluVal.jl")
include("adversarial/neurify.jl")
include("adversarial/fastLin.jl")
include("adversarial/fastLip.jl")
include("adversarial/dlv.jl")
export ReluVal, Neurify, FastLin, FastLip, DLV

const TOL = Ref(sqrt(eps()))
set_tolerance(x::Real) = (TOL[] = x)
export set_tolerance

end
