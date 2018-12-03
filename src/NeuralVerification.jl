module NeuralVerification

__precompile__(false)

using Reexport
# for Feasibility
@reexport using JuMP
@reexport using MathProgBase.SolverInterface
@reexport using GLPKMathProgInterface
@reexport using PicoSAT # needed for Planet
@reexport using SCS     # needed for Certify and Duality
# for Reachability
@reexport using LazySets
@reexport using Polyhedra
@reexport using CDDLib
#@reexport using LinearAlgebra

import LazySets: dim, HalfSpace # dim is necessary to avoid conflict with Polyhedra.dim, HalfSpace is not defined unless imported

abstract type Solver end # no longer needed

# NOTE: the first 3 can probably be unified in one file.
include("utils/activation.jl")
include("utils/network.jl")
include("utils/problem.jl")
include("utils/util.jl")
export
    Solver,
    Network,
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

solve(m::Model) = JuMP.solve(m) ## TODO find a place for this
export solve

# TODO: consider creating sub-modules for each of these.
include("optimization/utils/constraints.jl")
include("optimization/utils/objectives.jl")
include("optimization/utils/variables.jl")
include("optimization/reverify.jl")
include("optimization/convDual.jl")
include("optimization/duality.jl")
include("optimization/certify.jl")
include("optimization/iLP.jl")
include("optimization/mipVerify.jl")
export Reverify, ConvDual, Duality, Certify, ILP, MIPVerify

include("reachability/utils/reachability.jl")
include("reachability/exactReach.jl")
include("reachability/maxSens.jl")
include("reachability/ai2.jl")
export ExactReach, MaxSens, Ai2

include("satisfiability/bab.jl")
include("satisfiability/planet.jl")
include("satisfiability/sherlock.jl")
include("satisfiability/reluplex.jl")
export BAB, Planet, Sherlock, Reluplex

include("adversarial/reluVal.jl")
include("adversarial/FastLin.jl")
include("adversarial/FastLip.jl")
include("adversarial/dlv.jl")
export ReluVal, FastLin, FastLip, DLV

end