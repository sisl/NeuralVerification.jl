module NeuralVerification

using Reexport
# for Feasibility
@reexport using JuMP
@reexport using MathProgBase.SolverInterface
@reexport using GLPKMathProgInterface
# for Reachability
@reexport using LazySets
import LazySets.dim # necessary to avoid conflict with Polyhedra.dim
@reexport using Polyhedra
@reexport using CDDLib

abstract type Solver end

# NOTE: the first 3 can be unified in one file.
include("utils/activation.jl")
include("utils/network.jl")
include("utils/problem.jl")
include("utils/util.jl")

# TODO: consider creating sub-modules for each of these.
include("optimization/utils/constraints.jl")
include("optimization/utils/objectives.jl")
include("optimization/utils/variables.jl")
include("optimization/reverify.jl")
include("optimization/convDual.jl")
include("optimization/duality.jl")

include("reachability/utils/reachability.jl")
include("reachability/exactReach.jl")
include("reachability/maxSens.jl")
include("reachability/reluVal.jl")
include("reachability/ai2.jl")


export
    Solver,
    Network,
    Problem,
    Result,
    #potentially belong in sub-modules:
    Reverify,
    MaxSens,
    ExactReach,
    ReluVal,
    Ai2,
    ConvDual,
    Duality,
    # necessary functions:
    read_nnet,
    solve,
    forward_network,
    check_inclusion
end