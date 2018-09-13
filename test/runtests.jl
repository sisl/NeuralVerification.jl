
# for Reverify
using JuMP
using MathProgBase.SolverInterface
using GLPKMathProgInterface

# for reachability solvers
using LazySets
using Polyhedra
using CDDLib

# TODO create include heirarchy in a NeuralVerification.jl file with exports etc.
include("../solver/solver.jl")
include("../src/utils/activation.jl")
include("../src/utils/problem.jl")
include("../src/utils/util.jl")

# TODO include all the solvers in the include heirarchy as well. Potentially as their own submodules
include("../src/feasibility/reverify.jl")
small_nnet = read_nnet("../examples/networks/small_nnet.txt")
A = Matrix{Float64}(2,1)
A[1,1] = 1
A[2,1] = -1

inputSet = HPolytope(A, [1.0,1.0])
outputSet = HPolytope(A,[1.0,1.0])
problem = Problem(small_nnet, inputSet, outputSet)
solver = Reverify(GLPKSolverMIP(), 1000.0)
solve(solver, problem)


include("../src/reachability/maxSens.jl")

inputSet = HPolytope(A, [1.0,1.0])
outputSet = HPolytope(A,[100.0,1.0])
resolution = 0.3
problem = Problem(small_nnet, inputSet, outputSet)
solver = MaxSens(resolution)
solve(solver, problem)

include("../src/reachability/exactReach.jl")

# inputSet = Constraints(eye(1),[1.0],[1.0],[-1.0])        => [1, -1]x <= [0, 2]
# outputSet = Constraints(zeros(1,1),[0],[100.0],[-1.0])   => [0, -0]x <= [100, 1] NOTE is this correct?
inputSet = HPolytope(A, [0.0, 2.0])
A2 = zeros(2, 1)
outputSet = HPolytope(A2, [100.0, 1.0])
problem = Problem(small_nnet, inputSet, outputSet)
solver = ExactReach()
solve(solver, problem)


include("../src/reachability/reluVal.jl")

solver = ReluVal(2)
inputSet = high_dim_interval([-1.0],[1.0])
outputSet = high_dim_interval([-1.0],[50.0])
problem = Problem(small_nnet, inputSet, outputSet)
solve(solver, problem)