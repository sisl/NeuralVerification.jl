
using NeuralVerification
using Base.Test

at = @__DIR__

small_nnet = read_nnet("$at/../examples/networks/small_nnet.txt")
A = Matrix{Float64}(undef, 2,1)
A[1] = 1
A[2] = -1

# TODO can we unify all of the reachability test conditions?
### reverify
inputSet  = HPolytope(A, [1.0,1.0])
outputSet = HPolytope(A, [1.0,1.0])
problem_reverify = Problem(small_nnet, inputSet, outputSet)
solver_reverify = Reverify(GLPKSolverMIP(), 1000.0)

### maxSens
inputSet  = HPolytope(A, [1.0,1.0])
outputSet = HPolytope(A, [100.0,1.0])
resolution = 0.3
problem_maxSens = Problem(small_nnet, inputSet, outputSet)
solver_maxSens = MaxSens(resolution)

### exactReach
# inputSet = Constraints(eye(1),[1.0],[1.0],[-1.0])        => [1, -1]x <= [0, 2]
# outputSet = Constraints(zeros(1,1),[0],[100.0],[-1.0])   => [0, -0]x <= [100, 1] NOTE is this correct?
A2 = zeros(2, 1)
inputSet  = HPolytope(A,  [0.0, 2.0])
outputSet = HPolytope(A2, [100.0, 1.0])
problem_exactReach = Problem(small_nnet, inputSet, outputSet)
solver_exactReach = ExactReach()

#reluVal
inputSet  = Hyperrectangle(low = [-1.0], high = [1.0])
outputSet = Hyperrectangle(low = [-1.0], high = [50.0])
problem_reluVal = Problem(small_nnet, inputSet, outputSet)
solver_reluVal = ReluVal(2)


@test solve(solver_reverify,   problem_reverify).status == :True  # True means infeasible (NOTE: is that intuitive?)
@test solve(solver_maxSens,    problem_maxSens)         == :True
@test solve(solver_exactReach, problem_exactReach)      == :True
@test solve(solver_reluVal,    problem_reluVal).status  == :False