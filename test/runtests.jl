
using NeuralVerification

# TODO can we unify all of the reachability test conditions?

at = @__DIR__

small_nnet = read_nnet("$at/../examples/networks/small_nnet.txt")
A = Matrix{Float64}(2,1)
A[1] = 1
A[2] = -1

#reverify

inputSet = HPolytope(A, [1.0,1.0])
outputSet = HPolytope(A,[1.0,1.0])
problem = Problem(small_nnet, inputSet, outputSet)
solver = Reverify(GLPKSolverMIP(), 1000.0)
@test solve(solver, problem).status == :True  # True means infeasible (NOTE: is that intuitive?)

#maxSens

inputSet = HPolytope(A, [1.0,1.0])
outputSet = HPolytope(A,[100.0,1.0])
resolution = 0.3
problem = Problem(small_nnet, inputSet, outputSet)
solver = MaxSens(resolution)
@test solve(solver, problem) == true

#exactReach

# inputSet = Constraints(eye(1),[1.0],[1.0],[-1.0])        => [1, -1]x <= [0, 2]
# outputSet = Constraints(zeros(1,1),[0],[100.0],[-1.0])   => [0, -0]x <= [100, 1] NOTE is this correct?
inputSet = HPolytope(A, [0.0, 2.0])
A2 = zeros(2, 1)
outputSet = HPolytope(A2, [100.0, 1.0])
problem = Problem(small_nnet, inputSet, outputSet)
solver = ExactReach()
@test solve(solver, problem) == true


#reluVal

solver = ReluVal(2)
inputSet = Hyperrectangle(low = [-1.0], high = [1.0])
outputSet = Hyperrectangle(low = [-1.0], high = [50.0])
problem = Problem(small_nnet, inputSet, outputSet)
solve(solver, problem)