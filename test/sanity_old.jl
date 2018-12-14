using NeuralVerification
using Test

macro no_error(ex)
    quote
        try $(esc(ex))
            true
        catch
            false
        end
    end
end


at = @__DIR__

small_nnet = read_nnet("$at/../examples/networks/small_nnet.txt")
A = Matrix{Float64}(undef, 2,1)
A[1:2] = [1, -1]

# TODO can we unify all of the reachability test conditions?
### reverify
inputSet  = HPolytope(A, [1.0, 1.0])
outputSet = HPolytope(A, [1.0, 1.0])
problem_reverify = Problem(small_nnet, inputSet, outputSet)
solver_reverify = NSVerify(GLPKSolverMIP(), 1000.0)

### maxSens
inputSet  = HPolytope(A, [1.0, 1.0])
outputSet = HPolytope(A, [100.0, 1.0])
problem_maxSens = Problem(small_nnet, inputSet, outputSet)
solver_maxSens = MaxSens(0.3)

### exactReach
inputSet  = HPolytope(A,           [0.0, 2.0])
outputSet = HPolytope(zeros(2, 1), [100.0, 1.0])
problem_exactReach = Problem(small_nnet, inputSet, outputSet)
solver_exactReach = ExactReach()

#reluVal
inputSet  = Hyperrectangle(low = [-1.0], high = [1.0])
outputSet = Hyperrectangle(low = [-1.0], high = [50.0])
problem_reluVal = Problem(small_nnet, inputSet, outputSet)
solver_reluVal = ReluVal(2)

### Ai2
input  = HPolytope(A, [0.0, 0.0])
output = HPolytope(A, [100.0, 0.0])
problem_ai2 = Problem(small_nnet, input, output)
solver_ai2 = Ai2()

### BaB
#inputSet  = Hyperrectangle([-1.0], [0.5])
#outputSet = HPolytope(ones(1,1), [18.5])
#problem_bab = Problem(small_nnet, inputSet, outputSet)
#solver_bab = BaB(0.1, GLPKSolverMIP())

### Certify
inputSet = Hyperrectangle([1.0], [1.0])
outputSet = HPolytope(ones(1,1), [2.1])
problem_certify = Problem(small_nnet, inputSet, outputSet)
solver_certify = Certify(SCSSolver())

### ConvDual
inputSet  = Hyperrectangle([0.0], [1.0])
outputSet = HPolytope(-ones(1,1), [1.0])
problem_convdual = Problem(small_nnet, inputSet, outputSet)
solver_convdual = ConvDual()

### Duality
inputSet  = Hyperrectangle([1.0], [1.0])
outputSet = HPolytope(ones(1,1), [120.0])
problem_duality = Problem(small_nnet, inputSet, outputSet)
solver_duality = Duality(GLPKSolverMIP())

### iLP
inputSet  = Hyperrectangle([-2.0], [.5])
outputSet = HPolytope(ones(1,1),[72.5])
problem_ilp = Problem(small_nnet, inputSet, outputSet)
solver_ilp = ILP(GLPKSolverMIP(), 1)

### MIPVerify
inputSet  = Hyperrectangle([0.0], [.5])
outputSet = HPolytope(ones(1,1), [102.5])
problem_mipverify = Problem(small_nnet, inputSet, outputSet)
solver_mipverify = MIPVerify(GLPKSolverMIP())

### Planet
inputSet  = Hyperrectangle([-1.0], [0.5])
outputSet = HPolytope(ones(1,1), [2.5])
problem_planet = Problem(small_nnet, inputSet, outputSet)
solver_planet = Planet(GLPKSolverMIP())

### Reluplex
inputSet  = Hyperrectangle([1.0],[.2])
outputSet = Hyperrectangle([2.0,], [100.0])
problem_reluplex = Problem(small_nnet, inputSet, outputSet)
solver_reluplex = Reluplex()

### Sherlock
inputSet  = Hyperrectangle([2.0], [.5])
outputSet = Hyperrectangle([10.0], [10.0])
problem_sherlock = Problem(small_nnet, inputSet, outputSet)
solver_sherlock = Sherlock(GLPKSolverMIP(), 0.1)

#@test @no_error solve(solver_reverify,   problem_reverify)
#@test @no_error solve(solver_maxSens,    problem_maxSens)
@test @no_error solve(solver_exactReach, problem_exactReach)
@test @no_error solve(solver_reluVal,    problem_reluVal)
@test @no_error solve(solver_ai2,        problem_ai2)
@test @no_error solve(solver_bab,        problem_bab)
@test @no_error solve(solver_certify,    problem_certify)
@test @no_error solve(solver_convdual,   problem_convdual)
@test @no_error solve(solver_duality,    problem_duality)
@test @no_error solve(solver_ilp,        problem_ilp)
@test @no_error solve(solver_mipverify,  problem_mipverify)
@test @no_error solve(solver_planet,     problem_planet)
@test @no_error solve(solver_reluplex,   problem_reluplex)
@test @no_error solve(solver_sherlock,   problem_sherlock)