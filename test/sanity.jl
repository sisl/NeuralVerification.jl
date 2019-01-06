# sanity checks 

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

inputSet  = Hyperrectangle(low = [-1.0], high = [1.0])
outputSet = Hyperrectangle(low = [-1.0], high = [70.0])
problem_sat_hyper_hyper = Problem(small_nnet, inputSet, outputSet)


inputSet  = Hyperrectangle(low = [-1.0], high = [1.0])
outputSet = Hyperrectangle(low = [-1.0], high = [50.0])
problem_unsat_hyper_hyper = Problem(small_nnet, inputSet, outputSet)

A = [1.0, -1.0]
outputSet = HPolytope(A[1:1, :], [10.0])




# Problems

inputSet  = Hyperrectangle(low = [-1.0], high = [1.0])
outputSet = Hyperrectangle(low = [-1.0], high = [70.0])

# Group 1
# Input: HPolytope, Output: HPolytope

# SAT 
#MaxSens
solver = MaxSens(1.0, false)

#Ai2
solver = Ai2()

#ExactReach
solver = ExactReach()


# UNSAT
#MaxSens
solver = MaxSens(1.0, false)

#Ai2
solver = Ai2()

#ExactReach
solver = ExactReach()


# Group 2
# Input: HPolytope, Output: HPolytope

# SAT 



# UNSAT



# Group 3
# Input: HPolytope, Output: HPolytope



# Group 4
# Input: HPolytope, Output: HPolytope



# Group 5
# Input: Hyperrectangle, Output: Hyperrectangle

# ReluVal
# SAT
solver = ReluVal(max_iter = 1)
result = solve(solver, problem_sat_hyper_hyper)
@test result.status != :UNSAT

# UNSAT 
result = solve(solver, problem_unsat_hyper_hyper)
@test result.status == :UNSAT

# DLV
# SAT
solver = DLV(1.0)
result = solve(solver, problem_sat_hyper_hyper)
@test result.status != :SAT

# UNSAT 
result = solve(solver, problem_unsat_hyper_hyper)
@test result.status == :UNSAT


# Sherlock
# SAT
optimizer = GLPKSolverMIP()
solver = Sherlock(optimizer, 1.0)
result = solve(solver, problem_sat_hyper_hyper)
@test result.status != :SAT

# UNSAT 
result = solve(solver, problem_unsat_hyper_hyper)
@test result.status == :UNSAT

# BaB
optimizer = GLPKSolverMIP()
solver = BaB(optimizer, 0.1)
result = solve(solver, problem_sat_hyper_hyper)
@test result.status != :SAT

# UNSAT 
result = solve(solver, problem_unsat_hyper_hyper)
@test result.status == :UNSAT

# Group 6
# Input: Hyperrectangle, Output: Hyperrectangle

# Planet - ISSUES
optimizer = GLPKSolverMIP()
solver = Planet(optimizer)
result = solve(solver, problem_sat_hyper_hyper)
#@test result.status != :SAT

# UNSAT 
result = solve(solver, problem_unsat_hyper_hyper)
#@test result.status == :UNSAT

# Reluplex
# Planet - ISSUES
solver = Reluplex()
result = solve(solver, problem_sat_hyper_hyper)
@test result.status != :SAT

# UNSAT 
result = solve(solver, problem_unsat_hyper_hyper)
@test result.status == :UNSAT
