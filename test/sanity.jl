# sanity checks 

using NeuralVerification
using Test
using LinearAlgebra

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

small_nnet = read_nnet("$at/../examples/networks/small_nnet.nnet")

inputSet  = Hyperrectangle(low = [-1.0], high = [1.0])
outputSet = Hyperrectangle(low = [-1.0], high = [70.0])

# -1.0 < x < 1.0, -1.0 < y < 70.0
problem_sat_hyper_hyper = Problem(small_nnet, inputSet, outputSet)

inputSet  = Hyperrectangle(low = [-1.0], high = [1.0])
outputSet = Hyperrectangle(low = [-1.0], high = [50.0])

# -1.0 < x < 1.0, -1.0 < y < 50.0
problem_unsat_hyper_hyper = Problem(small_nnet, inputSet, outputSet)


A0 = Matrix{Float64}(I, 1, 1)
A1 = -Matrix{Float64}(I, 1, 1)
A = vcat(A0, A1)

b0 = ones(1)*1.0
b1 = -ones(1)*-1.0
b = vcat(b0, b1)

inputSet = HPolytope(A, b)

A = [1.0, -1.0]
outputSet = HPolytope(A[1:1, :], [10.0])

#  -1.0 < x < 1.0, y < 10.0
problem_unsat_hpoly_hpoly = Problem(small_nnet, inputSet, outputSet)

outputSet = HPolytope(A[1:1, :], [100.0])

#  -1.0 < x < 1.0, y < 100.0
problem_sat_hpoly_hpoly = Problem(small_nnet, inputSet, outputSet)

A0 = Matrix{Float64}(I, 1, 1)
A1 = -Matrix{Float64}(I, 1, 1)
A = vcat(A0, A1)

b0 = ones(1)*60.0
b1 = -ones(1)*40.0
b = vcat(b0, b1)

outputSet = HPolytope(A, b)


# -1.0 < x < 1.0, 40.0 < y < 60.0
problem_sat_hpoly_hpoly_bounded = Problem(small_nnet, inputSet, outputSet)

A0 = Matrix{Float64}(I, 1, 1)
A1 = -Matrix{Float64}(I, 1, 1)
A = vcat(A0, A1)

b0 = ones(1)*110.0
b1 = -ones(1)*100.0
b = vcat(b0, b1)

outputSet = HPolytope(A, b)

# -1.0 < x < 1.0, 100.0 < y < 110.0
problem_unsat_hpoly_hpoly_bounded = Problem(small_nnet, inputSet, outputSet)

inputSet  = Hyperrectangle(low = [-1.0], high = [1.0])
A = [1.0, -1.0]
outputSet = HPolytope(A[1:1, :], [10.0])

# -1.0 < x < 1.0, y < 10.0
problem_unsat_hyper_hpoly = Problem(small_nnet, inputSet, outputSet)

A = [1.0, -1.0]
outputSet = HPolytope(A[1:1, :], [100.0])
# -1.0 < x < 1.0, y < 100.0
problem_sat_hyper_hpoly = Problem(small_nnet, inputSet, outputSet)

# Problems

# Group 1
# Input: HPolytope, Output: HPolytope



#MaxSens
# SAT
solver = MaxSens(1.0, false)
result = solve(solver, problem_sat_hpoly_hpoly_bounded)
print("Maxsens SAT: ")
print(result.status)
print("\n")

# UNSAT

result = solve(solver, problem_unsat_hpoly_hpoly_bounded)
print("Maxsens UNSAT: ")
print(result.status)
print("\n")

# ExactReach
# SAT
solver = ExactReach()
result = solve(solver, problem_sat_hpoly_hpoly_bounded)
print("ExactReach SAT: ")
print(result.status)
print("\n")

# UNSAT

result = solve(solver, problem_unsat_hpoly_hpoly_bounded)
print("ExactReach UNSAT: ")
print(result.status)
print("\n")


#Ai2
solver = Ai2()
#Ai2
solver = Ai2()
#result = solve(solver, problem_sat_hpoly_hpoly_bounded)
print("Ai2 SAT: ")
#print(result.status)
print("\n")

# UNSAT

#result = solve(solver, problem_unsat_hpoly_hpoly_bounded)
print("Ai2 UNSAT: ")
#print(result.status)
print("\n")


solver = ExactReach()


# Group 2
# Input: HPolytope, Output: HPolytope

# NSVerify
optimizer = GLPKSolverMIP()
solver = NSVerify(optimizer, 5.0)

# SAT
result = solve(solver, problem_sat_hyper_hpoly)
print("NSVerify SAT: ")
print(result.status)
print("\n")

# UNSAT
result = solve(solver, problem_unsat_hyper_hpoly)
print("NSVerify UNSAT: ")
print(result.status)
print("\n")

# MIPVerify
optimizer = GLPKSolverMIP()
solver = MIPVerify(optimizer)

# SAT
result = solve(solver, problem_sat_hyper_hpoly)
print("MIPVerify SAT: ")
print(result.status)
print("\n")

# UNSAT
result = solve(solver, problem_unsat_hyper_hpoly)
print("MIPVerify UNSAT: ")
print(result.status)
print("\n")

# ILP
optimizer = GLPKSolverMIP()
solver = ILP(optimizer, 1)

# SAT
result = solve(solver, problem_sat_hyper_hpoly)
print("ILP SAT: ")
print(result.status)
print("\n")

# UNSAT
result = solve(solver, problem_unsat_hyper_hpoly)
print("ILP UNSAT: ")
print(result.status)
print("\n")

# Group 3
# Input: HPolytope, Output: HPolytope

# convDual
optimizer = GLPKSolverMIP()
solver = ConvDual()

# SAT
result = solve(solver, problem_sat_hyper_hpoly)
print("convDual SAT: ")
print(result.status)
print("\n")

# UNSAT
result = solve(solver, problem_unsat_hyper_hpoly)
print("convDual UNSAT: ")
print(result.status)
print("\n")

# Duality
optimizer = GLPKSolverMIP()
solver = Duality(optimizer)

# SAT
result = solve(solver, problem_sat_hyper_hpoly)
print("Duality SAT: ")
print(result.status)
print("\n")

# UNSAT
result = solve(solver, problem_unsat_hyper_hpoly)
print("Duality UNSAT: ")
print(result.status)
print("\n")

# Group 4
# Input: HPolytope, Output: HPolytope

# FastLin
solver = FastLin(10, 10.0, 1.0)
# SAT
result = solve(solver, problem_sat_hyper_hpoly)
print("FastLin SAT: ")
print(result.status)
print("\n")

# UNSAT
#result = solve(solver, problem_unsat_hyper_hpoly)
print("FastLin UNSAT: ")
#print(result.status)
print("\n")

# FastLip
solver = FastLip(10, 10.0, 1.0)
# SAT
result = solve(solver, problem_sat_hyper_hpoly)
print("FastLip SAT: ")
print(result.status)
print("\n")

# UNSAT
result = solve(solver, problem_unsat_hyper_hpoly)
print("FastLip UNSAT: ")
print(result.status)
print("\n")

# Group 5
# Input: Hyperrectangle, Output: Hyperrectangle

# ReluVal
solver = ReluVal(max_iter = 1)
# SAT
result = solve(solver, problem_sat_hyper_hyper)
print("ReluVal SAT: ")
print(result.status)
print("\n")

# UNSAT
result = solve(solver, problem_unsat_hyper_hyper)
print("ReluVal UNSAT: ")
print(result.status)
print("\n")

# DLV
optimizer = GLPKSolverMIP()
solver = DLV(1.0)
# SAT
result = solve(solver, problem_sat_hyper_hyper)
print("DLV SAT: ")
print(result.status)
print("\n")

# UNSAT
result = solve(solver, problem_unsat_hyper_hyper)
print("DLV UNSAT: ")
print(result.status)
print("\n")

# Sherlock
optimizer = GLPKSolverMIP()
solver = Sherlock(optimizer, 1.0)
# SAT
result = solve(solver, problem_sat_hyper_hyper)
print("Sherlock SAT: ")
print(result.status)
print("\n")

# UNSAT
result = solve(solver, problem_unsat_hyper_hyper)
print("Sherlock UNSAT: ")
print(result.status)
print("\n")

# BaB
optimizer = GLPKSolverMIP()
solver = BaB(optimizer, 0.1)
# SAT
result = solve(solver, problem_sat_hyper_hyper)
print("BaB SAT: ")
print(result.status)
print("\n")

# UNSAT
result = solve(solver, problem_unsat_hyper_hyper)
print("BaB UNSAT: ")
print(result.status)
print("\n")

# GROUP 6
# Input: Hyperrectangle, Output: Hyperrectangle

# Planet
optimizer = GLPKSolverMIP()
solver = Planet(optimizer)
# SAT
result = solve(solver, problem_sat_hyper_hyper)
print("Planet SAT: ")
print(result.status)
print("\n")

# UNSAT
result = solve(solver, problem_unsat_hyper_hyper)
print("Planet UNSAT: ")
print(result.status)
print("\n")

# Reluplex
solver=Reluplex()
# SAT
result = solve(solver, problem_sat_hyper_hyper)
print("Reluplex SAT: ")
print(result.status)
print("\n")

# UNSAT
result = solve(solver, problem_unsat_hyper_hyper)
print("Reluplex UNSAT: ")
print(result.status)
print("\n")

