using NeuralVerification, LazySets, Test, LinearAlgebra
import NeuralVerification: ReLU, Id
include("runtime_aux.jl")

# Group 1

# Small Problem: problem_small_HH
# Solvers: ExactReach(), Ai2(), MaxSens(resolution = 0.6)

# ACAS Problem: problem_acas_HH
# Solvers: ExactReac() - timesout, Ai2(), MaxSens(1.0, false)



# Group 2

# Small Problem: problem_small_HrH
# Solvers: NSVerify(GLPKSolverMIP(), 1000.0), MIPVerify(GLPKSolverMIP()),
# 	ILP(GLPKSolverMIP(), 1), 

# ACAS Problem: problem_acas_HrH
# Solvers: NSVerify(), MIPVerify(), ILP()



# Group 3

# Small Problem: problem_small_HH
# Solvers: Duality(), ConvDual(), 

# ACAS Problem: problem_acas_HrH
# Solvers: Duality(), ConvDual()



# Group 4

# Small Problem: problem_small_HrH
# Solvers: FastLin(), FastLip(), MIPVerify(), ILP()

# ACAS Problem: problem_acas_HrH
# Solvers: FastLin(), FastLip(), MIPVerify(), ILP()



# Group 5

# Small Problem: problem_small_RR
# Solvers: ReluVal(), DLV(), Sherlock(), BaB()

# ACAS Problem: problem_acas1_RR
# Solvers: ReluVal(), DLV(), Sherlock(), BaB()



# Group 6

# Small Problem: problem_small_HH
# Solvers: Planet(), Reluplex(), ReluVal()

# ACAS Problem: problem_acas_HrH
# Solvers: Planet(), Reluplex(), ReluVal()

test_runtime(problem_small_HH, ExactReach())