# FastLin, FastLip, MIPVerify, ILP

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

# Problem type - input:Hyperrectangle, output:Hpolytope
print("###### Group 4 - input:Hyperrectangle, output:HpolyTope (one constraint) ######\n")
# Small MNIST Network 

mnist_small = read_nnet("$at/../examples/networks/mnist_small.nnet")

# entry 23 in MNIST datset
input_center = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,121,254,136,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,13,230,253,248,99,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,118,253,253,225,42,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,61,253,253,253,74,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,32,206,253,253,186,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,211,253,253,239,69,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,254,253,253,133,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,142,255,253,186,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,149,229,254,207,21,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,54,229,253,254,105,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,152,254,254,213,26,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,112,251,253,253,26,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,29,212,253,250,149,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,36,214,253,253,137,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,75,253,253,253,59,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,93,253,253,189,17,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,253,253,84,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,43,235,253,126,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,99,248,253,119,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,225,235,49,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#output_center = [-1311.1257826380004,4633.767704436501,-654.0718535670002,-1325.349417307,1175.2361184373997,-1897.8607293569007,-470.3405972940001,830.8337987382,-377.7467076115001,572.3674015264198]

in_epsilon = 1 #0-255
out_epsilon = 10 #logit domain

input_low = input_center .- in_epsilon
input_high = input_center .+ in_epsilon

inputSet = Hyperrectangle(low=input_low, high=input_high)

A = Matrix(undef, 2, 1)
A = [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ,0.0]'
b = [0.0]
outputSet = HPolytope(A, b)

########### Small ############
# Problem type - input:Hyperrectangle, output:Hpolytope (one constraint)

problem_hyperrect_oneineq_small = Problem(mnist_small, inputSet, outputSet)
print("\n\n################## Small ##################\n")

# FastLin
print("\nFastLin - Small")
solver = FastLin(10, 10.0, 1.0)
timed_result = @timed solve(solver, problem_hyperrect_oneineq_small)
print(" - Time: " * string(timed_result[2]) * " s")
print(" - Output: ")
print(timed_result[1])

# FastLip
print("\nFastLip - Small")
solver = FastLip(10, 10.0, 1.0)
timed_result = @timed solve(solver, problem_hyperrect_oneineq_small)
print(" - Time: " * string(timed_result[2]) * " s")
print(" - Output: ")
print(timed_result[1])

# MIPVerify
print("\nMIPVerify - Small")
optimizer = GLPKSolverMIP()
solver = MIPVerify(optimizer)
timed_result = @timed solve(solver, problem_hyperrect_oneineq_small)
print(" - Time: " * string(timed_result[2]) * " s")
print(" - Output: ")
print(timed_result[1])

# ILP
print("\nILP - Small")
optimizer = GLPKSolverMIP()
solver = ILP(optimizer, 1)
timed_result = @timed solve(solver, problem_hyperrect_oneineq_small)
print(" - Time: " * string(timed_result[2]) * " s")
print(" - Output: ")
print(timed_result[1])

########### Deep ############
print("\n\n################## Deep ##################\n")

mnist_deep = read_nnet("$at/../examples/networks/mnist_large.nnet")
problem_hyperrect_oneineq_large = Problem(mnist_deep, inputSet, outputSet)

# FastLin
print("\nFastLin - Deep")
solver = FastLin(10, 10.0, 1.0)
timed_result = @timed solve(solver, problem_hyperrect_oneineq_large)
print(" - Time: " * string(timed_result[2]) * " s")
print(" - Output: ")
print(timed_result[1])

# FastLip
print("\nFastLip - Deep")
solver = FastLip(10, 10.0, 1.0)
timed_result = @timed solve(solver, problem_hyperrect_oneineq_large)
print(" - Time: " * string(timed_result[2]) * " s")
print(" - Output: ")
print(timed_result[1])

# MIPVerify
print("\nMIPVerify - Deep")
optimizer = GLPKSolverMIP()
solver = MIPVerify(optimizer)
timed_result = @timed solve(solver, problem_hyperrect_oneineq_large)
print(" - Time: " * string(timed_result[2]) * " s")
print(" - Output: ")
print(timed_result[1])

# ILP
print("\nILP - Deep")
optimizer = GLPKSolverMIP()
solver = ILP(optimizer, 1)
timed_result = @timed solve(solver, problem_hyperrect_oneineq_large)
print(" - Time: " * string(timed_result[2]) * " s")
print(" - Output: ")
print(timed_result[1])

########### Wide ############
print("\n\n################## Wide ##################\n")

mnist_wide = read_nnet("$at/../examples/networks/mnist-1-100.nnet")
problem_hyperrect_oneineq_wide = Problem(mnist_wide, inputSet, outputSet)

# FastLin
print("\nFastLin - Wide")
solver = FastLin(10, 10.0, 1.0)
timed_result = @timed solve(solver, problem_hyperrect_oneineq_wide)
print(" - Time: " * string(timed_result[2]) * " s")
print(" - Output: ")
print(timed_result[1])

# FastLip
print("\nFastLip - Wide")
solver = FastLip(10, 10.0, 1.0)
timed_result = @timed solve(solver, problem_hyperrect_oneineq_wide)
print(" - Time: " * string(timed_result[2]) * " s")
print(" - Output: ")
print(timed_result[1])

# MIPVerify
print("\nMIPVerify - Wide")
optimizer = GLPKSolverMIP()
solver = MIPVerify(optimizer)
timed_result = @timed solve(solver, problem_hyperrect_oneineq_wide)
print(" - Time: " * string(timed_result[2]) * " s")
print(" - Output: ")
print(timed_result[1])

# ILP
print("\nILP - Wide")
optimizer = GLPKSolverMIP()
solver = ILP(optimizer, 1)
timed_result = @timed solve(solver, problem_hyperrect_oneineq_wide)
print(" - Time: " * string(timed_result[2]) * " s")
print(" - Output: ")
print(timed_result[1])

print("\n\n################## Acas ##################\n")

acas_nnet = read_nnet("$at/../examples/networks/ACASXU_run2a_1_1_tiny_4.nnet")

b_upper = [0.58819589, 0.4999999 , -0.4999999, 0.52838384, 0.4]
b_lower = [0.21466922, 0.11140846, -0.4999999, 0.52838384, 0.4]

inputSet = Hyperrectangle(low=b_lower, high=b_upper)

A = Matrix(undef, 2, 1)
A = [1.0, 0.0, 0.0, 0.0, -1.0]'
b = [0.0]
outputSet = HPolytope(A, b)


problem_hyperrectangle_polytope_acas = Problem(acas_nnet, inputSet, outputSet)


# FastLin
print("\nFastLin - ACAS")
solver = FastLin(10, 10.0, 1.0)
timed_result = @timed solve(solver, problem_hyperrectangle_polytope_acas)
print(" - Time: " * string(timed_result[2]) * " s")
print(" - Output: ")
print(timed_result[1])

# FastLip
print("\nFastLip - ACAS")
solver = FastLip(10, 10.0, 1.0)
timed_result = @timed solve(solver, problem_hyperrectangle_polytope_acas)
print(" - Time: " * string(timed_result[2]) * " s")
print(" - Output: ")
print(timed_result[1])

# MIPVerify
print("\nMIPVerify - ACAS")
optimizer = GLPKSolverMIP()
solver = MIPVerify(optimizer)
timed_result = @timed solve(solver, problem_hyperrectangle_polytope_acas)
print(" - Time: " * string(timed_result[2]) * " s")
print(" - Output: ")
print(timed_result[1])

# ILP
print("\nILP - ACAS")
optimizer = GLPKSolverMIP()
solver = ILP(optimizer, 1)
timed_result = @timed solve(solver, problem_hyperrectangle_polytope_acas)
print(" - Time: " * string(timed_result[2]) * " s")
print(" - Output: ")
print(timed_result[1])

