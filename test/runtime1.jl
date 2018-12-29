# ExactReach, ai2, maxSens

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

# Problem type - input:Hyperrectangle, output:Hyperrectangle
print("###### Problem type - input:Hyperrectangle, output:Hyperrectangle ######\n")
# Small MNIST Network 

mnist_small = read_nnet("$at/../examples/networks/mnist_small.nnet")

# entry 23 in MNIST datset
input_center = [0.0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,121,254,136,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,13,230,253,248,99,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,118,253,253,225,42,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,61,253,253,253,74,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,32,206,253,253,186,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,211,253,253,239,69,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,254,253,253,133,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,142,255,253,186,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,149,229,254,207,21,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,54,229,253,254,105,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,152,254,254,213,26,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,112,251,253,253,26,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,29,212,253,250,149,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,36,214,253,253,137,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,75,253,253,253,59,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,93,253,253,189,17,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,253,253,84,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,43,235,253,126,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,99,248,253,119,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,225,235,49,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
output_center = [-1311.1257826380004,4633.767704436501,-654.0718535670002,-1325.349417307,1175.2361184373997,-1897.8607293569007,-470.3405972940001,830.8337987382,-377.7467076115001,572.3674015264198]

in_epsilon = 1.0 #0-255
out_epsilon = 10.0 #logit domain

input_low = input_center .- in_epsilon
input_high = input_center .+ in_epsilon

output_low = output_center .- out_epsilon
output_high = output_center .+ out_epsilon

inputSet = Hyperrectangle(low=input_low, high=input_high)
outputSet = Hyperrectangle(low=output_low, high=output_high)

print("###### Problem type - input:HPolytope, output:HPolytope ######\n")

problem_hyperrectangle_hyperrectangle_small = Problem(mnist_small, inputSet, outputSet)
solver = MaxSens(.3, false)
#print("Ai2 - Small")
#@time solve(solver, problem_hyperrectangle_hyperrectangle_small)

# Unbounded polytope
#A = Matrix{Float64}(I, 784, 784)
#b = zeros(784)
#inputSet = HPolytope(A, b)

# Bounded Polytope
A0 = Matrix{Float64}(I, 784, 784)
A1 = -Matrix{Float64}(I, 784, 784)
A = vcat(A0, A1)

b0 = ones(784)*255.0
b1 = -ones(784)*250.0
b = vcat(b0, b1)

inputSet = HPolytope(A, b)

A = Matrix(undef, 2, 1)
A = [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ,0.0]'
b = [0.0]
outputSet = HPolytope(A, b)

# version with polytopes as seen in the repo, doesnt work either
problem_polytope_polytope_small = Problem(mnist_small, inputSet, outputSet)
solver = MaxSens(1.0, false)
print("\nMaxSens - Small - polytopes")
#timed_result =@timed solve(solver, problem_polytope_polytope_small)
print(" - Time: " * string(timed_result[2]) * " s")
print(" - Output: ")
print(timed_result[1])

solver = Ai2()
print("\nAi2 - Small")
#timed_result = @timed solve(solver, problem_polytope_polytope_small)
print(" - Time: " * string(timed_result[2]) * " s")
print(" - Output: ")
print(timed_result[1])

solver = ExactReach()
print("\nExactReach - Small")
#timed_result = @timed solve(solver, problem_polytope_polytope_small)
print(" - Time: " * string(timed_result[2]) * " s")
print(" - Output: ")
print(timed_result[1])

# ACAS

acas_nnet = read_nnet("$at/../examples/networks/ACASXU_run2a_1_1_tiny_4.nnet")

# ACAS PROPERTY 10 - modified

A0 = Matrix{Float64}(I, 5, 5)
A1 = -Matrix{Float64}(I, 5, 5)
A = vcat(A0, A1)

b_upper = [0.58819589, 0.4999999 , -0.4999999, 0.52838384, 0.4]
b_lower = [0.21466922, 0.11140846, -0.4999999, 0.52838384, 0.4]

b = vcat(b_upper, b_lower)
inputSet = HPolytope(A, b)


A = Matrix(undef, 2, 1)
A = [1.0, 0.0, 0.0, 0.0, -1.0]'
b = [0.0]
outputSet = HPolytope(A, b)

problem_polytope_polytope_acas = Problem(acas_nnet, inputSet, outputSet)

solver = MaxSens(1.0, false)
print("\nMaxSens - ACAS")
timed_result = @timed solve(solver, problem_polytope_polytope_acas)
print(" - Time: " * string(timed_result[2]) * " s")
print(" - Output: ")
print(timed_result[1])

solver = Ai2()
print("\nAi2 - ACAS")
timed_result = @timed solve(solver, problem_polytope_polytope_acas)
print(" - Time: " * string(timed_result[2]) * " s")
print(" - Output: ")
print(timed_result[1])

solver = ExactReach()
print("\nExactReach - ACAS")
timed_result = @timed solve(solver, problem_polytope_polytope_acas)
print(" - Time: " * string(timed_result[2]) * " s")
print(" - Output: ")
print(timed_result[1])


