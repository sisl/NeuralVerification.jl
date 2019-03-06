# ExactReach, ai2, maxSens

using NeuralVerification, LazySets, Test, LinearAlgebra, GLPKMathProgInterface
import NeuralVerification: ReLU, Id

macro no_error(ex)
    quote
        try $(esc(ex))
            true
        catch
            false
        end
    end
end

small_nnet_file = "$(@__DIR__)/../examples/networks/small_nnet.nnet"
small_nnet_id_file = "$(@__DIR__)/../examples/networks/small_nnet_id.nnet"

mnist1_file = "$(@__DIR__)/../examples/networks/mnist1.nnet"
mnist2_file = "$(@__DIR__)/../examples/networks/mnist2.nnet"
mnist3_file = "$(@__DIR__)/../examples/networks/mnist3.nnet"
mnist4_file = "$(@__DIR__)/../examples/networks/mnist4.nnet"

small_nnet  = read_nnet(small_nnet_file, last_layer_activation = ReLU())
small_nnet_id  = read_nnet(small_nnet_id_file, last_layer_activation = Id())

mnist1 = read_nnet(mnist1_file, last_layer_activation = Id())
mnist2 = read_nnet(mnist2_file, last_layer_activation = Id())
mnist3 = read_nnet(mnist3_file, last_layer_activation = Id())
mnist4 = read_nnet(mnist4_file, last_layer_activation = Id())



println("###### Problem type - input:HPolytope, output:HPolytope ######")
println("###### Network: small_nnet, problem: holds              ######")

in_hyper  = Hyperrectangle(low = [-0.9], high = [0.9])
in_hpoly  = convert(HPolytope, in_hyper)
#out_superset    = Hyperrectangle(low = [30.0], high = [80.0])

problem_holds    = Problem(small_nnet, in_hyper, HPolytope([HalfSpace([1.], 100.)]))  

optimizer = GLPKSolverMIP()
solver = Planet(optimizer)
println("$(typeof(solver)) - small_nnet")
timed_result =@timed solve(solver, problem_holds)
println(" - Time: " * string(timed_result[2]) * " s")
println(" - Output: ")
println(timed_result[1])
println("")
#=
solver=Reluplex()
println("$(typeof(solver)) - small_nnet")
timed_result =@timed solve(solver, problem_holds)
println(" - Time: " * string(timed_result[2]) * " s")
println(" - Output: ")
println(timed_result[1])
println("")

solver = ReluVal(max_iter = 2)
println("$(typeof(solver)) - small_nnet")
timed_result =@timed solve(solver, problem_holds)
println(" - Time: " * string(timed_result[2]) * " s")
println(" - Output: ")
println(timed_result[1])
println("")

=#


println("###### Problem type - input:HPolytope, output:HPolytope ######")


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

A = Matrix(undef, 2, 1)
A = [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ,0.0]'
b = [0.0]
outputSet = HPolytope(A, b)

problem_mnist1 = Problem(mnist1, inputSet, outputSet)
## MNIST1 ##
print("\n\n\n")
println("###### Network: mnist1                                  ######")
print("\n\n\n")



optimizer = GLPKSolverMIP()
solver = Planet(optimizer)
println("$(typeof(solver)) - mnist1")
timed_result =@timed solve(solver, problem_mnist1)
print(" - Time: " * string(timed_result[2]) * " s")
print(" - Output: ")
print(timed_result[1])
println("")
#=
solver = Reluplex()
println("$(typeof(solver)) - mnist1")
timed_result =@timed solve(solver, problem_mnist1)
println(" - Time: " * string(timed_result[2]) * " s")
println(" - Output: ")
println(timed_result[1])
println("")

solver = ReluVal(max_iter = 2)
println("$(typeof(solver)) - mnist1")
timed_result =@timed solve(solver, problem_mnist1)
print(" - Time: " * string(timed_result[2]) * " s")
print(" - Output: ")
print(timed_result[1])
println("")

=#
problem_mnist2 = Problem(mnist2, inputSet, outputSet)
## MNIST1 ##
print("\n\n\n")
println("###### Network: mnist2                                  ######")
print("\n\n\n")



optimizer = GLPKSolverMIP()
solver = Planet(optimizer)
println("$(typeof(solver)) - mnist2")
timed_result =@timed solve(solver, problem_mnist2)
print(" - Time: " * string(timed_result[2]) * " s")
print(" - Output: ")
print(timed_result[1])
println("")

#=
solver = Reluplex()
println("$(typeof(solver)) - mnist2")
timed_result =@timed solve(solver, problem_mnist2)
println(" - Time: " * string(timed_result[2]) * " s")
println(" - Output: ")
println(timed_result[1])
println("")

solver = ReluVal(max_iter = 2)
println("$(typeof(solver)) - mnist2")
timed_result =@timed solve(solver, problem_mnist2)
print(" - Time: " * string(timed_result[2]) * " s")
print(" - Output: ")
print(timed_result[1])
println("")
=#
problem_mnist3 = Problem(mnist3, inputSet, outputSet)
## MNIST3 ##
print("\n\n\n")
println("###### Network: mnist3                                  ######")
print("\n\n\n")



optimizer = GLPKSolverMIP()
solver = Planet(optimizer)
println("$(typeof(solver)) - mnist3")
timed_result =@timed solve(solver, problem_mnist3)
print(" - Time: " * string(timed_result[2]) * " s")
print(" - Output: ")
print(timed_result[1])
println("")
#=
solver = Reluplex()
println("$(typeof(solver)) - mnist3")
timed_result =@timed solve(solver, problem_mnist3)
println(" - Time: " * string(timed_result[2]) * " s")
println(" - Output: ")
println(timed_result[1])
println("")

solver = ReluVal(max_iter = 2)
println("$(typeof(solver)) - mnist3")
timed_result =@timed solve(solver, problem_mnist3)
print(" - Time: " * string(timed_result[2]) * " s")
print(" - Output: ")
print(timed_result[1])
println("")
=#
problem_mnist4 = Problem(mnist4, inputSet, outputSet)
## MNIST4 ##
print("\n\n\n")
println("###### Network: mnist4                                  ######")
print("\n\n\n")
#=

optimizer = GLPKSolverMIP()
solver = Planet(optimizer)
println("$(typeof(solver)) - mnist4")
timed_result =@timed solve(solver, problem_mnist4)
print(" - Time: " * string(timed_result[2]) * " s")
print(" - Output: ")
print(timed_result[1])
println("")

solver=Reluplex()
println("$(typeof(solver)) - mnist4")
timed_result =@timed solve(solver, problem_mnist4)
println(" - Time: " * string(timed_result[2]) * " s")
println(" - Output: ")
println(timed_result[1])
println("")

solver = ReluVal(max_iter = 2)
println("$(typeof(solver)) - mnist4")
timed_result =@timed solve(solver, problem_mnist4)
print(" - Time: " * string(timed_result[2]) * " s")
print(" - Output: ")
print(timed_result[1])
println("")
=#
# ACAS
print("\n\n\n")
println("###### Problem type - input:HPolytope, output:HPolytope ######")
print("\n\n\n")
println("###### Network: acas                                    ######")
print("\n\n\n")

#acas_file = "$(@__DIR__)/../examples/networks/ACASXU_run2a_1_1_tiny_4.nnet"
acas_file = "$(@__DIR__)/../examples/networks/ACASXU_run2a_4_5_batch_2000.nnet"
#ACASXU_run2a_4_5_batch_2000.nnet

acas_nnet = read_nnet(acas_file, last_layer_activation = Id())

# ACAS PROPERTY 10 - modified
# Original input range: 
# LOWER BOUND: array([[ 0.21466922,  0.11140846, -0.4999999 ,  0.3920202 ,  0.15      ]])
# UPPER BOUND: array([[ 0.58819589,  0.4999999 , -0.49840835,  0.66474747,  0.65      ]])
A0 = Matrix{Float64}(I, 5, 5)
A1 = -Matrix{Float64}(I, 5, 5)
A = vcat(A0, A1)


#b_lower = [0.21466922, 0.11140846, -0.4999999, 0.52838384, 0.4]
#b_upper = [0.58819589, 0.4999999 , -0.4999999, 0.52838384, 0.4]

#b_lower = [ 0.21466922,  0.11140846, -0.4999999 ,  0.3920202 ,  0.15      ]
#b_upper = [ 0.58819589,  0.4999999 , -0.49840835,  0.66474747,  0.65      ]

#b_lower = [ 0.21466922,  0.11140846, -0.4999999 ,  0.3920202 ,  0.20      ]
#b_upper = [ 0.58819589,  0.4999999 , -0.49840835,  0.66474747,  0.65      ]

b_lower = [ 0.21466922,  0.11140846, -0.4999999 ,  0.3920202 ,  0.4      ]
b_upper = [ 0.58819589,  0.4999999 , -0.49840835,  0.66474747,  0.4      ]


in_hyper  = Hyperrectangle(low = b_lower, high = b_upper)
inputSet = convert(HPolytope, in_hyper)

#inputSet = HPolytope(A, b)


#A = Matrix(undef, 2, 1)
#A = [1.0, 0.0, 0.0, 0.0, -1.0]'
#b = [0.0]
#outputSet = HPolytope(A, b)

outputSet = HPolytope([HalfSpace([1.0, 0.0, 0.0, 0.0, -1.0], 0.0)])

problem_polytope_polytope_acas = Problem(acas_nnet, in_hyper, outputSet)

print("\n\n\n")
println("###### Network: acas                                  ######")
print("\n\n\n")


optimizer = GLPKSolverMIP()
solver = Planet(optimizer)
println("$(typeof(solver)) - acas")
timed_result =@timed solve(solver, problem_polytope_polytope_acas)
println(" - Time: " * string(timed_result[2]) * " s")
println(" - Output: ")
println(timed_result[1])
println("")

#=

solver = ReluVal(max_iter = 20)
println("$(typeof(solver)) - acas")
timed_result =@timed solve(solver, problem_polytope_polytope_acas)
print(" - Time: " * string(timed_result[2]) * " s")
print(" - Output: ")
print(timed_result[1])
println("")


solver=Reluplex()
println("$(typeof(solver)) - acas")
timed_result =@timed solve(solver, problem_polytope_polytope_acas)
println(" - Time: " * string(timed_result[2]) * " s")
println(" - Output: ")
println(timed_result[1])
println("")
=#