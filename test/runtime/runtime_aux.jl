using NeuralVerification, LazySets, Test, LinearAlgebra
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

net_path = joinpath(@__DIR__, "../networks/")

function test_runtime(problem, solver)
	solvername = string(typeof(solver))
	println("Solver: "*solvername)
	timed_result =@timed solve(solver, problem)
	println(" - Time: " * string(timed_result[2]) * " s")
	println(" - Output: ")
	println(timed_result[1])
end


# Problem: Small NN, input: HPolytope, output: HPolytope

small_nnet_file = net_path * "small_nnet.nnet"
small_nnet_id_file = net_path * "small_nnet_id.nnet"
small_nnet  = read_nnet(small_nnet_file, last_layer_activation = ReLU())
small_nnet_id  = read_nnet(small_nnet_id_file, last_layer_activation = Id())

in_hyper  = Hyperrectangle(low = [-0.9], high = [0.9])
in_hpoly  = convert(HPolytope, in_hyper)
out_superset    = Hyperrectangle(low = [30.0], high = [80.0])

problem_small_HH    = Problem(small_nnet, in_hpoly, convert(HPolytope, out_superset))

# Problem: Small NN, input: Hyperrectangle, output: HPolytope

in_hyper  = Hyperrectangle(low = [-0.9], high = [0.9])
problem_small_RH = Problem(small_nnet, in_hyper, HPolytope([HalfSpace([1.], 100.)]))

# Problem: Small NN, input: Hyperrectangle, output: Hyperrectangle

out_hyper    = Hyperrectangle(low = [30.0], high = [80.0])
problem_small_RR = Problem(small_nnet, in_hyper, out_hyper)

# Problem: ACAS, input: HPolytope, output: HPolytope

acas_file = net_path * "ACASXU_run2a_4_5_batch_2000.nnet"
acas_nnet = read_nnet(acas_file, last_layer_activation = Id())

A0 = Matrix{Float64}(I, 5, 5)
A1 = -Matrix{Float64}(I, 5, 5)
A = vcat(A0, A1)
b_lower = [ 0.21466922,  0.11140846, -0.4999999 ,  0.3920202 ,  0.4      ]
b_upper = [ 0.58819589,  0.4999999 , -0.49840835,  0.66474747,  0.4      ]
in_hyper  = Hyperrectangle(low = b_lower, high = b_upper)
inputSet = convert(HPolytope, in_hyper)

A = Matrix(undef, 2, 1)
A = [1.0, 0.0, 0.0, 0.0, -1.0]'
b = [0.0]
outputSet = HPolytope(A, b)

problem_acas_HH = Problem(acas_nnet, inputSet, outputSet)

# Problem: ACAS, input: Hyperrectangle, output: HPolytope

in_hyper  = Hyperrectangle(low = b_lower, high = b_upper)
problem_acas_HrH = Problem(acas_nnet, in_hyper, outputSet)

# Problem: ACAS, input: Hyperrectangle, output: HPolytope

in_hyper  = Hyperrectangle(low = b_lower, high = b_upper)
out_hyper  = Hyperrectangle(low = [-0.1], high = [3.0])
problem_acas_RR = Problem(acas_nnet, in_hyper, out_hyper)

# Problem: ACAS1 (one output), input: Hyperrectangle, output: Hyperrectangle
acas_file = net_path * "ACASXU_run2a_4_5_batch_2000_out1.nnet"
#ACASXU_run2a_4_5_batch_2000.nnet
acas_nnet1 = read_nnet(acas_file, last_layer_activation = Id())
in_hyper  = Hyperrectangle(low = b_lower, high = b_upper)
out_hyper  = Hyperrectangle(low = [-0.1], high = [3.0])

problem_acas1_RR = Problem(acas_nnet1, in_hyper, out_hyper)

