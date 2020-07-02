# Test by creating a problem, then writing it, then reading it back in
# afterwards make sure that they are equal.
@testset "Read and write set test" begin
    # Construct the problem from a network, input set, and output set
    cartpole_nnet_file = "$(@__DIR__)/../examples/networks/cartpole_nnet.nnet" # 4 --> 16 --> 16 --> 16 --> 2
    nnet = read_nnet(small_nnet_file)
    input_set = Hyperrectangle([1.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0])
    output_set = HalfSpace([1.0, -1.0], 5)
    problem = Problem(nnet, input_set, output_set)

    # Write out the problem
    nnet_file = string(tempname(), ".nnet")
    input_set_file = string(tempname(), ".txt")
    output_set_file = string(tempname(), ".txt")

    write_problem(nnet_file, input_set_file, output_set_file, problem)
    problem_in = read_problem(nnet_file, input_set_file, output_set_file)

    @test problem == problem_in
end
