using LazySets, Test, LinearAlgebra, GLPKMathProgInterface
using NeuralVerification
using JLD2, FileIO

# The following two test is based on a 4 layer network, node number of each layer:
# 1, 6, 4, 3, 1.

# We can consider this network is in the form of y=f(x), where x and y are 1d variables.
# The graph of this function looks like \/\/\/\/

# To get a precise rechable set, the verification algorithm needs to fully split
# the input set to reduce all over-approximation.

# Please make sure 'max_iter' of the solver is large enough to get a fully split.

function test_holds(solver, eps)
    @load "fully_split.jld2" x_min x_max y_min y_max w_nnet
    inputSet = Hyperrectangle(low = [x_min], high = [x_max])
    outputSet  = Hyperrectangle(low = [y_min - eps], high = [y_max + eps])
    problem = Problem(w_nnet, inputSet, outputSet);
    result = solve(solver, problem)
    @test result.status == :holds
end

function test_violated(solver, eps)
    @load "fully_split.jld2" x_min x_max y_min y_max w_nnet
    inputSet = Hyperrectangle(low = [x_min], high = [x_max])
    outputSet  = Hyperrectangle(low = [y_min + eps], high = [y_max - eps])
    problem = Problem(w_nnet, inputSet, outputSet);
    result = solve(solver, problem)
    @test result.status == :violated
end

@testset "Fully split, ReluVal" begin
    eps = 1e-1
    test_holds(ReluVal(max_iter=1000), eps)
    test_violated(ReluVal(max_iter=1000), eps)
end

@testset "Fully split, Neurify" begin
    eps = 1e-1
    test_holds(Neurify(max_iter=100), eps)
    test_violated(Neurify(max_iter=100), eps)
end

