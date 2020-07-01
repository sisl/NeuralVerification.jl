using LazySets, Test, LinearAlgebra
using NeuralVerification
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

@testset "Splitting Test" begin

    net_file = "$(@__DIR__)/../examples/networks/R2_R2.nnet"
    # This is a 3 layer simple network mapping from R2 to R2.

    # y1 =  x1 + x2 - 1
    # y2 = -x1 + x2 - 1

    # z1 = ReLU(y1)
    # z2 = ReLU(y2)

    # o1 = 1*z1 + 0*z2 + 0
    # o2 = 0*z1 + 1*z2 + 0

    net = read_nnet(net_file, last_layer_activation = Id())
    A = [1. 0.; 0. 1.; -1. 0.; 0. -1.;]
    b = [1.1, 1.1, 0.1, 0.1]
    X = Hyperrectangle(low=[-1,-1], high=[1,1])
    # The true output set is a union of two segments : (0,0)->(0,1) U (0,0)->(1,0)
    Y = HPolytope(A, b) # Output constraints. Hyperrectangle: -0.1, -0.1, 1.1, 1.1
    problem = Problem(net, X, Y)

    @testset "Neurify" begin
        solver = Neurify()
        @test @no_error result = solve(solver, problem)
        result = solve(solver, problem)
        @test result.status == :holds
    end

    @testset "ReluVal" begin
        solver = ReluVal()
        @test @no_error result = solve(solver, problem)
        result = solve(solver, problem)
        @test result.status == :holds
    end

end