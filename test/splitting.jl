@testset "Splitting Test" begin

    net_file = net_path * "R2_R2.nnet"

    net = read_nnet(net_file, last_layer_activation = Id())
    A = [1. 0.; 0. 1.; -1. 0.; 0. -1.;]
    b = [1.1, 1.1, 0.1, 0.1]
    X = Hyperrectangle(low=[-1,-1], high=[1,1])
    # The true output set is a union of two segments : (0,0)->(0,1) U (0,0)->(1,0)
    Y = HPolytope(A, b) # Output constraints. Hyperrectangle: -0.1, -0.1, 1.1, 1.1
    problem = Problem(net, X, Y)

    @testset "Neurify" begin
        solver = Neurify()
        result = solve(solver, problem)
        @test result.status == :holds
    end

    @testset "ReluVal" begin
        solver = ReluVal()
        result = solve(solver, problem)
        @test result.status == :holds
    end

end