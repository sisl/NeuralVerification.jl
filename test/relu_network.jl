
@testset "ReLU Last Layer" begin

    small_nnet_file = net_path * "small_nnet.nnet"
    # small_nnet encodes the simple function 24*max(x + 1.5, 0) + 18.5
    small_nnet = read_nnet(small_nnet_file, last_layer_activation = ReLU())

    # The input set is [-0.9:0.9]
    in_hyper  = Hyperrectangle(low = [-0.9], high = [0.9])
    in_hpoly  = convert(HPolytope, in_hyper)

    # Output region is entirely contained in this interval:
    out_superset    = Hyperrectangle(low = [30.0], high = [80.0])    # 20.0 ≤ y ≤ 90.0
    # Includes some points in the output region but not all:
    out_overlapping = Hyperrectangle(low = [-1.0], high = [50.0])    # -1.0 ≤ y ≤ 50.0

    @testset "Group 1" begin
        problem_holds    = Problem(small_nnet, in_hpoly, convert(HPolytope, out_superset))
        problem_violated = Problem(small_nnet, in_hpoly, convert(HPolytope, out_overlapping))

        for solver in [MaxSens(resolution = 0.6), ExactReach(), Ai2(), Ai2h(), Box()]
            holds    = solve(solver, problem_holds)
            violated = solve(solver, problem_violated)

            @testset "$(typeof(solver))" begin
                @test holds.status    ∈ (:holds, :unknown)
                @test violated.status ∈ (:violated, :unknown)
            end
        end

    end

    @testset "Group 2, 3, 4, 6" begin
        # halfspace only constraints:
        problem_holds    = Problem(small_nnet, in_hyper, HPolytope([HalfSpace([1.], 100.)]))     # y < 100.0
        problem_violated = Problem(small_nnet, in_hyper, HPolytope([HalfSpace([1.], 10.)]))      # y < 10.0

        group2 = [NSVerify(), MIPVerify(), ILP()]
        group3 = [Duality()] # hypothetically also ConvDual
        group4 = [FastLin(), FastLip()]
        group6 = [Reluplex(), Planet()]

        for solver in [group2; group3; group4; group6]
            holds    = solve(solver, problem_holds)
            violated = solve(solver, problem_violated)

            @testset "$(typeof(solver))" begin
                @test holds.status    ∈ (:holds, :unknown)
                @test violated.status ∈ (:violated, :unknown)
            end
        end

        # ConvDual can not handle ReLU networks at present.
        # We should ignore the result even if this particular network is trivial
        holds    = solve(ConvDual(), problem_holds)
        violated = solve(ConvDual(), problem_violated)
        @test_skip holds.status    ∈ (:holds, :unknown)
        @test_skip violated.status ∈ (:violated, :unknown)
    end

    @testset "Group 5" begin
        problem_holds    = Problem(small_nnet, in_hyper, out_superset)
        problem_violated = Problem(small_nnet, in_hyper, out_overlapping)

        for solver in [ReluVal(max_iter = 10), DLV(), Sherlock(ϵ = 0.5), BaB(), Neurify()]
            holds    = solve(solver, problem_holds)
            violated = solve(solver, problem_violated)

            @testset "$(typeof(solver))" begin
                @test holds.status    ∈ (:holds, :unknown)
                @test violated.status ∈ (:violated, :unknown)
            end
        end
    end

    @testset "Certify" begin
        ### Certify - only works for single hidden layer
        tiny_nnet = read_nnet(net_path * "tiny_nnet.nnet")
        solver_certify = Certify()
        in_set  = Hyperrectangle([2.0], [.5])
        out_set = HPolytope(ones(1,1), [2.5])
        problem_certify = Problem(tiny_nnet, in_set, out_set)
        @test @no_error solve(solver_certify, problem_certify)
    end
end
