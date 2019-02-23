
@testset "Id Last Layer" begin

    small_nnet_file = "$(@__DIR__)/../examples/networks/small_nnet_id.nnet"
    small_nnet = read_nnet(small_nnet_file, last_layer_activation = Id())

    # The input set is always in ∈[-1:1]
    input_hyper  = Hyperrectangle(low = [-0.9], high = [0.9])
    input_hpoly  = convert(HPolytope, input_hyper)

    out_hyper_70_23  = Hyperrectangle(low = [-70.0], high = [-23.0]) # superset of the output
    out_hyper_100_45 = Hyperrectangle(low = [-100.0], high = [-45.0]) # includes points in the output region

    problem_sat_hyper_hyper           = Problem(small_nnet, input_hyper, out_hyper_70_23)                          # 40.0 < y < 60.0
    problem_unsat_hyper_hyper         = Problem(small_nnet, input_hyper, out_hyper_100_45)                         # -1.0 < y < 50.0
    problem_sat_hpoly_hpoly_bounded   = Problem(small_nnet, input_hpoly, convert(HPolytope, out_hyper_70_23))
    problem_unsat_hpoly_hpoly_bounded = Problem(small_nnet, input_hpoly, convert(HPolytope, out_hyper_100_45))
    # halfspace constraints:
    problem_sat_hyper_hs              = Problem(small_nnet, input_hyper, HPolytope([HalfSpace([1.], -10.0)]))      # y < -10.0
    problem_unsat_hyper_hs            = Problem(small_nnet, input_hyper, HPolytope([HalfSpace([-1.], -20.0)]))     # y > 20.0

    @testset "Group 1" begin
        for solver in [MaxSens(resolution = 0.6), ExactReach(), Ai2()]
            sat   = solve(solver, problem_sat_hpoly_hpoly_bounded)
            unsat = solve(solver, problem_unsat_hpoly_hpoly_bounded)

            @test sat.status ∈ (:SAT, :Unknown)
            @test unsat.status ∈ (:UNSAT, :Unknown)
        end

    end

    @testset "Group 2, 3, 4, 6" begin
        glpk = GLPKSolverMIP()

        group2 = [S(optimizer = glpk) for S in (NSVerify, MIPVerify, ILP)]
        group3 = [ConvDual(), Duality(optimizer = glpk)]
        group4 = [FastLin(), FastLip()]
        group6 = [Reluplex(), Planet()]

        for solver in [group2; group3; group4; group6]
            sat   = solve(solver, problem_sat_hyper_hs)
            unsat = solve(solver, problem_unsat_hyper_hs)

            @test sat.status ∈ (:SAT, :Unknown)
            @test unsat.status ∈ (:UNSAT, :Unknown)
        end
    end

    @testset "Group 5" begin
        for solver in [ReluVal(max_iter = 10), DLV(), Sherlock(glpk, 0.5), BaB(optimizer = glpk)]
            sat   = solve(solver, problem_sat_hyper_hyper)
            unsat = solve(solver, problem_unsat_hyper_hyper)

            @test sat.status ∈ (:SAT, :Unknown)
            @test unsat.status ∈ (:UNSAT, :Unknown)
        end
    end

    @testset "Certify" begin
        ### Certify - only works for single hidden layer
        tiny_nnet = read_nnet("$(@__DIR__)/../examples/networks/tiny_nnet.nnet")
        solver_certify = Certify()
        inputSet  = Hyperrectangle([2.0], [.5])
        outputSet = HPolytope(ones(1,1), [2.5])
        problem_certify = Problem(tiny_nnet, inputSet, outputSet)
        @test @no_error solve(solver_certify, problem_certify)
    end
end