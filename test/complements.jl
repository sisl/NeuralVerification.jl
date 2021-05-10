
@testset "PolytopeComplements" begin

    @testset "Basic" begin

        HS = HalfSpace([-1.0, -1.0], 0.0)
        PC = Complement(HS)

        @test [5.0, 5.0]   ∈ HS && [5.0, 5.0]   ∉ PC
        @test [-1.0, -1.0] ∉ HS && [-1.0, -1.0] ∈ PC

        @test Complement(PC) == HS
        # @test Complement(HS) == PC

        # Hyperrectangle contained in HS
        hr = Hyperrectangle(low = [1.0, 1.0], high = [2.0, 2.0])


        @test is_intersection_empty(hr, PC) == true
        @test is_intersection_empty(hr, HS) == false
        @test (hr ⊆ PC) == false
        @test (hr ⊆ HS) == true
        # Hyperrectangle overlapping with HS
        hr = Hyperrectangle(low = [-1.0, -1.0], high = [1.0, 1.0])
        @test is_intersection_empty(hr, PC) == false
        @test is_intersection_empty(hr, HS) == false
        @test !(hr ⊆ PC)
        @test !(hr ⊆ HS)

        # Test some other sets
        @test @no_error Complement(Hyperrectangle(ones(2), ones(2)))
        @test @no_error Complement(Ball2(ones(2), 1.0))
        @test @no_error Complement(Ball1(ones(3), 1.0))
        @test @no_error Complement(Zonotope(ones(4), ones(4, 2)))
        @test @no_error Complement(convert(HPolytope, hr))
    end

    @testset "Solvers with PCs" begin

        small_nnet = read_nnet(net_path * "small_nnet_id.nnet")
        in_hyper  = Hyperrectangle(low = [-0.9], high = [0.9])

        # Output sets that are the PolytopeComplements of the complements of the output sets used in the regular tests.
        problem_holds    = Problem(small_nnet, in_hyper, Complement(HPolytope([HalfSpace([-1.0], 10.0)])))
        problem_violated = Problem(small_nnet, in_hyper, Complement(HPolytope([HalfSpace([1.0], -20.0)])))

        for solver in [NSVerify(), MIPVerify(), ILP(), Reluplex(), Planet()]
            holds    = solve(solver, problem_holds)
            violated = solve(solver, problem_violated)

            @testset "$(typeof(solver))" begin
                @test holds.status    ∈ (:holds, :unknown)
                @test violated.status ∈ (:violated, :unknown)
            end
        end
    end
end