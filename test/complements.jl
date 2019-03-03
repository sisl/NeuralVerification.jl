
@testset "PolytopeComplements" begin

    @testset "Basic" begin

        HS = HalfSpace([-1.0, -1.0], 0.0)
        PC = PolytopeComplement(HS)

        @test [5.0, 5.0] ∈ HS && [5.0, 5.0] ∉ PC
        @test [-1.0, -1.0]   ∉ HS && [-1.0, -1.0]   ∈ PC

        @test complement(PC) === HS
        @test complement(HS) === PC

        # Hyperrectangle contained in HS
        hr = Hyperrectangle(low = [1.0, 1.0], high = [2.0, 2.0])

        @show vertices_list(hr)

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
        @test @no_error complement(Hyperrectangle(ones(2), ones(2)))
        @test @no_error complement(Ball2(ones(2), 1.0))
        @test @no_error complement(Ball1(ones(3), 1.0))
        @test @no_error complement(Zonotope(ones(4), ones(4, 2)))
        @test @no_error complement(convert(HPolytope, hr))
    end
end