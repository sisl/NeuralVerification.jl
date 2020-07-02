# Test by creating a network, writing it, then reading it back in and make sure that all weights and biases match
@testset "Read and write set test" begin
    # Create a HalfSpace, Hyperrectangle, Hyperpolytope, Polytope, PolytopeComplement, and Zonotope
    # and make sure they are written and read correctly

    # Create the objects
    half_space = HalfSpace([1, 2, 3], 4)
    hyperrectangle = Hyperrectangle([1, 2, 3], [4, 5, 6])
    polytope = HPolytope([1 2 3; 4 5 6; 7 8 9; 10 11 12], [1; 2; 3; 4])
    polytope_complement = PolytopeComplement(polytope)
    zonotope = Zonotope([1; 2], [1 2; 3 4])

    # Give them each filenames
    half_space_file = string(tempname(), ".txt")
    hyperrectangle_file = string(tempname(), ".txt")
    polytope_file = string(tempname(), ".txt")
    polytope_complement_file = string(tempname(), ".txt")
    zonotope_file = string(tempname(), ".txt")

    # Write all to file
    write_set(half_space_file, half_space)
    write_set(hyperrectangle_file, hyperrectangle)
    write_set(polytope_file, polytope)
    write_set(polytope_complement_file, polytope_complement)
    write_set(zonotope_file, zonotope)

    # Read back in from file
    half_space_in = read_set(half_space_file)
    hyperrectangle_in = read_set(hyperrectangle_file)
    polytope_in = read_set(polytope_file)
    polytope_complement_in = read_set(polytope_complement_file)
    zonotope_in = read_set(zonotope_file)

    # Test that they are equal
    @test half_space == half_space_in
    @test hyperrectangle == hyperrectangle_in
    @test polytope == polytope_in
    @test polytope_complement == polytope_complement_in
    @test zonotope == zonotope_in
end
