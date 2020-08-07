using Random
using LinearAlgebra
using Cbc
using Test

"""
    make_random_network(layer_sizes::Vector{Int}, [min_weight = -1.0], [max_weight = 1.0], [min_bias = -1.0], [max_bias = 1.0], [rng = 1.0])
    read_layer(output_dim::Int, f::IOStream, [act = ReLU()])

Generate a network with random weights and bias. The first layer is treated as the input.
The values for the weights and bias will be uniformly drawn from the range between min_weight
and max_weight and min_bias and max_bias respectively. The last layer will have an ID()
activation function and the rest will have ReLU() activation functions. Allow a random number
generator(rng) to be passed in. This allows for seeded random network generation.
"""
function make_random_network(layer_sizes::Vector{Int}, min_weight = -1.0, max_weight = 1.0, min_bias = -1.0, max_bias = 1.0, rng=MersenneTwister())
    # Create each layer based on the layer_size
    layers = []
    for index in 1:(length(layer_sizes)-1)
        cur_size = layer_sizes[index]
        next_size = layer_sizes[index+1]
        # Use Id activation for the last layer - otherwise use ReLU activation
        if index == (length(layer_sizes)-1)
            cur_activation = NeuralVerification.Id()
        else
            cur_activation = NeuralVerification.ReLU()
        end

        # Dimension: num_out x num_in
        cur_weights = min_weight .+ (max_weight - min_weight) * rand(rng, Float64, (next_size, cur_size))
        cur_weights = reshape(cur_weights, (next_size, cur_size)) # for edge case where 1 dimension is equal to 1 this keeps it from being a 1-d vector

        # Dimension: num_out x 1
        cur_bias = min_bias .+ (max_bias - min_bias) * rand(rng, Float64, (next_size))
        push!(layers, NeuralVerification.Layer(cur_weights, cur_bias, cur_activation))
    end

    return Network(layers)
end


"""
    function make_random_query_file(num_networks_for_size::Array{Int, 1},
                                layer_sizes::Array{Array{Int, 1}},
                                network_dir,
                                input_set_dir,
                                output_set_dir,
                                query_file
                                ; [min_weight = -1.0],
                                [max_weight = 1.0],
                                [min_bias = -1.0],
                                [max_bias = 1.0],
                                network_files=[],
                                [rng=MersenneTwister()])


Generates a random query file. num_networks_for_size gives the number of networks
to make for each shape. The network_dir, input_set_dir, output_set_dir, and query_file
variables give the location to write out the files - these will be written relative
to where the function is called from. These relative paths will be stored in the query_file.

An example call would look like:
NeuralVerification.make_random_query_file([3, 3], [[1, 3, 1], [2, 5, 2]], "test_rand/networks", "test_rand/input_sets", "test_rand/output_sets", "test_rand/query_file.txt")

Which will make 3 networks with shape [1, 3, 1] and 3 networks with shape [2, 5, 2].

If network_files is non-empty, it should have len(num_networks_for_size) network files
with the corresponding shapes. Instead of generating random networks, we will use
these networks as we make our queries. It will use the same network multiple times if
an entry of num_networks_for_size is larger than 1.
"""
function make_random_query_file(num_networks_for_size::Array{Int, 1},
                                layer_sizes::Array{Array{Int, 1}},
                                network_dir,
                                input_set_dir,
                                output_set_dir,
                                query_file
                                ; min_weight = -1.0,
                                max_weight = 1.0,
                                min_bias = -1.0,
                                max_bias = 1.0,
                                network_files=[],
                                rng=MersenneTwister()
                                )

    for (shape_index, (num_networks, layer_sizes)) in enumerate(zip(num_networks_for_size, layer_sizes))
        for i = 1:num_networks
            # Create a random network
            if (length(network_files)) == 0
                network = make_random_network(layer_sizes, min_weight, max_weight, min_bias, max_bias, rng)
            else
                network = read_nnet(network_files[i])
            end
            num_inputs = layer_sizes[1]
            num_outputs = layer_sizes[end]

            # Create several input sets and a variety of corresponding output sets.
            halfspace_output = HalfSpace(rand(num_outputs) .- 0.5, rand() - 0.5)

            hyperrectangle_input_center = rand(num_inputs) .- 0.5
            hyperrectangle_input_radius = rand(num_inputs)
            hyperrectangle_input = Hyperrectangle(hyperrectangle_input_center, hyperrectangle_input_radius)

            hyperrcube_input_radius = rand() * ones(num_inputs)
            hypercube_input = Hyperrectangle(hyperrectangle_input_center, hyperrcube_input_radius)

            hyperrectangle_output_center = rand(num_outputs) .- 0.5
            hyperrectangle_output_radius = rand(num_outputs)
            hyperrectangle_output = Hyperrectangle(hyperrectangle_output_center, hyperrectangle_output_radius)

            hpolytope_in_one  = convert(HPolytope, hyperrectangle_input)
            hpolytope_out_one = convert(HPolytope, hyperrectangle_output)

            # Generate two orthogonal matrices which will be used to rotate the hyperrectangle
            # to create a rotated polytope
            rand_in = rand(num_inputs, num_inputs)
            rand_out = rand(num_outputs, num_outputs)
            Q_in, R_in = qr(rand_in)
            Q_out, R_out = qr(rand_out)

            A_in, b_in = tosimplehrep(hyperrectangle_input)
            A_out, b_out = tosimplehrep(hyperrectangle_output)
            hpolytope_in_two = HPolytope(A_in * Q_in, b_in)
            hpolytope_out_two = HPolytope(A_out * Q_out, b_out)

            # Create polytope complements from the two created polytopes
            polytopecomplement_in_one = PolytopeComplement(hpolytope_in_one)
            polytopecomplement_out_one = PolytopeComplement(hpolytope_out_one)

            polytopecomplement_in_two = PolytopeComplement(hpolytope_in_two)
            polytopecomplement_out_two = PolytopeComplement(hpolytope_out_two)

            # Ignore zonotopes for now

            # Create a problem with HP/HP, HR/PC, HR(uniform)/HS, HR/HS, HR/HR
            hp_hp_problem_one = Problem(network, hpolytope_in_one, hpolytope_out_one)
            hp_hp_problem_two = Problem(network, hpolytope_in_two, hpolytope_out_two)
            hr_pc_problem_one = Problem(network, hyperrectangle_input, polytopecomplement_out_one)
            hr_pc_problem_two = Problem(network, hyperrectangle_input, polytopecomplement_out_two)
            hr_uniform_hs_problem = Problem(network, hypercube_input, halfspace_output)
            hr_hs_problem = Problem(network, hyperrectangle_input, halfspace_output)
            hr_hr_problem = Problem(network, hyperrectangle_input, hyperrectangle_output)

            problems = [hp_hp_problem_one, hp_hp_problem_two, hr_pc_problem_one, hr_pc_problem_two, hr_uniform_hs_problem, hr_hs_problem, hr_hr_problem]
            identifier_string = replace(string(layer_sizes, "_", i), " "=>"") # remove all spaces

            base_filenames = string.(["hp_hp_one_", "hp_hp_two_", "hr_pc_one_", "hr_pc_two_", "hr_uniform_hs", "hr_hs_", "hr_hr_"], identifier_string)
            network_filenames = joinpath.(network_dir, fill("rand_"*identifier_string*".nnet", length(problems))) # all use the same network
            input_filenames = joinpath.(input_set_dir, base_filenames.*"_inputset.json")
            output_filenames = joinpath.(output_set_dir, base_filenames.*"_outputset.json")

            write_problem.(network_filenames, input_filenames, output_filenames, problems; query_file=query_file)
        end

    end
end


"""
    function make_random_test_set()

A function that generates the set of random query files to be used in testing.
We generate three different test sets - small, medium, and large.
"""
function make_random_test_sets()
    # Tiny test set for Ai2h
    NeuralVerification.make_random_query_file([3, 3],
                                                [[1, 2, 1], [1, 3, 2, 1]],
                                                "test/test_sets/random/tiny/networks",
                                                "test/test_sets/random/tiny/input_sets",
                                                "test/test_sets/random/tiny/output_sets",
                                                "test/test_sets/random/tiny/query_file_tiny.txt")


    # Small, medium, and large should be tractable for all solvers except Ai2h
    NeuralVerification.make_random_query_file([3, 3],
                                              [[1, 3, 1], [2, 5, 2]],
                                              "test/test_sets/random/small/networks",
                                              "test/test_sets/random/small/input_sets",
                                              "test/test_sets/random/small/output_sets",
                                              "test/test_sets/random/small/query_file_small.txt")

    NeuralVerification.make_random_query_file([5, 5, 5],
                                              [[1, 8, 1], [4, 8, 4], [1, 10, 4, 1]],
                                              "test/test_sets/random/medium/networks",
                                              "test/test_sets/random/medium/input_sets",
                                              "test/test_sets/random/medium/output_sets",
                                              "test/test_sets/random/medium/query_file_medium.txt")
end

function write_to_query_file(network_file::String, input_file::String, output_file::String, query_file::String)
    mkpath(dirname(query_file))
    open(query_file, "a") do f
        println(f, string(network_file, " ", input_file, " ", output_file))
    end
end

"""
    write_problem(network_file::String, input_file::String, output_file::String, problem::Problem; [query_file = ""])

Write a the information from a problem to files. The input set, output set, and network will
each be written to a separate file. If a query file is given it will append this test to the corresponding query file
"""
function write_problem(network_file::String, input_file::String, output_file::String, problem::Problem; query_file="")
    write_nnet(network_file, problem.network)
    write_set(input_file, problem.input)
    write_set(output_file, problem.output)

    # If a query file is given, append this problem to the file by writing out
    # the network file, input file, and output file names
    if query_file != ""
        write_to_query_file(network_file, input_file, output_file, query_file)
    end
end


"""
    get_set_type_string(set::Union{PolytopeComplement, LazySet})

Return a string corresponding to the object's type. This will be used to store
these sets to file.
"""
function get_set_type_string(set::Union{PolytopeComplement, LazySet})
    if set isa HalfSpace
        return "HalfSpace"
    elseif set isa Hyperrectangle
        return "Hyperrectangle"
    elseif set isa HPolytope
        return "HPolytope"
    elseif set isa PolytopeComplement
        return "PolytopeComplement"
    elseif set isa Zonotope
        return "Zonotope"
    else
        return ""
    end
end

"""
    write_set(filename::String, set::Union{PolytopeComplement, LazySet})

Write a set in JSON format to a file along with its type. This supports HalfSpace, Hyperrectangle,
HPolytope, PolytopeComplement, and Zonotope objects.
"""
function write_set(filename::String, set::Union{PolytopeComplement, LazySet})
    # Only support hyperrectangle, hyperpolytope, zonotope, polytope complement, and halfspace
    type_string = get_set_type_string(set)
    @assert type_string != ""

    # Save the type and the object itself
    output_dict = Dict()
    output_dict["type"] = type_string
    output_dict["set"] = set

    # Write to file
    mkpath(dirname(filename))
    open(filename, "w") do f
        JSON2.write(f, output_dict)
    end
end


"""
    read_problem(network_file::String, input_file::String, output_file::String)

Take in a network, input set, and output set and return the corresponding problem.
"""
function read_problem(network_file::String, input_file::String, output_file::String)
    network = read_nnet(network_file)
    input_set = read_set(input_file)
    output_set = read_set(output_file)
    return Problem(network, input_set, output_set)
end

"""
    read_set(filename::String)

Read in a set from a file. This supports HalfSpace, Hyperrectangle,
HPolytope, PolytopeComplement, and Zonotope objects.
"""
function read_set(filename::String)
    json_string = read(filename, String)
    dict = JSON2.read(json_string)
    type = dict[:type]
    set = dict[:set]

    # helper function to convert from Any arrays to Float arrays
    function convert_float(elem)
        if elem isa Array
            return convert(Array{Float64}, elem)
        else
            return convert(Float64, elem)
        end
    end

    # Based on the type we interpret the stored JSON data and creates the corresponding object
    if type == "HalfSpace"
        return HalfSpace(convert_float(set[:a]), convert_float(set[:b]))
    elseif type == "Hyperrectangle"
        return Hyperrectangle(convert_float(set[:center]), convert_float(set[:radius]))
    elseif type == "HPolytope"
        constraints = set[:constraints]
        half_spaces = [HalfSpace(convert_float(constraints[i][:a]), convert_float(constraints[i][:b])) for i = 1:length(constraints)]
        return HPolytope(half_spaces)
    elseif type == "PolytopeComplement"
        constraints = set[:P][:constraints]
        half_spaces = [HalfSpace(convert_float(constraints[i][:a]), convert_float(constraints[i][:b])) for i = 1:length(constraints)]
        return PolytopeComplement(HPolytope(half_spaces))
    elseif type == "Zonotope"
        center = set[:center]
        generator_list = set[:generators]
        generator_matrix = reshape(generator_list, (length(center), :)) # Check the reshape here
        return Zonotope(convert_float(center), convert_float(generator_matrix))
    else
        @assert false
    end
end

"""
function query_line_to_problem(line::String; [base_dir=""])

Take in a line from a query file and read in then return the corresponding problem.
This line will be in the format:
some/path/network.nnet other/path/input_set.json third/path/output_set.json

It splits it apart by spaces then reads in the problem.
"""
function query_line_to_problem(line::String; base_dir="")
    network_file, input_file, output_file = string.(split(line, " "))
    return read_problem(joinpath(base_dir, network_file), joinpath(base_dir, input_file), joinpath(base_dir, output_file))
end

"""
    get_valid_solvers(problem::Problem, solvers = get_all_solvers_to_test(); [solver_types_allowed = []], [solver_types_to_remove = []])

Return a list of valid instantitated solvers for a particular problem. This will depend
on compatibility with the input set, output set, and network.

If a list of solvers is given, it filters that list. Otherwise, it gets a default list
from get_all_solvers_to_test() and uses that list of solvers.

solver_types_allowed and solver_types_to_remove filters the list of solvers further
"""
function get_valid_solvers(problem::Problem; solvers = [], solver_types_allowed=[], solver_types_to_remove=[])
    # Fill the list of solvers if it isn't given, then filter based on solver_types_allowed and solver_types_to_remove
    if (length(solvers) == 0)
        solvers = get_all_solvers_to_test()
    end
    solvers = allow_and_remove_solvers(solvers, solver_types_allowed, solver_types_to_remove)

    # Filter out solvers that don't work with the input set, output set, or network
    return filter(solver->solver_works_with_input_set(solver, problem.input)
            && solver_works_with_output_set(solver, problem.output)
            && solver_works_with_network(solver, problem.network), solvers)
end

"""
    solver_works_with_input_set(solver, input_set)

Return true if the solver can handle the input given by input_set.
Return false otherwise.
"""
function solver_works_with_input_set(solver, input_set)
    # Define which solvers work with which input sets
    half_space_solver_types = Union{}
    hyperrectangle_solver_types = Union{NSVerify, MIPVerify, ILP, Duality, ConvDual, Certify, FastLin, FastLip, ReluVal, DLV, Sherlock, BaB, Planet, Reluplex, ExactReach, Ai2, MaxSens}
    hyperpolytope_solver_types = Union{ExactReach, Ai2, MaxSens}
    polytope_complement_solver_types = Union{}
    zonotope_solver_types = Union{}

    # Check to see for each supported input set which solvers work with it
    if input_set isa HalfSpace
        return solver isa half_space_solver_types
    elseif input_set isa Hyperrectangle
        # Duality and ConvDual require uniform HR (hypercube) input set
        if (solver isa Union{Duality, ConvDual})
            return all(iszero.(input_set.radius .- input_set.radius[1]))
        else
            return solver isa hyperrectangle_solver_types
        end
    elseif input_set isa HPolytope
        return solver isa hyperpolytope_solver_types
    elseif input_set isa PolytopeComplement
        return solver isa polytope_complement_solver_types
    elseif input_set isa Zonotope
        return solver isa zonotope_solver_types
    else
        @assert false "Unsupported input set"
    end
end

"""
    solver_works_with_output_set(solver, output_set)

Return true if the solver can handle the output given by output_set.
Return false otherwise.
"""
function solver_works_with_output_set(solver, output_set)
    half_space_solver_types = Union{Duality, ConvDual, Certify, FastLin, FastLip, NSVerify, MIPVerify, ILP, Planet, Reluplex}
    hyperrectangle_solver_types = Union{ReluVal, DLV, Sherlock, BaB, ExactReach, Ai2, MaxSens}
    hyperpolytope_solver_types = Union{ExactReach, Ai2, MaxSens}
    polytope_complement_solver_types = Union{NSVerify, MIPVerify, ILP, Planet, Reluplex}
    zonotope_solver_types = Union{}

    if output_set isa HalfSpace
        return solver isa half_space_solver_types
    elseif output_set isa Hyperrectangle
        # For DLV, Sherlock, and BaB it must be 1-D
        if (solver isa Union{DLV, Sherlock, BaB})
            return length(output_set.center) == 1
        else
            return solver isa hyperrectangle_solver_types
        end
    elseif output_set isa HPolytope
        # We will assume that the polytope is bounded
        return solver isa hyperpolytope_solver_types
    elseif output_set isa PolytopeComplement
        return solver isa polytope_complement_solver_types
    elseif output_set isa Zonotope
        return solver isa zonotope_solver_types
    else
        @assert false "Unsupported output set"
    end
end

"""
    needs_polytope_input(solver)

    Returns true if the solver requires a polytope.
    For example, Ai2 can handle hyperrectangle input
    only if it is converted to a hyperpolytope first.
"""
function needs_polytope_input(solver)
    return solver isa Union{Ai2, MaxSens}
end
"""
    needs_polytope_output(solver)

    Returns true if the solver requires a polytope.
    For example, Ai2 can handle hyperrectangle output
    only if it is converted to a hyperpolytope first.
"""
function needs_polytope_output(solver)
    return solver isa Union{Ai2, MaxSens}
end

"""
    solver_work_with_network(solver, network)

Return true if the solver can handle the network given by network.
Return false otherwise.
"""
function solver_works_with_network(solver, network)
    if (solver isa Certify)
        return length(network.layers) == 2 # certify only works with 1 hidden layer
    else
        return true
    end
end


"""
    is_complete(solver)

Return whether a solver is complete or not.
"""
function is_complete(solver)
    complete_solvers = Union{ExactReach, NSVerify, MIPVerify, ReluVal, Planet, Reluplex}
    return solver isa complete_solvers
end

"""
    get_all_solvers_to_test()

Return all solver configurations that we'd like to test
"""
function get_all_solvers_to_test()
    return [
            ExactReach(),
            Ai2z(),
            Box(),
            MaxSens(resolution = 0.6),
            MaxSens(resolution = 0.3),
            NSVerify(optimizer=Cbc.Optimizer),
            MIPVerify(optimizer=Cbc.Optimizer),
            ILP(optimizer=Cbc.Optimizer),
            Duality(optimizer=Cbc.Optimizer),
            ConvDual(),
            Certify(),
            FastLin(),
            FastLip(),
            ReluVal(max_iter = 500),
            DLV(optimizer=Cbc.Optimizer),
            Sherlock(ϵ = 0.5, optimizer=Cbc.Optimizer),
            Sherlock(ϵ = 0.1, optimizer=Cbc.Optimizer),
            BaB(optimizer=Cbc.Optimizer),
            Planet(optimizer=Cbc.Optimizer),
            Reluplex(optimizer=Cbc.Optimizer)
            ]
end

"""
    allow_and_remove_solvers(solvers, solver_types_allowed, solver_types_to_remove)

Takes a list of solvers. If solver_types_allowed is non-empty then it only allows those types.
If solver_types_to_remove is non-empty then it filters the list of solvers and removes all
types in solver_types_to_remove.
"""
function allow_and_remove_solvers(solvers, solver_types_allowed, solver_types_to_remove)
    filtered_list = deepcopy(solvers)
    # Filter the solvers if types allowed or removed are given (versus default)
    if (length(solver_types_allowed) >= 1)
        filtered_list = filter(solver->solver isa Union{solver_types_allowed...}, filtered_list)
    end
    if (length(solver_types_to_remove) >= 1)
        filtered_list = filter(solver->!(solver isa Union{solver_types_to_remove...}), filtered_list)
    end
    return filtered_list
end



#=
    Utils for running the tests themselves
=#

"""
test_query_file(file_name::String; [solvers = []], [solver_types_allowed = []], [solver_types_to_remove =[]], [solver_types_to_report=[]])

    Run a set of benchmarks on a query file given by file_name.
    Compares results for consistency, but has no ground truth to compare against.

    Will filter the set of available solvers by type using solver_types_allowed ans solver_types_to_remove.
    Will use a default list unless a solvers list is given
    Solver types to report tells which tests to report back for easier interpretation of the results if you're
    just debugginig a certain solver

"""

function test_query_file(file_name::String; all_solvers = [], solver_types_allowed=[], solver_types_to_remove=[], solver_types_to_report=[])
    path, file = splitdir(file_name)
    @testset "Correctness Tests on $(file)" begin
        queries = readlines(file_name)
        for (index, line) in enumerate(queries)
            println("Testing on line $(index): ", line)
            @testset "Test on line: $index" begin
                problem = NeuralVerification.query_line_to_problem(line; base_dir="$(@__DIR__)/../../")
                solvers = NeuralVerification.get_valid_solvers(problem; solvers=all_solvers, solver_types_allowed=solver_types_allowed, solver_types_to_remove=solver_types_to_remove=solver_types_to_remove)
                if (!any([solver isa Union{solver_types_to_report...} for solver in solvers]))
                    @warn "Skipping line because no solver in solvers_to_report will be used"
                    continue # If none of the solvers we're interested in are going to be run, then skip solving on this line
                end
                # Solve the problem with each solver that applies for the problem, then compare the results
                results = Vector(undef, length(solvers))
                for (solver_index, solver) in enumerate(solvers)
                    cur_problem = deepcopy(problem)
                    # Workaround to have ConvDual and FastLip take in halfspace output sets as expected
                    if ((solver isa NeuralVerification.ConvDual || solver isa NeuralVerification.FastLip)) && cur_problem.output isa NeuralVerification.HalfSpace
                        cur_problem = NeuralVerification.Problem(cur_problem.network, cur_problem.input, NeuralVerification.HPolytope([cur_problem.output])) # convert to a HPolytope b/c ConvDual takes in a HalfSpace as a HPolytope for now
                    end

                    # Workaround to have ExactReach, Ai2, and MaxSens take in HR inputs and outputs
                    if (NeuralVerification.needs_polytope_input(solver) && cur_problem.input isa NeuralVerification.Hyperrectangle)
                        cur_problem = NeuralVerification.Problem(cur_problem.network, LazySets.convert(HPolytope, cur_problem.input), cur_problem.output)
                    end
                    if (NeuralVerification.needs_polytope_output(solver) && cur_problem.output isa NeuralVerification.Hyperrectangle)
                        cur_problem = NeuralVerification.Problem(cur_problem.network, cur_problem.input, LazySets.convert(HPolytope, cur_problem.output))
                    end

                    # Try-catch while solving to handle a GLPK bug by attempting to run with Gurobi instead when this bug shows up
                    try
                        results[solver_index] = NeuralVerification.solve(solver, cur_problem)
                    catch e
                        if e isa GLPK.GLPKError && e.msg == "invalid GLPK.Prob"
                            @warn "Caught GLPK error on $(typeof(solver))"
                            results[solver_index] = NeuralVerification.CounterExampleResult(:unknown) # Known issue with GLPK so ignore this error and just push an unknown result
                        else
                            # Print the error and make sure we still get its stack
                            for (exc, bt) in Base.catch_stack()
                               showerror(stdout, exc, bt)
                               println()
                            end
                            throw(e)
                        end
                    end
                end

                # Just sees if each pair agrees
                # if one or both return unknown then we can't make a comparison
                tested_line = false

                for (i, j) in [(i, j) for i = 1:length(solvers) for j = (i+1):length(solvers)]
                    if (length(solver_types_to_report) == 0) || (solvers[i] isa Union{solver_types_to_report...}) || (solvers[j] isa Union{solver_types_to_report...})
                        @testset "Comparing $(typeof(solvers[i])) with $(typeof(solvers[j]))" begin
                            solver_one_complete = NeuralVerification.is_complete(solvers[i])
                            solver_two_complete = NeuralVerification.is_complete(solvers[j])
                            # Both complete
                            if (solver_one_complete && solver_two_complete)
                                tested_line = true
                                # Results match
                                @test results[i].status == results[j].status
                            # Solver one complete, solver two incomplete
                            elseif (solver_one_complete && !solver_two_complete)
                                tested_line = true
                                # Results match or solver two unknown or solver one holds solver two violated (bc incomplete)
                                @test ((results[i].status == results[j].status) || (results[j].status == :unknown) || (results[i].status == :holds && results[j].status == :violated))
                            # Solver one incomplete, solver two complete
                            elseif (!solver_one_complete && solver_two_complete)
                                tested_line = true
                                # Results match or solver one unknown or solver two holds solver one violated (bc incomplete)
                                @test ((results[i].status == results[j].status) || (results[i].status == :unknown) || ((results[i].status == :violated) && (results[j].status == :holds)))
                            # Neither are complete
                            else
                                if (results[i].status != results[j].status)
                                    @warn "Neither solver complete but results disagree: $(typeof(solvers[i])) vs. $(typeof(solvers[j])) is $(results[i].status) vs. $(results[j].status)"
                                end
                                # Results match or solver one unknown or solver two unknown or
                                # no test since any mix of outcomes could be justified with two incomplete ones
                            end
                        end
                    end
                end
                if (!tested_line)
                    @warn "Didn't test line $(index): $(line)"
                end
            end
        end
    end
end


"""
test_solvers(; solver_types=[], test_set="small", solvers=[])

    A wrapper around test_query_file that makes it easier to specify the test set.
    test_set can be: "tiny", "small", "medium", or "previous_issues". Typical usage looks like:

    test_solvers(solver_types_to_report=[ExactReach, ReluVal], test_set="small")
    or
    test_solvers(test_set="small")
    or if you'd like to not run a solver at all you can specify this as follows:
    test_solvers(test_set="small", solver_types_to_remove=[ExactReach, Reluplex])


    test_set: tiny, small, medium, or previous_issues. Describes which set of tests to run.
    all _solvers: An optional list of the solver instances that you'd like to use. This can be used to have control
        over the hyperparameters that each solver is instantiated with.
    solver_types_allowed: A list of solver types. If empty, all are allowed. This will filter the solvers that tests are run on to only include solvers whose type is in this list.
    solver_types_to_remove: A List of solver types. If empty, none are removed. This will filter the solvers that the tests are run on to not include any solvers whose type is in thsi list.
        Helpful if you are trying to run queries where a certain solver would be slow, or give frequent errors and you want to ignore those.
    solver_types_to_report: A list of solver types to report test results on. This can be helpful to have a smaller list of results to look through.
"""
function test_correctness(;test_set="small", all_solvers = [], solver_types_allowed=[], solver_types_to_remove=[], solver_types_to_report=[])
    test_set_to_file_name= Dict("tiny" => "$(@__DIR__)/../../test/test_sets/random/tiny/query_file_tiny.txt",
                                "small" => "$(@__DIR__)/../../test/test_sets/random/small/query_file_small.txt",
                                "medium" => "$(@__DIR__)/../../test/test_sets/random/medium/query_file_medium.txt",
                                "previous_issues" => "$(@__DIR__)/../../test/test_sets/previous_issues/query_file_previous_issues.txt")
    @assert test_set in keys(test_set_to_file_name) "Unsupported test_set. tiny, small, medium, and previous_issues are supported"
    query_file_name = test_set_to_file_name[test_set]
    test_query_file(query_file_name; all_solvers=all_solvers, solver_types_allowed=solver_types_allowed, solver_types_to_remove=solver_types_to_remove, solver_types_to_report=solver_types_to_report)
    println("Finished testing")
end
