"""
test_query_file(file_name::String; [solvers = []], [solver_types_allowed = []], [solver_types_to_remove =[]])

    Run a set of benchmarks on a query file given by file_name.
    Compares results for consistency, but has no ground truth to compare against.

    Will filter the set of available solvers by type using solver_types_allowed ans solver_types_to_remove.
    Will use a default list unless a solvers list is given

"""

function test_query_file(file_name::String; solvers = [], solver_types_allowed=[], solver_types_to_remove=[])
    path, file = splitdir(file_name)
    @testset "Correctness Tests on $(file)" begin
        queries = readlines(file_name)
        for (index, line) in enumerate(queries)
            println("Testing on line: ", line)
            @testset "Test on line: $index" begin
                cur_problem = NeuralVerification.query_line_to_problem(line; base_dir="$(@__DIR__)/../")
                solvers = NeuralVerification.get_valid_solvers(cur_problem; solvers=solvers, solver_types_allowed=solver_types_allowed, solver_types_to_remove=solver_types_to_remove=solver_types_to_remove)

                # Solve the problem with each solver that applies for the problem, then compare the results
                results = []
                for solver in solvers
                    println("Solving on: ", typeof(solver))

                    # Workaround to have ConvDual and FastLip take in halfspace output sets as expected
                    if ((solver isa NeuralVerification.ConvDual || solver isa NeuralVerification.FastLip)) && cur_problem.output isa NeuralVerification.HalfSpace
                        cur_problem = NeuralVerification.Problem(cur_problem.network, cur_problem.input, NeuralVerification.HPolytope([cur_problem.output])) # convert to a HPolytope b/c ConvDual takes in a HalfSpace as a HPolytope for now
                    end

                    # Try-catch while solving to handle a GLPK bug by attempting to run with Gurobi instead when this bug shows up
                    try
                        push!(results, NeuralVerification.solve(solver, cur_problem))
                    catch e
                        if e isa GLPK.GLPKError && e.msg == "invalid GLPK.Prob" && solver isa NeuralVerification.Sherlock
                            push!(results, NeuralVerification.CounterExampleResult(:unknown)) # Known issue with GLPK so ignore this error and just push an unknown result
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
                println("Starting comparisons")
                for (i, j) in [(i, j) for i = 1:length(solvers) for j = (i+1):length(solvers)]
                    @testset "Comparing $(typeof(solvers[i])) with $(typeof(solvers[j]))" begin
                        solver_one_complete = NeuralVerification.is_complete(solvers[i])
                        solver_two_complete = NeuralVerification.is_complete(solvers[j])
                        println("Comparing ", i, " to ", j)
                        # Both complete
                        if (solver_one_complete && solver_two_complete)
                            # Results match
                            @test results[i].status == results[j].status
                        # Solver one complete, solver two incomplete
                        elseif (solver_one_complete && !solver_two_complete)
                            # Results match or solver two unknown or solver one holds solver two violated (bc incomplete)
                            @test ((results[i].status == results[j].status) || (results[j].status == :unknown) || (results[i].status == :holds && results[j].status == :violated))
                        # Solver one incomplete, solver two complete
                        elseif (!solver_one_complete && solver_two_complete)
                            # Results match or solver one unknown or solver two holds solver one violated (bc incomplete)
                            @test ((results[i].status == results[j].status) || (results[i].status == :unknown) || ((results[i].status == :violated) && (results[j].status == :holds)))
                        # Neither are complete
                        else
                            # Results match or solver one unknown or solver two unknown or
                            # no test since any mix of outcomes could be justified with two incomplete ones
                        end
                    end
                end
            end
        end
    end
end

file_name_small = "$(@__DIR__)/../test/test_sets/random/small/query_file_small.txt"
file_name_medium = "$(@__DIR__)/../test/test_sets/random/medium/query_file_medium.txt"
file_name_large = "$(@__DIR__)/../test/test_sets/random/large/query_file_large.txt"

file_name_previous_issues = "$(@__DIR__)/../test/test_sets/previous_issues/query_file_previous_issues.txt"
file_name_control_networks = "$(@__DIR__)/../test/test_sets/control_networks/query_file_control_small.txt"

println("Starting tests on small random")
test_query_file(file_name_small; solver_types_to_remove=[NeuralVerification.ExactReach])
println("Starting tests on medium random")
#test_query_file(file_name_medium)
println("Starting tests on large random")
#test_query_file(file_name_large)
println("Starting tests on previous issues")
test_query_file(file_name_previous_issues)
#println("Starting tests on control networks")
#test_query_file(file_name_control_networks)
