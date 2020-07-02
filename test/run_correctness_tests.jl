"""
    Run a set of benchmarks given by

"""

file_name = "$(@__DIR__)/../examples/query_files/identity_network_queries.txt"
path, file = splitdir(file_name)
@testset "Correctness Tests on $(file)" begin

    queries = readlines(file_name)
    for (index, line) in enumerate(queries)
        @testset "Test on line: $index" begin
            cur_problem = NeuralVerification.query_line_to_problem(line)
            solvers = NeuralVerification.get_valid_solvers(cur_problem)

            # Solve the problem with each solver that applies for the problem, then compare the results
            results = []
            for solver in solvers
                push!(results, solve(solver, cur_problem))
            end

            # Just sees if each pair agrees
            # if one or both return unknown then we can't make a comparison
            for (i, j) in [(i, j) for i = 1:length(solvers) for j = (i+1):length(solvers)]
                @testset "Comparing $(typeof(solvers[i])) with $(typeof(solvers[j]))" begin
                    solver_one_complete = is_complete(solvers[i])
                    solver_two_complete = is_complete(solvers[j])

                    # Both complete
                    if (solver_one_complete && solver_two_complete)
                        # Results match
                        @test results[i].status == results[j].status
                    # Solver one complete, solver two incomplete
                    elseif (solver_one_complete && !solver_two_complete)
                        # Results match or solver two unknown or solver one holds solver two violated (bc incomplete)
                        @test results[i].status == results[j].status || results[j] == :unknown || (results[i] == :holds && results[j] == :violated)
                    # Solver one incomplete, solver two complete
                    elseif (!solver_one_complete && solver_two_complete)
                        # Results match or solver one unknown or solver two holds solver one violated (bc incomplete)
                        @test results[i].status == results[j].status || results[i] == :unknown || (results[i] == :violated && results[j] == :holds)
                    # Neither are complete
                    else
                        # Results match or solver one unknown or solver two unknown or
                        # no test since any mix of outcomes could be justified with two incomplete ones
                        @test true
                    end
                end
            end
        end
    end
end

# write_problem("$(@__DIR__)/../examples/networks/group1.nnet", "$(@__DIR__)/../examples/input_sets/identity_network_group1_holds.json", "$(@__DIR__)/../examples/output_sets/identity_network_group1_holds.json", problem_holds; query_file="$(@__DIR__)/../examples/query_files/identity_network_queries.txt")
# write_problem("$(@__DIR__)/../examples/networks/group1.nnet", "$(@__DIR__)/../examples/input_sets/identity_network_group1_violated.json", "$(@__DIR__)/../examples/output_sets/identity_network_group1_violated.json", problem_violated; query_file="$(@__DIR__)/../examples/query_files/identity_network_queries.txt")
