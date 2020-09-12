@testset "Full Correctness Test Set" begin
    solver_types_to_remove = solver_types_to_remove=[NeuralVerification.ExactReach]
    test_correctness(;test_set="tiny", solver_types_to_remove=solver_types_to_remove)
    test_correctness(;test_set="small", solver_types_to_remove=solver_types_to_remove)
    test_correctness(;test_set="medium", solver_types_to_remove=solver_types_to_remove)
    #test_correctness(;test_set="previous_issues", solver_types_to_remove=solver_types_to_remove)
end
