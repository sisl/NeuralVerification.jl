# Setup problem
network = NeuralVerification.read_nnet("./examples/networks/issue_net.nnet")
input_set = NeuralVerification.Hyperrectangle([-0.42076139039401705], [0.42346836042582625])
output_set = NeuralVerification.Hyperrectangle([-0.13583423942984774], [0.4613520242736866])
problem = NeuralVerification.Problem(network, input_set, output_set)

NeuralVerification.write_problem("test/test_sets/previous_issues/networks/issue_111_network2.nnet", "test/test_sets/previous_issues/input_sets/issue_111_input_set2.json", "test/test_sets/previous_issues/output_sets/issue_111_output_set2.json", problem; query_file="test/test_sets/previous_issues/query_file_previous_issues.txt")
println(NeuralVerification.solve(NeuralVerification.Sherlock(ϵ = 0.5), problem))
