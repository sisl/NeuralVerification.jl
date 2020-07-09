import NeuralVerification

# Create and write the problem
nnet = NeuralVerification.read_nnet("$(@__DIR__)/networks/small_nnet.nnet")
input_set = NeuralVerification.Hyperrectangle([1.0], [3.0])
output_set = NeuralVerification.HalfSpace([5.0], 1.0)
problem = NeuralVerification.Problem(nnet, input_set, output_set)

NeuralVerification.write_problem("./test_problem/network.nnet",
              "./test_problem/input.json",
              "./test_problem/output.json",
              problem
              ; query_file="./test_problem/query_file.txt")

# Read in a problem from a query file and then solve it
query_lines = readlines("./test_problem/query_file.txt")
new_problem = NeuralVerification.query_line_to_problem(query_lines[1])
NeuralVerification.solve(NeuralVerification.Duality(), new_problem)

# Can also read the problem directly
problem = NeuralVerification.read_problem("./test_problem/network.nnet", "./test_problem/input.json", "./test_problem/output.json")
