# Setup problem
network = NeuralVerification.read_nnet("./test/test_sets/random/medium/networks/rand_[4,8,4]_1.nnet")
input_set = NeuralVerification.HPolytope{Float64}([NeuralVerification.HalfSpace{Float64}([1.0, 0.0, 0.0, 0.0], 0.3845579517032085), NeuralVerification.HalfSpace{Float64}([0.0, 1.0, 0.0, 0.0], 0.8331193586918577), NeuralVerification.HalfSpace{Float64}([0.0, 0.0, 1.0, 0.0], 0.6607352829885846), NeuralVerification.HalfSpace{Float64}([0.0, 0.0, 0.0, 1.0], 0.270232549769154), NeuralVerification.HalfSpace{Float64}([-1.0, 0.0, 0.0, 0.0], 0.022984248429274823), NeuralVerification.HalfSpace{Float64}([0.0, -1.0, 0.0, 0.0], 1.061033037362016), NeuralVerification.HalfSpace{Float64}([0.0, 0.0, -1.0, 0.0], 0.734914583454296), NeuralVerification.HalfSpace{Float64}([0.0, 0.0, 0.0, -1.0], 0.7494557834041424)])
output_set = NeuralVerification.HPolytope{Float64}(NeuralVerification.HalfSpace{Float64}[NeuralVerification.HalfSpace{Float64}([1.0, 0.0, 0.0, 0.0], 0.47919249361761995), HalfSpace{Float64}([0.0, 1.0, 0.0, 0.0], 0.21600299802246092), HalfSpace{Float64}([0.0, 0.0, 1.0, 0.0], 0.13753126667274596), HalfSpace{Float64}([0.0, 0.0, 0.0, 1.0], 0.536256411404834), HalfSpace{Float64}([-1.0, 0.0, 0.0, 0.0], 0.05623084997936845), HalfSpace{Float64}([0.0, -1.0, 0.0, 0.0], -0.02937453504189702), HalfSpace{Float64}([0.0, 0.0, -1.0, 0.0], 0.7988290732045045), HalfSpace{Float64}([0.0, 0.0, 0.0, -1.0], 1.1397183250614469)])
problem = NeuralVerification.Problem(network, input_set, output_set)

# problem = NeuralVerification.query_line_to_problem("test/test_sets/random/medium/networks/rand_[4,8,4]_1.nnet test/test_sets/random/medium/input_sets/hp_hp_one_[4,8,4]_1_inputset.json test/test_sets/random/medium/output_sets/hp_hp_one_[4,8,4]_1_outputset.json")
# println(problem.input)
# println(problem.output)

# Exact Reach issue
println(NeuralVerification.solve(NeuralVerification.ExactReach(), problem))
