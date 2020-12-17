# The following two test is based on a 4 layer network.
# We can consider this network is in the form of y=f(x), where x and y are 1d variables.
# The graph of this function looks like \/\/\/\/
# To get a precise rechable set, the verification algorithm needs to fully split
# the input set to reduce all over-approximation.

w_nnet = read_nnet(net_path * "spiky_nnet.nnet", last_layer_activation = NeuralVerification.Id())

ϵ = 1e-1

x_min = 1.0
x_max = 100.0
y_min = 42.51485148514851
y_max = 100.0


problem_holds = Problem(w_nnet,
                           Hyperrectangle(low = [x_min], high = [x_max]),
                           Hyperrectangle(low = [y_min - ϵ], high = [y_max + ϵ]));

problem_violated = Problem(w_nnet,
                           Hyperrectangle(low = [x_min], high = [x_max]),
                           Hyperrectangle(low = [y_min + ϵ], high = [y_max - ϵ]));

# NOTE: 'max_iter' of the solver must be large enough to fully split.
@time @testset "Fully split, ReluVal" begin
    solver = ReluVal(max_iter = 1000)
    @test solve(solver, problem_holds).status == :holds
    @test solve(solver, problem_violated).status == :violated
end

@time @testset "Fully split, Neurify" begin
    solver = Neurify(max_iter = 1000)
    @test solve(solver, problem_holds).status == :holds
    @test solve(solver, problem_violated).status == :violated
end

