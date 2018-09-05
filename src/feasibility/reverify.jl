# include("../../solver/solver.jl")
# include("../utils/problem.jl")

struct ReverifySolver <: Solver
	model::Model
	m::Float64
	ReverifySolver(model, m) = new(model, m)
end

#=
Initialize JuMP variables corresponding to neurons and deltas of network for problem
=#
function init_nnet_vars(model::Model, network::Network)

    layers   = network.layers
    neurons = Array{Array{Variable}}(length(layers) + 1) # +1 for input layer
    deltas  = Array{Array{Variable}}(length(layers) + 1)
    # input layer is treated differently from other layers
    # NOTE: this double-counts layers[1]. Was that the intent?
    input_layer_n = size(first(layers).weights, 2)
    all_layers_n  = [length(l.bias) for l in layers]
    all_n         = [input_layer_n; all_layers_n]

    for (i, n) in enumerate(all_n)
        neurons[i] = @variable(model, [1:n], basename = "layer $i neuron-")
        deltas[i]  = @variable(model, [1:n], basename = "layer $i delta-", category = :Bin)
    end

    return neurons, deltas
end


#=
Add input/output constraints to model
=#
function add_io_constraints(model::Model, problem::FeasibilityProblem, neuron_vars::Array{Array{Variable}})
    in_A,  in_b  = tosimplehrep(problem.input)
    out_A, out_b = tosimplehrep(problem.output)

    @constraint(model,  in_A*first(neuron_vars) .<= in_b)
    @constraint(model, out_A*last(neuron_vars)  .<= out_b)
    return nothing
end

#=
Encode problem as an MIP following Reverify algorithm
=#
function encode(solver::ReverifySolver, problem::FeasibilityProblem)

    neurons, deltas = init_nnet_vars(solver.model, problem.network)
    add_io_constraints(solver.model, problem, neurons)

    for (i, layer) in enumerate(problem.network.layers)
        for j in 1:length(layer.bias)
            lbound = layer.weights*neurons[i] + layer.bias # Is it faster in julia to move this out of the j loop?
            ubound = lbound + solver.m*deltas[i+1][j]
            @constraints(solver.model, begin
                neurons[i+1][j] .>= lbound
                neurons[i+1][j] .<= ubound
                neurons[i+1][j] >= 0
                neurons[i+1][j] <= solver.m*(1-deltas[i+1][j])
            end)
        end
    end
end

#=
Solve encoded problem and return status
Some solvers (e.g. GLPKSolverMIP()) will provide additional information beyond status
=#
function solveMIP(solver::ReverifySolver)
	status = solve(solver.model)
	return status
end

