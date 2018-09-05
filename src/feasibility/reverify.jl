include("../solver/solver.jl")
include("../utils/problem.jl")

struct ReverifySolver <: Solver
	model::Model
	m::Float64
	ReverifySolver(model, m) = new(model, m)
end

#=
Initialize JuMP variables corresponding to neurons and deltas of network for problem
=#
function init_nnet_vars(solver::ReverifySolver, problem::FeasibilityProblem)

    layers   = problem.network.layers
    neurons = Array{Array{Variable}}(length(layers) + 1) # +1 for input layer
    deltas  = Array{Array{Variable}}(length(layers) + 1)
    # input layer is treated differently from other layers
    # NOTE: this double-counts layers[1]. Was that the intent?
    input_layer_n = size(layers[1].weights, 2)
    all_layers_n  = [length(l.bias) for l in layers]
    all_n         = [input_layer_n; all_layers_n]

    for (i, n) in enumerate(all_n)
        neurons[i] = @variable(solver.model, [1:n], basename = "layer $i neuron-")
        deltas[i]  = @variable(solver.model, [1:n], basename = "layer $i delta-", category = :Bin)
    end

    return neurons, deltas
end


#=
Add input/output constraints to model
=#
function add_io_constraints(model::Model, problem::Problem, neuron_vars::Array{Array{Variable}}, delta_vars::Array{Array{Variable}})
    add_constraints(model, neuron_vars[1], problem.input)
    add_constraints(model, neuron_vars[length(problem.network.layers)+1], problem.output)
end

#=
Encode problem as an MIP following Reverify algorithm
=#
function encode(solver::ReverifySolver, problem::Problem)
    neuron_vars, delta_vars = init_nnet_vars(solver, problem)
    add_io_constraints(solver.model, problem, neuron_vars, delta_vars)
    for i = 1:length(problem.network.layers)
        layer = problem.network.layers[i]
        for j in length(layer.bias)
            @constraint(solver.model, neuron_vars[i+1][j] .>= layer.weights*neuron_vars[i] + layer.bias)
            @constraint(solver.model, neuron_vars[i+1][j] .<= layer.weights*neuron_vars[i] + layer.bias + solver.m*delta_vars[i+1][j])
            @constraint(solver.model, neuron_vars[i+1][j] >= 0)
            @constraint(solver.model, neuron_vars[i+1][j] <= solver.m*(1-delta_vars[i+1][j]))
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

