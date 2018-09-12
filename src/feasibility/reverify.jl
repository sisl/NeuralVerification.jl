
struct ReverifySolver{O<:AbstractMathProgSolver} <: Solver
	m::Float64
    optimizer::O
end

#=
Initialize JuMP variables corresponding to neurons and deltas of network for problem
=#
function init_nnet_vars(model::Model, network::Network)

    layers = network.layers
    neurons = Vector{Vector{Variable}}(length(layers) + 1) # +1 for input layer
    deltas  = Vector{Vector{Variable}}(length(layers) + 1)
    # input layer is treated differently from other layers
    # NOTE: this double-counts layers[1]. Was that the intent?
    input_layer_n = size(first(layers).weights, 2)
    all_layers_n  = [length(l.bias) for l in layers]
    insert!(all_layers_n, 1, input_layer_n)

    for (i, n) in enumerate(all_layers_n)
        neurons[i] = @variable(model, [1:n])
        deltas[i]  = @variable(model, [1:n], Bin)
    end

    return neurons, deltas
end


#=
Add input/output constraints to model
=#
function add_io_constraints(model::Model, problem::FeasibilityProblem, neuron_vars::Vector{Vector{Variable}})
    in_A,  in_b  = tosimplehrep(problem.input)
    out_A, out_b = tosimplehrep(problem.output)

    @constraint(model,  in_A*first(neuron_vars) .<= in_b)
    @constraint(model, out_A*last(neuron_vars)  .<= out_b)
    return nothing
end

#=
Encode problem as an MIP following Reverify algorithm
=#
function encode(solver::ReverifySolver, model::Model, problem::FeasibilityProblem)
    neurons, deltas = init_nnet_vars(model, problem.network)
    add_io_constraints(model, problem, neurons)
    for (i, layer) in enumerate(problem.network.layers)
        lbounds = layer.weights * neurons[i] + layer.bias
        dy      = solver.m*(deltas[i+1])  # TODO rename variable
        for j in 1:length(layer.bias)
            ubounds = lbounds + dy[j]
            @constraints(model, begin
                                    neurons[i+1][j] .>= lbounds
                                    neurons[i+1][j] .<= ubounds
                                    neurons[i+1][j]  >= 0.0
                                    neurons[i+1][j]  <= solver.m-dy[j]
                                end)
        end
    end
end

#=
optional argument [optimizer] is where the desired JuMP solver goes.
NOTE: extendin JuMP.solve ...good idea?
=#
function solve(solver::ReverifySolver, problem::FeasibilityProblem)
    model = JuMP.Model(solver = solver.optimizer)
    encode(solver, model, problem)
    status = JuMP.solve(model)
    return status
end


#=
Solve encoded problem and return status
Some solvers (e.g. GLPKSolverMIP()) will provide additional information beyond status
=#
# function solveMIP(solver::ReverifySolver)
	# status = solve(solver.model)
	# return status
# end

