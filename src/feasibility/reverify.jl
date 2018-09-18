struct Reverify{O<:AbstractMathProgSolver} <: Feasibility
	optimizer::O
	m::Float64 # The big M in the linearization
end

Reverify(x) = Reverify(x, 1000.0)

function interpret_result(solver::Reverify, status, neurons)
    if status == :Optimal
        # To do: return adversarial case
        return Result(:False, getvalue(neurons))
    end
    if status == :Infeasible
        return Result(:True)
    end
    return Result(:Undertermined)
end
#=
Encode problem as an MIP following Reverify algorithm
=#
function encode(solver::Reverify, model::Model, problem::Problem)
    neurons, deltas = init_nnet_vars(solver, model, problem.network)
    add_input_constraint(model, problem.input, first(neurons))
    add_complementary_output_constraint(model, problem.output, last(neurons))
    for (i, layer) in enumerate(problem.network.layers)
        lbounds = layer.weights * neurons[i] + layer.bias
        dy = solver.m*(deltas[i+1])  # TODO rename variable
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
    return neurons[1] #delta unimportant
end

