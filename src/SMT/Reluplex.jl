struct Reluplex{O<:AbstractMathProgSolver} <: SMT
    optimizer::O
    m::Float64 # The big M in the linearization
end

Reluplex(x) = Reluplex(x, 1000.0)

function interpret_result(solver::Reverify, status, neurons)
    if status == :Optimal
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
        ubounds = lbounds + dy
        for j in 1:length(layer.bias)
            @constraints(model, begin
                                    neurons[i+1][j] >= lbounds[j]
                                    neurons[i+1][j] <= ubounds[j]
                                    neurons[i+1][j] >= 0.0
                                    neurons[i+1][j] <= solver.m-dy[j]
                                end)
        end
    end
    return neurons[1] #return input variable
end

