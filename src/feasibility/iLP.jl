# Iterative LP

struct ILP{O<:AbstractMathProgSolver} <: Feasibility
    optimizer::O
    max_iter::Int64
end

function solve(solver::ILP, problem::Problem)
    x = problem.input.center
    i = 0
    while i < solver.max_iter
        model = JuMP.Model(solver = solver.optimizer)
        act_pattern = get_activation(problem.network, x)
        println(act_pattern)
        input_neurons = encode(solver, model, problem, act_pattern)
        status = JuMP.solve(model)
        x = getvalue(input_neurons)
        if satisfy(problem.network, x, act_pattern)
            radius = maximum(abs.(x - problem.input.center))
            if radius >= minimum(problem.input.radius)
                return Result(:SAT, radius)
            else
                return Result(:UNSAT, radius)
            end
        end
        i += 1
    end
    return Result(:Unknown)
end

function encode(solver::ILP, model::Model, problem::Problem, act_pattern::Vector{Vector{Bool}})
    neurons, deltas = init_nnet_vars(solver, model, problem.network)
    add_complementary_output_constraint(model, problem.output, last(neurons))

    for (i, layer) in enumerate(problem.network.layers)
        (W, b, act) = (layer.weights, layer.bias, layer.activation)
        before_act = W * neurons[i] + b
        for j in 1:length(layer.bias) # For evey node
            if act_pattern[i][j]
                # @constraint(model, before_act[j] >= 0.0)
                @constraint(model, neurons[i+1][j] == before_act[j])
            else
                # @constraint(model, before_act[j] <= 0.0)
                @constraint(model, neurons[i+1][j] == 0.0)
            end
        end
    end

    # Objective: L∞ norm of the disturbance
    J = maximum(absolute(neurons[1] - problem.input.center))
    @objective(model, Min, J)

    return neurons[1]
end

function satisfy(nnet::Network, x::Vector{Float64}, act_pattern::Vector{Vector{Bool}})
    curr_value = x
    for (i, layer) in enumerate(nnet.layers)
        curr_value = layer.weights * curr_value + layer.bias
        for j in 1:length(curr_value)
            if act_pattern[i][j] && curr_value[j] < 0.0
                return false
            end
            if ~act_pattern[i][j] && curr_value[j] > 0.0
                return false
            end
        end
        curr_value = layer.activation(curr_value)
    end
    return true
end