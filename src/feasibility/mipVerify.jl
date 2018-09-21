# MIPVerify
#   Include maximum activation
#   Include presolve
# Only take half space input constraint
# Computes the allowable radius of input perturbations
struct MIPVerify{O<:AbstractMathProgSolver} <: Feasibility
    optimizer::O
end

function encode(solver::MIPVerify, model::Model, problem::Problem)
    bounds = get_bounds(problem)
    neurons, deltas = init_nnet_vars(solver, model, problem.network)
    add_complementary_output_constraint(model, problem.output, last(neurons))
    for (i, layer) in enumerate(problem.network.layers)
        (W, b, act) = (layer.weights, layer.bias, layer.activation)
        before_act = W * neurons[i] + b
        before_act_center = W * bounds[i].center + b
        before_act_radius = zeros(size(W,1))
        for j in 1:size(W, 1)
            before_act_radius[j] = sum(abs.(W[j, :]) .* bounds[i].radius)
        end
        before_act_rectangle = Hyperrectangle(before_act_center, before_act_radius)
        lower = low(before_act_rectangle)
        upper = high(before_act_rectangle)

        # For now assume only ReLU activation
        for j in 1:length(layer.bias) # For evey node
            if lower[j] >= 0.0
                @constraint(model, neurons[i+1][j] == before_act[j])
            elseif upper[j] <= 0.0 
                @constraint(model, neurons[i+1][j] == 0.0)
            else
                ubounds = lbounds + dy[j]
                @constraints(model, begin
                                    neurons[i+1][j] .>= before_act[j]
                                    neurons[i+1][j] .<= upper[j] * deltas[i+1][j]
                                    neurons[i+1][j]  >= 0.0
                                    neurons[i+1][j]  <= before_act[j] - lower[j] * (1 - deltas[i+1][j])
                                end)
            end
        end
    end

    # Objective: need to change to L∞ norm
    J = maximum(neurons[1] - problem.input.center)
    @objective(model, Min, J)
    return neurons[1]
end

function interpret_result(solver::MIPVerify, status, neurons)
    if status == :Infeasible
        return Result(:Undertermined)
    end
    J = maximum(getvalue(neurons) - problem.input.center)
    if J > maximum(problem.input.radius)
        return Result(:True, J)
    else
        return Result(:False, J)
    end
end