# Planet
struct Planet
    optimizer::AbstractMathProgSolver
end


function solve(solver::Planet, problem::Problem)   
    # Refine bounds
    status, bounds = tighten_bounds(solver, problem)
    if status != :Optimal 
        return Result(:SAT)
    end

    # partial assignment
    p = get_activation(problem.network, bounds)
    # Infered assignment
    extra, p = infer_node_phases(problem.network, p, bounds)

    println(p)
    println(extra)

    while satisfy(p)
        while extra != nothing
            # Perform unit propagation ...
        end

        extra = infer_node_phases(problem.network, p, bounds)
        if extra == nothing
            status, conflict = elastic_filtering(problem.network, p, bounds)
            if satisfy(p)
                if complete(p)
                    return Result(:UNSAT)
                end
                # add a new variable assignment
                # if p cannot be extended to a satisfying valuation
                # 
            end
        end
    end
    return Result(:SAT)
end

function satisfy(p::Vector{Vector{Int64}})
    for i in 1:length(p)
        for j in 1:length(p[i])
            if p[i][j] < -1
                return false
            end
        end
    end
    return true
end

function complete(p::Vector{Vector{Int64}})
    for i in 1:length(p)
        for j in 1:length(p[i])
            if p[i][j] == 0
                return false
            end
        end
    end
    return true
end

function elastic_filtering(nnet::Network, p::Vector{Vector{Int}}, bounds::Vector{Hyperrectangle})
    model = JuMP.Model(solver = optimizer)

    neurons = init_nnet_vars(solver, model, problem.network)
    add_input_constraint(model, problem.input, first(neurons))
    add_complementary_output_constraint(model, problem.output, last(neurons))
    encode_lp_constraint(model, problem.network, bounds, neurons)
    slack, J = encode_slack_constraint(model, problem.network, p, neurons)

    conflict = Vector{Tuple{Int64, Int64}}(0)
    while true
        @objective(model, Min, J)
        status = JuMP.solve(model)
        if status != :Optimal
            return (:Infeasible, conflict) # return the conflicts in p
        end
        
        s_value = getvalue(slack)
        (m, index) = max_slack(s_value)
        if m > 0.0
            add!(conflict, Truple{Int64, Int64}[index])
            @constraint(model, slack[index[1]][index[2]] == 0.0)
        else
            return (:Feasible, conflict) # partial assignment p is feasible
        end
    end
end

function max_slack(x::Vector{Vector{Float64}})
    m = 0.0
    index = (0, 0)
    for i in 1:length(x)
        for j in 1:length(x[i])
            if x[i][j] > m
                m = x[i][j]
                index = (i, j)
            end
        end
    end
    return (m, index)
end

function tighten_bounds(solver::Planet, problem::Problem)
    # bounds from interval arithmatic (call MaxSens)
    bounds = get_bounds(problem)

    model = JuMP.Model(solver = optimizer)

    neurons = init_nnet_vars(solver, model, problem.network)
    add_input_constraint(model, problem.input, first(neurons))
    add_complementary_output_constraint(model, problem.output, last(neurons))
    encode_lp_constraint(model, problem.network, bounds, neurons)

    n_layer = length(problem.network.layers)
    J = sum(sum(neurons[i]) for i in 1:n_layer+1)

    # Git tight lower bounds
    @objective(model, Min, J)
    status = JuMP.solve(model)
    if status == :Optimal
        lower = getvalue(neurons)
    else
        return (:Infeasible, [])
    end

    # Git tight upper bounds
    @objective(model, Max, J)
    status = JuMP.solve(model)
    if status == :Optimal
        upper = getvalue(neurons)
    else
        return (:Infeasible, [])
    end

    println(lower, upper)
    new_bounds = Vector{Hyperrectangle}(n_layer + 1)
    for i in 1:n_layer + 1
        new_bounds[i] = Hyperrectangle(low = lower[i], high = upper[i])
    end
    return (:Optimal, new_bounds)
end

function init_nnet_vars(solver::Planet, model::Model, network::Network)
    layers = network.layers
    neurons = Vector{Vector{Variable}}(length(layers) + 1)

    input_layer_n = size(first(layers).weights, 2)
    all_layers_n  = [length(l.bias) for l in layers]
    insert!(all_layers_n, 1, input_layer_n)

    for (i, n) in enumerate(all_layers_n)
        neurons[i] = @variable(model, [1:n])
    end

    return neurons
end

function encode_lp_constraint(model::Model, nnet::Network, bounds::Vector{Hyperrectangle}, neurons)
    for (i, layer) in enumerate(nnet.layers)
        (W, b, act) = (layer.weights, layer.bias, layer.activation)
        before_act = W * neurons[i] + b
        before_act_rectangle = linear_transformation(layer, bounds[i])
        lower, upper = low(before_act_rectangle), high(before_act_rectangle)
        # For now assume only ReLU activation
        for j in 1:length(layer.bias) # For evey node
            if lower[j] > 0.0
                @constraint(model, neurons[i+1][j] == before_act[j])
            elseif upper[j] < 0.0 
                @constraint(model, neurons[i+1][j] == 0.0)
            else # Here use triangle relaxation
                @constraints(model, begin
                                    neurons[i+1][j] >= before_act[j]
                                    neurons[i+1][j] <= upper[j] / (upper[j] - lower[j]) * (before_act[j] - lower[j])
                                    neurons[i+1][j] >= 0.0
                                end)
            end
        end
    end
    return nothing
end

function encode_slack_constraint(model::Model, nnet::Network, p::Vector{Vector{Int64}}, neurons)
    slack = Vector{Vector{Variable}}(length(nnet.layers))
    sum_slack = 0.0
    for (i, layer) in enumerate(nnet.layers)
        (W, b, act) = (layer.weights, layer.bias, layer.activation)
        before_act = W * neurons[i] + b
        slack[i] = @variable(model, [1:length(b)])
        for j in length(b)
            if p[i][j] != 0
                sum_slack += slack[i][j]
                if p[i][j] == 1
                    @constraint(model, neurons[i+1][j] + slack[i][j] >= before_act[j])
                else
                    @constraint(model, neurons[i+1][j] == 0.0)
                    @constraint(model, slack[i][j] >= before_act[j])
                end
            end
        end
    end
    return slack, sum_slack
end

function init_var(nnet::Network)
    # For every node, there are three modes
    # 1: activated
    # 0: unknown
    # -1: not activated
    ψ = Vector{Vector{Int64}}(length(nnet.layers))
    for (i, layer) in enumerate(nnet.layers)
        ψ[i] = fill(0, length(layer.bias))
    end
    return ψ
end

# To be implemented
function infer_node_phases(nnet::Network, p::Vector{Vector{Int64}}, bounds::Vector{Hyperrectangle})
    new_p = Vector{Vector{Int64}}(length(nnet.layers))
    extra = Vector{Tuple{Int64, Int64}}()
    for (i, layer) in enumerate(nnet.layers)
        new_p[i] = p[i]
        for j in 1:length(layer.bias)
            if p[i][j] == 0
                add!(extra, Tuple{Int64, Int64}[(i, j)])
            end
        end
    end
    return extra, new_p
end
