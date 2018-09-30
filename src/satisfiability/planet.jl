# Planet
struct Planet
    optimizer::AbstractMathProgSolver
end

function solve(solver::Planet, problem::Problem)   
    # refine bounds
    status, bounds = tighten_bounds(solver, problem) # 3.1
    if status != :Optimal 
        return Result(:SAT)
    end

    # partial assignment
    nnet = problem.network
    p = get_activation(nnet, bounds)
    ψ = init_ψ(get_list(p))

    # compute extra conditions
    extra = infer_node_phases(nnet, p, bounds) # 3.4
    status, tight = get_tight_clause(nnet, p, bounds) # 3.3
    if length(tight) > 0
        append!(extra, Any[tight])
    end
    soln = PicoSAT.solve(ψ)

    # main loop to compute the SAT problem
    while soln != :unsatisfiable
        if length(extra) > 0
            append!(ψ, extra)
            soln = PicoSAT.solve(ψ)
            if soln == :unsatisfiable
                return Result(:SAT)
            end
        end
        status, conflict = elastic_filtering(nnet, soln, bounds) # 3.2
        if status == :Infeasible
            extra = conflict
            break
        else
            return Result(:UNSAT)
        end
    end
    return Result(:SAT)
end

function get_list(p::Vector{Vector{Int64}})
    list = Vector{Int64}(0)
    for i in 1:length(p)
        for j in 1:length(p[i])
            append!(list, Int64[p[i][j]])
        end
    end
    return list
end

function get_assignment(nnet::Network, list::Vector{Int64})
    p = Vector{Vector{Int64}}(length(nnet.layers))
    n = 0
    for (i, layer) in enumerate(nnet.layers)
        p[i] = fill(0, length(layer.bias))
        for j in 1:length(p[i])
            p[i][j] = ifelse(list[n+j] > 0, 1, -1)
        end 
        n += length(p[i])
    end
    return p
end

function get_node_id(nnet::Network, x::Tuple{Int64, Int64})
    n = 0
    for i in 1:x[1]-1
        n += length(nnet.layers[i].bias)
    end
    return n + x[2]
end

function get_node_id(nnet::Network, n::Int64)
    i = 0
    j = n
    while j > length(nnet.layers[i+1].bias)
        i += 1
        j = j - length(nnet.layers[i].bias)
    end
    return (i, j)
end

function elastic_filtering(nnet::Network, p::Vector{Vector{Int64}}, bounds::Vector{Hyperrectangle})
    model = JuMP.Model(solver = optimizer)

    neurons = init_nnet_vars(solver, model, nnet)
    add_input_constraint(model, problem.input, first(neurons))
    add_complementary_output_constraint(model, problem.output, last(neurons))
    encode_lp_constraint(model, nnet, bounds, neurons)
    slack, J = encode_partial_assignment(model, nnet, p, neurons, true)

    conflict = Vector{Int64}(0)
    while true
        @objective(model, Min, J)
        status = JuMP.solve(model)
        if status != :Optimal
            return (:Infeasible, conflict) # return the conflicts in p
        end
        
        s_value = getvalue(slack)
        (m, index) = max_slack(s_value)
        if m > 0.0
            append!(conflict, Any[-p[index[1]][index[2]] * get_node_id(nnet, index)])
            @constraint(model, slack[index[1]][index[2]] == 0.0)
        else
            return (:Feasible, conflict) # partial assignment p is feasible
        end
    end
end

elastic_filtering(nnet::Network, list::Vector{Int64}, bounds::Vector{Hyperrectangle}) = elastic_filtering(nnet, get_assignment(nnet, list), bounds)

function get_tight_clause(nnet::Network, p::Vector{Vector{Int64}}, bounds::Vector{Hyperrectangle})
    model = JuMP.Model(solver = optimizer)

    neurons = init_nnet_vars(solver, model, nnet)
    add_input_constraint(model, problem.input, first(neurons))
    add_complementary_output_constraint(model, problem.output, last(neurons))
    encode_lp_constraint(model, nnet, bounds, neurons)
    encode_partial_assignment(model, nnet, p, neurons, false)
    
    J = 0.0
    for i in 1:length(p)
        for j in 1:length(p[i])
            if p[i][j] == 0
                J += neurons[i+1][j]
            end
        end
    end

    @objective(model, Min, J)
    status = JuMP.solve(model)
    if status != :Optimal
        return (:Infeasible, [])
    end

    v = getvalue(neurons)
    tight = Vector{Int64}(0)
    complete = :Complete
    for i in 1:length(p)
        for j in 1:length(p[i])
            if p[i][j] == 0 
                if v[i+1][j] > 0
                    append!(tight, Any[get_node_id(nnet, (i,j))])
                else
                    complete = :Incomplete
                end
            end
        end
    end
    return (complete, tight)
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

function encode_partial_assignment(model::Model, nnet::Network, p::Vector{Vector{Int64}}, neurons, slack::Bool)
    if slack
        slack_var = Vector{Vector{Variable}}(length(nnet.layers))
        sum_slack = 0.0
    end
    for (i, layer) in enumerate(nnet.layers)
        (W, b, act) = (layer.weights, layer.bias, layer.activation)
        before_act = W * neurons[i] + b
        if slack
            slack_var[i] = @variable(model, [1:length(b)])
        end
        for j in length(b)
            if p[i][j] != 0
                if slack
                    sum_slack += slack_var[i][j]
                    before_act[j] -= slack_var[i][j]
                end
                if p[i][j] == 1
                    @constraint(model, neurons[i+1][j] >= before_act[j])
                else
                    @constraint(model, neurons[i+1][j] == 0.0)
                    @constraint(model, 0.0 >= before_act[j])
                end
            end
        end
    end
    if slack
        return slack_var, sum_slack
    else
        return nothing
    end
end

function init_ψ(p_list::Vector{Int64})
    ψ = Vector{Vector{Int64}}(length(p_list))
    for i in 1:length(p_list)
        if p_list[i] == 0
            ψ[i] = [i, -i]
        else
            ψ[i] = [p_list[i]*i]
        end
    end
    return ψ
end

# To be implemented
function infer_node_phases(nnet::Network, p::Vector{Vector{Int64}}, bounds::Vector{Hyperrectangle})
    extra = Vector{Vector{Int64}}(0)
    return extra
end
