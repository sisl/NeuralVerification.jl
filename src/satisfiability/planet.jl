# Planet
struct Planet
    optimizer::AbstractMathProgSolver
    eager::Bool # Default false
end

Planet(x::AbstractMathProgSolver) = Planet(x, false)

function solve(solver::Planet, problem::Problem)
    @assert ~solver.eager "Eager implementation not supported yet"
    # refine bounds
    status, bounds = tighten_bounds(problem, solver.optimizer) # 3.1
    if status != :Optimal
        return BasicResult(:SAT)
    end

    # partial assignment
    nnet = problem.network
    p = get_activation(nnet, bounds)
    ψ = init_ψ(get_list(p))

    # compute extra conditions
    #=
    extra = infer_node_phases(nnet, p, bounds) # 3.4
    status, tight = get_tight_clause(nnet, p, bounds) # 3.3
    if length(tight) > 0
        append!(extra, Any[tight])
    end
    =#
    soln = PicoSAT.solve(ψ)

    # main loop to compute the SAT problem
    while soln != :unsatisfiable
        status, conflict = elastic_filtering(nnet, soln, bounds, solver.optimizer) # 3.2
        if status == :Infeasible
            append!(ψ, Any[conflict])
            soln = PicoSAT.solve(ψ)
        else
            return BasicResult(:UNSAT)
        end
    end
    return BasicResult(:SAT)
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

function elastic_filtering(nnet::Network, p::Vector{Vector{Int64}}, bounds::Vector{Hyperrectangle}, optimizer::AbstractMathProgSolver)
    model = JuMP.Model(solver = optimizer)
    neurons = init_neurons(model, nnet)
    add_input_constraint(model, problem.input, first(neurons))
    add_complementary_output_constraint(model, problem.output, last(neurons))
    encode_Δ_lp(model, nnet, bounds, neurons)
    slack = encode_slack_lp(model, nnet, p, neurons)
    J = min_sum_all(model, slack)
    conflict = Vector{Int64}(0)
    while true
        status = solve(model)
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

elastic_filtering(nnet::Network, list::Vector{Int64}, bounds::Vector{Hyperrectangle}, optimizer::AbstractMathProgSolver) = elastic_filtering(nnet, get_assignment(nnet, list), bounds, optimizer)

function get_tight_clause(nnet::Network, p::Vector{Vector{Int64}}, bounds::Vector{Hyperrectangle})
    model = JuMP.Model(solver = optimizer)

    neurons = init_neurons(solver, model, nnet)
    add_input_constraint(model, problem.input, first(neurons))
    add_complementary_output_constraint(model, problem.output, last(neurons))
    encode_Δ_lp(model, nnet, bounds, neurons)
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
    status = solve(model)
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

function tighten_bounds(problem::Problem, optimizer::AbstractMathProgSolver)
    bounds = get_bounds(problem)
    model = JuMP.Model(solver = optimizer)
    neurons = init_neurons(model, problem.network)
    add_input_constraint(model, problem.input, first(neurons))
    add_complementary_output_constraint(model, problem.output, last(neurons))
    encode_Δ_lp(model, problem.network, bounds, neurons)

    J = min_sum_all(model, neurons)
    status = solve(model)
    if status == :Optimal
        lower = getvalue(neurons)
    else
        return (:Infeasible, [])
    end

    J = max_sum_all(model, neurons)
    status = solve(model)
    if status == :Optimal
        upper = getvalue(neurons)
    else
        return (:Infeasible, [])
    end

    new_bounds = Vector{Hyperrectangle}(length(neurons))
    for i in 1:length(neurons)
        new_bounds[i] = Hyperrectangle(low = lower[i], high = upper[i])
    end
    return (:Optimal, new_bounds)
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
        for j in 1:length(b)
            if p[i][j] != 0
                if slack
                    sum_slack += slack_var[i][j]
                    if p[i][j] == 1
                        @constraint(model, neurons[i+1][j] == before_act[j] + slack_var[i][j])
                        @constraint(model, before_act[j] + slack_var[i][j] >= 0.0)
                    else
                        @constraint(model, neurons[i+1][j] == 0.0)
                        @constraint(model, 0.0 >= before_act[j] - slack_var[i][j])
                    end
                else
                    if p[i][j] == 1
                        @constraint(model, neurons[i+1][j] <= before_act[j])
                    else
                        @constraint(model, neurons[i+1][j] == 0.0)
                        @constraint(model, 0.0 >= before_act[j])
                    end
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
