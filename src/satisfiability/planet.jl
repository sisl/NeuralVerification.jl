# Planet
struct Planet
    optimizer::AbstractMathProgSolver
    eager::Bool # Default false
end

Planet(x::AbstractMathProgSolver) = Planet(x, false)

function solve(solver::Planet, problem::Problem)
    @assert ~solver.eager "Eager implementation not supported yet"
    # refine bounds
    status, bounds = tighten_bounds(problem, solver.optimizer)
    status == :Optimal || return BasicResult(:SAT)
    ψ = init_ψ(bounds)
    δ = PicoSAT.solve(ψ)
    opt = solver.optimizer
    # main loop to compute the SAT problem
    while δ != :unsatisfiable
        status, conflict = elastic_filtering(problem, δ, bounds, opt)
        status == :Infeasible || return BasicResult(:UNSAT)
        append!(ψ, Any[conflict])
        δ = PicoSAT.solve(ψ)
    end
    return BasicResult(:SAT)
end

function init_ψ(bounds::Vector{Hyperrectangle})
    ψ = Vector{Vector{Int64}}(undef, 0)
    index = 0
    for i in 2:length(bounds)
        lower, upper = low(bounds[i]), high(bounds[i])
        for j in 1:length(lower)
            index += 1
            lower[j] > 0 && push!(ψ, [index])
            upper[j] < 0 && push!(ψ, [-index])
            lower[j] <= 0 <= upper[j] && push!(ψ, [index, -index])
        end
    end
    return ψ
end

function elastic_filtering(problem::Problem, δ::Vector{Vector{Int64}}, bounds::Vector{Hyperrectangle}, optimizer::AbstractMathProgSolver)
    model = JuMP.Model(solver = optimizer)
    neurons = init_neurons(model, problem.network)
    add_input_constraint(model, problem.input, first(neurons))
    add_complementary_output_constraint(model, problem.output, last(neurons))
    encode_Δ_lp(model, problem.network, bounds, neurons)
    slack = encode_slack_lp(model, problem.network, δ, neurons)
    J = min_sum_all(model, slack)
    conflict = Vector{Int64}()
    while true
        status = solve(model)
        status == :Optimal || return (:Infeasible, conflict)
        (m, index) = max_slack(getvalue(slack))
        m > 0.0 || return (:Feasible, conflict)
        node = -δ[index[1]][index[2]] * get_node_id(problem.network, index)
        append!(conflict, Any[node])
        @constraint(model, slack[index[1]][index[2]] == 0.0)
    end
end

elastic_filtering(problem::Problem, list::Vector{Int64}, bounds::Vector{Hyperrectangle}, optimizer::AbstractMathProgSolver) = elastic_filtering(problem, get_assignment(problem.network, list), bounds, optimizer)

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
    status == :Optimal || return (:Infeasible, [])
    lower = getvalue(neurons)

    J = max_sum_all(model, neurons)
    status = solve(model)
    status == :Optimal || return (:Infeasible, [])
    upper = getvalue(neurons)

    new_bounds = Vector{Hyperrectangle}(undef, length(neurons))
    for i in 1:length(neurons)
        new_bounds[i] = Hyperrectangle(low = lower[i], high = upper[i])
    end
    return (:Optimal, new_bounds)
end

function get_assignment(nnet::Network, list::Vector{Int64})
    p = Vector{Vector{Int64}}(undef, length(nnet.layers))
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

# function init_ψ(p_list::Vector{Int64})
#     ψ = Vector{Vector{Int64}}(undef, length(p_list))
#     for i in 1:length(p_list)
#         if p_list[i] == 0
#             ψ[i] = [i, -i]
#         else
#             ψ[i] = [p_list[i]*i]
#         end
#     end
#     return ψ
# end

# function get_list(p::Vector{Vector{Int64}})
#     list = Vector{Int64}()
#     for i in 1:length(p)
#         for j in 1:length(p[i])
#             append!(list, Int64[p[i][j]])
#         end
#     end
#     return list
# end


# To be implemented
# function infer_node_phases(nnet::Network, p::Vector{Vector{Int64}}, bounds::Vector{Hyperrectangle})
#     extra = Vector{Vector{Int64}}()
#     return extra
# end

# function encode_partial_assignment(model::Model, nnet::Network, p::Vector{Vector{Int64}}, neurons, slack::Bool)
#     if slack
#         slack_var = Vector{Vector{Variable}}(undef, length(nnet.layers))
#         sum_slack = 0.0
#     end

#     for (i, layer) in enumerate(nnet.layers)
#         (W, b, act) = (layer.weights, layer.bias, layer.activation)
#         before_act = W * neurons[i] + b
#         if slack
#             slack_var[i] = @variable(model, [1:length(b)])
#         end
#         for j in 1:length(b)
#             if p[i][j] != 0
#                 if slack
#                     sum_slack += slack_var[i][j]
#                     if p[i][j] == 1
#                         @constraint(model, neurons[i+1][j] == before_act[j] + slack_var[i][j])
#                         @constraint(model, before_act[j] + slack_var[i][j] >= 0.0)
#                     else
#                         @constraint(model, neurons[i+1][j] == 0.0)
#                         @constraint(model, 0.0 >= before_act[j] - slack_var[i][j])
#                     end
#                 else
#                     if p[i][j] == 1
#                         @constraint(model, neurons[i+1][j] <= before_act[j])
#                     else
#                         @constraint(model, neurons[i+1][j] == 0.0)
#                         @constraint(model, 0.0 >= before_act[j])
#                     end
#                 end
#             end
#         end
#     end
#     if slack
#         return slack_var, sum_slack
#     else
#         return nothing
#     end
# end

# function get_tight_clause(problem::Problem, p::Vector{Vector{Int64}}, bounds::Vector{Hyperrectangle})
#     model = JuMP.Model(solver = optimizer)

#     neurons = init_neurons(solver, model, problem.network)
#     add_input_constraint(model, problem.input, first(neurons))
#     add_complementary_output_constraint(model, problem.output, last(neurons))
#     encode_Δ_lp(model, problem.network, bounds, neurons)
#     encode_partial_assignment(model, problem.network, p, neurons, false)

#     J = 0.0
#     for i in 1:length(p)
#         for j in 1:length(p[i])
#             if p[i][j] == 0
#                 J += neurons[i+1][j]
#             end
#         end
#     end

#     @objective(model, Min, J)
#     status = solve(model)
#     if status != :Optimal
#         return (:Infeasible, [])
#     end

#     v = getvalue(neurons)
#     tight = Vector{Int64}()
#     complete = :Complete
#     for i in 1:length(p)
#         for j in 1:length(p[i])
#             if p[i][j] == 0
#                 if v[i+1][j] > 0
#                     append!(tight, Any[get_node_id(problem.network, (i,j))])
#                 else
#                     complete = :Incomplete
#                 end
#             end
#         end
#     end
#     return (complete, tight)
# end
