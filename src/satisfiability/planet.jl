"""
    Planet(optimizer, eager::Bool)

Planet integrates a SAT solver (`PicoSAT.jl`) to find an activation pattern that maps a feasible input to an infeasible output.

# Problem requirement
1. Network: any depth, ReLU activation
2. Input: hyperrectangle or hpolytope
3. Output: halfspace

# Return
`BasicResult`

# Method
Binary search of activations (0/1) and pruning by optimization. Our implementation is non eager.
- `optimizer` default `GLPKSolverMIP()`;
- `eager` default `false`;

# Property
Sound and complete.

# Reference
[R. Ehlers, "Formal Verification of Piece-Wise Linear Feed-Forward Neural Networks,"
in *International Symposium on Automated Technology for Verification and Analysis*, 2017.](https://arxiv.org/abs/1705.01320)

[https://github.com/progirep/planet](https://github.com/progirep/planet)
"""
@with_kw struct Planet
    optimizer::AbstractMathProgSolver
    eager::Bool                        = false
end

Planet(x::AbstractMathProgSolver) = Planet(optimizer = x)

function solve(solver::Planet, problem::Problem)
    @assert ~solver.eager "Eager implementation not supported yet"
    # Refine bounds. The bounds are values after activation
    status, bounds = tighten_bounds(problem, solver.optimizer)
    status == :Optimal || return CounterExampleResult(:SAT)
    ψ = init_ψ(problem.network, bounds)
    δ = PicoSAT.solve(ψ)
    opt = solver.optimizer
    # Main loop to compute the SAT problem
    while δ != :unsatisfiable
        status, conflict = elastic_filtering(problem, δ, bounds, opt)
        status == :Infeasible || return CounterExampleResult(:UNSAT, conflict)
        append!(ψ, Any[conflict])
        δ = PicoSAT.solve(ψ)
    end
    return CounterExampleResult(:SAT)
end

function init_ψ(nnet::Network, bounds::Vector{Hyperrectangle})
    ψ = Vector{Vector{Int64}}(undef, 0)
    index = 0
    for i in 1:length(bounds)-1
        before_act_bound = linear_transformation(nnet.layers[i], bounds[i])
        lower = low(before_act_bound)
        upper = high(before_act_bound)
        for j in 1:length(lower)
            index += 1
            lower[j] > 0 && push!(ψ, [index])
            upper[j] < 0 && push!(ψ, [-index])
            lower[j] <= 0 <= upper[j] && push!(ψ, [index, -index])
        end
    end
    return ψ
end

function elastic_filtering(problem::Problem, δ::Vector{Vector{Bool}}, bounds::Vector{Hyperrectangle}, optimizer::AbstractMathProgSolver)
    model = Model(solver = optimizer)
    neurons = init_neurons(model, problem.network)
    add_set_constraint!(model, problem.input, first(neurons))
    add_complementary_set_constraint!(model, problem.output, last(neurons))
    encode_Δ_lp!(model, problem.network, bounds, neurons, TriangularRelaxedLP())
    LP = encode_slack_lp!(model, problem.network, neurons, δ, SlackLP())
    slack = LP.slack
    min_sum!(model, slack)
    conflict = Vector{Int64}()
    act = get_activation(problem.network, bounds)
    while true
        status = solve(model, suppress_warnings = true)
        status == :Optimal || return (:Infeasible, conflict)
        (m, index) = max_slack(getvalue(slack), act)
        m > 0.0 || return (:Feasible, getvalue(neurons[1]))
        # activated neurons get a factor of (-1)
        coeff = δ[index[1]][index[2]] ? -1 : 1
        node = coeff * get_node_id(problem.network, index)
        push!(conflict, node)
        @constraint(model, slack[index[1]][index[2]] == 0.0)
    end
end

elastic_filtering(problem::Problem, list::Vector{Int64}, bounds::Vector{Hyperrectangle}, optimizer::AbstractMathProgSolver) = elastic_filtering(problem, get_assignment(problem.network, list), bounds, optimizer)

function max_slack(x::Vector{Vector{Float64}}, act)
    m = 0.0
    index = (0, 0)
    for i in 1:length(x)
        for j in 1:length(x[i])
            if x[i][j] > m && act[i][j] == 0 # Only return undetermined nodes
                m = x[i][j]
                index = (i, j)
            end
        end
    end
    return (m, index)
end

function tighten_bounds(problem::Problem, optimizer::AbstractMathProgSolver)
    bounds = get_bounds(problem)
    model = Model(solver = optimizer)
    neurons = init_neurons(model, problem.network)
    add_set_constraint!(model, problem.input, first(neurons))
    add_complementary_set_constraint!(model, problem.output, last(neurons))
    encode_Δ_lp!(model, problem.network, bounds, neurons)

    o = min_sum!(model, neurons)
    status = solve(model, suppress_warnings = true)
    status == :Optimal || return (:Infeasible, bounds)
    lower = getvalue(neurons)

    o = max_sum!(model, neurons)
    status = solve(model, suppress_warnings = true)
    status == :Optimal || return (:Infeasible, bounds)
    upper = getvalue(neurons)

    new_bounds = Vector{Hyperrectangle}(undef, length(neurons))
    for i in 1:length(neurons)
        new_bounds[i] = Hyperrectangle(low = lower[i], high = upper[i])
    end
    return (:Optimal, new_bounds)
end

function get_assignment(nnet::Network, list::Vector{Int64})
    p = Vector{Vector{Bool}}(undef, length(nnet.layers))
    n = 0
    for (i, layer) in enumerate(nnet.layers)
        p[i] = zeros(Bool, length(layer.bias))
        for j in 1:length(p[i])
            if list[n+j] > 0
                p[i][j] = true
            end
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
#     model = Model(solver = optimizer)

#     neurons = init_neurons(solver, model, problem.network)
#     add_set_constraint!(model, problem.input, first(neurons))
#     add_complementary_output_constraint(model, problem.output, last(neurons))
#     encode_Δ_lp!(model, problem.network, bounds, neurons)
#     encode_partial_assignment(model, problem.network, p, neurons, false)

#     o = 0.0
#     for i in 1:length(p)
#         for j in 1:length(p[i])
#             if p[i][j] == 0
#                 o += neurons[i+1][j]
#             end
#         end
#     end

#     @objective(model, Min, o)
#     status = solve(model, suppress_warnings = true)
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
