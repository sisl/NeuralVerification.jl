"""
    Planet(optimizer, eager::Bool)

Planet integrates a SAT solver (`PicoSAT.jl`) to find an activation pattern that maps a feasible input to an infeasible output.

# Problem requirement
1. Network: any depth, ReLU activation
2. Input: hyperrectangle or bounded hpolytope
3. Output: PolytopeComplement

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
@with_kw struct Planet <: Solver
    optimizer = GLPK.Optimizer
    eager::Bool = false
end

function solve(solver::Planet, problem::Problem)
    @assert ~solver.eager "Eager implementation not supported yet"
    isbounded(problem.input) || UnboundedInputError("Planet does not accept unbounded input sets.")
    # Refine bounds. The bounds are values after activation
    status, bounds = tighten_bounds(problem, solver.optimizer)
    status == OPTIMAL || return CounterExampleResult(:holds)
    ψ = init_ψ(problem.network, bounds)
    δ = PicoSAT.solve(ψ)
    opt = solver.optimizer
    # Main loop to compute the SAT problem
    while δ != :unsatisfiable
        status, conflict = elastic_filtering(problem, δ, bounds, opt)
        status == INFEASIBLE || return CounterExampleResult(:violated, conflict)
        push!(ψ, conflict)
        δ = PicoSAT.solve(ψ)
    end
    return CounterExampleResult(:holds)
end

function init_ψ(nnet::Network, bounds::Vector{Hyperrectangle})
    ψ = Vector{Vector{Int64}}()
    index = 0
    for i in 1:length(bounds)-1
        index = set_activation_pattern!(ψ, nnet.layers[i], bounds[i], index)
    end
    return ψ
end
function set_activation_pattern!(ψ::Vector{Vector{Int64}}, L::Layer{ReLU}, bound::Hyperrectangle, index::Int64)
    before_act_bound = approximate_affine_map(L, bound)
    lower = low(before_act_bound)
    upper = high(before_act_bound)
    for j in 1:length(lower)
        index += 1
        lower[j] > 0 && push!(ψ, [index])
        upper[j] < 0 && push!(ψ, [-index])
        lower[j] <= 0 <= upper[j] && push!(ψ, [index, -index])
    end
    return index
end
function set_activation_pattern!(ψ::Vector{Vector{Int64}}, L::Layer{Id}, bound::Hyperrectangle, index::Int64)
    n = n_nodes(L)
    for j in 1:n
        index += 1
        push!(ψ, [index])
    end
    return index
end


function elastic_filtering(problem::Problem, δ::Vector{Vector{Bool}}, bounds::Vector{Hyperrectangle}, optimizer)
    model = Model(optimizer)
    neurons = init_neurons(model, problem.network)
    add_set_constraint!(model, problem.input, first(neurons))
    add_complementary_set_constraint!(model, problem.output, last(neurons))
    encode_network!(model, problem.network, neurons, bounds, TriangularRelaxedLP())
    SLP = encode_network!(model, problem.network, neurons, δ, SlackLP())
    slack = SLP.slack
    min_sum!(model, slack)
    conflict = Vector{Int64}()
    act = get_activation(problem.network, bounds)
    while true
        optimize!(model)
        termination_status(model) == OPTIMAL || return (INFEASIBLE, conflict)
        (m, index) = max_slack(value.(slack), act)
        m > 0.0 || return (:Feasible, value.(neurons[1]))
        # activated neurons get a factor of (-1)
        coeff = δ[index[1]][index[2]] ? -1 : 1
        node = coeff * get_node_id(problem.network, index)
        push!(conflict, node)
        @constraint(model, slack[index[1]][index[2]] == 0.0)
    end
end

function elastic_filtering(problem::Problem,
                           list::Vector{Int64},
                           bounds::Vector{Hyperrectangle},
                           optimizer)
    return elastic_filtering(problem,
                             get_assignment(problem.network, list),
                             bounds,
                             optimizer)
end

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

# Only use tighten_bounds for feasibility check
# pre_activation_bounds tells whether (i) the bounds passed in, and (ii) the bounds returned should be pre-activation bounds.
function tighten_bounds(problem::Problem, optimizer; pre_activation_bounds=false, use_output_constraints=true, bounds = get_bounds(problem, !pre_activation_bounds))
    model = Model(optimizer)
    neurons = init_neurons(model, problem.network)
    add_set_constraint!(model, problem.input, first(neurons))
    if (use_output_constraints)
        add_complementary_set_constraint!(model, problem.output, last(neurons))
    end
    encode_network!(model, problem.network, neurons, bounds, TriangularRelaxedLP(); pre_activation=pre_activation_bounds)

    new_bounds = deepcopy(bounds)
    new_bounds[1] = problem.input
    for i in 2:length(neurons)
        layer = problem.network.layers[i-1]
        lower = low(bounds[i])
        upper = high(bounds[i])
        for j in 1:length(neurons[i])
            neuron = neurons[i][j]

            # Have the objective be z (post-activation) or z_hat (pre-activation)
            if (pre_activation_bounds)
                objective = dot(problem.network.layers[i-1].weights[j, :], neurons[i-1]) + problem.network.layers[i-1].bias[j]
            else
                objective = neuron
            end

            # Find the lower bound
            @objective(model, Min, objective)
            optimize!(model)
            if (use_output_constraints == false)
                # should never have an infeasible status if you have no output constraints
                @assert termination_status(model) != INFEASIBLE "Tighten bounds should always be feasible without output consrainits"
            end
            termination_status(model) == OPTIMAL || return (INFEASIBLE, bounds)
            lower[j] = value(objective)

            # Find the upper bound
            @objective(model, Max, objective)
            optimize!(model)
            upper[j] = value(objective)

            # Update the bounds and re-generate the model.
            new_bounds[i] = Hyperrectangle(low = lower, high = upper)
            model = Model(optimizer)
            neurons = init_neurons(model, problem.network)
            add_set_constraint!(model, problem.input, first(neurons))
            if (use_output_constraints)
                add_complementary_set_constraint!(model, problem.output, last(neurons))
            end
            encode_network!(model, problem.network, neurons, new_bounds, TriangularRelaxedLP(); pre_activation=pre_activation_bounds)
        end
        new_bounds[i] = Hyperrectangle(low = lower, high = upper)
    end
    return (OPTIMAL, new_bounds)
end


function get_assignment(nnet::Network, list::Vector{Int64})
    p = Vector{Vector{Bool}}(undef, length(nnet.layers))
    ℓ_start = 1
    for (i, layer) in enumerate(nnet.layers)
        ℓ_next = ℓ_start + n_nodes(layer)
        p[i] = get_assignment(layer, list[ℓ_start:ℓ_next-1])
        ℓ_start = ℓ_next
    end
    return p
end
get_assignment(L::Layer{ReLU}, list::Vector{Int64}) = list .> 0
get_assignment(L::Layer{Id},   list::Vector{Int64}) = trues(length(list))

function get_node_id(nnet::Network, x::Tuple{Int64, Int64})
    # All the nodes in the previous layers
    n = sum(n_nodes.(nnet.layers[1:x[1]-1]))

    # Plus the previous nodes in the current layer
    return n + x[2]
end

# NOTE: not used -
# function get_node_id(nnet::Network, n::Int64)
#     i = 0
#     j = n
#     while j > length(nnet.layers[i+1].bias)
#         i += 1
#         j = j - length(nnet.layers[i].bias)
#     end
#     return (i, j)
# end
