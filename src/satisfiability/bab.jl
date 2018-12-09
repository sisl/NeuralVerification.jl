# BaB
# Input: Hyperrectangle
# Output: Half space
# BaB estimate whether the half space constraint can be violated or not

struct BaB
    ϵ::Float64
    optimizer::AbstractMathProgSolver
end

BaB() = BaB(0.1, GLPKSolverMIP())

function solve(solver::BaB, problem::Problem)
    # Modify the network to output c y - d
    # Constraint needs to satisfy max(c y - d) < 0
    c, d = tosimplehrep(problem.output)
    last_layer = Layer(c, -d, Id())
    nnet = Network(vcat(problem.network.layers, last_layer))

    global_lb, x_star = concrete_bound(nnet, problem.input, :max)
    global_ub = approx_bound(nnet, problem.input, solver.optimizer, :max)
    
    doms = Tuple{Any, Hyperrectangle}[(global_ub, problem.input)]
    while global_ub - global_lb > solver.ϵ && global_up > 0 && global_lb <= 0
        dom = pick_out(doms) # pick_out implements the search strategy
        subdoms = split_dom(dom[2]) # split implements the branching rule
        for i in 1:length(subdoms)
            dom_lb, x = concrete_bound(nnet, subdoms[i], :max) # Sample
            dom_ub = approx_bound(nnet, subdoms[i], solver.optimizer, :max)
            dom_lb > global_lb && ((global_lb, x_star) = (dom_lb, x))
            dom_ub > global_lb && add_domain!(doms, (dom_ub, subdoms[i]))
        end 
        global_ub = doms[1][1]
    end

    global_ub <= 0 && return CounterExampleResult(:SAT)
    global_lb > 0 && return CounterExampleResult(:UNSAT, x_star)
    return CounterExampleResult(:Unknown)
end

# Always pick the first dom
function pick_out(doms)
    return doms[1]
end

function add_domain!(doms::Vector{Tuple{Float64, Hyperrectangle}}, new::Tuple{Float64, Hyperrectangle})
    rank = length(doms) + 1
    for i in 1:length(doms)
        if new[1] >= doms[i][1]
            rank = i
            break
        end
    end
    insert!(doms, rank, new)
end

# Always split the longest input dimention
# To do: merge with "split_input" in reluVal
function split_dom(dom::Hyperrectangle)
    max_value, index_to_split = findmax(dom.radius)
    return split_interval(dom, index_to_split)
end

# For simplicity
function concrete_bound(nnet::Network, subdom::Hyperrectangle, type::Symbol)
    points = [subdom.center, low(subdom), high(subdom)]
    values = Vector{Float64}(undef, 0)
    for p in points
        push!(values, sum(compute_output(nnet, p)))
    end
    value, index = ifelse(type == :min, findmin(values), findmax(values))
    return (value, points[index])
end


function approx_bound(nnet::Network, subdom::Hyperrectangle, optimizer::AbstractMathProgSolver, type::Symbol)
    bounds = get_bounds(nnet, subdom)
    model = JuMP.Model(solver = optimizer)
    neurons = init_neurons(model, nnet)
    add_input_constraint(model, subdom, first(neurons))
    encode_Δ_lp(model, nnet, bounds, neurons)
    J = sum(last(neurons)) # there is only one output node
    ifelse(type == :min, @objective(model, Min, J), @objective(model, Max, J))
    status = solve(model)
    status == :Optimal && return getvalue(J)
    error("Could not find lower bound for subdom: ", subdom)
end