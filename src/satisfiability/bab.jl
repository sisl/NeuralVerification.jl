# BaB
# Input: Hyperrectangle
# Output: Hyperrectangle
# BaB estimate whether the half space constraint can be violated or not

struct BaB
    ϵ::Float64
    optimizer::AbstractMathProgSolver
end

BaB() = BaB(0.1, GLPKSolverMIP())

function solve(solver::BaB, problem::Problem)
    (u_approx, u, x_u) = output_bound(solver, problem, :max)
    (l_approx, l, x_l) = output_bound(solver, problem, :min)
    bound = Hyperrectangle(low = [l], high = [u])
    reach = Hyperrectangle(low = [l_approx], high = [u_approx])
    return interpret_result(reach, bound, problem.output, x_l, x_u)
end

function interpret_result(reach, bound, output, x_l, x_u)
    if high(reach) > high(output) && low(reach) < low(output) 
        return ReachabilityResult(:SAT, reach)
    end
    high(bound) > high(output)    && return CounterExampleResult(:UNSAT, x_u)
    low(bound) < low(output)      && return CounterExampleResult(:UNSAT, x_l)
    return RechabilityResult(:Unknown, reach)
end

function output_bound(solver::BaB, problem::Problem, type::Symbol)
    nnet = problem.network  
    global_concrete, x_star = concrete_bound(nnet, problem.input, type)
    global_approx = approx_bound(nnet, problem.input, solver.optimizer, type)
    doms = Tuple{Any, Hyperrectangle}[(global_approx, problem.input)]
    index = ifelse(type == :max, 1, -1)
    while index * (global_approx - global_concrete) > solver.ϵ
        dom = pick_out(doms) # pick_out implements the search strategy
        subdoms = split_dom(dom[2]) # split implements the branching rule
        for i in 1:length(subdoms)
            dom_concrete, x = concrete_bound(nnet, subdoms[i], type) # Sample
            dom_approx = approx_bound(nnet, subdoms[i], solver.optimizer, type)
            if index * (dom_concrete - global_concrete) > 0
                (global_concrete, x_star) = (dom_concrete, x)
            end
            if index * (dom_approx - global_concrete) > 0
                add_domain!(doms, (dom_approx, subdoms[i]))
            end
        end 
        global_approx = doms[1][1]
    end
    return (global_approx, global_concrete, x_star)
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


function approx_bound(nnet::Network, dom::Hyperrectangle, optimizer::AbstractMathProgSolver, type::Symbol)
    bounds = get_bounds(nnet, dom)
    model = JuMP.Model(solver = optimizer)
    neurons = init_neurons(model, nnet)
    add_input_constraint(model, dom, first(neurons))
    encode_Δ_lp(model, nnet, bounds, neurons)
    index = ifelse(type == :max, 1, -1)
    J = sum(last(neurons))
    @objective(model, Max, index * J)
    status = solve(model)
    status == :Optimal && return getvalue(J)
    error("Could not find bound for dom: ", dom)
end