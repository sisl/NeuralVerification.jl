# Sherlock
# Input constraint HPolytope
# Output: 1D Hyperrectangle
struct Sherlock
    global_solver::AbstractMathProgSolver
    delta::Float64
end

function solve(solver::Sherlock, problem::Problem)
    (x_u, u) = output_bound(solver, problem, true) # true for upper bound, false for lower bound
    (x_l, l) = output_bound(solver, problem, false)

    udiff = u - high(problem.output)[1] # TODO: doesn't this assume 1-d input?
    ldiff = l - low(problem.output)[1]

    udiff <= 0 <= ldiff       && return CounterExampleResult(:SAT)
    udiff > solver.delta      && return CounterExampleResult(:UNSAT, x_u)
    ldiff < solver.delta      && return CounterExampleResult(:UNSAT, x_l)

    return CounterExampleResult(:UNKNOWN)
end

function output_bound(solver::Sherlock, problem::Problem, upper::Bool)
    x = sample(problem.input)
    while true
        (x, bound) = local_search(solver, problem, x, upper)
        bound += ifelse(upper, solver.delta, -solver.delta)
        (x_new, bound_new, feasibile) = global_search(problem, solver, bound, upper)
        if feasibile
            (x, bound) = (x_new, bound_new)
        else
            return (x, bound)
        end
    end
end

# Choose the first vertex
function sample(set::AbstractPolytope)
    x = vertices_list(set)
    return x[1]
end

function local_search(solver::Sherlock, problem::Problem, x::Vector{Float64}, upper::Bool)
    nnet = problem.network

    act_pattern = get_activation(nnet, x)
    gradient = get_gradient(nnet, x)

    model = Model(solver = solver.global_solver)

    neurons = init_neurons(model, nnet)
    add_set_constraint!(model, problem.input, first(neurons))
    encode_lp(model, nnet, act_pattern, neurons)

    J = gradient * neurons[1]
    if upper
        @objective(model, Max, J[1])
    else
        @objective(model, Min, J[1])
    end


    solve(model)

    x_new = getvalue(neurons[1])
    bound_new = compute_output(nnet, x_new)
    return (x_new, bound_new[1])
end

global_search(problem::Problem, solver::Sherlock, bound::Float64, upper::Bool) = global_search(problem.network, problem.input, solver.global_solver, bound, upper)

function global_search(nnet::Network, input::AbstractPolytope, optimizer::AbstractMathProgSolver, bound::Float64, upper::Bool)
    # Call Reverify for global search
    if (upper)    h = HalfSpace([1.0], bound)
    else          h = HalfSpace([-1.0], -bound)
    end
    output_set = HPolytope([h])

    problem = Problem(nnet, input, output_set)
    solver  = Reverify(optimizer)
    result  = solve(solver, problem)
    if result.status == :UNSAT
        x = result.counter_example
        bound = compute_output(nnet, x)
        return (x, bound[1], true)
    else
        return ([], 0.0, false)
    end
end