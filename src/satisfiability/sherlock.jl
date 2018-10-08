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
    if u <= high(problem.output)[1] && l >= low(problem.output)[1]
        return Result(:SAT)
    elseif u - high(problem.output)[1] > solver.delta 
        return Result(:UNSAT, x_u)
    elseif low(problem.output)[1] - l > solver.delta
        return Result(:UNSAT, x_l)
    else
        return Result(:UNKNOWN)
    end
end

function output_bound(solver::Sherlock, problem::Problem, upper::Bool)
    x = sample(problem.input)
    while true
        (x, bound) = local_search(problem.network, x, problem.input, solver.global_solver, upper)
        bound += ifelse(upper, solver.delta, -solver.delta)
        (x_new, bound_new, feasibile) = global_search(problem.network, bound, problem.input, solver.global_solver, upper)
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

function local_search(nnet::Network, x::Vector{Float64}, inputSet::AbstractPolytope, optimizer::AbstractMathProgSolver, upper::Bool)
    act_pattern = get_activation(nnet, x)
    gradient = get_gradient(nnet, x)

    model = Model(solver = optimizer)

    neurons = init_neurons(solver, model, problem.network)
    add_input_constraint(model, problem.input, first(neurons))
    encode_lp_constraint(model, problem.network, act_pattern, neurons)

    J = gradient * neurons[1]
    @objective(model, ifelse(upper, Max, Min), J[1])

    JuMP.solve(model)

    x_new = getvalue(neurons[1])
    bound_new = compute_output(nnet, x_new)
    return (x_new, bound_new[1])
end

function global_search(nnet::Network, bound::Float64, inputSet::AbstractPolytope, optimizer::AbstractMathProgSolver, upper::Bool)
    # Call Reverify for global search
    outputSet = HPolytope()
    if upper
        h = HalfSpace([1.0], bound)
    else
        h = HalfSpace([-1.0], -bound)
    end
    addconstraint!(outputSet, h)
    problem = Problem(nnet, inputSet, outputSet)
    solver = Reverify(optimizer)
    result = solve(solver, problem)
    if result.status == :SAT
        x = result.counter_example
        bound = compute_output(nnet, x)
        return (x, bound[1], true)
    else
        return ([], 0.0, false)
    end
end