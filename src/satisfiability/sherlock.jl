# Sherlock
# Input constraint HPolytope
# Output: 1D Hyperrectangle
struct Sherlock
    ϵ::Float64
    optimizer::AbstractMathProgSolver
end

Sherlock() = Sherlock(1.0, GLPKSolverMIP())

function solve(solver::Sherlock, problem::Problem)
    (x_u, u) = output_bound(solver, problem, :max)
    (x_l, l) = output_bound(solver, problem, :min)
    bound = Hyperrectangle(low = [l], high = [u])
    reach = Hyperrectangle(low = [l - solver.ϵ], high = [u + solver.ϵ])
    return interpret_result(reach, bound, problem.output, x_l, x_u) # This function is defined in bab.jl
end

function output_bound(solver::Sherlock, problem::Problem, type::Symbol)
    x = sample(problem.input)
    while true
        (x, bound) = local_search(solver, problem, x, type)
        bound += ifelse(type == :max, solver.ϵ, -solver.ϵ)
        (x_new, bound_new, feasibile) = global_search(problem, solver, bound, type)
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

function local_search(solver::Sherlock, problem::Problem, x::Vector{Float64}, type::Symbol)
    nnet = problem.network

    act_pattern = get_activation(nnet, x)
    gradient = get_gradient(nnet, x)

    model = Model(solver = solver.optimizer)

    neurons = init_neurons(model, nnet)
    add_input_constraint(model, problem.input, first(neurons))
    encode_lp(model, nnet, act_pattern, neurons)

    J = gradient * neurons[1]
    index = ifelse(type == :max, 1, -1)
    @objective(model, Max, index * J[1])

    solve(model)

    x_new = getvalue(neurons[1])
    bound_new = compute_output(nnet, x_new)
    return (x_new, bound_new[1])
end

global_search(problem::Problem, solver::Sherlock, bound::Float64, type::Symbol) = global_search(problem.network, problem.input, solver.optimizer, bound, type)

function global_search(nnet::Network, input::AbstractPolytope, optimizer::AbstractMathProgSolver, bound::Float64, type::Symbol)
    # Call Reverify for global search
    index = ifelse(type == :max, 1.0, -1.0)
    h = HalfSpace([index], index * bound)
    output_set = HPolytope([h])

    problem = Problem(nnet, input, output_set)
    solver  = NSVerify(optimizer)
    result  = solve(solver, problem)
    if result.status == :UNSAT
        x = result.counter_example
        bound = compute_output(nnet, x)
        return (x, bound[1], true)
    else
        return ([], 0.0, false)
    end
end