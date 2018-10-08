# MIPVerify
#   Include maximum activation
#   Include presolve
# Only take half space input constraint
# Computes the allowable radius of input perturbations
struct MIPVerify{O<:AbstractMathProgSolver}
    optimizer::O
end

function solve(solver::MIPVerify, problem::Problem)
    model = JuMP.Model(solver = solver.optimizer)
    neurons = init_neurons(model, problem.network)
    deltas = init_deltas(model, problem.network)
    add_complementary_output_constraint(model, problem.output, last(neurons))
    bounds = get_bounds(problem)
    encode_mip_constraint(model, problem.network, bounds, neurons, deltas)
    J = max_disturbance(model, first(neurons) - problem.input.center)
    status = JuMP.solve(model)

    if status == :Infeasible
        return Result(:SAT)
    end
    if getvalue(J) >= minimum(problem.input.radius)
        return Result(:SAT)
    else
        return Result(:UNSAT, J)
    end
end