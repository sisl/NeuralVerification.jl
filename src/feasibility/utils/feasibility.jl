abstract type Feasibility end

# General structure for Feasibility Problems
function solve(solver::Feasibility, problem::Problem)
    model = JuMP.Model(solver = solver.optimizer)
    encode(solver, model, problem)
    status = JuMP.solve(model)
    return interpret_result(solver, status)
end

