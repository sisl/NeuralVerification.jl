abstract type Feasibility end

# General structure for Feasibility Problems
function solve(solver::Feasibility, problem::Problem)
    model = JuMP.Model(solver = solver.optimizer)
    encode(solver, model, problem)
    status = JuMP.solve(model)
    if status == :Optimal
        # To do: return adversarial case
        return Result(:False)
    end
    if status == :Infeasible
        return Result(:True)
    end
    return Result(:Undertermined)
end

