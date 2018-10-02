struct Reverify{O<:AbstractMathProgSolver} <: Feasibility
    optimizer::O
    m::Float64 # The big M in the linearization
end

Reverify(x) = Reverify(x, 1000.0)

function solve(solver::Reverify, problem::Problem)
    model = JuMP.Model(solver = solver.optimizer)
    neurons = init_neurons(model, problem.network)
    deltas = init_deltas(model, problem.network)
    add_input_constraint(model, problem.input, first(neurons))
    add_complementary_output_constraint(model, problem.output, last(neurons))
    encode_mip_constraint(model, problem.network, solver.m, neurons, deltas)
    status = JuMP.solve(model)
    if status == :Optimal
        return Result(:UNSAT, getvalue(first(neurons)))
    end
    if status == :Infeasible
        return Result(:SAT)
    end
    return Result(:Unknown)
end

