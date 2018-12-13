struct ILP{O<:AbstractMathProgSolver}
    optimizer::O
    max_iter::Int64
end

ILP() = ILP(GLPKSolverMIP(), 10)

function solve(solver::ILP, problem::Problem)
    x = problem.input.center
    i = 0
    while i < solver.max_iter
        model = JuMP.Model(solver = solver.optimizer)
        act_pattern = get_activation(problem.network, x)

        neurons = init_neurons(model, problem.network)
        add_complementary_output_constraint(model, problem.output, last(neurons))
        encode_relaxed_lp(model, problem.network, act_pattern, neurons)
        J = max_disturbance(model, first(neurons) - problem.input.center)

        status = solve(model)
        if status != :Optimal
            return AdversarialResult(:Unknown)
        end
        x = getvalue(first(neurons))
        if satisfy(problem.network, x, act_pattern)
            radius = getvalue(J)
            if radius >= minimum(problem.input.radius)
                return AdversarialResult(:SAT, radius)
            else
                return AdversarialResult(:UNSAT, radius)
            end
        end
        i += 1
    end
    return AdversarialResult(:Unknown)
end

function satisfy(nnet::Network, x::Vector{Float64}, act_pattern::Vector{Vector{Bool}})
    curr_value = x
    for (i, layer) in enumerate(nnet.layers)
        curr_value = layer.weights * curr_value + layer.bias
        for (j, val) in enumerate(curr_value)
            act = act_pattern[i][j]
            act  && val < 0.0 && return false
            !act && val > 0.0 && return false
        end
        curr_value = layer.activation(curr_value)
    end
    return true
end

"""
    ILP(optimizer, max_iter)

ILP iteratively solves a linearized primal optimization to compute maximum allowable disturbance.

# Problem requirement
1. Network: any depth, ReLU activation
2. Input: hyperrectangle
3. Output: halfspace

# Return
`AdversarialResult`

# Method
Iteratively solve a linear encoding of the problem. 
Default `optimizer` is `GLPKSolverMIP()`. Default `max_iter` is `10`.

# Property
Sound but not complete.
"""
ILP