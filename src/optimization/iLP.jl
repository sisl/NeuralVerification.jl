"""
    ILP(optimizer, max_iter)

ILP iteratively solves a linearized primal optimization to compute maximum allowable disturbance.
It iteratively adds the linear constraint to the problem.

# Problem requirement
1. Network: any depth, ReLU activation
2. Input: hyperrectangle
3. Output: halfspace or PolytopeComplement

# Return
`AdversarialResult`

# Method
Iteratively solve a linear encoding of the problem.
It only considers the linear piece of the network that has the same activation pattern as the reference input.
Default `optimizer` is `GLPKSolverMIP()`.
We provide both iterative method and non-iterative method to solve the LP problem.
Default `iterative` is `true`.

# Property
Sound but not complete.

# Reference
[O. Bastani, Y. Ioannou, L. Lampropoulos, D. Vytiniotis, A. Nori, and A. Criminisi,
"Measuring Neural Net Robustness with Constraints,"
in *Advances in Neural Information Processing Systems*, 2016.](https://arxiv.org/abs/1605.07262)
"""
@with_kw struct ILP{O<:AbstractMathProgSolver}
    optimizer::O    = GLPKSolverMIP()
    iterative::Bool = true
end

function solve(solver::ILP, problem::Problem)
    nnet = problem.network
    x = problem.input.center
    model = Model(solver = solver.optimizer)
    δ = get_activation(nnet, x)
    neurons = init_neurons(model, nnet)
    add_complementary_set_constraint!(model, problem.output, last(neurons))
    o = max_disturbance!(model, first(neurons) - problem.input.center)

    if !solver.iterative
        encode_lp!(model, nnet, neurons, δ)
        status = solve(model, suppress_warnings = true)
        status != :Optimal && return AdversarialResult(:Unknown)
        return interpret_result(solver, getvalue(o), problem.input)
    end

    encode_relaxed_lp!(model, nnet, neurons, δ)
    while true
        status = solve(model, suppress_warnings = true)
        status != :Optimal && return AdversarialResult(:Unknown)
        x = getvalue(first(neurons))
        matched, index = match_activation(nnet, x, δ)
        if matched
            return interpret_result(solver, getvalue(o), problem.input)
        end
        add_constraint!(model, nnet, neurons, δ, index)
    end
end

function interpret_result(solver::ILP, o, input)
    if o >= maximum(input.radius)
        return AdversarialResult(:holds, o)
    else
        return AdversarialResult(:violated, o)
    end
end

function add_constraint!(model::Model,
                         nnet::Network,
                         z::Vector{Vector{Variable}},
                         δ::Vector{Vector{Bool}},
                         index::Tuple{Int64, Int64})
    i, j = index
    layer = nnet.layers[i]
    val = layer.weights[j, :]' * z[i] + layer.bias[j]
    if δ[i][j]
        @constraint(model, val >= 0.0)
    else
        @constraint(model, val <= 0.0)
    end
end

function match_activation(nnet::Network, x::Vector{Float64}, δ::Vector{Vector{Bool}})
    curr_value = x
    for (i, layer) in enumerate(nnet.layers)
        curr_value = layer.weights * curr_value + layer.bias
        for (j, val) in enumerate(curr_value)
            act = δ[i][j]
            if act && val < -0.0001 # Floating point operation
                return (false, (i, j))
            end
            if !act && val > 0.0001 # Floating point operation
                return (false, (i, j))
            end
        end
        curr_value = layer.activation(curr_value)
    end
    return (true, (0, 0))
end
