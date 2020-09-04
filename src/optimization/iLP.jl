"""
    ILP(optimizer, max_iter)

ILP iteratively solves a linearized primal optimization to compute maximum allowable disturbance.
It iteratively adds the linear constraint to the problem.

# Problem requirement
1. Network: any depth, ReLU activation
2. Input: hyperrectangle
3. Output: PolytopeComplement

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
@with_kw struct ILP <: Solver
    optimizer = GLPK.Optimizer
    iterative::Bool = true
end

function solve(solver::ILP, problem::Problem)
    nnet = problem.network
    x = problem.input.center
    model = Model(solver)
    model[:δ] = δ = get_activation(nnet, x)
    z = init_vars(model, nnet, :z, with_input=true)
    add_complementary_set_constraint!(model, problem.output, last(z))
    o = max_disturbance!(model, first(z) - problem.input.center)

    if !solver.iterative
        encode_network!(model, nnet, StandardLP())
        optimize!(model)
        termination_status(model) != OPTIMAL && return AdversarialResult(:unknown)
        x = value(first(z))
        return interpret_result(solver, x, problem.input)
    end

    encode_network!(model, nnet, LinearRelaxedLP())
    while true
        optimize!(model)
        termination_status(model) != OPTIMAL && return AdversarialResult(:unknown)
        x = value(first(z))
        matched, index = match_activation(nnet, x, δ)
        if matched
            return interpret_result(solver, x, problem.input)
        end
        add_constraint!(model, nnet, z, δ, index)
    end
end

function interpret_result(solver::ILP, x, input)
    radius = abs.(x .- center(input))

    if all(radius .>= radius_hyperrectangle(input))
        return AdversarialResult(:holds, minimum(radius))
    else
        return AdversarialResult(:violated, minimum(radius))
    end
end

function add_constraint!(model::Model,
                         nnet::Network,
                         z::Vector{Vector{VariableRef}},
                         δ::Vector{Vector{Bool}},
                         (i, j)::Tuple{Int64, Int64})
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
        matched, j = match_activation(layer, δ[i], curr_value)

        !matched && return false, (i,j)

        curr_value = layer.activation(curr_value)
    end
    return (true, nothing)
end

match_activation(L::Layer{Id}, args...) = (true, nothing)
function match_activation(L::Layer{ReLU}, δᵢ, val, tol = 1e-4)
    for (j, val) in enumerate(val)
        act = δᵢ[j]
        if act
            val < -tol && return false, j
        else
            val > tol  && return false, j
        end

    end
    return (true, nothing)
end