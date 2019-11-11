"""
    Certify(optimizer)

Certify uses semidefinite programming to compute over-approximated certificates for a neural network with only one hidden layer.

# Problem requirement
1. Network: one hidden layer, any activation that is differentiable almost everywhere whose derivative is bound by 0 and 1
2. Input: hypercube
3. Output: halfspace

# Return
`BasicResult`

# Method
Semindefinite programming.
Default `optimizer` is `SCSSolver()`.

# Property
Sound but not complete.

# Reference
[A. Raghunathan, J. Steinhardt, and P. Liang,
"Certified Defenses against Adversarial Examples,"
*ArXiv Preprint ArXiv:1801.09344*, 2018.](https://arxiv.org/abs/1801.09344)
"""
@with_kw struct Certify
    optimizer = SCS.Optimizer
end

# can pass keyword args to optimizer
# Certify(optimizer::DataType = SCSSolver; kwargs...) =  Certify(optimizer(;kwargs...))

function solve(solver::Certify, problem::Problem)
    @assert length(problem.network.layers) == 2 "Certify only handles Networks that have one hidden layer. Got $(length(problem.network.layers)) total layers"
    model = Model(solver)
    c, d = problem.output.a, problem.output.b
    W = problem.network.layers[1].weights
    v = problem.network.layers[2].weights' * c
    M = get_M(v, W)
    n = size(M, 1)
    P = @variable(model, [1:n, 1:n], PSD)
    # Compute value
    output = c' * compute_output(problem.network, problem.input.center) .- d
    epsilon = problem.input.radius[1]
    o = output .+ epsilon/4 * tr(M*P)
    # Specify problem
    @constraint(model, diag(P) .<= ones(n))
    @objective(model, Max, o)
    optimize!(model)
    if value(o) <= 0
        return BasicResult(:holds)
    else
        return BasicResult(:unknown)
    end
end

# M is used in the semidefinite program
function get_M(v::Vector, W::Matrix)
    m = W' * Diagonal(v)
    mxs, mys = size(m)
    o = ones(size(W, 2), 1)
    M = [zeros(1, 1+mxs)    o'*m;
         zeros(mxs, 1+mxs)     m;
         m'*o  m' zeros(mys, mys)]
end


