# Certify based on semidefinite relaxation
# NN should only have one layer
# This method only works for half space output constraint
# c y <= d
# Input constraint needs to be a hyperrectangle with uniform radius
struct Certify{O<:AbstractMathProgSolver} <: Feasibility
    optimizer::O
end

function encode(solver::Certify, model::Model, problem::Problem)
    if length(problem.network.layers) > 2
      error("Network should only contain one hidden layer!")
    end
    c, d = tosimplehrep(problem.output)
    v = c * problem.network.layers[2].weights
    W = problem.network.layers[1].weights
    M = get_M(v[1, :], W)
    n = size(M, 1)

    # Cone type SDP not supported
    @variable(model, P[1:n, 1:n], SDP)

    # Compute cost
    Tr = M * P
    output = c * compute_output(problem.network, problem.input.center)
    epsilon = problem.input.radius[1]
    J = output - d[1] + epsilon/4 * sum(Tr[i, i] for i in 1:n)

    # Specify problem
    @SDconstraint(model, P <= eye(n))
    @objective(model, Max, J[1])
end

# True if J < 0
# Undertermined if otherwise
function interpret_result(solver::Certify, status)
    return Result(:True)
end

# M is used in the semidefinite program
function get_M(v::Vector{Float64}, W::Matrix{Float64})
    m = W' * diagm(v)
    o = ones(size(W, 2), 1)
    M = [zeros(1, 1+size(m, 1)) o'*m;
         zeros(size(m,1), 1+size(m, 1)) m;
         m'*o m' zeros(size(m, 2), size(m, 2))]
    return M
end