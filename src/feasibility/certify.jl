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
    output = c * compute_output(problem.network, problem.input.center) - d[1]
    epsilon = problem.input.radius[1]
    J = output + epsilon/4 * sum(Tr[i, i] for i in 1:n)

    # Specify problem
    @constraint(model, diag(P) .<= ones(n))
    @objective(model, Max, J[1])

    return J
end

# True if J < 0
# Undertermined if otherwise
function interpret_result(solver::Certify, status, J)
    # println("Upper bound: ", getvalue(J[1]))
    if getvalue(J[1]) <= 0
        return Result(:True)
    else
        return Result(:Undetermined)
    end
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