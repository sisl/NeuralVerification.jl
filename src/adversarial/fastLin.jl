# Input constraint is Hyperrectangle (uniform radius)
# Output constriant is HPolytope with one constraint
"""
    FastLin(maxIter::Int64, ϵ0::Float64, accuracy::Float64)

FastLin combines reachability analysis with binary search to find maximum allowable disturbance.

# Problem requirement
1. Network: any depth, ReLU activation
2. Input: hypercube
3. Output: AbstractPolytope

# Return
`AdversarialResult`

# Method
Reachability analysis by network approximation and binary search.
- `max_iter` is the maximum iteration in search, default `10`;
- `ϵ0` is the initial search radius, default `100.0`;
- `accuracy` is the stopping criteria, default `0.1`;

# Property
Sound but not complete.

# Reference
[T.-W. Weng, H. Zhang, H. Chen, Z. Song, C.-J. Hsieh, D. Boning, I. S. Dhillon, and L. Daniel,
"Towards Fast Computation of Certified Robustness for ReLU Networks,"
*ArXiv Preprint ArXiv:1804.09699*, 2018.](https://arxiv.org/abs/1804.09699)
"""
@with_kw struct FastLin <: Solver
    maxIter::Int64    = 10
    ϵ0::Float64       = 100.0
    accuracy::Float64 = 0.1
end

function solve(solver::FastLin, problem::Problem)
    @assert is_hypercube(problem.input)

    ϵ_upper = 2 * max(solver.ϵ0, maximum(problem.input.radius))
    ϵ = fill(maximum(problem.input.radius), solver.maxIter+1)
    ϵ_lower = 0.0
    n_input = dim(problem.input)
    for i = 1:solver.maxIter
        input_bounds = Hyperrectangle(problem.input.center, fill(ϵ[i], n_input))
        output_bounds = forward_network(solver, problem.network, input_bounds)
        if issubset(output_bounds, problem.output)
            ϵ_lower = ϵ[i]
            ϵ[i+1] = (ϵ[i] + ϵ_upper) / 2
            abs(ϵ[i] - ϵ[i+1]) > solver.accuracy || break
        else
            ϵ_upper = ϵ[i]
            ϵ[i+1] = (ϵ[i] + ϵ_lower) / 2
        end
    end
    if ϵ_lower > maximum(problem.input.radius)
        return AdversarialResult(:holds, ϵ_lower)
    else
        return AdversarialResult(:violated, ϵ_lower)
    end
end

function forward_network(solver::FastLin, nnet::Network, input::Hyperrectangle)
    # use MaxSens to compute bounds:
    L, U = get_bounds(nnet, input.center, input.radius[1])
    output = Hyperrectangle(low = last(L), high = last(U))
    return output
end

# function forward_network(ϵ::Float64, nnet::Network, input::Hyperrectangle)

#     len = length(nnet.layers)

#     act_pattern = Vector{Float64}()
#     A = Vector{Matrix{Float64}}()
#     l = Vector{Vector{Float64}}()
#     u = Vector{Vector{Float64}}()

#     x0 = input.center
#     println("in FN")
#     for m in 1:len
#         println("looping $m")
#         new_l, new_u = get_next_bounds(nnet.layers, x0, ϵ, A, l, act_pattern)

#         update_A!(A, m, new_l, new_u)

#         push!(l, new_l)
#         push!(u, new_u)
#     end

#     return Hyperrectangle(low = last(l), high = last(u))
# end

# #=
# input: input data vector,
# p : ℓp norm,
# ϵ: maximum ℓp-norm perturbation
# l[k], u[k], k∈[m] : layer-wise bounds
# =#
# function get_next_bounds(layers, input_center, ϵ, A, l, act_pattern)
#     W0 = layers[1].weights
#     m = length(A) + 1   # A is of length 0 on the first pass, and is not necessary then.

#     T, H = get_TH(l, A, act_pattern)

#     ## TODO can the order of the for loops be reversed?
#     for j in 1:n_nodes(layers[m])  # for each node `j` in the mth layer:
#         vj = W0[j, :]*input_center + layers[m].bias[j]
#         μ_upper = μ_lower = 0
#         for k in 1:m-1
#             μ_upper = μ_upper - A[k][j, :] * T[k][:, j]     # Eq 6
#             μ_lower = μ_lower - A[k][j, :] * H[k][:, j]
#             vj += A[k][j, :] * layers[m].bias               # Eq 7
#         end

#         γᵁ = vj + ϵ*sum(abs, W0[j, :]) + μ_upper     #||A[0][j, :]||[q] => qnorm of A
#         γᴸ = vj + ϵ*sum(abs, W0[j, :]) + μ_lower
#     end

#     push!(act_pattern, relaxed_ReLU.(γᴸ, γᵁ))

#     return γᴸ, γᵁ
# end

# function get_TH(l, A, act_pattern)
#     T = Vector{SparseMatrixCSC{Float64}}(undef, length(l))
#     H = Vector{SparseMatrixCSC{Float64}}(undef, length(l))

#     for k in 1:length(l)  # for all previous layers, (current one not included)

#         nᵏ = length(act_pattern[k]) # number nodes in the kth layer
#         lᵏ = l[k]

#         T[k] = spzeros(length(lᵏ), nᵏ) # reset/initialize each one
#         H[k] = spzeros(length(lᵏ), nᵏ) # reset/initialize each one

#         for r in 1:length(lᵏ) # NOTE: what does the length of the lower bound represent? The dimensionality of the input space? In the paper they use Iᵏ
#             if 0.0 < act_pattern[k][r] < 1.0
#                 # A⁺ = findall(...)  ## for Julia 0.7+
#                 A⁺ᵣ = find(a->a>0, A[k][:, r])            # positive indices
#                 A⁻ᵣ = setdiff(eachindex(A[k][:, r]), A⁺)  # negative indices
#                 T[k][r, A⁺] = lᵏ[A⁺ᵣ]
#                 H[k][r, A⁻] = lᵏ[A⁻ᵣ]
#             end
#         end
#     end
#     return T, H
# end

# # function update_A!(A, m, l, u)

# #     for k in reverse(1:m-1)
# #         if k == m-1
# #             D = get_D(l[m-1], u[m-1])  # diagonal matrix according to Eq. 5
# #             A[m-1] = layers[m].weights * D
# #         else
# #             A[k] = A[m-1] * A[k]
# #         end
# #     end
# # end
# # Identical to:
# # function update_A!(A, m, l, u)
# #     D = get_D(l, u)  # diagonal matrix according to Eq. 5
# #     A[m-1] = layers[m].weights * D
# #     for k in 1:m-2
# #         A[k] = A[m-1] * A[k]
# #     end
# # end
# function update_A!(A, l, u)
#     D = Diagonal(relaxed_ReLU.(l, u))  # diagonal matrix according to Eq. 5
#     WD = layers[m].weights * D

#     map!(a->WD*a, A, A)
#     push!(A, WD)  # consider pushing I and mapping WD onto it along with everything else
# end
