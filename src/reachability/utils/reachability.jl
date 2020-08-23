# General structure for reachability methods

# Performs layer-by-layer propagation
# It is called by all solvers under reachability
# TODO: also called by ReluVal and FastLin, so move to general utils (or to network.jl)
function forward_network(solver, nnet::Network, input)
    reach = input
    for layer in nnet.layers
        reach = forward_layer(solver, layer, reach)
    end
    return reach
end

# Checks whether the reachable set belongs to the output constraint
# It is called by all solvers under reachability
# Note vertices_list is not defined for HPolytope: to be defined
function check_inclusion(reach::Vector{<:LazySet}, output)
    for poly in reach
        issubset(poly, output) || return ReachabilityResult(:violated, reach)
    end
    return ReachabilityResult(:holds, reach)
end

function check_inclusion(reach::P, output) where P<:LazySet
    return ReachabilityResult(issubset(reach, output) ? :holds : :violated, [reach])
end

# return a vector so that append! is consistent with the relu forward_partition
forward_partition(act::Id, input) = [input]

function forward_partition(act::ReLU, input)
    N = dim(input)
    N > 30 && @warn "Got dim(X) == $N in `forward_partition`. Expecting 2ᴺ = $(2^big(N)) output sets."

    output = HPolytope{Float64}[]

    for h in 0:(big"2"^N)-1
        P = Diagonal(1.0.*digits(h, base = 2, pad = N))
        orthant = HPolytope(Matrix(I - 2.0P), zeros(N))
        S = intersection(input, orthant)
        if !isempty(S)
            push!(output, linear_map(P, S))
        end
    end
    return output
end



###################### For ReluVal and Neurify ######################

struct SymbolicInterval{F<:AbstractPolytope}
    Low::Matrix{Float64}
    Up::Matrix{Float64}
    domain::F
end

# Data to be passed during forward_layer
struct SymbolicIntervalGradient{F<:AbstractPolytope, N<:Real}
    sym::SymbolicInterval{F}
    LΛ::Vector{Vector{N}}
    UΛ::Vector{Vector{N}}
end
# Data to be passed during forward_layer
const SymbolicIntervalMask = SymbolicIntervalGradient{<:Hyperrectangle, Bool}

function _init_symbolic_grad_general(domain, N)
    n = dim(domain)
    I = Matrix{Float64}(LinearAlgebra.I(n))
    Z = zeros(n)
    symbolic_input = SymbolicInterval([I Z], [I Z], domain)
    symbolic_mask = SymbolicIntervalGradient(symbolic_input,
                                             Vector{Vector{N}}(),
                                             Vector{Vector{N}}())
end
function init_symbolic_grad(domain)
    VF = Vector{HalfSpace{Float64, Vector{Float64}}}
    domain = HPolytope(VF(constraints_list(domain)))
    _init_symbolic_grad_general(domain, Float64)
end
function init_symbolic_mask(interval)
    _init_symbolic_grad_general(interval, Bool)
end



domain(sym::SymbolicInterval) = sym.domain
domain(grad::SymbolicIntervalGradient) = domain(grad.sym)
_sym(sym::SymbolicInterval) = sym
_sym(grad::SymbolicIntervalGradient) = grad.sym

upper(sym::SymbolicInterval) = AffineMap(@view(sym.Up[:, 1:end-1]), sym.domain, @view(sym.Up[:, end]))
lower(sym::SymbolicInterval) = AffineMap(@view(sym.Low[:, 1:end-1]), sym.domain, @view(sym.Low[:, end]))
upper(grad::SymbolicIntervalGradient) = upper(grad.sym)
lower(grad::SymbolicIntervalGradient) = lower(grad.sym)

upper_bound(a::AbstractVector, set::LazySet) = a'σ(a, set)
lower_bound(a::AbstractVector, set::LazySet) = a'σ(-a, set) # ≡ -ρ(-a, set)
bounds(a::AbstractVector, set::LazySet) = (a'σ(-a, set), a'σ(a, set))  # (lower, upper)

upper_bound(S::LazySet, j::Integer) = upper_bound(Arrays.SingleEntryVector(j, dim(S), 1.0), S)
lower_bound(S::LazySet, j::Integer) = lower_bound(Arrays.SingleEntryVector(j, dim(S), 1.0), S)
bounds(S::LazySet, j::Integer) = (lower_bound(S, j), upper_bound(S, j))

const _SymIntOrGrad = Union{SymbolicInterval, SymbolicIntervalGradient}
LazySets.dim(sym::_SymIntOrGrad) = size(_sym(sym).Up, 1)
LazySets.high(sym::_SymIntOrGrad) = [upper_bound(upper(sym), j) for j in 1:dim(sym)]
LazySets.low(sym::_SymIntOrGrad) = [lower_bound(lower(sym), j) for j in 1:dim(sym)]
# radius of the symbolic interval in the direction of the
# jth generating vector. This is not the axis aligned radius,
# or the bounding radius, but rather a radius with respect to
# a node in the network. Equivalent to the upper-upper
# bound minus the lower-lower bound
function radius(sym::_SymIntOrGrad, j::Integer)
    upper_bound(upper(sym), j) - lower_bound(lower(sym), j)
end
