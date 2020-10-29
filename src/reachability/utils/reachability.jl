##### General structure for reachability methods #####

# Performs layer-by-layer propagation
# It is called by all solvers under reachability
# TODO: also called by ReluVal and FastLin, so move to general utils (or to network.jl)
# collect_f is a fcn mapping from a tuple of the pre-activation and
# post-activation reachable sets for a layer to the desired object to be collected.
# It defaults to last which collects the post-activation reachable sets.
# Overapproximation can be incorporated into this collection function
function forward_network(solver, nnet, input;
                         collect=false, before_act=false, transformation=identity)

    if collect
        f = before_act ? first : last
        return _forward_network_collect(solver, nnet, input, transformation∘f)
    else
        @assert !before_act && transformation == identity "before_act and transformation are only supported for `collect=true`"
        return _forward_network(solver, nnet, input)
    end
end

# these two functions can instead just be exposed without the leading "_",
function _forward_network(solver, nnet, input)
    Z = input
    for layer in nnet.layers
        Ẑ, Z = forward_layer(solver, layer, Z)
    end
    return Z
end

function _forward_network_collect(solver,
                                  nnet,
                                  input,
                                  collect_f=last)
    zs = (input, input)
    elem = collect_f(zs)
    collected = [(zs = forward_layer(solver, l, zs[2]); collect_f(zs))
                 for l in nnet.layers]
    return [elem; collected]
end

# Get bounds from the reachable set propagated by a solver
function get_bounds(solver, nnet, input; before_act=false)
    f = before_act ? first : last
    # Collect the bounds for any solver.
    # If we have a list of objects then we wrap
    # them in a ConvexHullArray to perform the overapproximation
    function overapprox(set)
        if set isa AbstractVector
            return overapproximate(ConvexHullArray(set), Hyperrectangle)
        end
        return overapproximate(set, Hyperrectangle)
    end
    return _forward_network_collect(solver, nnet, input, overapprox∘f)
end

function forward_layer(solver, layer, reach)
    Ẑ = forward_linear(solver, layer, reach)
    Z = forward_act(solver, layer, Ẑ)
    return Ẑ, Z
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

# Define overapproximate for SymbolicIntervalMasks and SymbolicIntervalGradients
LazySets.overapproximate(set::_SymIntOrGrad, ::Type{Hyperrectangle}) = Hyperrectangle(low=low(set), high=high(set))
