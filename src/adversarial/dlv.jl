"""
    DLV(ϵ::Float64)

DLV searches layer by layer for counter examples in hidden layers.

# Problem requirement
1. Network: any depth, any activation (currently only support ReLU)
2. Input: Hyperrectangle
3. Output: AbstractPolytope

# Return
`CounterExampleResult`

# Method
The following operations are performed layer by layer. for layer i
1. determine a reachable set from the reachable set in layer i-1
2. determine a search tree in the reachable set by refining the search tree in layer i-1
3. Verify
    - True -> continue to layer i+1
    - False -> counter example

The argument `ϵ` is the resolution of the initial search tree. Default `1.0`.

# Property
Sound but not complete.

# Reference
[X. Huang, M. Kwiatkowska, S. Wang, and M. Wu,
"Safety Verification of Deep Neural Networks,"
in *International Conference on Computer Aided Verification*, 2017.](https://arxiv.org/abs/1610.06940)

[https://github.com/VeriDeep/DLV](https://github.com/VeriDeep/DLV)
"""
@with_kw struct DLV <: Solver
    optimizer = GLPK.Optimizer
    ϵ::Float64 = 1.0
end
# TODO: create types for the two mapping cases, since they are now both unstable and boxed
# also check out how to get performance out of closures, since that can be an issue in julia

function solve(solver::DLV, problem::Problem)
    # The list of etas
    η = get_bounds(problem)
    # The list of sample intervals
    δ = Vector{Vector{Float64}}(undef,length(η))
    δ[1] = fill(solver.ϵ, dim(η[1]))

    if issubset(last(η), problem.output)
        return CounterExampleResult(:holds)
    end

    output = compute_output(problem.network, problem.input.center)
    for (i, layer) in enumerate(problem.network.layers)
        δ[i+1] = get_manipulation(layer, δ[i], η[i+1])
        if i == length(problem.network.layers)
            mapping = x -> (x ∈ problem.output)
        else
            forward_nnet = Network(problem.network.layers[i+1:end])
            mapping = x -> (compute_output(forward_nnet, x) ∈ problem.output)
        end
        var, y = bounded_variation(η[i+1], mapping, δ[i+1])  # TODO rename "var"

        if var
            backward_nnet = Network(problem.network.layers[1:i])
            status, x = backward_map(solver, y, backward_nnet, η[1:i+1])
            if status
                return CounterExampleResult(:violated, x)
            end
        end
    end
    return ReachabilityResult(:violated, [last(η)])
end

# For simplicity, we just cut the sample interval into half
function get_manipulation(layer::Layer, δ::Vector{Float64}, bound::Hyperrectangle)
    δ_new = abs.(layer.weights) * δ ./ 2
    return δ_new
end

# Try to find an input x that arrives at output y
function backward_map(solver::DLV, y::Vector{Float64}, nnet::Network, bounds::Vector{Hyperrectangle})
    output = Hyperrectangle(y, zeros(size(y)))
    input = first(bounds)
    model = Model(solver)
    model[:bounds] = bounds
    z = init_vars(model, nnet, :z, with_input=true)
    δ = init_vars(model, nnet, :δ, binary=true)
    add_set_constraint!(model, input, first(z))
    add_set_constraint!(model, output, last(z))
    encode_network!(model, nnet, BoundedMixedIntegerLP())
    o = max_disturbance!(model, first(z) - input.center)
    optimize!(model)
    if termination_status(model) == OPTIMAL
        return (true, value(first(z)))
    else
        return (false, [])
    end
end

function bounded_variation(bound::Hyperrectangle, mapping::Function, δ::Vector{Float64})
    # step 1: check whether the boundary points have the same class
    var, y = uniform_boundary_class(bound, mapping)
    var && return (var, y)
    # step 2: check the 0-variation in all dimension
    var, y = zero_variation(bound, mapping, δ)
    var && return (var, y)
    return (false, similar(y, 0))
end

function uniform_boundary_class(bound::Hyperrectangle, mapping::Function)
    y = bound.center
    for i = 1:dim(bound)
        y[i] += bound.radius[i]
        mapping(y) || return (true, y)
        y[i] -= 2 * bound.radius[i]
        mapping(y) || return (true, y)
        y[i] += bound.radius[i]
    end
    return (false, similar(y, 0))
end

function zero_variation(bound::Hyperrectangle, mapping::Function, δ::Vector{Float64})
    y = bound.center
    for i = 1:dim(bound)
        z = deepcopy(y)
        while maximum(z - high(bound)) < 0
            z[i] += δ[i]
            mapping(z) || return (true, z)
        end
        z = deepcopy(y)
        while minimum(z - high(bound)) > 0
            z[i] -= δ[i]
            mapping(z) || return (true, z)
        end
    end
    return (false, similar(y, 0))
end
