# DLV
struct DLV
    ϵ::Float64
end

function solve(solver::DLV, problem::Problem)
    # The list of etas
    η = get_bounds(problem)
    # The list of sample intervals
    δ = Vector{Vector{Float64}}(undef,length(η))
    δ[1] = fill(solver.ϵ, dim(η[1]))

    output = compute_output(problem.network, problem.input.center)
    # for layer i
    # 1. determine a region w.r.t. definition 6
    # 2. determine a minipulation set Δ w.r.t. definition 7
    # 3. Verify
    # True -> continue to i+1
    # False -> adversarial example
    for (i, layer) in enumerate(problem.network.layers)
        δ[i+1] = get_manipulation(layer, δ[i], η[i+1])
        if i == length(problem.network.layers)
            var, y = bounded_variation(η[i+1], x->(x∈problem.output), δ[i+1])
        else
            forward_nnet = Network(problem.network.layers[i+1:end])
            var, y = bounded_variation(η[i+1], x->(compute_output(forward_nnet, x)∈problem.output), δ[i+1])
        end
        if var > 0
            backward_nnet = Network(problem.network.layers[1:i])
            status, x = backward_map(y, backward_nnet, η[1:i+1])
            if status
                return CounterExampleResult(:UNSAT, x)
            end
        end
    end
    return CounterExampleResult(:SAT)
end

# For simplicity, we just cut the sample interval into half
function get_manipulation(layer::Layer, δ::Vector{Float64}, bound::Hyperrectangle)
    δ_new = abs.(layer.weights) * δ ./ 2
    return δ_new
end

# Try to find an input x that arrives at output y
function backward_map(y::Vector{Float64}, nnet::Network, bounds::Vector{Hyperrectangle})
    output = Hyperrectangle(y, zeros(size(y)))
    input = first(bounds)
    model = Model(solver = GLPKSolverMIP())
    neurons = init_neurons(model, nnet)
    deltas = init_deltas(model, nnet)
    add_input_constraint(model, input, first(neurons))
    add_output_constraint(model, output, last(neurons))
    encode_mip_constraint(model, nnet, bounds, neurons, deltas)
    J = max_disturbance(model, first(neurons) - input.center)
    status = solve(model)
    if status == :Optimal
        return (true, getvalue(first(neurons)))
    else
        return (false, [])
    end
end

# Here we implement single-path search (greedy)
# For simplicity, we partition the reachable set by dimension
function bounded_variation(bound, mapping, δ)
    var = 0
    y = bound.center
    # step 1: check whether the boundary points have the same class
    for i = 1:dim(bound)

        y[i] += bound.radius[i]
        !mapping(y) && return (1, y)

        y[i] -= 2 * bound.radius[i]
        !mapping(y) && return (1, y)
        y[i] += bound.radius[i]
    end
  
    # step 2: check the 0-variation in all dimension
    for i = 1:dim(bound)
        z = y
        while maximum(z - high(bound)) < 0
            z[i] += δ[i]
            !mapping(z) && return (1, y)
        end
        z = y
        while minimum(z - high(bound)) > 0
            z[i] -= δ[i]
            !mapping(z) && return (1, y)
        end
    end
    return (0, [])
end