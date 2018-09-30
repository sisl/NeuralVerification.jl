abstract type SMT end

import JuMP: GenericAffExpr

# General structure for Feasibility Problems
function solve(solver::Feasibility, problem::Problem)
    model = JuMP.Model(solver = solver.optimizer)
    var = encode(solver, model, problem)
    status = JuMP.solve(model)
    return interpret_result(solver, status, var)
end

function get_bounds(problem::Problem)
    solver = MaxSens()
    bounds = Vector{Hyperrectangle}(length(problem.network.layers) + 1)
    bounds[1] = problem.input
    for (i, layer) in enumerate(problem.network.layers)
        bounds[i+1] = forward_layer(solver, layer, bounds[i])
    end
    return bounds
end

# NEED TO FIX DELTAS
function init_nnet_vars(solver::Feasibility, model::Model, network::Network)
    layers = network.layers
    neurons = Vector{Vector{Variable}}(length(layers) + 1) # +1 for input layer
    deltas  = Vector{Vector{Variable}}(length(layers) + 1)
    # input layer is treated differently from other layers
    input_layer_n = size(first(layers).weights, 2)
    all_layers_n  = [length(l.bias) for l in layers]
    insert!(all_layers_n, 1, input_layer_n)

    for (i, n) in enumerate(all_layers_n)
        neurons[i] = @variable(model, [1:n]) # To do: name the variables
        deltas[i]  = @variable(model, [1:n], Bin)
    end

    return neurons, deltas
end

# NEED TO IMPLEMENT
function add_slack_variables(model::Model, output::AbstractPolytope, neuron_vars::Vector{Variable})
    return nothing
end

function add_input_constraint(model::Model, input::HPolytope, neuron_vars::Vector{Variable})
    in_A,  in_b  = tosimplehrep(input)
    @constraint(model,  in_A * neuron_vars .<= in_b)
    return nothing
end

function add_input_constraint(model::Model, input::Hyperrectangle, neuron_vars::Vector{Variable})
    @constraint(model,  neuron_vars .<= high(input))
    @constraint(model,  neuron_vars .>= low(input))
    return nothing
end

function add_output_constraint(model::Model, output::AbstractPolytope, neuron_vars::Vector{Variable})
    out_A, out_b = tosimplehrep(output)
    @constraint(model, out_A * neuron_vars .<= out_b)
    return nothing
end