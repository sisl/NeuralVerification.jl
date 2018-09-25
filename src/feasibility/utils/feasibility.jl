abstract type Feasibility end

import JuMP: GenericAffExpr

# General structure for Feasibility Problems
function solve(solver::Feasibility, problem::Problem)
    model = JuMP.Model(solver = solver.optimizer)
    var = encode(solver, model, problem)
    status = JuMP.solve(model)
    return interpret_result(solver, status, var)
end

# Presolve to determine the bounds of variables
# This function calls maxSens to compute the bounds
# Bounds are computed AFTER activation function
function get_bounds(problem::Problem)
    solver = MaxSens()
    bounds = Vector{Hyperrectangle}(length(problem.network.layers) + 1)
    bounds[1] = problem.input
    for (i, layer) in enumerate(problem.network.layers)
        bounds[i+1] = forward_layer(solver, layer, bounds[i])
    end
    return bounds
end

#=
Initialize JuMP variables corresponding to neurons and deltas of network for problem
=#
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

function absolute{V<:GenericAffExpr}(v::V)
    m = first(v).m
    @variable(m, aux >= 0)
    @constraint(m, aux >= v)
    @constraint(m, aux >= -v)
    return aux
end

function absolute{V<:GenericAffExpr}(v::Array{V})
    m = first(first(v).vars).m
    @variable(m, aux[1:length(v)] >= 0)
    @constraint(m, aux .>= v)
    @constraint(m, aux .>= -v)
    return aux
end

#=
Add input/output constraints to model
=#
function add_complementary_output_constraint(model::Model, output::AbstractPolytope, neuron_vars::Vector{Variable})
    out_A, out_b = tosimplehrep(output)
    # Needs to take the complementary of output constraint  
    n = length(out_b)
    if n == 1
        # Here the output constraint is a half space
        # So the complementary is just out_A * y .> out_b
        @constraint(model, -out_A * neuron_vars .<= -out_b)
    else
        # Here the complementary is a union of different constraints
        # We use binary variable to encode the union of constraints
        out_deltas = @variable(model, [1:n], Bin)
        @constraint(model, sum(out_deltas) == 1)
        for i in 1:n
            @constraint(model, -out_A[i, :]' * neuron_vars * out_deltas[i] <= -out_b[i] * out_deltas[i])
        end
    end
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