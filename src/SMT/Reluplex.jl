struct Reluplex{O<:AbstractMathProgSolver} <: SMT
    optimizer::O
end

function init_nnet_vars(model::Model, network::Network)
    layers = network.layers
    #input layer and last layer have b_vars because they are unbounded
    b_vars = Vector{Vector{Variable}}(length(layers) + 1) # +1 for input layer
    #f_vars are always positive and used as front for ReLUs
    f_vars = Vector{Vector{Variable}}(length(layers) -1)
    
    input_layer_n = size(first(layers).weights, 2)
    all_layers_n  = [length(l.bias) for l in layers]
    insert!(all_layers_n, 1, input_layer_n)

    for (i, n) in enumerate(all_layers_n)
        b_vars[i] = @variable(model, [1:n]) # To do: name the variables
        if 1 < i < length(layers) + 1
            f_vars[i-1] = @variable(model, [1:n])
        end
    end
    return b_vars, f_vars
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

function encode(model::Model, problem::Problem, relustatus::Array{Array{Int64,1},1})
    #bs, fs = init_nnet_vars(model, problem.network)
    
    #TEST
    bs = [[@variable(m, b11)],
         [@variable(m, b21),@variable(m, b22)],
         [@variable(m, b31),@variable(m, b32)],  
         [@variable(m, b41)]]

    fs = [[@variable(m, f21),@variable(m, f22)],
          [@variable(m, f31),@variable(m, f32)]]
    
    #add_input_constraint(model, inputSet, bs[1])
    #add_output_constraint(model, problem.output, last(bs))

    for (i, layer) in enumerate(problem.network.layers)
        #print(string("i: ",i))
        (W, b, act) = (layer.weights, layer.bias, layer.activation)
        #before_act = W * neurons[i] + b
        
        #first layer is different
        if i == 1
            for j in 1:length(layer.bias)
               @constraint(model, -bs[2][j] + bs[1][1]*W[j] == -b[j]) 
            end
        elseif 1<i
            for j in 1:length(layer.bias) # For evey node
                @constraint(model, -bs[i+1][j] + dot(fs[i-1],W[j,:]) == -b[j])
            end
        end
    end

    # Adding linear constraints
    
    #first layer
    bounds = get_bounds(problem)
    
    
    #@constraint(model, bs[1] .<= bounds[1].center + bounds[1].radius)
    #@constraint(model, bs[1] .>= bounds[1].center - bounds[1].radius)
    
    for i in 1:length(bs)
        @constraint(model, bs[i] .<= bounds[i].center + bounds[i].radius)
        @constraint(model, bs[i] .>= bounds[i].center - bounds[i].radius)
    end
    
    # positivity contraint for f variables
    for i in 1:length(fs)
        @constraint(model, fs[i] .>= zeros(length(fs[i])))
    end
    # Objective: L∞ norm of the disturbance\
    @objective(m, Max, 0)
    return (bs, fs)
end

function check_relu_status(bs::Array{Array{JuMP.Variable,1},1}, 
        fs::Array{Array{JuMP.Variable,1},1})
    b_values = [getvalue(b) for b in bs[2:length(bs)-1]]
    f_values = [getvalue(f) for f in fs]
    return [x[1] .== x[2] for x in zip(b_values, f_values)]
end