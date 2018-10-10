using JuMP
using MathProgBase.SolverInterface
using GLPKMathProgInterface
using LazySets

include("../src/utils/activation.jl")
include("../src/utils/network.jl")
include("../src/utils/problem.jl")
include("../src/utils/util.jl")

include("../src/reachability/maxSens.jl")




struct Reluplex{O<:AbstractMathProgSolver} <: SMT
    optimizer::O
end

struct ReluplexState
    model::JuMP.Model
    b_vars::Array{Array{JuMP.Variable,1},1}
    f_vars::Array{Array{JuMP.Variable,1},1}
    relu_status::Array{Array{Int64,1},1}
    relus_left_to_fix::Array{BitArray{1},1}
    depth::Int64
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

function relu_to_fix(broken::Array{BitArray{1},1})
    for (i, layer) in enumerate(broken)
        for (j, node) in enumerate(layer)
            if node
                return(i, j)
            end 
        end
    end
end


function check_broken_relus(bs::Array{Array{JuMP.Variable,1},1}, 
        fs::Array{Array{JuMP.Variable,1},1})
    b_values = [getvalue(b) for b in bs[2:length(bs)-1]]
    f_values = [getvalue(f) for f in fs]
    
    return [(x[2] .== 0.0) .& (x[1] .> 0.0)  .| (x[2] .> 0.0) .& (x[2] .!= x[1]) for x in zip(b_values, f_values)]
end

function encode(model::Model, problem::Problem, relu_status::Array{Array{Int64,1},1})
    bs, fs = init_nnet_vars(model, problem.network)
    
    #TEST
    #bs = [[@variable(model, b11)],
    #     [@variable(model, b21),@variable(m, b22)],
    #     [@variable(model, b31),@variable(m, b32)],  
    #     [@variable(model, b41)]]

    #fs = [[@variable(model, f21),@variable(m, f22)],
    #      [@variable(model, f31),@variable(m, f32)]]
    
    for (i, layer) in enumerate(problem.network.layers)
        (W, b, act) = (layer.weights, layer.bias, layer.activation)
        
        #first layer is different
        if i == 1
            for j in 1:length(layer.bias)
               @constraint(model, -bs[2][j] + bs[1][1]*W[j] == -b[j]) 
            end
        elseif 1<i
            for j in 1:length(layer.bias) # For evey node
               # @constraint(model, -bs[i+1][j] + dot(fs[i-1],W[j,:]) == -b[j])
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
    
    # relu fix constraints
    for i in 1:length(relu_status)
       for j in 1:length(relu_status[i])
            if relu_status[i][j] == 1
                @constraint(model, bs[i+1][j] == fs[i][j])
                @constraint(model, bs[i+1][j] >= 0.0)
            elseif relu_status[i][j] == 2
                @constraint(model, bs[i+1][j] <= 0.0)
                @constraint(model, fs[i][j] == 0.0)
            end 
        end
    end
    
    @objective(m, Max, 0)
    return (bs, fs)
end


### FUNCTIONS THAT WERE NOT INCLUDED

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

function reluplexStep(step::ReluplexState)
    print(step.depth)
    print("\n")
    status = JuMP.solve(step.model)
    #print(step.model)
    
    if status == :Optimal
        #CHECK THAT RELUS ARE CORRECT, OTHERWISE START FIXING, CALL AGAIN
        found_broken_relu = false
        broken = check_broken_relus(step.b_vars, step.f_vars)

        for i in 1:length(broken)
           for j in 1:length(broken[i])
                if broken[i][j]
                    print("Broken Relu: ")
                    print(i)
                    print(j)
                    # Found a broken ReLU
                    found_broken_relu = true
                    # Can still try to fix
                    if step.relus_left_to_fix[i][j]
                        
                        m1 = Model(solver = GLPKSolverLP(method=:Exact))
                        m2 = Model(solver = GLPKSolverLP(method=:Exact))
                        
                        relu_status1 = deepcopy(step.relu_status)
                        relu_status2 = deepcopy(step.relu_status)
                        
                        new_relus_left_to_fix = deepcopy(step.relus_left_to_fix)
                        new_relus_left_to_fix[i][j] = false
                        
                        
                        
                        relu_status1[i][j] = 1
                        bs1, fs1 = encode(m1, problem, relu_status1)
                        newStep1 = ReluplexState(m1, bs1, fs1, relu_status1, new_relus_left_to_fix, step.depth +1)
                        #print("GOT TO NEW STEP 1")
                        reluplexStep(newStep1)
                        
                        relu_status2[i][j] = 2
                        bs2, fs2 = encode(m2, problem, relu_status2)
                        newStep2 = ReluplexState(m2, bs2, fs2, relu_status2, new_relus_left_to_fix, step.depth +1)
                        break
                        #print("GOT TO NEW STEP 2")
                        reluplexStep(newStep2)
                        
                    else
                    # No relus left to fix
                        return "UNSAT"
                    end
                end  
            end
            if !found_broken_relu
                print("AAAAA")
                return ("SAT", first(step.b_vars))
            end
        end
        
    elseif status == :Infeasible
        return "UNSAT"
        
    end
end

function solveReluplex(problem::Problem)
    relu_shape = [length(x.bias) for x in problem.network.layers[1:length(problem.network.layers)-1]]
    first_m = Model(solver = GLPKSolverLP(method=:Exact))
    relu_status = [zeros(Int, x) for x in relu_shape]
    first_bs, first_fs = encode(first_m, problem, relu_status)
    relus_left_to_fix = [trues(x) for x in relu_shape]
    firstStep = ReluplexState(first_m, first_bs, first_fs, relu_status, relus_left_to_fix, 1)
    
    return reluplexStep(firstStep)
end