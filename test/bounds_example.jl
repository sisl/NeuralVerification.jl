using NeuralVerification, LazySets, GLPK

# Take a vector of hyperrectangles, representing the bounds at each layer,
# and convert this to a single vectro of lower or upper bounds. The flag
# lower tells you whether to return lower or upper bounds. 
function all_bounds(bounds::Vector{<:Hyperrectangle}; lower=false)
    grouped_bounds = Vector{Float64}()
    for (i, hyperrectangle) in enumerate(bounds)
        if (lower)
            append!(grouped_bounds, low(hyperrectangle))
        else
            append!(grouped_bounds, high(hyperrectangle))
        end
    end
    return grouped_bounds
end

# Load an example network 
network_file = joinpath(@__DIR__, "networks/ACASXU_run2a_1_1_tiny_4.nnet")
nnet = NeuralVerification.read_nnet(network_file)

# Create a hypercube input set centered at 0 with radius given by radius.
num_inputs = size(nnet.layers[1].weights, 2)
radius = 0.01
input_set = Hyperrectangle(zeros(num_inputs), radius*ones(num_inputs))

# Set up a problem to use 
problem = Problem(nnet, input_set, nothing)

# Compute the bounds from an LP relaxation at each node (Planet)
planet_status, planet_bounds = NeuralVerification.tighten_bounds(problem, GLPK.Optimizer)

# Compute the groundtruth bounds. Note that you can pass in existing bounds 
# using the bounds keyword. These are assumed to be pre-activation bounds.
# These bounds can greatly impact the runtime of this. 
# Note, there is an issue with the application of bounds, as described in 
# https://github.com/sisl/NeuralVerification.jl/issues/205 
# and a workaround where bounds are applied after the fact can be found at 
# https://github.com/sisl/NeuralPriorityOptimizer.jl/blob/b212cfbebbf6006adcb70b811be269667e58058b/src/additional_optimizers.jl#L59-L118
groundtruth_status, groundtruth_bounds = NeuralVerification.tighten_bounds(problem, GLPK.Optimizer; bounds=planet_bounds, encoding=NeuralVerification.BoundedMixedIntegerLP())

# Interval arithmetic bounds. 
interval_arithmetic_bounds = NeuralVerification.get_bounds(nnet, input_set; before_act=true)


# Compute bounds from Ai2 with zonotpes and Ai2 with boxes.
# Note: Ai2z bounds end up being equivalent to convdual bounds empirically.  
# Note: Ai2 Boxes is equivalent to interval arithmetic.
ai2z_bounds = NeuralVerification.get_bounds(Ai2z(), nnet, input_set; before_act=true)
ai2_box_bounds = NeuralVerification.get_bounds(Box(), nnet, input_set; before_act=true)

# Convdual bounds. 
convdual_bounds = NeuralVerification.get_bounds(ConvDual(), nnet, input_set; before_act=true)

# Symbolic bound tightening in Reluval
# we first must wrap the input into a symbolic mask to be passed into 
# get_bounds with reluval 
reluval_input = NeuralVerification.init_symbolic_mask(input_set)
reluval_bounds = NeuralVerification.get_bounds(ReluVal(), nnet, reluval_input; before_act=true)

# Compute bounds from symbolic bound tightening in Neurify
# we again first wrap the input before computing bounds 
neurify_input = NeuralVerification.init_symbolic_grad(input_set)
neurify_bounds = NeuralVerification.get_bounds(Neurify(), nnet, neurify_input; before_act=true)

# We can use other bounds when initializing Planet's tighten bounds. 
# for instance, here we use ai2z's bounds and then tighten them with planet. 
# this intermediate step can make the tightening process faster, as the LPs 
# solved at each step are more tightly constrained. It still yields 
# the same bounds as planet initialized with other bounds. 
status, ai2z_planet_bounds = NeuralVerification.tighten_bounds(problem, GLPK.Optimizer; bounds=ai2z_bounds)


# For ease of comparing the bounds, we'll use a helper function all_bounds
# that combines all layers' bounds into a single vector.  
num_outputs = length(nnet.layers[end].bias)

println("Lower bounds of last layer")
println("==========================")
println("Ground truth (MIP): ", all_bounds(groundtruth_bounds; lower=true)[end-num_outputs+1:end])
println("interval arithmetic bounds: ", all_bounds(interval_arithmetic_bounds; lower=true)[end-num_outputs+1:end])
println("Ai2 Box: ", all_bounds(ai2_box_bounds; lower=true)[end-num_outputs+1:end])
println("ConvDual: ", all_bounds(convdual_bounds; lower=true)[end-num_outputs+1:end])
println("Ai2z: ", all_bounds(ai2z_bounds; lower=true)[end-num_outputs+1:end])
println("ReluVal: ", all_bounds(reluval_bounds; lower=true)[end-num_outputs+1:end])
println("Neurify: ", all_bounds(neurify_bounds; lower=true)[end-num_outputs+1:end])
println("Planet: ", all_bounds(planet_bounds; lower=true)[end-num_outputs+1:end])
println("Planet: ", all_bounds(ai2z_planet_bounds; lower=true)[end-num_outputs+1:end])


println("\nUpper bounds of last layer")
println("==========================")
println("Ground truth (MIP): ", all_bounds(groundtruth_bounds)[end-num_outputs+1:end])
println("interval arithmetic bounds: ", all_bounds(interval_arithmetic_bounds)[end-num_outputs+1:end])
println("Ai2 Box: ", all_bounds(ai2_box_bounds)[end-num_outputs+1:end])
println("ConvDual: ", all_bounds(convdual_bounds)[end-num_outputs+1:end])
println("Ai2z: ", all_bounds(ai2z_bounds)[end-num_outputs+1:end])
println("ReluVal: ", all_bounds(reluval_bounds)[end-num_outputs+1:end])
println("Neurify: ", all_bounds(neurify_bounds)[end-num_outputs+1:end])
println("Planet: ", all_bounds(planet_bounds)[end-num_outputs+1:end])
println("Ai2z then Planet: ", all_bounds(ai2z_planet_bounds)[end-num_outputs+1:end])
