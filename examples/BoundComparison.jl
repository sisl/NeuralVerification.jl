using LazySets
using PGFPlots
using Random
using NeuralVerification
using GLPK
using Gurobi
using NPZ

function get_groundtruth_bounds(network::Network, input_set::Hyperrectangle)
    solver = MIPVerify(optimizer=Gurobi.Optimizer)
    num_layers = length(network.layers) + 1 # include the input and output layers
    bounds = Vector{Hyperrectangle}(undef, num_layers)

    # Get tightened bounds to help with the solving process
    #status, LP_bounds = NeuralVerification.tighten_bounds(Problem(network, input_set, Nothing), Gurobi.Optimizer; pre_activation=false, use_output_constraints=false)
    reach, ai2z_bounds = forward_network(Ai2z(), nnet, input_set; get_bounds=true)
    ai2z_bounds_post_activation = Vector{Hyperrectangle}(undef, length(ai2z_bounds))
    ai2z_bounds_post_activation[1] = ai2z_bounds[1] # The input bounds should match
    for i = 2:length(ai2z_bounds)
        if (network.layers[i-1].activation == NeuralVerification.Id())
            ai2z_bounds_post_activation[i] = ai2z_bounds[i]
        elseif (network.layers[i-1].activation == NeuralVerification.ReLU())
            ai2z_bounds_post_activation[i] = Hyperrectangle(low=max.(low(ai2z_bounds[i]), 0.0), high=max.(high(ai2z_bounds[i]), 0.0))
        else
            @assert false "Unsupported activation for ground truth bounds"
        end
    end
    external_bounds = ai2z_bounds_post_activation

    bounds[1] = input_set
    for layer_index = 2:num_layers
        println("Finding bounds for layer: ", layer_index)
        # Number of nodes in the layer_index + 1st layer (including input as the 1st layer)
        num_nodes = size(network.layers[layer_index-1].weights, 1)
        lower_bounds = Vector{Float64}(undef, num_nodes)
        upper_bounds = Vector{Float64}(undef, num_nodes)
        for node_index = 1:num_nodes
            println("Finding values for layer ", layer_index, " node ", node_index)
            lower_bounds[node_index], upper_bounds[node_index] = NeuralVerification.get_bounds_for_node(solver, network, input_set, layer_index, node_index; bounds=external_bounds)
        end
        bounds[layer_index] = Hyperrectangle(low=lower_bounds, high=upper_bounds)
    end
    return bounds
end

layer_sizes = [1, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 1]
filename_start = "mnist_10x20"
use_mnist = true
# If we want to use an mnist example, then read in the network and
# create a query with the desired radius
if (use_mnist)
    nnet = NeuralVerification.read_nnet("/Users/castrong/Desktop/Research/NeuralOptimization.jl/Networks/MNIST/mnist10x20.nnet")
    num_layers = length(nnet.layers) + 1 # number of layers including input and output
    num_inputs = size(nnet.layers[1].weights, 2)
    num_outputs = length(nnet.layers[end].bias)

    # Load a file to use for your center input, and create a hypercube
    center_input = transpose(npzread("/Users/castrong/Downloads/Bound_Sample_With_Input/MNISTlabel_0_index_0_.npy")) # Transpose for AutoTaxi - transpose(npzread(example_input))
    input_radius = 0.004
    # Can't truncate based on upper and lower input sizes because leads to uneven radius!!!
    input_set = Hyperrectangle(low=(vec(center_input)[:] - input_radius * ones(num_inputs)), high=(vec(center_input)[:] + input_radius * ones(num_inputs))) # center and radius
    output_set = PolytopeComplement(HalfSpace([1.0, -1.0, 0, 0, 0, 0, 0, 0, 0, 0], 0.0)) # try to give a halfspace that doesn't give too much information
else
    nnet = NeuralVerification.make_random_network(layer_sizes; rng=MersenneTwister(0))
    num_layers = length(nnet.layers) + 1 # number of layers including input and output
    num_inputs = size(nnet.layers[1].weights, 2)
    num_outputs = length(nnet.layers[end].bias)

    # Range of inputs to consider and create your input and output sets
    x_low = -0.1
    x_high = 0.1
    input_set = Hyperrectangle(low=x_low * ones(num_inputs), high=x_high * ones(num_inputs)) # Create an input set from -1.0 to 1.0
    output_set = PolytopeComplement(HalfSpace(ones(num_outputs), 100000000.0)) # try to give a halfspace that doesn't give too much information
end

net_function = (x) -> NeuralVerification.compute_output(nnet, [x])[1]
nodes_per_layer = [size(layer.weights, 2) for layer in nnet.layers]
push!(nodes_per_layer, length(nnet.layers[end].bias))


problem = Problem(nnet, input_set, output_set)
polytope_problem = Problem(nnet, convert(HPolytope, input_set), HPolytope([HalfSpace(ones(num_outputs), 1.0)])) # use a different output set


# Compute ground truth bounds from MIPVerify
groundtruth_time = @elapsed groundtruth_bounds = get_groundtruth_bounds(nnet, input_set)

# Compute bounds from planet's tighten_bounds
planet_time = @elapsed optimal, planet_bounds = NeuralVerification.tighten_bounds(problem, Gurobi.Optimizer; pre_activation=true, use_output_constraints=false)
planet_output_lower = low(planet_bounds[num_layers])[1]
planet_output_upper = high(planet_bounds[num_layers])[1]


# Compute bounds from util get_bounds with interval arithmetic
ia_time = @elapsed ia_bounds = NeuralVerification.get_bounds(nnet, input_set, false) # get pre-activation bounds using interval arithmetic (IA)
ia_output_lower = low(ia_bounds[num_layers])[1]
ia_output_upper = high(ia_bounds[num_layers])[1]

# Compute bounds from ConvDual get_bounds
convdual_time = @elapsed convdual_lower, convdual_upper = NeuralVerification.get_bounds(nnet, input_set.center, input_set.radius[1]) # assumes uniform bounds!
pushfirst!(convdual_lower, low(input_set)) # For consistency with the other algorithms add the bounds from the input set
pushfirst!(convdual_upper, high(input_set))
convdual_output_lower = convdual_lower[num_layers][1]
convdual_output_upper = convdual_upper[num_layers][1]

# create convdual hyperrectangle bounds
convdual_bounds = [Hyperrectangle(low=convdual_lower[i], high=convdual_upper[i]) for i = 1:num_layers]

# Compute bounds from symbolic bound tightening in Reluval
reluval_time = @elapsed reach, reluval_bounds = forward_network(ReluVal(), nnet, NeuralVerification.init_symbolic_mask(input_set); get_bounds=true)

# Compute bounds from symbolic bound tightening in Neurify
neurify_time = @elapsed reach, neurify_bounds = forward_network(Neurify(), nnet, input_set; get_bounds=true)

# Compute bounds from Ai2z and Box
ai2z_time = @elapsed reach, ai2z_bounds = forward_network(Ai2z(), nnet, input_set; get_bounds=true)
ai2_box_time = @elapsed reach, ai2_box_bounds = forward_network(Box(), nnet, input_set; get_bounds=true)


labels = ["  Ground Truth  ", "  Interval Arithmetic  ", "  Planet  ", "  Symbolic Interval Propagation  ", "  Symbolic Linear Relaxation  ", "  Ai2z  ", "  ConvDual  ", "  Ai2 Box  "]
styles =  ["green", "blue", "cyan", "violet", "pink", "black", "orange", "gray"]
markers = ["star", "diamond", "x", "star", "triangle", "+", "square", "-"]
indices_to_plot = [6, 7, 8, 1, 2, 3, 4, 5]
#indices_to_plot = 1:length(labels)
indices_for_subplot = [6, 7, 1, 3]

bounds = [groundtruth_bounds, ia_bounds, planet_bounds, reluval_bounds, neurify_bounds, ai2z_bounds, convdual_bounds, ai2_box_bounds]
num_algs = length(labels)

# Define colors for plot
colors = colormap("RdBu", length(indices_to_plot))
color_dict = Dict()
num_colors = length(indices_to_plot)
for i = 1:num_colors
    define_color(string("mycolor", i), [colors[i].r, colors[i].g, colors[i].b])
    color_dict[indices_to_plot[i]] = string("mycolor", i)
end

function all_bounds(bounds::Vector{Hyperrectangle}; lower=false, include_input=false)
    grouped_bounds = Vector{Float64}()
    for (i, hyperrectangle) in enumerate(bounds)
        if (i > 1 || include_input)
            if (lower)
                append!(grouped_bounds, low(hyperrectangle))
            else
                append!(grouped_bounds, high(hyperrectangle))
            end
        end
    end
    return grouped_bounds
end

# Get a list with length equal to the number of nodes in the network
# (or # nodes - # inputs if include_input=false) which has the
# layer number for that node
function get_layer_indices(network; include_input=false)
    layer_sizes = [size(layer.weights, 2) for layer in network.layers]
    push!(layer_sizes, length(network.layers[end].bias))
    if (!include_input)
        popfirst!(layer_sizes)
    end
    indices = 1:sum(layer_sizes)
    layer_nums = []
    for (i, layer_size) in enumerate(layer_sizes)
        append!(layer_nums, i * ones(layer_size))
    end
    # Layer nums will start at 1 on the first hidden layer if include input is false
    return layer_nums
end

# Group all of the bounds into a single vector for each algorithm.
# This will create a list of length num_algorithms, where each element
# is a vector with all of its lower bounds / upper bounds.
groundtruth_index = 1
all_lower_bounds = all_bounds.(bounds; lower=true, include_input=false)
all_upper_bounds = all_bounds.(bounds; lower=false, include_input=false)
relative_gap = [(upper_bound - lower_bound) ./ (all_upper_bounds[groundtruth_index] - all_lower_bounds[groundtruth_index]) for (upper_bound, lower_bound) in zip(all_upper_bounds, all_lower_bounds)]

counts_best = zeros(length(relative_gap))
for i = 1:length(relative_gap[1])
    # exclude ground truth from the count
    cur_node_lowest = minimum([relative_gap[j][i]] for j = 1:length(relative_gap) if j != groundtruth_index)[1]
    println("Cur lowest: ", cur_node_lowest)
    for alg_index = 1:length(relative_gap)
        if (alg_index != groundtruth_index)
            println("Comparing: ", relative_gap[alg_index][i])
            if (relative_gap[alg_index][i] ≈ cur_node_lowest)
                counts_best[alg_index] = counts_best[alg_index] + 1
            end
        end
    end
end
println("Labels: ", labels[indices_to_plot])
println("Counts best: ", counts_best)

xs = collect(1:length(all_lower_bounds[1]))
accum_nodes = accumulate(+, nodes_per_layer[2:end]) # ignore input layer


groundtruth_gap = all_upper_bounds[groundtruth_index] - all_lower_bounds[groundtruth_index]
plots = Vector{Plots.Plot}()
full_plot = Axis(ymode="log", style="black", xlabel="Neuron", ylabel="Bound relative to ground truth", title="Bound Comparison, MNIST Network", width="12cm", height="8cm")
full_plot.legendStyle = "at={(0.5,-0.2)}, anchor = north, legend columns=4, column sep = 5"


num_layers_for_subplot = 4
num_nodes_for_subplot = accum_nodes[num_layers_for_subplot]
sub_plot = Axis(ymode="log")
max_height_subplot = -Inf
y_loc_high_below_sub = -Inf

for alg_index in indices_to_plot
    global max_height_subplot, y_loc_high_below_sub
    lower_bounds = all_lower_bounds[alg_index]
    upper_bounds = all_upper_bounds[alg_index]

    @assert !any(isnan.(relative_gap[alg_index]))

    # Plot the relative gap for this algorithm
    plot_for_alg = Plots.Linear(xs, relative_gap[alg_index], onlyMarks=true, mark=markers[alg_index], markSize=1.5, style=styles[alg_index], legendentry=labels[alg_index])
    plot_for_subplot = Plots.Linear(xs[1:num_nodes_for_subplot], relative_gap[alg_index][1:num_nodes_for_subplot], onlyMarks=true, mark=markers[alg_index], markSize=1.5, style=styles[alg_index])
    push!(full_plot, plot_for_alg)
    if (alg_index in indices_for_subplot)
        push!(sub_plot, plot_for_subplot)
        max_height_subplot = max(relative_gap[alg_index][1:num_nodes_for_subplot]..., max_height_subplot)
    end
    y_loc_high_below_sub = max(relative_gap[alg_index][1:num_nodes_for_subplot]..., y_loc_high_below_sub)
end

rectangle_bottom_y = y_loc_low*1.5
rectangle_top_y = max_height_subplot * 1.2

# Add coordinates to the main plot that we'll use for our sub plot
# push!(full_plot, Plots.Command(string(raw"\coordinate (insetSW) at (axis cs: ", -8.0, ", ", y_loc_high_below_sub * 5.0, ");")))
# push!(full_plot, Plots.Command(string(raw"\coordinate (insetSE) at (axis cs: ", 93.7, ", ", y_loc_high_below_sub * 5.0, ");")))
push!(full_plot, Plots.Command(string(raw"\coordinate (insetSW) at (axis cs: ", -10.0, ", ", y_loc_high_below_sub * 60.0, ");")))
push!(full_plot, Plots.Command(string(raw"\coordinate (insetSE) at (axis cs: ", 95.8, ", ", y_loc_high_below_sub * 60.0, ");")))
push!(full_plot, Plots.Command(string(raw"\coordinate (insetNW) at (axis cs: ", -10.0, ", ", 300000, ");")))
push!(full_plot, Plots.Command(string(raw"\coordinate (insetNE) at (axis cs: ", 95.8, ", ", 300000, ");")))
push!(full_plot, Plots.Command(string(raw"\coordinate (leftsubset) at (axis cs: 0.0, ", rectangle_bottom_y, ");")))
push!(full_plot, Plots.Command(string(raw"\coordinate (rightsubset) at (axis cs: ", num_nodes_for_subplot + 0.5,",", rectangle_bottom_y, ");")))
push!(full_plot, Plots.Command(string(raw"\coordinate (leftsubsettop) at (axis cs: 0.0, ", rectangle_top_y, ");")))
push!(full_plot, Plots.Command(string(raw"\coordinate (rightsubsettop) at (axis cs: ", num_nodes_for_subplot + 0.5,",", rectangle_top_y, ");")))
push!(full_plot, Plots.Command(raw"\draw[color=gray,dashed] (leftsubset) -- (insetSW);"))
push!(full_plot, Plots.Command(raw"\draw[color=gray,dashed] (rightsubset) -- (insetSE);"))
#push!(full_plot, Plots.Command(raw"\draw[color=black,solid] (leftsubsettop) -- (insetNW);"))
#push!(full_plot, Plots.Command(raw"\draw[color=black,solid] (rightsubsettop) -- (insetNE);"))


println("y loc high: ", y_loc_high_below_sub)
sub_plot.style = string("black, width = 6cm, height=3.7cm, at={(insetSW)}, anchor=south west, xticklabels={}, yticklabels={}")

# Draw dashed lines between each layer. Number of layers - 1 divisions
y_loc_low = 0.5 * minimum([minimum(relative_gap[i]) for i in indices_to_plot])
y_loc_high = 2.0 * maximum([maximum(relative_gap[i]) for i in indices_to_plot])
for index = 1:(length(nodes_per_layer) - 2)
    y_loc_high_temp = y_loc_high
    if (index <= 4)
        y_loc_high_temp = y_loc_high_below_sub
    end
    x_loc = accum_nodes[index] + 0.5
    cur_dash = Plots.Linear([x_loc, x_loc], [y_loc_low, y_loc_high_temp], style="dashed, black", mark="none")
    push!(full_plot, cur_dash)

    # ignore labeling the output layer, just label the hidden layers
    if (index == 1)
        string_x_loc = accum_nodes[index] / 2
    else
        string_x_loc = (accum_nodes[index] + accum_nodes[index-1])/2
    end
    push!(full_plot, Plots.Node(string(index), string_x_loc, y_loc_low, style="black"))

    # Add on the separating dashed lines to the subplot
    if (index < num_layers_for_subplot)
        push!(sub_plot, cur_dash)
        #push!(sub_plot, Plots.Node(string(index), string_x_loc, y_loc_low, style="black"))
    end
end

rectangle_top_y = max_height_subplot * 1.2
push!(full_plot, Plots.Command(string(raw"\draw (0, ", rectangle_bottom_y, ") rectangle(", num_nodes_for_subplot+0.5, ", ", rectangle_top_y, ");")))

save(string(filename_start, "_full_plot", num_layers, "layers.pdf"), [full_plot, sub_plot])
save(string(filename_start, "_full_plot", num_layers, "layers.tex"), [full_plot, sub_plot])

# Print summary of the times
println("Ground truth time: ", groundtruth_time)
println("Interval Arithmetic time: ", ia_time)
println("Ai2 Box time: ", ai2_box_time)
println("Planet time: ", planet_time)
println("Reluval time: ", reluval_time)
println("Neurify time: ", neurify_time)
println("Ai2z time: ", ai2z_time)
println("ConvDual time: ", convdual_time)

# if (num_inputs == 1)
#     output_plot = Axis([
#                 Plots.Linear(net_function, (x_low, x_high), xbins=100, style="red", legendentry="net dots"),
#                 Plots.Linear([x_low, x_high], [net_function(x_low), net_function(x_high)], onlyMarks=true, markSize=4, style="red", legendentry="Network Output"),
#                 Plots.Linear([x_low, x_high], [ia_output_upper, ia_output_upper], mark="diamond", markSize=4, style="blue",  legendentry="Interval arithmetic bounds"),
#                 Plots.Linear([x_low, x_high], [ia_output_lower, ia_output_lower], mark="diamond", markSize=4, style="blue"),
#                 Plots.Linear([x_low, x_high], [convdual_output_upper, convdual_output_upper], mark="square", markSize=4, style="yellow",  legendentry="ConvDual bounds"),
#                 Plots.Linear([x_low, x_high], [convdual_output_lower, convdual_output_lower], mark="square", markSize=4, style="yellow"),
#                 Plots.Linear([x_low, x_high], [planet_output_upper, planet_output_upper], mark="triangle", markSize=4, style="cyan",  legendentry="Planet bounds"),
#                 Plots.Linear([x_low, x_high], [planet_output_lower, planet_output_lower], mark="triangle", markSize=4, style="cyan"),
#             ], xlabel="input", ylabel="output", title="Network response")
#
#     output_plot.legendStyle = "at={(1.05,1.0)}, anchor=north west"
# end
# save(string(filename_start, "_output_plot_", num_layers, "layers.tex"), output_plot)
# save(string(filename_start, "_output_plot_", num_layers, "layers.pdf"), output_plot)
