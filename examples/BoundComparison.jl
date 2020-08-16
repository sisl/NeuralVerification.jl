using LazySets
using PGFPlots
using Random
using NeuralVerification
using GLPK

#layer_sizes = [1, 10, 10, 10, 1]
#layer_sizes = [1, 10, 10, 10, 10, 10, 10, 10, 10, 1]
layer_sizes = [1, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 1]


nnet = NeuralVerification.make_random_network(layer_sizes; rng=MersenneTwister(1))
num_layers = length(nnet.layers) + 1 # number of layers including input and output
num_inputs = size(nnet.layers[1].weights, 2)
num_outputs = length(nnet.layers[end].bias)

net_function = (x) -> NeuralVerification.compute_output(nnet, [x])[1]

# Range of inputs to consider
x_low = -1.0
x_high = 1.0

input_set = Hyperrectangle(low=x_low * ones(num_inputs), high=x_high * ones(num_inputs)) # Create an input set from -1.0 to 1.0
output_set = PolytopeComplement(HalfSpace(ones(num_outputs), 1000000.0)) # try to give a halfspace that doesn't give too much information
problem = Problem(nnet, input_set, output_set)

# Compute bounds from util get_bounds
ia_bounds = NeuralVerification.get_bounds(nnet, input_set, false) # get pre-activation bounds using interval arithmetic (IA)
ia_output_lower = low(ia_bounds[num_layers])[1]
ia_output_upper = high(ia_bounds[num_layers])[1]

# Compute bounds from ConvDual get_bounds
convdual_lower, convdual_upper = NeuralVerification.get_bounds(nnet, input_set.center, input_set.radius[1]) # assumes uniform bounds!
pushfirst!(convdual_lower, low(input_set)) # For consistency with the other algorithms add the bounds from the input set
pushfirst!(convdual_upper, high(input_set))
convdual_output_lower = convdual_lower[num_layers][1]
convdual_output_upper = convdual_upper[num_layers][1]

# create convdual hyperrectangle bounds
convdual_bounds = [Hyperrectangle(low=convdual_lower[i], high=convdual_upper[i]) for i = 1:num_layers]

# Compute bounds from planet's tighten_bounds
optimal, planet_bounds = NeuralVerification.tighten_bounds(problem, GLPK.Optimizer)
planet_output_lower = low(planet_bounds[num_layers])[1]
planet_output_upper = high(planet_bounds[num_layers])[1]

if (num_inputs == 1)
    output_plot = Axis([
                Plots.Linear(net_function, (x_low, x_high), xbins=100, style="red", legendentry="net dots"),
                Plots.Linear([x_low, x_high], [net_function(x_low), net_function(x_high)], onlyMarks=true, markSize=4, style="red", legendentry="Network Output"),
                Plots.Linear([x_low, x_high], [ia_output_upper, ia_output_upper], mark="diamond", markSize=4, style="blue",  legendentry="Interval arithmetic bounds"),
                Plots.Linear([x_low, x_high], [ia_output_lower, ia_output_lower], mark="diamond", markSize=4, style="blue", legendentry="Interval arithmetic Bounds"),
                Plots.Linear([x_low, x_high], [convdual_output_upper, convdual_output_upper], mark="square", markSize=4, style="yellow",  legendentry="ConvDual bounds"),
                Plots.Linear([x_low, x_high], [convdual_output_lower, convdual_output_lower], mark="square", markSize=4, style="yellow", legendentry="ConvDual bounds"),
                Plots.Linear([x_low, x_high], [planet_output_upper, planet_output_upper], mark="triangle", markSize=4, style="cyan",  legendentry="Planet bounds"),
                Plots.Linear([x_low, x_high], [planet_output_lower, planet_output_lower], mark="triangle", markSize=4, style="cyan", legendentry="Planet bounds"),
            ], xlabel="input", ylabel="output", title="Network response")

    output_plot.legendStyle = "at={(1.05,1.0)}, anchor=north west"
end

labels = ["IA", "ConvDual", "Planet"]
styles = ["blue", "yellow", "cyan"]
markers = ["diamond", "square", "triangle"]
bounds = [ia_bounds, convdual_bounds, planet_bounds]
num_algs = length(labels)

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

# Group all of the bounds into a single vector for each algorithm.
# This will create a list of length num_algorithms, where each element
# is a vector with all of its lower bounds / upper bounds.
all_lower_bounds = all_bounds.(bounds; lower=true, include_input=false)
all_upper_bounds = all_bounds.(bounds; lower=false, include_input=false)

sort_indices_lower = sortperm(all_lower_bounds[2]) # Sort all based on the first algorithm's bounds
sort_indices_upper = sortperm(all_upper_bounds[2]) # Sort all based on the first algorithm's bounds

xs = collect(1:length(all_lower_bounds[1]))
plots_lower = Vector{Plots.Linear}()
plots_upper = Vector{Plots.Linear}()
for (alg_index, (lower_bounds, upper_bounds)) in enumerate(zip(all_lower_bounds, all_upper_bounds))
    plot_for_alg_lower = Plots.Linear(xs, lower_bounds[sort_indices_lower], onlyMarks=true, mark=markers[alg_index], style=styles[alg_index], legendentry=labels[alg_index])
    plot_for_alg_upper = Plots.Linear(xs, upper_bounds[sort_indices_upper], onlyMarks=true, mark=markers[alg_index], style=styles[alg_index], legendentry=labels[alg_index])
    push!(plots_lower, plot_for_alg_lower)
    push!(plots_upper, plot_for_alg_upper)
end

# Create the new plots and write all plots to file
all_nodes_lower_plot = Axis(plots_lower)
all_nodes_upper_plot = Axis(plots_upper)

all_nodes_lower_plot.legendStyle = "at={(1.05,1.0)}, anchor=north west"
all_nodes_upper_plot.legendStyle = "at={(1.05,1.0)}, anchor=north west"

save(string("output_plot_", num_layers, "layers.svg"), output_plot)
save(string("all_lower_", num_layers, "layers.svg"), all_nodes_lower_plot)
save(string("all_upper_", num_layers, "layers.svg"), all_nodes_upper_plot)












################################################################################################
#
# lower_plots = Vector{Plots.Linear}()
# upper_plots = Vector{Plots.Linear}()
# for (alg_index, alg_bounds) in enumerate(bounds)
#     i = 0
#     xs = Vector{Int}()
#     lower_ys = Vector{Float64}()
#     upper_ys = Vector{Float64}()
#     for layer_index = 1:length(alg_bounds)
#         lower_bounds = low(alg_bounds[layer_index])
#         upper_bounds = high(alg_bounds[layer_index])
#         for node_index = 1:length(lower_bounds)
#             push!(xs, i)
#             push!(lower_ys, lower_bounds[node_index])
#             push!(upper_ys, upper_bounds[node_index])
#             i = i + 1
#         end
#     end
#
#     plot_for_alg_lower = Plots.Linear(xs, lower_ys, onlyMarks=true, mark=markers[alg_index], style=styles[alg_index], legendentry=labels[alg_index])
#     plot_for_alg_upper = Plots.Linear(xs, upper_ys, onlyMarks=true, mark=markers[alg_index], style=styles[alg_index], legendentry=labels[alg_index])
#     push!(lower_plots, plot_for_alg_lower)
#     push!(upper_plots, plot_for_alg_upper)
#
# end
# all_nodes_lower_plot = Axis(plots)
# all_nodes_upper_plot = Axis(plots)
#
# save("output_plot.svg", output_plot)
# save("all_lower.svg", all_nodes_lower_plot, )
# plot("all_upper.svg", all_nodes_upper_plot)
# for bounds_for_layer in zip(bounds...)
#     global i
#     println("Size of bounds for layer: ", length(bounds_for_layer))
#     # Iterate through each node in the layer
#     for node_index in length(low(bounds_for_layer[1]))
#         println("node index: ", node_index)
#         lower_bounds_for_node = [low(bounds_for_layer[j])[node_index] for j = 1:num_algs] # find the bounds for that node for each algorithm
#         upper_bounds_for_node = [high(bounds_for_layer[j])[node_index] for j = 1:num_algs] # find the bounds for that node for each algorithm
#
#         plot_for_node = Plots.Linear(i * ones(num_algs), lower_bounds_for_node, style="")
#         push!(plots, plot_for_node)
#         i = i + 1
#     end
# end


# all_nodes_plot = Axis(plots)
# all_nodes_plot
