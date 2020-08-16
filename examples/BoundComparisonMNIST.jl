using LazySets
using PGFPlots
using Random
using NeuralVerification
using GLPK
using NPZ

layer_sizes = [1, 10, 10, 10, 1]
#nnet = NeuralVerification.make_random_network(layer_sizes; rng=MersenneTwister(1))
nnet = NeuralVerification.read_nnet("/Users/castrong/Downloads/Bound_Sample_With_Input/mnist10x20.nnet")
center_input = transpose(npzread("/Users/castrong/Downloads/Bound_Sample_With_Input/MNISTlabel_0_index_0_.npy")) # Transpose for AutoTaxi - transpose(npzread(example_input))
input_radius = 0.016

num_layers = length(nnet.layers) + 1 # number of layers including input and output
num_inputs = size(nnet.layers[1].weights, 2)
num_outputs = length(nnet.layers[end].bias)

net_function = (x) -> NeuralVerification.compute_output(nnet, [x])[1]

input_set = Hyperrectangle(low=max.(vec(center_input)[:] - input_radius * ones(num_inputs), 0.0), high=min.(vec(center_input)[:] + input_radius * ones(num_inputs), 1.0)) # center and radius
output_set = PolytopeComplement(HalfSpace([1.0, -1.0, 0, 0, 0, 0, 0, 0, 0, 0], 0.0)) # try to give a halfspace that doesn't give too much information
problem = Problem(nnet, input_set, output_set)

# Compute bounds from util get_bounds
ia_bounds = NeuralVerification.get_bounds(nnet, input_set, false) # get pre-activation bounds using interval arithmetic (IA)
ia_output_lower = low(ia_bounds[num_layers])[1]
ia_output_upper = high(ia_bounds[num_layers])[1]

# Compute bounds from ConvDual get_bounds
@time convdual_lower, convdual_upper = NeuralVerification.get_bounds(nnet, input_set.center, input_set.radius[1]) # assumes uniform bounds!
pushfirst!(convdual_lower, low(input_set)) # For consistency with the other algorithms add the bounds from the input set
pushfirst!(convdual_upper, high(input_set))
convdual_output_lower = convdual_lower[num_layers][1]
convdual_output_upper = convdual_upper[num_layers][1]

# create convdual hyperrectangle bounds
convdual_bounds = [Hyperrectangle(low=convdual_lower[i], high=convdual_upper[i]) for i = 1:num_layers]

# Compute bounds from planet's tighten_bounds
@time optimal, planet_bounds = NeuralVerification.tighten_bounds(problem, GLPK.Optimizer)
planet_output_lower = low(planet_bounds[num_layers])[1]
planet_output_upper = high(planet_bounds[num_layers])[1]


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
                append!(grouped_bounds, max.(0, low(hyperrectangle)))
            else
                append!(grouped_bounds, max.(0, high(hyperrectangle)))
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
all_nodes_lower_plot = Axis(plots_lower, title="Lower bounds")
all_nodes_upper_plot = Axis(plots_upper, title="Upper bounds")

all_nodes_lower_plot.legendStyle = "at={(1.05,1.0)}, anchor=north west"
all_nodes_upper_plot.legendStyle = "at={(1.05,1.0)}, anchor=north west"

save(string("all_lower_", num_layers, "layers.svg"), all_nodes_lower_plot)
save(string("all_upper_", num_layers, "layers.svg"), all_nodes_upper_plot)
