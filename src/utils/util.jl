include("problem.jl")

using JuMP
using MathProgBase
using GLPKMathProgInterface

#=
Read in layer from nnet file and return a Layer struct containing its weights/biases
=#
function init_layer(model::Model, i::Int64, layerSizes::Array{Int64}, f::IOStream)
	 bias = Vector{Float64}(layerSizes[i+1])
     weights = Matrix{Float64}(layerSizes[i+1], layerSizes[i])
	 
	 # first read in weights
	 for r = 1: layerSizes[i+1]
	 	line = readline(f)
		record = split(line, ",")
		token = record[1]
		c = 1
		for c = 1: layerSizes[i]
			weights[r, c] = parse(Float64, token)
			token = record[c]
		end
	 end

	 # now read in bias
	 for r = 1: layerSizes[i+1]
	 	line = readline(f)
		record = split(line, ",")
		bias[r] = parse(Float64, record[1])
	 end

	 # activation function is set to ReLU as default
     return Layer(weights, bias, ReLU())
end

#=
Read in neural net from file and return Network struct 
=#	
function read_nnet(fname::String)
    f = open(fname)
	line = readline(f)
    while contains(line, "//") #skip comments
    	line = readline(f)
    end

    # read in layer sizes and populate array
	record = split(line, ",")
    nLayers = parse(Int64, record[1])
    record = split(readline(f), ",")
    layerSizes = Vector{Int64}(nLayers + 1)
    for i = 1: nLayers + 1
    	layerSizes[i] = parse(Int64, record[i])
    end

	# read past additonal information
    for i = 1: 5
    	line = readline(f)
    end	

	# initialize layers
	model = Model(solver=GLPKSolverMIP())
    layers = Vector{Layer}(nLayers)
	for i = 1:nLayers
        curr_layer = init_layer(model, i, layerSizes, f)
		layers[i] = curr_layer
	end

	return Network(layers)
end

#=
Compute output of an nnet for a given input vector
=#
function compute_output(nnet::Network, input::Vector{Float64})
	curr_value = input
	layers = nnet.layers
	for i = 1:length(layers) # layers does not include input layer (which has no weights/biases)
		curr_value = (layers[i].weights * curr_value) + layers[i].bias
	end
	return curr_value # would another name be better?
end