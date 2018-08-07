include("network.jl")
include("util.jl")

using JuMP
using MathProgBase
using GLPKMathProgInterface

function read_nnet(fname)
    f = open(fname)
	
	line = readline(f)
        while contains(line, "//") #skip comments
                line = readline(f)
        end

	record = split(line, ",")
        nLayers = parse(Int64, record[1])
        record = split(readline(f), ",")
        layerSizes = Vector{Int64}(nLayers + 1)
        for i = 1: nLayers + 1
                layerSizes[i] = parse(Int64, record[i])
        end

	# read past additonal information to get weight/bias values
    for i = 1: 5
        line = readline(f)
    end	

	# initialize layers and model of network

    layers = Vector{Layer}(nLayers + 1)

    # initialize input layer (which has no weights/bias)
	for i = 1:nLayers+1
                 curr_layer = init_layer(model, i, layerSizes, f)
		 layers[i] = curr_layer
	end

	return Network(layers, model)
end

function add_constraint(nnet::Network, neuron::Variable, expr::Expr)
	@constraint(nnet.model, neuron >= expr)
end
