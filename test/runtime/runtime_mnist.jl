using NeuralVerification, LazySets, Test, LinearAlgebra
import NeuralVerification: ReLU, Id
using PyPlot
using CSV
using Cbc



function read_input(fname::String)
    f = open(fname)
    allvals = chop(readline(f))
    arrvals = split(allvals,",")
    input_img = map(x->parse(Float64,x),arrvals)/255.0
    return input_img
end


vnn_mnist_networks = "$(@__DIR__)/../networks/vnn-comp-mnist/networks";
vnn_mnist_inputs = "$(@__DIR__)/../networks/vnn-comp-mnist/inputs";
input_labels = [7,2,1,0,4,1,4,9,5,9,0,6,9,0,1,5,9,7,3,4,9,6,6,5,4];

# property definition

# 256x2
network2  = read_nnet(vnn_mnist_networks*"/mnist-net_256x2.nnet", last_layer_activation = Id());
# 256x4
network4  = read_nnet(vnn_mnist_networks*"/mnist-net_256x4.nnet", last_layer_activation = Id());
# 256x6
network6  = read_nnet(vnn_mnist_networks*"/mnist-net_256x6.nnet", last_layer_activation = Id());

nws = Dict(2 => network2, 
           4 => network4, 
           6 => network6);


eps1 = .02
eps2 = .05