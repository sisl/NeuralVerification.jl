using NeuralVerification, LazySets, Test, LinearAlgebra
import NeuralVerification: ReLU, Id

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


function test_mnist_problem1(solver, network_no, image_no , problem,epsi)
    solvername = string(typeof(solver))
    timed_result = @timed solve(solver, problem)
    res = solvername*","*string(network_no)*","*string(image_no)*","*string(epsi)*","*string(timed_result[2])*","*string(timed_result[1].status)
    print("\n"*res)
    return solvername, timed_result[2], timed_result[1].status, res
end

function test_mnist_group1(solver)
    solvername = string(typeof(solver))
    open(solvername*".csv", "w") do f
        for network_no in [2,4,6]
            print("Network: "*string(network_no))
            for in_epsilon in [.02, .05]
                print("Epsilon: "*string(in_epsilon))
                for image_no in 1:25

                    input_center = read_input(vnn_mnist_inputs*"/image"*string(image_no));
                    inputSet = Hyperrectangle(low=input_center .- in_epsilon, high=input_center .+ in_epsilon)

                    aux = Matrix(1.0I, 10,10);
                    aux[:,input_labels[image_no]+1] .= -1.0;
                    A = aux[1:size(aux,1) .!= input_labels[image_no]+1,: ];
                    b=zeros(9);

                    outputSet = HPolytope(A, b);

                    mnistproblem = Problem(nws[network_no], inputSet, outputSet);
                    solvername, t, status, res = test_mnist_problem1(solver, network_no, image_no , mnistproblem,in_epsilon)
                    write(f, res)
                    write(f, "\n")
                end
            end
        end
    end
end

function test_mnist_group2(solver)
    solvername = string(typeof(solver))
    open(solvername*".csv", "w") do f
        for network_no in [2,4,6]
            print("\n Network: "*string(network_no)*"\n")
            for in_epsilon in [.02, .05]
                print("\n Epsilon: "*string(in_epsilon)*"\n")
                for image_no in 1:25


                    input_center = read_input(vnn_mnist_inputs*"/image"*string(image_no));
                    
                    inputSet = Hyperrectangle(input_center, ones(length(input_center))*in_epsilon)
                    #inputSet = Hyperrectangle(low=input_center .- in_epsilon, high=input_center .+ in_epsilon)

                    compare_to = (input_labels[image_no]+1) % 10
                    aux = Matrix(1.0I, 10,10);
                    aux[:,input_labels[image_no]+1] .= -1.0;
                    A = aux[1:size(aux,1) .== compare_to+1,: ];
                    b=zeros(1);

                    outputSet = HPolytope(A, b);

                    mnistproblem = Problem(nws[network_no], inputSet, outputSet);
                    solvername, t, status, res = test_mnist_problem1(solver, network_no, image_no , mnistproblem,in_epsilon)
                    write(f, res)
                    write(f, "\n")
                end
            end
        end
    end
end

# select group and solver

#test_mnist_group1(MaxSens())
test_mnist_group2(ConvDual())