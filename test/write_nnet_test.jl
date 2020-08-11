# Test by creating a network, writing it, then reading it back in and make sure that all weights and biases match
import NeuralVerification: Layer, ReLU, Id

@testset "Write + Read nnet test" begin

    # 3 --> 3 --> 2 --> 5
    l1 = Layer([3.0 2.0 1.0; 5.0 6.0 7.0; 8.0 9.0 10.0], [0.8; 1.0; 1.2], ReLU())
    l2 = Layer([1.5 2.5 3.5; 4.5 6.5 7.5], [-1.0; -3.0], ReLU())
    # The .nnet file doesn't store an activation at each layer,
    l3 = Layer([10.0 -1.0; -2.0 3.0; 4.0 5.0; 10.0 7.0; -3.5 -4.5], [0.0; -1.0; 0.0; 10.0; -10.0], Id())

    # Write out the network
    network_file = string(tempname(), ".nnet")
    println("File: ", network_file)
    nnet = Network([l1, l2, l3])
    write_nnet(network_file, nnet; header_text="test_header")

    # Read back in the network
    new_nnet = read_nnet(network_file)

    # Test that all weights, biases, and activations are the same
    @test new_nnet.layers[1].weights == l1.weights;
    @test new_nnet.layers[1].bias == l1.bias;
    @test new_nnet.layers[1].activation == l1.activation;

    @test new_nnet.layers[2].weights == l2.weights;
    @test new_nnet.layers[2].bias == l2.bias;
    @test new_nnet.layers[2].activation == l2.activation;

    @test new_nnet.layers[3].weights == l3.weights;
    @test new_nnet.layers[3].bias == l3.bias;
    @test new_nnet.layers[3].activation == l3.activation;
end
