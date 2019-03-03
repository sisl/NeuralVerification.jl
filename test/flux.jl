
using Flux
using NeuralVerification: ReLU, Id, Layer, network

@testset "Flux" begin

    DI = Dense(5,5)
    DR = Dense(5,5, relu)
    LI = Layer(rand(5,5), rand(5), Id())
    LR = Layer(rand(5,5), rand(5), ReLU())

    # just test that they don't error:
    @test network(Chain(DR,DR,DI)) isa Network
    @test Chain(Network([LR,LR,LI])) isa Chain

end