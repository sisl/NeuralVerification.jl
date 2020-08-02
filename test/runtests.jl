include("identity_network.jl")
include("relu_network.jl")
include("inactive_relus.jl")
if Base.find_package("Flux") != nothing
    include("flux.jl")
end
include("complements.jl")
include("splitting.jl")
include("mnist_1000.jl")
include("fully_split.jl")
include("write_nnet_test.jl")
