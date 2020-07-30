using NeuralVerification, LazySets, GLPKMathProgInterface, GLPK
using Test

import NeuralVerification: ReLU, Id

macro no_error(ex)
    quote
        try $(esc(ex))
            true
        catch e
            @error(e)
            false
        end
    end
end


include("identity_network.jl")
include("relu_network.jl")
include("inactive_relus.jl")
if Base.find_package("Flux") != nothing
    include("flux.jl")
end
include("complements.jl")
include("write_nnet_test.jl")
include("write_and_read_set_test.jl")
include("write_and_read_problem_test.jl")
include("run_correctness_tests.jl")
