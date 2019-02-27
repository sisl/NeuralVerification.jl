using NeuralVerification, LazySets, GLPKMathProgInterface
using Test

import NeuralVerification: ReLU, Id

macro no_error(ex)
    quote
        try $(esc(ex))
            true
        catch
            false
        end
    end
end

# function printtest(solver, problem_sat, problem_unsat)
#     println(typeof(solver))
#     res_sat = solve(solver, problem_sat)
#     res_unsat = solve(solver, problem_unsat)

#     function col(s, rev = false)
#         cols = [:green, :red]
#         rev && reverse!(cols)
#         s == :SAT   && return cols[1]
#         s == :UNSAT && return cols[2]
#         return (:yellow) #else
#     end

#     print("\tSAT test.   Result: "); printstyled("$(res_sat.status)\n", color = col(res_sat.status))
#     print("\tUNSAT test. Result: "); printstyled("$(res_unsat.status)\n", color = col(res_unsat.status, true))
#     println("_"^70, "\n")
# end
# printtest(solvers::Vector, p1, p2) = ([printtest(s, p1, p2) for s in solvers]; nothing)

include("identity_network.jl")
include("relu_network.jl")
include("inactive_relus.jl")
if Base.find_package("Flux") != nothing
    include("flux.jl")
end
