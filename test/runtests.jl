# sanity checks

using NeuralVerification
using Test

function printtest(solver, problem_sat, problem_unsat)
    println(typeof(solver))
    res_sat = solve(solver, problem_sat)
    res_unsat = solve(solver, problem_unsat)

    function col(s, rev = false)
        cols = [:green, :red]
        rev && reverse!(cols)
        s == :SAT   && return cols[1]
        s == :UNSAT && return cols[2]
        return (:yellow) #else
    end

    print("\tSAT test.   Result: "); printstyled("$(res_sat.status)\n", color = col(res_sat.status))
    print("\tUNSAT test. Result: "); printstyled("$(res_unsat.status)\n", color = col(res_unsat.status, true))
    println("_"^70, "\n")
end

printtest(solvers::Vector, p1, p2) = ([printtest(s, p1, p2) for s in solvers]; nothing)

at = @__DIR__
small_nnet = read_nnet("$at/../examples/networks/small_nnet.nnet")

# The input set is always [-1:1]
input_hyper  = Hyperrectangle(low = [-0.9], high = [0.9])
input_hpoly  = HPolytope(input_hyper)

out_hyper_30_80 = Hyperrectangle(low = [20.0], high = [90.0])
out_hyper_50    = Hyperrectangle(low = [-1.0], high = [50.0]) # includes points in the output region ie y > 30.5

problem_sat_hyper_hyper           = Problem(small_nnet, input_hyper, out_hyper_30_80)                      # 40.0 < y < 60.0
problem_unsat_hyper_hyper         = Problem(small_nnet, input_hyper, out_hyper_50)                         # -1.0 < y < 50.0
problem_sat_hpoly_hpoly_bounded   = Problem(small_nnet, input_hpoly, HPolytope(out_hyper_30_80))
problem_unsat_hpoly_hpoly_bounded = Problem(small_nnet, input_hpoly, HPolytope(out_hyper_50))
# halfspace constraints:
problem_sat_hyper_hs              = Problem(small_nnet, input_hyper, HPolytope([HalfSpace([1.], 100.)]))     # y < 100.0
problem_unsat_hyper_hs            = Problem(small_nnet, input_hyper, HPolytope([HalfSpace([1.], 10.)]))      # y < 10.0


# GROUP 1           # Input: HPolytope, Output: HPolytope
# group1 = [MaxSens(), ExactReach(), Ai2()]
group1 = [MaxSens(resolution = 0.6), ExactReach()] # Ai2 is 100% broken right now so dropping it
for solver in group1
    printtest(solver, problem_sat_hpoly_hpoly_bounded, problem_unsat_hpoly_hpoly_bounded)
    sat   = solve(solver, problem_sat_hpoly_hpoly_bounded)
    unsat = solve(solver, problem_unsat_hpoly_hpoly_bounded)

    @test sat.status ∈ (:SAT, :Unknown)
    @test unsat.status ∈ (:UNSAT, :Unknown)
end


# GROUP 2, 3, 4     # Input: HPolytope, Output: HPolytope
glpk = GLPKSolverMIP()
group2 = [S(optimizer = glpk) for S in (NSVerify, MIPVerify, ILP)]
group3 = [ConvDual(), Duality(optimizer = glpk)]
group4 = [FastLin(), FastLip()]
group6 = [Reluplex(), Planet(GLPKSolverMIP())]

for solver in [group2; group3; group4; group6]
    printtest(solver, problem_sat_hyper_hs, problem_unsat_hyper_hs)
    sat   = solve(solver, problem_sat_hyper_hs)
    unsat = solve(solver, problem_unsat_hyper_hs)

    @test sat.status ∈ (:SAT, :Unknown)
    @test unsat.status ∈ (:UNSAT, :Unknown)
end


# GROUP 5, 6        # Input: Hyperrectangle, Output: Hyperrectangle
group5 = [ReluVal(max_iter = 10), DLV(), Sherlock(glpk, 1.0), BaB(optimizer = glpk)]


for solver in [group5;]
    printtest(solver, problem_sat_hyper_hyper, problem_unsat_hyper_hyper)
    sat   = solve(solver, problem_sat_hyper_hyper)
    unsat = solve(solver, problem_unsat_hyper_hyper)

    @test sat.status ∈ (:SAT, :Unknown)
    @test unsat.status ∈ (:UNSAT, :Unknown)
end

macro no_error(ex)
    quote
        try $(esc(ex))
            true
        catch
            false
        end
    end
end

### Certify - only works for single hidden layer
tiny_nnet = read_nnet("$at/../examples/networks/tiny_nnet.nnet")
solver_certify = Certify()
inputSet  = Hyperrectangle([2.0], [.5])
outputSet = HPolytope(ones(1,1), [2.5])
problem_certify = Problem(tiny_nnet, inputSet, outputSet)
@test @no_error solve(solver_certify, problem_certify)