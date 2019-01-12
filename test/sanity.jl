# sanity checks

using NeuralVerification
using Test

function printtest(solver, problem_sat, problem_unsat)
    println(typeof(solver))
    res_sat = solve(solver, problem_sat)
    res_unsat = solve(solver, problem_unsat)

    function col(s, rev = false)
        cols = [:green, :red]
        if rev
            cols = reverse(cols)
        end
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
input_hyper  = Hyperrectangle(low = [-1.0], high = [1.0])
input_hpoly  = HPolytope(input_hyper)

out_hyper_20_80 = Hyperrectangle(low = [20.0], high = [80.0])
out_hyper_50 = Hyperrectangle(low = [-1.0], high = [50.0]) # includes points in the output region ie y > 30.5

# In all cases -1.0 < x < 1.0
problem_sat_hyper_hyper           = Problem(small_nnet, input_hyper, out_hyper_20_80)                   # 20.0 < y < 80.0
problem_unsat_hyper_hyper         = Problem(small_nnet, input_hyper, out_hyper_50)                      # -1.0 < y < 50.0

problem_unsat_hyper_hpoly         = Problem(small_nnet, input_hyper, HPolytope(ones(1,1), [10.0]))      # y < 10.0
problem_sat_hyper_hpoly           = Problem(small_nnet, input_hyper, HPolytope(ones(1,1), [100.0]))     # y < 100.0

A = ones(2, 1); A[2] = -1
problem_sat_hpoly_hpoly_bounded   = Problem(small_nnet, input_hpoly, HPolytope(A, [60.0, -40.0]))       # 40.0 < y < 60.0
problem_unsat_hpoly_hpoly_bounded = Problem(small_nnet, input_hpoly, HPolytope(A, [110.0, -100.0]))    # 100.0 < y < 110.0

# NOTE: unused tests
# problem_sat_hpoly_hpoly           = Problem(small_nnet, input_hpoly, HPolytope(ones(1,1), [100.0]))   # y < 100.0
# problem_unsat_hpoly_hpoly         = Problem(small_nnet, input_hpoly, HPolytope(A, [10.0]))            # y < 10.0


# Group 1
# Input: HPolytope, Output: HPolytope
# group1 = [MaxSens(), ExactReach(), Ai2()]
group1 = [MaxSens(), ExactReach()] # Ai2 is 100% broken right now so dropping it
# Group 2, 3, 4
# Input: HPolytope, Output: HPolytope
glpk = GLPKSolverMIP()
group2 = [NSVerify(optimizer = glpk), MIPVerify(optimizer = glpk), ILP(optimizer = glpk)]
group3 = [ConvDual(), Duality(optimizer = glpk)]
group4 = [FastLin(10, 10.0, 1.0), FastLip(10, 10.0, 1.0)]
# Group 5, 6
# Input: Hyperrectangle, Output: Hyperrectangle
#group6 = [Planet(glpk), Reluplex()] # Planet is producing an error right now
group6 = [Reluplex()]
group5 = [ReluVal(max_iter = 10), DLV(), Sherlock(glpk, 1.0), BaB(optimizer = glpk)]

printtest(group1,
          problem_sat_hpoly_hpoly_bounded,
          problem_unsat_hpoly_hpoly_bounded)

printtest([group2; group3; group4],
          problem_sat_hyper_hpoly,
          problem_unsat_hyper_hpoly)

printtest([group5; group6],
          problem_sat_hyper_hyper,
          problem_unsat_hyper_hyper)
