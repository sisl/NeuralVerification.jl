# sanity checks

using NeuralVerification, LazySets, GLPKMathProgInterface
using Test

function printtest(solver, problem_holds, problem_violated)
    println(typeof(solver))
    res_holds = solve(solver, problem_holds)
    res_violated = solve(solver, problem_violated)

    function col(s, rev = false)
        cols = [:green, :red]
        rev && reverse!(cols)
        s == :holds   && return cols[1]
        s == :violated && return cols[2]
        return (:yellow) #else
    end

    print("\tholds test.   Result: "); printstyled("$(res_holds.status)\n", color = col(res_holds.status))
    print("\tviolated test. Result: "); printstyled("$(res_violated.status)\n", color = col(res_violated.status, true))
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

problem_h_hyper_hyper           = Problem(small_nnet, input_hyper, out_hyper_30_80)                      # 40.0 < y < 60.0
problem_v_hyper_hyper         = Problem(small_nnet, input_hyper, out_hyper_50)                         # -1.0 < y < 50.0
problem_h_hpoly_hpoly_bounded   = Problem(small_nnet, input_hpoly, HPolytope(out_hyper_30_80))
problem_v_hpoly_hpoly_bounded = Problem(small_nnet, input_hpoly, HPolytope(out_hyper_50))
# halfspace constraints:
problem_h_hyper_hs              = Problem(small_nnet, input_hyper, HPolytope([HalfSpace([1.], 100.)]))     # y < 100.0
problem_v_hyper_hs            = Problem(small_nnet, input_hyper, HPolytope([HalfSpace([1.], 10.)]))      # y < 10.0


# GROUP 1           # Input: HPolytope, Output: HPolytope
group1 = [MaxSens(resolution = 0.6), ExactReach(), Ai2()]
for solver in group1
    printtest(solver, problem_h_hpoly_hpoly_bounded, problem_v_hpoly_hpoly_bounded)
    holds   = solve(solver, problem_h_hpoly_hpoly_bounded)
    violated = solve(solver, problem_v_hpoly_hpoly_bounded)

    @test holds.status ∈ (:holds, :Unknown)
    @test violated.status ∈ (:violated, :Unknown)
end


# GROUP 2, 3, 4     # Input: HPolytope, Output: HPolytope
glpk = GLPKSolverMIP()
group2 = [S(optimizer = glpk) for S in (NSVerify, MIPVerify, ILP)]
group3 = [ConvDual(), Duality(optimizer = glpk)]
group4 = [FastLin(), FastLip()]
group6 = [ReluVal(max_iter = 10), Reluplex(), Planet(GLPKSolverMIP())]

for solver in [group2; group3; group4; group6]
    printtest(solver, problem_h_hyper_hs, problem_v_hyper_hs)
    holds   = solve(solver, problem_h_hyper_hs)
    violated = solve(solver, problem_v_hyper_hs)

    @test holds.status ∈ (:holds, :Unknown)
    @test violated.status ∈ (:violated, :Unknown)
end


# GROUP 5, 6        # Input: Hyperrectangle, Output: Hyperrectangle
group5 = [ReluVal(max_iter = 10), DLV(), Sherlock(glpk, 1.0), BaB(optimizer = glpk)]


for solver in [group5;]
    printtest(solver, problem_h_hyper_hyper, problem_v_hyper_hyper)
    holds   = solve(solver, problem_h_hyper_hyper)
    violated = solve(solver, problem_v_hyper_hyper)

    @test holds.status ∈ (:holds, :Unknown)
    @test violated.status ∈ (:violated, :Unknown)
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