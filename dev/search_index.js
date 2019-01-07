var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "NeuralVerification.jl",
    "title": "NeuralVerification.jl",
    "category": "page",
    "text": ""
},

{
    "location": "#NeuralVerification.jl-1",
    "page": "NeuralVerification.jl",
    "title": "NeuralVerification.jl",
    "category": "section",
    "text": "A library of algorithms for verifying deep neural networks. At the moment, all of the algorithms are written under the assumption of feedforward, fully-connected NNs, and some of them also assume ReLU activation, but both of these assumptions will be relaxed in the future.Pages = [\"index.md\", \"problem.md\", \"solvers.md\", \"functions.md\", \"existing_implementations.md\"]\nDepth = 2"
},

{
    "location": "#Installation-1",
    "page": "NeuralVerification.jl",
    "title": "Installation",
    "category": "section",
    "text": "To download this library, clone it from the julia package manager like so:(v1.0) pkg> add https://github.com/sisl/NeuralVerification.jl"
},

{
    "location": "#Usage-1",
    "page": "NeuralVerification.jl",
    "title": "Usage",
    "category": "section",
    "text": ""
},

{
    "location": "#Initializing-a-solver-1",
    "page": "NeuralVerification.jl",
    "title": "Initializing a solver",
    "category": "section",
    "text": "First, we select the solver to be used, as this informs the input and output constraints that constitute the Problem. We initialize an instance of MaxSens with a custom resolution, which determines how small the input set is partitioned for the search. For more information about MaxSens and other solvers, see Solvers.using NeuralVerification\n\nsolver = MaxSens(resolution = 0.3)"
},

{
    "location": "#Creating-up-a-Problem-1",
    "page": "NeuralVerification.jl",
    "title": "Creating up a Problem",
    "category": "section",
    "text": "A Problem consists of a Network to be tested, a set representing the domain of the input test region, and another set representing the range of the output region. In this example, we use a small neural network with only one hidden layer consisting of 2 neurons.Note that the input and output sets may be of different types for different solvers. MaxSens requires a Hyperrectangle or HPolytope as its input and output constraints, so that is what we will use here:nnet = read_nnet(\"../../examples/networks/small_nnet.nnet\")\n\nA = reshape([-1.0, 1.0], 2, 1)\ninput  = HPolytope(A, [1.0, 1.0])\noutput = HPolytope(A, [1.0, 100.0])\n\nproblem = Problem(nnet, input, output)Note that in this example, input and output are each 1-dimensional sets (line segments) corresponding to the input and output dimensions of nnet.  The input region is bounded by [-1.0, 1.0], and the output region is bounded by [-1.0, 100.0]. This can be seen by carrying out A .* [1.0, 1.0], etc. to create the constraints associated with the sets input and output. For more information about HPolytopes and Hyperrectangles, which are commonly used to represent the input/output sets in this package, please refer to the LazySets documentation."
},

{
    "location": "#Calling-the-solver-1",
    "page": "NeuralVerification.jl",
    "title": "Calling the solver",
    "category": "section",
    "text": "To solve the problem, we simply call the solve function. This syntax is independent of the solver selected.solve(solver, problem)In this case, the solver returns a ReachabilityResult and indicates that the property is satisfied. That is, no input in the input region can produce a point that is outside of the specified output region.A result status of :SAT means that the specified input-output relationship is \"satisfied\", i.e. that the property being tested for in the network holds. A result status of :UNSAT means that the input-output relationship is \"unsatisfied\", i.e. that the property being tested for in the network does not hold. A status of :Unknown is also possible. All of the algorithms considered in this library are sound, but most are not complete; it is important to note these properties when interpreting the result status. For more information about result types, see Result.The accompanying Hyperrectangle{Float64}[] is an empty vector that is meaningless in this case. If the result was instead :UNSAT, then this vector would contain the reachable set (which exceeds the allowed output set). Note that since MaxSens uses Hyperrectangles to express the interval arithmetic used in the search, the vector type is Hyperrectangle. For other solvers that return ReachbilityResult, the reachable vector could contain other subtypes of AbstractPolytope."
},

{
    "location": "problem/#",
    "page": "Problem Definitions",
    "title": "Problem Definitions",
    "category": "page",
    "text": ""
},

{
    "location": "problem/#Problem-Definitions-1",
    "page": "Problem Definitions",
    "title": "Problem Definitions",
    "category": "section",
    "text": "	Pages = [\"problem.md\"]\n	Depth = 3"
},

{
    "location": "problem/#NeuralVerification.Problem",
    "page": "Problem Definitions",
    "title": "NeuralVerification.Problem",
    "category": "type",
    "text": "Problem(network, input, output)\n\nProblem definition for neural verification.\n\nnetwork is of type Network\ninput belongs to AbstractPolytope in LazySets.jl\noutput belongs to AbstractPolytope in LazySets.jl\n\nThe verification problem consists of: for all  points in the input set, the corresponding output of the network must belong to the output set.\n\n\n\n\n\n"
},

{
    "location": "problem/#Problem-1",
    "page": "Problem Definitions",
    "title": "Problem",
    "category": "section",
    "text": "Problem"
},

{
    "location": "problem/#NeuralVerification.Network",
    "page": "Problem Definitions",
    "title": "NeuralVerification.Network",
    "category": "type",
    "text": "Network(layers::Vector{Layer})\n\nNetwork consists of a Vector of layers.\n\nSee also: Layer\n\n\n\n\n\n"
},

{
    "location": "problem/#NeuralVerification.Layer",
    "page": "Problem Definitions",
    "title": "NeuralVerification.Layer",
    "category": "type",
    "text": "Layer(weights::Matrix{Float64}, bias::Vector{Float64}, activation::ActivationFunction)\n\nLayer consists of weights and bias for linear mapping, and activation for nonlinear mapping.\n\nSee also: Network\n\n\n\n\n\n"
},

{
    "location": "problem/#NeuralVerification.Node",
    "page": "Problem Definitions",
    "title": "NeuralVerification.Node",
    "category": "type",
    "text": "Node(w::Vector{Float64}, b::Float64, activation::ActivationFunction)\n\nNode consists of w and b for linear mapping, and activation for nonlinear mapping.\n\n\n\n\n\n"
},

{
    "location": "problem/#NeuralVerification.n_nodes-Tuple{NeuralVerification.Layer}",
    "page": "Problem Definitions",
    "title": "NeuralVerification.n_nodes",
    "category": "method",
    "text": "n_node(L::Layer)\n\nReturns the number of neurons in a layer.\n\n\n\n\n\n"
},

{
    "location": "problem/#Network-1",
    "page": "Problem Definitions",
    "title": "Network",
    "category": "section",
    "text": "Modules = [NeuralVerification]\nPages = [\"utils/network.jl\"]\nOrder = [:type, :function]"
},

{
    "location": "problem/#NeuralVerification.GeneralAct",
    "page": "Problem Definitions",
    "title": "NeuralVerification.GeneralAct",
    "category": "type",
    "text": "GeneralAct <: ActivationFunction\n\nWrapper type for a general activation function.\n\nUsage\n\nact = GeneralAct(tanh)\n\nact(0) == tanh(0)           # true\nact(10.0) == tanh(10.0)     # true\n\nact = GeneralAct(x->tanh.(x))\n\njulia> act(-2:2)\n5-element Array{Float64,1}:\n -0.9640275800758169\n -0.7615941559557649\n  0.0\n  0.7615941559557649\n  0.9640275800758169\n\n\n\n\n\n"
},

{
    "location": "problem/#NeuralVerification.PiecewiseLinear",
    "page": "Problem Definitions",
    "title": "NeuralVerification.PiecewiseLinear",
    "category": "type",
    "text": "PiecewiseLinear <: ActivationFunction\n\nActivation function that uses linear interpolation between supplied knots. An extrapolation condition can be set for values outside the set of knots. Default is Linear.\n\nPiecewiseLinear(knots_x, knots_y, [extrapolation = Line()])\n\nUsage\n\nkx = [0.0, 1.2, 1.7, 3.1]\nky = [0.0, 0.5, 1.0, 1.5]\nact = PiecewiseLinear(kx, ky)\n\nact(first(kx)) == first(ky) == 0.0\nact(last(kx))  == last(ky)  == 1.5\n\nact(1.0)    # 0.4166666666666667\nact(-102)   # -42.5\n\nact = PiecewiseLinear(kx, ky, Flat())\n\nact(-102)   # 0.0\nact(Inf)    # 1.5\n\nExtrapolations\n\nFlat()\nLine()\nconstant (supply a number as the argument)\nThrow() (throws bounds error)\n\nPiecewiseLinear uses Interpolations.jl.\n\n\n\n\n\n"
},

{
    "location": "problem/#NeuralVerification.Id",
    "page": "Problem Definitions",
    "title": "NeuralVerification.Id",
    "category": "type",
    "text": "Id <: ActivationFunction\n\nIdentity operator\n\n(Id())(x) -> x\n\n\n\n\n\n"
},

{
    "location": "problem/#NeuralVerification.Max",
    "page": "Problem Definitions",
    "title": "NeuralVerification.Max",
    "category": "type",
    "text": "Max <: ActivationFunction\n\n(Max())(x) -> max(maximum(x), 0)\n\n\n\n\n\n"
},

{
    "location": "problem/#NeuralVerification.ReLU",
    "page": "Problem Definitions",
    "title": "NeuralVerification.ReLU",
    "category": "type",
    "text": "ReLU <: ActivationFunction\n\n(ReLU())(x) -> max.(x, 0)\n\n\n\n\n\n"
},

{
    "location": "problem/#Activation-Functions-1",
    "page": "Problem Definitions",
    "title": "Activation Functions",
    "category": "section",
    "text": "Note that of the activation functions listed below, only ReLU is fully supported throughout the library.Modules = [NeuralVerification]\nPages = [\"utils/activation.jl\"]\nOrder = [:type]"
},

{
    "location": "problem/#NeuralVerification.Result",
    "page": "Problem Definitions",
    "title": "NeuralVerification.Result",
    "category": "type",
    "text": "Result\n\nSupertype of all result types.\n\nSee also: BasicResult, CounterExampleResult, AdversarialResult, ReachabilityResult\n\n\n\n\n\n"
},

{
    "location": "problem/#NeuralVerification.BasicResult",
    "page": "Problem Definitions",
    "title": "NeuralVerification.BasicResult",
    "category": "type",
    "text": "BasicResult(status::Symbol)\n\nResult type that captures whether the input-output constraint is satisfied. Possible status values:\n\n:SAT (io constraint is satisfied always)\n\n:UNSAT (io constraint is violated)\n\n:Unknown (could not be determined)\n\n\n\n\n\n"
},

{
    "location": "problem/#NeuralVerification.CounterExampleResult",
    "page": "Problem Definitions",
    "title": "NeuralVerification.CounterExampleResult",
    "category": "type",
    "text": "CounterExampleResult(status, counter_example)\n\nLike BasicResult, but also returns a counter_example if one is found (if status = :UNSAT). The counter_example is a point in the input set that, after the NN, lies outside the output set.\n\n\n\n\n\n"
},

{
    "location": "problem/#NeuralVerification.AdversarialResult",
    "page": "Problem Definitions",
    "title": "NeuralVerification.AdversarialResult",
    "category": "type",
    "text": "AdversarialResult(status, max_disturbance)\n\nLike BasicResult, but also returns the maximum allowable disturbance in the input (if status = :UNSAT).\n\n\n\n\n\n"
},

{
    "location": "problem/#NeuralVerification.ReachabilityResult",
    "page": "Problem Definitions",
    "title": "NeuralVerification.ReachabilityResult",
    "category": "type",
    "text": "ReachabilityResult(status, reachable)\n\nLike BasicResult, but also returns the output reachable set given the input constraint (if status = :UNSAT).\n\n\n\n\n\n"
},

{
    "location": "problem/#Results-1",
    "page": "Problem Definitions",
    "title": "Results",
    "category": "section",
    "text": "Result\nBasicResult\nCounterExampleResult\nAdversarialResult\nReachabilityResult"
},

{
    "location": "solvers/#",
    "page": "Solvers",
    "title": "Solvers",
    "category": "page",
    "text": ""
},

{
    "location": "solvers/#Solvers-1",
    "page": "Solvers",
    "title": "Solvers",
    "category": "section",
    "text": "The three basic verification methods are \"reachability\", \"optimization\", and \"search\". These are further divided into the five categories listed below. Note that all of the optimization methods use the JuMP.jl library.	Pages = [\"solvers.md\"]\n	Depth = 3"
},

{
    "location": "solvers/#Reachability-Methods-1",
    "page": "Solvers",
    "title": "Reachability Methods",
    "category": "section",
    "text": "These methods perform exact or approximate reachability analysis to determine the output set corresponding to a given input set. In addition, MaxSens, which computes lower and upper bounds for each layer, is called within other solver types in the form of get_bounds."
},

{
    "location": "solvers/#NeuralVerification.ExactReach",
    "page": "Solvers",
    "title": "NeuralVerification.ExactReach",
    "category": "type",
    "text": "ExactReach\n\nExactReach performs exact reachability analysis to compute the output reachable set for a network.\n\nProblem requirement\n\nNetwork: any depth, ReLU activation\nInput: HPolytope\nOutput: HPolytope\n\nReturn\n\nReachabilityResult\n\nMethod\n\nExact reachability analysis.\n\nProperty\n\nSound and complete.\n\nReference\n\nW. Xiang, H.-D. Tran, and T. T. Johnson, \"Reachable Set Computation and Safety Verification for Neural Networks with ReLU Activations,\" ArXiv Preprint ArXiv:1712.08163, 2017.\n\n\n\n\n\n"
},

{
    "location": "solvers/#ExactReach-1",
    "page": "Solvers",
    "title": "ExactReach",
    "category": "section",
    "text": "ExactReach"
},

{
    "location": "solvers/#NeuralVerification.Ai2",
    "page": "Solvers",
    "title": "NeuralVerification.Ai2",
    "category": "type",
    "text": "Ai2\n\nAi2 performs over-approximated reachability analysis to compute the over-approximated output reachable set for a network.\n\nProblem requirement\n\nNetwork: any depth, ReLU activation (more activations to be supported in the future)\nInput: HPolytope\nOutput: HPolytope\n\nReturn\n\nReachabilityResult\n\nMethod\n\nReachability analysis using split and join.\n\nProperty\n\nSound but not complete.\n\nReference\n\nT. Gehr, M. Mirman, D. Drashsler-Cohen, P. Tsankov, S. Chaudhuri, and M. Vechev, \"Ai2: Safety and Robustness Certification of Neural Networks with Abstract Interpretation,\" in 2018 IEEE Symposium on Security and Privacy (SP), 2018.\n\n\n\n\n\n"
},

{
    "location": "solvers/#Ai2-1",
    "page": "Solvers",
    "title": "Ai2",
    "category": "section",
    "text": "Ai2"
},

{
    "location": "solvers/#NeuralVerification.MaxSens",
    "page": "Solvers",
    "title": "NeuralVerification.MaxSens",
    "category": "type",
    "text": "MaxSens(resolution::Float64, tight::Bool)\n\nMaxSens performs over-approximated reachability analysis to compute the over-approximated output reachable set for a network.\n\nProblem requirement\n\nNetwork: any depth, any activation that is monotone\nInput: Hyperrectangle or HPolytope\nOutput: HPolytope\n\nReturn\n\nReachabilityResult\n\nMethod\n\nFirst partition the input space into small grid cells according to resolution. Then use interval arithmetic to compute the reachable set for each cell. Two versions of interval arithmetic is implemented with indicator tight. Default resolution is 1.0. Default tight = false.\n\nProperty\n\nSound but not complete.\n\nReference\n\nW. Xiang, H.-D. Tran, and T. T. Johnson, \"Output Reachable Set Estimation and Verification for Multi-Layer Neural Networks,\" ArXiv Preprint ArXiv:1708.03322, 2017.\n\n\n\n\n\n"
},

{
    "location": "solvers/#MaxSens-1",
    "page": "Solvers",
    "title": "MaxSens",
    "category": "section",
    "text": "MaxSens"
},

{
    "location": "solvers/#Primal-Optimization-Methods-1",
    "page": "Solvers",
    "title": "Primal Optimization Methods",
    "category": "section",
    "text": ""
},

{
    "location": "solvers/#Example-1",
    "page": "Solvers",
    "title": "Example",
    "category": "section",
    "text": "using NeuralVerification # hide\nnnet = read_nnet(\"../../examples/networks/small_nnet.nnet\")\ninput  = Hyperrectangle([0.0], [.5])\noutput = HPolytope(ones(1,1), [102.5])\n\nproblem = Problem(nnet, input, output)\n# set the JuMP solver with `optimizer` keyword or use default:\nsolver = MIPVerify(optimizer = GLPKSolverMIP())\n\nsolve(solver,  problem)"
},

{
    "location": "solvers/#NeuralVerification.NSVerify",
    "page": "Solvers",
    "title": "NeuralVerification.NSVerify",
    "category": "type",
    "text": "NSVerify(optimizer, m::Float64)\n\nNSVerify finds counter examples using mixed integer linear programming.\n\nProblem requirement\n\nNetwork: any depth, ReLU activation\nInput: hyperrectangle or hpolytope\nOutput: halfspace\n\nReturn\n\nCounterExampleResult\n\nMethod\n\nMILP encoding (using m). No presolve. Default optimizer is GLPKSolverMIP(). Default m is 1000.0 (should be large enough to avoid approximation error).\n\nProperty\n\nSound and complete.\n\nReference\n\nA. Lomuscio and L. Maganti, \"An Approach to Reachability Analysis for Feed-Forward Relu Neural Networks,\" ArXiv Preprint ArXiv:1706.07351, 2017.\n\n\n\n\n\n"
},

{
    "location": "solvers/#NSVerify-1",
    "page": "Solvers",
    "title": "NSVerify",
    "category": "section",
    "text": "NSVerify"
},

{
    "location": "solvers/#NeuralVerification.MIPVerify",
    "page": "Solvers",
    "title": "NeuralVerification.MIPVerify",
    "category": "type",
    "text": "MIPVerify(optimizer)\n\nMIPVerify computes maximum allowable disturbance using mixed integer linear programming.\n\nProblem requirement\n\nNetwork: any depth, ReLU activation\nInput: hyperrectangle\nOutput: halfspace\n\nReturn\n\nAdversarialResult\n\nMethod\n\nMILP encoding. Use presolve to compute a tight node-wise bounds first. Default optimizer is GLPKSolverMIP().\n\nProperty\n\nSound and complete.\n\nReference\n\nV. Tjeng, K. Xiao, and R. Tedrake, \"Evaluating Robustness of Neural Networks with Mixed Integer Programming,\" ArXiv Preprint ArXiv:1711.07356, 2017.\n\nhttps://github.com/vtjeng/MIPVerify.jl\n\n\n\n\n\n"
},

{
    "location": "solvers/#MIPVerify-1",
    "page": "Solvers",
    "title": "MIPVerify",
    "category": "section",
    "text": "MIPVerify"
},

{
    "location": "solvers/#NeuralVerification.ILP",
    "page": "Solvers",
    "title": "NeuralVerification.ILP",
    "category": "type",
    "text": "ILP(optimizer, max_iter)\n\nILP iteratively solves a linearized primal optimization to compute maximum allowable disturbance.\n\nProblem requirement\n\nNetwork: any depth, ReLU activation\nInput: hyperrectangle\nOutput: halfspace\n\nReturn\n\nAdversarialResult\n\nMethod\n\nIteratively solve a linear encoding of the problem. Default optimizer is GLPKSolverMIP(). Default max_iter is 10.\n\nProperty\n\nSound but not complete.\n\nReference\n\nO. Bastani, Y. Ioannou, L. Lampropoulos, D. Vytiniotis, A. Nori, and A. Criminisi, \"Measuring Neural Net Robustness with Constraints,\" in Advances in Neural Information Processing Systems, 2016.\n\n\n\n\n\n"
},

{
    "location": "solvers/#ILP-1",
    "page": "Solvers",
    "title": "ILP",
    "category": "section",
    "text": "ILP"
},

{
    "location": "solvers/#Dual-Optimization-Methods-1",
    "page": "Solvers",
    "title": "Dual Optimization Methods",
    "category": "section",
    "text": ""
},

{
    "location": "solvers/#NeuralVerification.Duality",
    "page": "Solvers",
    "title": "NeuralVerification.Duality",
    "category": "type",
    "text": "Duality(optimizer)\n\nDuality uses Lagrangian relaxation to compute over-approximated bounds for a network\n\nProblem requirement\n\nNetwork: any depth, any activation function that is monotone\nInput: hyperrectangle\nOutput: halfspace\n\nReturn\n\nBasicResult\n\nMethod\n\nLagrangian relaxation. Default optimizer is GLPKSolverMIP().\n\nProperty\n\nSound but not complete.\n\nReference\n\nK. Dvijotham, R. Stanforth, S. Gowal, T. Mann, and P. Kohli, \"A Dual Approach to Scalable Verification of Deep Networks,\" ArXiv Preprint ArXiv:1803.06567, 2018.\n\n\n\n\n\n"
},

{
    "location": "solvers/#Duality-1",
    "page": "Solvers",
    "title": "Duality",
    "category": "section",
    "text": "Duality"
},

{
    "location": "solvers/#NeuralVerification.ConvDual",
    "page": "Solvers",
    "title": "NeuralVerification.ConvDual",
    "category": "type",
    "text": "ConvDual\n\nConvDual uses convex relaxation to compute over-approximated bounds for a network\n\nProblem requirement\n\nNetwork: any depth, ReLU activation\nInput: hypercube\nOutput: halfspace\n\nReturn\n\nBasicResult\n\nMethod\n\nConvex relaxation with duality.\n\nProperty\n\nSound but not complete.\n\nReference\n\nE. Wong and J. Z. Kolter, \"Provable Defenses against Adversarial Examples via the Convex Outer Adversarial Polytope,\" ArXiv Preprint ArXiv:1711.00851, 2017.\n\nhttps://github.com/locuslab/convex_adversarial\n\n\n\n\n\n"
},

{
    "location": "solvers/#ConvDual-1",
    "page": "Solvers",
    "title": "ConvDual",
    "category": "section",
    "text": "ConvDual"
},

{
    "location": "solvers/#NeuralVerification.Certify",
    "page": "Solvers",
    "title": "NeuralVerification.Certify",
    "category": "type",
    "text": "Certify(optimizer)\n\nCertify uses semidefinite programming to compute over-approximated certificates for a neural network with only one hidden layer.\n\nProblem requirement\n\nNetwork: one hidden layer, any activation that is differentiable almost everywhere whose derivative is bound by 0 and 1\nInput: hypercube\nOutput: halfspace\n\nReturn\n\nBasicResult\n\nMethod\n\nSemindefinite programming. Default optimizer is SCSSolver().\n\nProperty\n\nSound but not complete.\n\nReference\n\nA. Raghunathan, J. Steinhardt, and P. Liang, \"Certified Defenses against Adversarial Examples,\" ArXiv Preprint ArXiv:1801.09344, 2018.\n\n\n\n\n\n"
},

{
    "location": "solvers/#Certify-1",
    "page": "Solvers",
    "title": "Certify",
    "category": "section",
    "text": "Certify"
},

{
    "location": "solvers/#Search-and-Reachability-Methods-1",
    "page": "Solvers",
    "title": "Search and Reachability Methods",
    "category": "section",
    "text": ""
},

{
    "location": "solvers/#NeuralVerification.ReluVal",
    "page": "Solvers",
    "title": "NeuralVerification.ReluVal",
    "category": "type",
    "text": "ReluVal(max_iter::Int64, tree_search::Symbol)\n\nReluVal combines symbolic reachability analysis with iterative interval refinement to minimize over-approximation of the reachable set.\n\nProblem requirement\n\nNetwork: any depth, ReLU activation\nInput: hyperrectangle\nOutput: hpolytope\n\nReturn\n\nCounterExampleResult or ReachabilityResult\n\nMethod\n\nSymbolic reachability analysis and iterative interval refinement (search).\n\nmax_iter default 10.\ntree_search default :DFS - depth first search.\n\nProperty\n\nSound but not complete.\n\nReference\n\nS. Wang, K. Pei, J. Whitehouse, J. Yang, and S. Jana, \"Formal Security Analysis of Neural Networks Using Symbolic Intervals,\" CoRR, vol. abs/1804.10829, 2018. arXiv: 1804.10829.\n\nhttps://github.com/tcwangshiqi-columbia/ReluVal\n\n\n\n\n\n"
},

{
    "location": "solvers/#ReluVal-1",
    "page": "Solvers",
    "title": "ReluVal",
    "category": "section",
    "text": "ReluVal"
},

{
    "location": "solvers/#NeuralVerification.FastLin",
    "page": "Solvers",
    "title": "NeuralVerification.FastLin",
    "category": "type",
    "text": "FastLin(maxIter::Int64, ϵ0::Float64, accuracy::Float64)\n\nFastLin combines reachability analysis with binary search to find maximum allowable disturbance.\n\nProblem requirement\n\nNetwork: any depth, ReLU activation\nInput: hypercube\nOutput: halfspace\n\nReturn\n\nAdversarialResult\n\nMethod\n\nReachability analysis by network approximation and binary search.\n\nmax_iter is the maximum iteration in search, default 10;\nϵ0 is the initial search radius, default 100.0;\naccuracy is the stopping criteria, default 0.1;\n\nProperty\n\nSound but not complete.\n\nReference\n\nT.-W. Weng, H. Zhang, H. Chen, Z. Song, C.-J. Hsieh, D. Boning, I. S. Dhillon, and L. Daniel, \"Towards Fast Computation of Certified Robustness for ReLU Networks,\" ArXiv Preprint ArXiv:1804.09699, 2018.\n\n\n\n\n\n"
},

{
    "location": "solvers/#FastLin-1",
    "page": "Solvers",
    "title": "FastLin",
    "category": "section",
    "text": "FastLin"
},

{
    "location": "solvers/#NeuralVerification.FastLip",
    "page": "Solvers",
    "title": "NeuralVerification.FastLip",
    "category": "type",
    "text": "FastLip(maxIter::Int64, ϵ0::Float64, accuracy::Float64)\n\nFastLip adds Lipschitz estimation on top of FastLin.\n\nProblem requirement\n\nNetwork: any depth, ReLU activation\nInput: hypercube\nOutput: halfspace\n\nReturn\n\nAdversarialResult\n\nMethod\n\nLipschitz estimation + FastLin. All arguments are for FastLin.\n\nmax_iter is the maximum iteration in search, default 10;\nϵ0 is the initial search radius, default 100.0;\naccuracy is the stopping criteria, default 0.1;\n\nProperty\n\nSound but not complete.\n\nReference\n\nT.-W. Weng, H. Zhang, H. Chen, Z. Song, C.-J. Hsieh, D. Boning, I. S. Dhillon, and L. Daniel, \"Towards Fast Computation of Certified Robustness for ReLU Networks,\" ArXiv Preprint ArXiv:1804.09699, 2018.\n\n\n\n\n\n"
},

{
    "location": "solvers/#FastLip-1",
    "page": "Solvers",
    "title": "FastLip",
    "category": "section",
    "text": "FastLip"
},

{
    "location": "solvers/#NeuralVerification.DLV",
    "page": "Solvers",
    "title": "NeuralVerification.DLV",
    "category": "type",
    "text": "DLV(ϵ::Float64)\n\nDLV searches layer by layer for counter examples in hidden layers.\n\nProblem requirement\n\nNetwork: any depth, any activation (currently only support ReLU)\nInput: hyperrectangle\nOutput: abstractpolytope\n\nReturn\n\nCounterExampleResult\n\nMethod\n\nThe following operations are performed layer by layer. for layer i\n\ndetermine a reachable set from the reachable set in layer i-1\ndetermine a search tree in the reachable set by refining the search tree in layer i-1\nVerify\nTrue -> continue to layer i+1\nFalse -> counter example\n\nThe argument ϵ is the resolution of the initial search tree. Default 1.0.\n\nProperty\n\nSound but not complete.\n\nReference\n\nX. Huang, M. Kwiatkowska, S. Wang, and M. Wu, \"Safety Verification of Deep Neural Networks,\" in International Conference on Computer Aided Verification, 2017.\n\nhttps://github.com/VeriDeep/DLV\n\n\n\n\n\n"
},

{
    "location": "solvers/#DLV-1",
    "page": "Solvers",
    "title": "DLV",
    "category": "section",
    "text": "DLV"
},

{
    "location": "solvers/#Search-and-Optimization-Methods-1",
    "page": "Solvers",
    "title": "Search and Optimization Methods",
    "category": "section",
    "text": ""
},

{
    "location": "solvers/#NeuralVerification.Sherlock",
    "page": "Solvers",
    "title": "NeuralVerification.Sherlock",
    "category": "type",
    "text": "Sherlock(optimizer, ϵ::Float64)\n\nSherlock combines local and global search to estimate the range of the output node.\n\nProblem requirement\n\nNetwork: any depth, ReLU activation, single output\nInput: hpolytope and hyperrectangle\nOutput: hyperrectangle (1d interval)\n\nReturn\n\nCounterExampleResult or ReachabilityResult\n\nMethod\n\nLocal search: solve a linear program to find local optima on a line segment of the piece-wise linear network. Global search: solve a feasibilty problem using MILP encoding (default calling NSVerify).\n\noptimizer default GLPKSolverMIP()\nϵ is the margin for global search, default 0.1.\n\nProperty\n\nSound but not complete.\n\nReference\n\nS. Dutta, S. Jha, S. Sanakaranarayanan, and A. Tiwari, \"Output Range Analysis for Deep Neural Networks,\" ArXiv Preprint ArXiv:1709.09130, 2017.\n\nhttps://github.com/souradeep-111/sherlock\n\n\n\n\n\n"
},

{
    "location": "solvers/#Sherlock-1",
    "page": "Solvers",
    "title": "Sherlock",
    "category": "section",
    "text": "Sherlock"
},

{
    "location": "solvers/#NeuralVerification.BaB",
    "page": "Solvers",
    "title": "NeuralVerification.BaB",
    "category": "type",
    "text": "BaB(optimizer, ϵ::Float64)\n\nBaB uses branch and bound to estimate the range of the output node.\n\nProblem requirement\n\nNetwork: any depth, ReLU activation, single output\nInput: hyperrectangle\nOutput: hyperrectangle (1d interval)\n\nReturn\n\nCounterExampleResult or ReachabilityResult\n\nMethod\n\nBranch and bound. For branch, it uses iterative interval refinement. For bound, it computes concrete bounds by sampling, approximated bound by optimization.\n\noptimizer default GLPKSolverMIP()\nϵ is the desired accurancy for termination, default 0.1.\n\nProperty\n\nSound and complete.\n\nReference\n\nR. Bunel, I. Turkaslan, P. H. Torr, P. Kohli, and M. P. Kumar, \"A Unified View of Piecewise Linear Neural Network Verification,\" ArXiv Preprint ArXiv:1711.00455, 2017.\n\n\n\n\n\n"
},

{
    "location": "solvers/#BaB-1",
    "page": "Solvers",
    "title": "BaB",
    "category": "section",
    "text": "BaB"
},

{
    "location": "solvers/#NeuralVerification.Planet",
    "page": "Solvers",
    "title": "NeuralVerification.Planet",
    "category": "type",
    "text": "Planet(optimizer, eager::Bool)\n\nPlanet integrates a SAT solver (PicoSAT.jl) to find an activation pattern that maps a feasible input to an infeasible output.\n\nProblem requirement\n\nNetwork: any depth, ReLU activation\nInput: hyperrectangle or hpolytope\nOutput: halfspace\n\nReturn\n\nBasicResult\n\nMethod\n\nBinary search of activations (0/1) and pruning by optimization. Our implementation is non eager.\n\noptimizer default GLPKSolverMIP();\neager default false;\n\nProperty\n\nSound and complete.\n\nReference\n\nR. Ehlers, \"Formal Verification of Piece-Wise Linear Feed-Forward Neural Networks,\" in International Symposium on Automated Technology for Verification and Analysis, 2017.\n\nhttps://github.com/progirep/planet\n\n\n\n\n\n"
},

{
    "location": "solvers/#Planet-1",
    "page": "Solvers",
    "title": "Planet",
    "category": "section",
    "text": "Planet"
},

{
    "location": "solvers/#NeuralVerification.Reluplex",
    "page": "Solvers",
    "title": "NeuralVerification.Reluplex",
    "category": "type",
    "text": "Reluplex(optimizer, eager::Bool)\n\nReluplex uses binary tree search to find an activation pattern that maps a feasible input to an infeasible output.\n\nProblem requirement\n\nNetwork: any depth, ReLU activation\nInput: hyperrectangle\nOutput: halfspace\n\nReturn\n\nCounterExampleResult\n\nMethod\n\nBinary search of activations (0/1) and pruning by optimization.\n\nProperty\n\nSound and complete.\n\nReference\n\nG. Katz, C. Barrett, D. L. Dill, K. Julian, and M. J. Kochenderfer, \"Reluplex: An Efficient SMT Solver for Verifying Deep Neural Networks,\" in International Conference on Computer Aided Verification, 2017.\n\n\n\n\n\n"
},

{
    "location": "solvers/#Reluplex-1",
    "page": "Solvers",
    "title": "Reluplex",
    "category": "section",
    "text": "Reluplex"
},

{
    "location": "functions/#",
    "page": "Helper Functions",
    "title": "Helper Functions",
    "category": "page",
    "text": ""
},

{
    "location": "functions/#NeuralVerification.read_nnet",
    "page": "Helper Functions",
    "title": "NeuralVerification.read_nnet",
    "category": "function",
    "text": "read_nnet(fname::String)\n\nRead in neural net from file and return Network struct\n\n\n\n\n\n"
},

{
    "location": "functions/#NeuralVerification.init_layer",
    "page": "Helper Functions",
    "title": "NeuralVerification.init_layer",
    "category": "function",
    "text": "init_layer(i::Int64, layerSizes::Array{Int64}, f::IOStream)\n\nRead in layer from nnet file and return a Layer struct containing its weights/biases\n\n\n\n\n\n"
},

{
    "location": "functions/#NeuralVerification.compute_output",
    "page": "Helper Functions",
    "title": "NeuralVerification.compute_output",
    "category": "function",
    "text": "compute_output(nnet::Network, input::Vector{Float64})\n\nPropagate a given vector through a nnet and compute the output.\n\n\n\n\n\n"
},

{
    "location": "functions/#NeuralVerification.get_activation",
    "page": "Helper Functions",
    "title": "NeuralVerification.get_activation",
    "category": "function",
    "text": "get_activation(nnet::Network, x::Vector{Float64})\n\nGiven a network, find the activation pattern of all neurons at a given point x. Assume ReLU. return Vector{Vector{Bool}}. Each Vector{Bool} refers to the activation pattern of a particular layer.\n\n\n\n\n\nget_activation(nnet::Network, input::Hyperrectangle)\n\nGiven a network, find the activation pattern of all neurons for a given input set. Assume ReLU. return Vector{Vector{Int64}}.\n\n1: activated\n0: undetermined\n-1: not activated\n\n\n\n\n\nget_activation(nnet::Network, bounds::Vector{Hyperrectangle})\n\nGiven a network, find the activation pattern of all neurons given the node-wise bounds. Assume ReLU. return Vector{Vector{Int64}}.\n\n1: activated\n0: undetermined\n-1: not activated\n\n\n\n\n\n"
},

{
    "location": "functions/#NeuralVerification.get_gradient",
    "page": "Helper Functions",
    "title": "NeuralVerification.get_gradient",
    "category": "function",
    "text": "get_gradient(nnet::Network, x::Vector)\n\nGiven a network, find the gradient at the input x\n\n\n\n\n\nget_gradient(nnet::Network, input::AbstractPolytope)\n\nGet lower and upper bounds on network gradient for a given input set. Return:\n\nLG::Vector{Matrix}: lower bounds\nUG::Vector{Matrix}: upper bounds\n\n\n\n\n\nget_gradient(nnet::Network, LΛ::Vector{Matrix}, UΛ::Vector{Matrix})\n\nGet lower and upper bounds on network gradient for given gradient bounds on activations Inputs:\n\nLΛ::Vector{Matrix}: lower bounds on activation gradients\nUΛ::Vector{Matrix}: upper bounds on activation gradients\n\nReturn:\n\nLG::Vector{Matrix}: lower bounds\nUG::Vector{Matrix}: upper bounds\n\n\n\n\n\nget_gradient(nnet::Network, LΛ::Vector{Vector{N}}, UΛ::Vector{Vector{N}}) where N\n\nGet lower and upper bounds on network gradient for given gradient bounds on activations Inputs:\n\nLΛ::Vector{Vector{N}}: lower bounds on activation gradients\nUΛ::Vector{Vector{N}}: upper bounds on activation gradients\n\nReturn:\n\nLG::Vector{Matrix}: lower bounds\nUG::Vector{Matrix}: upper bounds\n\n\n\n\n\n"
},

{
    "location": "functions/#NeuralVerification.act_gradient",
    "page": "Helper Functions",
    "title": "NeuralVerification.act_gradient",
    "category": "function",
    "text": "act_gradient(act::ReLU, z_hat::Vector{N}) where N\n\nComputing the gradient of an activation function at point z_hat. Currently only support ReLU.\n\n\n\n\n\n"
},

{
    "location": "functions/#NeuralVerification.act_gradient_bounds",
    "page": "Helper Functions",
    "title": "NeuralVerification.act_gradient_bounds",
    "category": "function",
    "text": "act_gradient_bounds(nnet::Network, input::AbstractPolytope)\n\nComputing the bounds on the gradient of all activation functions given an input set. Currently only support ReLU. Return:\n\nLΛ::Vector{Matrix}: lower bounds\nUΛ::Vector{Matrix}: upper bounds\n\n\n\n\n\n"
},

{
    "location": "functions/#NeuralVerification.interval_map",
    "page": "Helper Functions",
    "title": "NeuralVerification.interval_map",
    "category": "function",
    "text": "interval_map(W::Matrix{N}, l::Vector{N}, u::Vector{N}) where N\n\nSimple linear mapping on intervals Inputs:\n\nW::Matrix{N}: the linear mapping\nl::Vector{N}: the lower bound\nu::Vector{N}: the upper bound\n\nOutputs:\n\nl_new::Vector{N}: the lower bound after mapping\nu_new::Vector{N}: the upper bound after mapping\n\n\n\n\n\ninterval_map(W::Matrix{N}, l::Vector{N}, u::Vector{N}) where N\n\nSimple linear mapping on intervals Inputs:\n\nW::Matrix{N}: the linear mapping\nl::Matrix{N}: the lower bound\nu::Matrix{N}: the upper bound\n\nOutputs:\n\nl_new::Matrix{N}: the lower bound after mapping\nu_new::Matrix{N}: the upper bound after mapping\n\n\n\n\n\n"
},

{
    "location": "functions/#NeuralVerification.get_bounds",
    "page": "Helper Functions",
    "title": "NeuralVerification.get_bounds",
    "category": "function",
    "text": "get_bounds(nnet::Network, input::Hyperrectangle)\n\nThis function calls maxSens to compute node-wise bounds given a input set.\n\nReturn:\n\nbounds::Vector{Hyperrectangle}: bounds for all nodes AFTER activation. bounds[1] is the input set.\n\n\n\n\n\nget_bounds(problem::Problem)\n\nThis function calls maxSens to compute node-wise bounds given a problem.\n\nReturn:\n\nbounds::Vector{Hyperrectangle}: bounds for all nodes AFTER activation. bounds[1] is the input set.\n\n\n\n\n\nget_bounds(nnet::Network, input::Hyperrectangle, act::Bool)\n\nCompute bounds before or after activation by interval arithmetic. To be implemented.\n\nInputs:\n\nnnet::Network: network\ninput::Hyperrectangle: input set\nact::Bool: true for after activation bound; false for before activation bound\n\nReturn:\n\nbounds::Vector{Hyperrectangle}: bounds for all nodes AFTER activation. bounds[1] is the input set.\n\n\n\n\n\n"
},

{
    "location": "functions/#NeuralVerification.linear_transformation",
    "page": "Helper Functions",
    "title": "NeuralVerification.linear_transformation",
    "category": "function",
    "text": "linear_transformation(layer::Layer, input::Hyperrectangle)\n\nTransformation of a set considering linear mappings in a layer.\n\nInputs:\n\nlayer::Layer: a layer in a network\ninput::Hyperrectangle: input set\n\nReturn:\n\noutput::Hyperrectangle: set after transformation.\n\n\n\n\n\nlinear_transformation(layer::Layer, input::HPolytope)\n\nTransformation of a set considering linear mappings in a layer.\n\nInputs:\n\nlayer::Layer: a layer in a network\ninput::HPolytope: input set\n\nReturn:\n\noutput::HPolytope: set after transformation.\n\n\n\n\n\nlinear_transformation(W::Matrix, input::HPolytope)\n\nTransformation of a set considering a linear mapping.\n\nInputs:\n\nW::Matrix: a linear mapping\ninput::HPolytope: input set\n\nReturn:\n\noutput::HPolytope: set after transformation.\n\n\n\n\n\n"
},

{
    "location": "functions/#NeuralVerification.split_interval",
    "page": "Helper Functions",
    "title": "NeuralVerification.split_interval",
    "category": "function",
    "text": "split_interval(dom::Hyperrectangle, index::Int64)\n\nSplit a set into two at the given index.\n\nInputs:\n\ndom::Hyperrectangle: the set to be split\nindex: the index to split at\n\nReturn:\n\n(left, right)::Tuple{Hyperrectangle, Hyperrectangle}: two sets after split\n\n\n\n\n\n"
},

{
    "location": "functions/#Helper-Functions-1",
    "page": "Helper Functions",
    "title": "Helper Functions",
    "category": "section",
    "text": "    NeuralVerification.read_nnet\nNeuralVerification.init_layer\nNeuralVerification.compute_output\nNeuralVerification.get_activation\nNeuralVerification.get_gradient\nNeuralVerification.act_gradient\nNeuralVerification.act_gradient_bounds\nNeuralVerification.interval_map\nNeuralVerification.get_bounds\nNeuralVerification.linear_transformation\nNeuralVerification.split_interval"
},

{
    "location": "existing_implementations/#",
    "page": "Existing Implementations",
    "title": "Existing Implementations",
    "category": "page",
    "text": ""
},

{
    "location": "existing_implementations/#Existing-Implementations-1",
    "page": "Existing Implementations",
    "title": "Existing Implementations",
    "category": "section",
    "text": "MIPVerifyConvDualReluValSherlockPlanetPLNNDLV"
},

]}
