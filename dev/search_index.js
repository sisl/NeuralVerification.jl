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
    "location": "#Creating-a-Problem-1",
    "page": "NeuralVerification.jl",
    "title": "Creating a Problem",
    "category": "section",
    "text": "A Problem consists of a Network to be tested, a set representing the domain of the input test region, and another set representing the range of the output region. In this example, we use a small neural network with only one hidden layer consisting of 2 neurons.Note that the input and output sets may be of different types for different solvers. MaxSens requires a Hyperrectangle or HPolytope as its input and output constraints, so that is what we will use here:nnet = read_nnet(\"../../examples/networks/small_nnet.nnet\")\n\nA = reshape([-1.0, 1.0], 2, 1)\ninput  = HPolytope(A, [1.0, 1.0])\noutput = HPolytope(A, [1.0, 100.0])\n\nproblem = Problem(nnet, input, output)Note that in this example, input and output are each 1-dimensional sets (line segments) corresponding to the input and output dimensions of nnet.  The input region is bounded by [-1.0, 1.0], and the output region is bounded by [-1.0, 100.0]. This can be seen by carrying out A .* [1.0, 1.0], etc. to create the constraints associated with the sets input and output. For more information about HPolytopes and Hyperrectangles, which are commonly used to represent the input/output sets in this package, please refer to the LazySets documentation."
},

{
    "location": "#Calling-the-solver-1",
    "page": "NeuralVerification.jl",
    "title": "Calling the solver",
    "category": "section",
    "text": "To solve the problem, we simply call the solve function. This syntax is independent of the solver selected.solve(solver, problem)In this case, the solver returns a ReachabilityResult and indicates that the property is satisfied. That is, no input in the input region can produce a point that is outside of the specified output region.A result status of :holds means that the specified input-output relationship is \"satisfied\", i.e. that the property being tested for in the network holds. A result status of :violated means that the input-output relationship is \"unsatisfied\", i.e. that the property being tested for in the network does not hold. A status of :Unknown is also possible. All of the algorithms considered in this library are sound, but most are not complete; it is important to note these properties when interpreting the result status. For more information about result types, see Result.The accompanying Hyperrectangle{Float64}[] is an empty vector that is meaningless in this case. If the result was instead :violated, then this vector would contain the reachable set (which exceeds the allowed output set). Note that since MaxSens uses Hyperrectangles to express the interval arithmetic used in the search, the vector type is Hyperrectangle. For other solvers that return ReachbilityResult, the reachable vector could contain other subtypes of AbstractPolytope."
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
    "text": "	Pages = [\"problem.md\"]\n	Depth = 4"
},

{
    "location": "problem/#NeuralVerification.Problem",
    "page": "Problem Definitions",
    "title": "NeuralVerification.Problem",
    "category": "type",
    "text": "Problem{P, Q}(network::Network, input::P, output::Q)\n\nProblem definition for neural verification.\n\nThe verification problem consists of: for all  points in the input set, the corresponding output of the network must belong to the output set.\n\n\n\n\n\n"
},

{
    "location": "problem/#Problem-1",
    "page": "Problem Definitions",
    "title": "Problem",
    "category": "section",
    "text": "Problem"
},

{
    "location": "problem/#Input/Output-Sets-1",
    "page": "Problem Definitions",
    "title": "Input/Output Sets",
    "category": "section",
    "text": "Different solvers require problems formulated with particular input and output sets. The table below lists all of the solvers with their required input/output sets.HR = Hyperrectangle\nHS = HalfSpace\nHP = HPolytope\nPC = PolytopeComplementSolver Input set Output set\nExactReach HP HP (bounded[1])\nAI2 HP HP (bounded[1])\nMaxSens HR HP (bounded[1])\nNSVerify HR PC[2]\nMIPVerify HR PC[2]\nILP HR PC[2]\nDuality HR(uniform) HS\nConvDual HR(uniform) HS\nCertify HR HS\nFastLin HR HS\nFastLip HR HS\nReluVal HR HR\nDLV HR HR[3]\nSherlock HR HR[3]\nBaB HR HR[3]\nPlanet HR PC[2]\nReluplex HP PC[2][1]: This restriction is not due to a theoretic limitation, but rather to our implementation, and will eventually be relaxed.[2]: See PolytopeComplement for a justification of this output set restriction.[3]: The set can only have one output node. I.e. it must be a set of dimension 1.Note that solvers which require Hyperrectangles also work on HPolytopes by overapproximating the input set. This is likewise true for solvers that require HPolytopes converting a Hyperrectangle input to H-representation. Any set which can be made into the required set is converted, wrapped, or approximated appropriately."
},

{
    "location": "problem/#NeuralVerification.PolytopeComplement",
    "page": "Problem Definitions",
    "title": "NeuralVerification.PolytopeComplement",
    "category": "type",
    "text": "PolytopeComplement\n\nThe complement to a given set. Note that in general, a PolytopeComplement is not necessarily a convex set. Also note that PolytopeComplements are open by definition.\n\nExamples\n\njulia> H = Hyperrectangle([0,0], [1,1])\nHyperrectangle{Int64}([0, 0], [1, 1])\n\njulia> PC = complement(H)\nPolytopeComplement of:\n  Hyperrectangle{Int64}([0, 0], [1, 1])\n\njulia> center(H) ∈ PC\nfalse\n\njulia> high(H).+[1,1] ∈ PC\ntrue\n\n\n\n\n\n"
},

{
    "location": "problem/#PolytopeComplements-1",
    "page": "Problem Definitions",
    "title": "PolytopeComplements",
    "category": "section",
    "text": "Some optimization-based solvers work on the principle of a complementary output constraint. Essentially, they test whether a point is not in a set, by checking whether it is in the complement of the set (or vice versa). To represent the kinds of sets we are interested in for these solvers, we define the PolytopeComplement, which represents the complement of a convex set. Note that in general, the complement of a convex set is neither convex nor closed. Although it is possible to represent the complement of a HalfSpace as another HalfSpace, we require that it be specified as a PolytopeComplement to disambiguate the boundary.PolytopeComplement"
},

{
    "location": "problem/#NeuralVerification.Network",
    "page": "Problem Definitions",
    "title": "NeuralVerification.Network",
    "category": "type",
    "text": "A Vector of layers.\n\nNetwork([layer1, layer2, layer3, ...])\n\nSee also: Layer\n\n\n\n\n\n"
},

{
    "location": "problem/#NeuralVerification.Layer",
    "page": "Problem Definitions",
    "title": "NeuralVerification.Layer",
    "category": "type",
    "text": "Layer{F, N}\n\nConsists of weights and bias for linear mapping, and activation for nonlinear mapping.\n\nFields\n\nweights::Matrix{N}\nbias::Vector{N}\nactivation::F\n\nSee also: Network\n\n\n\n\n\n"
},

{
    "location": "problem/#NeuralVerification.Node",
    "page": "Problem Definitions",
    "title": "NeuralVerification.Node",
    "category": "type",
    "text": "Node{N, F}\n\nA single node in a layer.\n\nFields\n\nw::Vector{N}\nb::N\nactivation::F\n\n\n\n\n\n"
},

{
    "location": "problem/#NeuralVerification.n_nodes-Tuple{NeuralVerification.Layer}",
    "page": "Problem Definitions",
    "title": "NeuralVerification.n_nodes",
    "category": "method",
    "text": "n_nodes(L::Layer)\n\nReturns the number of neurons in a layer.\n\n\n\n\n\n"
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
    "text": "BasicResult(status::Symbol)\n\nResult type that captures whether the input-output constraint is satisfied. Possible status values:\n\n:holds (io constraint is satisfied always)\n\n:violated (io constraint is violated)\n\n:Unknown (could not be determined)\n\n\n\n\n\n"
},

{
    "location": "problem/#NeuralVerification.CounterExampleResult",
    "page": "Problem Definitions",
    "title": "NeuralVerification.CounterExampleResult",
    "category": "type",
    "text": "CounterExampleResult(status, counter_example)\n\nLike BasicResult, but also returns a counter_example if one is found (if status = :violated). The counter_example is a point in the input set that, after the NN, lies outside the output set.\n\n\n\n\n\n"
},

{
    "location": "problem/#NeuralVerification.AdversarialResult",
    "page": "Problem Definitions",
    "title": "NeuralVerification.AdversarialResult",
    "category": "type",
    "text": "AdversarialResult(status, max_disturbance)\n\nLike BasicResult, but also returns the maximum allowable disturbance in the input (if status = :violated).\n\n\n\n\n\n"
},

{
    "location": "problem/#NeuralVerification.ReachabilityResult",
    "page": "Problem Definitions",
    "title": "NeuralVerification.ReachabilityResult",
    "category": "type",
    "text": "ReachabilityResult(status, reachable)\n\nLike BasicResult, but also returns the output reachable set given the input constraint (if status = :violated).\n\n\n\n\n\n"
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
    "text": "ExactReach\n\nExactReach performs exact reachability analysis to compute the output reachable set for a network.\n\nProblem requirement\n\nNetwork: any depth, ReLU activation\nInput: HPolytope\nOutput: AbstractPolytope\n\nReturn\n\nReachabilityResult\n\nMethod\n\nExact reachability analysis.\n\nProperty\n\nSound and complete.\n\nReference\n\nW. Xiang, H.-D. Tran, and T. T. Johnson, \"Reachable Set Computation and Safety Verification for Neural Networks with ReLU Activations,\" ArXiv Preprint ArXiv:1712.08163, 2017.\n\n\n\n\n\n"
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
    "text": "Ai2\n\nAi2 performs over-approximated reachability analysis to compute the over-approximated output reachable set for a network.\n\nProblem requirement\n\nNetwork: any depth, ReLU activation (more activations to be supported in the future)\nInput: HPolytope\nOutput: AbstractPolytope\n\nReturn\n\nReachabilityResult\n\nMethod\n\nReachability analysis using split and join.\n\nProperty\n\nSound but not complete.\n\nReference\n\nT. Gehr, M. Mirman, D. Drashsler-Cohen, P. Tsankov, S. Chaudhuri, and M. Vechev, \"Ai2: Safety and Robustness Certification of Neural Networks with Abstract Interpretation,\" in 2018 IEEE Symposium on Security and Privacy (SP), 2018.\n\n\n\n\n\n"
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
    "text": "MaxSens(resolution::Float64, tight::Bool)\n\nMaxSens performs over-approximated reachability analysis to compute the over-approximated output reachable set for a network.\n\nProblem requirement\n\nNetwork: any depth, any activation that is monotone\nInput: Hyperrectangle or HPolytope\nOutput: AbstractPolytope\n\nReturn\n\nReachabilityResult\n\nMethod\n\nFirst partition the input space into small grid cells according to resolution. Then use interval arithmetic to compute the reachable set for each cell. Two versions of interval arithmetic is implemented with indicator tight. Default resolution is 1.0. Default tight = false.\n\nProperty\n\nSound but not complete.\n\nReference\n\nW. Xiang, H.-D. Tran, and T. T. Johnson, \"Output Reachable Set Estimation and Verification for Multi-Layer Neural Networks,\" ArXiv Preprint ArXiv:1708.03322, 2017.\n\n\n\n\n\n"
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
    "text": "NSVerify(optimizer, m::Float64)\n\nNSVerify finds counter examples using mixed integer linear programming.\n\nProblem requirement\n\nNetwork: any depth, ReLU activation\nInput: hyperrectangle or hpolytope\nOutput: PolytopeComplement\n\nReturn\n\nCounterExampleResult\n\nMethod\n\nMILP encoding (using m). No presolve. Default optimizer is GLPKSolverMIP(). Default m is 1000.0 (should be large enough to avoid approximation error).\n\nProperty\n\nSound and complete.\n\nReference\n\nA. Lomuscio and L. Maganti, \"An Approach to Reachability Analysis for Feed-Forward Relu Neural Networks,\" ArXiv Preprint ArXiv:1706.07351, 2017.\n\n\n\n\n\n"
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
    "text": "MIPVerify(optimizer)\n\nMIPVerify computes maximum allowable disturbance using mixed integer linear programming.\n\nProblem requirement\n\nNetwork: any depth, ReLU activation\nInput: hyperrectangle\nOutput: PolytopeComplement\n\nReturn\n\nAdversarialResult\n\nMethod\n\nMILP encoding. Use presolve to compute a tight node-wise bounds first. Default optimizer is GLPKSolverMIP().\n\nProperty\n\nSound and complete.\n\nReference\n\nV. Tjeng, K. Xiao, and R. Tedrake, \"Evaluating Robustness of Neural Networks with Mixed Integer Programming,\" ArXiv Preprint ArXiv:1711.07356, 2017.\n\nhttps://github.com/vtjeng/MIPVerify.jl\n\n\n\n\n\n"
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
    "text": "ILP(optimizer, max_iter)\n\nILP iteratively solves a linearized primal optimization to compute maximum allowable disturbance. It iteratively adds the linear constraint to the problem.\n\nProblem requirement\n\nNetwork: any depth, ReLU activation\nInput: hyperrectangle\nOutput: PolytopeComplement\n\nReturn\n\nAdversarialResult\n\nMethod\n\nIteratively solve a linear encoding of the problem. It only considers the linear piece of the network that has the same activation pattern as the reference input. Default optimizer is GLPKSolverMIP(). We provide both iterative method and non-iterative method to solve the LP problem. Default iterative is true.\n\nProperty\n\nSound but not complete.\n\nReference\n\nO. Bastani, Y. Ioannou, L. Lampropoulos, D. Vytiniotis, A. Nori, and A. Criminisi, \"Measuring Neural Net Robustness with Constraints,\" in Advances in Neural Information Processing Systems, 2016.\n\n\n\n\n\n"
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
    "text": "ReluVal(max_iter::Int64, tree_search::Symbol)\n\nReluVal combines symbolic reachability analysis with iterative interval refinement to minimize over-approximation of the reachable set.\n\nProblem requirement\n\nNetwork: any depth, ReLU activation\nInput: hyperrectangle\nOutput: AbstractPolytope\n\nReturn\n\nCounterExampleResult or ReachabilityResult\n\nMethod\n\nSymbolic reachability analysis and iterative interval refinement (search).\n\nmax_iter default 10.\ntree_search default :DFS - depth first search.\n\nProperty\n\nSound but not complete.\n\nReference\n\nS. Wang, K. Pei, J. Whitehouse, J. Yang, and S. Jana, \"Formal Security Analysis of Neural Networks Using Symbolic Intervals,\" CoRR, vol. abs/1804.10829, 2018. arXiv: 1804.10829.\n\nhttps://github.com/tcwangshiqi-columbia/ReluVal\n\n\n\n\n\n"
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
    "text": "FastLin(maxIter::Int64, ϵ0::Float64, accuracy::Float64)\n\nFastLin combines reachability analysis with binary search to find maximum allowable disturbance.\n\nProblem requirement\n\nNetwork: any depth, ReLU activation\nInput: hypercube\nOutput: AbstractPolytope\n\nReturn\n\nAdversarialResult\n\nMethod\n\nReachability analysis by network approximation and binary search.\n\nmax_iter is the maximum iteration in search, default 10;\nϵ0 is the initial search radius, default 100.0;\naccuracy is the stopping criteria, default 0.1;\n\nProperty\n\nSound but not complete.\n\nReference\n\nT.-W. Weng, H. Zhang, H. Chen, Z. Song, C.-J. Hsieh, D. Boning, I. S. Dhillon, and L. Daniel, \"Towards Fast Computation of Certified Robustness for ReLU Networks,\" ArXiv Preprint ArXiv:1804.09699, 2018.\n\n\n\n\n\n"
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
    "text": "DLV(ϵ::Float64)\n\nDLV searches layer by layer for counter examples in hidden layers.\n\nProblem requirement\n\nNetwork: any depth, any activation (currently only support ReLU)\nInput: Hyperrectangle\nOutput: AbstractPolytope\n\nReturn\n\nCounterExampleResult\n\nMethod\n\nThe following operations are performed layer by layer. for layer i\n\ndetermine a reachable set from the reachable set in layer i-1\ndetermine a search tree in the reachable set by refining the search tree in layer i-1\nVerify\nTrue -> continue to layer i+1\nFalse -> counter example\n\nThe argument ϵ is the resolution of the initial search tree. Default 1.0.\n\nProperty\n\nSound but not complete.\n\nReference\n\nX. Huang, M. Kwiatkowska, S. Wang, and M. Wu, \"Safety Verification of Deep Neural Networks,\" in International Conference on Computer Aided Verification, 2017.\n\nhttps://github.com/VeriDeep/DLV\n\n\n\n\n\n"
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
    "text": "Planet(optimizer, eager::Bool)\n\nPlanet integrates a SAT solver (PicoSAT.jl) to find an activation pattern that maps a feasible input to an infeasible output.\n\nProblem requirement\n\nNetwork: any depth, ReLU activation\nInput: hyperrectangle or hpolytope\nOutput: PolytopeComplement\n\nReturn\n\nBasicResult\n\nMethod\n\nBinary search of activations (0/1) and pruning by optimization. Our implementation is non eager.\n\noptimizer default GLPKSolverMIP();\neager default false;\n\nProperty\n\nSound and complete.\n\nReference\n\nR. Ehlers, \"Formal Verification of Piece-Wise Linear Feed-Forward Neural Networks,\" in International Symposium on Automated Technology for Verification and Analysis, 2017.\n\nhttps://github.com/progirep/planet\n\n\n\n\n\n"
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
    "text": "Reluplex(optimizer, eager::Bool)\n\nReluplex uses binary tree search to find an activation pattern that maps a feasible input to an infeasible output.\n\nProblem requirement\n\nNetwork: any depth, ReLU activation\nInput: hyperrectangle\nOutput: PolytopeComplement\n\nReturn\n\nCounterExampleResult\n\nMethod\n\nBinary search of activations (0/1) and pruning by optimization.\n\nProperty\n\nSound and complete.\n\nReference\n\nG. Katz, C. Barrett, D. L. Dill, K. Julian, and M. J. Kochenderfer, \"Reluplex: An Efficient SMT Solver for Verifying Deep Neural Networks,\" in International Conference on Computer Aided Verification, 2017.\n\n\n\n\n\n"
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
    "text": "MIPVerify\nConvDual\nReluVal\nSherlock\nPlanet\nPLNN\nDLV\nReluplex"
},

]}
