# NeuralVerification.jl

*A library of algorithms for verifying deep neural networks.
At the moment, many of the algorithms are written under the assumption of feedforward, fully-connected NNs.
Some of the algorithms also assume ReLU activation, but both of these assumptions will be relaxed in the future.*

```@contents
Pages = ["problem.md", "solvers.md", "functions.md"]
Depth = 2
```

## Setting up a problem
A `Problem` consists of a `Network` to be tested, a set representing the domain of `the input test region`, and another set representing the range of `the output region`.
In this example, we use a small neural network with only one hidden layer consisting of 2 neurons.
Note that the input and output sets may be different for different solvers.
Later, we will use the `MaxSens` solver, which requires `HPolytopes` as its input and output constraints, so this is what we will use here:
```julia
nnet = read_nnet("examples/networks/small_nnet.txt")

A = reshape([1.0, -1.0], 2, 1) # the type LazySets.HPolytope requires a matrix, rather than a vector, as input, so reshape accordingly.
input  = HPolytope(A, [1.0, 1.0])
output = HPolytope(A, [100.0, 1.0])

problem = Problem(small_nnet, input, output)
```
Note that in this example, `input` and `output` are each 1-dimensional polytopes (line segments).
The input region is within [-1.0, 1.0], and the output region is bounded by [-1.0, 100.0]
For more information about `HPolytope`s and `Hyperrectangle`s, which commonly used to represent the input/output sets, please refer to the `LazySets` documentation of those types
[](<!-- we probably want to include our own mini-doc covering some lazyset sets -->)

## Initializing a solver
Before setting up the problem, we had already decided which solver we wanted to use (since this informed our selection of the input and output sets).
So now that the problem is defined, we initialize an instance of `MaxSens` with a custom `resolution`.
The `resolution` determines how small the input set is partitioned for the search.
For more information about [`MaxSens`](@ref), see it's [documentation](@ref NeuralVerification.MaxSens). For other solvers, see [Solvers](@ref)
```julia
solver = MaxSens(resolution = 0.3)
```

## Solving the problem
To solve the problem, we simply call the `solve` function. This syntax is independent of the solver selected.
```julia
julia> solve(solver, problem)
ReachabilityResult(:SAT, Hyperrectangle{Float64}[])
```
In this case, the solver returns a [`ReachabilityResult`](@ref) and indicates that the property is satisfied.
That is, no input in the input region can produce a point that is outside of the specified output region.
The accompanying `Hyperrectangle{Float64}[]` is an empty vector that is meaningless in this case.
If the result was instead `:UNSAT`, then this vector would contain the reachable set that exceeds the output set.
Note that since `MaxSens` uses `Hyperrectangle`s to express the interval arithmetic used in the search, the vector type is `Hyperrectangle`.
In the case of other solvers that return `ReachbilityResult`, the `reachable` vector could contain other subtypes of `AbstractPolytope`.