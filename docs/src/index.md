# NeuralVerification.jl

*A library of algorithms for verifying deep neural networks.
At the moment, all of the algorithms are written under the assumption of feedforward, fully-connected NNs,
and some of them also assume ReLU activation, but both of these assumptions will be relaxed in the future.*

```@contents
Pages = ["index.md", "problem.md", "solvers.md", "functions.md", "existing_implementations.md"]
Depth = 2
```

## Installation
To download this library, clone it from the julia package manager like so:
```julia
(v1.0) pkg> add https://github.com/sisl/NeuralVerification.jl
```

## Usage
### Initializing a solver
First, we select the solver to be used, as this informs the input and output constraints that constitute the `Problem`.
We initialize an instance of `MaxSens` with a custom `resolution`, which determines how small the input set is partitioned for the search.
For more information about `MaxSens` and other solvers, see [Solvers](@ref).
```@example ex1
using NeuralVerification

solver = MaxSens(resolution = 0.3)
```

### Creating a Problem
A `Problem` consists of a `Network` to be tested, a set representing the domain of the *input test region*, and another set representing the range of the *output region*.
In this example, we use a small neural network with only one hidden layer consisting of 2 neurons.

Note that the input and output sets may be of different types for different solvers.
`MaxSens` requires a `Hyperrectangle` or `HPolytope` as its input and output constraints, so that is what we will use here:
```@example ex1
nnet = read_nnet("../../examples/networks/small_nnet.nnet")

A = reshape([-1.0, 1.0], 2, 1)
input  = HPolytope(A, [1.0, 1.0])
output = HPolytope(A, [1.0, 100.0])

problem = Problem(nnet, input, output)
```
Note that in this example, `input` and `output` are each 1-dimensional sets (line segments) corresponding to the input and output dimensions of `nnet`. [](Needs more explanation.)
The input region is bounded by [-1.0, 1.0], and the output region is bounded by [-1.0, 100.0].
This can be seen by carrying out `A .* [1.0, 1.0]`, etc. to create the constraints associated with the sets `input` and `output`.
For more information about `HPolytope`s and `Hyperrectangle`s, which are commonly used to represent the input/output sets in this package, please refer to the [`LazySets` documentation](https://juliareach.github.io/LazySets.jl/latest/index.html).

### Calling the solver
To solve the problem, we simply call the `solve` function. This syntax is independent of the solver selected.
```@example ex1
solve(solver, problem)
```
In this case, the solver returns a `ReachabilityResult` and indicates that the property is satisfied.
That is, no input in the input region can produce a point that is outside of the specified output region.

A result status of `:holds` means that the specified input-output relationship is "satisfied", i.e. that the property being tested for in the network holds.
A result status of `:violated` means that the input-output relationship is "unsatisfied", i.e. that the property being tested for in the network does not hold.
A status of `:Unknown` is also possible.
All of the algorithms considered in this library are sound, but most are not complete; it is important to note these properties when interpreting the result status.
For more information about result types, see [`Result`](@ref).

The accompanying `Hyperrectangle{Float64}[]` is an empty vector that is meaningless in this case.
If the result was instead `:violated`, then this vector would contain the reachable set (which exceeds the allowed output set).
Note that since `MaxSens` uses `Hyperrectangle`s to express the interval arithmetic used in the search, the vector type is `Hyperrectangle`.
For other solvers that return `ReachbilityResult`, the `reachable` vector could contain other subtypes of `AbstractPolytope`.
