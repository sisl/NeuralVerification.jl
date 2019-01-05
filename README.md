# NeuralVerification.jl

This library contains implementations of various methods to soundly verify deep neural networks.
In general, we verify whether a neural network satisfies certain input-output constraints.
The verification methods are divided into five categories:
* *Reachability methods:*
ExactReach, Ai2, MaxSens
* *Primal optimization methods:*
NSVerify, MIPVerify, ILP
* *Dual optimization methods:*
Duality, ConvDual, Certify
* *Search and reachability methods:*
ReluVal, FastLin, FastLip, DLV
* *Search and optimization methods:*
Sherlock, BaB, Planet, Reluplex

## Installation
To download this library, clone it from the julia package manager like so:
```julia
(v1.0) pkg> add https://github.com/sisl/NeuralVerification.jl
```

Please note that the implementations of the algorithms are pedagogical in nature, and so may not perform optimally.
Derivation and discussion of these algorithms is presented in _link to paper_.

*Note:* At present, `Ai2`, `ExactReach`, and `Duality` do not work in higher dimensions (e.g. image classification).
This is being addressed in [#9](@ref)

## Example Usage
### Choose a solver
```julia
using NeuralVerification

solver = BaB()
```
### Set up the problem
```julia
nnet = read_nnet("examples/networks/small_nnet.txt")
input_set  = Hyperrectangle(low = [-1.0], high = [1.0])
output_set = Hyperrectangle(low = [-1.0], high = [70.0])
problem = Problem(nnet, input_set, output_set)
```
### Solve
```julia
julia> result = solve(solver, problem)
CounterExampleResult(:UNSAT, [1.0])

julia> result.status
:UNSAT
```

<!-- A result status of `:UNSAT` means that the input-output relationship is "unsatisfied", i.e. that the property being tested for in the network does not hold.
A result status of `:SAT` means that the specified input-output relationship is "satisfied" (note the completeness/soundness properties of the chosen algorithm in interpretting `:SAT` and `:UNSAT`).
A status of `:Undetermined` is also possible. -->
For a full list of `Solvers` and their properties, requirements, and `Result` types, please refer to the documentation.