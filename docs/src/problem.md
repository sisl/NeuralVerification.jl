# Problem Definitions

```@contents
	Pages = ["problem.md"]
	Depth = 4
```

## Problem

```@docs
Problem
```

### Input/Output Sets

Different solvers require problems formulated with particular input and output sets.
The table below lists all of the solvers with their required input/output sets.

 - HR = `Hyperrectangle`
 - HS = `HalfSpace`
 - HP = `HPolytope`
 - PC = `PolytopeComplement`

[](TODO: review these. Not sure they're all correct.)

|        Solver        |  Input set  |    Output set    |
|----------------------|:-----------:|:----------------:|
| [`ExactReach`](@ref) | HP          | HP (bounded[^1]) |
| [`AI2`](@ref)        | HP          | HP (bounded[^1]) |
| [`MaxSens`](@ref)    | HR          | HP (bounded[^1]) |
| [`NSVerify`](@ref)   | HR          | PC[^2]           |
| [`MIPVerify`](@ref)  | HR          | PC[^2]           |
| [`ILP`](@ref)        | HR          | PC[^2]           |
| [`Duality`](@ref)    | HR(uniform) | HS               |
| [`ConvDual`](@ref)   | HR(uniform) | HS               |
| [`Certify`](@ref)    | HR          | HS               |
| [`FastLin`](@ref)    | HR          | HS               |
| [`FastLip`](@ref)    | HR          | HS               |
| [`ReluVal`](@ref)    | HR          | HR               |
| [`Neurify`](@ref)    | HP          | HP               |
| [`DLV`](@ref)        | HR          | HR[^3]           |
| [`Sherlock`](@ref)   | HR          | HR[^3]           |
| [`BaB`](@ref)        | HR          | HR[^3]           |
| [`Planet`](@ref)     | HR          | PC[^2]           |
| [`Reluplex`](@ref)   | HP          | PC[^2]           |

 [^1]: This restriction is not due to a theoretic limitation, but rather to our implementation, and will eventually be relaxed.

 [^2]: See [`PolytopeComplement`](@ref) for a justification of this output set restriction.

 [^3]: The set can only have one output node. I.e. it must be a set of dimension 1.

Note that solvers which require `Hyperrectangle`s also work on `HPolytope`s by overapproximating the input set. This is likewise true for solvers that require `HPolytope`s converting a `Hyperrectangle` input to H-representation. Any set which can be made into the required set is converted, wrapped, or approximated appropriately.

### PolytopeComplements

Some optimization-based solvers work on the principle of a complementary output constraint.
Essentially, they test whether a point *is not* in a set, by checking whether it is in the complement of the set (or vice versa).
To represent the kinds of sets we are interested in for these solvers, we define the `PolytopeComplement`, which represents the complement of a convex set.
Note that in general, the complement of a convex set is neither convex nor closed. [](would be good to include an image like the one in the paper that goes with AdversarialResult)

Although it is possible to represent the complement of a `HalfSpace` as another `HalfSpace`, we require that it be specified as a `PolytopeComplement` to disambiguate the boundary.

```@docs
PolytopeComplement
```


## Network

```@autodocs
Modules = [NeuralVerification]
Pages = ["utils/network.jl"]
Order = [:type, :function]
```

## Activation Functions
*Note that of the activation functions listed below, only `ReLU` is fully supported throughout the library.*
```@autodocs
Modules = [NeuralVerification]
Pages = ["utils/activation.jl"]
Order = [:type]
```

## Results

```@docs
Result
BasicResult
CounterExampleResult
AdversarialResult
ReachabilityResult
```