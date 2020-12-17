# Solvers

The three basic verification methods are "reachability", "optimization", and "search".
These are further divided into the five categories listed below.
Note that all of the optimization methods use the [JuMP.jl](https://github.com/JuliaOpt/JuMP.jl) library.


```@contents
	Pages = ["solvers.md"]
	Depth = 3
```

## Reachability-Based Methods
These methods perform exact or approximate reachability analysis to determine the output set corresponding to a given input set.
In addition, `MaxSens`, which computes lower and upper bounds for each layer, is called within other solver types in the form of [`get_bounds`](@ref).
### ExactReach
```@docs
ExactReach
```

### Ai2
```@docs
Ai2
```

### MaxSens
```@docs
MaxSens
```

### ReluVal
```@docs
ReluVal
```

### Neurify
```@docs
Neurify
```

### FastLin
```@docs
FastLin
```

### FastLip
```@docs
FastLip
```

### DLV
```@docs
DLV
```

## Optimization-Based Methods

#### Example
```@example optim
using NeuralVerification # hide
nnet = read_nnet("../../examples/networks/small_nnet.nnet")
input  = Hyperrectangle([0.0], [.5])
output = HPolytope(ones(1,1), [102.5])

problem = Problem(nnet, input, output)
# set the JuMP solver with `optimizer` keyword or use default:
solver = MIPVerify(optimizer = GLPKSolverMIP())

solve(solver,  problem)
```

### NSVerify
```@docs
NSVerify
```

### MIPVerify
```@docs
MIPVerify
```

### ILP
```@docs
ILP
```

### Duality
```@docs
Duality
```

### ConvDual
```@docs
ConvDual
```

### Certify
```@docs
Certify
```

### Sherlock
```@docs
Sherlock
```

### BaB
```@docs
BaB
```

### Planet
```@docs
Planet
```

### Reluplex
```@docs
Reluplex
```