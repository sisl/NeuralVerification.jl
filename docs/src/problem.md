# Problem Definitions

```@contents
	Pages = ["problem.md"]
	Depth = 3
```

## Problem

```@docs
Problem
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