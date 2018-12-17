# NeuralVerification.jl

This repo contains implementation of various methods to soundly verify deep neural networks. 
We verify whether a neural network satisfies certain input-output constraints. 
The verification methods are divided into five categories: 
* reachability methods:
ExactReach, Ai2, MaxSens
* primal optimization methods:
NSVerify, MIPVerify, ILP
* dual optimization methods:
Duality, ConvDual, Certify
* search and reachability methods:
ReluVal, FastLin, FastLip, DLV
* search and optimization methods:
Sherlock, BaB, Planet, Reluplex

## Example
Launch the module
```bash
using NeuralVerification
```
Set the problem
```bash
small_nnet = read_nnet("$at/../examples/networks/small_nnet.txt")
inputSet  = Hyperrectangle(low = [-1.0], high = [1.0])
outputSet = Hyperrectangle(low = [-1.0], high = [70.0])
problem = Problem(small_nnet, inputSet, outputSet)
```
Choose a solver
```bash
solver = BaB()
```
Solve!
```bash
solve(solver,    problem)
```
Result
```bash
CounterExampleResult(:UNSAT, [1.0])
```
