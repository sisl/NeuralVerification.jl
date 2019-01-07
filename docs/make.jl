using Documenter, NeuralVerification

makedocs(sitename = "NeuralVerification.jl",
         pages = ["index.md", "problem.md", "solvers.md", "functions.md", "existing_implementations.md"])


deploydocs(
    repo = "github.com/sisl/NeuralVerification.jl.git",
)