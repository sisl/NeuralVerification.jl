# Input constraint is Hyperrectangle (uniform radius)
# Output constriant is HPolytope with one constraint
struct FastLin
    maxIter::Int64
    ϵ0::Float64
    accuracy::Float64
end

function solve(solver::FastLin, problem::Problem)
    ϵ = fill(solver.ϵ0, solver.maxIter)
    ϵ_upper = 2 * solver.ϵ0
    ϵ_lower = 0.0
    c, d = tosimplehrep(problem.output)
    i = 1
    while ϵ[i] > solver.accuracy && i < solver.maxIter
        input_bounds = Hyperrectangle(problem.input.center, ϵ*ones(dim(problem.input)))
        # Here it uses reachability to comput the output bounds
        output_bounds = forward_network(solver, problem.network, input_bounds)
        # Binary search
        if issubset(output_bounds, problem.output)
            ϵ_lower = ϵ[i]
            ϵ[i+1] = (ϵ[i] + ϵ_upper) / 2
        else
            ϵ_upper = ϵ[i]
            ϵ[i+1] = (ϵ[i] + ϵ_lower) / 2
        end
        i = i+1
    end
    if ϵ[i] > minimum(problem.input.radius)
        return Result(:True, ϵ[i])
    else
        return Result(:False, ϵ[i])
    end
end

# To be implemented
function forward_layer(solver::FastLin, L::Layer, input::Hyperrectangle)
end