struct FastLip
    maxIter::Int64
    ϵ0::Float64
    accuracy::Float64
end

function solve(solver::FastLip, problem::Problem)
	bounds = get_bounds()
	
end