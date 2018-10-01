# BAB
struct BAB
	ϵ::Float64
end

function solve(solver::BAB, problem::Problem)
	global_up = Inf
	global_lb = -Inf

	doms = init_domain(global_lb, problem.input)
	while global_up - global_lb > solver.ϵ
		dom = pick_out(doms) # pick_out implements the search strategy
		subdoms = split(dom) # split implements the branching rule
		for i in 1:s
			dom_ub = compute_bounds(problem.network, subdoms[i], true) # bounding - call convDual
			dom_lb = compute_bounds(problem.network, subdoms[i], false)
			if dom_ub < global_ub
				global_up = dom_ub
				prune_domains(doms, global_ub)
			end
			if dom_lb < global_ub
				append!(doms, (dom_lb, subdoms[i]))
			end
		end
		global_lb = minimum(lb for (lb, dom) in doms)
	end
	return global_ub
end

# To be implemented
# Search strategy
function pick_out(doms)
end

# To be implemented
# Always split the longest input dimention
function split(dom)
end

# To be implemented
# Call convDual?
function compute_bounds(nnet::Network, subdoms, upper::Bool)
end
