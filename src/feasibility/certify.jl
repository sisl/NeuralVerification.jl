# Certify based on semidefinite relaxation

# M is used in the semidefinite program
function get_M(v::Vector{Float64}, W::Matrix{Float64})
	m = W' * diagm(v)
	o = ones(size(W, 2))
	M = [zeros(1, 1+size(m, 2)) o'*m;
		 zeros(size(m,1), 1+size(m, 2)) m;
		 m'*o m' zeros(size(m, 2), size(m, 2))]
	return M
end