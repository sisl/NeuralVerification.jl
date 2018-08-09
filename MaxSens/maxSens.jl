# Supporting functions
function getBounds(x::Vector{Float64},delta::Float64,w::Vector{Float64},theta::Float64)
    betaMax = w'*x+theta
    betaMin = w'*x+theta
    for i in 1:length(x)
        betaMax += abs(w[i])*delta
        betaMin -= abs(w[i])*delta
    end
    return [betaMax,betaMin]
end

function getGamma(actFunc::Function,input::Float64,beta::Vector{Float64})
    return max(abs(actFunc(beta[1])-actFunc(input)), abs(actFunc(beta[2])-actFunc(input)))
end

function ReLU(x::Float64)
    return max(x,0)
end

function ReLU(x::Vector{Float64})
    y = x
    for i in 1:length(x)
        y[i] = max(x[i],0)
    end
    return y
end

function gridPartition(inputSet::Constraints, size::Float64)
    n_dim = length(inputSet.upper)
    grid_list = Vector{Int64}(n_dim)
    n_grid = 1
    for i in 1:n_dim
        grid_list[i] = n_grid
        n_grid *= ceil((inputSet.upper[i]-inputSet.lower[i])/size)
    end
    n_grid = trunc(Int, n_grid)
    x = Vector{Vector{Float64}}(n_grid)
    for k in 1:n_grid
        number = k
        x[k] = Vector{Float64}(n_dim)
        for i in n_dim:-1:1
            id = div(number-1, grid_list[i])
            number = mod(number-1, grid_list[i])+1
            x[k][i] = inputSet.lower[i] + size/2 + size * id;
        end
    end
    return x
end