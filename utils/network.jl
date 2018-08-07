type Layer
    weights::Matrix{Float64}
    bias::Vector{Float64}
end

type Network
    layers::Vector{Layer} # layers includes output layer
end
