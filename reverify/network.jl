type Layer
    bias::Matrix{Float64}
    weights::Matrix{Float64}
end

type Network
    layers::Vector{Layer} # layers includes input & output layer
    layer_sizes::Vector{Int64} # store size of each layer in network
end
