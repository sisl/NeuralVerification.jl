using Random

"""
    make_random_network(layer_sizes::Vector{Int}, [min_weight = -1.0], [max_weight = 1.0], [min_bias = -1.0], [max_bias = 1.0], [rng = 1.0])
    read_layer(output_dim::Int, f::IOStream, [act = ReLU()])

Generate a network with random weights and bias. The first layer is treated as the input.
The values for the weights and bias will be uniformly drawn from the range between min_weight
and max_weight and min_bias and max_bias respectively. The last layer will have an ID()
activation function and the rest will have ReLU() activation functions. Allow a random number
generator(rng) to be passed in. This allows for seeded random network generation.
"""
function make_random_network(layer_sizes::Vector{Int}, min_weight = -1.0, max_weight = 1.0, min_bias = -1.0, max_bias = 1.0, rng=MersenneTwister(0))
    # Create each layer based on the layer_size
    layers = []
    for index in 1:(length(layer_sizes)-1)
        cur_size = layer_sizes[index]
        next_size = layer_sizes[index+1]
        # Use Id activation for the last layer - otherwise use ReLU activation
        if index == (length(layer_sizes)-1)
            cur_activation = NeuralVerification.Id()
        else
            cur_activation = NeuralVerification.ReLU()
        end

        # Dimension: num_out x num_in
        cur_weights = min_weight .+ (max_weight - min_weight) * rand(rng, Float64, (next_size, cur_size))
        cur_weights = reshape(cur_weights, (next_size, cur_size)) # for edge case where 1 dimension is equal to 1 this keeps it from being a 1-d vector

        # Dimension: num_out x 1
        cur_bias = min_bias .+ (max_bias - min_bias) * rand(rng, Float64, (next_size))
        push!(layers, NeuralVerification.Layer(cur_weights, cur_bias, cur_activation))
    end

    return Network(layers)
end


"""
    write_problem(network_file::String, input_file::String, output_file::String, problem::Problem)

Write a the information from a problem to files. The input set, output set, and network will
each be written to a separate file.
"""
function write_problem(network_file::String, input_file::String, output_file::String, problem::Problem)
    write_nnet(network_file, problem.network)
    write_set(input_file, problem.input)
    write_set(output_file, problem.output)
end

"""
    get_set_type_string(set::Union{PolytopeComplement, LazySet})

Return a string corresponding to the object's type. This will be used to store
these sets to file.
"""
function get_set_type_string(set::Union{PolytopeComplement, LazySet})
    if set isa HalfSpace
        return "HalfSpace"
    elseif set isa Hyperrectangle
        return "Hyperrectangle"
    elseif set isa HPolytope
        return "HPolytope"
    elseif set isa PolytopeComplement
        return "PolytopeComplement"
    elseif set isa Zonotope
        return "Zonotope"
    else
        return ""
    end
end

"""
    write_set(filename::String, set::Union{PolytopeComplement, LazySet})

Return a string corresponding to the object's type. This will be used to store
these sets to file.
"""
function write_set(filename::String, set::Union{PolytopeComplement, LazySet})
    # Only support hyperrectangle, hyperpolytope, zonotope, polytope complement, and halfspace
    type_string = get_set_type_string(set)
    @assert type_string != ""

    # Save the type and the object itself
    output_dict = Dict()
    output_dict["type"] = type_string
    output_dict["set"] = set

    # Write to file
    open(filename, "w") do f
        JSON2.write(f, output_dict)
    end
end


"""
    write_problem(network_file::String, input_file::String, output_file::String, problem::Problem)

Read in a network, input set, and output set and return the corresponding problem.
"""
function read_problem(network_file::String, input_file::String, output_file::String, problem::Problem)
    network = read_nnet(network_file)
    input_set = read_set(input_file)
    output_set = read_set(output_file)
    return Problem(network, input_set, output_set)
end

function read_set(filename)
    json_string = read(filename, String)
    dict = JSON2.read(json_string)
    type = dict[:type]
    set = dict[:set]

    # helper function to convert from Any arrays to Float arrays
    function convert_float(elem)
        if elem isa Array
            return convert(Array{Float64}, elem)
        else
            return convert(Float64, elem)
        end
    end

    if type == "HalfSpace"
        return HalfSpace(convert_float(set[:a]), convert_float(set[:b]))
    elseif type == "Hyperrectangle"
        return Hyperrectangle(convert_float(set[:center]), convert_float(set[:radius]))
    elseif type == "HPolytope"
        constraints = set[:constraints]
        half_spaces = [HalfSpace(convert_float(constraints[i][:a]), convert_float(constraints[i][:b])) for i = 1:length(constraints)]
        return HPolytope(half_spaces)
    elseif type == "PolytopeComplement"
        constraints = set[:P][:constraints]
        half_spaces = [HalfSpace(convert_float(constraints[i][:a]), convert_float(constraints[i][:b])) for i = 1:length(constraints)]
        return PolytopeComplement(HPolytope(half_spaces))
    elseif type == "Zonotope"
        center = set[:center]
        generator_list = set[:generators]
        generator_matrix = reshape(generator_list, (length(center), :)) # Check the reshape here
        return Zonotope(convert_float(center), convert_float(generator_matrix))
    else
        @assert false
    end
end

function run_correctness_tests_on_file(filename)

end
