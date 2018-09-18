
struct Ai2 end

function solve(solver::Ai2, problem::Problem)
    reach = forward_network(solver, problem.network, problem.input)
    isinc = check_inclusion(reach, problem.output)

    #TODO is this the correct return pattern?
    if isinc
        return Result(:True)
    else
        return Result(:False)
    end
end

forward_layer(solver::Ai2, layer::Layer, inputs::Vector{<:AbstractPolytope}) = forward_layer.(solver, layer, inputs)

function forward_layer(solver::Ai2, layer::Layer, input::AbstractPolytope)
    W, b, act = layer.weights, layer.bias, layer.activation
    outLinear = shiftcenter(linear_map(W, input), b)
    return transform(act, outLinear)
end

#=
In general, convert any polytope to a VPolytope to proceed.
TODO: create high performance Zonotype exception
=#
transform(f::ActivationFunction, poly::AbstractPolytope) = transform(f, VPolytope(vertices_list(poly)))
#=
define a version of transform for each activation
NOTE: the paper considers ReLU_i(ReLU_{i-1}(ReLU_{i-2}(...ReLU_1(V)))) where i denotes the dimension.
ReLU is (should be...) a commutive operation with respect to dimension, and so doing all ReLUs in parallel as considered in this transform should be equivalent
I.e. it shouldn't matter if you do: for vertices{for dims{ apply_ReLU }}, or for dims{for vertices{for apply_ReLU }}}
=#
function transform(f::ReLU, V::VPolytope) # NOTE: this form might be true for all ActivationFuntions, if they are commutive like ReLU
    new_vertices = [f(v) for v in vertices_list(V)]
    return VPolytope(new_vertices)
end

shiftcenter(zono::Zonotope, shift::Vector)         = Zonotope(zono.center + shift, zono.generators)
shiftcenter(poly::AbstractPolytope, shift::Vector) = shiftcenter(VPolytope(vertices_list(poly)), shift)

function shiftcenter(V::VPolytope, shift::Vector)
    shifted = [v + shift for v in vertices_list(V)]
    return VPolytope(shifted)
end


#=
meet and join will still become necessary in the zonotope case
The Case type is probably not necessary for correct dispatch
=#
# function meet(case::Case, H::HPolytope)
#     addconstraint!(H, constraint(case, H))
#     return H
# end

# function constraint(case::Type{Case}, n_dims, constraint_dim)
#     space = zeros(n_dims)
#     if case == Pos
#         space[constraint_dim] =  1.0
#     elseif case == Neg
#         space[constraint_dim] = -1.0
#     end
#     return HalfSpace(space, 0.0)
# end

# constraint(case::Case, poly::AbstractPolytope) = constraint(typeof(case), dim((poly), case.i)
