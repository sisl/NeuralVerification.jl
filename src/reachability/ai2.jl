
struct Ai2 end

function solve(solver::Ai2, problem::Problem)
    reach = forward_network(solver, problem.network, problem.input)

    #TODO is this the correct return pattern?
    if check_inclusion(reach, problem.output)
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

function transform(f::ReLU, P::AbstractPolytope)
    pos = meet(P, true)
    neg = meet(P, false)

    Vneg = VPolytope([f(v) for v in vertices_list(neg)])

    return convex_hull(pos, Vneg)
end

meet(V::VPolytope, pos) = tovrep(meet(tohrep(V), side))

function meet(H::HPolytope, pos)
    HH = deepcopy(H)
    meet!(HH, pos)
    return HH
end
function meet!(H::HPolytope{T}, pos::Bool) where T
    # constraints are given by ax <= b so (-) is required for a positive constraint
    if (pos)  d = -eye(dim(H))
    else      d =  eye(dim(H))
    end

    for i in size(d, 1)
        new_hs = HalfSpace(d[i, :], zero(T))
        addconstraint!(H, new_hs)
    end
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
