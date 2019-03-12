# This contains several objectives

##
# TODO make objectives have ! names
##
# Objective: L∞ norm of the disturbance
function max_disturbance!(model::Model, var)
    o = symbolic_infty_norm(var)
    @objective(model, Min, o)
    return o
end

function min_sum!(model::Model, var)
    o = sum(sum.(var))
    @objective(model, Min, o)
    return o
end

function max_sum!(model::Model, var)
    o = sum(sum.(var))
    @objective(model, Max, o)
    return o
end

# This is the default when creating a model. Only used for explicit-ness.
feasibility_problem!(model::Model) = nothing

function linear_objective!(mode::Model, map::HPolytope, var)
    c, d = tosimplehrep(map)
    o = c * var - d
    @objective(model, Min, o)
    return o
end
