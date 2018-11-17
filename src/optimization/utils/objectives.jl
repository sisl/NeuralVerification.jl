# This contains several objectives

##
# TODO make objectives have ! names
##
# Objective: L∞ norm of the disturbance
function max_disturbance(model::Model, var)
    J = maximum(symbolic_abs(var))
    @objective(model, Min, J)
    return J
end

function min_sum_all(model::Model, var)
    J = sum(sum.(var))
    @objective(model, Min, J)
    return J
end

function max_sum_all(model::Model, var)
    J = sum(sum.(var))
    @objective(model, Max, J)
    return J
end

function zero_objective(model::Model)
    @objective(model, Max, 0.0)
end

function linear_objective(mode::Model, map::HPolytope, var)
    c, d = tosimplehrep(map)
    J = c * var - d
    @objective(model, Min, J)
    return J
end
