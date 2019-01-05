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

function min_sum_all!(model::Model, var)
    o = sum(sum.(var))
    @objective(model, Min, o)
    return o
end

function max_sum_all!(model::Model, var)
    o = sum(sum.(var))
    @objective(model, Max, o)
    return o
end

function zero_objective!(model::Model)
    @objective(model, Max, 0.0)
end

function linear_objective!(mode::Model, map::HPolytope, var)
    c, d = tosimplehrep(map)
    o = c * var - d
    @objective(model, Min, o)
    return o
end
