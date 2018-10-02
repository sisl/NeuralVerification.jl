# This contains several objectives

# Objective: L∞ norm of the disturbance
function max_disturbance(model::Model, var)
    J = maximum(symbolic_abs(var))
    @objective(model, Min, J)
    return J
end