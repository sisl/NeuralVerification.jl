struct VariableBound
    finite::Bool # Whether the bound is finite or not
    bound::Float64 # The current value of the bound
    level::UInt32 # 
end

function set_bound(variableBound::VariableBound, bound::Float64)
	variableBound.finite = true
    variableBound.bound = bound
    return Result(:True)
end

function finite(variableBound::VariableBound)
	return variableBouund.finite
end

function getBound(variableBound::VariableBound)
	return variableBouund.finite
end

function set_level(variableBound::VariableBound, level::UInt32)
    variableBound.level = level
    return Result(:True)
end

function getLevel(variableBound::VariableBound)
	return variableBouund.level
end