# sanity checks 

using NeuralVerification
using Base.Test

macro no_error(ex)
    quote
        try $(esc(ex))
            true
        catch
            false
        end
    end
end