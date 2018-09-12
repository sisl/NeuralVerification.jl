# Sampling method 2: DeepTest

#=

# for steering angle theta:
    θₒ = {θₒ1, θₒ2 ..., θₒn}            # output for orignal images
    θ  = {θ1, θ2 ..., θn}               # manually created labels
    MSE_orig = mean(abs2.(θ .- θₒ))     # mean squared error
    abs2.(θ .- θt) <= λ*MSE_orig        # output should obey this


allowed transformations:
 * contrast,
 * fog,
 * rain,
 * rotation,
 * blur (gaussian, median, averaging, bilateral),
 * brightness,
 * shear,
 * translate
(maybe more)


Unclear:
* covInc operates inside the loop? I.e. coverage is measured between each given seed image and it's tranformed counterparts, not increasing monotonically.
* do you reset the random parameter P1 even in case that P1 worked previously?
* Are only two transformations allowed on an image? Online they have examples of 1:6, so is that a separate situation?
* What is the range of the allowed parameters? Surely a scaling of 100x and rotation of 130° are not appropriate.
    - table 4
=#

"""
Algorithm 1: Greedy search for combining image tranformations
to increase neuron coverage
    Input :Transformations T, Seed images I
    Output : Synthetically generated test images
    Variable : S: stack for storing newly generated images
    Tqueue: transformation queue

    Push all seed imgs ∈ I to Stack S
    genTests = ϕ
    while S is not empty do
        img = S.pop()
        Tqueue = ϕ
        numFailedTries = 0
        while numFailedTries ≤ maxFailedTries do
            if Tqueue is not empty then
                T1 = Tqueue.dequeue()
            else
                Randomly pick transformation T1 from T
            end
            Randomly pick parameter P1 for T1
            Randomly pick transformation T2 from T
            Randomly pick parameter P2 for T2
            newImage = ApplyTransforms(image, T1, P1, T2, P2)
            if covInc(newimage) then
                Tqueue.enqueue(T1)
                Tqueue.enqueue(T2)
                UpdateCoverage()
                genTest = genTests ∪ newimage S.push(newImage)
            else
                numFailedTries = numFailedTries + 1
            end
        end
     end
     return genTests
 """
# Algorithm 1: Greedy search for combining image tranformations to increase neuron coverage
function generate_tests(T::Vector{Tranformation}, I::Vector{Image}, nnet::Network)
    S = deepcopy(I)
    testcases = similar(I, 0)
    cov_tracker = Set()
    while !isempty(S)
        img = pop!(S)
        Tqueue = similar(T, 0)
        failed_tries = 0
        while failed_tries <= MAX_FAILED_TRIES  #reasonable value not mentioned
            T1 = isempty(Tqueue) ? rand(T) : pop!(Tqueue)
            T2 = rand(T)
            set_random_parameter!(T1)
            set_random_parameter!(T2)
            newimage = apply_transforms(img, T1, T2)          # Should always apply transform to img (or to newImage on future iterations)? Not clear in paper.
            if coverage_increase(newimage, nnet, cov_tracker) # covInc probably stands for coverage increase (not explained in paper)
                update_coverage!(newimage, nnet, cov_tracker)
                push!(testcases, newimage)
                push!(Tqueue, T1)
                push!(Tqueue, T2)
                push!(S, newImage)
            else
                failed_tries += 1
            end
        end
    end
    return testcases
end

coverage_increase(image, DNN, cov_tracker)  = true or false
update_coverage!(image, DNN, cov_tracker)   = should only update
coverage_increase!(image, DNN, cov_tracker) = should return true/false as well as update. One pass is more efficient.
set_random_parameter!(T::Transformation)    = probobaly have to create a transformation type. Best case scenario can use transformation operators from ImageTransformation.jl or some other package


function coverage_increase(img::AbstractArray, nnet::Network, cov_tracker::Set)

    return length(update_coverage(img, nnet, cov_tracker)) > length(cov_tracker)
end

function update_coverage(img, nnet, cov_tracker)
    new_cov_tracker = similar(cov_tracker)
    update_coverage!(img, nnet, new_cov_tracker)
    return new_cov_tracker
end

# might honestly be more efficient to keep an array of true/false
function update_coverage!(img, nnet, cov_tracker)
    for i in 1:length(nnet.layers)
        for j in 1:length(layers[i].bias)
            neuron_index = blah
            n = get_neuron(nnet, neuron_index)
            if out(n, img) > ACTIVATION_THRESHHOLD # 0.2 in the paper
                push(cov_tracker, neuron_index)
            elseif neuron_index ∈ cov_tracker
                pop!(cov_tracker, neuron_index)
            end
        end
    end
end
# requires implementing `out`, which works like compute_output(nnet, input, i, j)
# for the jth neuron in ith layer. Or compute_output(nnet, input, k) where k is the linear index

# CoordinateTransformations (which will probably be necessary) exports Transformation
abstract type Transformation end

# one way of doing things:
struct Rotation <: Transformation
    parameter
end
struct Scale <: Transformation
    parameter
end
struct Brightness <: Transformation
    parameter
end
struct Contrast <: Transformation
    parameter
end
struct Shear <: Transformation
    parameter
end
struct Translate <: Transformation
    parameter
end
struct Blur <: Transformation
    type  # function or Symbol
    parameter
end
struct Fog <: Transformation
    noisescale
    blending
end
struct Rain <: Transformation
    noisescale
    blending
end
#= range:

## Affine
    Translation  (tx, ty)   (10, 10)   to (100, 100) step (10, 10)
    Scale        (sx, sy)   (1.5, 1.5) to (6, 6)     step (0.5, 0.5)
    Shear        (sx, sy)   (−1.0, 0)  to (−0.1, 0)  step (0.1, 0)
    Rotation     (degree)   3          to 30         step 3

## Linear (color)
    Contrast     α (gain)   1.2  to   3.0  step 0.2
    Brightness   β (bias)   10   to   100  step 10

## Blur:
    averaging (kernel size)                         3x3, 4x4, 5x5, 6x6
    gaussian  (kernel size)                         3x3, 4x4, 7x7, 3x3  (repeat?)
    median    (aperture size)                       3, 5
    bilateral (diameter, sigmaColor, sigmaSpace)    9, 75, 75

=#


# Alternatively:

# covers shear, translate, scale, and rotation
struct AffineTransform{A<:AbstractArray, V<:AbstractVector} <: Transformation
    S::A
    b::V
end

# can cover constrast and brightness
# of the form constrast: (αx) or brightness: (x + β). Usually not both.
struct ColorTransform <: Transformation
    α::Float64
    β::Float64 # or int?
end

# still leaves fog, rain, and blur (maybe need their own types)
struct ConvolutionalTransform <: Transformation
    photoshop???
    # noisescale
    # blending_factor
end


# Yet another way
function transform!(A, tr::Symbol, params...)
    if tr == :Fog
        fog!(A, params...)
    elseif tr == :Rain
        rain!(A, params...)
    elseif tr == :Rotation
        rotate!(A, params...)
    elseif tr == :Scale
        scale!(A, params...)
    elseif tr == :Brightness
        brighten!(A, params...)
    elseif tr == :Contrast
        contrast!(A, params...)
    elseif tr == :Shear
        shear!(A, params...)
    elseif tr == :Translate
        translate!(A, params...)
    elseif tr == :Blur
        blur!(A, params...)
    end
end
# or similar to CoordinateTransformations
function transform(A, f, params...)
    return f(A, params...)
end
# however, generating params is type unstable. In the other cases Transforms themselves caused a type instability