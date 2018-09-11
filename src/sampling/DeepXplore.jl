
# Sampling method 1: DeepXplore
#=
Neuron coverage: the ratio of unique activated* neurons for all test inputs aand the total number of neurons in the DNN.
activated* : output value higher than a threshold. (often 0)

neurons:     N = {n₁, n₂ ...}
test inputs: T = {x⃗₁, x⃗₂, ...}
out(n, x⃗) returns the output value of n for x⃗

(neuron coverage)
NCov(T, x) = |{n| ∀x ∈ T, out(n, x) > t}| // |N|   --> N_activated / N_total

∴ NCov(dnn, coverage_tracker) = length(coverage_tracker) / n_neurons(dnn)

# [87] for details on gradients

∇f = ∇ₓf(θ, x⃗) = δy/δx⃗  --> where y = f(θ, x⃗) represents the activation function and theta and x are the parameters and test intput of a DNN



=#
"""
Algorithm 1 Test input generation via joint optimization
    Input: seed_set ← unlabeled inputs as the seeds
    dnns ← multiple DNNs under test
    λ1 ← parameter to balance output differences of DNNs (Equation
    2)
    λ2 ← parameter to balance coverage and differential behavior
    s ← step size in gradient ascent
    t ← threshold for determining if a neuron is activated
    p ← desired neuron coverage
    cov_tracker ← tracks which neurons have been activated

/* main procedure */
gen_test := empty set
for cycle(x ∈ seed_set) do // infinitely cycling through seed_set
    /* all dnns should classify the seed input to the same class */
    c = dnns[0].predict(x)
    d = randomly select one dnn from dnns
    while True do
        obj1 = COMPUTE_OBJ1(x, d, c, dnns, λ1)
        obj2 = COMPUTE_OBJ2(x, dnns, cov_tracker)
        obj = obj1 + λ2 · obj2
        grad = ∂obj / ∂x
        /*apply domain specific constraints to gradient*/
        grad = DOMAIN_CONSTRNTS(grad)
        x = x + s · grad //gradient ascent
        if d.predict(x) , (dnns-d).predict(x) then
            /* dnns predict x differently */
            gen_test.add(x)
            update cov_tracker
            break
    if DESIRED_COVERAGE_ACHVD(cov_tracker) then
        return gen_test

"""

# Algorithm 1: Test input generation via joint optimization:
function generate_tests(seed_set::Inputs,
                        dnns::Vector{Network},
                        λ1::Float64,            # parameter to balance output difference of DNNs (equation 2)
                        λ2::Float64,            # parameter to balance coverage and differential behavior
                        s,                      # step size in gradient ascent
                        t::Float64,             # threshold for neuron activation
                        p::Float64              # desired neuron coverage
                        )

    gen_test = []      # set of generated test cases
    cov_tracker = []   # tracks which neurons have been activated

    for x in cycle(seed_set) #cycle forever
        c = predict(dnn[0], x) # same as compute_ouput(?)
        d = rand(dnn)
        rest = filter(net->net!=d, dnns)

        diff_found = false
        while !diff_found
            obj1 = compute_obj1(x, d, c, dnns, λ1)
            obj2 = compute_obj2(x, dnns, cov_tracker)
            obj = obj1 + λ2*obj2
            grad = δobj / δx                # compute gradient. What is δx for the first gradient? s?
            grad = domain_constraints(grad) # domain specific constraints (such as 0-255 for pixels, etc.)
            x = x + s*grad

            prediction = predict(d, x)
            diff_found = false
            for dnn in rest
                if predict(dnn, x) != prediction
                    push!(gen_test, x)
                    update!(cov_tracker, d, x)
                    diff_found = true
                    break
                end
            end
        end

        if DESIRED_COVERAGE_ACHEIVED(d, cov_tracker)
            return gen_test
        end
    end
end

# utils:
# obj1 relates to difference in classification confidence
function compute_obj1(x, d, c, dnns, λ1)
    loss1 = sum(confidence(x, net, c) for net in dnns if net!=d)
    loss2 =     confidence(x, d, c)
    return loss1 - λ1*loss2
end

# obj2 relates to neuron coverage
function compute_obj2(x, dnns, cov_tracker)
    loss = 0
    for dnn in dnns
        n = rand(cov_tracker[:unselected]) # select a neuron n inactivated so far using cov_tracker
        loss += out(n, x)
    end
    return loss
end

function DESIRED_COVERAGE_ACHEIVED(cov_tracker, dnn, p)
    n_neurons = sum(length(dnn.layers[i].bias) for i in 1:length(dnn.layers))
    covered = length(cov_tracker)
    return covered/n_neurons > p
end

# Let Fk(x)[c] [[F(x, c, k)]] be the class probability that Fk predicts x to
# be c. We randomly select one neural network Fj (Algorithm 1
# line 6) and maximize the following objective function:
"""obj1(x) = Σk≢j Fk(x)[c] − λ1*Fj(x)[c] # for a constant Fj"""
obj1(x) = sum(F(x, c, k)- λ1*F(x, c, j) for k in 1:length(dnns), j in 1:length(dnns) if k != j)
# maximize obj2(x) = fn(x) such that fn(x) > t
"""obj_joint(x) = Σi≢j (Fi(x)[c] − λ1Fj(x)[c]) + λ2*fn(x)"""
obj_joint(x) = sum(F(x, c, i) - λ1*F(x, c, j) for i in 1:length(dnns), j in 1:length(dnns) if i != j) + λ2*fn(x)

