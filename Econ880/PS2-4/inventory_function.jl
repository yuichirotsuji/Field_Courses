using Parameters, LinearAlgebra, Random, Distributions, Optim, Statistics, DataFrames, StatFiles, CSVFiles 

@with_kw struct Primitives
    β::Float64 = 0.99   # discount factor
    α::Float64 = 2.0    # utility parameter for consumption
    λ::Float64 = -4.0   # "true" stockout penalty

    # grid for λ (just for log-likelihood function plot)
    λ_min::Float64 = -8.0
    λ_max::Float64 = 0.0
    nλ::Int64 = 101
    λ_grid::Vector{Float64} = range(λ_min, stop=λ_max, length=nλ)

    # grid for inventory stock
    ni::Int64 = 9
    i_min::Int64 = 0
    i_max::Int64 = 8
    i_grid::Vector{Int64} = collect(i_min:i_max) .|> Int64

    # grid for consumption shocks
    c_grid::Vector{Float64} = [0.0, 1.0]
    c₀::Float64 = c_grid[1]
    c₁::Float64 = c_grid[2]
    nc::Int64 = length(c_grid)

    # grid for prices
    p_grid::Vector{Float64} = [4.0, 1.0]
    p₁::Float64 = p_grid[1]
    p₀::Float64 = p_grid[2]
    np::Int64 = length(p_grid)

end

@with_kw mutable struct Results
    V::Vector{Float64}        # value function (with random shock) V(a;s)
    pol_func::Vector{Float64} # policy function a'(a;s) (may not be used)
    P::Vector{Float64}        # CCP vector (i.e. P(a=1|s))

    Π₀::Vector{Float64}       # flow profit vector when taking a=1
    Π₁::Vector{Float64}       # flow profit vector when taking a=0
    E₀::Vector{Float64}       # conditional expectation vector of ϵ₀
    E₁::Vector{Float64}       # conditional expectation vector of ϵ₁
end

#initialization function at everywhere
@everywhere function Initialize()
    prim = Primitives()

    V = zeros(Float64, prim.ni*prim.nc*prim.np)
    pol_func = zeros(Float64, prim.ni*prim.nc*prim.np)
    P = zeros(Float64, prim.ni*prim.nc*prim.np)
    Π₀ = zeros(Float64, prim.ni*prim.nc*prim.np)
    Π₁ = zeros(Float64, prim.ni*prim.nc*prim.np)
    E₀ = zeros(Float64, prim.ni*prim.nc*prim.np)
    E₁ = zeros(Float64, prim.ni*prim.nc*prim.np)
    res = Results(V, pol_func, P, Π₀, Π₁, E₀, E₁)

    return prim, res
end

#######################################################################
# data initalization
@with_kw struct S_data
    # Read state space data
    df_S = DataFrame(load("PS4_state_space.csv")) # load data from csv file
    S = Matrix(df_S[:,3:end])

    # Read transition matrices
    df_F₀ = DataFrame(load("PS4_transition_a0.csv")) # load data from csv file
    F₀ = Matrix(df_F₀[:,3:end])
    df_F₁ = DataFrame(load("PS4_transition_a1.csv")) # load data from csv file
    F₁ = Matrix(df_F₁[:,3:end])

    # Read simulated data
    df_D = DataFrame(load("PS4_simdata.csv")) # load data from csv file
    D = Matrix(df_D[:,2:end])

end

function get_data()
    data = S_data()
    return data
end

#######################################################################
# get (expected) flow profit of today
function get_Π(λ₀::Float64, a::Int64, i::Int64, c::Float64, p::Float64, prim::Primitives)
    if a == 1
        Π = prim.α*c - p
    elseif a == 0
        if i == 0
            Π = λ₀*c
        else
            Π = prim.α*c
        end
    end
    return Π
end

# Bellman operator (i.e. Γ(V))
function Bellman_V(prim::Primitives, res::Results, data::S_data)
    @unpack α, β, λ, i_grid, c_grid, p_grid, ni, nc, np = prim # unpack model primitives
    @unpack F₀, F₁ = data                                      # unpack transition matrix data

    V_next::Vector{Float64} = zeros(Float64, ni*nc*np) # container for next guess V_max (36*1 vector)

    # compute next guess of V
    for i_index in eachindex(i_grid)
        for c_index in eachindex(c_grid)
            for p_index in eachindex(p_grid)
                i = i_grid[i_index] # get today's inventory stock
                c = c_grid[c_index] # get today's consumption shock
                p = p_grid[p_index] # get today's price

                # get today's state (1-36)
                state = Int64((i_index) + (9*(c+floor(p_index*1.5)-1)))
                #println("current state = ", state)

                # compute next guess of V(s)
                V_next[state] = log(exp(get_Π(λ,0,i,c,p,prim)+β*(F₀[state,:]'*res.V))+ exp(get_Π(λ,1,i,c,p,prim)+β*(F₁[state,:]'*res.V))) + SpecialFunctions.γ

                #=
                # get (expected) flow profit of today
                Π₁ = α*c - p
                if i == 0
                    Π₀ = λ*c
                else
                    Π₀ = α*c
                end
                # compute next guess of V(s)
                V_next[state] = log(exp(Π₀+β*(F₀[state,:]'*res.V))+ exp(Π₁+β*(F₁[state,:]'*res.V))) + SpecialFunctions.γ
                =#
            end
        end
    end

    return V_next # return next guess of V_max (36*1 vector)
end

# Iteration for V(S)
function V_iterate(prim::Primitives, res::Results, data::S_data; tol::Float64 = 1e-14, max_iter = 5000)
    n = 0     # iteration counter
    err = 100 # initial error: some big number

    while err > tol && n < max_iter
        V_next = Bellman_V(prim, res, data)  # compute next guess of V
        err = maximum(abs.(V_next .- res.V)) # calculate sup(error)
        res.V .= V_next                      # update value function in struct
        n += 1                               # iteration count

        #println("Current error = ", err)
    end
    
    #println("Finar error = ", err)
    println("Value function converged in ", n, " iterations.")

end

#######################################################################
# CCP estimator using frequency
function CCP_estimator_freq(prim::Primitives, data::S_data)
    @unpack D = data
    P_choice::Vector{Float64} = zeros(Float64, prim.ni*prim.nc*prim.np) # container for cum choice
    P_freq::Vector{Float64} = zeros(Float64, prim.ni*prim.nc*prim.np)   # conatiner for frequency

    # inspect states and choices in the data
    for i in 1:size(D,1)
        state = D[i,2] + 1          # get state for each m        
        P_choice[state] += D[i,1]   # add a∈{0,1} to cum-choice vector 
        P_freq[state] += 1          # add 1 to freq vector
    end

    # calculate choice probability
    P::Vector{Float64} = zeros(Float64, prim.ni*prim.nc*prim.np)
    for r in 1:prim.ni*prim.nc*prim.np
        p₁ = P_choice[r] / P_freq[r]
        if p₁ < 0.001       # too little actions -> convert to P(s)=0.001
            P[r] = 0.001
        elseif p₁ > 0.999   # too many actions -> convert to P(s)=0.999
            P[r] = 0.999
        else                # for other cases, just use the frequency-based probability
            P[r] = p₁
        end
    end

    return P
end

#######################################################################
# compute complete vector of flow profit
function get_Π_vector(λ₀::Float64, prim::Primitives, res::Results)
    @unpack α, β, i_grid, c_grid, p_grid = prim
    for i_index in eachindex(i_grid)
        for c_index in eachindex(c_grid)
            for p_index in eachindex(p_grid)
                i = i_grid[i_index] # get today's inventory stock
                c = c_grid[c_index] # get today's consumption shock
                p = p_grid[p_index] # get today's price

                # get today's state (1-36)
                state = Int64((i_index) + (9*(c+floor(p_index*1.5)-1)))

                res.Π₀[state] = get_Π(λ₀,0,i,c,p,prim) # flow profit of today if a=0
                res.Π₁[state] = get_Π(λ₀,1,i,c,p,prim) # flow profit of today if a=1
            end
        end
    end
end

# get (Conditional) expectation of ϵₐ
function Exp_e(res::Results)
    res.E₀ =  SpecialFunctions.γ .- log.(1 .- res.P)
    res.E₁ = SpecialFunctions.γ .- log.(res.P)
end

# (CCP_implied) expected value function calculation
function V_CCP(prim::Primitives, res::Results, data::S_data)
    @unpack β, ni, nc, np = prim
    @unpack F₀, F₁ = data

    Exp_e(res) # get (Conditional) expectation of ϵₐ

    Π_bar = ((1 .- res.P) .* (res.Π₀+res.E₀)) + (res.P .* (res.Π₁+res.E₁)) # flow profit matrix (RHSof linear eq) 
    Fp = (((1 .- res.P) .* F₀) + (res.P .* F₁))                            # transition matrix
    A = 1.0*I(ni*nc*np) - β*Fp                                             # get LHS of linear EQ
    return A\Π_bar                                                         # solve for V by LU decomposition method
end

# CCP mapping (i.e. Ψ(P))
function CCP_mapping(prim::Primitives, res::Results, data::S_data)
    @unpack β = prim
    @unpack F₀, F₁ = data

    EV = V_CCP(prim, res, data)                       # get V via CCP-implied value
    v_tilde = (res.Π₁ + β*F₁*EV) - (res.Π₀ + β*F₀*EV) # get v_tilde using formula in slides
    P_next = 1 ./ (1 .+ exp.(-v_tilde))               # get next guess of P(s)

    return P_next
end

# Iteration for P(X)
function P_iterate(λ₀::Float64, prim::Primitives, res::Results, data::S_data; tol::Float64 = 1e-13, max_iter = 10000)
    n = 0     # iteration counter
    err = 100 # initial error: some big number

    get_Π_vector(λ₀,prim,res)             # get flow profit vectors (update struct)
    res.P = CCP_estimator_freq(prim,data) # Initial guess of P(X) 

    while err > tol && n < max_iter
        P_next = CCP_mapping(prim, res, data)             # compute next guess of P
        err = maximum(abs.(log.(P_next) .- log.(res.P)))  # calculate sup(log(error))
        res.P .= P_next                                   # update policy function in struct
        n += 1                                            # iteration count

        println("Current error = ", err)
    end
    
    #println("Finar error = ", err)
    println("Policy function converged in ", n, " iterations.")

end

#######################################################################
# compute log-liklihood
function nested_log_liklihood(λ₀::Float64, prim::Primitives, res::Results, data::S_data)

    P_iterate(λ₀,prim,res,data)             # get P(s) by policy function iteration

    L = 0                                   # log-liklihood is scalar-valued
    A = Vector(data.df_D[:,:choice])        # get action data
    S = Vector(data.df_D[:,:state_id]) .+ 1 # get state data

    # We calculate each indiviudal's choice probabilities and take sum to get log-likelihood
    for i_index in 1:length(A) 
        state = S[i_index]
        L += A[i_index]*log(res.P[state]) + (1.0-A[i_index])*log(1.0 - res.P[state])
    end

    return L
end

# a function to apply Optim.optimize package
function nll(λ₀)
    return -nested_log_liklihood(λ₀[1], prim, res, data) # objective function = -(nested_LL)
end

# likelihood function plot
function nested_ll_plot(prim::Primitives, res::Results, data::S_data)
    @unpack λ_grid, nλ = prim
    LL_val::Vector{Float64} = zeros(nλ)
    for (λ_index,λ) in enumerate(λ_grid)
        LL_val[λ_index] = nested_log_liklihood(λ,prim,res,data) # get the value of GMM(λ) and store it in memory
    end
    return LL_val
end
#######################################################################
