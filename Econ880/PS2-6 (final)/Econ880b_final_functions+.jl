using Parameters, LinearAlgebra, Random, Distributions, Optim, Statistics, DataFrames, StatFiles, CSVFiles 

#######################################################################
# Functions for parameter settings and result storage
#######################################################################
# declare parameters struct
@with_kw struct Primitives
    β::Float64 = 0.96   # discount factor
    α::Float64 = -0.75  # "true" loan size disutility parameter

    κₕ::Float64 = 5.0   # "true" refinancing cost (High type)
    κₗ::Float64 = 1.0   # "true" refinancing cost (Low type)

    # parameters for l (loan size)
    nl::Int64 = 8
    l_grid::Vector{Float64} = [50.0,100.0,150.0,200.0,250.0,300.0,350.0,400.0]

    # parameters for r (interest rate)
    nr::Int64 = 13
end

# declare results struct
@with_kw mutable struct Results
    V::Vector{Float64}        # value function V(a;s,κ) (may not be used)
    pol_func::Vector{Float64} # policy function a'(a|s,κ) (may not be used)

    Pₕ::Vector{Float64} # CCP vector for high type (i.e. P(a=1|s,κₕ))
    Pₗ::Vector{Float64} # CCP vector for low type (i.e. P(a=1|s,κₗ))

    Π₀::Vector{Float64} # flow profit vector when taking a=0 (i.e. not refinance), same for both type
    Π₁ₕ::Vector{Float64} # flow profit vector when taking a=1 (i.e. refinance) for high type
    Π₁ₗ::Vector{Float64} # flow profit vector when taking a=1 (i.e. refinance) for low type
    E₀ₕ::Vector{Float64} # conditional expectation vector of ϵ₀ for high type
    E₁ₕ::Vector{Float64} # conditional expectation vector of ϵ₁ for high type
    E₀ₗ::Vector{Float64} # conditional expectation vector of ϵ₀ for low type
    E₁ₗ::Vector{Float64} # conditional expectation vector of ϵ₁ for low type
end

# initialize parameters and results
@everywhere function Initialize()
    prim = Primitives()

    V = zeros(Float64, prim.nr*prim.nr*prim.nl)
    pol_func = zeros(Float64, prim.nr*prim.nr*prim.nl)
    Pₕ = zeros(Float64, prim.nr*prim.nr*prim.nl)
    Pₗ = zeros(Float64, prim.nr*prim.nr*prim.nl)
    Π₀ = zeros(Float64, prim.nr*prim.nr*prim.nl)
    Π₁ₕ = zeros(Float64, prim.nr*prim.nr*prim.nl)
    Π₁ₗ = zeros(Float64, prim.nr*prim.nr*prim.nl)
    E₀ₕ = zeros(Float64, prim.nr*prim.nr*prim.nl)
    E₁ₕ = zeros(Float64, prim.nr*prim.nr*prim.nl)
    E₀ₗ = zeros(Float64, prim.nr*prim.nr*prim.nl)
    E₁ₗ = zeros(Float64, prim.nr*prim.nr*prim.nl)
    res = Results(V, pol_func, Pₕ, Pₗ, Π₀, Π₁ₕ, Π₁ₗ, E₀ₕ, E₁ₕ, E₀ₗ, E₁ₗ)

    return prim, res
end

#######################################################################
# Functions for data loading
#######################################################################
# daclare data struct
@with_kw struct Loanpaymentdata
    df::DataFrame           # simulated dataset

    df_r_trans::DataFrame   # transition matrix for rₜ (raw dataframe)
    Fᵣ::Matrix{Float64}     # transition matrix for rₜ (13×13 matrix)

    df_r_grid::DataFrame    # interest rate grid (raw dataframe)
    r_grid::Vector{Float64} # interest rate grid (13×1 vector)

    F₀::Matrix{Float64}     # Full transition matrix F(s'|s,a=0) (1352×1352)
    F₁::Matrix{Float64}     # Full transition matrix F(s'|s,a=1) (1352×1352)
end

# a function to construct data struct, including the full transition matrices
function get_data()
    @unpack nl, nr = prim
    # Read simulated dataset
    df = DataFrame(load("Mortgage_simdata_students.dta"))      # load data from dta file

    # Read transition matrix
    df_r_trans = DataFrame(load("Transition_prob_matrix.csv")) # load data from csv file
    Fᵣ = Matrix(df_r_trans[:,2:end])

    # Read grid of interest rate
    df_r_grid = DataFrame(load("Interest_rate_grid.csv"))      # load data from csv file
    r_grid = Vector(df_r_grid[:,2])

    # create full transition matrix
    S = nr*nr*prim.nl # number of state (S = 1,352 = 13*13*8)

    F₀ = zeros(S,S) # F(s'|s,a=0): simple reprecation of Fᵣ
    for i in 1:nr*nl
        start = 13*i - 12
        last = 13*i
        F₀[start:last,start:last] = Fᵣ
    end

    F₁ = zeros(S,S) # F(s'|s,a=1): r₀' depends on today's interest rate
    F₁_mid = zeros(nr*nr,nr*nr)
    F₁_small = zeros(nr,nr*nr)
    for j in 1:nr
        start = 13*j - 12
        F₁_small[j,start:start+12] = Fᵣ[j,:]
    end
    for k in 1:nr
        start = 13*k - 12
        F₁_mid[start:start+12,:] = F₁_small
    end
    for l in 1:nl
        start = 169*l - 168
        last = 169*l
        F₁[start:last,start:last] = F₁_mid
    end

    data = Loanpaymentdata(df,df_r_trans,Fᵣ,df_r_grid,r_grid,F₀,F₁)
    return data
end

#######################################################################
# Functions for computing CCP mapping
#######################################################################
# a function to get (expected) flow profit
function get_Π(α::Float64,κ::Float64, a::Int64, r::Float64, r₀::Float64, l::Float64, prim::Primitives)
    if a == 1     # refinancing
        Π = prim.α*l*(r/200) - κ
    elseif a == 0 # not refinancing
        Π = prim.α*l*(r₀/200)
    end
    return Π
end

# compute complete vector of flow profit (depends on action: two (1352*1) vectors)
function get_Π_vector(α::Float64, κₕ::Float64, κₗ::Float64, prim::Primitives, res::Results, data::Loanpaymentdata)
    @unpack l_grid = prim
    @unpack r_grid = data
    for l_index in eachindex(l_grid)
        for r₀_index in eachindex(r_grid)
            for r_index in eachindex(r_grid)
                l = l_grid[l_index]   # get loan size (50K-400K)
                r₀ = r_grid[r₀_index] # get current loan's rate
                r = r_grid[r_index]   # get market interest rate

                # get state index (1-1352)
                state = (r_index) + (r₀_index-1)*13 + (l_index-1)*169
                #println(state) # to check whether index is correct

                res.Π₀[state] = get_Π(α,κₕ,0,r,r₀,l,prim)  # flow profit of today if a=0 (No refinance)
                res.Π₁ₕ[state] = get_Π(α,κₕ,1,r,r₀,l,prim) # flow profit of today if a=1 (refinance) of high type
                res.Π₁ₗ[state] = get_Π(α,κₗ,1,r,r₀,l,prim) # flow profit of today if a=1 (refinance) of low type
            end
        end
    end
end

# a function to get (Conditional) expectation of ϵₐ
function Exp_e(res::Results)
    res.E₀ₕ =  SpecialFunctions.γ .- log.(1 .- res.Pₕ)
    res.E₁ₕ = SpecialFunctions.γ .- log.(res.Pₕ)
    res.E₀ₗ =  SpecialFunctions.γ .- log.(1 .- res.Pₗ)
    res.E₁ₗ = SpecialFunctions.γ .- log.(res.Pₗ)
end

# (CCP_implied) expected value function calculation
function V_CCP(prim::Primitives, res::Results, data::Loanpaymentdata, high::Bool)
    @unpack β, nl, nr = prim
    @unpack F₀, F₁ = data

    Exp_e(res) # get (Conditional) expectation of ϵₐ

    if high == true # for high type
        Π_bar = ((1 .- res.Pₕ) .* (res.Π₀+res.E₀ₕ)) + (res.Pₕ .* (res.Π₁ₕ+res.E₁ₕ)) # flow profit matrix (RHS of linear eq) 
        Fp = (((1 .- res.Pₕ) .* F₀) + (res.Pₕ .* F₁))                              # transition matrix
        A = 1.0*I(nr*nr*nl) - β*Fp                                                # get LHS of linear eq
        return A\Π_bar                                                            # solve for V using LU decomposition
    elseif high == false # for low type
        Π_bar = ((1 .- res.Pₗ) .* (res.Π₀+res.E₀ₗ)) + (res.Pₗ .* (res.Π₁ₗ+res.E₁ₗ)) # flow profit matrix (RHS of linear eq) 
        Fp = (((1 .- res.Pₗ) .* F₀) + (res.Pₗ .* F₁))                             # transition matrix
        A = 1.0*I(nr*nr*nl) - β*Fp                                               # get LHS of linear eq
        return A\Π_bar                                                           # solve for V using LU decomposition
    end

end

# CCP mapping (i.e. Ψ(P))
function CCP_mapping(prim::Primitives, res::Results, data::Loanpaymentdata, high::Bool)
    @unpack β = prim
    @unpack F₀, F₁ = data

    if high == true # for high type
        EV = V_CCP(prim, res, data,true)                   # get V^bar via CCP-implied value
        v_tilde = (res.Π₁ₕ + β*F₁*EV) - (res.Π₀ + β*F₀*EV) # get v_tilde using formula in slides
        P_next = 1 ./ (1 .+ exp.(-v_tilde))                # get next guess of P(s)
    elseif high == false # for low type
        EV = V_CCP(prim, res, data,false)                  # get V^bar via CCP-implied value
        v_tilde = (res.Π₁ₗ + β*F₁*EV) - (res.Π₀ + β*F₀*EV) # get v_tilde using formula in slides
        P_next = 1 ./ (1 .+ exp.(-v_tilde))               # get next guess of P(s)
    end

    return P_next
end

# Iteration for P(s;κ)
function P_iterate(α::Float64, κₕ::Float64, κₗ::Float64, prim::Primitives, res::Results, data::Loanpaymentdata; tol::Float64 = 1e-13, max_iter = 1000)
    nₕ = 0     # iteration counter for high type iteration
    nₗ = 0     # iteration counter for low type iteration
    errₕ = 100 # initial error: some big number for high type iteration
    errₗ = 100 # initial error: some big number for low type iteration

    get_Π_vector(α,κₕ,κₗ,prim,res,data)             # get flow profit vectors (update struct)
    res.Pₕ = ones(prim.nr*prim.nr*prim.nl) * (1/2) # Initial guess of P(S;κₕ) 
    res.Pₗ = ones(prim.nr*prim.nr*prim.nl) * (1/2) # Initial guess of P(S;κₗ) 

    # high type (Pₕ) iteration
    while errₕ > tol && nₕ < max_iter
        Pₕ_next = CCP_mapping(prim, res, data,true)         # compute next guess of P
        errₕ = maximum(abs.(log.(Pₕ_next) .- log.(res.Pₕ))) # calculate sup(log(error))
        res.Pₕ .= Pₕ_next                                   # update policy function in struct
        nₕ += 1                                            # iteration count
        # println("Current error = ", errₕ)
    end
    #println("Policy function of high type converged in ", nₕ, " iterations.")

    # low type (Pₗ) iteration
    while errₗ > tol && nₗ < max_iter
        Pₗ_next = CCP_mapping(prim,res,data,false)          # compute next guess of P
        errₗ = maximum(abs.(log.(Pₗ_next) .- log.(res.Pₗ)))  # calculate sup(log(error))
        res.Pₗ .= Pₗ_next                                   # update policy function in struct
        nₗ += 1                                            # iteration count
        #println("Current error = ", errₗ)
    end
    #println("Policy function of low type converged in ", nₗ, " iterations.")

end

#######################################################################
# Functions to calculate log-likelihood
#######################################################################
#= 
# a function to compute log-liklihood
# (Note: parameters to estimate = (α,κₕ,κₗ,a), where π = exp(a)/(1+exp(a)))
# (Note2: this function is incorrect: please see)
function nested_log_liklihood(α::Float64, κₕ::Float64, κₗ::Float64, a::Float64, prim::Primitives, res::Results, data::Loanpaymentdata)

    P_iterate(α,κₕ,κₗ,prim,res,data)  # get P(s|high) and P(s|low) by policy function iteration

    # get action and state data from dataframe
    A = Float64.(data.df.refi)
    S = hcat(Float64.(data.df.loan_size),Float64.(data.df.rate),Float64.(data.df.rate_t))

    # create vectors to store likelihood contributions
    Lₕ::Vector{Float64} = zeros(length(A)) # for high type
    Lₗ::Vector{Float64} = zeros(length(A)) # for low type

    # calculate each indiviudal's choice probabilities by type
    for i_index in 1:length(A)
        # get state id given data
        state = Int((((S[i_index,1]/50)-1)*169.0) + (((S[i_index,2]/0.25 - 7.0) -1)*13.0) + (S[i_index,3]/0.25 - 7.0))
        #println(state)

        # calculate likelihood contributions
        Lₕ[i_index] = A[i_index]*res.Pₕ[state] + (1.0-A[i_index])*(1.0 - res.Pₕ[state]) # high type
        Lₗ[i_index] = A[i_index]*res.Pₗ[state] + (1.0-A[i_index])*(1.0 - res.Pₗ[state])  # low type
    end

    π = exp(a)/(1+exp(a))         # logit transformation for type probability
    L = sum(log.(π*Lₗ + (1-π)*Lₕ)) # calculate log-likelihood to evaluate
    return L
end

# a function to apply Optim.optimize package
function nll(θ)
    return -nested_log_liklihood(θ[1],θ[2],θ[3],θ[4], prim, res, data)/1000 # objective function = -(nested_LL)
end
=#

# a function to compute average log-liklihood
# (Note: parameters to estimate = (α,κₕ,κₗ,a), where π = exp(a)/(1+exp(a)))
function nested_log_liklihood_ave(α::Float64, κₕ::Float64, κₗ::Float64, a::Float64, prim::Primitives, res::Results, data::Loanpaymentdata)

    P_iterate(α,κₕ,κₗ,prim,res,data)  # get P(s|high) and P(s|low) by policy function iteration

    # get action and state data from dataframe
    A = Float64.(data.df.refi)
    S = hcat(Float64.(data.df.loan_size),Float64.(data.df.rate),Float64.(data.df.rate_t))
    id = Int64.(data.df.id)

    # create vectors to store likelihood contributions
    Lₕ::Vector{Float64} = zeros(length(A)) # for high type
    Lₗ::Vector{Float64} = zeros(length(A)) # for low type

    # calculate each indiviudal's choice probabilities by type
    for i_index in 1:length(A)
        # get state id given data
        state = Int((((S[i_index,1]/50)-1)*169.0) + (((S[i_index,2]/0.25 - 7.0) -1)*13.0) + (S[i_index,3]/0.25 - 7.0))
        #println(state)

        # calculate likelihood contributions
        Lₕ[i_index] = A[i_index]*res.Pₕ[state] + (1.0-A[i_index])*(1.0 - res.Pₕ[state]) # high type
        Lₗ[i_index] = A[i_index]*res.Pₗ[state] + (1.0-A[i_index])*(1.0 - res.Pₗ[state])  # low type
    end

    π = exp(a)/(1+exp(a)) # logit transformation for type probability
    L = 0.0               # a container for log-likelihood (scalar)

    # get log-likelihood contributions by each agent
    for i_index in 0:999
        indv = findall(x -> x == i_index, id) # get data index of individual i
        #println(length(indv))

        # calculate individual-time chain choice probability
        valₕ = 1.0
        valₗ = 1.0
        for it in eachindex(indv)
            valₕ *= Lₕ[indv[it]]
            valₗ *= Lₗ[indv[it]]
        end

        L += log((1-π)*valₕ+π*valₗ) # acutual contribution = weighted average by π
    end

    return L/1000 # evaluate average log-likelihood
end

# a function to apply Optim.optimize package
function nll_ave(θ)
    return -nested_log_liklihood_ave(θ[1],θ[2],θ[3],θ[4], prim, res, data) # objective function = -(nested_LL)
end