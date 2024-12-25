using Parameters, LinearAlgebra, Random, Distributions, Optim, Statistics, DataFrames, StatFiles, CSVFiles #import the libraries we want

# struct for primiteves
@with_kw struct Primitives
    N::Int64 = 6103                       # number of obs in dataset
    T::Vector{Int64} = collect(1985:2015) # market "grid"
    R::Int64 = 100                        # number of simulation draws
    #seed::Int64 = 1234 # (we don't use this - simulation draws are given)
    L::Int64 = 6                          # dim(Z) (i.e. number of instruments)

    # set λ_grid for l(λ) plot
    λ_min::Float64 = 0.0
    λ_max::Float64 = 1.0
    nλ::Int64 = 101
    λ_grid::Vector{Float64} = range(λ_min, stop=λ_max, length=nλ)
    λ_grid₂::Vector{Float64} = range(λ_min, stop=2.0, length=nλ)
end

# struct for simulation & estimation results
@with_kw mutable struct Results
    D::Vector{Float64}  # simulated mean utility vector
    W::Matrix{Float64}  # weighting matrix in GMM objective function
end

# a function to initialize primiteves and results
function Initialize()
    prim = Primitives()

    D = zeros(prim.N)
    W = 1.0*I(prim.L)
    res= Results(D, W)

    return prim, res
end

#　struct-initalization 
@with_kw struct Cardata
    # Read car characteristics data (from dta)
    df_X = Float64.(DataFrame(load("Car_demand_characteristics_spec1.dta"))) # load data from dta file
    sorted_df_X = sort(df_X, :Year)                                          # sort by year 
    X = Matrix(sorted_df_X[:,[:hp2wt, :size, :turbo, :trans, :Year_1986, :Year_1987, :Year_1988, :Year_1989, :Year_1990,
                            :Year_1991, :Year_1992, :Year_1993, :Year_1994, :Year_1995, :Year_1996, :Year_1997, :Year_1998,
                            :Year_1999, :Year_2000, :Year_2001, :Year_2002, :Year_2003, :Year_2004, :Year_2005, :Year_2006,
                            :Year_2007, :Year_2008, :Year_2009, :Year_2010, :Year_2011, :Year_2012, :Year_2013, :Year_2014, 
                            :Year_2015, :model_class_2, :model_class_3, :model_class_4, :model_class_5, :cyl_2, :cyl_4, :cyl_6,
                            :cyl_8, :drive_2, :drive_3, :Intercept, :price]])  # RHS variables 
    Z₁ = Matrix(sorted_df_X[:,[:hp2wt, :size, :turbo, :trans, :Year_1986, :Year_1987, :Year_1988, :Year_1989, :Year_1990,
                            :Year_1991, :Year_1992, :Year_1993, :Year_1994, :Year_1995, :Year_1996, :Year_1997, :Year_1998,
                            :Year_1999, :Year_2000, :Year_2001, :Year_2002, :Year_2003, :Year_2004, :Year_2005, :Year_2006,
                            :Year_2007, :Year_2008, :Year_2009, :Year_2010, :Year_2011, :Year_2012, :Year_2013, :Year_2014, 
                            :Year_2015, :model_class_2, :model_class_3, :model_class_4, :model_class_5, :cyl_2, :cyl_4, :cyl_6,
                            :cyl_8, :drive_2, :drive_3, :Intercept]])         # included exogenous variables (i.e. X\[price])

    # Read IV data (from dta)
    df_Z = Float64.(DataFrame(load("Car_demand_iv_spec1.dta"))) # load data from dta file
    sorted_df_Z = sort(df_Z, :Year)                             # sort by year
    Z = hcat(Z₁, Matrix(sorted_df_Z[:,[:i_import, :diffiv_local_0, :diffiv_local_1, :diffiv_local_2, :diffiv_local_3, :diffiv_ed_0]])) # all IVs (included & excluded)
    #Z₂ = hcat(Z₁, Matrix(sorted_df_Z[:,[:i_import, :blpiv_0, :blpiv_1, :blpiv_2, :blpiv_3, :blpiv_4, :blpiv_5]]))  # all IVs (included & excluded)

    # Read simulated income (from dta)
    df_Y = DataFrame(load("Simulated_type_distribution.dta")) # load data from dta file
    y_sim = Vector(Float64.(df_Y)[:,1])

end

# data initialization
function get_data()
    data = Cardata()
    return data
end

########################################################################
# simulate idiosyncratic component of the random utility (μ_ijt)
function μ_sim(λ::Float64, P::Vector{Float64}, y::Float64)
    m = λ*P*y   # use the definition of μ
    return m
end

# simulate individual choice probability (σ_ijt)
function choice_prob(D::Vector{Float64}, M::Vector{Float64})
    if length(D) == length(M)
        denom = 1 + sum(exp.(D) .* exp.(M)) # denominator of choice probability
        s = (exp.(D) .* exp.(M))/denom      # follow the formula for σ
        return s
    else
        println("The dimensions of δ and μ are different!!") # dimension mismatch indicator
    end    
end

# simulate market share (= average choice probability)
function share_simulator(D::Vector{Float64}, λ::Float64, P::Vector{Float64}, prim::Primitives, data::Cardata)
    @unpack y_sim = data

    share::Vector{Float64} = zeros(length(D))       # container for market share
    C::Matrix{Float64} = zeros(length(D), prim.R)   # container for individual choice probs

    for (i, y) in enumerate(y_sim)
        m = μ_sim(λ, P, y_sim[i])   # simulate μ(Jₜ×1 vector) of individual i
        c = choice_prob(D, m)       # simulate σ(Jₜ×1 vector) of individual i
        C[:,i] = c                  # store individual choice prob in memory
        share += c                  # sum up the individual choice prob
    end

    s = share/prim.R    # predictes share (or demand) vector (in market t)
    return s, C         # return simulated market share vector & individual choice probs matrix
end

# invert model-predicted market share with contraction mapping method
function σ_invert_contraction(D::Vector{Float64}, λ::Float64, P::Vector{Float64}, S::Vector{Float64}, prim::Primitives, data::Cardata)
    δ_next =  D + (log.(S) - log.((share_simulator(D, λ, P, prim, data)[1]))) # follow the contraction in BLP(1995) and get next guess of δ
    return δ_next
end

# get Jacobian from choice probability Matrix
function Jacob_σ(Σ::Matrix{Float64} ,prim::Primitives)
    Jₜ = size(Σ, 1) # get the number of product (=Jₜ, depends on market)

    own = (1.0*I(Jₜ)) .* (Σ * (1 .- Σ)')    # own derivative part: I*σ(1-σ)'
    cross = (1.0 .- 1.0*I(Jₜ)) .* (Σ * Σ')  # cross derivative part: (1-I)*σσ'
    return (1/prim.R)*(own - cross)         # return Jacobian metrix (Jₜ×Jₜ) 
end

# invert model-predicted market share with Newton method
function σ_invert_Newton(D::Vector{Float64}, λ::Float64, P::Vector{Float64}, S::Vector{Float64}, prim::Primitives, data::Cardata)
    σ,Σ = share_simulator(D, λ, P, prim, data)          # simulate choice probs
    Df = Jacob_σ(Σ, prim)                               # calculate Jacobian
    δ_next =  D - (inv(-Df ./ σ))*(log.(S) - log.(σ))   # get next guess of δ by Newton
    return δ_next, -Df ./ σ                             # return next guess and Df/σ matrix
end

# contraction only
function δ_iterate_contraction(λ::Float64, P::Vector{Float64}, S::Vector{Float64}, prim::Primitives, data::Cardata; tol::Float64 = 1e-12, max_iter::Int64 = 1000)
    n = 0                          # iteration counter
    err = 100                      # initial error (some big number)
    err_norm::Vector{Float64} = [] # vector for containing the evolution of error norm

    D::Vector{Float64} = zeros(length(P))   # initial guess of δ's
    while err > tol && n < max_iter 
        δ_next = σ_invert_contraction(D, λ, P, S, prim, data) # spit out new guess of δ

        err = maximum(abs.(δ_next .- D))    # reset error level
        err_norm = vcat(err_norm, err)      # add the error level to the vector
        D = δ_next                          # update δ(≡ D) in memory
        n+=1
    end

    println("Final error = ", err)
    println("δ converged in ", n, " iterations.")
    return D, err_norm # return simulated δ and evolution of norm
end

# mix of contraction and Newton
function δ_iterate_mix(λ::Float64, P::Vector{Float64}, S::Vector{Float64}, prim::Primitives, data::Cardata; tol₁::Float64 = 1.0, tol::Float64 = 1e-12, max_iter::Int64 = 1000)
    n = 0                          # iteration counter
    err = 100                      # initial error (some big number)
    err_norm::Vector{Float64} = [] # vector for containing the evolution of error norm

    D::Vector{Float64} = zeros(length(P))   # initial guess of δ's
    while err > tol && n < max_iter
        if err > tol₁ # first we use contraction
            δ_next = σ_invert_contraction(D, λ, P, S, prim, data) # get new guess of δ with contraction
            err = maximum(abs.(δ_next.-D))                        # reset error level
        else  # move to Newton after error is small enough
            δ, Df = σ_invert_Newton(D, λ, P, S, prim, data)       # get new guess of δ with newton
            δ_next = δ                                            # update next guess of δ
            err = maximum(abs.(Df*(δ_next.-D)))                   # reset error level
        end
        err_norm = vcat(err_norm, err)  # add the error level to the vector
        D = δ_next                      # update δ in memory
        n+=1
    end

    println("Final error = ", err)
    println("δ converged in ", n, " iterations.")
    return D, err_norm # return simulated δ and evolution of norm
end

########################################################################
# simulate δ's in all brands & markets
function δ_simulator_all(λ::Float64, prim::Primitives, data::Cardata)
    δ_all::Vector{Float64} = [] # container for simulated δs

    # simulate δ in all markets
    for (t_index,t) in enumerate(prim.T)
        price = data.df_X[data.df_X.Year .== t, :price] # get price vector in market t
        share = data.df_X[data.df_X.Year .== t, :share] # get share vector in market t       

        δ_all = vcat(δ_all, δ_iterate_mix(λ, price, share, prim, data)[1]) # add δ of market t
    end

    return δ_all
end

# estimate linear parameters with IV
function iv_estimator(res::Results, data::Cardata)
    @unpack D, W = res
    @unpack X, Z = data

    X_hat = Z*W*Z'*X                  # get X_hat
    return inv((X_hat)'*X)*(X_hat)'*D # return IV estimates
end

# GMM objective function
function GMM_obj(λ::Float64, prim::Primitives, res::Results, data::Cardata)
    @unpack X, Z = data

    res.D = δ_simulator_all(λ, prim, data)  # get (simulated) δ
    β = iv_estimator(res,data)              # get IV estimates
    ρ = res.D - X*β                         # compute ρ (i.e. residuals)
    return ρ'*Z*res.W*Z'*ρ                  # return GMM objective function value
end

# grid search & plot GMM objective function
function GMM_plot(grid::Vector{Float64}, prim::Primitives, res::Results, data::Cardata, tsls::Bool)
    G_val::Vector{Float64} = zeros(prim.nλ) # container for the values of GMM objective function

    # 2sls weight indicator (if not, use W stored in struct)
    if tsls == true
        res.W = inv(data.Z'*data.Z)
    end

    for (λ_index,λ) in enumerate(grid)
        G_val[λ_index] = GMM_obj(λ,prim,res,data) # get the value of GMM(λ) and store it in memory
    end

    return G_val
end

########################################################################
# solve GMM (a function to apply Optim.optimize package)
function G(λ)
    return GMM_obj(λ[1],prim,res,data)
end

########################################################################


