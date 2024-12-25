#cd("/Users/yuichirotsuji/Documents/Econ715/PS2-1")
using Parameters, LinearAlgebra, Random, Distributions, Optim, Statistics, DataFrames, StatFiles, CSVFiles #import the libraries we want

###################################################################
# model primitives setting 
@with_kw mutable struct Primitives
    n::Int64 = 50742 # number of observations
    k::Int64 = 3     # dim(β)
    r::Int64 = 100   # sample size

    B::Int64 = 1000  # Bootstrap replication
    S::Int64 = 5000  # Sample draw

    dist = Normal(0,1)
    α₉₅::Float64 = quantile(dist,0.975) 
end

@with_kw mutable struct Results
    β₀::Vector{Float64} # "true" value of β

    # container for simulated CI for asymptotic CI
    asym_V₀_CI₋::Array{Float64}
    asym_V₀_CI₊::Array{Float64}
    asym_V₁_CI₋::Array{Float64}
    asym_V₁_CI₊::Array{Float64}
    
    # container for simulated CI for nonparametric bootstrap CI
    nonparametric_V₀_CI₋::Array{Float64}
    nonparametric_V₀_CI₊::Array{Float64}
    nonparametric_V₁_CI₋::Array{Float64}
    nonparametric_V₁_CI₊::Array{Float64}

    # container for simulated CI for residual bootstrap CI
    residual_V₀_CI₋::Array{Float64}
    residual_V₀_CI₊::Array{Float64}
    residual_V₁_CI₋::Array{Float64}
    residual_V₁_CI₊::Array{Float64}

    # container for simulated CI for parametric bootstrap CI
    parametric_V₀_CI₋::Array{Float64}
    parametric_V₀_CI₊::Array{Float64}
    parametric_V₁_CI₋::Array{Float64}
    parametric_V₁_CI₊::Array{Float64}

end

# Initialize parameters and result containers
function Initialize()
    prim = Primitives()

    β₀ = zeros(prim.k)
    asym_V₀_CI₋ = zeros(prim.k, prim.S)
    asym_V₀_CI₊ = zeros(prim.k, prim.S)
    asym_V₁_CI₋ = zeros(prim.k, prim.S)
    asym_V₁_CI₊ = zeros(prim.k, prim.S)
    nonparametric_V₀_CI₋ = zeros(prim.k, prim.S)
    nonparametric_V₀_CI₊ = zeros(prim.k, prim.S)
    nonparametric_V₁_CI₋ = zeros(prim.k, prim.S)
    nonparametric_V₁_CI₊ = zeros(prim.k, prim.S)
    residual_V₀_CI₋ = zeros(prim.k, prim.S)
    residual_V₀_CI₊ = zeros(prim.k, prim.S)
    residual_V₁_CI₋ = zeros(prim.k, prim.S)
    residual_V₁_CI₊ = zeros(prim.k, prim.S)
    parametric_V₀_CI₋ = zeros(prim.k, prim.S)
    parametric_V₀_CI₊ = zeros(prim.k, prim.S)
    parametric_V₁_CI₋ = zeros(prim.k, prim.S)
    parametric_V₁_CI₊ = zeros(prim.k, prim.S)
    res = Results(β₀, asym_V₀_CI₋, asym_V₀_CI₊, asym_V₁_CI₋, asym_V₁_CI₊, nonparametric_V₀_CI₋, nonparametric_V₀_CI₊, nonparametric_V₁_CI₋, nonparametric_V₁_CI₊, residual_V₀_CI₋, residual_V₀_CI₊, residual_V₁_CI₋, residual_V₁_CI₊, parametric_V₀_CI₋, parametric_V₀_CI₊, parametric_V₁_CI₋, parametric_V₁_CI₊)

    return prim, res
end


#　Data loading
@with_kw struct CPSdata
    # Read CPS dataset
    df = Float64.(DataFrame(load("cps09mar_clean.dta"))) # load data from dta file
    Y = Vector(df[:,:ln_earnings])                       # LHS variable (log(earnings))
    X = Matrix(df[:,[:constant, :ued, :uexp]])           # RHS variables  
    #X₂ = Matrix(df[:,[:constant, :education, :exp]])
end

# Data initialization
function get_data()
    data = CPSdata()
    return data
end


###################################################################
# Functions for Q2
###################################################################
# function for OLS calculation
function OLS(Y::Vector{Float64}, X::Matrix{Float64})
    return (X'*X)\X'*Y  # we use LU decomposition
end

# function for residual making
function ϵ_maker(Y::Vector{Float64}, X::Matrix{Float64})
    β = OLS(Y,X)
    return Y - X*β
end

# function for getting homoskedastic variance estimates
function get_V_hat(Y::Vector{Float64}, X::Matrix{Float64})
    ϵ = ϵ_maker(Y,X)
    n = size(X,1)
    k = size(X,2)
    s² = sum(ϵ.^2)/(n-k)

    return s²*inv(X'*X)
end

# function for getting heteroskedastic variance estimates
function get_V_tilde(Y::Vector{Float64}, X::Matrix{Float64})
    ϵ = ϵ_maker(Y,X) 
    return inv(X'*X)*(X'*(X .*(ϵ.^2)))*inv(X'*X)
end

# function to obtain CI using asymptotic approximation
function get_CI(Y::Vector{Float64}, X::Matrix{Float64}, V::Matrix{Float64}, prim::Primitives, show::Bool)
    @unpack α₉₅ = prim
    β = OLS(Y,X)
    β_lower::Vector{Float64} = zeros(prim.k)
    β_upper::Vector{Float64} = zeros(prim.k)

    for k in 1:length(β)
        SE = sqrt(V[k,k])
        β_lower[k] = β[k] - α₉₅*SE
        β_upper[k] = β[k] + α₉₅*SE
        if show == true
            println("The 95% CI for β[",k,"] is [", β_lower[k],", ",β_upper[k],"].")
        end
    end

    if show == false
        return β_lower, β_upper
    end    
end

###################################################################
# Functions for Q3
###################################################################
# function to calculate bootstrap T-statistics
function get_T_star(boot_Y::Vector{Float64}, boot_X::Matrix{Float64}, β::Vector{Float64} )
    β_star = OLS(boot_Y, boot_X)
    s₀_star = sqrt.(diag(get_V_hat(boot_Y, boot_X)))
    s₁_star = sqrt.(diag(get_V_tilde(boot_Y, boot_X)))

    T₀_star = (β_star - β) ./s₀_star
    T₁_star = (β_star - β) ./s₁_star

    return T₀_star, T₁_star
end

# function to get CI (after calculating T-statistics)
function get_boot_pCI(T₀_boot::Matrix{Float64}, T₁_boot::Matrix{Float64}, β::Vector{Float64}, s₀::Vector{Float64}, s₁::Vector{Float64}, prim::Primitives)
    β₀_lower::Vector{Float64} = zeros(prim.k)
    β₀_upper::Vector{Float64} = zeros(prim.k)
    β₁_lower::Vector{Float64} = zeros(prim.k)
    β₁_upper::Vector{Float64} = zeros(prim.k)

    for k_index in 1:prim.k
        q₀_star⁻ = sort(T₀_boot[k_index,:])[Int(floor(prim.B*0.025))]
        q₀_star⁺ = sort(T₀_boot[k_index,:])[Int(floor(prim.B*0.975))]
        β₀_lower[k_index] = β[k_index] - s₀[k_index]*q₀_star⁺
        β₀_upper[k_index] = β[k_index] - s₀[k_index]*q₀_star⁻

        q₁_star⁻ = sort(T₁_boot[k_index,:])[Int(floor(prim.B*0.025))]
        q₁_star⁺ = sort(T₁_boot[k_index,:])[Int(floor(prim.B*0.975))]
        β₁_lower[k_index] = β[k_index] - s₁[k_index]*q₁_star⁺
        β₁_upper[k_index] = β[k_index] - s₁[k_index]*q₁_star⁻

    end

    return β₀_lower, β₀_upper, β₁_lower, β₁_upper
end

# function to perform nonparametric bootstrap
function nonparametric_Bootstrap_CI(Y::Vector{Float64}, X::Matrix{Float64}, prim::Primitives, show::Bool)
    @unpack k, B = prim

    # get OLS result using sample
    β = OLS(Y,X)
    s₀ = sqrt.(diag(get_V_hat(Y, X)))
    s₁ = sqrt.(diag(get_V_tilde(Y, X)))

    # container for bootstrap t-statistics
    T₀_boot::Matrix{Float64} = zeros(k,B)
    T₁_boot::Matrix{Float64} = zeros(k,B)
    n = length(Y)

    # get bootstrap t-statistics
    for b in 1:B
        boot_sample_id = rand(1:n,n) # get bootstrap sample id
        boot_Y = Y[boot_sample_id]   # get bootstrap Y
        boot_X = X[boot_sample_id,:] # get bootstrap X

        T₀_boot[:,b], T₁_boot[:,b] = get_T_star(boot_Y, boot_X, β)
    end

    β₀_lower, β₀_upper, β₁_lower, β₁_upper = get_boot_pCI(T₀_boot, T₁_boot, β, s₀, s₁, prim)

    # show(Q2) or return(Q3) result
    if show == true
        println("Homoskedastic CI: β[const]=[", β₀_lower[1], ",", β₀_upper[1],"], β[ed]=[", β₀_lower[2],",", β₀_upper[2], "], β[exp]=[", β₀_lower[3],",", β₀_upper[3], "]")
        println("Heteroskedastic CI: β[const]=[", β₁_lower[1], ",", β₁_upper[1],"], β[ed]=[", β₁_lower[2],",", β₁_upper[2], "], β[exp]=[", β₁_lower[3],",", β₁_upper[3], "]")
    elseif show == false
        return β₀_lower, β₀_upper, β₁_lower, β₁_upper
    end
end

# function to perform residual bootstrap
function residual_Bootstrap_CI(Y::Vector{Float64}, X::Matrix{Float64}, prim::Primitives, show::Bool)
    @unpack k, B = prim

    # get OLS result using sample
    β = OLS(Y,X)
    ϵ = ϵ_maker(Y,X)
    s₀ = sqrt.(diag(get_V_hat(Y, X)))
    s₁ = sqrt.(diag(get_V_tilde(Y, X)))

    # containers for bootstrap t-statistics
    T₀_boot::Matrix{Float64} = zeros(k,B)
    T₁_boot::Matrix{Float64} = zeros(k,B)
    n = length(Y)
    
    # get bootstrap t-statistics
    for b in 1:B
        boot_sample_id = rand(1:n,n) # get bootstrap sample id
        boot_ϵ = ϵ[boot_sample_id]   # get bootstrap residual
        boot_Y = X*β + boot_ϵ        # make bootstrap Y
        #boot_X = X

        T₀_boot[:,b], T₁_boot[:,b] = get_T_star(boot_Y, X, β)
    end

    β₀_lower, β₀_upper, β₁_lower, β₁_upper = get_boot_pCI(T₀_boot, T₁_boot, β, s₀, s₁, prim)

    # show(Q2) or return(Q3) result
     if show == true
        println("Homoskedastic CI: β[const]=[", β₀_lower[1], ",", β₀_upper[1],"], β[ed]=[", β₀_lower[2],",", β₀_upper[2], "], β[exp]=[", β₀_lower[3],",", β₀_upper[3], "]")
        println("Heteroskedastic CI: β[const]=[", β₁_lower[1], ",", β₁_upper[1],"], β[ed]=[", β₁_lower[2],",", β₁_upper[2], "], β[exp]=[", β₁_lower[3],",", β₁_upper[3], "]")
    elseif show == false
        return β₀_lower, β₀_upper, β₁_lower, β₁_upper
    end

end

# function to perform parametric bootstrap
function parametric_Bootstrap_CI(Y::Vector{Float64}, X::Matrix{Float64}, prim::Primitives, show::Bool)
    @unpack k, B = prim

    # get OLS result using sample
    β = OLS(Y,X)
    ϵ = ϵ_maker(Y,X)
    s² = sum(ϵ.^2)/(size(X,1)-size(X,2))
    dist = Normal(0,s²)
    s₀ = sqrt.(diag(get_V_hat(Y, X)))
    s₁ = sqrt.(diag(get_V_tilde(Y, X)))

    # containers for bootstrap t-statistics
    T₀_boot::Matrix{Float64} = zeros(k,B)
    T₁_boot::Matrix{Float64} = zeros(k,B)
    n = length(Y)
       
    # get bootstrap t-statistics
     for b in 1:B
        boot_sample_id = rand(1:n,n)
        boot_ϵ = rand(dist,size(X,1))
        boot_Y = X*β + boot_ϵ
        #boot_X = X

        T₀_boot[:,b], T₁_boot[:,b] = get_T_star(boot_Y, X, β)
    end

    # show(Q2) or return(Q3) result
    β₀_lower, β₀_upper, β₁_lower, β₁_upper = get_boot_pCI(T₀_boot, T₁_boot, β, s₀, s₁, prim)
    if show == true
        println("Homoskedastic CI: β[const]=[", β₀_lower[1], ",", β₀_upper[1],"], β[ed]=[", β₀_lower[2],",", β₀_upper[2], "], β[exp]=[", β₀_lower[3],",", β₀_upper[3], "]")
        println("Heteroskedastic CI: β[const]=[", β₁_lower[1], ",", β₁_upper[1],"], β[ed]=[", β₁_lower[2],",", β₁_upper[2], "], β[exp]=[", β₁_lower[3],",", β₁_upper[3], "]")
    elseif show == false
        return β₀_lower, β₀_upper, β₁_lower, β₁_upper
    end

end

###################################################################
# Functions for Q4
###################################################################
# function to simulate 8 different CIs
function CI_simulator(prim::Primitives, res::Results, data::CPSdata)
    @unpack n, r, S = prim

    for s in 1:S
        sample_id = randperm(n)[1:r]
        sample_Y = data.Y[sample_id]
        sample_X = data.X[sample_id, :]

        # asymptotic approximation CI
        V₀ = get_V_hat(sample_Y, sample_X)
        V₁ = get_V_tilde(sample_Y, sample_X)
        res.asym_V₀_CI₋[:,s], res.asym_V₀_CI₊[:,s] = get_CI(sample_Y, sample_X, V₀, prim, false)
        res.asym_V₁_CI₋[:,s], res.asym_V₁_CI₊[:,s] = get_CI(sample_Y, sample_X, V₁, prim, false)

        # nonparametric bootstrap CI
        res.nonparametric_V₀_CI₋[:,s], res.nonparametric_V₀_CI₊[:,s],res.nonparametric_V₁_CI₋[:,s], res.nonparametric_V₁_CI₊[:,s] = nonparametric_Bootstrap_CI(sample_Y, sample_X, prim, false)
        # residual bootstrap CI
        res.residual_V₀_CI₋[:,s], res.residual_V₀_CI₊[:,s],res.residual_V₁_CI₋[:,s], res.residual_V₁_CI₊[:,s] = residual_Bootstrap_CI(sample_Y, sample_X, prim, false)
        # parametric bootstrap CI
        res.parametric_V₀_CI₋[:,s], res.parametric_V₀_CI₊[:,s],res.parametric_V₁_CI₋[:,s], res.parametric_V₁_CI₊[:,s] = parametric_Bootstrap_CI(sample_Y, sample_X, prim, false)

        # code to know the progress (not necessary for actual calculation...)
        if s == Int(S/10)
            println("10% complete")
        elseif s == Int(S/2)
            println("50% complete")
        end    
    end
end

# function to calculate coverage probability
function get_coverage(CI₋::Array{Float64},CI₊::Array{Float64},prim::Primitives, res::Results)
    @unpack n, k, r, S = prim
    @unpack β₀ = res

    coverage_count::Vector{Float64} = zeros(k)

    # check wether each CI contains true β
    for s in 1:S
        for k_index in 1:k
            if β₀[k_index] > CI₋[k_index,s] && β₀[k_index] < CI₊[k_index,s]
                coverage_count[k_index] += 1.0
            end
        end
    end
    return coverage_count/S
end

# function to perform all comparison step
function coverage_comparison(prim::Primitives, res::Results, data::CPSdata)
    res.β₀ = OLS(data.Y,data.X) # get "ture" β
    CI_simulator(prim,res,data) # simulate CI

    # calculate coverage percentage for all CIs
    p₀ = get_coverage(res.asym_V₀_CI₋, res.asym_V₀_CI₊, prim, res)
    p₁ = get_coverage(res.asym_V₁_CI₋, res.asym_V₁_CI₊, prim, res)
    p₀_np = get_coverage(res.nonparametric_V₀_CI₋, res.nonparametric_V₀_CI₊, prim, res)
    p₁_np = get_coverage(res.nonparametric_V₁_CI₋, res.nonparametric_V₁_CI₊, prim, res)
    p₀_r = get_coverage(res.residual_V₀_CI₋, res.residual_V₀_CI₊, prim, res)
    p₁_r = get_coverage(res.residual_V₁_CI₋, res.residual_V₁_CI₊, prim, res)
    p₀_p = get_coverage(res.parametric_V₀_CI₋, res.parametric_V₀_CI₊, prim, res)
    p₁_p = get_coverage(res.parametric_V₁_CI₋, res.parametric_V₁_CI₊, prim, res)

    # display results
    println("************* Coverage percentages *************")
    println("Homoskedastic & asymptotid: ", p₀)
    println("Heteroskedastic & asymptotid: ", p₁)
    println("Homoskedastic & nonparametric bootstrap: ", p₀_np)
    println("Heteroskedastic & nonparametric bootstrap: ", p₁_np)
    println("Homoskedastic & residual bootstrap: ", p₀_r)
    println("Heteroskedastic & residual bootstrap: ", p₁_r)
    println("Homoskedastic & parametric bootstrap: ", p₀_p)
    println("Heteroskedastic & parametric bootstrap: ", p₁_p)
    println("************************************************")


end

###################################################################


