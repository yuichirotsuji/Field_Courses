# struct for primiteves
@with_kw struct Primitives
    N::Int64 = 16355   # number of obs in dataset
    nT::Int64 = 4      # number of outcomes

    R::Int64 = 100     # number of simulation draws
    seed::Int64 = 1234 # seed for simulation draw
end

# struct for simulation results
@with_kw mutable struct Results
    E::Array{Float64,3} # number of simulation draws
    P::Matrix{Float64}  # simulated choice probabilities
end

# a function to initialize primiteves and results
function Initialize()
    prim = Primitives()

    E = zeros(prim.R, 3, prim.N)
    P = zeros(prim.N, prim.nT)
    res= Results(E, P)

    return prim, res
end

#　struct-initalization (may not be used)
@with_kw struct Mortgagedata
    # Read mortgage data from dta file
    df = DataFrame(load("Mortgage_performance_data.dta")) # load data from dta file
    T = Float64.(df.duration)                   # create T
    Xr = select(df, [:score_0, :rate_spread, :i_large_loan, :i_medium_loan, :i_refinance, :age_r, :cltv, :dti, :cu, :first_mort_r,  :i_FHA, :i_open_year2, :i_open_year3, :i_open_year4, :i_open_year5]) #create x
    X = identity.(Array(Xr))
    Z₁ = Float64.(df.score_0) # create Z₁
    Z₂ = Float64.(df.score_1) # create Z₂
    Z₃ = Float64.(df.score_2) # create Z₃

    # import KPU sparce grid
    KPU_D1 = DataFrame(load("KPU_d1_l20.csv"))
    KPU_D2 = DataFrame(load("KPU_d2_l20.csv"))
    U₁ = Float64.(KPU_D1.x1)
    W₁ = Float64.(KPU_D1.weight)
    U₂ = Matrix(KPU_D2[:,1:2])
    W₂ = Float64.(KPU_D2.weight)
end

function get_data()
    data = Mortgagedata()
    return data
end

########################################################################
# Q1: Log-liklihood evaluation with quadrature
########################################################################
# log-transformation functions
function R_lb(u::Float64, a::Float64) # for lower bound (a,+∞)
    return -log(1-u) + a
end

# compute Prob(T=1|X,Z)
function prob_t1(α₁::Float64, β::Vector{Float64}, γ::Float64, ρ::Float64, X::Vector{Float64}, Z₁::Float64)
    σ₁ = 1/(1-ρ)               # variance of ϵ_i1 (determined by ρ)
    b = (-α₁-X'*β-Z₁*γ)/σ₁    # create thredhold value

    dist = Normal(0,1)         # get standard normal distribution
    choice_prob = cdf(dist, b) # Prob(T=1|X,Z) = Φ(b)
    return choice_prob
end

# compute Prob(T=2|X,Z)
function prob_t2(α₁::Float64, α₂::Float64, β::Vector{Float64}, γ::Float64, ρ::Float64, X::Vector{Float64}, Z₁::Float64, Z₂::Float64, U::Vector{Float64}, W::Vector{Float64})
    σ₁ = 1/(1-ρ)        # variance of ϵ_i1 (determined by ρ)
    a₁ = -α₁-X'*β-Z₁*γ  # lower range of integration = (a,+∞)
    a₂ = -α₂-X'*β-Z₂*γ  # part of the inner component of Φ(⋅)

    dist = Normal(0,1)                                 # get standard normal distribution
    Φ = [cdf(dist, a₂-ρ*R_lb(u,a₁)) for u in U]        # vector of function Φ(⋅)
    density = [pdf(dist, R_lb(u,a₁)/σ₁)/σ₁ for u in U] # vector of density

    J = 1 ./ (1 .- U)   # vector of Jacobian: ∂R/∂u = 1/1-u

    choice_prob = sum(Φ .* density .* J .* W) # compute choice probability
    return choice_prob
end

# compute Prob(T=3|X,Z)
function prob_t3(α₁::Float64, α₂::Float64, α₃::Float64, β::Vector{Float64}, γ::Float64, ρ::Float64, X::Vector{Float64}, Z₁::Float64, Z₂::Float64, Z₃::Float64, U::Matrix{Float64}, W::Vector{Float64})
    σ₁ = 1/(1-ρ)        # variance of ϵ_i1 (determined by ρ)
    a₁ = -α₁-X'*β-Z₁*γ  # lower range of integration of ϵ₁
    a₂ = -α₂-X'*β-Z₂*γ  # lower range of integration of ϵ₂
    a₃ = -α₃-X'*β-Z₃*γ  # part of the inner component of Φ(⋅)

    dist = Normal(0,1)                                 # get standard normal distribution
    Φ = [cdf(dist, a₃-ρ*R_lb(u,a₂)) for u in U[:,2]]   # vector of function Φ(⋅)

    dens_val = [R_lb(u₂,a₂) for u₂ in U[:,2]] - ρ*[R_lb(u₁,a₁) for u₁ in U[:,1]]             # values to evaluate first part of density
    density = [pdf(dist,d) for d in dens_val].*[pdf(dist, R_lb(u,a₁)/σ₁)/σ₁ for u in U[:,1]] # vector of density

    J₁ = 1 ./ (1 .- U[:,1])   # vector of Jacobian: ∂R₁/∂u₁ = 1/1-u₁
    J₂ = 1 ./ (1 .- U[:,2])   # vector of Jacobian: ∂R₂/∂u₂ = 1/1-u₂

    choice_prob = sum(Φ .* density .* J₁ .* J₂ .* W) # compute choice probability
    return choice_prob
end

# compute Prob(T=4|X,Z)
function prob_t4(α₁::Float64, α₂::Float64, α₃::Float64, β::Vector{Float64}, γ::Float64, ρ::Float64, X::Vector{Float64}, Z₁::Float64, Z₂::Float64, Z₃::Float64, U::Matrix{Float64}, W::Vector{Float64})
    σ₁ = 1/(1-ρ)        # variance of ϵ_i1 (determined by ρ)
    a₁ = -α₁-X'*β-Z₁*γ  # lower range of integration of ϵ₁
    a₂ = -α₂-X'*β-Z₂*γ  # lower range of integration of ϵ₂
    a₃ = -α₃-X'*β-Z₃*γ  # part of the inner component of Φ(⋅)

    dist = Normal(0,1)                                 # get standard normal distribution
    Φ = 1 .- [cdf(dist, a₃-ρ*R_lb(u,a₂)) for u in U[:,2]]   # vector of function Φ(⋅)

    dens_val = [R_lb(u₂,a₂) for u₂ in U[:,2]] - ρ*[R_lb(u₁,a₁) for u₁ in U[:,1]]             # values to evaluate first part of density
    density = [pdf(dist,d) for d in dens_val].*[pdf(dist, R_lb(u,a₁)/σ₁)/σ₁ for u in U[:,1]] # vector of density

    J₁ = 1 ./ (1 .- U[:,1])   # vector of Jacobian: ∂R₁/∂u₁ = 1/1-u₁
    J₂ = 1 ./ (1 .- U[:,2])   # vector of Jacobian: ∂R₂/∂u₂ = 1/1-u₂

    choice_prob = sum(Φ .* density .* J₁ .* J₂ .* W) # compute choice probability
    return choice_prob
end

# compute log-liklihood
function log_liklihood_quad(α₁::Float64, α₂::Float64, α₃::Float64, β::Vector{Float64}, γ::Float64, ρ::Float64, data::Mortgagedata)
    @unpack T, X, Z₁,Z₂,Z₃, U₁,W₁,U₂,W₂ = data
    L = 0 # log-liklihood is scalar-valued
    for (id,t) in enumerate(T) # We get each indiviudal's liklihood and take sum
        if t == 1.0      # people who chose T = 1
            L += log(prob_t1(α₁, β, γ, ρ, X[id,:], Z₁[id]))
        elseif t == 2.0
            L += log(prob_t2(α₁,α₂, β, γ, ρ, X[id,:], Z₁[id], Z₂[id], U₁, W₁))
        elseif t == 3.0
            L += log(prob_t3(α₁,α₂,α₃, β, γ, ρ, X[id,:], Z₁[id], Z₂[id], Z₃[id], U₂, W₂))
        elseif t == 4.0
            L += log(prob_t4(α₁,α₂,α₃, β, γ, ρ, X[id,:], Z₁[id], Z₂[id], Z₃[id], U₂, W₂))
        end
    end
    return L
end

########################################################################
# Q2. Simulated MLE
########################################################################
# a function to take simulation draws
function drawshocks(prim::Primitives)
    @unpack N, R, seed = prim
    Random.seed!(seed)
    dist = Normal(0,1)
    η = rand(dist, R, 3, N)
    return η
end

# a function to compute simulated individual choice probabilities
function prob_t_sim(α₁::Float64, α₂::Float64, α₃::Float64, β::Vector{Float64}, γ::Float64, ρ::Float64, X::Vector{Float64}, Z₁::Float64,Z₂::Float64,Z₃::Float64, prim::Primitives, e::Matrix{Float64})

    C::Vector{Int64} = zeros(prim.R)
    a₁ = -α₁-X'*β-Z₁*γ  # threshold of T=1/T=2
    a₂ = -α₂-X'*β-Z₂*γ  # threshold of T=2/T=3
    a₃ = -α₃-X'*β-Z₃*γ  # threshold of T=3/T=4

    for r in 1:prim.R
        ϵ₁ = (1/(1-ρ))*e[r,1] # simulate ϵ₁
        ϵ₂ = ρ*ϵ₁ + e[r,2]    # simulate ϵ₂
        ϵ₃ = ρ*ϵ₂ + e[r,3]    # simulate ϵ₃

        # get simulated choice
        if ϵ₁ < a₁ # T=1
            C[r] = 1
        elseif ϵ₁ > a₁ && ϵ₂ < a₂ # T=2
            C[r] = 2
        elseif ϵ₁ > a₁ && ϵ₂ > a₂ && ϵ₃ < a₃ # T=3
            C[r] = 3
        else # T=4
            C[r] = 4
        end
    end

    # get choice probability = (# of accept)/(# of draws)
    t₁ = count(c -> c == 1, C)/prim.R
    t₂ = count(c -> c == 2, C)/prim.R
    t₃ = count(c -> c == 3, C)/prim.R
    t₄ = count(c -> c == 4, C)/prim.R
    return [t₁ t₂ t₃ t₄]
end

# a function to get everyone's choice probabilities
function choice_prob_simulator(α₁::Float64, α₂::Float64, α₃::Float64, β::Vector{Float64}, γ::Float64, ρ::Float64, X::Matrix{Float64}, Z₁::Vector{Float64},Z₂::Vector{Float64},Z₃::Vector{Float64},prim::Primitives, res::Results)
    @unpack N = prim

    res.E = drawshocks(prim) # get simulation draw

    # apply individual choice probabilities simulator to everyone (and update res.P)
    for id in 1:N
        res.P[id,:] = prob_t_sim(α₁,α₂,α₃, β, γ, ρ, X[id,:], Z₁[id], Z₂[id], Z₃[id],prim, res.E[:,:,id])
    end

end

# compute (simulated) log-liklihood
function log_liklihood_sim(α₁::Float64, α₂::Float64, α₃::Float64, β::Vector{Float64}, γ::Float64, ρ::Float64, data::Mortgagedata,prim::Primitives, res::Results)
    @unpack_Mortgagedata data
    choice_prob_simulator(α₁,α₂,α₃, β, γ, ρ,X, Z₁,Z₂,Z₃,prim,res) # get choice probabilities

    L = 0 # log-liklihood is scalar-valued
    for (id,t) in enumerate(T) # We get each indiviudal's liklihood and take sum
        if t == 1.0      # people who chose T = 1
            L += log(res.P[id, 1])
        elseif t == 2.0  # people who chose T = 2
            if res.P[id, 2] == 0.0
                L += log(res.P[id, 2]+1e-10)
            else
                L += log(res.P[id, 2])
            end
        elseif t == 3.0  # people who chose T = 3
            if res.P[id, 3] == 0.0
                L += log(res.P[id, 2]+1e-10)
            else
                L += log(res.P[id, 3])
            end
        elseif t == 4.0  # people who chose T = 4
            L += log(res.P[id, 4])
        end
    end
    return L
end

########################################################################
# Q3: Choice probabilities
########################################################################
# Get choice probabilities using quadrature (as a (n×4) matrix)
function choice_prob_Quad(α₁::Float64, α₂::Float64, α₃::Float64, β::Vector{Float64}, γ::Float64, ρ::Float64, data::Mortgagedata)
    @unpack_Mortgagedata data
    P::Matrix{Float64} = zeros(length(T),4)
    for i in 1:length(T)
        P[i, 1] = prob_t1(α₁, β, γ, ρ, X[i,:], Z₁[i])
        P[i, 2] = prob_t2(α₁,α₂, β, γ, ρ, X[i,:], Z₁[i], Z₂[i], U₁, W₁)
        P[i, 3] = prob_t3(α₁,α₂,α₃, β, γ, ρ, X[i,:], Z₁[i], Z₂[i], Z₃[i], U₂, W₂)
        P[i, 4] = prob_t4(α₁,α₂,α₃, β, γ, ρ, X[i,:], Z₁[i], Z₂[i], Z₃[i], U₂, W₂)
    end
    return P
end

########################################################################
# Q4: Quadrature ML solver using BFGS
########################################################################
# functions for using BFGS with Optim.optimize package
function LLm(θ::Vector{Float64},data::Mortgagedata)
    return -log_liklihood_quad(θ[1], θ[2], θ[3], θ[6:end], θ[4], θ[5], data) #objective function = -(LL)
end
 
########################################################################
