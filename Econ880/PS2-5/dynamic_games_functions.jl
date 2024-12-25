using Parameters, LinearAlgebra, Random, Distributions, Optim, Statistics, DataFrames, StatFiles, CSVFiles 

# declear primitives struct
@with_kw struct Primitives
    β::Float64 = 1/1.05 # discount factor
    α::Float64 = 0.06   # investment success parameter
    δ::Float64 = 0.1    # capital depreciation parameter
    a::Float64 = 40.0   # demand intercept parameter
    b::Float64 = 10.0   # demand slope parameter

    # capacity grid
    nq::Int64 = 10
    q_grid::Vector{Float64} = [0.0,5.0,10.0,15.0,20.0,25.0,30.0,35.0,40.0,45.0]

    #
    B::Int64 = 1000
    T::Int64 = 25
    seed::Int64 = 456
end

# declear results struct
@with_kw mutable struct Results
    x::Matrix{Float64} # strategy x(ω) 
    V::Matrix{Float64} # value function V(ω)

    Q::Matrix{Float64} # industry transition matrix Q(ω',ω)
    Ω::Vector{Int64} # 
end

# declear data struct
@with_kw struct csvdata
    df::DataFrame
    Π::Matrix{Float64} # profit matrix(10*10)
end

#initialization function at everywhere
@everywhere function Initialize()
    prim = Primitives()

    x = zeros(prim.nq,prim.nq)
    V = zeros(prim.nq,prim.nq)
    Q = zeros(prim.nq*prim.nq,prim.nq*prim.nq)
    Ω = zeros(Int64,prim.B)
    res = Results(x,V,Q,Ω)

    df = DataFrame(load("cournotProfits.csv")) # load profit data from csv file
    # create profit matrix
    Π = zeros(prim.nq,prim.nq)
    for (j,q₁) in enumerate(prim.q_grid)
        for (k,q₂) in enumerate(prim.q_grid)
            ω = findall(df.capacityFirm1 .== q₁ .&& df.capacityFirm2 .== q₂)
            Π[j,k] = df.profitFirm1[ω[1]]
        end
    end
    data = csvdata(df, Π)

    return prim, res, data
end

#######################################################################
# a function to get capacity evolution probability (given x and q)
# (Note: This function returns a (10*1) vector)
function capacity_evolution(q_index::Int64, x::Float64, prim::Primitives)
    @unpack α, δ = prim
    Δq ::Vector{Float64} = zeros(prim.nq) # container for capacity evolution probability

    if q_index == 1           # lowest capacity level: no capacity decrease
        Δq[q_index+1] = (α*x)/(1 + α*x)
        Δq[q_index] = 1/(1 + α*x)

    elseif q_index == prim.nq # highest capacity level: no capacity increase
        Δq[q_index] = (1-δ)/(1 + α*x) + (α*x)/(1 + α*x)
        Δq[q_index-1] = δ/(1 + α*x)

    else # the other capacity levels: all transitions are likely
        Δq[q_index+1] = ((1-δ)*α*x)/(1 + α*x)
        Δq[q_index] = (1-δ)/(1 + α*x) + (δ*α*x)/(1 + α*x)
        Δq[q_index-1] = δ/(1 + α*x)    
    end

    return Δq # return capacity evolution probability vector (10*1)
end

# a function to get continuation value (given ω)
function get_W(i::Int64, j::Int64, prim::Primitives, res::Results)
    @unpack x, V = res

    v = V[i,:]
    p = capacity_evolution(j,x[j,i],prim)

    return sum(v .* p)
end

# a function to get BR(ω)
function get_BR(i::Int64, j::Int64, prim::Primitives, res::Results)
    @unpack α, β, δ, nq = prim

    W₀ = get_W(i,j,prim,res) # continuation value

    if i == 1
        W₋ = get_W(i,j,prim,res)
    else
        W₋ = get_W(i-1,j,prim,res)
    end

    if i == nq
        W₊ = get_W(i,j,prim,res)
    else
        W₊ = get_W(i+1,j,prim,res)
    end

    sol = (sqrt(β*α*(((1-δ)*(W₊ - W₀)) + (δ*(W₀ - W₋))))-1)/α

    return max(0.0, sol)
end

# a function to get E[W(ω)] in value function iteration
function get_EW(i::Int64, j::Int64, x::Float64, prim::Primitives, res::Results)
    @unpack α, β, δ, nq = prim

    W_next::Vector{Float64} = zeros(nq)
    W_next[i] = get_W(i,j,prim,res) # continuation value
    if i ≠ 1
        W_next[i-1] = get_W(i-1,j,prim,res)
    end
    if i ≠ nq
        W_next[i+1] = get_W(i+1,j,prim,res)
    end

    p = capacity_evolution(i,x,prim)

    return sum(W_next .* p)
end

# Bellman operator
function Bellman(prim::Primitives, res::Results, data::csvdata)
    @unpack α, β, δ, nq, q_grid = prim
    @unpack Π = data

    x_next::Matrix{Float64} = zeros(nq, nq) # container for next guess of x(ω)
    V_next::Matrix{Float64} = zeros(nq, nq) # container for next guess of V(ω)

    # compute next guess of x and V
    for i in 1:nq
        for j in 1:nq
            inves = get_BR(i,j,prim,res)    # get R(x|ω) using closed form solution
            x_next[i,j] = inves             # store x(ω)
            V_next[i,j] = Π[i,j] - inves + β*(get_EW(i,j,inves,prim,res)) # compute V(ω) and store
        end
    end

    return x_next, V_next # return next guess of x(ω) and V(ω)
end 

# x(ω) and V(ω) iterator
function xV_iterate(prim::Primitives, res::Results, data::csvdata; tol_x::Float64 = 1e-13,  tol_V::Float64 = 1e-10, max_iter = 5000)
    n = 0       # iteration counter
    err_x = 100 # initial error for investment x: some big number
    err_V = 100 # initial error for value function V: some big number

    while (err_x > tol_x || err_V > tol_V) && n < max_iter
        x_next, V_next = Bellman(prim,res,data) # compute next guess of x and V
        err_x = maximum(abs.(x_next .- res.x))  # calculate sup(error) of x
        err_V = maximum(abs.(V_next .- res.V))  # calculate sup(error) of V
        res.x .= x_next                         # update x(ω) in struct
        res.V .= V_next                         # update V(ω) in struct
        n += 1                                  # iteration count
        println("Current error of x = ", err_x, " Current error of V = ", err_V)
    end
    
    #println("Finar error = ", err)
    println("x(ω) and V(ω) converged in ", n, " iterations.")
end

#######################################################################
# a function to get an industry transition matrix Q(ω',ω)(100×100)
function get_Q(prim::Primitives, res::Results)
    @unpack nq = prim
    @unpack x = res

    # loop for each state and compute transition
    for i in 1:nq
        for j in 1:nq
            state = nq*(i-1) + j # get state index (1-100)
            #println(state)
            pᵢ = capacity_evolution(i,x[i,j],prim) # get Pr(q₁'|q₁,x(ω))
            pⱼ = capacity_evolution(j,x[j,i],prim) # get Pr(q₂'|q₂,x(ω))

            # compute Q(ω',ω) for given ω = (i,j)
            Q_part::Vector{Float64} = []
            for s in 1:nq
                Q_part = vcat(Q_part,pᵢ[s]*pⱼ)
            end

            res.Q[state,:] = Q_part # update Q(ω',ω) in struct
        end
    end
end

#######################################################################
# a function to simulate market structure evolution
function market_evol_simulator(ω_init_index::Vector{Int64}, prim::Primitives, res::Results)
    @unpack nq, T, seed, B = prim
    @unpack Q = res

    Random.seed!(seed) # get seed for simulation

    # loop for B(=1,000) times
    for b in 1:B
        state = nq*(ω_init_index[1]-1) + ω_init_index[2] # get initial state
        D::Vector{Float64} = rand(T)                     # draw random number to get transition state

        # loop for T(=25) years
        for t in 1:T
            ϵ = D[t]                  # get random number in [0,1]
            F = cumsum(Q[state,:])    # a "CDF" for transition
            state = findfirst(F .> ϵ) # get next state
        end
        res.Ω[b] = state # record the state at T=25 in struct
    end
end

# a function to get industry state distribution
function get_industry_dist(prim::Primitives, res::Results, data::csvdata)
    @unpack Ω = res
    @unpack df = data

    ω_mat::Matrix{Int64} = zeros(Int64,prim.nq,prim.nq)
    ω_vec::Matrix{Int64} = zeros(prim.B,2)

    # get (i,j) from each state number stored in Ω
    for (b,state) in enumerate(Ω)
        i = Int(df.capacityFirm1[state]/5 + 1)
        j = Int(df.capacityFirm2[state]/5 + 1)
        ω_mat[i,j] += 1
        ω_vec[b,1] = i
        ω_vec[b,2] = j
    end

    return ω_mat, ω_vec
end
#######################################################################