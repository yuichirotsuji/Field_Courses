using Parameters, LinearAlgebra, Random, Interpolations, Optim

@with_kw struct Primitives
    β::Float64 = 0.99    #discount factor
    α::Float64 = 0.36    #cobb-douglas parameter
    δ::Float64 = 0.025   #capital depreciation rate
    ē::Float64 = 0.3271  #labor efficiency parameter

    ϵ_grid::Vector{Float64} = [1.0, 0.0] #grid points of employed/unemployed 
    nϵ::Int64 = length(ϵ_grid)           #number of employment states 
    M::Matrix{Float64} = [0.9624 0.0376; 0.5 0.5] #transition matrix of employment
    unemp::Float64 = M[1, 2] / (M[1, 2] + M[2, 2]) #SS unemployment rate
    L = ē * (1 - unemp) # Fixed in Aiyagari because no aggregate uncertainty & exogenous labor supply

    #grids for capital
    k_min::Float64 = 0.0001
    k_max::Float64 = 20.0
    nk::Int64 = 101
    k_grid::Vector{Float64} = range(k_min, stop=k_max, length=nk)

end

@with_kw mutable struct Results
    V::Array{Float64, 2}           # value function, dims (k, ϵ)
    k_policy::Array{Float64, 2}    # capital policy function, dims (k, ϵ)

    K::Float64                     # aggregate capital
    w::Float64                     # wage
    r::Float64                     # interest rate
    μ::Array{Float64, 2}           # distribution of agents
end

function Initialize()
    prim = Primitives()
    @unpack_Primitives prim

    V = zeros(nk, nϵ)
    k_policy = zeros(nk, nϵ)

    K = 12.0
    w = (1-α) * (K / L) ^ α
    r = α * (L / K) ^ (1-α)

    μ = ones(prim.nk, prim.nϵ) / (prim.nk * prim.nϵ)

    res = Results(V, k_policy, K, w, r, μ)
    return prim, res
end


function u(c::Float64; ε::Float64=1e-16)
    if c > ε
        return log(c)
    else
        return log(ε) + (c - ε) / ε
    end
end


function Bellman(prim::Primitives, res::Results)
    @unpack_Primitives prim
    @unpack_Results res

    V_next = zeros(nk, nϵ)
    k_next = zeros(nk, nϵ)

    #linear interpolation for employed value function
    interp1 = interpolate(V[:, 1], BSpline(Linear()))
    extrap1 = extrapolate(interp1, Line())              # gives linear extrapolation off grid
    V1_interp = scale(extrap1, range(k_min, k_max, nk)) # has to be scaled on increasing range object

    #linear interpolation for unemployed value function
    interp0 = interpolate(V[:, 2], BSpline(Linear()))
    extrap0 = extrapolate(interp0, Line())
    V0_interp = scale(extrap0, range(k_min, k_max, nk))

    #=
    # can also do bilinear interpolation and only interpolate once
    interp = interpolate(V, BSpline(Linear())) # baseline interpolation of value function
    extrap = extrapolate(interp, Line()) # gives linear extrapolation off grid
    V_interp = scale(extrap, range(k_min, k_max, nk), 1:nϵ) # has to be scaled on increasing range object
    =#

    for (ϵ_index, ϵ) in enumerate(ϵ_grid)
        p = M[ϵ_index, :]
        for (k_index, k) in enumerate(k_grid)
            budget = w * ϵ + (1 + r - δ) * k

            obj(k_prime) = -u(budget - k_prime) - β * (p[1] * V1_interp(k_prime) + p[2] * V0_interp(k_prime))
            res = optimize(obj, 0.0, budget)

            if res.converged
                V_next[k_index, ϵ_index] = -res.minimum
                k_next[k_index, ϵ_index] = res.minimizer
            else
                error("Optimization did not converge")
            end
        end
    end

    return V_next, k_next
end

function VFI(prim, res; tol=1e-10, max_iter=20_000)
    error = 100 * tol
    iter = 0

    while error > tol && iter < max_iter
        V_next, k_next = Bellman(prim, res)
        error = maximum(abs.(V_next - res.V))
        res.V = V_next
        res.k_policy = k_next
        iter += 1
    end

    if iter == max_iter
        println("Maximum iterations reached in VFI")
    elseif error < tol
        println("Converged in VFI after $iter iterations")
    end
end


function μ_iterate(prim, res)
    # slightly different than Huggett because policy functions off the grid

    @unpack_Primitives prim
    @unpack_Results res

    μ_next = zeros(nk, nϵ)

    for k_index in eachindex(k_grid)
        for ϵ_index in eachindex(ϵ_grid)
            k_prime = res.k_policy[k_index, ϵ_index]
            for ϵ_next in eachindex(ϵ_grid)
                if k_prime < k_min
                    μ_next[1, ϵ_next] += M[ϵ_index, ϵ_next] * μ[k_index, ϵ_index]
                elseif k_prime > k_max
                    μ_next[end, ϵ_next] += M[ϵ_index, ϵ_next] * μ[k_index, ϵ_index]
                else
                    # find 2 nearest grid points in k_grid to k_prime
                    index_high = searchsortedfirst(k_grid, k_prime)
                    k_high = k_grid[index_high]
                    index_low = index_high - 1
                    k_low = k_grid[index_low]
                    

                    # split the probability mass between the two points based on distance
                    weight_high = (k_prime - k_low) / (k_high - k_low)
                    weight_low = 1 - weight_high
                    μ_next[index_low, ϵ_next] += weight_low * M[ϵ_index, ϵ_next] * μ[k_index, ϵ_index]
                    μ_next[index_high, ϵ_next] += weight_high * M[ϵ_index, ϵ_next] * μ[k_index, ϵ_index]
                end
            end
        end
    end

    return μ_next
end


function SteadyDist(prim, res; tol=1e-10, max_iter=10_000)
    error = 100 * tol
    iter = 0

    while error > tol && iter < max_iter
        μ_next = μ_iterate(prim, res)
        error = maximum(abs.(μ_next - res.μ))
        res.μ = μ_next
        iter += 1
    end

    if iter == max_iter
        println("Maximum iterations reached in SteadyDist")
    elseif error < tol
        println("Converged in SteadyDist after $iter iterations")
    end
end


function SteadyStateCapital(prim, res)
    @unpack_Primitives prim
    function r_error(r)
        res.r = r
        res.K = (α * L ^ (1-α) / r) ^ (1 / (1-α))
        res.w = (1-α) * (res.K / L) ^ α
        VFI(prim, res)
        SteadyDist(prim, res)
        capital_supply = sum(res.μ .* res.k_policy)
        return (capital_supply - res.K)^2
    end

    opt = optimize(r_error, .01, .04)
    res.r = opt.minimizer
    res.K = (α * L ^ (1-α) / res.r) ^ (1 / (1-α))
    res.w = (1-α) * (res.K / L) ^ α
    VFI(prim, res)
    SteadyDist(prim, res)
end


function SolveModel()
    prim, res = Initialize()
    SteadyStateCapital(prim, res)
    return prim, res
end