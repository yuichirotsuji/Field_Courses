#=
 This file contains functions for solving the Krusell-Smith model.
 Most of the frameworks are based on "Aiyagari.jl" and "KS_skelton.jl". 
=#

using Parameters, LinearAlgebra, Random, Interpolations, Optim
#################################################################
# 1.1. Primitives struct
@with_kw struct Primitives
    β::Float64 = 0.99           # discount factor
    α::Float64 = 0.36           # capital share
    δ::Float64 = 0.025          # depreciation rate
    ē::Float64 = 0.3271         # labor productivity

    z_grid::Vector{Float64} = [1.01, .99]      # grid for TFP shocks
    z_g::Float64 = z_grid[1]
    z_b::Float64 = z_grid[2]
    nz::Int64 = length(z_grid)

    ϵ_grid::Vector{Float64} = [1, 0]           # grid for employment shocks
    nϵ::Int64 = length(ϵ_grid)

    nk::Int64 = 31
    k_min::Float64 = 0.001
    k_max::Float64 = 15.0
    k_grid::Vector{Float64} = range(k_min, stop=k_max, length=nk) # grid for capital, start coarse

    nK::Int64 = 17
    K_min::Float64 = 11.0
    K_max::Float64 = 15.0
    K_grid::Vector{Float64} = range(K_min, stop=K_max, length=nK) # grid for aggregate capital, start coarse

end

# 1.2. Results struct
@with_kw mutable struct Results
    Z::Vector{Float64}                      # aggregate shocks
    E::Matrix{Float64}                      # employment shocks

    V::Array{Float64, 4}                    # value function, dims (k, ϵ, K, z)
    k_policy::Array{Float64, 4}             # capital policy, similar to V

    a₀::Float64                             # constant for capital LOM, good times
    a₁::Float64                             # coefficient for capital LOM, good times
    b₀::Float64                             # constant for capital LOM, bad times
    b₁::Float64                             # coefficient for capital LOM, bad times
    R²::Float64                             # R² for capital LOM

    K_path::Vector{Float64}                 # path of capital

end

######################### Part 2 - generate shocks #########################
# 2.1. Shock struct
@with_kw struct Shocks
    #parameters of transition matrix:
    d_ug::Float64 = 1.5 # Unemp Duration (Good Times)
    u_g::Float64 = 0.04 # Fraction Unemp (Good Times)
    d_g::Float64 = 8.0  # Duration (Good Times)
    u_b::Float64 = 0.1  # Fraction Unemp (Bad Times)
    d_b::Float64 = 8.0  # Duration (Bad Times)
    d_ub::Float64 = 2.5 # Unemp Duration (Bad Times)

    #transition probabilities for aggregate states
    pgg::Float64 = (d_g-1.0)/d_g
    pgb::Float64 = 1.0 - (d_b-1.0)/d_b
    pbg::Float64 = 1.0 - (d_g-1.0)/d_g
    pbb::Float64 = (d_b-1.0)/d_b

    #transition probabilities for aggregate states and staying unemployed
    pgg00::Float64 = (d_ug-1.0)/d_ug
    pbb00::Float64 = (d_ub-1.0)/d_ub
    pbg00::Float64 = 1.25*pbb00
    pgb00::Float64 = 0.75*pgg00

    #transition probabilities for aggregate states and becoming employed
    pgg01::Float64 = (u_g - u_g*pgg00)/(1.0-u_g)
    pbb01::Float64 = (u_b - u_b*pbb00)/(1.0-u_b)
    pbg01::Float64 = (u_b - u_g*pbg00)/(1.0-u_g)
    pgb01::Float64 = (u_g - u_b*pgb00)/(1.0-u_b)

    #transition probabilities for aggregate states and becoming unemployed
    pgg10::Float64 = 1.0 - (d_ug-1.0)/d_ug
    pbb10::Float64 = 1.0 - (d_ub-1.0)/d_ub
    pbg10::Float64 = 1.0 - 1.25*pbb00
    pgb10::Float64 = 1.0 - 0.75*pgg00

    #transition probabilities for aggregate states and staying employed
    pgg11::Float64 = 1.0 - (u_g - u_g*pgg00)/(1.0-u_g)
    pbb11::Float64 = 1.0 - (u_b - u_b*pbb00)/(1.0-u_b)
    pbg11::Float64 = 1.0 - (u_b - u_g*pbg00)/(1.0-u_g)
    pgb11::Float64 = 1.0 - (u_g - u_b*pgb00)/(1.0-u_b)

    # Markov Transition Matrix
    Mgg::Array{Float64,2} = [pgg11 pgg01
                            pgg10 pgg00]

    Mbg::Array{Float64,2} = [pgb11 pgb01
                            pgb10 pgb00]

    Mgb::Array{Float64,2} = [pbg11 pbg01
                            pbg10 pbg00]

    Mbb ::Array{Float64,2} = [pbb11 pbb01
                             pbb10 pbb00]

    M::Array{Float64,2} = [pgg*Mgg pgb*Mgb
                          pbg*Mbg pbb*Mbb]

    # aggregate transition matrix
    Mzz::Array{Float64,2} = [pgg pbg
                            pgb pbb]
end

# 2.2. Simulation struct
@with_kw struct Simulations
    T::Int64 = 11_000           # number of periods to simulate
    N::Int64 = 5_000            # number of agents to simulate
    seed::Int64 = 1234          # seed for random number generator

    V_tol::Float64 = 1e-1       # tolerance for value function iteration
    V_max_iter::Int64 = 50       # maximum number of iterations for value function

    burn::Int64 = 1_000         # number of periods to burn for regression
    reg_tol::Float64 = 1e-5     # tolerance for regression coefficients
    reg_max_iter::Int64 = 5000 # maximum number of iterations for regression
    λ::Float64 = 0.5            # update parameter for regression coefficients

    K_initial::Float64 = 11.55   # initial aggregate capital (from Q(1))
end

# 2.3. Functions to generate shocks
# Function to get the next state given today's state
function sim_Markov(current_index::Int64, Π::Matrix{Float64})
    # Generate a random number between 0 and 1
    rand_num = rand() 
    # Get the cumulative sum of the probabilities in the current row
    cumulative_sum = cumsum(Π[current_index, :]) 
    # Find the next state index based on the random number
    next_index = searchsortedfirst(cumulative_sum, rand_num)

    return next_index
end

# Generate a sequence of aggregate shocks
function DrawShocks(prim::Primitives, sho::Shocks, sim::Simulations)
    @unpack z_grid, ϵ_grid = prim
    @unpack T, N, seed = sim
    @unpack M, Mzz = sho
    Random.seed!(seed)

    # Simulate a path of technology shocks
    Sz::Vector{Int64} = zeros(T) # Vector for aggregate state index (1(good) or 2(bad))
    Z::Vector{Float64} = zeros(T)  # Vector for aggregate productivity shock
    Sz[1] = 1               # Initial state = 1(good)
    Z[1] = z_grid[Sz[1]]    # Initial productivity = zg
    for t_index in 2:T
        Sz[t_index] = sim_Markov(Sz[t_index-1], Mzz)
        Z[t_index] = z_grid[Sz[t_index]]
    end

    # Simulate N = 5000 sequences of ϵ shocks of lengt T = 11000
    Se::Matrix{Int64} = zeros(N, T) # Vector for individual state index
    E::Matrix{Float64} = zeros(N, T)  # Vector for individual employment status
    for n_index in 1:N
        Se[n_index, 1] = 1                      # Initial state = good and employed
        E[n_index, 1] = ϵ_grid[Se[n_index, 1]]  # Initial employment status = 1
        for t_index in 2:T
            next_state = sim_Markov(Se[n_index, t_index-1], M)
            if next_state == 1 || next_state == 3
                E[n_index, t_index] = 1
                Se[n_index, t_index] = 2*Sz[t_index] - 1
            else
                E[n_index, t_index] = 0
                Se[n_index, t_index] = 2*Sz[t_index]
            end
        end
    end

    return Z, E
end

function Initialize()
    prim = Primitives()
    sho = Shocks()
    sim = Simulations()
    Z, E = DrawShocks(prim, sho, sim)

    V = zeros(prim.nk, prim.nϵ, prim.nK, prim.nz)
    k_policy = zeros(prim.nk, prim.nϵ, prim.nK, prim.nz)

    a₀ = 0.095
    a₁ = 0.999
    b₀ = 0.085
    b₁ = 0.999
    R² = 0.0

    K_path = zeros(sim.T)

    res = Results(Z, E, V, k_policy, a₀, a₁, b₀, b₁, R², K_path)
    return prim, sho, sim, res
end

######################### Part 3 - HH Problem #########################
# Define the utility function, with stitching function for numerical optimization
# (Same for skelton: No need to change)
function u(c::Float64; ε::Float64 = 1e-16)
    if c > ε
        return log(c)
    else # a linear approximation for stitching function
        # ensures smoothness for numerical optimization
        return log(ε) - (ε - c) / ε
    end
end

# Bellman operator for value function iteration
function Bellman(prim::Primitives, res::Results, sho::Shocks)
    @unpack_Primitives prim
    @unpack_Results res
    @unpack M, u_g, u_b = sho

    V_next = zeros(nk, nϵ, nK, nz) #next guess of value function
    k_next = zeros(nk, nϵ, nK, nz) #next guess of policy function

    #linear interpolation for employed & good state (i.e. state_index = 1)
    interp_g1 = interpolate(V[:,1,:,1], BSpline(Linear()))  # baseline interpolation of value function
    extrap_g1 = extrapolate(interp_g1, Line())              # gives linear extrapolation off grid
    Vg1_interp = scale(extrap_g1, range(k_min, k_max, nk), range(K_min, K_max, nK))

    #linear interpolation for unemployed & good state (i.e. state_index = 2)
    interp_g0 = interpolate(V[:,2,:,1], BSpline(Linear()))  # baseline interpolation of value function
    extrap_g0 = extrapolate(interp_g0, Line())              # gives linear extrapolation off grid
    Vg0_interp = scale(extrap_g0, range(k_min, k_max, nk), range(K_min, K_max, nK))

    #linear interpolation for employed & bad state (i.e. state_index = 3)
    interp_b1 = interpolate(V[:,1,:,2], BSpline(Linear()))  # baseline interpolation of value function
    extrap_b1 = extrapolate(interp_b1, Line())              # gives linear extrapolation off grid
    Vb1_interp = scale(extrap_b1, range(k_min, k_max, nk), range(K_min, K_max, nK))
    
    #linear interpolation for unemployed & bad state (i.e. state_index = 4)
    interp_b0 = interpolate(V[:,2,:,2], BSpline(Linear()))  # baseline interpolation of value function
    extrap_b0 = extrapolate(interp_b0, Line())              # gives linear extrapolation off grid
    Vb0_interp = scale(extrap_b0, range(k_min, k_max, nk), range(K_min, K_max, nK))

    # Iteration step for each state variables
    for (ϵ_index, ϵ) in enumerate(ϵ_grid)
        for (z_index, z) in enumerate(z_grid)
            # determine today's state (corresponding the row/column of transition matrix M)
            today = 1
            if  ϵ_index  == 2
                today += 1
            end
            if z_index == 2
                today += 2
            end
            p = M[today,:]      # transition probability given today's state

            # (exogenous) labor supply
            if z_index == 1
                L = ē*(1-u_g)
            elseif z_index == 2
                L = ē*(1-u_b)
            end

            for (K_index, K) in enumerate(K_grid)
                r = α*z*(K/L)^(α-1)     # model-implied interest rate today
                w = (1-α)*z*(K/L)^α     # model-implied wage today

                # create K' based on the guesses of first moment
                if z_index == 1
                    K_prime = exp(a₀ + a₁*log(K))
                else
                    K_prime = exp(b₀ + b₁*log(K))
                end

                # optimization over k'(choice variable for the HH)
                for (k_index, k) in enumerate(k_grid)
                    budget = w * ϵ + (1 + r - δ) * k # budget constraint
                    # objective function (to be minimized) is -(u(c)+E[V'])
                    obj(k_prime) = -u(budget - k_prime) - β * (p[1] * Vg1_interp(k_prime, K_prime) + p[2] * Vg0_interp(k_prime, K_prime) +p[3] * Vb1_interp(k_prime, K_prime) + p[4] * Vb0_interp(k_prime, K_prime))

                    res = optimize(obj, 0.0, budget) # optimize using Opt.optimize
                    V_next[k_index, ϵ_index, K_index, z_index] = -res.minimum  # V'(guess) = -(min(obj))
                    k_next[k_index, ϵ_index, K_index, z_index] = res.minimizer # k'(guess)        
                end
            end
        end
    end

    return V_next, k_next #next guess of value and policy functions
end

# Value Funtion Iteration (almost the same as in Aiyagari.jl)
function VFI(prim::Primitives, res::Results, sim::Simulations, sho::Shocks)
    @unpack_Simulations sim
    error = 100 * V_tol
    iter = 0

    while error > V_tol && iter < V_max_iter
        V_next, k_next = Bellman(prim, res, sho)
        error = maximum(abs.(V_next - res.V))/maximum(abs.(V_next))
        res.V = V_next
        res.k_policy = k_next
        iter += 1
        #println("Currebt distance = ", error, ", at iteration ", iter)
    end

    #=
    if iter == V_max_iter
        println("Maximum iterations reached in VFI")
    elseif error < V_tol
        println("Converged in VFI after ", iter, " iterations")
    end
    =#
    
end

########################### Part 4 - Solve model ###########################
# Simulate the path of K (aggregate capital) from simulation results
function SimulateCapitalPath(prim::Primitives, res::Results, sim::Simulations)
    @unpack_Primitives prim
    @unpack_Results res
    @unpack_Simulations sim

    #bilinear interpolation for policy functions
    #linear interpolation for employed & good state (i.e. state_index = 1)
    interp_g1 = interpolate(k_policy[:,1,:,1], BSpline(Linear()))  # baseline interpolation of value function
    extrap_g1 = extrapolate(interp_g1, Line())              # gives linear extrapolation off grid
    k_prime_g1_interp = scale(extrap_g1, range(k_min, k_max, nk), range(K_min, K_max, nK))
    
    #linear interpolation for unemployed & good state (i.e. state_index = 2)
    interp_g0 = interpolate(k_policy[:,2,:,1], BSpline(Linear()))  # baseline interpolation of value function
    extrap_g0 = extrapolate(interp_g0, Line())              # gives linear extrapolation off grid
    k_prime_g0_interp = scale(extrap_g0, range(k_min, k_max, nk), range(K_min, K_max, nK))
    
    #linear interpolation for employed & bad state (i.e. state_index = 3)
    interp_b1 = interpolate(k_policy[:,1,:,2], BSpline(Linear()))  # baseline interpolation of value function
    extrap_b1 = extrapolate(interp_b1, Line())              # gives linear extrapolation off grid
    k_prime_b1_interp = scale(extrap_b1, range(k_min, k_max, nk), range(K_min, K_max, nK))
        
    #linear interpolation for unemployed & bad state (i.e. state_index = 4)
    interp_b0 = interpolate(k_policy[:,2,:,2], BSpline(Linear()))  # baseline interpolation of value function
    extrap_b0 = extrapolate(interp_b0, Line())              # gives linear extrapolation off grid
    k_prime_b0_interp = scale(extrap_b0, range(k_min, k_max, nk), range(K_min, K_max, nK))

    k_sim::Array{Float64} = zeros(N,T) #matrix for storing individual capital choice

    # generate k and K of t=1 using initial conditions(k0 = K0 = K_initial, z=z_g and P(ϵ=1)=1-u_g=0.96)
    for n_index in 1:N
        Random.seed!(567)
        rand_num = rand()
        if rand_num < 0.96
            k_sim[n_index, 1] = k_prime_g1_interp(K_initial, K_initial)
        else
            k_sim[n_index, 1] = k_prime_g0_interp(K_initial, K_initial)
        end
    end
    K_path[1] = (1/N)*sum(k_sim[:,1])

    # simulate t = 2 to T
    for t_index in 2:T
        for n_index in 1:N
            # determine the state of i_th individual in time t and simulate k'
            if Z[t_index] == 1.01 && E[n_index, t_index] == 1.0     #good and employed
                k_sim[n_index, t_index] = k_prime_g1_interp(k_sim[n_index, t_index-1],K_path[t_index-1])
            elseif Z[t_index] == 1.01 && E[n_index, t_index] == 0.0 #good and unemployed
                k_sim[n_index, t_index] = k_prime_g0_interp(k_sim[n_index, t_index-1],K_path[t_index-1])
            elseif Z[t_index] == 0.99 && E[n_index, t_index] == 1.0 #bad and employed
                k_sim[n_index, t_index] = k_prime_b1_interp(k_sim[n_index, t_index-1],K_path[t_index-1])
            elseif Z[t_index] == 0.99 && E[n_index, t_index] == 0.0 #bad and unemployed
                k_sim[n_index, t_index] = k_prime_b0_interp(k_sim[n_index, t_index-1],K_path[t_index-1])
            end
        end
        K_path[t_index] = (1/N)*sum(k_sim[:,t_index]) #aggregate capital = sum(k_i × μ_i)
    end
    #println("The least capital choice among people is ", minimum(k_sim))
end

function EstimateRegression(prim::Primitives, res::Results, sim::Simulations)
    @unpack_Primitives prim
    @unpack_Results res
    @unpack_Simulations sim

    # get simulated data from res-struct
    K_simdata::Matrix{Float64} = zeros(T,4)
    K_simdata[:,1] = K_path
    K_simdata[:,2] = ones(Float64, sim.T)
    K_simdata[1,3] = K_initial
    for t_index in 2:T
        K_simdata[t_index,3] = K_path[t_index-1]
    end
    K_simdata[:,4] = Z

    K_panel = K_simdata[burn+1:T,:] # the panel we use in regression is from t=1001

    # OLS regression in good states
    K_panel_good = K_panel[K_panel[:,4] .== 1.01, :]
    Yg = log.(K_panel_good[:,1])
    Xg = K_panel_good[:,2:3]
    Xg[:,2] = log.(K_panel_good[:,3])
    a_hat = inv(Xg'*Xg)*Xg'*Yg
    #return a_hat
    aₒ_next = a_hat[1]
    a₁_next = a_hat[2]

    # OLS regression in bad states
    K_panel_bad = K_panel[K_panel[:,4] .== 0.99, :]
    Yb = log.(K_panel_bad[:,1])
    Xb = K_panel_bad[:,2:3]
    Xb[:,2] = log.(K_panel_bad[:,3])
    b_hat = inv(Xb'*Xb)*Xb'*Yb    
    #return b_hat
    bₒ_next = b_hat[1]
    b₁_next = b_hat[2]

    # Get R²
    Yg_hat = Xg*a_hat                  # predicted value of Yg
    Yb_hat = Xb*b_hat                  # predicted value of Yb
    RSS_g = sum((Yg - Yg_hat).^2)      # Residual sum of square for good state
    RSS_b = sum((Yb - Yb_hat).^2)      # Residual sum of square for bad state
    TSS_g = sum((Yg .- mean(Yg)).^2)   # Total sum of square for good state
    TSS_b = sum((Yb .- mean(Yb)).^2)   # Total sum of square for bad state
    res.R² = 1 - (RSS_g+RSS_b)/(TSS_g+TSS_b) # Get (total) R²
 
    return aₒ_next, a₁_next, bₒ_next, b₁_next
end

function SolveModel(prim::Primitives, sho::Shocks, sim::Simulations, res::Results)
    error_reg = 100 * sim.reg_tol
    iter_reg = 0

    while error_reg >sim.reg_tol && iter_reg < sim.reg_max_iter
        VFI(prim,res,sim,sho) # Get next guess of value and policy functions
        SimulateCapitalPath(prim,res,sim) # simulate a panel using updated k'
        a₀_next, a₁_next, b₀_next, b₁_next = EstimateRegression(prim,res,sim) # regress log(K') on (1, log(K)) using updated k' and K_path

        error_reg = abs(a₀_next - res.a₀) + abs(a₁_next - res.a₁) + abs(b₀_next - res.b₀) + abs(b₁_next - res.b₁)
        # Get next guess of parameters
        res.a₀ = sim.λ*a₀_next + (1-sim.λ)*res.a₀
        res.a₁ = sim.λ*a₁_next + (1-sim.λ)*res.a₁
        res.b₀ = sim.λ*b₀_next + (1-sim.λ)*res.b₀
        res.b₁ = sim.λ*b₁_next + (1-sim.λ)*res.b₁
        iter_reg += 1
        println("Current iteration count : ", iter_reg, ", regression error: ", error_reg)
    end

    println(" ")
    println("******* Results ********")    
    println("Final regression error: ", error_reg)
    if iter_reg == sim.reg_max_iter
        println("Maximum iterations reached in regression")
    elseif error_reg < sim.reg_tol
        println("Converged after ", iter_reg, " iterations")
    end
    println("a₀ =", res.a₀)
    println("a₁ =", res.a₁)
    println("b₀ =", res.b₀)
    println("b₁ =", res.b₁)
    println("R² =", res.R²)
    println("************************")    

end