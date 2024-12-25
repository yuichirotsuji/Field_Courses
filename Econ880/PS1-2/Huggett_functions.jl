# We followed the instruction in "PS2_pseudocode" to derive q_star.
# (Step # in this code corresponds to numbers on first page of the pseudocode)

########################################################################
# Initialization of the algorithm (Step 1.)
########################################################################
#parameters struct at everywhere
@everywhere @with_kw struct Primitives
    β::Float64 = 0.9932 #discount factor
    α::Float64 = 1.5 #CRRA parameter

    a_min::Float64 = -2.0 
    a_max::Float64 = 5.0
    na::Int64 = 1000 #capital grid
    a_grid::SharedVector{Float64} = SharedVector(collect(range(start=a_min, stop=a_max, length=na)))

    ns::Int64 = 2 #number of states (e and u)
    s_grid::SharedVector{Float64} = SharedVector([1, 0.5]) #earnings in each state
    Π::SharedArray{Float64, 2} = SharedArray([0.97 0.03; 0.5 0.5]) #transition matrix
end

#functions struct at everywhere
@everywhere @with_kw mutable struct Results
    val_func::SharedArray{Float64, 2} #value function v(a,s:q)
    pol_func::SharedArray{Float64, 2} #policy function a'(a,s:q)
    q::Float64 #bond price
    μ::SharedArray{Float64, 2} #stationary distribution of asset
    μ_w::SharedArray{Float64, 2} #stationary distribution of wealth
    ES::Float64 #excess supply of asset
end

#initialization function at everywhere
@everywhere function Initialize()
    prim = Primitives()
    val_func = SharedArray{Float64, 2}(zeros(Float64, prim.na, prim.ns))
    pol_func = SharedArray{Float64, 2}(zeros(Float64, prim.na, prim.ns))
    q = 0.995 #initial bond price (something between β and 1.0)
    μ = SharedArray{Float64, 2}(fill(1/(prim.na*prim.ns), prim.na, prim.ns)) # initial distribution(perfectly equal)
    μ_w = SharedArray{Float64, 2}(zeros(Float64, prim.na, prim.ns)) # initial distribution(zeros)
    ES = 10.0 #inital excess supply (some big number)
    res = Results(val_func, pol_func, q, μ, μ_w, ES)
    prim, res
end

########################################################################
# Value Function Iteration (Step 2.)
########################################################################
#Bellman Operator at everywhere, stochastic version
@everywhere function Bellman(prim::Primitives,res::Results)
    @unpack val_func, q = res #unpack value function
    @unpack β, α, a_grid, s_grid, na, ns, Π = prim #unpack model primitives
    v_next = SharedArray{Float64, 2}(zeros(Float64, na, ns)) #next guess of value function to fill

    @sync @distributed for s_index in eachindex(s_grid) #parallelization for z_grid loop
        for a_index in eachindex(a_grid) #we devide k_index loop for @distributed macro to work well

            a = a_grid[a_index] #value of a
            s = s_grid[s_index] #value of s(earnings)
            candidate_max = -Inf #bad candidate max
            budget = s + a #budget(= earings + asset)

            for ap_index in 1:na #loop over possible selections of a'
                c = budget - q*a_grid[ap_index] #consumption given a' selection
                if c > 0 #check for positivity
                    val = (c^(1-α)-1)/(1-α) + β*(Π[s_index, 1] * val_func[ap_index, 1] + Π[s_index, 2]*val_func[ap_index, 2]) #compute (expected) value
                    if val > candidate_max #check for new max value
                        candidate_max = val #update max value
                        res.pol_func[a_index, s_index] = a_grid[ap_index] #update policy function
                    end
                end
            end
            v_next[a_index, s_index] = candidate_max #update value function
        end
    end
    v_next #return next guess of value function
end

#Iteration
function V_iterate(prim::Primitives, res::Results, tol::Float64 = 1e-6, err::Float64 = 100.0)
    n = 0

    while err > tol
        v_next = Bellman(prim, res)
        err = maximum(abs.(v_next .- res.val_func))
        res.val_func .= v_next
        n += 1
    end
    println("Value function converged in ", n, " iterations.")
end

#solve the model
function Solve_model(prim::Primitives, res::Results)
    V_iterate(prim, res) #in this case, all we have to do is the value function iteration!
end

########################################################################
# Make T-star operator and find stationary distribution (Step 3.)
########################################################################
#T-star operator
@everywhere function T_star(prim::Primitives,res::Results)
    @unpack pol_func, μ = res #unpack policy function and distribution
    @unpack a_grid, s_grid, na, ns, Π = prim #unpack model primitives
    μ_next = SharedArray{Float64, 2}(zeros(Float64, na, ns)) #next guess of stationary distribution

    @sync @distributed for s_index in eachindex(s_grid) #parallelization for s_grid loop
        s = s_grid[s_index] #value of s(earnings) today
        for a_index in eachindex(a_grid) #we devide a_index loop for @distributed macro to work well
            a = a_grid[a_index] #value of a today
            ap = pol_func[a_index, s_index] #extract the value of policy function a'(a,s)
            
            for ap_index in 1:na #loop over all transition of a' (This may be inefficient - would be better if we stopped at a' = a_index)
                if ap == a_grid[ap_index] #indicator function 1{a' = g(a,b)}
                    for sp_index in 1:ns
                        μ_next[ap_index, sp_index] += Π[s_index, sp_index]*μ[a_index, s_index] #add probability mass
                    end
                end
            end
            
        end
    end
    μ_next #return next guess of stationary distribution
end

#solve for stationary distribution using the T-star operator
@everywhere function Solve_SD(prim::Primitives, res::Results, tol_μ::Float64 = 1e-7, err_μ::Float64 = 100.0) #apply T-star operator to μ (in res-struct)
    n_μ = 0 #iteration counter

    while err_μ > tol_μ
        μ_next = T_star(prim, res) #apply T-star operator to μ (in res-struct)
        err_μ = maximum(abs.(μ_next .- res.μ))
        res.μ .= μ_next # update μ in res-struct
        n_μ += 1
    end
    #println("μ converged to statinonary distribution in ", n_μ, " iterations.")
end

########################################################################
# Calculate excess supply and find equilibrium bond price (Step 4.)
########################################################################
#calculate excess supply with a function
@everywhere function Calc_ExcessSupply(res::Results)
    res.ES = sum(res.pol_func .* res.μ)
end

#solve for q_star
@everywhere function Solve_Huggett(prim::Primitives, res::Results, tol_ES::Float64 = 1e-3)
    #elements for binary search of q_star
    @unpack β = prim
    q_upper = 1.0
    q_lower = β

    n_q = 0 #iteration counter
    while abs(res.ES) > tol_ES && n_q < 1000 # we will stop at 1000 iteration
        Solve_model(prim, res) #V(a,s) and a'(a,s) are updated and stored in res-struct
        Solve_SD(prim, res) #μ is updated (given new a'(a,s)) and stored in res-struct
        Calc_ExcessSupply(res) #Excess supply is updated (given new a'(a,s) and μ) and stored in res-struct

        #binary search for q_star
        if res.ES > 0
            q_lower = res.q
            res.q = (q_upper + q_lower)/2 
        else
            q_upper = res.q
            res.q = (q_upper + q_lower)/2 
        end

        println("Current iteration count is ", n_q, ".")
        n_q += 1
    end

    println("The market clearing bond price is found in ", n_q, " iterations.")

end

########################################################################
########################################################################
# Wealth / Lorentz / Gini  
########################################################################
# wealth (= income + asset) distribution generator
@everywhere function Wealth_dist(prim::Primitives, res::Results)
    @unpack μ, μ_w = res
    @unpack na, ns = prim

    for a_index in 1:na
        w_index_e = min(a_index + 143, 1000) #asset + employed income(1)
        w_index_u = min(a_index + 71, 1000) #asset + unemployed income(.5)

        μ_w[w_index_e, 1] = μ[a_index, 1] #wealth distribution of employed people 
        μ_w[w_index_u, 2] = μ[a_index, 2] #wealth distribution of unemployed people
    end
end

# Lorenz curve generator
@everywhere function Lorenz_curve(prim::Primitives, res::Results)
    @unpack a_grid, na = prim
    @unpack μ_w = res
    total_μ_w = μ_w[:,1] + μ_w[:,2] #total mass of people in each wealth level

    cum_mass::Array{Float64} = zeros(na) #cumulative mass of people (x-axis of Lorenz curve)
    cum_wealth::Array{Float64} = zeros(na) #cumulative mass of wealth (y-axis of Lorenz curve is cum_wealth/total_wealth)

    # make cumulative mass and wealth in a recursive way
    cum_mass[1] = total_μ_w[1] 
    cum_wealth[1] = a_grid[1]*total_μ_w[1]
    for a_index in 2:na
        cum_mass[a_index] = total_μ_w[a_index] + cum_mass[a_index - 1]
        cum_wealth[a_index] = a_grid[a_index]*total_μ_w[a_index] + cum_wealth[a_index - 1]
    end    

    return cum_mass, cum_wealth ./ cum_wealth[na] # x_grid and y_grid of Lorenz curve
end

#Gini coefficients culculator
@everywhere function Gini_coef(prim::Primitives, Lorenz::Vector{Float64}, cum::Vector{Float64})
    @unpack a_grid, na = prim

    dx::Vector{Float64} =  zeros(Float64, na)
    for a_index in 2:na
        dx[a_index] = cum[a_index] - cum[a_index - 1]
    end
    
    return 2*sum((cum-Lorenz) .* dx)
    
end

########################################################################
# Welfare calculation
########################################################################
# λ-calculator
@everywhere function Calc_lambda(prim::Primitives, res::Results, W::Float64)
    @unpack β, α, a_grid, s_grid, na, ns = prim
    @unpack val_func = res

    λ::Array{Float64, 2} = zeros(Float64, na, ns)
    for s_index in 1:2
        for a_index in eachindex(a_grid) 
            λ[a_index, s_index] = ((W + 1.0/((1.0-α)*(1.0-β)))/(val_func[a_index, s_index] + 1.0/((1.0-α)*(1.0-β))))^(1.0/(1.0-α)) -1.0
        end
    end

    return λ
end

