########################################################################
# Initialization of the algorithm
########################################################################
#parameters struct at everywhere
@everywhere @with_kw mutable struct Primitives
    β::Float64 = 0.8 #discount factor
    θ::Float64 = 0.64 #production function parameter

    ns::Int64 = 5 #number of productivity states
    s_grid::SharedVector{Float64} = SharedVector([3.98e-4, 3.58, 6.82, 12.18, 18.79]) #productivity shocks
    F::SharedArray{Float64} = SharedArray([0.6598 0.2600 0.0416 0.0331 0.0055; 
                                           0.1997 0.7201 0.0420 0.0326 0.0056; 
                                           0.2000 0.2000 0.5555 0.0344 0.0101; 
                                           0.2000 0.2000 0.2502 0.3397 0.0101; 
                                           0.2000 0.2000 0.2500 0.3400 0.0100]) #transition matrix
    ν::SharedVector{Float64} = SharedVector([0.37, 0.4631, 0.1102, 0.0504, 0.0063]) #entrant distribution

    cf::Float64 = 10.0 #Incumbents fixed cost
    ce::Float64 = 5.0 #Entry cost
    A::Float64 = 0.005 #Labor disutility parameter

    α::Float64 = 1.0 #variance parameter of random shock
end

#functions struct at everywhere
@everywhere @with_kw mutable struct Results
    val_func::SharedArray{Float64} #value function W(s;p)
    pol_func::SharedArray{Float64} #exit policy function x'(s;p)
    val_func_U::SharedArray{Float64} #value function (with random shock) U(s;p)
    σ::SharedArray{Float64} #exit probability (with random shock) σ(s;p)

    Nd::SharedArray{Float64} #labor demand N^d(s;p)
    Π::SharedArray{Float64} #profit function π(s;p)
    p::Float64 #output price
    μ::SharedVector{Float64} #distribution of firms μ(s;p)
    M::Float64 #mass of new entrants

    EC::Float64 #Free entry condition for finding equilibrium price EC(p)
    LMC::Float64 #Labor makert clearing conditions for finding equilibrium mass of entrants LMC(μ,M)   
end

#initialization function at everywhere
@everywhere function Initialize()
    prim = Primitives()

    val_func = SharedArray{Float64}(zeros(Float64, prim.ns)) 
    pol_func = SharedArray{Float64}(zeros(Float64, prim.ns))
    val_func_U = SharedArray{Float64}(zeros(Float64, prim.ns))
    σ = SharedArray{Float64}(zeros(Float64, prim.ns))

    Nd = SharedArray{Float64}(zeros(Float64, prim.ns))
    Π = SharedArray{Float64}(zeros(Float64, prim.ns))
    p = 0.7 #initial output price guess
    μ = SharedVector{Float64}([0.2, 0.2, 0.2, 0.2, 0.2]) # initial guess of distribution(perfectly equal)
    M = 2.0 #Initial guess of mass of entrants

    EC = 100.0 #some big number
    LMC = 100.0 #some big number

    res = Results(val_func, pol_func, val_func_U, σ, Nd, Π, p, μ, M, EC, LMC)
    return prim, res
end

########################################################################
#######################################################################
# 1. Standard H-R setup
#######################################################################
#######################################################################
# Functions for Solving firm's problem (Algorithm 1.(a))
########################################################################
#Bellman Operator at everywhere to find W(s;p) and x'(s;p)
function Bellman(prim::Primitives, res::Results)
    @unpack β, θ, s_grid, ns, F, cf = prim #unpack model primitives
    @unpack p = res #unpack output price
    W_next = SharedArray{Float64}(zeros(Float64, ns)) #next guess of value function to fill

    @sync @distributed for s_index in eachindex(s_grid) #loop over s (Today's productivity)
        s = s_grid[s_index] #value of s (current productivity)
        nd = (θ*p*s)^(1/(1-θ)) #labor demand (closed form solution)
        profit = p*s*(nd^θ) - nd - p*cf #profit of firm given nd

        res.Nd[s_index] = nd #update labor demand in res-struct
        res.Π[s_index] = profit #update profit in res-struct
        W_next[s_index] = profit  #profit π(s;p) is flow payoff

        entry_val = β*(sum(F[s_index, :] .* res.val_func)) #compute continuation value of entry
        if entry_val < 0 #check whether the continuation value is smaller than 0 (= exit value)
            res.pol_func[s_index] = 1.0 #update policy function (exit = 1) (and value = profit only, no need to update W_next)
        else 
            W_next[s_index] += entry_val #add continuation value to the value function when stay/enter is profitable
            res.pol_func[s_index] = 0.0 #update policy function (stay/enter = 0)
        end    
    end

    return W_next #return next guess of value function

end

#Iteration for value function W(s;p) and policy function x'(s;p)
function W_iterate(prim::Primitives, res::Results, tol::Float64 = 1e-6, err::Float64 = 100.0)
    n = 0 #iteration counter

    while err > tol
        W_next = Bellman(prim, res) #update value funtion
        err = maximum(abs.(W_next .- res.val_func)) #calculate sup(error)
        res.val_func .= W_next #update value function
        n += 1
    end
    #println("Value function converged in ", n, " iterations.")
end

########################################################################
# Calculate EC(p) and find the equilibrium price (Algorithm 1.(b))
#######################################################################
# Entry condition (EC(p)) calculator
@everywhere function EC_culc(prim::Primitives, res::Results)
    res.EC = (sum(res.val_func .* prim.ν)/res.p) - prim.ce   
end

# Find equilibrium price
@everywhere function Solve_price(prim::Primitives, res::Results, tol_EC::Float64 = 1e-6)
    p_upper = 15.0 #guess of upper bound of price
    p_lower = 0.0 #guess of lower bound of price

    n_p = 0 #iteration counter

    while abs(res.EC) > tol_EC && n_p < 1000
        W_iterate(prim, res) #W(s;p) and x'(s;p) are updated and stored in res-struct
        EC_culc(prim, res) #EC is updated and stored in res-struct
 
        #bisection search
        if res.EC > 0
            p_upper = res.p
        else
            p_lower = res.p
        end
        res.p = (p_upper+p_lower)/2 #update price in res-struct

        n_p += 1
        #println("Current iteration count is ", n_p, ".")
    end
    println("The equilibrium price is found in ", n_p, " iterations.")
end

########################################################################
# Calculate EC(p) and find equilibrium price (Algorithm 2.(a) and (b))
#######################################################################
#T-star operator
@everywhere function T_star(prim::Primitives,res::Results)
    @unpack pol_func, μ, M = res #unpack policy functions and distribution
    @unpack s_grid, ns, F, ν = prim #unpack model primitives
    μ_next = SharedVector{Float64}(zeros(Float64, ns)) #next guess of stationary distribution

    @sync @distributed for s_index in eachindex(s_grid) #parallelization for s_grid loop
        s = s_grid[s_index] #value of s(productivity) today
        xp = pol_func[s_index] #extract policy function x'(s)

        if xp == 0.0 # Indicator of stay/entry
            for sp_index in 1:ns #add to μ'(s',M)
                μ_next[sp_index] += F[s_index, sp_index]*μ[s_index] + M*F[s_index, sp_index]*ν[s_index]
            end
        end
    end
    res.μ .= μ_next #update μ to μ' = T_star(μ)
end

#Labor market clearing (LMC(μ, M)) calculator
@everywhere function LMC_culc(prim::Primitives, res::Results)
    net_Nd = sum(res.Nd .* res.μ) + res.M*sum(res.Nd .* prim.ν) #net labor demand
    net_Ns = (1/prim.A) - sum(res.Π .* res.μ) - res.M*sum(res.Π .* prim.ν) #net labor supply (coming from FOC of HH's problem)
    res.LMC = net_Nd - net_Ns
end

#Solve for stationary distribution (μ_star, M_star)
@everywhere function Solve_M(prim::Primitives, res::Results, tol_M::Float64 = 1e-6) #apply T-star operator to μ (in res-struct)

    n_M = 0 #iteration counter

    while abs(res.LMC) > tol_M && n_M < 10000
        T_star(prim, res) #μ(s;p) is updated and stored in res-struct
        LMC_culc(prim, res) #LMC(μ, M) is updated and stored in res-struct

        #search for equilibrium entrant mass (using the method explained in footnote 3 in PS2)
        adj_M = res.M*0.0001 
        if res.LMC > 0
            res.M -= adj_M
        else
            res.M += adj_M
        end

        n_M += 1
        #println("Current excess demand is ", res.LMC, ".")
    end
    println("The equilibrium mass of entrants is found in ", n_M, " iterations.")

end

#######################################################################
# Solve the entire model and show results
#######################################################################
#solve the entire model and print model moments
function Solve_model(prim::Primitives, res::Results)
    Solve_price(prim, res) 
    Solve_M(prim,res)

    println("")
    println("Model moments (cf = ", prim.cf, ")")
    println("***********************")
    println("Price level = ", res.p)
    println("Mass of Incumbents = ", sum((1 .- res.pol_func) .* res.μ))
    println("Mass of Entrants = ", res.M)
    println("Mass of Exits = ", sum(res.pol_func .* res.μ))
    println("Aggregate Labor = ", (sum(res.Nd .* res.μ) + res.M*sum(res.Nd .* prim.ν)))
    println("Labor of Incumbents = ", sum(res.Nd .* res.μ))
    println("Labor of Entrants = ", res.M*sum(res.Nd .* prim.ν))
    println("Fraction of Labor in Entrants = ", res.M*sum(res.Nd .* prim.ν)/(sum(res.Nd .* res.μ) + res.M*sum(res.Nd .* prim.ν)))
    println("***********************")
end


#######################################################################
#######################################################################
# 2. Adding random disturbances to action values
#######################################################################
# Functions for Solving firm's problem (Algorithm 1.- 4.)
#######################################################################
# Bellman operator to find U(s;p)
function Bellman_U(prim::Primitives, res::Results)
    @unpack β, θ, s_grid, ns, F, cf, α = prim #unpack model primitives
    @unpack p = res #unpack output price
    U_next = SharedArray{Float64}(zeros(Float64, ns)) #next guess of value function U(s;p) to fill

    @sync @distributed for s_index in eachindex(s_grid) #loop over s (Today's productivity)
        s = s_grid[s_index] #value of s (current productivity)
        nd = (θ*p*s)^(1/(1-θ)) #labor demand (closed form solution, same as standard case)
        profit = p*s*(nd^θ) - nd - p*cf #profit of firm given nd (same as standard case)

        res.Nd[s_index] = nd #update labor demand in res-struct
        res.Π[s_index] = profit #update profit in res-struct

        entry_val = β*(sum(F[s_index, :] .* res.val_func_U)) #continuation value

        V_enter = profit + entry_val #V(x=0)(s;p) (equation (12))
        V_exit = profit #V(x=1)(s;p) (equation (13))

        #use the log-sum-exp trick to avoid an overflow (I asked chatGPT how to overcome this kind of overflow)
        m = max(α*V_enter, α*V_exit)
        U_next[s_index] = (SpecialFunctions.γ/α) + ((m + log(exp(α*V_enter - m) + exp(α*V_exit - m)))/α) #new guess of U(s_i;p) 

        #update exit choice probability (closed form, with tiny transformation to avoid an overflow)
        res.σ[s_index] =  1/(1 + exp(α*(V_enter - V_exit))) 
    end

    return U_next #return next guess of value function U(s;p)

end

#Iteration for value function U(s;p) and choice probability σ(s;p)
function U_iterate(prim::Primitives, res::Results, tol_U::Float64 = 1e-6, err_U::Float64 = 100.0)
    n_U = 0

    while err_U > tol_U && n_U < 1000
        U_next = Bellman_U(prim, res)
        err_U = maximum(abs.(U_next .- res.val_func_U))
        res.val_func_U .= U_next #update value function U(s;p)
        n_U += 1
        #println(err_U)
    end

    #println("Value function converged in ", n_U, " iterations.")
end

########################################################################
# Find equilibrium price, cross-sectional destribution and mass of entrants
# (Note: most of the functions are close to the standard case)
#######################################################################
# Entry condition with U(s;p)
@everywhere function EC_culc_U(prim::Primitives, res::Results)
    res.EC = (sum(res.val_func_U .* prim.ν)/res.p) - prim.ce   
end

# Find equilibrium price with U(s;p)
@everywhere function Solve_price_U(prim::Primitives, res::Results, tol_EC::Float64 = 1e-4)
    n_p = 0 #iteration counter

    while abs(res.EC) > tol_EC && n_p < 100000
        U_iterate(prim, res) #U(s;p) and σ(s;p) are updated and stored in res-struct
        EC_culc_U(prim, res) #EC is updated and stored in res-struct

        #search for equilibrium price
        adj_p = res.p*0.00001
        if res.EC > 0
            res.p -= adj_p
        else
            res.p += adj_p
        end

        n_p += 1
        #println(res.EC, ", p = ", res.p)
        #println("Current iteration count is ", n_p, ".")
    end
    println("The equilibrium price is found in ", n_p, " iterations.")
end

#T-star operator for stochastic case
@everywhere function T_star_stochastic(prim::Primitives,res::Results)
    @unpack σ, μ, M = res #unpack policy functions and distribution
    @unpack s_grid, ns, F, ν = prim #unpack model primitives
    μ_next = SharedVector{Float64}(zeros(Float64, ns)) #next guess of stationary distribution

    @sync @distributed for s_index in eachindex(s_grid) #parallelization for s_grid loop
        s = s_grid[s_index] #value of s(productivity) today
        xp = σ[s_index] #extract exit probability σ(s;p)

        #calculation of μ'(s',M) (now we can use (1-σ(s;p)) for exit probability)
        for sp_index in 1:ns
            μ_next[sp_index] += (1-xp)*F[s_index, sp_index]*μ[s_index] + (1-xp)*M*F[s_index, sp_index]*ν[s_index]        
        end
    end
    res.μ .= μ_next #update μ to μ' = T_star(μ) in res-struct
end

#Labor market clearing (LMC(μ, M)) calculator is ths same as standard case

#Solve for stationary distribution (μ_star, M_star) in stochastic case
@everywhere function Solve_M_Stochastic(prim::Primitives, res::Results, tol_M::Float64 = 1e-6) #apply T-star operator to μ (in res-struct)

    n_M = 0 #iteration counter

    while abs(res.LMC) > tol_M && n_M < 10000
        T_star_stochastic(prim, res) #μ(s;p) is updated and stored in res-struct
        LMC_culc(prim, res) #LMC(μ, M) is updated and stored in res-struct

        #search for equilibrium entrant mass
        adj_M = res.M*0.0001 
        if res.LMC > 0
            res.M -= adj_M
        else
            res.M += adj_M
        end

        n_M += 1
        #println("Current excess demand is ", res.LMC, ".")
    end
    println("The equilibrium mass of entrants is found in ", n_M, " iterations.")

end

#solve the entire model and print model moments
function Solve_model_Stochastic(prim::Primitives, res::Results)
    Solve_price_U(prim, res) 
    Solve_M_Stochastic(prim,res)
    
    println("")
    println("Model moments: (cf = ", prim.cf, ", α = ", prim.α, ")")
    println("***********************")
    println("Price level = ", res.p)
    println("Mass of Incumbents = ", sum((1 .- res.σ) .* res.μ))
    println("Mass of Entrants = ", res.M)
    println("Mass of Exits = ", sum(res.σ .* res.μ))
    println("Aggregate Labor = ", (sum(res.Nd .* res.μ) + res.M*sum(res.Nd .* prim.ν)))
    println("Labor of Incumbents = ", sum(res.Nd .* res.μ))
    println("Labor of Entrants = ", res.M*sum(res.Nd .* prim.ν))
    println("Fraction of Labor in Entrants = ", res.M*sum(res.Nd .* prim.ν)/(sum(res.Nd .* res.μ) + res.M*sum(res.Nd .* prim.ν)))
    println("***********************")
end
