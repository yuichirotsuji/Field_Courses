#cd("/Users/yuichirotsuji/Documents/Econ880/PS1-2")
using Distributed
addprocs(2)
@everywhere using Parameters, Plots, SharedArrays #import the libraries we want
include("Huggett_functions.jl") #import the functions that solve the model
@everywhere prim,res = Initialize()
@time Solve_model(prim,res) #This takes about 25 sec

########################################################################
# Make plot (II(a.))
########################################################################
#policy function plot with 45-degree line
@unpack val_func, pol_func = res
@unpack a_grid = prim
using Plots
plot(a_grid, pol_func[:,1], title="Policy Function a'(a,s:q)",ylabel = "policy a'(a,s:q)", label = "Employed", xlabel = "asset holdings a")
plot!(a_grid, pol_func[:,2], label = "Unemployed")
plot!(a_grid,a_grid,label = "45 degree",color="red",linestyle=:dash)
savefig("Policy_Functions.png") #used in Q.II(a)

########################################################################
# Derive market clearing price and stationary distribution (II(b.))
#######################################################################
#sloving for equilibrium price
@time Solve_Huggett(prim, res) #this takes about 80 sec
res.q #q_star = 0.99427

#Wealth distribution plot
Wealth_dist(prim,res)
@unpack a_grid = prim
@unpack μ_w = res
plot(a_grid, μ_w[:,1], title="Wealth distribution",ylabel = "density", label = "Employed", xlabel = "Wealth(income + asset)")
plot!(a_grid, μ_w[:,2], label = "Unemployed")
savefig("Wealth_Distribution.png") #used in Q.II(b)

########################################################################
# Lorenz curve and Gini index (II(c.))
#######################################################################
#Lorenz curve 
cum, Lorenz = Lorenz_curve(prim,res)
plot(cum, Lorenz, title="Lorenz Curve",ylabel = "Fraction of wealth", label = "Lorenz curve", xlabel = "Fraction of people")
plot!(cum, cum, color="red",linestyle=:dash, xlim=(0.0,1.0), ylim = (-0.1,1.0), label = false)
savefig("Lorenz_curve.png") #used in Q.II(c)

#Gini coefficient
G = Gini_coef(prim, Lorenz, cum) #Gini = 0.3844

########################################################################
# Welfare distribution (III)
#######################################################################
@unpack μ, val_func = res
@unpack α, β, a_grid, s_grid = prim

#(a.) plot λ
WFB = (((0.9434*1) + (0.0566*0.5))^(1-α) - 1)/((1-α)*(1-β)) #use the result in slides: WFB = -4.252
λ = Calc_lambda(prim, res, WFB)
using Plots
plot(a_grid, λ[:,1], title="Consumption Equivalents λ(a,s)",ylabel = "λ(a,s)", label = "Employed", xlabel = "asset holdings a")
plot!(a_grid, λ[:,2], label = "Unemployed")
savefig("lambda_plot.png") #used in Q.III(a)

#(b.) Derive WFB, WINC, WG
WINC = sum(μ .* val_func) #WINC = -4.456
WG = sum(λ .* μ) #WG = 0.0013

#(c.) Derive population who prefer complete market
CM_favor = sum((λ .>= 0) .* μ) #54.07%%