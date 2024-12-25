#######################################################################
#Initialize parameters and functions/distributions
#######################################################################
#cd("/Users/yuichirotsuji/Documents/Econ880/PS1-3")
using Distributed
#addprocs(2)
#import Pkg; Pkg.add("SpecialFunctions")
@everywhere using Parameters, Plots, SharedArrays, SpecialFunctions #import the libraries we want
include("H-R_functions.jl") #import the functions that solve the model

#######################################################################
# Solve for model 1 (Standard H-R setup)
#######################################################################
# Q1 & Q2: Solve the entire models and compute moments
@everywhere prim,res = Initialize()
@time Solve_model(prim,res) #This takes about 1-2 seconds

# Q3: Plot decision rules
using Plots
@unpack pol_func = res
plot(pol_func, title="Decision rules of exit",ylabel = "Probability of exit", label = "Standard", xlabel = "Productivity level") #Standard case

#######################################################################
# Solve for model 2(Adding random disturbances to action values)
#######################################################################
# For α = 1.0
# Q1 & Q2: Solve the entire models and compute moments
@everywhere prim,res = Initialize()
prim.α = 1.0 #set α = 1.0
@time Solve_model_Stochastic(prim,res) #This takes about 60 seconds

# Q3: Plot decision rules
@unpack s_grid = prim
@unpack σ = res
plot!(σ, label = "TV1 α=1") #exit probability plot of TV1, α=1

###
# For α = 2.0 case
# Q1 & Q2: Solve the entire models and compute moments
@everywhere prim,res = Initialize()
prim.α = 2.0 #set α = 2.0
@time Solve_model_Stochastic(prim,res) #This takes about 5-10 seconds

# Q3: Plot decision rules
@unpack σ = res
plot!(σ, label = "TV1 α=2") #exit probability plot of TV1, α=2
savefig("Exit_probability_10.png") #used in Q3

#######################################################################
#######################################################################
# Q4: Check decision rules when c_f = 15
#######################################################################
#resolve all the models with different fixed cost (cf = 15)
###
#Standard case
@everywhere prim,res = Initialize() #Initialize all stuff
prim.cf = 15.0 #change cf to 15
@time Solve_model(prim,res) #This takes about 1-2 seconds
using Plots
@unpack pol_func = res
plot(pol_func, title="Decision rules of exit when cf = 15",ylabel = "Probability of exit", label = "Standard", xlabel = "Productivity level") #Standard case

###
#TV1, α = 1.0 case
@everywhere prim,res = Initialize() #Initialize all stuff
prim.cf = 15.0 #change cf to 15
prim.α = 1.0 #set α = 1.0
@time Solve_model_Stochastic(prim,res) #This takes about 30 seconds
@unpack σ = res
plot!(σ, label = "TV1 α=1") #exit probability plot of TV1, α=1

###
#TV2, α = 2.0 case
# Q1 & Q2: Solve the entire models and compute moments
@everywhere prim,res = Initialize() #Initialize all stuff
prim.cf = 15.0 #change cf to 15
prim.α = 2.0 #set α = 2.0
res.p = 0.8 #due to the slow convergence, we changed the initial guess of price here
@time Solve_model_Stochastic(prim,res) #This takes about 10-20 seconds after adjusting the initial guess
@unpack σ = res
plot!(σ, label = "TV1 α=2") #exit probability plot of TV1, α=2
savefig("Exit_probability_15.png") #used in Q4
#######################################################################
