########################################################################
# Script for solving Aiyagari model (Q1 and Q2)
# Author: Yuichiro Tsuji
########################################################################
#cd("/Users/yuichirotsuji/Documents/Econ880/PS1-7(midterm)")
using Distributed
@everywhere using Parameters, LinearAlgebra, Random, Interpolations, Optim #import the libraries we want
include("Aiyagari.jl")  #import the functions from the provided file
prim,res = SolveModel() #solve the model first and update structs

######################################################
# Q1. Compute socially optimal capital
######################################################
@unpack_Primitives prim
# Compute K^* 
println("The socially optimal steady state level of capital is ", (((1/α)*((1/β)-(1-δ)))^(1/(α-1)))*L, ".")

######################################################
# Q2. Derive and plot the policy function in the Aiyagari
######################################################
# plot the policy functions
using Plots
plot(prim.k_grid, res.k_policy[:,1], title="Policy Functions k'(k,ϵ)",ylabel = "policy k'(k,ϵ)", label = "Employed", xlabel = "capital today(k)") # k'(k,ϵ=1)
plot!(prim.k_grid, res.k_policy[:,2], label = "Unemployed") # k'(k,ϵ=1)
savefig("Policy_Functions_Aiyagari.png") #used in Q.2

# Get the aggregate capital level
println("The steady state level of capital in the Aiyagari model is ", res.K, ".")

########################################################################
