########################################################################
# Script for solving Krusell-Smith model
# Author: Yuichiro Tsuji
########################################################################
#cd("/Users/yuichirotsuji/Documents/Econ880/PS1-7(midterm)")
using Distributed
@everywhere using Parameters, LinearAlgebra, Random, Interpolations, Optim, Statistics #import the libraries we want
include("KS_function.jl")  #import the functions from the provided file

########################################################################
# solve K-S model
prim, sho, sim, res = Initialize()
@time SolveModel(prim,sho,sim,res) # takes about 1 min

# plot simulation results of K
using Plots
plot(res.K_path,title="Simulated path of aggregate capital",ylabel = "aggregate capital K", label = "Simulated K", xlabel = "time")
savefig("K_path.png") #used in Q.3
plot(res.K_path[1001:11000], title="Simulated path of aggregate capital for t>1000",ylabel = "aggregate capital K", label = "Simulated K", xlabel = "time")
savefig("K_path_burned.png") #used in Q.3

# plot policy functions
using Plots
plot(res.k_policy[:,1,1,1], title="Policy functions in Krusell-Smith",ylabel = "Policy k'(k,ϵ;K,z)", label = "Good & Employed", xlabel = "individual capital k")
plot!(res.k_policy[:,2,1,1], label = "Good & Unemployed")
plot!(res.k_policy[:,1,1,2], label = "Bad & Employed")
plot!(res.k_policy[:,2,1,2], label = "Bad & Unemployed")
savefig("Policy_Functions_KS.png") #used in Q.3

########################################################################
