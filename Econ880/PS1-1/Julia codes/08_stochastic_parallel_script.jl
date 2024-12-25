#############
#Peform DP using parallelization
using Distributed
addprocs(2)
@everywhere using Parameters, Plots, SharedArrays #import the libraries we want
include("07_stochastic_parallel_functions.jl") #import the functions that solve our growth model
@everywhere prim, res = Initialize() #initialize primitive and results structs
@time Solve_model(prim, res) #solve the model!

#############
#Make plots
@unpack val_func, pol_func = res
@unpack k_grid = prim

#value function
plot(k_grid, val_func[:,1], title="Value Function V(K,Z)",ylabel = "value V(K,Z)", label = "Z = 1.25", xlabel = "capital K")
plot!(k_grid, val_func[:,2], label = "Z = 0.2",xlabel = "capital K")
savefig("Value_Functions.png") #used in Q2

#policy functions
plot(k_grid, pol_func[:,1], title="Policy Function K'(K,Z)", ylabel = "policy K'(K)", label = "Z = 1.25", xlabel = "capital K", linestyle=:solid)
plot!(k_grid, pol_func[:,2], title="Policy Function K'(K,Z)", label = "Z = 0.2", linestyle=:solid)
plot!(k_grid,k_grid,label = "45 degree",color="red",linestyle=:dash)
savefig("Policy_Functions.png") #used in Q3

#changes in policy function
pol_func_δ = pol_func.-k_grid
plot(k_grid, pol_func_δ[:,1], title="Saving Policy Function K'(K,Z) - K",ylabel = "saving policy K'(K,Z) - K", label = "Z=1.25",xlabel = "capital K")
plot!(k_grid, pol_func_δ[:,2], label = "Z = 0.2")
savefig("Policy_Functions_Changes.png") #used in Q3

println("All done!")
################################
