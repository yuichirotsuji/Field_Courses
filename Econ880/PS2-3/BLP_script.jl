########################################################################
# Script for solving BLP
########################################################################
cd("/Users/yuichirotsuji/Documents/Econ880/PS2-3")
using Distributed
using Parameters, LinearAlgebra, Random, Distributions, Optim, Statistics, DataFrames, StatFiles, CSVFiles #import the libraries we want
include("BLP_function.jl")  # import the functions from the provided file
prim,res = Initialize()     # primitives initialization
data = get_data()           # data initialization

########################################################################
# Q1. Demand inversion
# (See BLP_function.jl for actual functions)
########################################################################
# using 1985 market data
price₁ = data.df_X[data.df_X.Year .== 1985, :price]
share₁ = data.df_X[data.df_X.Year .== 1985, :share]
λₚ = 0.6 # value of λₚ = 0.6

# contraction only
@time e_evol_contraction = δ_iterate_contraction(λₚ, price₁, share₁, prim, data)[2]
using Plots
plot(e_evol_contraction[2:end], title= "Evolution of the error norm", ylabel = "sup(log(S)-log(σ))", label = "Contraction mapping", xlabel = "Iterations")

# mix of contraction and Newton
@time e_evol_mix = δ_iterate_mix(λₚ, price₁, share₁, prim, data)[2]
plot!(e_evol_mix[2:end], title= "Evolution of the error norm", label = "Contraction mapping & Newton")
plot!(ones(length(e_evol_contraction[2:end])),label = "Contraction-Newton threshold")
savefig("error_norm_evolution.png")

########################################################################
# Q2. GMM objective function plot
# (See BLP_function.jl for actual functions)
########################################################################
# GMM objective function grid search with 2sls weighting matrix
@time G_val = GMM_plot(prim.λ_grid,prim,res,data,true) # about 2 mins
# plot GMM objective functions
using Plots
plot(prim.λ_grid, G_val, title = "GMM objective function plot", ylabel = "value of GMM objective function", label = "2SLS weighting matrix", xlabel = "λ")
savefig("GMM_plot.png")

#= (Test for extended plot: we don't use this result in the document)
#@time G_val₂ = GMM_plot(prim.λ_grid₂,prim,res,data) # about 3 mins
#plot(prim.λ_grid₂, G_val₂, title = "GMM objective function plot", ylabel = "value of GMM objective function", label = "G(λ)", xlabel = "λ")
#savefig("GMM_plot_2.png")
=#

########################################################################
# Q3. Parameters estimation
# (See BLP_function.jl for actual functions)
########################################################################
# first stage GMM function minimization with BFGS method
res.W = inv(data.Z' * data.Z) # initial W = 2sls weight
@time opt_GMM = optimize(λ -> G(λ),[λₚ],BFGS(),Optim.Options(iterations = 1000)) # about 45 sec
λ_star = opt_GMM.minimizer # λ₁ = 0.6476
G_star = opt_GMM.minimum   # G(λ₁) = 245.23155

# update W in res-struct
res.D = δ_simulator_all(λ_star[1], prim, data)  # update δ
ρ₂ = res.D - data.X*iv_estimator(res,data)      # get new ρ
res.W = inv((data.Z .* ρ₂)'*(data.Z .* ρ₂))     # update W

# plot GMM objective function with optimal weighting matrix
@time G_val₂ = GMM_plot(prim.λ_grid,prim,res,data,false) # about 2-3 mins
plot(prim.λ_grid, G_val, label = "2SLS weighting matrix")
plot!(prim.λ_grid, G_val₂, title = "GMM objective functions and estimates comparison", ylabel = "value of GMM objective function",  xlabel = "λ", label = "Optimal weighting matrix")
scatter!([0.6476,0.5933], [245.23155,161.89165], label = "λₚ^hat") # GMM estimates plot
savefig("GMM_plot_comparison.png")

#= for making GMM plot with optimal W only (we didn't use this in our write-up document)
plot(prim.λ_grid, G_val₂, title = "G(λ) with optimal weighting matrix", ylabel = "value of GMM objective function", label = "Optimal weighting matrix",xlabel = "λ")
savefig("GMM_plot_Optimal.png")
=#

# second stage GMM estimation with optimal weighting matrix
@time opt_GMM₂ = optimize(λ -> G(λ),[λₚ],BFGS(),Optim.Options(g_tol = 1e-2,iterations = 1000)) # about 30 sec
λ_star₂ = opt_GMM₂.minimizer # λ₂ = 0.5933
G_star = opt_GMM₂.minimum    # G(λ₂) 161.89165

########################################################################

