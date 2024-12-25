########################################################################
# Script for solving inventory control model
########################################################################
cd("/Users/yuichirotsuji/Documents/Econ880/PS2-4")
using Distributed
using Parameters, LinearAlgebra, Random, Distributions, Optim, Statistics, SpecialFunctions ,DataFrames, StatFiles, CSVFiles, LatexPrint #import the libraries we want
include("inventory_function.jl")  #import the functions from the provided file

prim,res = Initialize() # primitives initialization
data = get_data()       # data initialization

########################################################################
# Q1. Value function iteration
# (See inventory_function.jl for actual functions)
########################################################################
@time V_iterate(prim,res,data) # about 0.2 sec, 3217 iterations
lap(hcat(data.S,res.V))        # get LaTeX representation (used in Q1)

########################################################################
# Q2. CCP mapping
# (See inventory_function.jl for actual functions)
########################################################################
res.P = CCP_estimator_freq(prim,data) # Get P_hat(CCP)

λ = -4.0                    # "true" value of λ
get_Π_vector(λ, prim, res)  # get flow profit vector at true parameters
Vp = V_CCP(prim, res, data) # compute CCP-impled expected value function
lap(hcat(res.P,Vp))         # get LaTeX representation (used in Q2)

#= 
@time P_iterate(-4.0,prim,res,data)   # about 0.2 sec, 7 iterations
=#

########################################################################
# Q3. LL(λ) calculation
# (See inventory_function.jl for actual functions)
########################################################################
#=
 Actual likelihood function is calculated in "nested_log_liklihood" function.
=#
# plot log-likelihood
LL_val = nested_ll_plot(prim, res, data)
using Plots
plot(prim.λ_grid, LL_val, title = "Nested log-likelihood function plot", ylabel = "value of l(λ)", label = "Nested Log-likelihood", xlabel = "λ")
vline!([-4.0], label = "true λ")
savefig("nested_likelihood_plot.png")

########################################################################
# Q4. MLE estimation
# (See inventory_function.jl for actual functions)
########################################################################
# Simplex method (about 1 sec)
@time opt_nll = optimize(λ₀ -> nll(λ₀),[λ],Optim.Options(iterations = 1000))
opt_nll.minimizer # λ_simplex = -4.02438

# BFGS method (about 2 sec)
@time opt_nll_BFGS = optimize(λ₀ -> nll(λ₀),[λ],BFGS(),Optim.Options(iterations = 1000))
λ_hat = opt_nll_BFGS.minimizer[1] # λ_BFGS = -4.02437
nll_hat = -opt_nll_BFGS.minimum   # maximum likelihood = -2633.152156

########################################################################
