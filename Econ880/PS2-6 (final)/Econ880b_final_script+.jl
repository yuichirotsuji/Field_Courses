########################################################################
# Econ880b take-home exam
# Author: Yuichiro Tsuji
# (Script for solving refinancing model)
########################################################################
#cd("/Users/yuichirotsuji/Documents/Econ880/PS2-6 (final)")
using Distributed
using Parameters, LinearAlgebra, Random, Distributions, Optim, Statistics, SpecialFunctions ,DataFrames, StatFiles, CSVFiles, LatexPrint #import the libraries we want
include("Econ880b_final_functions+.jl")  #import the functions from the provided file

prim,res = Initialize() # primitives initialization
data = get_data()       # data initialization

########################################################################
# Q3. Policy function iteration and plot
# (See Econ880b_final_functions.jl for actual functions)
########################################################################
# get Pr(a=1|s,κ) using policy function iteration
@time P_iterate(prim.α,prim.κₕ,prim.κₗ,prim,res,data)

# plot Pr(a=1|s,κ)
using Plots
r_diff = collect(-3.0:0.25:3.0)                   # r-r₀ grid
cp_100_h = vcat(res.Pₕ[326:338],res.Pₕ[171:182])   # get P(a=1|l=100, high)
plot(r_diff,cp_100_h,title = "Refinancing probability Pr(a=1|s,κ) plot", ylabel = "Pr(a=1|s,κ)", label = "High,l=100", xlabel = "r-r₀")
cp_100_l = vcat(res.Pₗ[326:338],res.Pₗ[171:182])   # get P(a=1|l=100, low)
plot!(r_diff,cp_100_l,label = "Low, l=100")
cp_300_h = vcat(res.Pₕ[1002:1014],res.Pₕ[847:858]) # get P(a=1|l=300, high)
plot!(r_diff,cp_300_h,label = "High,l=300")
cp_300_l = vcat(res.Pₗ[1002:1014],res.Pₗ[847:858])  # get P(a=1|l=300, low)
plot!(r_diff,cp_300_l,label = "Low, l=300")
savefig("CCP_plot.png") # used in Q3

########################################################################
# Q6. Estimation via MLE with BLGS
# (See Econ880b_final_functions.jl for actual functions)
########################################################################
# initial guess of parameters (α,κₕ,κₗ,a)
# (Note: When using parameters in Q3, the iteration will immediately finish)
guess = [prim.α,prim.κₕ+0.1,prim.κₗ-0.1,0.1] 

# perform MLE with BFGS method (about 3-5 min with mac M1 chip)
@time opt_nll_ave_BFGS = optimize(θ -> nll_ave(θ),guess,BFGS(),Optim.Options(show_trace = true, iterations = 1000))
θ_hat = opt_nll_ave_BFGS.minimizer       # θ_BFGS =(-0.750,4.995,0.973,-0.005) (=(α,κₕ,κₗ,a))
π_hat = exp(θ_hat[4])/(1+exp(θ_hat[4]))  # π_hat = 0.4987
opt_average = -opt_nll_ave_BFGS.minimum  # maximum average likelihood = -1.332

#= (A wrong MLE in ver1)
# perform MLE with BFGS method (about 10-15 min with mac M1 tip)
@time opt_nll_BFGS = optimize(θ -> nll(θ),guess,BFGS(),Optim.Options(iterations = 1000))
λ_hat = opt_nll_BFGS.minimizer          # θ_BFGS =(-0.750,5.850,0.897,-0.540) (=(α,κₕ,κₗ,a))
π_hat = exp(λ_hat[4])/(1+exp(λ_hat[4])) # π_hat = 0.368
nll_hat = -opt_nll_BFGS.minimum         # maximum likelihood = -1347.220
=#
#= (just for experiment with Simplex method)
# Simplex method (about 3 sec)
@time opt_nll = optimize(θ -> nll(θ),guess,Optim.Options(iterations = 1000))
opt_nll.minimizer
@time opt_nll_ave = optimize(θ -> nll_ave(θ),guess,Optim.Options(g_tol = 1e-2, show_trace = true, iterations = 1000))
opt_nll_ave.minimizer
=#


########################################################################
