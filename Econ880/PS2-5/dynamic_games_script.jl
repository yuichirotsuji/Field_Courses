########################################################################
# Script for solving dynamic games
########################################################################
#cd("/Users/yuichirotsuji/Documents/Econ880/PS2-5")
using Distributed
using Parameters, LinearAlgebra, Random, Distributions, Optim, Statistics, SpecialFunctions ,DataFrames, StatFiles, CSVFiles, LatexPrint #import the libraries we want
include("dynamic_games_functions.jl")  # import the functions from the provided file
prim,res,data = Initialize()           # primitives & data initialization

########################################################################
# Q1. Profit matrix
# (See dynamic_games_functions.jl for actual functions)
########################################################################
# (Note: The profit matrix is created by running Initialize() function)
# plofit matrix surface plot (as FIG.1 in Besanko and Doraszelski (2004))
using Plots
surface(prim.q_grid,prim.q_grid,data.Π, title = "Profit π(ω) plot", xlabel = "i", ylabel = "j", zlabel = "π(i,j)" )
savefig("plofit.png")

########################################################################
# Q2. Define MPE
########################################################################
#=
(See the document for the definition)
=#

########################################################################
# Q3. Solve for MPE
# (See inventory_function.jl for actual functions)
########################################################################
# iteration
@time xV_iterate(prim,res,data)
lap(res.x)

# MPE investment strategy surface plot
using Plots
surface(prim.q_grid,prim.q_grid,res.x, title = "MPE investment strategy x(ω) plot", xlabel = "i", ylabel = "j", zlabel = "x(i,j)" , camera=(50,20))
savefig("MPE_surface.png")

# get industry transition matrix
get_Q(prim,res)

########################################################################
# Q4. Industry evolution simulation
# (See dynamic_games_function.jl for actual functions)
########################################################################
# simulate market strcture evolution for B=1,000 times
@time market_evol_simulator([1,1],prim,res)

# get industy state distribution and plot
ω_mat, ω_vec = get_industry_dist(prim,res,data)
surface(prim.q_grid,prim.q_grid,ω_mat/1000, title = "Industry state distribution at T=25", xlabel = "i", ylabel = "j", zlabel = "a(i,j)")
savefig("indsutry_dist_25.png")
histogram2d(ω_vec[:,1], ω_vec[:,2],title = "Industry state distribution (frequency) at T=25", xlabel = "i", ylabel = "j")
savefig("indsutry_hist_25.png")

########################################################################

