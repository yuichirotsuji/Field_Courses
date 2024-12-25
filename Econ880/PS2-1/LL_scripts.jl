########################################################################
# Script for solving single choice model
########################################################################
#cd("/Users/yuichirotsuji/Documents/Econ880/PS2-1")
using Distributed
@everywhere using Parameters, LinearAlgebra, Random, Interpolations, Optim, Statistics #import the libraries we want
include("LL_functions.jl")  #import the functions from the provided file

########################################################################

########################################################################
# Q1. Evaluation
# (See LL_functions.jl for actual functions)
########################################################################
#import Pkg; Pkg.add("StatFiles")
using DataFrames, StatFiles
df = DataFrame(load("Mortgage_performance_data.dta")) # load data from dta file
Y = Float64.(df.i_close_first_year)                   # create Y 
X = select(df, [:i_large_loan, :i_medium_loan, :rate_spread, :i_refinance, :age_r, :cltv, :dti, :cu, :first_mort_r, :score_0, :score_1, :i_FHA, :i_open_year2, :i_open_year3, :i_open_year4, :i_open_year5]) #create x
X = identity.(Array(X))

# evaluates functions with β₀=-1.0 and β=0
β₀ = -1.0
β = zeros(Float64, size(X,2))
println("The value of log-likelihood is ", log_liklihood(β₀, β, Y, X), ".") #log-liklihood
println("The value of Score is ",score(β₀, β, Y, X), ".")                   #score
println("The value of Hessian is ",hessian(β₀, β,  X), ".")                 #Hessian

########################################################################
# Q2. Numerical derivarives
# (See LL_functions.jl for actual functions)
########################################################################
# Numerical first derivative
@time first_derivatives(β₀, β, Y, X, 1e-6)  # less than 1 sec
# Numerical second derivative
@time second_derivatives(β₀, β, Y, X, 1e-3) # 2-3 sec

########################################################################
# Q3. Newton's method
# (See LL_functions.jl for actual functions)
########################################################################
guess = vcat(β₀, β)   # initial guess = values used in Q1
@time Newton(β₀, β, Y, X, guess) #converged in less than 1 sec

# We also used the Optim.optimize package and got the same result
@time opt_Newton = optimize(LL, Score, Hessian, guess) #converged in 1 sec
opt_Newton.minimizer

########################################################################
# Q4. BFGS and Simplex(Nelder-Mead)
########################################################################
# BFGS method
@time opt_bfgs = optimize(LL,guess,BFGS()) # about 120 seconde to converge
opt_bfgs.minimizer # used in Q4

# Simplex method (Note: this is the default method in Julia)
@time opt_simplex = optimize(LL,guess,Optim.Options(iterations=10000)) # about 15 seconds to converge
opt_simplex.minimizer # used in Q4

########################################################################
