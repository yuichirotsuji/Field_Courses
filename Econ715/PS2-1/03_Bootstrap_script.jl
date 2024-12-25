########################################################################
# Script for Bootstrap
########################################################################
#cd("/Users/yuichirotsuji/Documents/Econ715/PS2-1")
using Parameters, LinearAlgebra, Random, Distributions, Optim, Statistics, DataFrames, StatFiles, Plots #import the libraries we want
include("02_Bootstrap_functions.jl")  # import the functions from the provided file
prim,res = Initialize()     # parameters and result initialization
data = get_data()           # data initialization

#######################################################################
# Q1: OLS with n=100 and asymptotic analysis
# (Please see "Bootstrap_functions.jl" for actual functions.)
########################################################################
# use first 100 observations
Y₁ = data.Y[1:prim.r]
X₁ = data.X[1:prim.r,:]

##(a) OLS estimation
@time β₁ = OLS(Y₁,X₁)

##(b) Homoskedastic SE and asymptotic CI
V₀ = get_V_hat(Y₁, X₁)
SE₀ = sqrt.(diag(V₀))
get_CI(Y₁, X₁, V₀, prim, true)

##(c) Heteroskedastic SE and asymptotic CI
V₁ = get_V_tilde(Y₁, X₁)
SE₁ = sqrt.(diag(V₁))
get_CI(Y₁, X₁, V₁, prim, true)

#######################################################################
# Q2: Bootstrap estimation
# (Please see "Bootstrap_functions.jl" for actual functions.)
########################################################################
Random.seed!(1234)
##(a) Nonparametric Bootstrap
@time nonparametric_Bootstrap_CI(Y₁, X₁, prim, true)
##(b) Residual Bootstrap
@time residual_Bootstrap_CI(Y₁, X₁, prim, true)
##(c) Parametric Bootstrap
@time parametric_Bootstrap_CI(Y₁, X₁, prim, true)

#######################################################################
# Q3: Coverage percentage
# (Please see "Bootstrap_functions.jl" for actual functions.)
########################################################################
prim,res = Initialize()
@time coverage_comparison(prim,res,data) # about 5 mins

#######################################################################
# Q4: Coverage percentage with different settings
# (Please see "Bootstrap_functions.jl" for actual functions.)
########################################################################
# Case I: B = 100
prim,res = Initialize()
prim.B = 100
@time coverage_comparison(prim,res,data) # about 30 sec

# Case II: (sample size) = 200
prim,res = Initialize()
prim.r = 200
@time coverage_comparison(prim,res,data) # about 8 mins

# Case III: B = 100, (sample size) = 200
prim,res = Initialize()
prim.B = 100
prim.r = 200
@time coverage_comparison(prim,res,data)  # about 1 min

########################################################################
