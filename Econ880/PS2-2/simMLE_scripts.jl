########################################################################
# Script for solving multivariate probit model
########################################################################
cd("/Users/yuichirotsuji/Documents/Econ880/PS2-2")
#import Pkg; Pkg.add("CSVFiles")
using Parameters, LinearAlgebra, Random, Distributions, Optim, Statistics, DataFrames, StatFiles, CSVFiles #import the libraries we want
include("SimMLE_functions.jl")  #import the functions from the provided file
prim,res = Initialize() # primitives initialization
data = get_data() # data initialization

########################################################################
# Q1. Log-liklihood evaluation with quadrature
# (See simMLE_functions.jl for actual functions)
########################################################################
# Log-liklihood evaluation with parameters in Q3
ll_quad = log_liklihood_quad(0.0,-1.0,-1.0,zeros(size(data.X,2)),0.3,0.5,data)

########################################################################
# Q2. Simulated Log-liklihood with accept/reject
# (See simMLE_functions.jl for actual functions)
########################################################################
# Log-liklihood evaluation with parameters in Q3
ll_sim = log_liklihood_sim(0.0,-1.0,-1.0,zeros(size(data.X,2)),0.3,0.5,data,prim,res)
#(Note: The value log-liklihood is -∞ due to the existance of zero simulated choice probability)

########################################################################
# Q3: Choice probabilities
########################################################################
# Get choice probabilities using quadrature
P_Quad = choice_prob_Quad(0.0,-1.0,-1.0,zeros(size(data.X,2)), 0.3,0.5, data)
using Plots
# draw histogram
histogram(P_Quad[:,1],title="Predicted choice probabilities using Quadrature",ylabel = "frequency", label = "Pr(Tᵢ=1)", xlabel = "Pr(Tᵢ=t)")
histogram!(P_Quad[:,2], label = "Pr(Tᵢ=2)")
histogram!(P_Quad[:,3], label = "Pr(Tᵢ=3)")
histogram!(P_Quad[:,4], label = "Pr(Tᵢ=4)")
savefig("choiceprob_quad_hist.png") #used in Q.3

# (Note: simulated choice probabilities are stored in res.P after compiling Q2)
# draw histogram
histogram(res.P[:,1],title="Preducted choice probabilities using Accept/Reject",ylabel = "frequency", label = "Pr(Tᵢ=1)", xlabel = "Pr(Tᵢ=t)")
histogram!(res.P[:,2], label = "Pr(Tᵢ=2)")
histogram!(res.P[:,3], label = "Pr(Tᵢ=3)")
histogram!(res.P[:,4], label = "Pr(Tᵢ=4)")
savefig("choiceprob_sim_hist.png") #used in Q.3

# scatter plot
scatter(P_Quad[:,1], res.P[:,1],label = "Pr(Tᵢ=1)",title="Predicted choice probabilities scatter plot",ylabel = "choice probabilities by simulation", xlabel = "choice probabilities by quadrature")
scatter!(P_Quad[:,2], res.P[:,2],label = "Pr(Tᵢ=2)")
scatter!(P_Quad[:,3], res.P[:,3],label = "Pr(Tᵢ=3)")
scatter!(P_Quad[:,4], res.P[:,4],label = "Pr(Tᵢ=4)")
plot!(range(0.0,stop=1.0,length=50),range(0.0,stop=1.0,length=50),label = "45-degree line")
savefig("choiceprob_scatterplot.png") #used in Q.3

########################################################################
# Q4. Log-liklihood evaluation with quadrature
########################################################################
# initiai guess = values in Q3
guess = [0.0,-1.0,-1.0,0.3,0.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

# minimize negative log-liklihood with BFGS
@time opt_bfgs = optimize(θ -> LLm(θ),guess,LBFGS(),Optim.Options(g_tol = 1e-2,iterations = 1000)) # about 300 minutes(!) for convergence
θ_hat = opt_bfgs.minimizer # used in Q4

########################################################################
#= (Below is the result I got in Nov 19, 2024)

17798.516699 seconds (10.72 G allocations: 32.743 TiB, 10.56% gc time, 0.00% compilation time)
 * Status: success

 * Candidate solution
    Final objective value:     1.137407e+04

 * Found with
    Algorithm:     L-BFGS

 * Convergence measures
    |x - x'|               = 4.00e-06 ≰ 0.0e+00
    |x - x'|/|x'|          = 7.03e-07 ≰ 0.0e+00
    |f(x) - f(x')|         = 1.05e-08 ≰ 0.0e+00
    |f(x) - f(x')|/|f(x')| = 9.19e-13 ≰ 0.0e+00
    |g(x)|                 = 7.24e-03 ≤ 1.0e-02

 * Work counters
    Seconds run:   17767  (vs limit Inf)
    Iterations:    185
    f(x) calls:    570
    ∇f(x) calls:   570

20-element Vector{Float64}:
 α₁: 5.690737238120582
 α₂: 3.17725999211734
 α₃: 2.577716860551023
 γ : -0.09796440748071818
 ρ : 0.49375984819261604
 β (15×1 vector) : 
  0.21770028717974138
 -0.32486992067685566
 -0.494240031978615
 -0.340630869317614
 -0.08174495914864287
 -0.20762274999939145
  0.06531456790648958
 -0.13185490478706158
 -0.5230682112433558
 -0.2602337751384156
 -0.3706425566511351
 -0.6195479620825963
 -0.16248446550053275
  0.07106576798685955
  0.08144006034590598

=#

