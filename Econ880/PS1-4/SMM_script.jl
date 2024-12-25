#######################################################################
#Initialize parameters and functions/distributions
#######################################################################
#cd("/Users/yuichirotsuji/Documents/Econ880/PS1-4")
using Distributed
#addprocs(2)
@everywhere using Random, Distributions, Parameters, Plots, SharedArrays, Optim #import the libraries we want
include("SMM_functions.jl") #import the functions that solve the model
@everywhere prim,res = Initialize()

#######################################################################
# Q2: Generate "true" data. 
# (Note: Run this part first!!!)
#######################################################################
True_data_initialize(prim, res) #generate "true" AR(1) data and iid shock for simulation - we'll use this in remaining parts
plot(res.x_true, title="True data",ylabel = "x(t)", label = "True AR(1)", xlabel = "time") #plot the "true" data
savefig("True_AR1.png") #used in Q2
res.M_T
#######################################################################
# Q3: Generate model-based data
#######################################################################
# will be included in next part

#######################################################################
# Q4: l=2 (mean & variance) case
# (Note:From now on, we specify the moments we use by "moments" argument in each functions.) 
#######################################################################
#(a) plot J_TH(ρ,σ)
res.W2 = SharedArray([1.0 0.0; 0.0 1.0]) #initialize weighting matrix to I2
@time J_val = J_TH_plot(prim, res, [true, true, false]) #generate valus of J_TH for plot (about 5 seconds)
surface(prim.σ_grid, prim.ρ_grid, J_val, title="SMM objective function plot (m1 & m2)", xlabel = "σ", ylabel = "ρ",zlabel = "J_TH", seriescolor=:viridis, camera = (20,25)) #3D surface plot of J_TH
savefig("J_TH_12.png") #used in Q4(a)

#(a)-(d):Solve for SMM
@time SMM_solver(prim,res,[true,true,false]) #takes about 2 seconds (most of them are for compilation)

#= Actually we do the following in the SMM_solver;
    #(a)get b1_hat
        b_hat_finder(prim, res, [true, true, false]) #find b^1_TH using W=I
    #(b)get W_star_hat and compute b2_hat 
        W_star_calculator(prim, res, [true,true,false]) #Update W with Optimal waighting matrix by Newey-West
        b_hat_finder(prim, res, [true, true, false]) #find b^2_TH using updated W
    #(c)numerical derivative, compute SE and AGS
        b_derivarives(prim,res,[true, true, false]) #numerical derivatives
        V_b_hat(prim,res,[true, true, false])
        AGS(prim,res,[true, true, false])
    #(d)calculate J-statistics
        J_test(prim, res, [true, true, false]) #conduct J-test
=#

#######################################################################
# Q5: l=2 (variance & autocorrelation) case
#######################################################################
#(a) plot J_TH(ρ,σ)
res.W2 = SharedArray([1.0 0.0; 0.0 1.0]) #initialize weighting matrix to I2
@time J_val = J_TH_plot(prim, res, [false, true, true]) #generate valus of J_TH for plot (2-3 seconds)
surface(prim.σ_grid, prim.ρ_grid, J_val, title="SMM objective function plot (m2 & m3)", xlabel = "σ", ylabel = "ρ",zlabel = "J_TH", seriescolor=:viridis, camera = (20,25)) #3D surface plot of J_TH
savefig("J_TH_23.png") #used in Q5(b)

#(a)-(d):Solve for SMM
@time SMM_solver(prim,res,[false, true, true]) #less than 1 sec if compilation is done in Q4

#######################################################################
# Q6: l=3 (mean & variance & autocorrelation) case
#######################################################################
#(a) plot J_TH(ρ,σ)
res.W3 = SharedArray([1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]) #initialize weighting matrix to I3
@time J_val = J_TH_plot(prim, res, [true, true, true]) #generate valus of J_TH for plot (2-3 seconds)
surface(prim.σ_grid, prim.ρ_grid, J_val, title="SMM objective function plot (m1 & m2 & m3)", xlabel = "σ", ylabel = "ρ",zlabel = "J_TH", seriescolor=:viridis, camera = (20,25)) #3D surface plot of J_TH
savefig("J_TH_123.png") #used in Q6(b)

#(a)-(d):Solve for SMM
@time SMM_solver(prim,res,[true, true, true])  #less than 1 sec if compilation is done in Q4

#(e) Bootstrap
@time ρ1, ρ2, σ1, σ2 = Bootstrap(prim,res) #takes about 1-2 min: 
histogram([ρ1, ρ2], fillalpha = 0.7, title="Bootstrap distribution of ρ_hat",label = ["ρ1" "ρ2"], bins=range(0.35,0.65,length=100),xlabel="Estimates of ρ", ylabel="Frequency") #histogram for ρ1 and ρ2
savefig("hist_Boot_ρ.png") #used in Q6(e)
histogram([σ1,σ2], fillalpha = 0.8, title="Bootstrap distribution of σ_hat", label = ["σ1" "σ2"], bins=range(0.8,1.2,length=100), xlabel="Estimates of σ", ylabel="Frequency") #histogram for σ1 and σ2
savefig("hist_Boot_σ.png") #used in Q6(e)


#######################################################################
# Q7: Indirect inderence
# (Note:From now on, we specify the oreders of MA(N) by "Order" argument in each functions.) 
####################################0###################################
#Intialize components in structs and "true" data and shocks for simulation
@everywhere prim,res = Initialize()
True_data_initialize(prim, res)

#Indirect inference for N=1,2,3
@time IND_solver(prim, res, 1) #Takes about 1 sec
@time IND_solver(prim, res, 2) #Takes about 2-3 sec
@time IND_solver(prim, res, 3) #Takes about 3-5 sec
#######################################################################
