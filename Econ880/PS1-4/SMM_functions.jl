########################################################################
# Initialization of the algorithm
########################################################################
#parameters struct at everywhere
@everywhere @with_kw mutable struct Primitives
    seed::Int64 = 1234 #seed for generating true data(shocks)
    seed_m::Int64 = 456 #seed for generating simulated data(shocks)
    ρ_o::Float64 = 0.5 #true coefficient parameter
    σ_o::Float64 = 1.0 #true variance parameter
    x_o::Float64 = 0.0 #initial value of true data
    T::Int64 = 200 #data length
    m::Int64 = 3 #number of (maximum) moments we use
    l::Int64 = 2 #number of parameters
    iT::Int64 = 4 #NW lag parameter
    H::Int64 = 10 #number of simulations
    Δ::Float64 = 1e-7 #small value for numerical derivatives
    B::Int64 = 1000 #number of bootstrap sampling

    #parameters for 3-D plot
    ρ_min::Float64 = 0.35 
    ρ_max::Float64 = 0.65
    ρ_grid::SharedVector{Float64} = SharedVector(collect(ρ_min:0.005:ρ_max))
    σ_min::Float64 = 0.8
    σ_max::Float64 = 1.2
    σ_grid::SharedVector{Float64} = SharedVector(collect(σ_min:0.005:σ_max))
end

#functions struct at everywhere
@everywhere @with_kw mutable struct Results
    x_true::SharedVector{Float64} #true data
    ϵ_true::SharedVector{Float64} #true shock vector
    y_b::SharedArray{Float64} #model-based data
    e_sim::SharedArray{Float64} #model shocks
    M_T::SharedVector{Float64} #data moments
    M_TH::SharedVector{Float64} #model moments

    b_TH::Vector{Float64} #estimates for b=(ρ, σ)
    W2::SharedArray{Float64} #2-demensional weighting matrix
    W3::SharedArray{Float64} #3-demensional weighting matrix
    ∇g2::SharedArray{Float64} #Jacobian matrix of moment estimators when m = 2
    ∇g3::SharedArray{Float64} #Jacobian matrix of moment estimators when m = 3
    V::SharedArray{Float64} #2*2 variance-covariance matrix of b_TH
end

#initialization function at everywhere
@everywhere function Initialize()
    prim = Primitives()
    ϵ_true = SharedVector(zeros(Float64, prim.T))
    x_true = SharedVector(zeros(Float64, prim.T))
    y_b = SharedArray(zeros(Float64, prim.T, prim.H))
    e_sim = SharedArray(zeros(Float64, prim.T, prim.H))
    M_T = SharedVector(zeros(Float64, prim.m)) 
    M_TH = SharedVector(zeros(Float64, prim.m))
    b_TH = SharedVector(zeros(Float64, prim.l))
    W2 = SharedArray([1.0 0.0;
                      0.0 1.0]) #2*2 identity matrix
    W3 = SharedArray([1.0 0.0 0.0;
                      0.0 1.0 0.0;
                      0.0 0.0 1.0]) #3*3 identity matrix
    ∇g2 = SharedArray(zeros(Float64, 2, prim.l))
    ∇g3 = SharedArray(zeros(Float64, 3, prim.l))
    V = SharedArray(zeros(Float64, 2, prim.l)) 
    res = Results(ϵ_true, x_true, y_b, e_sim, M_T, M_TH, b_TH, W2, W3, ∇g2, ∇g3, V)

    return prim, res
end

########################################################################
# "True" data generator
########################################################################
# true shocks (=ϵ_t) for true data
@everywhere function true_shock_generator(prim::Primitives)
    Random.seed!(prim.seed)
    dist = Normal(0, prim.σ_o)
    ϵ = rand(dist, prim.T)
    return ϵ
end

# generate true data using true shocks
@everywhere function true_data_generator(prim::Primitives, res::Results)
    x_true = Vector(zeros(Float64, prim.T))
    x_true[1] = prim.ρ_o*prim.x_o + res.ϵ_true[1]
    for t_index in 2:prim.T
        x_true[t_index] = prim.ρ_o*x_true[t_index - 1] + res.ϵ_true[t_index]
    end
    return x_true
end

#true data moment calculator
@everywhere function data_moment_calculator(prim::Primitives, res::Results)
    @unpack T = prim 
    @unpack x_true, M_T = res

    M_T[1] = sum(x_true)/T #first moment(mean)
    M_T[2] = sum((x_true .- M_T[1]).^2)/T #second moment(variance)

    #thrid moment(autocorrelation(1))
    γ = (x_true[1] - M_T[1])*(prim.x_o - M_T[1]) #initial value
    for t_index in 2:T
        γ += (x_true[t_index] - M_T[1])*(x_true[t_index - 1] - M_T[1])
    end
    M_T[3] = γ/(T*M_T[2])
end

# draw shocks (=e_t) for simulation
@everywhere function model_shock_generator(prim::Primitives)
    Random.seed!(prim.seed_m)
    dist = Normal(0, 1)
    return rand(dist, prim.T, prim.H)
end

# do all the above intial data generation process
@everywhere function True_data_initialize(prim::Primitives, res::Results)
    res.ϵ_true = true_shock_generator(prim)
    res.x_true = true_data_generator(prim,res)
    data_moment_calculator(prim,res)
    res.e_sim = model_shock_generator(prim)
end

########################################################################
# Model-based data simulation
########################################################################
#simulates T(=200) length data (y_b) for H(=10) times for given ρ and σ
@everywhere function model_data_generator(prim::Primitives, res::Results, ρ::Float64, σ::Float64)
    @unpack T, H, = prim 

    e = σ*res.e_sim #generate randpm shocks using e_t stored in res-struct 

    y_b = Array{Float64}(zeros(T, H)) #container for simulated values
    for h_index in 1:H #simulate data using the model data generation process
        y_b[1, h_index] = e[1, h_index]
        for t_index in 2:T
            y_b[t_index, h_index] = ρ*y_b[t_index - 1, h_index] + e[t_index, h_index]
        end
    end
    return y_b
end

#calculate model-based moments using y_b stored in res-struct
@everywhere function model_moment_calculator(prim::Primitives, res::Results)
    @unpack T, H = prim 
    @unpack y_b = res

    M_TH = Vector{Float64}(zeros(prim.m)) #container of moment values

    #first moment(mean)
    M_TH[1] = sum(y_b)/(T*H) 
    #second moment(variance) and third moment(autocorrelation(1))
    s = 0
    γ = 0
    for h_index in 1:H
        y_bar = sum(y_b[:,h_index])/T
        for t_index in 1:T
            s += (y_b[t_index, h_index] - y_bar)^2
        end
    
        for t_index in 2:T
            γ += (y_b[t_index, h_index] - y_bar)*(y_b[t_index - 1, h_index] - y_bar)
        end
    end
    M_TH[2] = s/(T*H)
    M_TH[3] = γ/s

    return M_TH
end

########################################################################
# Functions for SMM_solver
# (Note:From now on, we specify the moments we use by "moments" argument in each functions.) 
########################################################################
# calculate the value of the objective function J_TH(b) given data, parameters b=(ρ,σ) and model moments
@everywhere function J_TH_calculator(prim::Primitives, res::Results, b::Vector{Float64}, moments::Vector{Bool})
    @unpack M_T, M_TH = res

    res.y_b = model_data_generator(prim, res, b[1], b[2]) #y(b) is generated for given b=(ρ,σ) and stored in res-struct
    M_TH = model_moment_calculator(prim, res) #moments of y(b) is generated and stored in res-struct

    #get data and simulated moments vector(M_T and M_TH) using moment specification
    data_moment::Vector{Float64} = [] 
    model_moment::Vector{Float64} = []
    for moment_index in eachindex(moments)
        if moments[moment_index] == true
            data_moment = vcat(data_moment, M_T[moment_index])
            model_moment = vcat(model_moment, M_TH[moment_index])
        end
    end

    if length(data_moment) == 2
        return (data_moment - model_moment)'*res.W2*(data_moment - model_moment)
    elseif length(data_moment) == 3
        return (data_moment - model_moment)'*res.W3*(data_moment - model_moment)
    end

end

#calculate the value of the objective function J_TH(b) given data and model moments
@everywhere function J_TH_plot(prim::Primitives, res::Results, moments::Vector{Bool})
    @unpack ρ_grid, σ_grid = prim
    J_val::SharedArray{Float64} = zeros(Float64, length(ρ_grid), length(σ_grid))

    @sync @distributed for ρ_index in eachindex(ρ_grid)
        for σ_index in eachindex(σ_grid)
            J_val[ρ_index,σ_index] = J_TH_calculator(prim, res, [ρ_grid[ρ_index], σ_grid[σ_index]], moments)
        end
    end
    return J_val

end

#get b_hat using Optim.optimize function
@everywhere function b_hat_finder(prim::Primitives, res::Results, moments::Vector{Bool})
    J_TH_star = optimize(b -> J_TH_calculator(prim, res, b, moments),[0.5,1.0])
    res.b_TH = J_TH_star.minimizer #update the esimates for parameters in res-struct
    return res.b_TH
end

# calculate the estimator for the optimal weighting matrix
@everywhere function W_star_calculator(prim::Primitives, res::Results, moments::Vector{Bool})
    @unpack T, H, iT = prim
    @unpack y_b, b_TH, M_TH = res

    y_b = model_data_generator(prim, res, b_TH[1], b_TH[2]) #update y(b_TH) in res-struct using b1_TH
    M_TH = model_moment_calculator(prim, res) #update model moments using y(b1_TH)

    dim = count(x -> x == true, moments) #specify the number of moments condition we use

    model_moment::Vector{Float64} = []
    for moment_index in eachindex(moments)
        if moments[moment_index] == true
            model_moment = vcat(model_moment, M_TH[moment_index])
        end
    end

    #loop for constucting the estimates of optimal weighting matrix
    S_y_TH = zeros(dim, dim) # container for W_star_hat
    #moment-specific computation of W_star_hat
    if moments[1] == true && moments[2] == true && moments[3] == false #Q4(mean & variance)
        Γ_zero = zeros(dim,dim) #compute Γ_0TH first
        for h_index in 1:H
            y_bar = sum(y_b[:,h_index])/T
            for t_index in 1:T
                model_val_zero = [y_b[t_index, h_index], (y_b[t_index, h_index] - y_bar)^2]
                Γ_zero += (model_val_zero - model_moment)*(model_val_zero - model_moment)'
            end
        end
        S_y_TH += (1/(T*H))*Γ_zero

        for j_index in 1:iT #compute Γ_j,TH
            Γ_jTH = zeros(dim,dim)
            for h_index in 1:H
                y_bar = sum(y_b[:,h_index])/T
                for t_index in j_index+1:T
                    model_val= [y_b[t_index, h_index], (y_b[t_index, h_index] - y_bar)^2]
                    lagged_val = [y_b[t_index-j_index, h_index], (y_b[t_index-j_index, h_index] - y_bar)^2]
                    Γ_jTH += (model_val - model_moment)*(lagged_val - model_moment)'
                end
            end
            Γ_sum = (1/(H*T))*Γ_jTH
            S_y_TH += (1-(j_index/(iT+1)))*(Γ_sum + Γ_sum')
        end        
        res.W2 = ((1+(1/H))*S_y_TH)^(-1) #update 2*2 weighting matrix in res-struct
        return res.W2

    elseif moments[1] == false && moments[2] == true && moments[3] == true #Q5(variance & autocorrelation)
        Γ_zero = zeros(dim,dim) #compute Γ_0TH first
        for h_index in 1:H
            y_bar = sum(y_b[:,h_index])/T
            var_y = sum((y_b[:,h_index] .- y_bar).^2)/T
            
            for t_index in 2:T
                model_val_zero = [(y_b[t_index, h_index] - y_bar)^2, (y_b[t_index, h_index] - y_bar)*(y_b[t_index - 1, h_index] - y_bar)/var_y]
                Γ_zero += (model_val_zero - model_moment)*(model_val_zero - model_moment)'
            end
        end
        S_y_TH += (1/(T*H))*Γ_zero

        for j_index in 1:iT
            Γ_jTH = zeros(dim,dim)
            for h_index in 1:H
                y_bar = sum(y_b[:,h_index])/T
                var_y = sum((y_b[:,h_index] .- y_bar).^2)/T
                for t_index in j_index+2:T
                    model_val= [(y_b[t_index, h_index] - y_bar)^2, (y_b[t_index, h_index] - y_bar)*(y_b[t_index - 1, h_index] - y_bar)/var_y]
                    lagged_val = [(y_b[t_index-j_index, h_index] - y_bar)^2, (y_b[t_index-j_index, h_index] - y_bar)*(y_b[t_index-j_index-1, h_index] - y_bar)/var_y]
                    Γ_jTH += (model_val - model_moment)*(lagged_val - model_moment)'
                end
            end
            Γ_sum = (1/(H*T))*Γ_jTH
            S_y_TH += (1-(j_index/(iT+1)))*(Γ_sum + Γ_sum')
        end        

        res.W2 = ((1+(1/H))*S_y_TH)^(-1) #update 2*2 weighting matrix in res-struct
        return res.W2

    elseif dim == 3 #Q6(mean & variance & autocorrelation)
        Γ_zero = zeros(dim,dim) #compute Γ_0TH first
        for h_index in 1:H
            y_bar = sum(y_b[:,h_index])/T
            var_y = sum((y_b[:,h_index] .- y_bar).^2)/T          
            for t_index in 2:T
                model_val_zero = [y_b[t_index, h_index],(y_b[t_index, h_index] - y_bar)^2, (y_b[t_index, h_index] - y_bar)*(y_b[t_index - 1, h_index] - y_bar)/var_y]
                Γ_zero += (model_val_zero - model_moment)*(model_val_zero - model_moment)'
            end
        end
        S_y_TH += (1/(T*H))*Γ_zero
        
        for j_index in 1:iT
            Γ_sum = zeros(dim,dim)
            for h_index in 1:H
                y_bar = sum(y_b[:,h_index])/T
                var_y = sum((y_b[:,h_index] .- y_bar).^2)/T
                for t_index in j_index+2:T
                    model_val= [y_b[t_index, h_index], (y_b[t_index, h_index] - y_bar)^2, (y_b[t_index, h_index] - y_bar)*(y_b[t_index - 1, h_index] - y_bar)/var_y]
                    lagged_val = [y_b[t_index-j_index, h_index], (y_b[t_index-j_index, h_index] - y_bar)^2, (y_b[t_index-j_index, h_index] - y_bar)*(y_b[t_index-j_index- 1, h_index] - y_bar)/var_y]
                    Γ_sum += (model_val - model_moment)*(lagged_val - model_moment)'
                end
            end
            Γ_sum = (1/(H*T))*Γ_sum
            S_y_TH += (1-(j_index/(iT+1)))*(Γ_sum + Γ_sum')
        end
        res.W3 = ((1+(1/H))*S_y_TH)^(-1) #update 3*3 weighting matrix in res-struct
        return res.W3
    end    
end

# taking numerical derivetives (w.r.t. b_TH)
@everywhere function b_derivarives(prim::Primitives, res::Results, moments::Vector{Bool})
    @unpack Δ = prim
    @unpack y_b, b_TH, M_TH, ∇g2, ∇g3 = res

    y_b = model_data_generator(prim, res, b_TH[1], b_TH[2]) #update y(b_TH) in res-struct using b2_TH
    M_TH = model_moment_calculator(prim, res) #update model moments using y(b2_TH)

    #generating values for numerical derivatives
    diff_ρ_data::Array{Float64} =  model_data_generator(prim, res, b_TH[1]-Δ, b_TH[2])
    diff_ρ_moment::Vector{Float64} = diff_moment_calculator(prim, res, diff_ρ_data)
    diff_σ_data::Array{Float64} =  model_data_generator(prim, res, b_TH[1], b_TH[2]-Δ)
    diff_σ_moment::Vector{Float64} = diff_moment_calculator(prim, res, diff_σ_data)

    model_moment::Vector{Float64} = []
    model_moment_Δρ::Vector{Float64} = []
    model_moment_Δσ::Vector{Float64} = []
    for moment_index in eachindex(moments)
        if moments[moment_index] == true
            model_moment = vcat(model_moment, M_TH[moment_index])
            model_moment_Δρ = vcat(model_moment_Δρ, diff_ρ_moment[moment_index])
            model_moment_Δσ = vcat(model_moment_Δσ, diff_σ_moment[moment_index])
        end
    end

    if length(model_moment) == 2 #return numerical derivetives for 3D
        ∇g2[:,1] = (model_moment - model_moment_Δρ)/Δ
        ∇g2[:,2] = (model_moment - model_moment_Δσ)/Δ
        return ∇g2
    elseif length(model_moment) == 3 #return numerical derivetives for 3D
        ∇g3[:,1] = (model_moment - model_moment_Δρ)/Δ
        ∇g3[:,2] = (model_moment - model_moment_Δσ)/Δ
        return ∇g3
    end    
end

# a function to generate moments for numerical derivatives
@everywhere function diff_moment_calculator(prim::Primitives, res::Results, y_b::Array{Float64})
    @unpack T, H = prim 
    
    M_TH = Vector{Float64}(zeros(prim.m)) #container of moment values
    
    #first moment(mean)
    M_TH[1] = sum(y_b)/(T*H) 
    #second moment(variance) and third moment(autocorrelation(1))
    s = 0
    γ = 0
    for h_index in 1:H
        y_bar = sum(y_b[:,h_index])/T
        for t_index in 1:T
            s += (y_b[t_index, h_index] - y_bar)^2
        end
        
        for t_index in 2:T
            γ += (y_b[t_index, h_index] - y_bar)*(y_b[t_index - 1, h_index] - y_bar)
        end
    end
    M_TH[2] = s/(T*H)
    M_TH[3] = γ/s    
    return M_TH 
end

# Computing asymptotic variance-covariance matrix using numerical derivatives
@everywhere function V_b_hat(prim::Primitives, res::Results, moments::Vector{Bool})
    @unpack W2, W3, ∇g2, ∇g3 = res
    dim = count(x -> x == true, moments) #specify the number of moments condition we use

    if dim == 2 #use 2-D Jacobian(numerical deribatives) and weighting matrix
        res.V = (1/prim.T)*inv((∇g2)'*W2*∇g2)
    elseif dim == 3 #use 3-D Jacobian(numerical deribatives) and weighting matrix
        res.V = (1/prim.T)*inv((∇g3)'*W3*∇g3)
    end
    println("se(ρ_hat) = ", sqrt(res.V[1,1]), ", and se(σ_hat) = " , sqrt(res.V[2,2]), ".")
end

# Computing small sample AGS statistics
@everywhere function AGS(prim::Primitives, res::Results, moments::Vector{Bool})
    @unpack ∇g2, ∇g3 = res
    dim = count(x -> x == true, moments) #specify the number of moments condition we use

    if dim == 2 #use 2-D Jacobian(numerical deribatives) and identity matrix
        AGS = -inv((∇g2)'*[1.0 0.0; 0.0 1.0]*(∇g2))*(∇g2)'*[1.0 0.0; 0.0 1.0]
    elseif dim == 3 #use 2-D Jacobian(numerical deribatives) and identity matrix
        AGS = -inv((∇g3)'*[1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]*(∇g3))*(∇g3)'*[1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    end
    println("The small sample AGS statistics is ", AGS, ".")
end

# Computing J-statistics
@everywhere function J_test(prim::Primitives, res::Results, moments::Vector{Bool})
    @unpack T, H = prim
    J_TH = J_TH_calculator(prim, res, res.b_TH, moments) #culculate the value of objective function, using b^2_TH and W^*
    println("The value of J-statistics is " ,T*(H/(1+H))*J_TH, ".")
end

#Including all procedure for SMM
@everywhere function SMM_solver(prim::Primitives, res::Results, moments::Vector{Bool})
    println("****************************")
    dim = count(x -> x == true, moments) #specify the number of moments condition we use

    # initialize weighting matrices
    if dim == 2
        res.W2 = SharedArray([1.0 0.0;
                              0.0 1.0]) 
    elseif dim == 3
        res.W3 = SharedArray([1.0 0.0 0.0;
                              0.0 1.0 0.0;
                              0.0 0.0 1.0])
    end

    #(a)Find b1_hat using a solver
    b_hat_finder(prim, res, moments)
    println("b1_hat_TH = ", res.b_TH, ".")

    #(b) get W_star_hat
    W_star_calculator(prim, res, moments)
    if dim == 2
        println("The estimates for the optimal weighting matrix is ", res.W2, ".")
    elseif dim == 3
        println("The estimates for the optimal weighting matrix is ", res.W3, ".")
    end
    b_hat_finder(prim, res, moments)
    println("b2_hat_TH = ", res.b_TH, ".")

    #(c) numerical derivative, se and AGS
    b_derivarives(prim,res,moments)
    if dim == 2
        println("The numerical Jacobian matrix; ∇g = ", res.∇g2, ".")
    elseif dim == 3
        println("The numerical Jacobian matrix; ∇g =  ", res.∇g3, ".")
    end
    V_b_hat(prim, res, moments)
    AGS(prim, res, moments)

    #(d) calculate J-statistics
    J_test(prim, res, moments)

    println("****************************") #All are done!
end

########################################################################
# Function for Bootstrap
#######################################################################
@everywhere function Bootstrap(prim,res)
    ρ1_hat::SharedVector{Float64} = zeros(Float64, prim.B)
    ρ2_hat::SharedVector{Float64} = zeros(Float64, prim.B)
    σ1_hat::SharedVector{Float64} = zeros(Float64, prim.B)
    σ2_hat::SharedVector{Float64} = zeros(Float64, prim.B)

    for B_index in 1:prim.B
        #draw ϵ_t and generate "true" data
        Random.seed!()
        dist = Normal(0, prim.σ_o)
        res.ϵ_true = rand(dist, prim.T)
        res.x_true = true_data_generator(prim,res)
        data_moment_calculator(prim,res)

        #draw e_t^h and simulate shock
        Random.seed!()
        dist = Normal(0, 1)
        res.e_sim = rand(dist, prim.T, prim.H)  

        #use the procedure we made above to find b1_TH and b2_TH
        b_hat_finder(prim::Primitives, res::Results, [true,true,true])
        ρ1_hat[B_index] = res.b_TH[1]
        σ1_hat[B_index] = res.b_TH[2]
        W_star_calculator(prim, res, [true,true,true]) 
        b_hat_finder(prim, res, [true, true, true]) 
        ρ2_hat[B_index] = res.b_TH[1]
        σ2_hat[B_index] = res.b_TH[2]
    end

    #print bootstrap estimates (= average of B times simulations)
    println("ρ1_bar = ", (1/prim.B)*sum(ρ1_hat), "ρ2_bar = ", (1/prim.B)*sum(ρ2_hat))
    println("σ1_bar = ", (1/prim.B)*sum(σ1_hat), "σ2_bar = ", (1/prim.B)*sum(σ2_hat))
    #return vectors of simulated estimates
    return ρ1_hat, ρ2_hat, σ1_hat, σ2_hat
end

########################################################################
########################################################################
# Functions for indirect inference
# (Note:From now on, we specify the oreders of MA(N) by "Order" argument in each functions.) 
########################################################################
# generating MA(N) model from "true" data
@everywhere function true_MA_generator(prim::Primitives, res::Results, θ::Vector{Float64}, Order::Int64)
    x_ma = zeros(Float64, prim.T) # MA(1) process container

    if Order  == 1 #MA(1)
        x_ma[1] = res.ϵ_true[1]
        for t_index in 2:prim.T
            x_ma[t_index] = res.ϵ_true[t_index] + (θ[1] * res.ϵ_true[t_index- 1])  
        end

    elseif Order == 2 #MA(2)
        x_ma[1] = res.ϵ_true[1]
        x_ma[2] = res.ϵ_true[2] +  (θ[1] * res.ϵ_true[1])
        for t_index in 3:prim.T
            x_ma[t_index] = res.ϵ_true[t_index] + (θ[1] * res.ϵ_true[t_index- 1]) + (θ[2] * res.ϵ_true[t_index- 2])
        end
    elseif Order == 3 #MA(3)
        x_ma[1] = res.ϵ_true[1]
        x_ma[2] = res.ϵ_true[2] +  (θ[1] * res.ϵ_true[1])
        x_ma[3] = res.ϵ_true[3] +  (θ[1] * res.ϵ_true[2]) +  (θ[2] * res.ϵ_true[1])
        for t_index in 4:prim.T
            x_ma[t_index] = res.ϵ_true[t_index] + (θ[1] * res.ϵ_true[t_index- 1]) + (θ[2] * res.ϵ_true[t_index- 2])+ (θ[3] * res.ϵ_true[t_index- 3])
        end
    end

    return x_ma
end

# calculating difference between true AR(1) and MA(N) (objective function to be minimized)
@everywhere function true_MA_SE(prim::Primitives, res::Results, θ::Vector{Float64}, Order::Int64)
    x_ma = true_MA_generator(prim, res, θ, Order)
    return sum((x_ma .- res.x_true).^2)
end

# estimating MA(N) parameters from "true" data
@everywhere function true_MA_estimator(prim::Primitives, res::Results, Order::Int64)
    if Order == 1 #MA(1)
        MA_mindist = optimize(θ -> true_MA_SE(prim, res, θ, Order),[0.5]) 
    elseif Order == 2 #MA(2)
        MA_mindist = optimize(θ -> true_MA_SE(prim, res, θ, Order),[0.5,0.25]) 
    elseif Order == 3 #MA(3)
        MA_mindist = optimize(θ -> true_MA_SE(prim, res, θ, Order),[0.5,0.25,0.1125])
    end

    θ_hat = MA_mindist.minimizer #get estimates for θ_hat

    #get estimates for s_hat
    x_ma_hat = true_MA_generator(prim, res, θ_hat, Order)
    s_hat = sqrt((1/(prim.T-1))*sum((x_ma_hat .- (sum(x_ma_hat)/prim.T)).^2))

    return θ_hat, s_hat
end

# simulate MA(N) data with guess of b = (ρ,σ)
@everywhere function MA_simulator(prim::Primitives, res::Results, h::Int64, θ::Vector{Float64}, σ::Float64, Order::Int64)
    @unpack T, H, = prim 
    @unpack e_sim, y_b = res

    x_ma = zeros(Float64, prim.T) # MA(1) process container
    e = σ * e_sim

    if Order == 1
        x_ma[1] = e[1, h]
        for t_index in 2:prim.T
            x_ma[t_index] = e[t_index, h] + (θ[1] * e[t_index- 1, h])
        end
    elseif Order == 2
        x_ma[1] = e[1, h]
        x_ma[2] = e[2, h] + (θ[1] * e[1, h])
        for t_index in 3:prim.T
            x_ma[t_index] = e[t_index, h] + (θ[1] * e[t_index- 1, h]) + (θ[2] * e[t_index- 2, h])
        end
    elseif Order == 3
        x_ma[1] = e[1, h]
        x_ma[2] = e[2, h] + (θ[1] * e[1, h])
        x_ma[3] = e[3, h] + (θ[1] * e[2, h]) + (θ[2] * e[1, h])
        for t_index in 4:prim.T
            x_ma[t_index] = e[t_index, h] + (θ[1] * e[t_index- 1, h]) + (θ[2] * e[t_index- 2, h]) + (θ[3] * e[t_index- 3, h])
        end
    end

    return x_ma
end

# calculating difference between simulated AR(1) and MA(N) (objective function to be minimized)
@everywhere function sim_MA_SE(prim::Primitives, res::Results, h::Int64, θ::Vector{Float64}, σ::Float64, Order::Int64)
    x_ma = MA_simulator(prim, res, h, θ, σ, Order)
    return sum((x_ma .- res.y_b[:,h]).^2)
end

# estimating MA(N) parameters from simulated data
@everywhere function sim_MA_estimator(prim::Primitives, res::Results, σ::Float64, Order::Int64)
    θ_hat = Vector{Vector{Float64}}(undef, prim.H)
    s_hat = zeros(Float64, prim.H)

    for h_index in 1:prim.H #get simulated estimates using Order specifiction
        if Order == 1
            MA_mindist = optimize(θ -> sim_MA_SE(prim, res, h_index, θ, σ, Order),[0.5]) 
        elseif Order == 2
            MA_mindist = optimize(θ -> sim_MA_SE(prim, res, h_index, θ, σ, Order),[0.5,0.25]) 
        elseif Order == 3
            MA_mindist = optimize(θ -> sim_MA_SE(prim, res, h_index, θ, σ, Order),[0.5,0.25,0.1125]) 
        end
        θ_hat[h_index] = MA_mindist.minimizer
        x_ma_hat = MA_simulator(prim, res, h_index, θ_hat[h_index], σ, Order)
        s_hat[h_index] = sqrt((1/(prim.T-1))*sum((x_ma_hat .- (sum(x_ma_hat)/prim.T)).^2))
    end
    θ_bar = (1/prim.H)*sum(θ_hat) #get θ^bar from simulation results
    s_bar = (1/prim.H)*sum(s_hat) #get s^bar from simulation results

    return θ_bar, s_bar
end

# the objective function for Indirect Inference (as in 7.v.)
@everywhere function IND_finder(prim::Primitives, res::Results, b::Vector{Float64}, Order::Int64)
    res.y_b = model_data_generator(prim,res,b[1],b[2]) #Given b=(ρ,σ), simulate AR(1)
    θ_hat, s_hat = true_MA_estimator(prim,res,Order) #get model-based estimates
    θ_bar, s_bar = sim_MA_estimator(prim,res,b[2],Order) #get simulated estimates
    return (θ_hat - θ_bar)'*(θ_hat - θ_bar) + (s_bar - s_hat)^2 #return the value of IND objective function
end

# Solve the entire estimation procedure, with specification of N ("Order" in the argument)
@everywhere function IND_solver(prim::Primitives, res::Results, Order::Int64)
    θ_hat, s_hat = true_MA_estimator(prim,res,Order)
    println("Data moments for N = ", Order, " are θ_hat = ", θ_hat, ", and s_hat = ", s_hat, ".")

    opt = optimize(b -> IND_finder(prim, res, b, Order),[0.4,1.1])
    b_hat = opt.minimizer
    println("Estimates of model parameters for N = ", Order, " are ρ = ", b_hat[1], ", and σ = ", b_hat[2], ".")
end