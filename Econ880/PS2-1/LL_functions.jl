########################################################################
# Q1: Functions to compute l(β), J, H
########################################################################
# a function to compute Prob(Y=1|X)
function choice_prob(β₀::Float64, β::Vector{Float64}, X::Vector{Float64})
    return exp(β₀ + X'*β)/(1 + exp(β₀ + X'*β))
end

# compute log-liklihood
function log_liklihood(β₀::Float64, β::Vector{Float64}, Y::Vector{Float64}, X::Matrix{Float64})
    L = 0 # log-liklihood is scalar-valued
    for i_index in 1:length(Y) # We get each indiviudal's variables and take sum
        L += Y[i_index]*log(choice_prob(β₀, β, X[i_index,:])) + (1.0-Y[i_index])*log(1-choice_prob(β₀, β, X[i_index,:]))
    end
    return L
end

# compute Score
function score(β₀::Float64, β::Vector{Float64}, Y::Vector{Float64}, X::Matrix{Float64})
    J::Vector{Float64} = zeros(size(X,2)+1) # score is a (k×1) vector
    for i_index in 1:length(Y) # We get each indiviudal's variables and take sum
        J += (Y[i_index]-choice_prob(β₀, β, X[i_index,:]))*vcat(1,X[i_index,:])
    end
    return J
end

# compute Hessian
function hessian(β₀::Float64, β::Vector{Float64}, X::Matrix{Float64})
    H::Matrix{Float64} = zeros(size(X,2)+1,size(X,2)+1) # Hessian is a (k×k) matrix
    for i_index in 1:length(Y)  # We get each indiviudal's variables and take sum
        H -= (choice_prob(β₀, β, X[i_index,:]))*(1-choice_prob(β₀, β, X[i_index,:]))*vcat(1,X[i_index,:])*(vcat(1,X[i_index,:]))'
    end
    return H
end

########################################################################
# Q2: Numerical derivative
########################################################################
# a function to take numerical first derivatives
@everywhere function first_derivatives(β₀::Float64, β::Vector{Float64}, Y::Vector{Float64}, X::Matrix{Float64}, Δ::Float64)
    ∇f::Vector{Float64} = zeros(size(X,2)+1)
    ∇f[1] = (log_liklihood(β₀+Δ, β, Y, X) - log_liklihood(β₀, β, Y, X))/Δ

    for i_index in 1:size(X,2)
        Δβ::Vector{Float64} = copy(β) # make a copy of β
        Δβ[i_index] += Δ              # increase ith element by Δ
        ∇f[i_index+1] = (log_liklihood(β₀, Δβ, Y, X) - log_liklihood(β₀, β, Y, X))/Δ #taking numerical partial derivatives for ith element
    end

    return ∇f
end

# a function to take numerical second derivatives
@everywhere function second_derivatives(β₀::Float64, β::Vector{Float64}, Y::Vector{Float64}, X::Matrix{Float64}, Δ::Float64)
    Hf::Matrix{Float64} = zeros(size(X,2)+1,size(X,2)+1)

    # first row and column: we have to treat β₀ separately
    Hf[1,1] = (log_liklihood(β₀+Δ, β, Y, X)-(2*log_liklihood(β₀, β, Y, X))+log_liklihood(β₀-Δ, β, Y, X))/Δ^2
    for i_index in 1:size(X,2)
        Δβ₊::Vector{Float64} = copy(β)
        Δβ₊[i_index] += Δ
        Δβ₋::Vector{Float64} = copy(β)
        Δβ₋[i_index] -= Δ
        Hf[1, i_index+1] = (log_liklihood(β₀+Δ, Δβ₊, Y, X)-log_liklihood(β₀+Δ, Δβ₋, Y, X)-log_liklihood(β₀-Δ, Δβ₊, Y, X)+log_liklihood(β₀-Δ, Δβ₋, Y, X))/(4*Δ^2)
        Hf[i_index+1, 1] = (log_liklihood(β₀+Δ, Δβ₊, Y, X)-log_liklihood(β₀-Δ, Δβ₊, Y, X)-log_liklihood(β₀+Δ, Δβ₋, Y, X)+log_liklihood(β₀-Δ, Δβ₋, Y, X))/(4*Δ^2)
    end

    # the other rows and columns: simple loop
    for i_index in 1:size(X,2)
        for j_index in 1:size(X,2)             
            if i_index == j_index #own-second derivatives
                Δβ₊::Vector{Float64} = copy(β)
                Δβ₊[i_index] += Δ
                Δβ₋::Vector{Float64} = copy(β)
                Δβ₋[i_index] -= Δ
                Hf[i_index+1, j_index+1] = (log_liklihood(β₀, Δβ₊, Y, X)-(2*log_liklihood(β₀, β, Y, X))+log_liklihood(β₀, Δβ₋, Y, X))/Δ^2
            else #cross- second derivatives
                # + and + term
                Δβ₊₊::Vector{Float64} = copy(β)
                Δβ₊₊[i_index] += Δ
                Δβ₊₊[j_index] += Δ
                # + and - term
                Δβ₊₋::Vector{Float64} = copy(β)
                Δβ₊₋[i_index] += Δ
                Δβ₊₋[j_index] -= Δ
                # + and - term
                Δβ₋₊::Vector{Float64} = copy(β)
                Δβ₋₊[i_index] -= Δ
                Δβ₋₊[j_index] += Δ
                # - and - term
                Δβ₋₋::Vector{Float64} = copy(β)
                Δβ₋₋[i_index] -= Δ
                Δβ₋₋[j_index] -= Δ

                Hf[i_index+1, j_index+1] = (log_liklihood(β₀, Δβ₊₊, Y, X)-log_liklihood(β₀, Δβ₋₊, Y, X)-log_liklihood(β₀, Δβ₊₋, Y, X)+log_liklihood(β₀, Δβ₋₋, Y, X))/(4*Δ^2)
            end
        end
    end
    return Hf
end

########################################################################
# Q3: Newton's method
########################################################################
function Newton(β₀::Float64, β::Vector{Float64}, Y::Vector{Float64}, X::Matrix{Float64}, guess::Vector{Float64})
    β_guess = copy(guess) # initial guess: the values used in Q1
    tol = 1e-12           # tolerance level: used 1e-12, as in the lecture slides
    err = 100             # initial error: some big number
    iter = 0              # iteration counter

    while err > tol && iter < 10000
        β_next = β_guess - inv(hessian(β_guess[1],β_guess[2:end],X))*score(β_guess[1],β_guess[2:end],Y,X) # calculate next guess
        err = (sum((β_next - β_guess).^2))^(1/2) # error = Euclidean norm
        β_guess = β_next                         # undate the guess of β
        iter += 1
    end

    println("The Newton's algorithm converged in ", iter, " iterations.")
    return β_guess

end

# functions for using Newton method with Optim.optimize package
function LL(b::Vector{Float64})
    return -log_liklihood(b[1], b[2:end], Y, X) #objective function = -(LL)
end

function Score(S::Array{Float64,1}, b::Vector{Float64})
    S .= -score(b[1], b[2:end], Y, X) # negative score for minimizing -(LL)
    return S
end

function Hessian(H::Array{Float64,2}, b::Vector{Float64})
    H .= -hessian(b[1], b[2:end], X) # negative hessian for minimizing -(LL)
    return H
end

########################################################################
# Q4: BFGS and Simplex
########################################################################
# we used the Optim.optimize package in Julia.
# (See LL_script.jl for the actual implementation)

########################################################################
