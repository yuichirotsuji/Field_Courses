#= 
This code is a parallelized version of the VFI code for the neoclassical growth model.
The main difference is that the Bellman operator is parallelized using the @distributed macro.
September 2024
=#
#=
This is a stochastic version of parallelization functions.
=#

#parameters struct at everywhere
@everywhere @with_kw struct Primitives
    β::Float64 = 0.99
    δ::Float64 = 0.025
    α::Float64 = 0.36
    k_min::Float64 = 0.01
    k_max::Float64 = 90.0
    nk::Int64 = 1000
    k_grid::SharedVector{Float64} = SharedVector(collect(range(start=k_min, stop=k_max, length=nk)))

    #primitives for stochastic version
    nz::Int64 = 2 #number of state gird points (good and bad)
    z_grid::SharedVector{Float64} = SharedVector([1.25, 0.2]) #state grid
    Π::SharedArray{Float64, 2} = SharedArray([0.977 0.023; 0.074 0.926]) #transition matrix
end

#functions struct at everywhere
@everywhere @with_kw mutable struct Results
    val_func::SharedArray{Float64, 2}
    pol_func::SharedArray{Float64, 2}
end

#initialization function at everywhere
@everywhere function Initialize()
    prim = Primitives()
    val_func = SharedArray{Float64, 2}(zeros(Float64, prim.nk, prim.nz))
    pol_func = SharedArray{Float64, 2}(zeros(Float64, prim.nk, prim.nz))
    res = Results(val_func, pol_func)
    prim, res
end

#Bellman Operator at everywhere, stochastic version
@everywhere function Bellman(prim::Primitives,res::Results)
    @unpack val_func = res #unpack value function
    @unpack k_grid, z_grid, β, δ, α, nk, nz, Π = prim #unpack model primitives
    v_next = SharedArray{Float64, 2}(zeros(Float64, prim.nk, prim.nz)) #next guess of value function to fill

    @sync @distributed for z_index in eachindex(z_grid) #parallelization for z_grid loop
        for k_index in eachindex(k_grid) #we devide k_index loop for @distributed macro to work well

            k = k_grid[k_index] #value of k
            z = z_grid[z_index] #value of z
            candidate_max = -Inf #bad candidate max
            budget = z*k^α + (1-δ)*k #budget

            for kp_index in 1:nk #loop over possible selections of k'
                c = budget - k_grid[kp_index] #consumption given k' selection
                if c > 0 #check for positivity
                    val = log(c) + β*(Π[z_index, 1] * val_func[kp_index, 1] + Π[z_index, 2]*val_func[kp_index, 2]) #compute (expected) value
                    if val > candidate_max #check for new max value
                        candidate_max = val #update max value
                        res.pol_func[k_index, z_index] = k_grid[kp_index] #update policy function
                    end
                end
            end
            v_next[k_index, z_index] = candidate_max #update value function
        end
    end
    v_next #return next guess of value function
end

#Iteration
function V_iterate(prim::Primitives, res::Results; tol::Float64 = 1e-6, err::Float64 = 100.0)
    n = 0

    while err > tol
        v_next = Bellman(prim, res)
        err = maximum(abs.(v_next .- res.val_func))
        res.val_func .= v_next
        n += 1
    end
    println("Value function converged in ", n, " iterations.")
end

#solve the model
function Solve_model(prim::Primitives, res::Results)
    V_iterate(prim, res) #in this case, all we have to do is the value function iteration!
end

##################################################
