using LinearAlgebra
using ITensors
using ITensorMPS
using JLD2, FileIO
using Statistics
using Base.Threads
using Random

BLAS.set_num_threads(1)

N_vals = [6, 8, 10, 14, 18, 22, 26, 30] 
σ_vals = [0.0, 0.001, 0.002]           
μ = 1.0
num_sweeps = 20

max_bond_dim = 10000 
precision_cutoff = eps(Float64) 

function gen_full_conn(N::Int, σ::Float64; μ::Float64=1.0)
    A = zeros(Float64, N, N)
    for i in 1:N
        for j in (i+1):N
            weight = μ + σ * randn()
            A[i, j] = weight
            A[j, i] = weight 
        end
    end
    return A
end

function create_xxz_mpo(N::Int, A::Matrix{Float64}, sites)
    J = -1.0
    Δ = -1.0
    
    os = OpSum()
    for i in 1:N
        for j in (i+1):N
            if A[i,j] != 0.0
                os += A[i,j] * (J/2), "S+", i, "S-", j
                os += A[i,j] * (J/2), "S-", i, "S+", j
                os += A[i,j] * (J*Δ), "Sz", i, "Sz", j
            end
        end
    end
    return MPO(os, sites)
end

function run_simulation()
    output_path = joinpath(@__DIR__, "high_prec_data.jld2")
    data_lock = ReentrantLock()
    
    data = isfile(output_path) ? load(output_path) : Dict{String, Any}()

    all_tasks = vec(collect(Iterators.product(N_vals, σ_vals)))
    pending_tasks = filter(t -> !haskey(data, "N=$(t[1])/sigma=$(t[2])"), all_tasks)
    
    sort!(pending_tasks, by = x -> x[1], rev=true)

    total_work = length(pending_tasks)
    if total_work == 0
        println("All tasks completed.")
        return
    end

    task_channel = Channel{Tuple{Int, Float64}}(total_work)
    for t in pending_tasks
        put!(task_channel, t)
    end
    close(task_channel)

    println("Starting simulation on $(Threads.nthreads()) threads.")
    println("Pending tasks: $total_work")

    workers = map(1:Threads.nthreads()) do w_id
        Threads.@spawn begin
            for (N, σ) in task_channel
                try
                    A = gen_full_conn(N, σ; μ=μ)
                    sites = siteinds("S=1/2", N)
                    H = create_xxz_mpo(N, A, sites)
                    psi0 = randomMPS(sites, 2)
                    
                    sweeps = Sweeps(num_sweeps)
                    setmaxdim!(sweeps, max_bond_dim)
                    setcutoff!(sweeps, precision_cutoff) 
                    setnoise!(sweeps, 1E-6, 1E-10, 0.0)

                    energy, psi = dmrg(H, psi0, sweeps; outputlevel=0)
                    
                    center_b = N ÷ 2
                    orthogonalize!(psi, center_b)
                    u, s, v = svd(psi[center_b], (linkind(psi, center_b-1), siteind(psi, center_b)))
                    
                    sv = Float64[]
                    for n in 1:dim(s, 1)
                        push!(sv, s[n, n])
                    end
                    
                    norm_factor = sqrt(sum(sv.^2))
                    sv = sv ./ norm_factor
                    
                    lock(data_lock) do
                        key = "N=$N/sigma=$σ"
                        data[key] = sv
                        
                        temp_path = output_path * ".tmp"
                        save(temp_path, data)
                        mv(temp_path, output_path, force=true)
                        
                        println("Worker $w_id DONE: N=$N, σ=$σ")
                    end

                catch e
                    println("Worker $w_id FAILED on N=$N, σ=$σ. Error: $e")
                finally
                    GC.gc()
                end
            end
        end
    end

    wait.(workers)
    println("All simulations complete.")
end

run_simulation()