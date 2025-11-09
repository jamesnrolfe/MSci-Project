using Statistics, Random
using ITensors, ITensorMPS, LinearAlgebra
using Base.Threads
using JLD2 

Random.seed!(1234)
global jld_lock = ReentrantLock() 


"""
Creates an MPS for an alternating "Up", "Dn" initial state.
"""
function create_MPS(L::Int)
    sites = siteinds("S=1/2", L; conserve_qns=true)
    initial_state = [isodd(i) ? "Up" : "Dn" for i in 1:L]
    ψ₀ = MPS(sites, initial_state)
    return ψ₀, sites
end

"""
Creates a weighted adjacency matrix for a completely connected graph.
"""
function create_weighted_adj_mat(N::Int, σ::Float64; μ::Float64=1.0)
    if σ == 0.0
        A = ones(Float64, N, N)
        A -= Matrix{Float64}(I, N, N)
        return A
    end
    A = zeros(Float64, N, N)
    for i in 1:N, j in (i+1):N
        weight = μ + σ * randn()
        A[i, j] = A[j, i] = weight
    end
    return A
end


"""
Creates a weighted adjacency matrix for an 'outlier' graph.
"""
function create_outlier_adj_mat(N::Int, σ::Float64; μ::Float64=1.0)
    A = zeros(Float64, N, N)

    # Create the (N-1)x(N-1) fully connected subgraph
    for i in 1:(N-1), j in (i+1):(N-1)
        weight = (σ == 0.0) ? μ : (μ + σ * randn())
        A[i, j] = A[j, i] = weight
    end

    # Create the single connection for the outlier (spin N) to spin 
    outlier_weight = (σ == 0.0) ? μ : (μ + σ * randn())
    A[1, N] = A[N, 1] = outlier_weight

    return A
end


"""
Creates the MPO for the XXZ Hamiltonian on a graph with weighted interactions.
"""
function create_weighted_xxz_mpo(N::Int, adj_mat, sites; J::Float64, Δ::Float64)
    ampo = OpSum()
    for i in 1:N-1
        for j in (i+1):N
            coupling_strength = adj_mat[i, j]
            if coupling_strength != 0.0
                ampo += coupling_strength * (J / 2), "S+", i, "S-", j
                ampo += coupling_strength * (J / 2), "S-", i, "S+", j
                ampo += coupling_strength * (J * Δ), "Sz", i, "Sz", j
            end
        end
    end
    return MPO(ampo, sites)
end


"""
Runs DMRG and returns the maximum bond dimension of the ground state.
"""
function calculate_gs_entanglement(
    N::Int,
    adj_mat,
    sites;
    J::Float64,
    Δ::Float64,
    num_sweeps::Int,
    max_bond_dim_limit::Int,
    cutoff::Float64
)
    ψ₀ = MPS(sites, [isodd(i) ? "Up" : "Dn" for i in 1:N])
    H_mpo = create_weighted_xxz_mpo(N, adj_mat, sites; J=J, Δ=Δ)

    sweeps = Sweeps(num_sweeps)
    setmaxdim!(sweeps, max_bond_dim_limit)
    setcutoff!(sweeps, cutoff)
    setnoise!(sweeps, LinRange(1E-6, 1E-10, num_sweeps)...)

    _, ψ_gs = dmrg(H_mpo, ψ₀, sweeps; outputlevel=0)

    return maxlinkdim(ψ_gs)
end






function main_outlier_comparison()
    println("--- Starting Part 3: Outlier Graph Entanglement Comparison ---")

    N_range = [8, 10, 12, 14, 16]
    sigma_values = [0.0, 0.01, 0.02] 
    num_graphs_avg = 10 

    J_coupling = -1.0 
    Delta_coupling = -1.0 
    num_sweeps = 20
    max_bond_dim_limit = 200 
    cutoff = 1E-10
    μ = 1.0

    filename = joinpath(@__DIR__, "outlier_graphs_data.jld2")
    println("Data saved to: $filename")

    init_results_dict() = Dict(σ => (avg=zeros(Float64, length(N_range)),
                                      err=zeros(Float64, length(N_range)))
                                      for σ in sigma_values)

    local results_standard, results_outlier

    if isfile(filename)
        println("Found existing data file")
        try
            jldopen(filename, "r") do file
                N_range_loaded = read(file, "N_range")
                sigma_values_loaded = read(file, "sigma_values")
                
                if N_range_loaded == N_range && sigma_values_loaded == sigma_values
                    results_standard = read(file, "results_standard")
                    results_outlier = read(file, "results_outlier")
                else
                    println("WARNING: Parameters in file do not match. creating file.")
                    results_standard = init_results_dict()
                    results_outlier = init_results_dict()
                end
            end
        catch e
            println("WARNING: Could not load existing file Error: $e")
            results_standard = init_results_dict()
            results_outlier = init_results_dict()
        end
    else
        println("No existing data file found. creating file.")
        results_standard = init_results_dict()
        results_outlier = init_results_dict()
        
        # Save initial file with parameters
        jldsave(filename; 
            results_standard, 
            results_outlier, 
            N_range, 
            sigma_values,
            J_coupling,
            Delta_coupling,
            num_graphs_avg,
            num_sweeps,
            max_bond_dim_limit
        )
    end

    println("Parameters: N=$(N_range), σ=$(sigma_values), J=$(J_coupling), Δ=$(Delta_coupling)\n")

    for (i, N) in enumerate(N_range)
        println("--- N = $N ---")

        if results_standard[sigma_values[1]].avg[i] != 0.0
            println("Skipping N = $N, results already loaded.")
            continue
        end

        sites = siteinds("S=1/2", N; conserve_qns=true)

        for σ in sigma_values
            num_runs = (σ == 0.0) ? 1 : num_graphs_avg
            
            entanglement_standard = zeros(Float64, num_runs)
            entanglement_outlier = zeros(Float64, num_runs)

            Threads.@threads for k in 1:num_runs
                adj_mat_std = create_weighted_adj_mat(N, σ; μ=μ)
                entanglement_standard[k] = calculate_gs_entanglement(
                    N, adj_mat_std, sites;
                    J=J_coupling, Δ=Delta_coupling, num_sweeps=num_sweeps,
                    max_bond_dim_limit=max_bond_dim_limit, cutoff=cutoff
                )

                adj_mat_out = create_outlier_adj_mat(N, σ; μ=μ)
                entanglement_outlier[k] = calculate_gs_entanglement(
                    N, adj_mat_out, sites;
                    J=J_coupling, Δ=Delta_coupling, num_sweeps=num_sweeps,
                    max_bond_dim_limit=max_bond_dim_limit, cutoff=cutoff
                )
            end

            avg_ent_std = mean(entanglement_standard)
            err_ent_std = (num_runs > 1) ? std(entanglement_standard) : 0.0
            
            avg_ent_out = mean(entanglement_outlier)
            err_ent_out = (num_runs > 1) ? std(entanglement_outlier) : 0.0

            results_standard[σ].avg[i] = avg_ent_std
            results_standard[σ].err[i] = err_ent_std
            results_outlier[σ].avg[i] = avg_ent_out
            results_outlier[σ].err[i] = err_ent_out

            println("  σ = $σ")
            println("    Standard Graph: Avg. Max Bond Dim = $(round(avg_ent_std, digits=2)) ± $(round(err_ent_std, digits=2))")
            println("    Outlier Graph:  Avg. Max Bond Dim = $(round(avg_ent_out, digits=2)) ± $(round(err_ent_out, digits=2))")
        end

        try
            lock(jld_lock) do
                jldopen(filename, "r+") do file
                    if haskey(file, "results_standard")
                        delete!(file, "results_standard")
                    end
                    if haskey(file, "results_outlier")
                        delete!(file, "results_outlier")
                    end
                    file["results_standard"] = results_standard
                    file["results_outlier"] = results_outlier
                end
            end
            println("Checkpoint saved for N = $N")
        catch e
            println("WARNING: Could not save checkpoint for N = $N. Error: $e")
        end
    end

    println("Calculations finished. Final data is in $filename.")
end

main_outlier_comparison()