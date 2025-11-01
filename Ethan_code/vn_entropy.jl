using Random, Statistics
using ITensors, ITensorMPS, LinearAlgebra
using JLD2
using Base.Threads

Random.seed!(1234);

function create_MPS(L::Int)
    sites = siteinds("S=1/2", L; conserve_qns=true)
    initial_state = [isodd(i) ? "Up" : "Dn" for i in 1:L]
    ψ₀ = randomMPS(sites, initial_state)
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
Creates the MPO for the XXZ Hamiltonian on a graph with weighted interactions.
"""
function create_weighted_xxz_mpo(N::Int, adj_mat, sites; J::Float64, Δ::Float64)
    ampo = OpSum()
    for i in 1:N-1
        for j in i+1:N
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
Alternative manual entropy calculation.
This is what `ITensors.entropy` does under the hood.
"""
function calculate_entropy_manual(ψ::MPS, b::Int)
    orthogonalize!(ψ, b)
    
    # Check if bond b-1 exists (i.e., if b > 1)
    if b == 1
        # We are at the first site, no left bond
        s_b = siteind(ψ, b)
        U, S, V = svd(ψ[b], (s_b,))
    else
        # We are at site b > 1, so bond b-1 exists
        l_b_minus_1 = linkind(ψ, b - 1)
        s_b = siteind(ψ, b)
        U, S, V = svd(ψ[b], (l_b_minus_1, s_b))
    end

    S_ent = 0.0

    for n in 1:dim(S, 1)
        p = S[n, n]^2
        if p > 1e-12 
            S_ent -= p * log(p)
        end
    end
    return S_ent
end

"""
MODIFIED: Function now accepts parameters and results dictionary.
"""
function run_entropy_simulation(
    entropy_results::Dict{Float64, Vector{Float64}},
    N_range,
    sigma_values_to_run,
    num_graphs_avg::Int,
    num_sweeps::Int,
    max_bond_dim_limit::Int,
    cutoff::Float64,
    μ::Float64,
    J_coupling::Float64,
    Delta_coupling::Float64
)

    println("Starting entropy simulations...")
    filename_entropy = joinpath(@__DIR__, "vn_entropy_data.jld2")

    for σ in sigma_values_to_run
        
        # --- ADDED: Skip logic ---
        if haskey(entropy_results, σ)
            println("Skipping σ = $σ, results already loaded/computed.")
            continue
        end
        # ---
        
        println("Running for σ = $σ")
        
        entropies_for_N = zeros(Float64, length(N_range))

        Threads.@threads for i in 1:length(N_range)
            N = N_range[i]
            b = N ÷ 2 

            num_runs = (σ == 0.0) ? 1 : num_graphs_avg
            entropies_for_avg = zeros(Float64, num_runs) 

            for run in 1:num_runs
                ψ₀, sites = create_MPS(N)
                adj_mat = create_weighted_adj_mat(N, σ; μ=μ)
                H_mpo = create_weighted_xxz_mpo(N, adj_mat, sites; J=J_coupling, Δ=Delta_coupling)

                sweeps = Sweeps(num_sweeps)
                setmaxdim!(sweeps, max_bond_dim_limit)
                setcutoff!(sweeps, cutoff)
                noise_vals = LinRange(1E-6, 1E-10, num_sweeps)
                setnoise!(sweeps, noise_vals...)

                _, ψ_gs = dmrg(H_mpo, ψ₀, sweeps; outputlevel=0)

                S = calculate_entropy_manual(ψ_gs, b)

                entropies_for_avg[run] = S
            end

            avg_entropy = mean(entropies_for_avg)
            entropies_for_N[i] = avg_entropy 
            
            println("  Completed N = $N for σ = $σ (Avg. Entropy = $avg_entropy)")
        end

        entropy_results[σ] = entropies_for_N
        
        # --- ADDED: Checkpointing after each sigma ---
        try
            jldsave(filename_entropy; entropy_results, N_range_used = N_range, sigma_values_used = sigma_values_to_run)
            println("Checkpoint saved for σ = $σ")
        catch e
            println("WARNING: Could not save checkpoint for σ = $σ. Error: $e")
        end
        # ---
    end

    println("Simulations finished.")
    # No return needed, modifies entropy_results in-place
end


"""
MODIFIED: Main function to handle parameters, loading, and checkpointing.
"""
function main()
    println("Starting calculations...")

    N_range = 10:1:100 
    sigma_values_to_run = [0.0, 0.002] 
    num_graphs_avg = 10 
    
    num_sweeps = 30
    max_bond_dim_limit = 250
    cutoff = 1E-10
    μ = 1.0
    J_coupling = -0.5
    Delta_coupling = 0.5 
    
    filename_entropy = joinpath(@__DIR__, "vn_entropy_data.jld2")
    local entropy_results 

    if isfile(filename_entropy)
        println("Found existing data file. Loading progress...")
        try
            loaded_data = jldopen(filename_entropy, "r")
            N_range_loaded = read(loaded_data, "N_range_used")
            sigma_values_loaded = read(loaded_data, "sigma_values_used")

            if N_range_loaded == N_range && sigma_values_loaded == sigma_values_to_run
                println("Parameters match. Resuming...")
                entropy_results = read(loaded_data, "entropy_results")
            else
                println("WARNING: Parameters in file do not match current script. Starting from scratch.")
                entropy_results = Dict{Float64, Vector{Float64}}()
            end
            close(loaded_data)
        catch e
            println("WARNING: Could not load existing file. Starting from scratch. Error: $e")
            entropy_results = Dict{Float64, Vector{Float64}}()
        end
    else
        println("No existing data file found. Starting from scratch.")
        entropy_results = Dict{Float64, Vector{Float64}}()
    end

    run_entropy_simulation(
        entropy_results,
        N_range,
        sigma_values_to_run,
        num_graphs_avg,
        num_sweeps,
        max_bond_dim_limit,
        cutoff,
        μ,
        J_coupling,
        Delta_coupling
    )

    println("Calculations finished. Final data save...")
    jldsave(filename_entropy; entropy_results, N_range_used = N_range, sigma_values_used = sigma_values_to_run)
    println("Data saved successfully.\n")
end

main()

