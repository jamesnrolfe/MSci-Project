using Statistics, Random
using ITensors, ITensorMPS, LinearAlgebra
using JLD2
using Base.Threads

Random.seed!(1234);

# ---
# MARK: MPO and MPS Creation Functions (from user files)
# ---

"""
[cite: 106-107]
"""
function create_MPS(L::Int)
    sites = siteinds("S=1/2", L; conserve_qns=true)
    initial_state = [isodd(i) ? "Up" : "Dn" for i in 1:L]
    ψ₀ = MPS(sites, initial_state) 
    return ψ₀, sites
end

"""
Creates a weighted adjacency matrix for a completely connected graph.
[cite: 107-108]
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
Creates the MPO for the XXZ Hamiltonian on a dense graph.
[cite: 108-110]
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
NEW: Creates the MPO for a 1D Disordered XXZ Chain.
(Styled to match the functions above)
"""
function create_disordered_chain_mpo(N::Int, sites; J::Float64, Δ::Float64, σ::Float64, μ::Float64=1.0)
    ampo = OpSum()
    
    # Create N-1 random couplings for the N-1 bonds
    couplings = [μ + σ * randn() for _ in 1:(N-1)]

    for i in 1:(N-1)
        # Only add nearest-neighbor terms
        coupling_strength = couplings[i]
        
        ampo += coupling_strength * (J / 2), "S+", i, "S-", i+1
        ampo += coupling_strength * (J / 2), "S-", i, "S+", i+1
        ampo += coupling_strength * (J * Δ), "Sz", i, "Sz", i+1
    end
    return MPO(ampo, sites)
end


# ---
# MARK: Simulation Runners
# ---

"""
Runs Model A: Dense Disordered Graph.
Structure mirrors avg_err_bd.jl 
"""
function run_model_A_dense(
    results::Dict,
    N_range,
    sigma_values,
    num_graphs_avg::Int,
    num_sweeps::Int,
    max_bond_dim_limit::Int,
    cutoff::Float64,
    μ::Float64,
    J::Float64,
    Δ::Float64,
    filename::String
)
    println("--- 🔬 Starting Model A: Dense Disordered Graph ---")
    
    Threads.@threads for i in 1:length(N_range)
        N = N_range[i] 

        # Check if this N is already done
        if haskey(results, sigma_values[1]) && results[sigma_values[1]].avg[i] != 0.0
            println("Model A: Skipping N = $N, results already loaded.")
            continue
        end
        
        for σ in sigma_values
            bond_dims_for_avg = zeros(Float64, num_graphs_avg)
            
            # For σ=0, no averaging is needed
            num_runs = (σ == 0.0) ? 1 : num_graphs_avg
            
            for k in 1:num_runs
                ψ₀, sites = create_MPS(N)
                adj_mat = create_weighted_adj_mat(N, σ; μ=μ)
                H_mpo = create_weighted_xxz_mpo(N, adj_mat, sites; J=J, Δ=Δ)

                sweeps = Sweeps(num_sweeps)
                setmaxdim!(sweeps, max_bond_dim_limit)
                setcutoff!(sweeps, cutoff)
                # Add noise as in the entropy script [cite: 135-136]
                setnoise!(sweeps, LinRange(1E-6, 1E-10, num_sweeps)...)

                _, ψ_gs = dmrg(H_mpo, ψ₀, sweeps; outputlevel=0)
                bond_dims_for_avg[k] = maxlinkdim(ψ_gs)
            end

            avg_dim = mean(bond_dims_for_avg[1:num_runs])
            std_dev = (num_runs > 1) ? std(bond_dims_for_avg[1:num_runs]) : 0.0
            
            results[σ].avg[i] = avg_dim
            results[σ].err[i] = std_dev
        end
        println("Model A: Completed N = $N")

        # Save checkpoint (inside N loop, as in user file)
        try
            # Must lock to prevent race condition on file write
            lock(jld_lock) do
                # Re-open file to save *all* results, not just this thread's
                jldopen(filename, "r+") do file
                    if haskey(file, "results_dense")
                        delete!(file, "results_dense")
                    end
                    file["results_dense"] = results
                end
            end
            println("Model A: Checkpoint saved for N = $N")
        catch e
            println("WARNING: Model A: Could not save checkpoint for N = $N. Error: $e")
        end
    end
    println("--- ✅ Model A: Finished ---")
end


"""
Runs Model B: 1D Disordered Chain.
Structure mirrors avg_err_bd.jl 
"""
function run_model_B_1d(
    results::Dict,
    N_range,
    sigma_values,
    num_graphs_avg::Int,
    num_sweeps::Int,
    max_bond_dim_limit::Int,
    cutoff::Float64,
    μ::Float64,
    J::Float64,
    Δ::Float64,
    filename::String
)
    println("--- 🔬 Starting Model B: 1D Disordered Chain ---")
    
    Threads.@threads for i in 1:length(N_range)
        N = N_range[i] 

        # Check if this N is already done
        if haskey(results, sigma_values[1]) && results[sigma_values[1]].avg[i] != 0.0
            println("Model B: Skipping N = $N, results already loaded.")
            continue
        end
        
        for σ in sigma_values
            bond_dims_for_avg = zeros(Float64, num_graphs_avg)
            
            num_runs = (σ == 0.0) ? 1 : num_graphs_avg
            
            for k in 1:num_runs
                ψ₀, sites = create_MPS(N)
                # Use the NEW MPO function
                H_mpo = create_disordered_chain_mpo(N, sites; J=J, Δ=Δ, σ=σ, μ=μ)

                sweeps = Sweeps(num_sweeps)
                setmaxdim!(sweeps, max_bond_dim_limit)
                setcutoff!(sweeps, cutoff)
                setnoise!(sweeps, LinRange(1E-6, 1E-10, num_sweeps)...)

                _, ψ_gs = dmrg(H_mpo, ψ₀, sweeps; outputlevel=0)
                bond_dims_for_avg[k] = maxlinkdim(ψ_gs)
            end

            avg_dim = mean(bond_dims_for_avg[1:num_runs])
            std_dev = (num_runs > 1) ? std(bond_dims_for_avg[1:num_runs]) : 0.0
            
            results[σ].avg[i] = avg_dim
            results[σ].err[i] = std_dev
        end
        println("Model B: Completed N = $N")

        # Save checkpoint (inside N loop)
        try
            # Must lock to prevent race condition on file write
            lock(jld_lock) do
                jldopen(filename, "r+") do file
                    if haskey(file, "results_1d")
                        delete!(file, "results_1d")
                    end
                    file["results_1d"] = results
                end
            end
            println("Model B: Checkpoint saved for N = $N")
        catch e
            println("WARNING: Model B: Could not save checkpoint for N = $N. Error: $e")
        end
    end
    println("--- ✅ Model B: Finished ---")
end


# ---
# MARK: Main Execution
# ---

println("Starting calculations...")

# --- Global Parameters ---
N_range = 10:1:75  # Reduced N_range for faster test
sigma_values = [0.0, 0.001, 0.002]
num_graphs_avg = 10
num_sweeps = 30
max_bond_dim_limit = 250
cutoff = 1E-10
μ = 1.0

# J and Δ from the report's entropy script 
J_coupling = -0.5
Delta_coupling = 0.5

filename = joinpath(@__DIR__, "comparative_spike_analysis.jld2")
println("Data file: $filename\n")

# A lock for thread-safe file writing
global jld_lock = ReentrantLock()

# Helper function to initialize the results dictionaries
init_results_dict() = Dict(σ => (avg=zeros(Float64, length(N_range)),
                                  err=zeros(Float64, length(N_range))) 
                                  for σ in sigma_values)

# --- Load or Initialize Results ---
# This logic mirrors avg_err_bd.jl 
local results_dense, results_1d

if isfile(filename)
    println("Found existing data file. Loading progress...")
    try
        global results_dense, results_1d
        
        jldopen(filename, "r") do file
            N_range_loaded = read(file, "N_range")
            sigma_values_loaded = read(file, "sigma_values")

            if N_range_loaded == N_range && sigma_values_loaded == sigma_values
                println("Parameters match. Resuming...")
                results_dense = read(file, "results_dense")
                results_1d = read(file, "results_1d")
            else
                println("WARNING: Parameters in file do not match. Starting from scratch.")
                results_dense = init_results_dict()
                results_1d = init_results_dict()
            end
        end
    catch e
        println("WARNING: Could not load existing file. Starting from scratch. Error: $e")
        global results_dense = init_results_dict()
        global results_1d = init_results_dict()
    end
else
    println("No existing data file found. Starting from scratch.")
    global results_dense = init_results_dict()
    global results_1d = init_results_dict()
    
    # Create the file so threads can open it with "r+"
    jldsave(filename; 
        results_dense, 
        results_1d, 
        N_range, 
        sigma_values
    )
end


# --- Run Simulations ---
run_model_A_dense(
    results_dense,
    N_range,
    sigma_values,
    num_graphs_avg,
    num_sweeps,
    max_bond_dim_limit,
    cutoff,
    μ,
    J_coupling,
    Delta_coupling,
    filename
)

run_model_B_1d(
    results_1d,
    N_range,
    sigma_values,
    num_graphs_avg,
    num_sweeps,
    max_bond_dim_limit,
    cutoff,
    μ,
    J_coupling,
    Delta_coupling,
    filename
)

println("Calculations finished. Final data save...")
# Save everything one last time [cite: 122]
jldsave(filename; 
    results_dense, 
    results_1d, 
    N_range, 
    sigma_values
)
println("Data saved successfully.\n")