using Statistics, Random
using ITensors, ITensorMPS, LinearAlgebra
using JLD2
using Base.Threads

Random.seed!(1234);

# --- Helper Functions from conc_star.jl ---

function create_MPS(L::Int)
    sites = siteinds("S=1/2", L; conserve_qns=true)
    # Create an alternating "Up, Dn, Up, Dn..." initial state
    initial_state = [isodd(i) ? "Up" : "Dn" for i in 1:L]
    ψ₀ = MPS(sites, initial_state) 
    return ψ₀, sites
end

function create_weighted_star_adj_mat(N::Int, σ::Float64; μ::Float64=1.0)
    A = zeros(Float64, N, N)
    if N < 2
        return A
    end
    # Star graph: Node 1 is connected to all others
    for j in 2:N
        weight = (σ == 0.0) ? μ : (μ + σ * randn())
        A[1, j] = A[j, 1] = weight
    end
    return A
end

function create_weighted_xxz_mpo(N::Int, adj_mat, sites; J::Float64, Δ::Float64)
    ampo = OpSum()
    for i in 1:N-1
        for j in (i+1):N 
            coupling_strength = adj_mat[i, j]
            if coupling_strength != 0.0
                # Add XXZ Hamiltonian terms
                ampo += coupling_strength * (J / 2), "S+", i, "S-", j
                ampo += coupling_strength * (J / 2), "S-", i, "S+", j
                ampo += coupling_strength * Δ, "Sz", i, "Sz", j
            end
        end
    end
    H = MPO(ampo, sites)
    return H
end

function init_results_structure(sigma_values)
    # Same structure as conc_star.jl
    return Dict(σ => Dict() for σ in sigma_values)
end

# --- Concurrence Finding Structure (from stars.ipynb logic) ---

"""
    concurrence_of_rho(rho_mat)

Calculates the concurrence for a 4x4 reduced density matrix.
Uses Wootters' formula.
"""
function concurrence_of_rho(rho_mat::Matrix{C}) where {C <: Complex}
    # Ensure matrix is Hermitian
    rho_mat = (rho_mat + dagger(rho_mat)) / 2.0
    
    # Define the Ry matrix (spin-flip)
    ry_mat = [0 0 0 -1; 0 0 1 0; 0 1 0 0; -1 0 0 0] + 0im
    
    # Calculate ρ_tilde
    sqrt_rho = sqrt(rho_mat)
    rho_tilde = sqrt_rho * ry_mat * conj(rho_mat) * ry_mat * sqrt_rho
    
    # Get eigenvalues
    eigvals_rt = real.(eigvals(rho_tilde))
    
    # Ensure eigenvalues are non-negative due to numerical precision
    sqrt_eigvals = sqrt.(max.(0.0, eigvals_rt))
    
    # Sort eigenvalues in descending order
    lambda = sort(sqrt_eigvals, rev=true)
    
    # Concurrence formula
    # FIX: Renamed variable from 'C' to 'conc' to avoid conflict
    conc = max(0.0, lambda[1] - lambda[2] - lambda[3] - lambda[4])
    return conc
end

"""
    get_rho_ij(ψ::MPS, i::Int, j::Int)

Calculates the 2-site reduced density matrix ρ_ij for sites i and j.
Assumes i < j.
"""
function get_rho_ij(ψ::MPS, i::Int, j::Int)
    sites = siteinds(ψ)
    s_i = sites[i]
    s_j = sites[j]
    
    # Orthogonalize MPS at site i
    orthogonalize!(ψ, i)
    
    # Contract all tensors between i and j (inclusive)
    phi = ψ[i]
    for k in (i+1):j
        phi *= ψ[k]
    end
    
    # Contract with dag(phi) to get RDM, priming only i and j
    rho = prime(phi, s_i, s_j) * dag(phi)
    
    # Convert to a 4x4 matrix
    rho_mat = matrix(rho, (s_i, s_j), (prime(s_i), prime(s_j)))
    
    # Normalize to fix potential DMRG normalization or numerical issues
    tr_rho = tr(rho_mat)
    if !isapprox(tr_rho, 1.0; atol=1e-5)
        # println("Warning: Trace(ρ_$i,$j) = $tr_rho. Renormalizing.")
        rho_mat ./= tr_rho
    end
    
    return rho_mat
end

"""
    calculate_concurrence(ψ::MPS, i::Int, j::Int)

Main function to compute concurrence between sites i and j.
"""
function calculate_concurrence(ψ::MPS, i::Int, j::Int)
    # Ensure i < j for the RDM function
    if i > j
        i, j = j, i
    end
    
    # Get 2-site RDM
    rho_ij_mat = get_rho_ij(ψ, i, j)
    
    # Calculate concurrence from RDM
    # Ensure it's ComplexF64 for the linear algebra
    return concurrence_of_rho(ComplexF64.(rho_ij_mat))
end


# --- Main Simulation Function (Modified) ---

function run_simulation_central_concurrence(
    results, N_range, sigma_values, 
    num_graphs, J, Δ, 
    num_sweeps, max_bond_dim_limit, cutoff, μ
)

    # Use @threads for parallel execution over sigma
    @threads for σ in sigma_values
        σ_key = σ
        if !haskey(results, σ_key)
            results[σ_key] = Dict()
        end
        
        for N in N_range
            N_key = N
            # Check if this (N, σ) combination is already done
            if haskey(results[σ_key], N_key) && !isempty(results[σ_key][N_key])
                println("Skipping N=$N, σ=$σ (already computed).")
                continue
            end
            println("Starting N=$N, σ=$σ...")
            
            # This will store the sum of concurrences over all graphs
            # We will average at the end
            concurrence_results_sum = Dict{Int, Float64}()
            
            for g in 1:num_graphs
                ψ₀, sites = create_MPS(N)
                adj_mat = create_weighted_star_adj_mat(N, σ; μ=μ)
                H = create_weighted_xxz_mpo(N, adj_mat, sites; J=J, Δ=Δ)
                
                sweeps = Sweeps(num_sweeps)
                setmaxdim!(sweeps, max_bond_dim_limit)
                setcutoff!(sweeps, cutoff)
                
                # Run DMRG
                energy, ψ = dmrg(H, ψ₀, sweeps; outputlevel=0)
                
                # --- MODIFIED CONCURRENCE LOOP ---
                # This is the "concurrence finding structure" requested
                # Only calculate for pairs (1, j) where j = 2...N
                for j in 2:N
                    C_1j = calculate_concurrence(ψ, 1, j)
                    
                    # Initialize sum if first time
                    if !haskey(concurrence_results_sum, j)
                        concurrence_results_sum[j] = 0.0
                    end
                    concurrence_results_sum[j] += C_1j
                end
                
            end # end graph loop (g)
            
            # --- Averaging and Storing ---
            # "store them under the N value"
            avg_concurrence_values = Dict{Int, Float64}()
            for (j, C_sum) in concurrence_results_sum
                avg_concurrence_values[j] = C_sum / num_graphs
            end
            
            # Store this dictionary of {j => C_1j}
            results[σ_key][N_key] = avg_concurrence_values
            
            println("Finished N=$N, σ=$σ. Avg C(1,2) = $(round(get(avg_concurrence_values, 2, 0.0), 4))")
            
        end # end N loop
    end # end sigma loop
    return results
end

# --- Main Execution Block ---

# --- Set Parameters ---
# User request: N = 4:2:12
N_range = 4:2:12
# User request: sigma = 0.002
sigma_values = [0.002] 

# Parameters from conc_star.jl
J = 1.0                  
Δ = 1.0                  
num_graphs = 100            # Number of random graphs to average over for the given sigma
num_sweeps = 10
max_bond_dim_limit = 100
cutoff = 1E-10
μ = 1.0                  

# New filename as requested
filename = joinpath(@__DIR__, "conc_central_data.jld2") 

# --- Load/Initialize Results ---
# (Identical logic to conc_star.jl)
if isfile(filename)
    println("Found existing data file: $filename")
    println("Loading progress...")
    try
        global results = jldopen(filename, "r") do file
            # Check if parameters match
            N_range_loaded = read(file, "N_range")
            sigma_values_loaded = read(file, "sigma_values")
            
            if N_range_loaded == N_range && sigma_values_loaded == sigma_values
                println("Parameters match. Resuming...")
                read(file, "results")
            else
                println("WARNING: Parameters in file do not match current script. Starting from scratch.")
                init_results_structure(sigma_values)
            end
        end
    catch e
        println("WARNING: Could not load existing file. Starting from scratch. Error: $e")
        global results = init_results_structure(sigma_values)
    end
else
    println("No existing data file found. Starting from scratch.")
    global results = init_results_structure(sigma_values)
end

# --- Run Simulation ---
run_simulation_central_concurrence(
    results,
    N_range,
    sigma_values,
    num_graphs, J, Δ,
    num_sweeps, max_bond_dim_limit, cutoff, μ
)

# --- Save Results ---
println("Simulation complete. Saving results to $filename...")
try
    jldopen(filename, "w") do file
        file["results"] = results
        file["N_range"] = N_range
        file["sigma_values"] = sigma_values
        file["J"] = J
        file["Delta"] = Δ
        file["mu"] = μ
        file["num_graphs"] = num_graphs
        file["num_sweeps"] = num_sweeps
    end
    println("Save successful.")
catch e
    println("ERROR: Could not save results to $filename. Error: $e")
end

display(results)
println("\nDone.")