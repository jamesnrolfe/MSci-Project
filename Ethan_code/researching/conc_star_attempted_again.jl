using Statistics, Random
using ITensors, ITensorMPS, LinearAlgebra
using JLD2
using Base.Threads

Random.seed!(1234);

# --- Helper Functions (Reused from conc_star.jl) ---

"""
Initializes a product state MPS "Up, Dn, Up, Dn..."
"""
function create_MPS(L::Int)
    sites = siteinds("S=1/2", L; conserve_qns=true)
    initial_state = [isodd(i) ? "Up" : "Dn" for i in 1:L]
    ψ₀ = MPS(sites, initial_state) 
    return ψ₀, sites
end

"""
Creates the adjacency matrix for a star graph.
Node 1 is the center, connected to all others (2...N).
Weights are drawn from N(μ, σ^2).
"""
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

"""
Creates the XXZ Hamiltonian MPO from the weighted adjacency matrix.
"""
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

"""
Initializes the nested dictionary for storing results.
"""
function init_results_structure(sigma_values)
    return Dict(σ => Dict() for σ in sigma_values)
end

# --- New Concurrence Calculation Functions ---

"""
    concurrence_of_rho(rho_mat)

Calculates the concurrence for a 4x4 reduced density matrix `rho_mat`.
Uses Wootters' formula: C(ρ) = max(0, λ₁ - λ₂ - λ₃ - λ₄),
where λᵢ are the sqrt of the eigenvalues of (√ρ) * R * (√ρ) in descending order,
and R = (σʸ ⊗ σʸ) ρ* (σʸ ⊗ σʸ).
"""
function concurrence_of_rho(rho_mat::Matrix{C}) where {C <: Complex}
    # Ensure matrix is Hermitian (removes small imaginary parts from numerical error)
    rho_mat = (rho_mat + dagger(rho_mat)) / 2.0
    
    # Define the Ry matrix (spin-flip) in the standard |↑↑⟩, |↑↓⟩, |↓↑⟩, |↓↓⟩ basis
    ry_mat = [0 0 0 -1; 0 0 1 0; 0 1 0 0; -1 0 0 0] + 0im
    
    # Calculate ρ_tilde = (√ρ) * (σʸ ⊗ σʸ) * ρ* * (σʸ ⊗ σʸ) * (√ρ)
    # This is numerically more stable than R = (√ρ) * R_op * (√ρ)
    
    # Need to be careful with sqrt of non-positive-definite matrices (from numerics)
    # SVD is a robust way to get sqrt(ρ)
    U, S, V = svd(rho_mat)
    sqrt_rho = U * Diagonal(sqrt.(max.(0.0, S))) * V'
    
    # Calculate the R matrix
    R_mat = sqrt_rho * ry_mat * conj(rho_mat) * ry_mat * sqrt_rho
    
    # Get eigenvalues of R
    eigvals_R = real.(eigvals(R_mat))
    
    # Ensure eigenvalues are non-negative due to numerical precision
    sqrt_eigvals = sqrt.(max.(0.0, eigvals_R))
    
    # Sort eigenvalues in descending order
    lambda = sort(sqrt_eigvals, rev=true)
    
    # Concurrence formula
    conc = max(0.0, lambda[1] - lambda[2] - lambda[3] - lambda[4])
    return conc
end

"""
    get_rho_ij(ψ::MPS, i::Int, j::Int)

Calculates the 2-site reduced density matrix ρ_ij for sites i and j.
Assumes i < j.
This function "concatenates" the tensors as requested.
"""
function get_rho_ij(ψ::MPS, i::Int, j::Int)
    sites = siteinds(ψ)
    s_i = sites[i]
    s_j = sites[j]
    
    # Orthogonalize MPS at site i
    orthogonalize!(ψ, i)
    
    # Contract all tensors between i and j (inclusive)
    # This tensor 'phi' will have physical indices (s_i, s_j)
    # and link indices from link(i-1) and link(j)
    phi = ψ[i]
    for k in (i+1):j
        phi *= ψ[k]
    end
    
    # Contract with dag(phi) to get RDM
    # We prime only s_i and s_j, so contracting with dag(phi)
    # traces out all other sites AND the boundary links.
    rho = prime(phi, s_i, s_j) * dag(phi)
    
    # Convert the resulting ITensor to a 4x4 matrix
    # The order (s_i, s_j) determines the basis
    
    # --- FIX ---
    # OLD: rho_mat = matrix(rho, (s_i, s_j), (prime(s_i), prime(s_j)))
    # NEW: Pass only the row indices. ITensors will automatically
    # use the remaining indices (prime(s_i), prime(s_j)) as the columns.
    # This is a more robust call and avoids the permute error.
    rho_mat = matrix(rho, (s_i, s_j))
    
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

Main helper function to compute concurrence between sites i and j.
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

"""
Main simulation loop.
Calculates C(1, j) for j=2...N for a star graph.
Averages over `num_graphs` and stores the dictionary
{j => C_1j} for each (N, σ) pair.
"""
function run_simulation_central_concurrence(
    results, N_range, sigma_values, 
    num_graphs, J, Δ, 
    num_sweeps, max_bond_dim_limit, cutoff, μ
)

    # --- FIX ---
    # Removed `@threads` from the loop below.
    # With sigma_values = [0.002] (or any single value),
    # it only has one element, so threading provides no benefit
    # and causes a "World Age" error (TaskFailedException).
    for σ in sigma_values
        σ_key = σ
        if !haskey(results, σ_key)
            # This might race, but it's okay.
            # A more robust way would be a lock or atomic,
            # but for this structure, it's generally fine.
            results[σ_key] = Dict()
        end
        
        for N in N_range
            N_key = N
            # Check if this (N, σ) combination is already done
            if haskey(results[σ_key], N_key)
                # Note: We check for 'haskey' now, not 'isempty', 
                # as a 0.0 value is valid.
                println("Skipping N=$N, σ=$σ (already computed).")
                continue
            end
            println("Starting N=$N, σ=$σ...")
            
            # --- MODIFIED LOGIC ---
            # This will store the sum of the *standard deviations*
            # of concurrence from each graph. We average this at the end.
            sum_of_std_devs = 0.0
            
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
                # We now collect all C(1,j) values for *this graph*
                # in a temporary list.
                concurrences_this_graph = Float64[]
                
                for j in 2:N
                    C_1j = calculate_concurrence(ψ, 1, j)
                    push!(concurrences_this_graph, C_1j)
                end
                
                # --- NEW STEP: Calculate StdDev for this graph ---
                # Calculate the standard deviation of the concurrences
                # we just found. This quantifies the "non-equality".
                if length(concurrences_this_graph) > 1
                    std_dev_C = std(concurrences_this_graph)
                    sum_of_std_devs += std_dev_C
                end
                
            end # end graph loop (g)
            
            # --- MODIFIED Averaging and Storing ---
            # Instead of a dict, we now store the average *standard deviation*
            avg_std_dev = (num_graphs > 0) ? (sum_of_std_devs / num_graphs) : 0.0
            
            # Store this single float value
            results[σ_key][N_key] = avg_std_dev
            
            # Print the new result
            println("Finished N=$N, σ=$σ. Avg StdDev[C(1,j)] = $(round(avg_std_dev, 5))")
            
        end # end N loop
    end # end sigma loop
    return results
end

# --- Main Execution Block ---

# --- Set Parameters ---
# User request: N = 4:2:12
const N_range = 4:2:4
# User request: sigma = 0.002
const sigma_values = [0.002] 

# Parameters from conc_star.jl
const J = 1.0                  
const Δ = 1.0                  
const num_graphs = 10         # Number of random graphs to average over
const num_sweeps = 10
const max_bond_dim_limit = 250
const cutoff = 1E-10
const μ = 1.0                  

# New filename as requested
const filename = joinpath(@__DIR__, "conc_star_data_new.jld2") 

# --- Load/Initialize Results ---
# (Identical logic to conc_star.jl, but with new filename)
local results # Use 'local' to assign to global 'results' in try/catch

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
println("Running simulation...")
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

println("\n--- Final Results ---")
display(results)
println("\nDone.")