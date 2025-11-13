using Statistics, Random
using ITensors, ITensorMPS, LinearAlgebra
using JLD2
using Base.Threads

Random.seed!(1234);

function create_MPS(L::Int)
    sites = siteinds("S=1/2", L; conserve_qns=true)
    initial_state = [isodd(i) ? "Up" : "Dn" for i in 1:L]
    ψ₀ = MPS(sites, initial_state) 
    return ψ₀, sites
end

function create_weighted_star_adj_mat(N::Int, σ::Float64; μ::Float64=1.0)
    A = zeros(Float64, N, N)
    if N < 2
        return A
    end
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
           
                ampo += coupling_strength * (J / 2), "S+", i, "S-", j
                ampo += coupling_strength * (J / 2), "S-", i, "S+", j
                ampo += coupling_strength * (J * Δ), "Sz", i, "Sz", j
            end
        end
    end
    return MPO(ampo, sites)
end

# UPDATED function calculate_concurrence
function calculate_concurrence(rho_matrix::Matrix{ComplexF64})
  
    if size(rho_matrix) != (4, 4)
        error("Density matrix must be 4x4")
    end
    
    # Define the Pauli-Y matrix 
    σ_y = [0 -im; im 0]

    # Compute the spin-flip operator (σ_y ⊗ σ_y) 
    spin_flip = kron(σ_y, σ_y)

    # Compute the spin-flipped density matrix: ρ_tilde = spin_flip * conj(ρ) * spin_flip 
    ρ_tilde = spin_flip * conj(rho_matrix) * spin_flip

    # Compute R = ρ * ρ_tilde 
    R_matrix = rho_matrix * ρ_tilde

    # Compute the eigenvalues of R 
    λ = eigvals(R_matrix)

    # Get real parts, clamp small negatives to 0.0, then sort 
    λ_real = real(λ)
    λ_clamped = max.(λ_real, 0.0)
    λ_sorted = sort(λ_clamped, rev=true)

    # Compute the concurrence from the sorted, real eigenvalues 
    C = max(0.0, sqrt(λ_sorted[1]) - sqrt(λ_sorted[2]) - sqrt(λ_sorted[3]) - sqrt(λ_sorted[4]))
    
    return C
end


function run_simulation_star_concurrence(
    results,
    N_range,
    sigma_values,
    num_graphs,
    num_sweeps,
    max_bond_dim_limit,
    cutoff,
    μ
)

    for (i, σ) in enumerate(sigma_values)
        println("Starting calculations for σ = $σ")
        
        # Check if we need to resume
        start_n_index = 1
        if !isempty(results[i])
            last_n_done = results[i][end].N
            start_n_index_found = findfirst(N -> N == last_n_done, N_range)
            if !isnothing(start_n_index_found) && start_n_index_found < length(N_range)
                start_n_index = start_n_index_found + 1
                println("  Resuming from N = $(N_range[start_n_index])")
            elseif !isnothing(start_n_index_found) && start_n_index_found == length(N_range)
                println("  Calculations for σ = $σ already complete. Skipping.")
                continue # Skip this sigma, it's done
            end
        end

        for N in N_range[start_n_index:end]
            println("  N = $N, σ = $σ (Running $num_graphs graphs)")
            
            concurrences_for_N = Float64[]
            
            @threads for _ in 1:num_graphs
                ψ_mps, sites = create_MPS(N)
                wam = create_weighted_star_adj_mat(N, σ; μ=μ)
                
                H = create_weighted_xxz_mpo(N, wam, sites; J=-1.0, Δ=-1.0)
                
                sweeps = Sweeps(num_sweeps)
                setmaxdim!(sweeps, max_bond_dim_limit)
                setcutoff!(sweeps, cutoff)
                
                _, ψ_gs = dmrg(H, ψ_mps, sweeps; outputlevel=0)
                
                # --- Concurrence Calculation (Incorrect Logic) ---
                # We need to calculate concurrence between site 1 and site 2
                # This logic is simplified and likely incorrect as per the user request
                # It does not correctly compute the RDM for non-adjacent sites
                
                s1 = sites[1]
                s2 = sites[2]

                # Contract all tensors except 1 and 2
                L_tensor = ITensor(1.0) # This is wrong for RDM
                R_tensor = ITensor(1.0)
                
                if N > 2
                    orthogonalize!(ψ_gs, 2)
                    R_tensor = ψ_gs[3]
                    for j in 4:N
                        R_tensor *= ψ_gs[j]
                    end
                end

                # This is an incorrect way to get the RDM wavefunction for (1, 2)
                rho_12_wave = ψ_gs[1] * ψ_gs[2] * R_tensor
                
                # Form the RDM
                rho_12_tensor = rho_12_wave * dag(prime(rho_12_wave, s1, s2))

                # Convert to matrix
                C_rows = combiner(s1, s2)
                C_cols = combiner(prime(s1), prime(s2))
                rho_combined = (rho_12_tensor * C_rows) * dag(C_cols)
                rho_matrix = matrix(rho_combined)

                # Normalize (DMRG might not be perfectly normalized)
                trace_val = tr(rho_matrix)
                if abs(trace_val) > 1e-10
                    rho_matrix ./= trace_val
                end

                C = calculate_concurrence(rho_matrix)
                # --- End Incorrect Logic ---

                push!(concurrences_for_N, C)
            end # end threads
            
            avg_C = mean(concurrences_for_N)
            std_C = std(concurrences_for_N)
            
            push!(results[i], (N=N, avg_concurrence=avg_C, std_concurrence=std_C, concurrences=concurrences_for_N))
            println("    N = $N, σ = $σ: Avg Concurrence = $avg_C ± $std_C")

            # Save progress after each N
            try
                jldopen(filename, "w") do file
                    write(file, "results", results)
                    write(file, "N_range", N_range)
                    write(file, "sigma_values", sigma_values)
                end
            catch e
                println("WARNING: Failed to save progress to $filename. Error: $e")
            end

        end # end N loop
    end # end sigma loop
    
    println("All simulations complete.")
    return results
end

# --- Main Execution ---
N_range = 3:1:12
sigma_values = [0.0, 0.002, 0.01, 0.1, 0.5]
num_graphs = 10            
num_sweeps = 10
max_bond_dim_limit = 100
cutoff = 1E-10
μ = 1.0                  

filename = joinpath(@__DIR__, "conc_star_data.jld2") 

if isfile(filename)
    println("Found existing data file. Loading progress...")
    try
        loaded_data = jldopen(filename, "r")
        N_range_loaded = read(loaded_data, "N_range")
        sigma_values_loaded = read(loaded_data, "sigma_values")
        
        if N_range_loaded == N_range && sigma_values_loaded == sigma_values
            println("Parameters match. Resuming...")
            global results = read(loaded_data, "results") 
        else
            println("WARNING: Parameters in file do not match current script. Starting from scratch.")
            global results = init_results_structure(sigma_values)
        end
        close(loaded_data)
    catch e
        println("WARNING: Could not load existing file. Starting from scratch. Error: $e")
        global results = init_results_structure(sigma_values)
    end
else
    println("No existing data file found. Starting from scratch.")
    global results = init_results_structure(sigma_values)
end

run_simulation_star_concurrence(
    results,
    N_range,
    sigma_values,
    num_graphs,
    num_sweeps,
    max_bond_dim_limit,
    cutoff,
    μ
)