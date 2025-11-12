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

"""
Creates a weighted adjacency matrix for a star graph (center spin 1).
Applies disorder 'σ' to the couplings.
"""
function create_weighted_star_adj_mat(N::Int, σ::Float64; μ::Float64=1.0)
    A = zeros(Float64, N, N)
    if N < 2
        return A
    end
    # Connect center (1) to all outer spins (2...N)
    for j in 2:N
        weight = (σ == 0.0) ? μ : (μ + σ * randn())
        A[1, j] = A[j, 1] = weight
    end
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
Calculates concurrence for a 2-qubit (4x4) density matrix.
"""
function calculate_concurrence(rho_matrix::Matrix{ComplexF64})
    if size(rho_matrix) != (4, 4)
        error("Density matrix must be 4x4")
    end

    sy = [0.0 -1.0im; 1.0im 0.0]
    sy_sy = kron(sy, sy)

    rho_tilde = sy_sy * conj(rho_matrix) * sy_sy
    R_matrix = rho_matrix * rho_tilde
    eigvals_R = eigvals(R_matrix)
    
    lambdas = sort(sqrt.(complex(eigvals_R)), by=real, rev=true)

    C = max(0.0, real(lambdas[1]) - real(lambdas[2]) - real(lambdas[3]) - real(lambdas[4]))
    return C
end

function run_simulation_star_concurrence(
    results::Dict,
    N_range,
    sigma_values,
    num_graphs_avg::Int,
    num_sweeps::Int,
    max_bond_dim_limit::Int,
    cutoff::Float64,
    μ::Float64,
    filename::String
)
    
    println("Data will be saved to: $filename")

    Threads.@threads for i in 1:length(N_range)
        N = N_range[i] 

        if isassigned(results[sigma_values[1]].avg, i)
             println("Skipping N = $N, results already loaded/computed.")
            continue
        end
        
        for σ in sigma_values

            concurrence_for_avg_all_pairs = [zeros(Float64, num_graphs_avg) for _ in 2:N]
            
            for k in 1:num_graphs_avg
                ψ₀, sites = create_MPS(N)
                
                adj_mat = create_weighted_star_adj_mat(N, σ; μ=μ)
                H_mpo = create_weighted_xxz_mpo(N, adj_mat, sites; J=-1.0, Δ=-1.0)

                sweeps = Sweeps(num_sweeps)
                setmaxdim!(sweeps, max_bond_dim_limit)
                setcutoff!(sweeps, cutoff)

                _, ψ_gs = dmrg(H_mpo, ψ₀, sweeps; outputlevel=0)
    
                if N < 2
                    continue
                end

                # --- MODIFIED: Calculate RDM for all pairs (1, j) ---
                
                # Orthogonalize MPS with center at site 1
                orthogonalize!(ψ_gs, 1)
                s1 = sites[1]
                
                # C_tensor will be the contraction of tensors from 1 up to j
                C_tensor = ψ_gs[1] 
                
                for j_pair in 2:N
                    # Contract the next tensor in the chain
                    C_tensor *= ψ_gs[j_pair]
                    s_j = sites[j_pair]
                    
                    # Get site indices for all sites *between* 1 and j (i.e., 2, 3, ..., j-1)
                    # These are the sites we need to trace out.
                    prime_map_sites = [sites[k] for k in 2:(j_pair-1)] # Empty if j_pair = 2

                    # Contract C_tensor with its conjugate, priming only the
                    # intermediate sites (prime_map_sites) to trace them out.
                    # This leaves us with a tensor for sites 1 and j_pair.
                    rho_1j_tensor = C_tensor * dag(prime(C_tensor, prime_map_sites...))

                    # Convert the RDM tensor to a standard 4x4 matrix
                    C_rows = combiner(s1, s_j)
                    C_cols = combiner(prime(s1), prime(s_j))
                    
                    rho_combined = (rho_1j_tensor * C_rows) * dag(C_cols)
                    rho_matrix = matrix(rho_combined)

                    C = calculate_concurrence(complex(rho_matrix))
                    
                    # Store result (j_pair=2 -> index 1, j_pair=3 -> index 2, etc.)
                    concurrence_for_avg_all_pairs[j_pair - 1][k] = C
                end
                # --- End of modified RDM block ---
            end

            if N >= 2
                # Calculate mean and std dev for each pair (1,j)
                avg_conc_all_pairs = [mean(concurrence_for_avg_all_pairs[j_idx]) for j_idx in 1:(N-1)]
                std_dev_all_pairs = [std(concurrence_for_avg_all_pairs[j_idx]) for j_idx in 1:(N-1)]
                
                results[σ].avg[i] = avg_conc_all_pairs
                results[σ].err[i] = std_dev_all_pairs
            else
                results[σ].avg[i] = Float64[] # Store empty array for N < 2
                results[σ].err[i] = Float64[]
            end
        end
        println("Completed N = $N")

        # Save checkpoint
        try
            jldsave(filename; results, N_range, sigma_values)
            println("Checkpoint saved for N = $N")
        catch e
            println("WARNING: Could not save checkpoint for N = $N. Error: $e")
        end
    end
    println("...calculations finished.")
end

"""
Initializes the results dictionary.
We use Vector{Any} because each element results[σ].avg[i] will be a Vector{Float64}
of a different length (N-1), so the outer vector can't be strongly typed.
"""
function init_results_structure(N_range_len, sigma_values)
    return Dict(σ => (avg=Vector{Any}(undef, N_range_len), 
                     err=Vector{Any}(undef, N_range_len)) 
                     for σ in sigma_values)
end


println("Starting Star Graph Concurrence calculations...")

# Parameters
N_range = 4:1:20
sigma_values = [0.002] 
num_graphs_avg = 10            
num_sweeps = 10
max_bond_dim_limit = 100
cutoff = 1E-10
μ = 1.0                     

filename = joinpath(@__DIR__, "weighted_conc_data.jld2") 

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
            global results = init_results_structure(length(N_range), sigma_values)
        end
        close(loaded_data)
    catch e
        println("WARNING: Could not load existing file. Starting from scratch. Error: $e")
        global results = init_results_structure(length(N_range), sigma_values)
    end
else
    println("No existing data file found. Starting from scratch.")
    global results = init_results_structure(length(N_range), sigma_values)
end
# --- End of modified block ---


run_simulation_star_concurrence(
    results,
    N_range,
    sigma_values,
    num_graphs_avg,
    num_sweeps,
    max_bond_dim_limit,
    cutoff,
    μ,
    filename
)

println("Calculations finished. Final data save...")
jldsave(filename; results, N_range, sigma_values)
println("Data saved successfully.\n")