using ITensors
using ITensorMPS
using JLD2
using LinearAlgebra
using Printf

# --- Helper Function: Calculate Entanglement Entropy ---
function get_half_chain_entropy(psi::MPS)
    N = length(psi)
    b = div(N, 2) # Half-chain bond
    
    # Orthogonalize to the bond to ensure valid SVD
    psi_loc = copy(psi)
    orthogonalize!(psi_loc, b)
    
    # Perform SVD across the bond
    # U, S, V = svd(psi[b], (linkind(psi, b-1), siteind(psi, b)))
    # Easier ITensor syntax: just get singular values from the bond
    inds_left = uniqueinds(psi_loc[b], psi_loc[b+1])
    _, S, _ = svd(psi_loc[b], inds_left)
    
    SvN = 0.0
    for n in 1:dim(S, 1)
        p = S[n, n]^2
        if p > 1E-12
            SvN -= p * log(p)
        end
    end
    return SvN
end

function compress_noise_vs_clean()
    input_file = joinpath(@__DIR__, "noise_vs_clean_data.jld2")
    output_file = joinpath(@__DIR__, "noise_vs_clean_data_compressed.jld2")
    
    if !isfile(input_file)
        println("Error: Input file '$input_file' not found.")
        return
    end

    println("Loading raw simulation data...")
    # Loading the specific dictionaries created in noise_vs_clean.jl
    data = load(input_file)
    results_clean = data["results_clean"]
    results_noise = data["results_noise"]
    
    compressed_db = Dict()
    
    println("\nProcessing and Compressing...")
    println("Strategy: Calculate Fidelity/Entropy, Keep Clean MPS, Drop Noisy MPS")
    println("-"^85)
    @printf("%-4s | %-10s | %-10s | %-12s | %-12s | %-15s\n", 
            "N", "Fidelity", "MaxDim(N)", "S_vN(Clean)", "S_vN(Noise)", "Action")
    println("-"^85)

    # Sort keys to process in order
    all_Ns = sort(collect(keys(results_clean)))

    for N in all_Ns
        # Ensure we have the pair
        if !haskey(results_noise, N)
            println("Warning: Missing noise data for N=$N, skipping.")
            continue
        end

        psi_clean = results_clean[N]
        psi_noise = results_noise[N]

        # --- CRITICAL: FIX INDICES ---
        # Because noise_vs_clean.jl generated sites separately for clean/noise,
        # the ITensor indices (IDs) won't match. We must force them to match
        # to calculate overlap.
        sites_clean = siteinds(psi_clean)
        # Replace indices of noise MPS to match clean MPS
        replace_siteinds!(psi_noise, sites_clean)

        # --- 1. Calculate Fidelity ---
        # |<clean|noise>|^2
        ov = inner(psi_clean, psi_noise)
        fidelity = abs(ov)^2

        # --- 2. Calculate Entropies (Half-chain) ---
        S_clean = get_half_chain_entropy(psi_clean)
        S_noise = get_half_chain_entropy(psi_noise)

        # --- 3. Get Bond Dimensions ---
        chi_clean = maxlinkdim(psi_clean)
        chi_noise = maxlinkdim(psi_noise)

        # --- 4. Build Entry ---
        entry = Dict(
            "N" => N,
            "fidelity" => fidelity,
            "entropy_clean" => S_clean,
            "entropy_noise" => S_noise,
            "maxlinkdim_clean" => chi_clean,
            "maxlinkdim_noise" => chi_noise,
            
            # Retention Strategy:
            # We KEEP the Clean MPS (useful baseline for future correlations)
            "mps_clean" => psi_clean 
            # We DROP the Noisy MPS (stats extracted, heavy object removed)
        )

        compressed_db[N] = entry

        @printf("%-4d | %-10.6f | %-10d | %-12.4f | %-12.4f | %-15s\n", 
                N, fidelity, chi_noise, S_clean, S_noise, "Compressed")
    end

    println("-"^85)
    println("Saving compressed database to $output_file...")
    jldsave(output_file; compressed_results=compressed_db)

    # --- File Size Stats ---
    s_in = filesize(input_file) / 1024^2
    s_out = filesize(output_file) / 1024^2
    
    println("\nCompression Summary:")
    @printf("Original Data:   %.2f MB\n", s_in)
    @printf("Compressed Data: %.2f MB\n", s_out)
    @printf("Space Saved:     %.1f%%\n", 100 * (1 - s_out/s_in))
end

compress_noise_vs_clean()