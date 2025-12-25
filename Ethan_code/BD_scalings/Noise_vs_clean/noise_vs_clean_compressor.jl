using ITensors
using ITensorMPS
using JLD2
using LinearAlgebra
using Printf
using Statistics

function compress_noise_vs_clean()
    input_file = joinpath(@__DIR__, "noise_vs_clean_data.jld2")
    output_file = joinpath(@__DIR__, "noise_vs_clean_data_compressed.jld2")
    
    if !isfile(input_file)
        println("Error: Input file '$input_file' not found.")
        return
    end

    println("Loading raw simulation data...")
    # Using IOStream to avoid file locking issues
    data = jldopen(input_file, "r"; iotype=IOStream)
    
    if !haskey(data, "results_clean") || !haskey(data, "results_noise")
        println("Error: Raw data file is missing 'results_clean' or 'results_noise'.")
        close(data)
        return
    end

    results_clean = read(data, "results_clean")
    results_noise = read(data, "results_noise")
    close(data)

    # Dictionary to store lightweight analysis results
    analysis_db = Dict{Int, Dict{Symbol, Any}}()
    
    println("\nExtracting Analysis Data...")
    println("Metrics: Fidelity, Full Renyi Spectrum (0, 0.5, 1, 2), Order Parameters")
    println("-"^110)
    @printf("%-4s | %-10s | %-12s | %-12s | %-12s | %-15s\n", 
            "N", "Fidelity", "SvN(Clean)", "SvN(Noise)", "M_stagg", "Status")
    println("-"^110)

    all_Ns = sort(collect(keys(results_clean)))

    for N in all_Ns
        if !haskey(results_noise, N); continue; end

        psi_clean = results_clean[N]
        psi_noise = results_noise[N]

        # ---------------------------------------------------------
        # 1. Basic Complexity (Bond Dimension)
        # ---------------------------------------------------------
        chi_clean = maxlinkdim(psi_clean)
        chi_noise = maxlinkdim(psi_noise)

        # ---------------------------------------------------------
        # 2. Fidelity (Overlap)
        # ---------------------------------------------------------
        sites_clean = siteinds(psi_clean)
        replace_siteinds!(psi_noise, sites_clean)
        fidelity = abs(inner(psi_clean, psi_noise))^2

        # ---------------------------------------------------------
        # 3. Entanglement Analysis (Multi-Alpha)
        # ---------------------------------------------------------
        function get_full_entanglement(psi, bond)
            psi_c = copy(psi)
            orthogonalize!(psi_c, bond)
            
            # SVD for Schmidt Spectrum
            _, S, _ = svd(psi_c[bond], uniqueinds(psi_c[bond], psi_c[bond+1]))
            
            # Extract spectrum as standard vector
            sv = [S[n, n] for n in 1:dim(S, 1)]
            
            # Normalized Probabilities
            p = sv.^2
            p = p ./ sum(p)
            p_nz = p[p .> 1e-16] # Filter zeros for Log

            # Calculate Renyi Dictionary
            entropies = Dict{Float64, Float64}()
            
            # Alpha = 0 (Hartley: Log of Rank)
            entropies[0.0] = log(length(p_nz))
            
            # Alpha = 1 (Von Neumann: Limit alpha->1)
            entropies[1.0] = -sum(p_nz .* log.(p_nz))
            
            # Alpha = 0.5 and 2.0 (Standard Renyi)
            for alpha in [0.5, 2.0]
                entropies[alpha] = (1.0 / (1.0 - alpha)) * log(sum(p_nz.^alpha))
            end

            return entropies, sv
        end
        
        mid_b = N ÷ 2
        ent_clean, spec_clean = get_full_entanglement(psi_clean, mid_b)
        ent_noise, spec_noise = get_full_entanglement(psi_noise, mid_b)

        # ---------------------------------------------------------
        # 4. Order Parameter: Staggered Magnetization
        # ---------------------------------------------------------
        # M_stagg = (1/N) * sum_i (-1)^i * <Sz_i>
        # Used to detect Neel order
        mag_c = expect(psi_clean, "Sz")
        mag_n = expect(psi_noise, "Sz")
        
        # Calculate Staggered Mag
        # Note: site indices are 1..N. (-1)^i flips sign every site.
        mst_c = sum([mag_c[i] * (-1)^i for i in 1:N]) / N
        mst_n = sum([mag_n[i] * (-1)^i for i in 1:N]) / N

        # ---------------------------------------------------------
        # 5. Spin Correlations
        # ---------------------------------------------------------
        C_clean = Matrix{Float64}(correlation_matrix(psi_clean, "Sz", "Sz"))
        C_noise = Matrix{Float64}(correlation_matrix(psi_noise, "Sz", "Sz"))

        # ---------------------------------------------------------
        # 6. Store Everything
        # ---------------------------------------------------------
        analysis_db[N] = Dict(
            :fidelity       => fidelity,
            :chi_clean      => chi_clean,
            :chi_noise      => chi_noise,
            # Entropies (Dicts)
            :entropy_clean  => ent_clean,
            :entropy_noise  => ent_noise,
            # Spectra (Vectors)
            :spectrum_clean => spec_clean,
            :spectrum_noise => spec_noise,
            # Order Parameters (Scalars)
            :stagg_mag_clean => abs(mst_c),
            :stagg_mag_noise => abs(mst_n),
            # Correlations (Matrices)
            :C_clean        => C_clean,
            :C_noise        => C_noise
        )

        @printf("%-4d | %-10.6f | %-12.4f | %-12.4f | %-12.4f | %-15s\n", 
                N, fidelity, ent_clean[1.0], ent_noise[1.0], abs(mst_c), "Done")
    end

    println("-"^110)
    println("Saving extended analysis file to $output_file...")

    if isfile(output_file); rm(output_file; force=true); end
    jldsave(output_file; iotype=IOStream, analysis_results=analysis_db)
    
    println("Done.")
end

compress_noise_vs_clean()