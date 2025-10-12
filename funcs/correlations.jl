function calculate_correlations_dmrg(ψ::MPS, sites, range=(-0.25, 0.25); save_fig::Bool=false)
    """
    Calculate spin-spin correlations using correlation_matrix function.
    """
    N = length(ψ)
    
    # Calculate <Sz_i Sz_j> correlations using built-in function
    sz_correlations = correlation_matrix(ψ, "Sz", "Sz")
    
    display(heatmap(sz_correlations, 
            xlabel="Site j", ylabel="Site i",
            title="Sz-Sz Correlations (DMRG)",
            color=:RdBu, clims=range))

    if save_fig
        png("sz_sz_correlations_dmrg.png")
    end
    
    return sz_correlations
end

function calculate_correlations_exact(ψ::Vector{ComplexF64}, range=(-0.25, 0.25); save_fig::Bool=false)
    """
    Calculate spin-spin correlations <Sz_i Sz_j> from the full wavefunction array.
    """
    N = Int(log2(length(ψ))) # number of sites
    sz_correlations = zeros(Float64, N, N)
    
    # Define Sz operator for a single spin-1/2 site
    sz = 0.5 * [1.0 0.0; 0.0 -1.0]
    
    sz_ops = Vector{Matrix{ComplexF64}}(undef, N)
    for i in 1:N
        op = 1.0 # start with placeholder, will be built
        for j in 1:N
            if j == i
                op = kron(op, sz)
            else
                op = kron(op, I(2))
            end
        end
        sz_ops[i] = op
    end

    # calc correlations using <ψ|Sz_i Sz_j|ψ> via matrix multiplication
    for i in 1:N
        for j in 1:N
            if i == j
                # Diagonal: <Sz_i^2> = <ψ|Sz_i * Sz_i|ψ>
                sz_correlations[i, j] = real(ψ' * (sz_ops[i] * sz_ops[i] * ψ))
            else
                # Off-diagonal: <Sz_i * Sz_j> = <ψ|Sz_i * Sz_j|ψ>
                sz_correlations[i, j] = real(ψ' * (sz_ops[i] * sz_ops[j] * ψ))
            end
        end
    end
    
    display(heatmap(sz_correlations, 
            xlabel="Site j", ylabel="Site i",
            title="Sz-Sz Correlations (Exact)",
            color=:RdBu, clims=range))

    if save_fig
        png("sz_sz_correlations_exact.png")
    end

    return sz_correlations
end
