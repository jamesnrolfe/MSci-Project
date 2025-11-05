function phase_diagram_J_Δ_dmrg(N::Int, J_range, Δ_range; save_fig::Bool=false)
    
    energies = zeros(length(J_range), length(Δ_range))
    
    adj_mat = generate_chain_adjacency_matrix(N)
    
    for (i, J) in enumerate(J_range)
        for (j, Δ) in enumerate(Δ_range)
            if J < 0 # ferromagnetic, need to give a different initial state
                init_state = ["Up" for _ in 1:N] # ferromagnetic state
            else
                init_state = [isodd(k) ? "Up" : "Dn" for k = 1:N] # antiferromagnetic state
            end
            ψ0, sites = create_custom_MPS(N, 100, init_state; conserve_qns=false)
            H = create_xxz_hamiltonian_mpo(N, adj_mat, J, Δ, sites)
            E, ψ = solve_xxz_hamiltonian_dmrg(H, ψ0, 5)
            
            energies[i, j] = E
        end
    end
    
    # Plot phase diagram
    h = heatmap(Δ_range, J_range, energies, 
           xlabel="Δ", ylabel="J", 
           title="Ground State Energy")
    
    if save_fig
        savefig(h, "phase_diagram_J_Δ_N$(N).png")
    end

    return h, energies # plot and the raw data
end

function phase_diagram_J_Δ_exact(N::Int, J_range, Δ_range; save_fig::Bool=false)
    """
    Generate phase diagram using exact diagonalization for XXZ model.
    Returns ground state energies for all combinations of J and Δ.
    """
    
    energies = zeros(length(J_range), length(Δ_range))
    
    for (i, J) in enumerate(J_range)
        for (j, Δ) in enumerate(Δ_range)
            # Create Hamiltonian for this (J, Δ) combination
            H = get_xxz_hamiltonian_exact(N, J, Δ)
            
            # Solve for ground state
            eigens, ground_state_energy, ψ = solve_xxz_hamiltonian_exact(H)
            energies[i, j] = ground_state_energy

        end
    end
    
    # Create phase diagram plot
    p = heatmap(Δ_range, J_range, energies, 
               xlabel="Δ", ylabel="J", 
               title="Ground State Energy (Exact, N=$N)",
               aspect_ratio=:auto)
    
    if save_fig
        savefig(p, "phase_diagram_exact_N$(N).png")
    end

    return p, energies # plot and the raw data
end

function phase_space_magnetisation_J_Δ_exact(N, J_range, Δ_range; save_fig::Bool=false)
    magnetizations = zeros(length(J_range), length(Δ_range))

    for (i, J) in enumerate(J_range)
        for (j, Δ) in enumerate(Δ_range)
            H = get_xxz_hamiltonian_exact(N, J, Δ)
            eigens, ground_state_energy, ψ = solve_xxz_hamiltonian_exact(H)

            # Calculate total magnetization using <ψ|Sz_total|ψ>
            total_magnetization = 0.0
            
            # Create total Sz operator: sum of Sz_i over all sites
            sz = 0.5 * [1.0 0.0; 0.0 -1.0]
            
            for site in 1:N
                # Build Sz operator for site i
                sz_i = 1.0
                for k in 1:N
                    if k == site
                        sz_i = kron(sz_i, sz)
                    else
                        sz_i = kron(sz_i, I(2))
                    end
                end
                
                # Add expectation value <ψ|Sz_i|ψ>
                total_magnetization += real(ψ' * sz_i * ψ)
            end
            
            # Average magnetization per site
            magnetizations[i, j] = total_magnetization / N
        end
    end

    p = heatmap(Δ_range, J_range, magnetizations,
            xlabel="Δ", ylabel="J",
            title="Average Magnetization per Site (Exact, N=$N)",
            color=:RdBu,
            clims=(-0.5, 0.5),
            aspect_ratio=:auto)
    
    if save_fig
        savefig(p, "magnetization_phase_diagram_N$(N).png")
    end
    
    return p, magnetizations # plot and raw data
end