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
Creates an adjacency matrix for a star graph with the center at spin 1.
No disorder (homogeneous couplings).
"""
function create_star_adj_mat(N::Int)
    A = zeros(Float64, N, N)
    if N < 2
        return A
    end
    # Connect center (1) to all outer spins (2...N)
    for j in 2:N
        A[1, j] = 1.0
        A[j, 1] = 1.0
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
C(ρ) = max(0, λ₁ - λ₂ - λ₃ - λ₄)
where λᵢ are the sorted sqrt(eigenvalues) of (ρ * ρ̃)
and ρ̃ = (σʸ ⊗ σʸ)ρ*(σʸ ⊗ σʸ)
"""
function calculate_concurrence(rho_matrix::Matrix{ComplexF64})
    if size(rho_matrix) != (4, 4)
        error("Density matrix must be 4x4")
    end

    # Pauli-Y matrix
    sy = [0.0 -1.0im; 1.0im 0.0]
    # σʸ ⊗ σʸ
    sy_sy = kron(sy, sy)

    # ρ̃ = (σʸ ⊗ σʸ)ρ*(σʸ ⊗ σʸ)
    rho_tilde = sy_sy * conj(rho_matrix) * sy_sy

    # R = ρ * ρ̃
    R_matrix = rho_matrix * rho_tilde

    # Eigenvalues of R
    eigvals_R = eigvals(R_matrix)

    # λᵢ are the square roots of the eigenvalues of R, sorted by real part
    lambdas = sort(sqrt.(complex(eigvals_R)), by=real, rev=true)

    # C = max(0, λ₁ - λ₂ - λ₃ - λ₄)
    C = max(0.0, real(lambdas[1]) - real(lambdas[2]) - real(lambdas[3]) - real(lambdas[4]))
    return C
end


println("Starting Star Graph Concurrence calculation (Case 1)...")


N_range = 4:1:12 
concurrence_results = zeros(Float64, length(N_range))

# DMRG parameters
num_sweeps = 10
max_bond_dim_limit = 100 
cutoff = 1E-10

for (i, N) in enumerate(N_range)
    println("Running N = $N...")

    ψ₀, sites = create_MPS(N)
    adj_mat = create_star_adj_mat(N)

    H_mpo = create_weighted_xxz_mpo(N, adj_mat, sites; J=-1.0, Δ=-1.0)

    # 2. Run DMRG
    sweeps = Sweeps(num_sweeps)
    setmaxdim!(sweeps, max_bond_dim_limit)
    setcutoff!(sweeps, cutoff)
    
    _, ψ_gs = dmrg(H_mpo, ψ₀, sweeps; outputlevel=0)

    if N < 2
        concurrence_results[i] = 0.0
        continue
    end

    # 3. Calculate 2-site RDM for center (1) and one outer spin (2)
    #    using the manual ITensors.jl contraction method.
    
    # Ensure the MPS is orthogonalized at site 1
    orthogonalize!(ψ_gs, 1)
    
    # Get the site indices
    # 'sites' is already the vector of site indices,
    # so we just index into it.
    s1 = sites[1]
    s2 = sites[2]

    # Contract the "ket" tensors for sites 1 and 2
    phi = ψ_gs[1] * ψ_gs[2]
    
    # Create the RDM by contracting the "ket" (phi) with the "bra" (its dagger)
    # dag(prime(phi, s1, s2)) creates the "bra" with primed site indices
    rho_12_tensor = phi * dag(prime(phi, s1, s2))

    # 4. Convert RDM tensor to a standard 4x4 matrix
    
    # Use combiners to explicitly group row and column indices.
    # This is a more robust method for QN tensors.
    C_rows = combiner(s1, s2)
    C_cols = combiner(prime(s1), prime(s2))

    # Apply the combiners to contract (s1,s2) into 'C_rows'
    # and (s1',s2') into 'C_cols'.
    rho_combined = (rho_12_tensor * C_rows) * C_cols

    # rho_combined is now a 2-index tensor (rows, cols).
    # Convert this directly to a dense Julia matrix.
    rho_matrix = matrix(rho_combined)

    C = calculate_concurrence(rho_matrix)
    concurrence_results[i] = C

    println("N = $N, Concurrence(Spin 1, Spin 2) = $C")
end

println("...calculations finished.")



println("\n" * "="^30)
println("Final Results: Concurrence vs. N (Star Graph)")
for i in 1:length(N_range)
    println("N = $(N_range[i]), C = $(concurrence_results[i])")
end

filename = joinpath(@__DIR__, "star_conc_data.jld2")
try
    jldsave(filename; N_range, concurrence_results)
    println("\nResults saved successfully to $filename")
catch e
    println("\nWARNING: Could not save results. Error: $e")
end