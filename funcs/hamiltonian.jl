"""For creating Hamiltonian operators in different ways."""

using LinearAlgebra
using ITensors 
using ITensorMPS

function get_xxz_hamiltonian_exact(L::Int, J::Float64, Δ::Float64)
    """
    Constructs the Hamiltonian for the XXZ spin chain model with L sites, coupling constant J, and anisotropy parameter Δ.
    """
    dim = 2^L

    # define the pauli matrices we need
    sx = 0.5 * [0.0 1.0; 1.0 0.0] # pauli X
    sy = 0.5 * [0.0 -im; im 0.0] # pauli Y
    sz = 0.5 * [1.0 0.0; 0.0 -1.0] # pauli Z
    id = I(2) # 2D identity from LinearAlgebra

    # init as a zero matrix
    H = zeros(ComplexF64, dim, dim)

    # now we can construct the hamiltonian (i.e. fill it with the right values)
    # loop over all the sites 0 to L-1
    for i in 1:L-1
        j = i + 1 # j is the adjacent site to i (to the right)

        # we need to construct the full terms for each interaction
        term_xx = 1.0
        term_yy = 1.0
        term_zz = 1.0

        for site in 1:L
            if site == i
                # if we are on the ith site, we want to add the sx, sy, sz terms
                term_xx = kron(term_xx, sx)
                term_yy = kron(term_yy, sy)
                term_zz = kron(term_zz, sz)
            elseif site == j 
                # if we are on the jth site, we want to add the sx, sy, sz terms
                term_xx = kron(term_xx, sx)
                term_yy = kron(term_yy, sy)
                term_zz = kron(term_zz, sz)
            else
                # otherwise we add the identity
                term_xx = kron(term_xx, id)
                term_yy = kron(term_yy, id)
                term_zz = kron(term_zz, id)
            end
        end

        # now we can add these terms to the hamiltonian
        H += J * term_xx + J * term_yy + J * Δ * term_zz
    end

    return H
end

function solve_xxz_hamiltonian_exact(H)
    """Solves an exact Hamiltonian matrix H, returning the eigenvalues and ground state."""
    eigens = eigen(Hermitian(H))
    eigenvalues = eigens.values
    ground_state = minimum(eigenvalues)
    ψ = eigens.vectors[:, argmin(eigenvalues)]
    return eigens, ground_state, ψ
end

function create_xxz_hamiltonian_mpo(N, adj_mat, J, Δ, sites)
    """Create the XXZ Hamiltonian as an MPO given an adjacency matrix."""
    ampo = OpSum()
    for i = 1:N-1
        for j = i+1:N
            if adj_mat[i, j] == 1  # if there is some entanglement between i and j
                # XX and YY terms: S+S- + S-S+ = 2(SxSx + SySy)
                # So to get J(SxSx + SySy), we need J/2 * (S+S- + S-S+)
                ampo += J/2, "S+", i, "S-", j
                ampo += J/2, "S-", i, "S+", j
                # ZZ term
                ampo += J*Δ, "Sz", i, "Sz", j
            end
        end
    end
    H = MPO(ampo, sites)
    return H
end

function solve_xxz_hamiltonian_dmrg(H, ψ0, sweeps::Int=10, bond_dim::Int=1000, cutoff::Float64=1E-14)
    """Solves the XXZ Hamiltonian using DMRG with given parameters. Returns the ground state energy and MPS. """
    swps = Sweeps(sweeps)
    setmaxdim!(swps, bond_dim)
    setcutoff!(swps, cutoff)
    E, ψ = dmrg(H, ψ0, swps; outputlevel=0)
    return E, ψ # only ground state and ground state wavefunction
end