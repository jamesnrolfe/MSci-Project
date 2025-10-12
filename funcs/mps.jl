"""For creating and handling MPS states."""

using ITensors
using ITensorMPS

function create_custom_MPS(L::Int, Χ::Int, init_state::Vector{String}; conserve_qns::Bool=true)
    """Create a random MPS for a spin-1/2 chain of length L with bond dimension Χ and a custom initial state."""
    # create a site set for a spin-1/2 chain
    sites = siteinds("S=1/2", L; conserve_qns=conserve_qns) # conserve total Sz

    # create a random MPS with bond dimension Χ
    ψ0 = randomMPS(sites, init_state)
    return ψ0, sites
end

function create_MPS(L::Int, Χ::Int; conserve_qns::Bool=true)
    """Create a random MPS for a spin-1/2 chain of length L with bond dimension Χ."""
    # create a site set for a spin-1/2 chain
    sites = siteinds("S=1/2", L; conserve_qns=conserve_qns) # conserve total Sz

    # create a random MPS with bond dimension Χ
    init_state = [isodd(i) ? "Up" : "Dn" for i = 1:L] # antiferromagnetic ground state
    # THIS IS IMPORTANT - SEE NOTE BELOW
    # it sets the subspace of states we are allowed to explore
    # for example, this init_state means we only explore states with total Sz = 0 (i.e. zero magnetisation)
    # this is a reasonable assumption for positive J, but not for negative J
    # if we want to explore ferromagnetic states (negative J), we would need a different init_state
    # USE create_custom_MPS TO SET A DIFFERENT INIT STATE
    ψ0 = randomMPS(sites, init_state)
    return ψ0, sites
end

function mps_to_array(ψ::MPS)
    """
    Convert an MPS wavefunction to a full state vector (array) with correct basis ordering.

    #! SHOULD DEPRICATE - USE ITensors.contract(ψ) TO GET FULL TENSOR
    """
    N = length(ψ)
    
    # contract the MPS to get full tensor
    ψ_tensor = ITensors.contract(ψ)
    # convert to array - this gives the correct ITensor ordering
    ψ_array = array(ψ_tensor)
    
    # reshape to vector
    ψ_vector = reshape(ψ_array, (2^N,))
    
    # Convert ITensor basis ordering to standard Kronecker product ordering
    # ITensor uses reverse ordering compared to standard tensor products
    ψ_corrected = zeros(ComplexF64, 2^N)
    
    for i in 0:(2^N-1)
        # convert index to binary representation
        binary_rep = digits(i, base=2, pad=N)
        
        # ITensor uses reverse site ordering, so flip the binary representation
        itensor_idx = 0
        for j in 1:N
            itensor_idx += binary_rep[N+1-j] * 2^(j-1)
        end
        
        ψ_corrected[i+1] = ψ_vector[itensor_idx+1]
    end
    
    return ψ_corrected
end