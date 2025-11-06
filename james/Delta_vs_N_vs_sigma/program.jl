using Pkg
Pkg.add([ 
    "ITensors", 
    "ITensorMPS", 
    "LinearAlgebra", 
    "Statistics", 
    "JLD2"
])

using Random
using Base.Threads
using ITensorMPS
using ITensors
using LinearAlgebra
using Statistics
using JLD2

J = -1
σ_vals = [0.0, 0.001, 0.002]
Δ_vals = -1:0.5:1
N_vals = 10:1:100

MAX_BOND_DIM = 500
NUM_SWEEPS = 30

AVG_OVER_GRAPHS = 5

OVERALL_DATA_STORAGE_FILE = "overall_data.jld2"
TEMP_DATA_STORAGE_FILE = "temp_data.jld2"

if isfile(TEMP_DATA_STORAGE_FILE)
    TEMP_DATA = jldopen(TEMP_DATA_STORAGE_FILE, "r") do file
        
        read(file, "data")
    end
    println("Loaded temp data file.")
else
    TEMP_DATA = Dict()
    println("No temp data file found. Creating new one.")
end

function generate_fully_connected_am(N::Int)
    """
    Create an unweighted adjacency matrix for a fully connected graph of N nodes.
    """
    A = ones(Int, N, N)
    for i in 1:N; A[i, i] = 0; end # remove self-loops
    return A
end

function generate_fully_connected_wam(N::Int, σ::Float64; μ::Float64=1.0)
    """
    Create a weighted adjacency matrix for a fully connected graph of N nodes. μ should always be one really.
    """

    if σ == 0.0 # if no randomness, return unweighted fully connected graph
        return generate_fully_connected_am(N)
    end
 
    A = zeros(Float64, N, N)
    for i in 1:N
        for j in (i+1):N
            weight = μ + σ * randn() # weight from normal distribution with mean μ and std σ
            A[i, j] = weight
            A[j, i] = weight 
        end
    end
    return A
end 

function create_MPS(L::Int, conserve_qns::Bool=true)
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

function create_xxz_hamiltonian_mpo(N, adj_mat, J, Δ, sites)
    """Create the XXZ Hamiltonian as an MPO given an adjacency matrix."""
    ampo = OpSum()
    for i = 1:N-1
        for j = i+1:N
            weight = adj_mat[i, j]
            if weight != 0.0
                # XX and YY terms: S+S- + S-S+ = 2(SxSx + SySy)
                # So to get J(SxSx + SySy), we need J/2 * (S+S- + S-S+)
                ampo += weight * J/2, "S+", i, "S-", j
                ampo += weight * J/2, "S-", i, "S+", j
                # ZZ term
                ampo += weight * J * Δ, "Sz", i, "Sz", j
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

function run_simulation(Δ, σ, J, N_vals)

    shuffled_N_vals = shuffle(N_vals)
    local_averaging = σ == 0.0 ? 1 : AVG_OVER_GRAPHS

    println("    Averaging $local_averaging times.")

    data = Dict(TEMP_DATA) # (J, Delta, sigma, N) = (avg_bond_dim, err)
    println(data)

    rl_lock = ReentrantLock()

    # Open a file to log (J, Δ, σ, N) values
    log_file_path = "simulation_log.txt"

    @threads for i in 1:length(N_vals)
        N = shuffled_N_vals[i]  # Define N at the start of the loop

        println("    Computing (J=$J, Δ=$Δ, σ=$σ, N=$N)...")

        # Check if we have already done this N before
        already_done = false
        lock(rl_lock) do
            if haskey(data, (J, Δ, σ, N))
                println("        Already completed (J=$J, Δ=$Δ, σ=$σ, N=$N). Skipping")
                already_done = true
            end
        end

        if already_done
            continue  # Skip this iteration of the loop
        end

        Χs = []

        for _ in 1:local_averaging
            wam = generate_fully_connected_wam(N, σ)
            ψ_mps, sites = create_MPS(N)
            H = create_xxz_hamiltonian_mpo(N, wam, J, Δ, sites)
            _, ψ_gs = solve_xxz_hamiltonian_dmrg(H, ψ_mps, NUM_SWEEPS, MAX_BOND_DIM, 1e-10)
            bond_dim = maxlinkdim(ψ_gs)
            push!(Χs, bond_dim)
        end

        # Calculate average bond dimension and error
        avg_bond_dim = mean(Χs)
        error = σ != 0.0 ? Statistics.std(Χs) : 0

        # Save the result for this N
        lock(rl_lock) do
            data[(J, Δ, σ, N)] = (avg_bond_dim, error)
            jldsave(TEMP_DATA_STORAGE_FILE; data)
            println("        Found results $avg_bond_dim ± $error for (J=$J, Δ=$Δ, σ=$σ, N=$N)")
            println("        Computed and saved results for (J=$J, Δ=$Δ, σ=$σ, N=$N).")

            # Write (J, Δ, σ, N) to the log file
            open(log_file_path,"a") do sim_log
                println(sim_log,"Completed: J=$J, Δ=$Δ, σ=$σ, N=$N")
            end
        end
    end

    close(log_file)  # Close the log file
    return data
end

function main()

    println("Running simulation...")

    overall_data = Dict()

    for Δ in Δ_vals, σ in σ_vals
        println("Starting simulation for Δ=$Δ and σ=$σ...")
        run_data = run_simulation(Δ, σ, J, N_vals)
        
        merge!(overall_data, run_data)

    end

    jldsave(OVERALL_DATA_STORAGE_FILE; overall_data)
    
    println("Simulation complete. Data saved to 'overall_data.jld2'.")
end

main()