using ProgressMeter

include("../funcs/adjacency.jl")  
include("../funcs/mps.jl")  
include("../funcs/hamiltonian.jl")

J_vals = [-1.0, -0.5, 0.5, 1.0]
Δ_vals = [-1.0, -0.5, 0.5, 1.0]
σ_vals = [0.0, 0.001, 0.002]
N_vals = shuffle(10:2:100) # number of nodes to test
# do them in a random order to make eta for progress meter more accurate - larger N take longer

MAX_BOND_DIM = 1000
NUM_GRAPHS_TO_AVG = 5
NUM_SWEEPS = 30


function run_simulation(J, Δ, N, σ)
    # Placeholder for the actual simulation code
    local_averaging = σ != 0.0 ? NUM_GRAPHS_TO_AVG : 1 # if sigma is 0, only need to run once
    bond_dims = Int[]
    
    for _ in 1:local_averaging
        wam = generate_fully_connected_wam(N, σ)
        ψ_mps, sites = create_MPS(N)
        H = create_xxz_hamiltonian_mpo(N, wam, J, Δ, sites)
        _, ψ_gs = solve_xxz_hamiltonian_dmrg(H, ψ_mps, NUM_SWEEPS, MAX_BOND_DIM, 1e-10)
        bond_dim = maxlinkdim(ψ_gs)
        push!(bond_dims, bond_dim)
    end
    
    avg_bond_dim = Statistics.mean(bond_dims)
    # standard error on the mean
    if σ != 0.0
        error = Statistics.std(bond_dims) / sqrt(length(bond_dims)) # std/sqrt(sample size)
    else
        error = 0.0
    end
    return (avg_bond_dim, error)

end



function main()

    data = Dict{Tuple{Float64, Float64, Int64, Float64}, Any}() # to store results
    # should be stored in the form data[(J, Δ, N, σ)] = result

    for J in J_vals
        for Δ in Δ_vals
            progress_meter = Progress(length(N_vals) * length(σ_vals), show_eta=true, desc="J=$J, Δ=$Δ")
            for N in N_vals
                for σ in σ_vals
                    # Run the simulation and store the result
                    result, error = run_simulation(J, Δ, N, σ)
                    data[(J, Δ, N, σ)] = (result, error)
                    next!(progress_meter)
                end
            end
        end
    end
end

main()