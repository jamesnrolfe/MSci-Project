using ProgressMeter
using Random
using Statistics
using ThreadsX
using JLD2

include("../funcs/adjacency.jl")  
include("../funcs/mps.jl")  
include("../funcs/hamiltonian.jl")

J_vals = [-1.0, 1.0]
Δ_vals = [-1.0, 1.0]
σ_vals = [0.0, 0.002]
N_vals = shuffle(10:2:75) # number of nodes to test
# do them in a random order to make eta for progress meter more accurate - larger N take longer

MAX_BOND_DIM = 500
NUM_GRAPHS_TO_AVG = 3
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
    # Try to load existing data
    datafile = "james/j_vs_delta_vs_N_data.jld2"
    data = Dict{Tuple{Float64, Float64, Int64, Float64}, Any}()
    if isfile(datafile)
        try
            loaded = JLD2.load(datafile, "data")
            if loaded isa Dict
                data = loaded
                println("Loaded existing data with $(length(data)) entries.")
            end
        catch e
            println("Could not load existing data: $e")
        end
    end

    num_completed = Threads.Atomic{Int}(length(data))
    total_runs = length(J_vals) * length(Δ_vals) * length(N_vals) * length(σ_vals)
    avg_time_per_run = Threads.Atomic{Float64}(0.0)

    println("Starting simulations...")
    println("Total runs to complete: $total_runs - $(num_completed[]) already completed.")

    N_results = Vector{Any}(undef, length(N_vals))
    shuffled_N_indices = shuffle(1:length(N_vals))

    for J in J_vals, Δ in Δ_vals
        println("\nRunning simulations for J=$J, Δ=$Δ")
        Threads.@threads for i in 1:length(N_vals)
            n_idx = shuffled_N_indices[i]
            N = N_vals[n_idx]
            N_data = Dict{Float64, Tuple{Float64, Float64}}()
            for σ in σ_vals
                key = (J, Δ, N, σ)
                if haskey(data, key)
                    println("Skipping J=$J, Δ=$Δ, N=$N, σ=$σ (already computed)")
                    continue
                end
                timer_start = time()
                result, error = run_simulation(J, Δ, N, σ)
                timer_end = time()
                N_data[σ] = (result, error)
                data[key] = (result, error)
                Threads.atomic_add!(num_completed, 1)
                avg_time_per_run[] = ((avg_time_per_run[] * (num_completed[] - 1)) + (timer_end - timer_start)) / num_completed[]
                est_time_remaining = avg_time_per_run[] * (total_runs - num_completed[])
                est_time_remaining_hrs = est_time_remaining / 3600
                println("J=$J, Δ=$Δ, N=$N, σ=$σ: $result ± $error")
                println("       -> Time taken: $(timer_end - timer_start) seconds")
                println("       -> Estimated time remaining: $(round(est_time_remaining_hrs, digits=2)) hours")
                # Save after each new result
                try
                    JLD2.jldsave(datafile; data)
                catch e
                    println("Error saving data: $e")
                end
            end
            N_results[n_idx] = (J, Δ, N, N_data)
        end
    end

    println("Simulations complete.")
    println("Data saved to $datafile")
end

main()