using ProgressMeter
using Random
using Statistics
using ThreadsX
using JLD2
using Plots

include("../james/funcs/adjacency.jl")  
include("../james/funcs/mps.jl")  
include("../james/funcs/hamiltonian.jl")

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

function load_and_plot_surface(sigma_vals)
    println("Loading data for plotting...")

    datafile = "james/j_vs_delta_vs_N_data.jld2"
    data = jldopen(datafile, "r") do file
        read(file, "data")
    end

    for σ in sigma_vals
        # Collect unique sorted values
        Js = sort(unique([J for (J, Δ, N, σ_key) in keys(data) if σ_key == σ]))
        Deltas = sort(unique([Δ for (J, Δ, N, σ_key) in keys(data) if σ_key == σ]))
        Ns = sort(unique([N for (J, Δ, N, σ_key) in keys(data) if σ_key == σ]))

        # Prepare a 3D array for bond_dims
        bond_dim_array = fill(NaN, length(Js), length(Deltas), length(Ns))
        for ((J, Δ, N, σ_key), (bond_dim, error)) in data
            if σ_key == σ
                i = findfirst(isequal(J), Js)
                j = findfirst(isequal(Δ), Deltas)
                k = findfirst(isequal(N), Ns)
                bond_dim_array[i, j, k] = bond_dim
            end
        end

        # Plot for each N as a surface in (J, Δ)
        for (k, N) in enumerate(Ns)
            zvals = bond_dim_array[:, :, k]
            if all(isnan, zvals)
                continue
            end

        end
        
    end
end

# main()
# load_and_plot_surface([0.002])

function load_and_plot_scatter(sigma_vals)
    println("Loading data for plotting...")

    datafile = "james/j_vs_delta_vs_N_data.jld2"
    data = jldopen(datafile, "r") do file
        read(file, "data")
    end

    for σ in sigma_vals
        Js, Deltas, Ns, bond_dims = Float64[], Float64[], Int[], Float64[]
        for ((J, Δ, N, σ_key), (bond_dim, _)) in data
            if σ_key == σ
                push!(Js, J)
                push!(Deltas, Δ)
                push!(Ns, N)
                push!(bond_dims, bond_dim)
            end
        end
        if !isempty(Js)
            plt = scatter3d(Js, Deltas, Ns, marker_z=bond_dims, xlabel="J", ylabel="Δ", zlabel="N",
                      color=:viridis, markersize=8, title="Bond Dimension for σ=$σ")
            display(plt)
        end
    end
end

# main()
# load_and_plot_scatter([0.002])

function plot_phase_diagrams(sigma; N_vals_to_plot=nothing)
    datafile = "james/j_vs_delta_vs_N_data.jld2"
    data = jldopen(datafile, "r") do file
        read(file, "data")
    end

    # Get all unique values
    Js = sort(unique([J for (J, Δ, N, σ_key) in keys(data) if σ_key == sigma]))
    Deltas = sort(unique([Δ for (J, Δ, N, σ_key) in keys(data) if σ_key == sigma]))
    Ns = sort(unique([N for (J, Δ, N, σ_key) in keys(data) if σ_key == sigma]))

    if N_vals_to_plot === nothing
        N_vals_to_plot = Ns
    end

    nrows = ceil(Int, sqrt(length(N_vals_to_plot)))
    ncols = ceil(Int, length(N_vals_to_plot) / nrows)
    plt = plot(layout=(nrows, ncols), size=(300*ncols, 300*nrows))

    for (i, N) in enumerate(N_vals_to_plot)
        z = fill(NaN, length(Js), length(Deltas))
        for (j, J) in enumerate(Js), (k, Δ) in enumerate(Deltas)
            key = (J, Δ, N, sigma)
            if haskey(data, key)
                z[j, k] = data[key][1]  # bond_dim
            end
        end
        contourf!(plt[i], Js, Deltas, z', xlabel="J", ylabel="Δ", title="N=$N", color=:viridis)
    end
    display(plt)
end

# Example usage:
plot_phase_diagrams(0.002; N_vals_to_plot=[10, 20, 30, 40, 50, 60, 70])

function plot_3d_surfaces(sigma; N_vals_to_plot=nothing)
    datafile = "james/j_vs_delta_vs_N_data.jld2"
    data = jldopen(datafile, "r") do file
        read(file, "data")
    end

    Js = sort(unique([J for (J, Δ, N, σ_key) in keys(data) if σ_key == sigma]))
    Deltas = sort(unique([Δ for (J, Δ, N, σ_key) in keys(data) if σ_key == sigma]))
    Ns = sort(unique([N for (J, Δ, N, σ_key) in keys(data) if σ_key == sigma]))

    if N_vals_to_plot === nothing
        N_vals_to_plot = Ns
    end

    println("Js: ", Js)
    println("Deltas: ", Deltas)
    println("N_vals_to_plot: ", N_vals_to_plot)

    plt = plot3d(title="Bond Dimension Surfaces for σ=$sigma", xlabel="J", ylabel="Δ", zlabel="Bond Dim")

    colors = [:red, :blue, :green, :orange, :purple, :brown, :pink, :gray, :black, :cyan]

    for (idx, N) in enumerate(N_vals_to_plot)
        z = fill(NaN, length(Js), length(Deltas))
        for (j, J) in enumerate(Js), (k, Δ) in enumerate(Deltas)
            key = (J, Δ, N, sigma)
            if haskey(data, key)
                z[j, k] = data[key][1]  # bond_dim
            end
        end
        # Replace NaN values with 0.0
        z = replace(z, NaN => 0.0)
        println("Surface for N=$N: ", z)
        if all(isnan, z)
            println("Skipping N=$N as all values are NaN.")
            continue
        end
        surface!(plt, Js, Deltas, z', color=colors[mod1(idx, length(colors))], label="N=$N", alpha=0.7, colorbar=false)
    end
    display(plt)
end
# plot_3d_surfaces(0.002; N_vals_to_plot=[10, 20, 30, 40, 50, 60, 70])


function plot_delta_vs_N_vs_bond_dim(sigma; J_vals_to_plot=nothing, N_vals_to_plot=nothing)
    datafile = "james/j_vs_delta_vs_N_data.jld2"
    data = jldopen(datafile, "r") do file
        read(file, "data")
    end

    keys_sorted_by_N = sort(collect(keys(data)), by=x -> x[3])  # Convert keys to an array before sorting

    Js = [key[1] for key in keys_sorted_by_N if key[4] == sigma]  # Extract J values
    Deltas = [key[2] for key in keys_sorted_by_N if key[4] == sigma]  # Extract Δ values
    Ns = [key[3] for key in keys_sorted_by_N if key[4] == sigma]  # Extract N values

    if J_vals_to_plot === nothing
        J_vals_to_plot = Js
    end
    if N_vals_to_plot === nothing
        N_vals_to_plot = Ns
    end

    colors = [:red, :blue, :green, :orange, :purple, :brown, :pink, :gray, :black, :cyan]

    for J in J_vals_to_plot
        plt = plot3d(title="Δ vs N vs Bond Dim for J=$J, σ=$sigma", xlabel="Δ", ylabel="N", zlabel="Bond Dim")

        z = fill(NaN, length(N_vals_to_plot), length(Deltas))  # Initialize z as a matrix
        for (k, N) in enumerate(N_vals_to_plot)
            for (i, Δ) in enumerate(Deltas)
                key = (J, Δ, N, sigma)
                if haskey(data, key)
                    z[k, i] = data[key][1]  # bond_dim
                end
            end
        end
        # Replace NaN values with 0.0
        z = replace(z, NaN => 0.0)
        println("Surface for J=$J: ", z)
        if all(isnan, z)
            println("Skipping J=$J as all values are NaN.")
            continue
        end
        surface!(plt, Deltas, N_vals_to_plot, z, color=:viridis, alpha=0.7)
        display(plt)
    end
end

# # Example usage:
# plot_delta_vs_N_vs_bond_dim(0.002; J_vals_to_plot=[-1.0, 1.0])