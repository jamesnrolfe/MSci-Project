using Markdown
using InteractiveUtils
using Graphs, Random, Statistics
using Plots, Colors
using ITensors, ITensorMPS, LinearAlgebra
using JLD2 
using Base.Threads  # <-- 1. IMPORT THREADS

# seed for reproducibility
Random.seed!(1234);


function create_MPS(L::Int)
    sites = siteinds("S=1/2", L; conserve_qns=true)
    initial_state = [isodd(i) ? "Up" : "Dn" for i in 1:L]
    ψ₀ = randomMPS(sites, initial_state)
    return ψ₀, sites
end

"""
Creates a weighted adjacency matrix for a completely connected graph.
"""
function create_weighted_adj_mat(N::Int, σ::Float64; μ::Float64=1.0)
    if σ == 0.0
        A = ones(Float64, N, N)
        A -= Matrix{Float64}(I, N, N)
        return A
    end
    A = zeros(Float64, N, N)
    for i in 1:N, j in (i+1):N
        weight = μ + σ * randn()
        A[i, j] = A[j, i] = weight
    end
    return A
end

"""
Creates the MPO for the XXZ Hamiltonian on a graph with weighted interactions.
"""
function create_weighted_xxz_mpo(N::Int, adj_mat, sites; J::Float64, Δ::Float64)
    ampo = OpSum()
    for i in 1:N-1
        for j in i+1:N
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



# runs the simulation over a 2D parameter space (N and σ)
# and stores the results in a matrix suitable for a surface plot.
function run_simulation_and_plot_sigma()

    N_range = 10:1:100
    sigma_range = 0.0:0.0001:0.002 
    num_graphs_avg = 10

    num_sweeps = 30
    max_bond_dim_limit = 250
    cutoff = 1E-10
    μ = 1.0

    # Matrix to store the average bond dimension for each (N, σ) pair
    avg_bond_dims = zeros(Float64, length(N_range), length(sigma_range))


    for (i, N) in enumerate(N_range)
        for (j, σ) in enumerate(sigma_range)
            
            # --- 2. PRE-ALLOCATE THE ARRAY ---
            # We change from `bond_dims_for_avg = Float64[]` to:
            bond_dims_for_avg = zeros(Float64, num_graphs_avg)
            
            # --- 3. ADD THE THREADS MACRO ---
            # We change `for _ in 1:num_graphs_avg` to:
            Threads.@threads for k in 1:num_graphs_avg
                # This whole block now runs in parallel
                ψ₀, sites = create_MPS(N)
                adj_mat = create_weighted_adj_mat(N, σ; μ=μ)
                H_mpo = create_weighted_xxz_mpo(N, adj_mat, sites; J=-0.5, Δ=0.5)

                sweeps = Sweeps(num_sweeps)
                setmaxdim!(sweeps, max_bond_dim_limit)
                setcutoff!(sweeps, cutoff)

                _, ψ_gs = dmrg(H_mpo, ψ₀, sweeps; outputlevel=0)
                
                # --- 4. WRITE TO INDEX INSTEAD OF PUSH! ---
                # We change `push!(bond_dims_for_avg, ...)` to:
                bond_dims_for_avg[k] = maxlinkdim(ψ_gs)
            end
            avg_bond_dims[i, j] = mean(bond_dims_for_avg)
        end
        println("Completed N = $N") # Feedback for each completed N
    end


    # Convert ranges to arrays for plotting
    N_values = collect(N_range)
    sigma_values = collect(sigma_range)

    # Note: plotlyjs() and plot() will not produce a visible plot
    # in a batch script. They are harmless but will be ignored.
    plotlyjs() 
    
    plt = plot(N_values, sigma_values, avg_bond_dims',
        st=:surface,
        title="Maximum Bond Dimension",
        xlabel="System Size",
        ylabel="σ",
        zlabel="Maximum Bond Dimension",
        camera=(50, 30),      # Adjust viewing angle (azimuth, elevation)
        c=cgrad(:inferno),    # Use a colormap 
        legend=false
    )
    
    return plt, avg_bond_dims, N_range, sigma_range
end

println("Starting calculations...")
plt, avg_bond_dims, N_range, sigma_range = run_simulation_and_plot_sigma();

println("Calculations finished. Saving data...")
filename = "surface_plot_sigma_data.jld2"
jldsave(filename; avg_bond_dims, N_range, sigma_range)
println("Data saved successfully.\n")
