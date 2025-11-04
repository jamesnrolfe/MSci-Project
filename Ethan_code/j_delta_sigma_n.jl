using Random, Statistics
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

"""
Internal function to run the DMRG simulation and average results.
"""
function perform_dmrg_avg(N, σ, J, Δ, μ, num_graphs_avg, num_sweeps, max_bond_dim, cutoff)
    local_averaging = σ != 0.0 ? num_graphs_avg : 1
    bond_dims = Int[]
    
    for _ in 1:local_averaging
        adj_mat = create_weighted_adj_mat(N, σ; μ=μ)
        ψ_mps, sites = create_MPS(N)
        H = create_weighted_xxz_mpo(N, adj_mat, sites; J=J, Δ=Δ)
        
        sweeps = Sweeps(num_sweeps)
        setmaxdim!(sweeps, max_bond_dim)
        setcutoff!(sweeps, cutoff)

        _, ψ_gs = dmrg(H, ψ_mps, sweeps; outputlevel=0)
        bond_dim = maxlinkdim(ψ_gs)
        push!(bond_dims, bond_dim)
    end
    
    avg_bond_dim = Statistics.mean(bond_dims)
    if σ != 0.0
        error = Statistics.std(bond_dims) / sqrt(length(bond_dims))
    else
        error = 0.0
    end
    return (avg_bond_dim, error)
end

"""
Main simulation function, structured to match the other files.
"""
function run_simulation_all(
    data::Dict,
    data_lock::SpinLock,
    J_vals,
    Δ_vals,
    N_vals,
    σ_vals,
    num_graphs_avg,
    num_sweeps,
    max_bond_dim,
    cutoff,
    μ
)
    filename = joinpath(@__DIR__, "j_delta_sigma_n_data.jld2")
    println("Data will be saved to: $filename")

    for N in N_vals
        

        first_key = (J_vals[1], Δ_vals[1], N, σ_vals[1])
        if haskey(data, first_key)
            println("Skipping N = $N, results already loaded/computed.")
            continue
        end

        println("Running for N = $N")
        for J in J_vals
            for Δ in Δ_vals
                Threads.@threads for σ in σ_vals
                    key = (J, Δ, N, σ)
                    
                    if haskey(data, key)
                        continue
                    end

                    result, error = perform_dmrg_avg(N, σ, J, Δ, μ, num_graphs_avg, num_sweeps, max_bond_dim, cutoff)
                    
                    lock(data_lock) do
                        data[key] = (result, error)
                    end
                end
            end
        end

        println("Completed N = $N")
        try
            jldsave(filename; data, J_vals, Δ_vals, N_vals, σ_vals)
            println("Checkpoint saved for N = $N")
        catch e
            println("WARNING: Could not save checkpoint for N = $N. Error: $e")
        end
    end
    println("...calculations finished.")
end


println("Starting calculations...")

J_vals = [-1.0, -0.5, 0.5, 1.0]
Δ_vals = [-1.0, -0.5, 0.5, 1.0]
σ_vals = [0.0, 0.001, 0.002]
N_vals = shuffle(10:2:100) 

MAX_BOND_DIM = 1000
NUM_GRAPHS_TO_AVG = 5
NUM_SWEEPS = 30
CUTOFF = 1E-10
Μ = 1.0 

filename = joinpath(@__DIR__, "j_delta_sigma_n_data.jld2")
data_lock = SpinLock() 
local data 

if isfile(filename)
    println("Found existing data file. Loading progress...")
    try
        loaded_data = jldopen(filename, "r")
        
        params_match = (
            read(loaded_data, "J_vals") == J_vals &&
            read(loaded_data, "Δ_vals") == Δ_vals &&
            read(loaded_data, "σ_vals") == σ_vals &&
            Set(read(loaded_data, "N_vals")) == Set(N_vals) 
        )

        if params_match
            println("Parameters match. Resuming...")
            data = read(loaded_data, "data")
        else
            println("WARNING: Parameters in file do not match current script. Starting from scratch.")
            data = Dict{Tuple{Float64, Float64, Int64, Float64}, Any}()
        end
        close(loaded_data)
    catch e
        println("WARNING: Could not load existing file. Starting from scratch. Error: $e")
        data = Dict{Tuple{Float64, Float64, Int64, Float64}, Any}()
    end
else
    println("No existing data file found. Starting from scratch.")
    data = Dict{Tuple{Float64, Float64, Int64, Float64}, Any}()
end

run_simulation_all(
    data,
    data_lock,
    J_vals,
    Δ_vals,
    N_vals,
    σ_vals,
    NUM_GRAPHS_TO_AVG,
    NUM_SWEEPS,
    MAX_BOND_DIM,
    CUTOFF,
    Μ
)

println("Calculations finished. Final data save...")
jldsave(filename; data, J_vals, Δ_vals, N_vals, σ_vals)
println("Data saved successfully.\n")