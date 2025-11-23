using JLD2
using FileIO
using Plots
using Statistics

filename = joinpath(@__DIR__, "high_prec_spec_data_0.0.jld2")

if !isfile(filename)
    error("Data file not found: $filename")
end



data = load(filename)
results = data["entanglement_spectrum_results"] # This is a Dict{Int, Vector{Float64}}
N_values = data["N_values"]
sigma_val = data["σ_val"]




entropies = Float64[]
sorted_N = sort(collect(keys(results)))
filter!(n -> n <= 100, sorted_N)

largest_N = maximum(sorted_N)
largest_N_spectrum = Float64[]

for N in sorted_N
    coeffs = results[N]
    
    # Calculate probabilities p = λ^2
    p = coeffs .^ 2
    
    # Filter out numerical zeros to avoid log(0)
    filter!(x -> x > 1e-20, p)
    
    # Determine normalization factor (effective number of graphs)
    num_graphs = sum(p) 
    
    # Calculate Von Neumann Entropy
    # S = - sum(p * log(p)) / num_graphs
    # (This is equivalent to averaging the entropy of each realization)
    S = -sum(p .* log.(p)) / num_graphs
    
    push!(entropies, S)
    
    if N == largest_N
        global largest_N_spectrum = coeffs
    end
end

p1 = plot(
    sorted_N, 
    entropies, 
    marker = :circle,
    label = "S_VN",
    xlabel = "System Size (N)", 
    ylabel = "Von neumann Entropy",
    title = "Von Neumann Entropy Scaling (σ = $sigma_val)",
    legend = :bottomright,
    grid = true
)

num_graphs_int = round(Int, sum(largest_N_spectrum .^ 2))

unique_coeffs = largest_N_spectrum[1:num_graphs_int:end]
probs = unique_coeffs .^ 2
filter!(x -> x > 1e-20, probs)
xi = -log.(probs)

p2 = plot(
    1:length(xi), 
    xi, 
    seriestype = :scatter,
    markersize = 3,
    color = :crimson,
    label = "   ",
    xlabel = "Coefficient Index", 
    ylabel = "Scmidt Coefficients",
    title = "Schmidt Sprectrum (N = $largest_N, σ = $sigma_val)",
    grid = true
)

final_plot = plot(p1, p2, layout = (1, 2), size = (900, 450), margin=5Plots.mm)

savefig(final_plot, joinpath(@__DIR__, "high_prec_spec_plot_0.0.png"))
