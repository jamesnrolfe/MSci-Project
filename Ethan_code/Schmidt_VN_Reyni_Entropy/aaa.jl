using JLD2
using FileIO

# --- Configuration ---
input_filename = joinpath(@__DIR__, "high_prec_spec_data_0.0_200.jld2")
output_filename = joinpath(@__DIR__, "high_prec_spec_data_0.0.jld2")
max_N = 100

# --- Main Processing ---
if !isfile(input_filename)
    error("Input file not found: $input_filename. Please make sure it is in the same directory.")
end

println("Loading data from $input_filename...")
data = load(input_filename)

# Extract original data
original_results = data["entanglement_spectrum_results"]
original_N_values = data["N_values"]
sigma_val = data["σ_val"]
precision = haskey(data, "precision") ? data["precision"] : "unknown"

println("Original N values: $original_N_values")

# Filter N_values
# We handle both StepRange (e.g., 10:10:200) and Vectors
new_N_values = filter(x -> x <= max_N, collect(original_N_values))

# Filter Results Dictionary
new_results = Dict{Int, Vector{Float64}}()
for (N, coeffs) in original_results
    if N <= max_N
        new_results[N] = coeffs
    end
end

println("New N values: $new_N_values")
println("Saving filtered data to $output_filename...")

# Save to new file
save(output_filename, Dict(
    "entanglement_spectrum_results" => new_results,
    "N_values" => new_N_values,
    "σ_val" => sigma_val,
    "precision" => precision
))

println("Done! Created $output_filename")