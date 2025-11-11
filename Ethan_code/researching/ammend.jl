using JLD2
using Plots # Added for plotting

# Define the file to modify
filename = joinpath(@__DIR__,"full_bd_data.jld2")

# --- Define operations ---
sigma_to_remove = 0.001
indices_to_fix = [66, 75]

println("Loading data from $filename...")

local results, N_range, sigma_values

# --- 1. Load all data from the file ---
try
    jldopen(filename, "r") do file
        # Convert the lazy JLD2.SerializedDict to a standard in-memory Dict
        global results = Dict(read(file, "results"))
        global N_range = read(file, "N_range")
        global sigma_values = read(file, "sigma_values")
    end
catch e
    println("ERROR: Could not read data from '$filename'. Aborting. Error: $e")
    return
end

println("Data loaded.")

# --- 2. Remove data for sigma = 0.001 ---
println("\n--- Removing data for sigma = $sigma_to_remove ---")
local new_sigma_values

# Remove the key from the 'results' dictionary
if haskey(results, sigma_to_remove)
    delete!(results, sigma_to_remove)
    println("Successfully removed entry for $sigma_to_remove from 'results'.")
else
    println("Key $sigma_to_remove not found in 'results'. No change made.")
end

# Filter the 'sigma_values' array to remove the value
new_sigma_values = filter(s -> s != sigma_to_remove, sigma_values)

if length(new_sigma_values) < length(sigma_values)
    println("Successfully removed $sigma_to_remove from 'sigma_values' array.")
else
    println("Value $sigma_to_remove not found in 'sigma_values' array. No change made.")
end

println("Remaining sigma keys in 'results': $(keys(results))")

# --- 3. Modify the remaining data in memory (Interpolation) ---
println("\n--- Applying linear interpolation to indices: $indices_to_fix ---")
try
    # Loop over the *remaining* keys in the results dictionary
    for sigma_key in keys(results)
        println("--- Processing sigma = $sigma_key ---")
        
        avg_data = results[sigma_key].avg
        err_data = results[sigma_key].err
        
        for idx in indices_to_fix
            N_val = N_range[idx]
            println("Fixing data at index $idx (N = $N_val)")

            # Interpolate 'avg' data
            avg_before = avg_data[idx - 1]
            avg_after  = avg_data[idx + 1]
            avg_new    = (avg_before + avg_after) / 2.0
            
            println("    AVG: Changed $(avg_data[idx]) -> $avg_new")
            avg_data[idx] = avg_new # Modify the array in-place
            
            # Interpolate 'err' data
            err_before = err_data[idx - 1]
            err_after  = err_data[idx + 1]
            err_new    = (err_before + err_after) / 2.0
            
            println("    ERR: Changed $(err_data[idx]) -> $err_new")
            err_data[idx] = err_new # Modify the array in-place
        end
    end
catch e
    println("ERROR: An error occurred during interpolation. Aborting. Error: $e")
    return
end

# --- 4. Save the modified data back to the file ---
try
    jldsave(filename; 
        results=results, 
        N_range=N_range, 
        sigma_values=new_sigma_values # Save the *new* sigma array
    )
    println("\nSuccessfully saved modified and interpolated data back to $filename.")
catch e
    println("ERROR: Could not save modified data to '$filename'. Error: $e")
end

# --- 5. Print the final data ---
println("\n--- Final Data Structure ---")
println(results)


# --- 6. PLOTTING SECTION ---
println("\n--- Starting Plot Generation ---")

# --- Define file paths ---
# We already have the connected data in memory (in 'results', 'N_range', 'new_sigma_values')
linear_data_file = joinpath(@__DIR__, "lin_bd_data.jld2")
plot_filename = joinpath(@__DIR__, "lin_con_comparison_plot.png")

if !isfile(linear_data_file)
    println("ERROR: Linear data file not found: $linear_data_file. Aborting plot.")
    return
end

# --- Assign in-memory data to new variables for clarity ---
results_connected = results
N_range_connected = N_range
sigma_values_touse = new_sigma_values # These are [0.0, 0.002]

local results_linear, N_range_linear

# --- Load Linear Data ---
try
    jldopen(linear_data_file, "r") do file
        global results_linear = read(file, "results")
        global N_range_linear = read(file, "N_range")
        # We don't need sigma_values_linear, we will use sigma_values_touse
    end
    println("Loaded linear data from $linear_data_file")
catch e
    println("ERROR: Could not load data from $linear_data_file. Error: $e")
    return
end

# --- Create Plot ---
gr()

p = plot(
    title = "Bond Dimension Comparison: Connected vs. Linear Models",
    xlabel = "System Size (N)",
    ylabel = "Average Max Bond Dimension",
    legend = :topleft,
    size = (1000, 600)
)

colors = [:blue, :purple, :red]

for (idx, σ) in enumerate(sigma_values_touse)
    color = colors[idx]
    
    # Plot Connected Data (from memory)
    if haskey(results_connected, σ)
        plot!(p,
            N_range_connected,
            results_connected[σ].avg,
            ribbon = results_connected[σ].err,
            label = "Connected (σ=$σ)",
            color = color,
            linestyle = :solid,
            fillalpha = 0.15
        )
    end
    

end

# --- Save Plot ---
try
    savefig(p, plot_filename)
    println("Plot saved successfully to $plot_filename")
catch e
    println("ERROR: Could not save plot. Error: $e")
end

println("\nScript finished.")