using JLD2

# Define the file to modify and the key to remove
filename = joinpath(@__DIR__,"full_bd_data.jld2")

indices_to_fix = [66, 75]

println("Loading data from $filename...")

local results, N_range, sigma_values

# --- 1. Load all data from the file ---
try
    jldopen(filename, "r") do file
        global results = read(file, "results")
        global N_range = read(file, "N_range")
        global sigma_values = read(file, "sigma_values")
    end
catch e
    println("ERROR: Could not read data from '$filename'. Aborting. Error: $e")
    exit()
end

println("Data loaded. Applying linear interpolation to indices: $indices_to_fix")

# --- 2. Modify the data in memory ---
try
    for sigma_key in keys(results)
        println("--- Processing sigma = $sigma_key ---")
        
        # Get references to the arrays
        avg_data = results[sigma_key].avg
        err_data = results[sigma_key].err
        
        for idx in indices_to_fix
            N_val = N_range[idx]
            println("Fixing data at index $idx (N = $N_val)")

            # --- Interpolate 'avg' data ---
            avg_before = avg_data[idx - 1]
            avg_after  = avg_data[idx + 1]
            avg_new    = (avg_before + avg_after) / 2.0
            
            println("    AVG: Changed $(avg_data[idx]) -> $avg_new (from $avg_before and $avg_after)")
            avg_data[idx] = avg_new # Modify the array
            
            # --- Interpolate 'err' data ---
            err_before = err_data[idx - 1]
            err_after  = err_data[idx + 1]
            err_new    = (err_before + err_after) / 2.0
            
            println("    ERR: Changed $(err_data[idx]) -> $err_new (from $err_before and $err_after)")
            err_data[idx] = err_new # Modify the array
        end
    end
catch e
    println("ERROR: An error occurred during interpolation. Aborting. Error: $e")
    exit()
end

# --- 3. Save the modified data back to the file ---
try
    jldsave(filename; 
        results=results, 
        N_range=N_range, 
        sigma_values=sigma_values
    )
    println("\nSuccessfully saved interpolated data back to $filename.")
catch e
    println("ERROR: Could not save modified data to '$filename'. Error: $e")
end

println("Script finished.")