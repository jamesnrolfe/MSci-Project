using JLD2
using Plots
using Statistics

println("Starting Julia plotting script...")

data_filename = joinpath(@__DIR__, "conc_star_data.jld2")
plot_filename = joinpath(@__DIR__, "conc_star_plot.png")

# Check if data file exists
if !isfile(data_filename)
    println("ERROR: Data file not found: $data_filename")
    println("Please run the simulation script first to generate this file.")
else
    println("Loading data from $data_filename...")
    
    # Load the data
    file = jldopen(data_filename, "r")
    results = read(file, "results")
    N_range = read(file, "N_range")
    sigma_values = read(file, "sigma_values")
    close(file)
    
    
    
    if isempty(sigma_values)
        error("No sigma values found in data file.")
    end
    σ = sigma_values[1] 
    
    data_for_sigma = results[σ]
    
    plot_N_values = Int[]
    y_values_mean = Float64[]
    y_values_error = Float64[] # Std dev *between* pairs

    # We loop over N_range and check if N exists as a key in the results
    for N in N_range
        # Check if this data point was computed and stored
        if haskey(data_for_sigma, N)
            # Access the data using N as the key
            data_N = data_for_sigma[N]
            
            # data_N.avg is the Vector{Float64} [C(1,2), C(1,3), ..., C(1,N)]
            avg_concurrence_vector = data_N.avg
            
            if !isempty(avg_concurrence_vector)
                # Calculate the average concurrence over all pairs j
                mean_C = mean(avg_concurrence_vector)
                
                # Calculate the std dev of concurrence over all pairs j
                std_C = std(avg_concurrence_vector)
                
                push!(plot_N_values, N)
                push!(y_values_mean, mean_C)
                push!(y_values_error, std_C)
            end
        end
    end
    
    if isempty(plot_N_values)
        println("No valid data found to plot. Exiting.")
    else
        println("Data processed. Generating plot...")

        plt = plot(
            plot_N_values,
            y_values_mean,
            yerror = y_values_error,
            title = "Average Concurrence (Center to Outer) vs. System Size",
            xlabel = "System Size (N)",
            ylabel = "Average Concurrence C(1, j)",
            label = "Avg. C(1, j) (over j) for σ=$σ",
            legend = :topright,
            marker = :circle,
            markersize = 4,
            linewidth = 2,
            xticks = N_range, 
            grid = true,
            gridstyle = :dash,
            gridalpha = 0.5
        )
        
        # Save the plot
        savefig(plt, plot_filename)
        
        println("Plot saved successfully as $plot_filename")
    end
end