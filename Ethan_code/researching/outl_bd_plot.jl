using JLD2, Plots

function create_comparison_plot()
    println("Loading data...")
    
    # --- This file matches your new data script ---
    data_filename = joinpath(@__DIR__, "outlier_graphs_data.jld2")
    plot_filename = joinpath(@__DIR__, "outlier_graphs_plot.png")

    if !isfile(data_filename)
        println("ERROR: Data file not found: $data_filename")
        return
    end

    local results_standard, results_outlier, N_range, sigma_values
    try
        jldopen(data_filename, "r") do file
            # --- Loads the new variable names ---
            results_standard = read(file, "results_standard")
            results_outlier = read(file, "results_outlier")
            N_range = read(file, "N_range")
            sigma_values = read(file, "sigma_values")
        end
    catch e
        println("ERROR: Could not load data from $data_filename. Error: $e")
        return
    end


    gr()
    
    p = plot(
        title = "Bond Dimension Comparison: Standard vs. Outlier Graphs",
        xlabel = "System Size (N)",
        ylabel = "Average Max Bond Dimension",
        legend = :topleft,
        size = (1000, 600)
    )

    # Define colors and linestyles for clarity
    colors = [:blue, :red] # Colors for sigma_values[1] and sigma_values[2]
    
    for (idx, σ) in enumerate(sigma_values)
        color = colors[idx]
        
        # --- Plots the "Standard" (fully connected) data ---
        plot!(p,
            N_range,
            results_standard[σ].avg,
            ribbon = results_standard[σ].err,
            label = "Standard (σ=$σ)",
            color = color,
            linestyle = :solid,
            fillalpha = 0.15
        )
        
        # --- Plots the "Outlier" data ---
        plot!(p,
            N_range,
            results_outlier[σ].avg,
            ribbon = results_outlier[σ].err,
            label = "Outlier (σ=$σ)",
            color = color,
            linestyle = :dash,
            fillalpha = 0.15
        )
    end

    try
        savefig(p, plot_filename)
        println("Plot saved successfully to $plot_filename")
    catch e
        println("ERROR: Could not save plot. Error: $e")
    end
end

create_comparison_plot()