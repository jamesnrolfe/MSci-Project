using JLD2
using Plots

# Set the backend for Plots.jl
gr()

"""
Main function to load data and generate the plot.
"""
function create_plot()
    filename = joinpath(@__DIR__, "outlier_graphs_data.jld2")

    if !isfile(filename)
        println("Error: Data file not found!")
        println("Please run 'outlier_graphs.jl' first to generate 'outlier_graphs_data.jld2'.")
        return
    end

    println("Loading data from $filename...")

    # Load all data from the JLD2 file
    local N_range, sigma_values, results_standard, results_outlier, J_coupling, Delta_coupling
    try
        jldopen(filename, "r") do file
            N_range = read(file, "N_range")
            sigma_values = read(file, "sigma_values")
            results_standard = read(file, "results_standard")
            results_outlier = read(file, "results_outlier")
            J_coupling = read(file, "J_coupling")
            Delta_coupling = read(file, "Delta_coupling")
        end
    catch e
        println("Error loading data from JLD2 file: $e")
        return
    end

    println("Data loaded. Generating plots...")

    # --- Amended Layout Logic ---
    # We will create a 3x3 grid, one plot for each N
    if length(N_range) != 9
        println("Warning: Expected 9 N-values for a 3x3 grid, but found $(length(N_range)).")
        println("Adjusting layout to fit all N values...")
    end
    
    num_plots = length(N_range)
    # Calculate layout, aiming for 3 columns
    num_cols = 3
    num_rows = ceil(Int, num_plots / num_cols)
    plot_layout = (num_rows, num_cols) # e.g., (3, 3) for 9 plots

    println("Found $num_plots N values. Creating a $plot_layout plot grid.")

    plot_list = [] # Array to hold each individual subplot
    
    # X-axis for the bar charts will be sigma values
    sigma_categories = string.(sigma_values)

    for (i, N) in enumerate(N_range)
        # Extract data for this N across all sigmas
        avg_std_for_N = [results_standard[σ].avg[i] for σ in sigma_values]
        err_std_for_N = [results_standard[σ].err[i] for σ in sigma_values]
        avg_out_for_N = [results_outlier[σ].avg[i] for σ in sigma_values]
        err_out_for_N = [results_outlier[σ].err[i] for σ in sigma_values]

        # Combine data for grouped bar chart
        # hcat creates a matrix where each column is a group
        data_matrix = hcat(avg_std_for_N, avg_out_for_N)
        error_matrix = hcat(err_std_for_N, err_out_for_N)

        # Create the plot for this N
        p = groupedbar(
            sigma_categories,
            data_matrix;
            yerr = error_matrix,
            label = ["Standard" "Outlier"],
            legend = :topleft,
            xlabel = "Disorder (σ)",
            ylabel = "Avg. Max Bond Dim",
            title = "System Size N = $N"
        )

        push!(plot_list, p)
    end

    # Combine all plots into a single figure with a main title
    main_title = "Entanglement vs. Disorder for Standard vs. Outlier Graphs (J=$J_coupling, Δ=$Delta_coupling)"
    
    # Adjust size for a 3x3 grid.
    final_plot = plot(
        plot_list...;
        layout = plot_layout,
        plot_title = main_title,
        size = (1500, 1500) # (width, height) - good for 3x3
    )

    # Save the final plot
    output_filename = "outlier_entanglement_grid_plot.png"
    savefig(final_plot, output_filename)

    println("Plot successfully saved to $output_filename")
end

# Run the plot creation
create_plot()