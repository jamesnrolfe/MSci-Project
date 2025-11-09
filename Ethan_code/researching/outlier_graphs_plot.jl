using JLD2
using Plots

gr()

"""
Main function to load data and generate the plot.
"""
function create_plot()
    filename = joinpath(@__DIR__, "outlier_graphs_data.jld2")

    if !isfile(filename)
        println("Error: Data file not found!")
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


    num_plots = length(sigma_values)
    plot_layout = (1, num_plots) 

    println("Found $num_plots sigma values. Creating a (1, $num_plots) plot grid.")

    plot_list = [] # Array to hold each individual subplot

    for σ in sigma_values

        # Extract data for this sigma
        avg_std = results_standard[σ].avg
        err_std = results_standard[σ].err
        avg_out = results_outlier[σ].avg
        err_out = results_outlier[σ].err

        # Create the plot for this sigma
        p = plot(
            N_range,
            avg_std;
            yerr = err_std,
            label = "Standard Fully Connected",
            legend = :topleft,
            marker = :circle,
            xlabel = "System Size (N)",
            ylabel = "Avg. Max Bond Dim",
            title = "σ = $σ"
        )

        plot!(
            p,
            N_range,
            avg_out;
            yerr = err_out,
            label = "Outlier Graph",
            marker = :square
        )

        push!(plot_list, p)
    end

    main_title = "Standard vs. Outlier Graph Entanglement (J=$J_coupling, Δ=$Delta_coupling)"
    
    final_plot = plot(
        plot_list...;
        layout = plot_layout,
        plot_title = main_title,
        size = (500 * num_plots, 500) # (1500, 500) for 3 plots
    )

    output_filename = "outlier_graphs_plot.png"
    savefig(final_plot, output_filename)

    println("Plot successfully saved to $output_filename")
end

create_plot()