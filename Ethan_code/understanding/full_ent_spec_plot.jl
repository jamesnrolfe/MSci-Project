using JLD2, Plots

"""
Loads the saved entanglement spectrum data and generates the 8-panel plot.
"""
function load_and_plot_spectrum()
    
    filename = joinpath(@__DIR__, "full_ent_spec_data.jld2")
    println("Loading data from $filename...")

    if !isfile(filename)
        println("Error: Data file '$filename' not found.")
        println("Please run 'entanglement_spectrum.jl' first to generate the data.")
        return
    end

    # Load all data from the JLD2 file
    loaded_data = load(filename)
    entanglement_spectrum_results = loaded_data["entanglement_spectrum_results"]
    max_coeffs_to_store = loaded_data["max_coeffs_to_store"]
    
    N_values_sorted = sort(collect(keys(entanglement_spectrum_results)))

    println("Simulations data loaded. Generating plot...")

    plot_list = []
    
    # Set y-axis limits
    y_max = 0.35 
    
    for N in N_values_sorted
        coeffs_to_plot = entanglement_spectrum_results[N]
        
        # Create the bar plot for this panel
        p = bar(coeffs_to_plot,
                title="$N nodes",
                legend=false,
                ylims=(0, y_max),
                xlims=(0, max_coeffs_to_store + 1))
        
        # Add labels only to the outer plots
        if N in [20, 60]
            ylabel!(p, "Coefficient Values")
        end
        if N in [60, 70, 80, 90]
            xlabel!(p, "Schmidt Coefficients")
        end

        push!(plot_list, p)
    end

    # Combine all 8 plots into a 2x4 grid
    main_plot = plot(plot_list..., 
                     layout=(2, 4), 
                     plot_title="Full Entanglement Spectrum for σ=0.002",
                     size=(1600, 800))
    
    output_filename = "entanglement_spectrum_plot.png"
    savefig(main_plot, output_filename)
    println("Plot saved to $output_filename")
end

load_and_plot_spectrum()
println("\nPlotting script finished.")