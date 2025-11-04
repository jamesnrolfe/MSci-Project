using JLD2, Plots

function create_comparison_plot()
    println("Loading data...")
    
    data_filename = joinpath(@__DIR__, "lin_con_comparison_data.jld2")
    plot_filename = joinpath(@__DIR__, "lin_con_comparison_plot.png")

    if !isfile(data_filename)
        println("ERROR: Data file not found: $data_filename")
        println("Please run 'lin_con_comparison.jl' first to generate the data.")
        return
    end

    local results_connected, results_linear, N_range, sigma_values
    try
        jldopen(data_filename, "r") do file
            results_connected = read(file, "results_connected")
            results_linear = read(file, "results_linear")
            N_range = read(file, "N_range")
            sigma_values = read(file, "sigma_values")
        end
    catch e
        println("ERROR: Could not load data from $data_filename. Error: $e")
        return
    end

    println("Data loaded. Generating plot...")

    gr()
    
    p = plot(
        title = "Bond Dimension Comparison: Connected vs. Linear Models",
        xlabel = "System Size (N)",
        ylabel = "Average Max Bond Dimension",
        legend = :topleft,
        size = (1000, 600)
    )

    colors = [:blue, :purple, :red]
    
    for (idx, σ) in enumerate(sigma_values)
        color = colors[idx]
        
        plot!(p,
            N_range,
            results_connected[σ].avg,
            ribbon = results_connected[σ].err,
            label = "Connected (σ=$σ)",
            color = color,
            linestyle = :solid,
            fillalpha = 0.15
        )
        
        plot!(p,
            N_range,
            results_linear[σ].avg,
            ribbon = results_linear[σ].err,
            label = "Linear (σ=$σ)",
            color = color,
            linestyle = :dash,
            fillalpha = 0.15
        )
    end

    try
        savefig(p, plot_filename)
        println("--- ✅ Plot saved successfully to $plot_filename ---")
    catch e
        println("ERROR: Could not save plot. Error: $e")
    end
end

create_comparison_plot()