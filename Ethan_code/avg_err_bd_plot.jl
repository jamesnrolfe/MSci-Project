using JLD2, Plots

filename = joinpath(@__DIR__, "avg_err_bd_data.jld2")

function load_data_and_plot(data_file)
    println("Loading data from $data_file...")
    if !isfile(data_file)
        println("Error: Data file '$data_file' not found.")
        println("Please run 'avg_err_bd.jl' first to generate the data.")
        return
    end
    

    data = load(data_file)
    results_data = data["results"]
    N_range_data = data["N_range"]

    N_values = collect(N_range_data)
    sigma_values = [0.0, 0.001, 0.002]
    colors = Dict(0.0 => :gold, 0.001 => :darkviolet, 0.002 => :firebrick)

    plotlyjs() 
    

    plt = plot(
        title="Average Bond Dimension vs. System Size",
        xlabel="System Size (N)",
        ylabel="Avg. Max Bond Dimension",
        legend=:bottomright
    )

    for σ in sigma_values

        plot!(plt, N_values[1:66], results_data[σ].avg[1:66],
            yerror=results_data[σ].err,  
            label="σ = $σ",
            lw=1.5,
            marker=:circle,
            markersize=3,
            color=colors[σ]
        )
    end
    
    output_filename = joinpath(@__DIR__, "avg_err_bd_plot.png")
    savefig(plt, output_filename)
    println("Plot saved successfully to $output_filename")
end

load_data_and_plot(filename)