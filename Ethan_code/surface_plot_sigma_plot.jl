using JLD2, Plots, Colors

filename = joinpath(@__DIR__, "surface_plot_sigma_data.jld2")

function load_data_and_plot(data_file)
    println("Loading data from $data_file...")
    if !isfile(data_file)
        println("Error: Data file '$data_file' not found.")
        println("Please run 'surface_plot_sigma.jl' first to generate the data.")
        return 
    end
    
    data = load(data_file)
    avg_bond_dims = data["avg_bond_dims"]
    N_range = data["N_range"]
    sigma_range = data["sigma_range"]

    N_values = collect(N_range)
    sigma_values = collect(sigma_range)

    plotlyjs() 
    
    plt = plot(N_values, sigma_values, avg_bond_dims',
        st=:surface,
        title="Average Bond Dimension vs. (N, σ)",
        xlabel="System Size (N)",
        ylabel="σ",
        zlabel="Avg. Max Bond Dimension",
        camera=(50, 30),
        c=cgrad(:inferno),
        legend=false
    )
    
    output_filename = joinpath(@__DIR__, "surface_plot_sigma_plot.png")
    savefig(plt, output_filename)
    println("Plot saved successfully to $output_filename")
end

println("--- Loading and Plotting from File ---")
load_data_and_plot(filename)
println("Plotting script finished.\n")