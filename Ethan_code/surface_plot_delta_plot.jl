using JLD2, Plots, Colors

filename = joinpath(@__DIR__, "surface_plot_delta_data.jld2")

function load_data_and_plot(data_file)
    println("Loading data from $data_file...")
    if !isfile(data_file)
        println("Error: Data file '$data_file' not found.")
        println("Please run 'surface_plot_delta.jl' first to generate the data.")
        return 
    end
    
    data = load(data_file)
    max_bond_dims = data["max_bond_dims"]
    N_range = data["N_range"]
    delta_range = data["delta_range"]

    N_values = collect(N_range)
    delta_values = collect(delta_range) 

    plotlyjs() 
    
    plt = plot(N_values, delta_values, max_bond_dims',
        st=:surface,
        title="Maximum Bond Dimension χ vs. (N, Δ)",
        xlabel="System Size N",
        ylabel="Anisotropy Δ",  
        zlabel="Max Bond Dimension χ",
        camera=(-25, 25),      
        c=cgrad(:inferno),    
        legend=false,
        automargin = true,
        autosize = true,
    )
    
    output_filename = joinpath(@__DIR__, "surface_plot_delta_plot.png")
    # output_filename = joinpath(@__DIR__, "surface_plot_delta_plot.html")
    savefig(plt, output_filename)
    println("Plot saved successfully to $output_filename")
end

println("--- Loading and Plotting from File ---")
load_data_and_plot(filename)
println("Plotting script finished.\n")

