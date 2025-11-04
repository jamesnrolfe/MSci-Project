using JLD2, Plots, Colors

function load_and_generate_plots()
    plotlyjs() 
    
    filename = joinpath(@__DIR__, "j_delta_sigma_n_data.jld2")
    println("Loading data from $filename...")

    if !isfile(filename)
        println("Error: Data file '$filename' not found.")
        println("Please run 'j_delta_sigma_n.jl' first to generate the data.")
        return
    end
    
    loaded_data = load(filename)
    results_data = loaded_data["data"]
    J_vals = loaded_data["J_vals"]
    Δ_vals = loaded_data["Δ_vals"]
    N_vals = loaded_data["N_vals"]
    σ_vals = loaded_data["σ_vals"]

    N_values_sorted = sort(collect(N_vals))
    sigmas_sorted = sort(collect(σ_vals))

    colors = Dict(
        0.0 => :purple, 
        0.001 => :darkorange, 
        0.002 => :teal
    )

    println("Generating plots for each (J, Δ) combination...")

    for J in J_vals
        for Δ in Δ_vals
            
            plt_title = "Avg. Bond Dim (J=$J, Δ=$Δ)"
            plt = plot(title=plt_title,
                       xlabel="Number of Nodes (N)",
                       ylabel="Avg. Bond Dimension",
                       legend=:topleft,
                       grid=true,
                       yscale=:log) 

            for σ in sigmas_sorted
                avg_dims = Float64[]
                errors = Float64[]
                
                for N in N_values_sorted
                    key = (J, Δ, N, σ)
                    if haskey(results_data, key)
                        avg_dim, err = results_data[key]
                        push!(avg_dims, avg_dim)
                        push!(errors, err) 
                    else
                        push!(avg_dims, NaN) 
                        push!(errors, NaN)
                        println("Warning: Missing data for key $key")
                    end
                end
                
                color = get(colors, σ, :auto) 
                label_str = "σ=$σ"
                
                plot!(plt, N_values_sorted, avg_dims, 
                      label=label_str, 
                      color=color,
                      ribbon=errors,     
                      fillalpha=0.15,    
                      linewidth=1.5,
                      marker=:circle,
                      markersize=3)
            end
            
            J_str = replace("$J", "-" => "neg")
            Δ_str = replace("$Δ", "-" => "neg")
            output_filename = "j_delta_sigma_n_$(J_str)_D_$(Δ_str).png"
            output_path = joinpath(@__DIR__, output_filename)
            
            try
                savefig(plt, output_path)
                println("Saved plot: $output_path")
            catch e
                println("ERROR: Could not save plot $output_filename. Error: $e")
            end
        end
    end
end

load_and_generate_plots()
println("\nPlotting script finished.")