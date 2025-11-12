using Graphs
using Plots
using GraphRecipes
using LinearAlgebra
using NetworkLayout


Random.seed!(42) 

"""
Creates a weighted adjacency matrix for a completely connected graph.
(Copied from your script)
"""
function create_weighted_adj_mat(N::Int, σ::Float64; μ::Float64=1.0)
    if σ == 0.0
        A = ones(Float64, N, N)
        A -= Matrix{Float64}(I, N, N)
        return A
    end
    
    A = zeros(Float64, N, N)
    for i in 1:N, j in (i+1):N
        weight = μ + σ * randn()
        A[i, j] = A[j, i] = weight
    end
    return A
end

"""
Main function to generate and save the graph visualization.
"""
function visualize_clustering()
    println("--- Starting Part 4: Disordered Graph Visualization ---")

    # --- Parameters ---
    N = 8      
    μ = 1.0     
    σ = 0.75    

    adj_mat = create_weighted_adj_mat(N, σ; μ=μ)
    
    # Create a Graph object from the adjacency matrix
    g = Graphs.Graph(adj_mat) 

    println("Pre-computing spring layout...")
    layout_coords = spring(g) 
    
    # Extract x and y coordinates into separate vectors
    x_coords = [p[1] for p in layout_coords]
    y_coords = [p[2] for p in layout_coords]
    
    
    # Define a colormap
    cmap = cgrad(:bwr, [μ - 2σ, μ, μ + 2σ], categorical=false)
    
    # Create an attribute matrix for edge width
    edge_width_matrix = (abs.(adj_mat) ./ μ) .* 2

    println("Generating plot for N=$N, σ=$σ...")

    # Set backend
    gr() # default GR backend for Plots

    
    # Create the plot
    p = graphplot(
        g,
        # Pass coordinates directly to x and y keywords
        x = x_coords,
        y = y_coords,
        names = 1:N,
        fontsize = 10,
        nodesize = 0.5,
        nodeshape = :circle,
        nodecolor = :lightblue,
        
        # Edge properties
        edge_width = edge_width_matrix, 
        edge_z = adj_mat,           
        seriescolor = cmap,
        
        title = "Disordered Couplings (N = $N, σ = $σ, μ = $μ)\nYellow = Strong, Black = Weak",
        colorbar_title = "Coupling Strength (J_ij)",
        size = (800, 800),
        colorbar = true,
        dpi = 150
    )

    filename = joinpath(@__DIR__, "dis_graphs_plot_$(N)_$(σ).png")
    savefig(p, filename)
    
    println("Visualisation complete.")
    println("Plot saved to: $filename")


end

visualize_clustering()