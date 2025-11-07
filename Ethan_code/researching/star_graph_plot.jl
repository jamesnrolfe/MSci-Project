using Plots, Graphs, GraphRecipes
using Random

# Set a seed for the random graph generation to be reproducible
Random.seed!(42);

"""
This is a helper function to generate a sample adjacency matrix.
It's needed to generate a sample adjacency matrix for plotting.
"""
function create_weighted_star_adj_mat(N::Int, σ::Float64; μ::Float64=1.0)
    A = zeros(Float64, N, N)
    if N < 2
        return A
    end
    # Connect center (1) to all outer spins (2...N)
    for j in 2:N
        # If sigma is 0.0, use 'μ'. Otherwise, add noise.
        weight = (σ == 0.0) ? μ : (μ + σ * randn())
        A[1, j] = A[j, 1] = weight
    end
    return A
end

"""
Generates and plots a single realization of a weighted star graph.
"""
function plot_weighted_graph(N::Int, σ::Float64; μ::Float64=1.0)
    println("Generating sample graph for N=$N, σ=$σ...")
    
    # 1. Create the adjacency matrix and graph object
    adj_mat = create_weighted_star_adj_mat(N, σ; μ=μ)
    g = SimpleGraph(adj_mat) # Graphs.jl handles this from the matrix

    # 2. Set up manual layout for the star graph
    # Center node (1) is at (0,0)
    loc_x = zeros(Float64, N)
    loc_y = zeros(Float64, N)
    
    # Outer nodes (2...N) are in a circle
    radius = 1.0
    angles = range(0, 2π, length=N) # N angles, but we only use N-1
    for i in 2:N
        loc_x[i] = radius * cos(angles[i-1])
        loc_y[i] = radius * sin(angles[i-1])
    end

    # 3. Get edge widths from weights
    edge_weights = []
    for edge in edges(g)
        # Get the weight from our adjacency matrix
        push!(edge_weights, adj_mat[src(edge), dst(edge)])
    end
    
    # Normalize weights for plotting
    # Map weights to a line width, e.g., 0.5 to 8
    max_w = maximum(abs.(edge_weights))
    if max_w == 0.0 max_w = 1.0 end # Avoid division by zero
    
    # Ensure line widths are positive and scaled
    line_widths = [max(0.1, 6.0 * abs(w) / max_w) for w in edge_weights]

    # 4. Create the plot
    p2 = plot(
        g,
        x = loc_x,
        y = loc_y,
        names = 1:N,
        nodecolor = 1, # Color node 1 differently
        markersize = 15,
        markerstrokewidth = 0,
        edgewidth = line_widths, # Use our calculated widths
        linecolor = :darkgrey,
        axis_buffer = 0.1,
        framestyle = :none, # Clean up the plot
        legend = false,
        title = "Weighted Star Graph (N=$N, σ=$σ)"
    )
    
    # Save the plot
    output_filename = "star_graph_N$(N)_sigma$(σ).png"
    savefig(p2, output_filename)
    println("Saved sample graph plot to $output_filename")
    display(p2) # Show the plot
end


# --- Main execution ---

# Plot a sample weighted graph
#    (e.g., N=10 and sigma=0.5 to show inhomogeneity)
plot_weighted_graph(10, 0.5; μ=1.0)

println("Graph plotting script finished.")