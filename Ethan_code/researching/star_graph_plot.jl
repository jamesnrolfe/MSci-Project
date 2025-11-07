using Graphs
using GraphPlot         # For gplot
using Random
using Colors            # For colorant"green", "red", "blue"
using Compose           # For saving the plot
using Cairo             # For the PNG backend

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
    for j in 2:N
        weight = (σ == 0.0) ? μ : (μ + σ * randn())
        A[1, j] = A[j, 1] = weight
    end
    return A
end

"""
Generates and plots a single realization of a weighted star graph
using the GraphPlot.gplot method.
"""
function plot_weighted_graph(N::Int, σ::Float64; μ::Float64=1.0)
    println("Generating sample graph for N=$N, σ=$σ...")
    
    # 1. Create the graph from the adjacency matrix
    adj_mat = create_weighted_star_adj_mat(N, σ; μ=μ)
    g = SimpleGraph(adj_mat) 

    # 2. Define the star layout (Node 1 in center)
    loc_x = zeros(Float64, N)
    loc_y = zeros(Float64, N)
    
    radius = 1.0
    angles = range(0, 2π, length=N)
    for i in 2:N
        loc_x[i] = radius * cos(angles[i-1])
        loc_y[i] = radius * sin(angles[i-1])
    end

    # 3. Define edge widths and colors based on weight
    # gplot requires these to be vectors, one entry per edge
    max_w = maximum(abs.(adj_mat))
    if max_w == 0.0 max_w = 1.0 end 

    edge_widths = Float64[]
    edge_colors = Colorant[] # gplot uses Colorant types

    for edge in edges(g)
        i, j = src(edge), dst(edge)
        weight = adj_mat[i, j]
        
        # Width logic (thicker for stronger weight)
        width = max(0.5, 6.0 * abs(weight) / max_w)
        push!(edge_widths, width)
        
        # Color logic (blue for positive, red for negative)
        color = weight >= 0 ? colorant"blue" : colorant"red"
        push!(edge_colors, color)
    end

    # 4. Define node attributes
    node_labels = [string(i) for i in 1:N]
    node_fill_colors = colorant"green" # gplot attribute for node color
    
    # Use noderad for a fixed node size (in layout units)
    # The `nodesize` from the notebook is relative and can be inconsistent
    node_radii = fill(0.05, N) 

    # 5. Create the plot using GraphPlot.gplot
    p = GraphPlot.gplot(
        g,
        loc_x,
        loc_y,
        nodelabel = node_labels,
        nodelabelsize = 3.0,
        noderad = node_radii,
        nodefillc = node_fill_colors,
        edgelinewidth = edge_widths,
        edgestrokec = edge_colors
    )
    
    # 6. Save the plot using Compose.jl
    output_filename = joinpath(@__DIR__, "star_graph_plot_N$(N)_σ$(σ).png")
    
    # Use draw(PNG(...), p) instead of savefig
    draw(PNG(output_filename, 16cm, 12cm), p)
    
    println("Saved sample graph plot to $output_filename")
end


# --- Run the function ---
plot_weighted_graph(10, 0.002; μ=1.0)

println("Graph plotting script finished.")