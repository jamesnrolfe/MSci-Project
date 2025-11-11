using Markdown
using InteractiveUtils
using Graphs, Random, GraphPlot, Plots, Colors, GraphRecipes
using ITensors, ITensorMPS, LinearAlgebra
using Plots, JLD2

"""
Creates a weighted adjacency matrix for a star graph with N nodes Node 1 is the center.
"""
function weighted_star_adjmat(N::Int, σ::Float64; μ::Float64=1.0)
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
Places node 1 at the center (0,0) and all other nodes in a circle around it.
"""
function star_layout(g::AbstractGraph)
    N = nv(g)
    locs_x = zeros(Float64, N)
    locs_y = zeros(Float64, N)
    
    locs_x[1] = 0.0
    locs_y[1] = 0.0
    
    num_outer = N - 1
    angles = range(0, 2π, length=num_outer + 1)[1:num_outer] 
    
    for (i, node_idx) in enumerate(2:N)
        locs_x[node_idx] = cos(angles[i])
        locs_y[node_idx] = sin(angles[i])
    end
    
    return (locs_x, locs_y)
end



N_nodes = 8
sigma = 0.02
mu = 1

adjmat = weighted_star_adjmat(N_nodes, sigma)
G = Graphs.SimpleGraph(adjmat)

# Create an N x N matrix initialized with a transparent color
edge_color_matrix = fill(colorant"transparent", N_nodes, N_nodes)

for edge in edges(G)
    i, j = src(edge), dst(edge)
    weight = adjmat[i, j]
    
    local color # Use local variable
    if weight > mu
        color = colorant"salmon" # > 1.0
    elseif weight < mu
        color = colorant"violet"  # < 1.0
    else
        color = colorant"grey" # 1.0
    end
    
    # Assign the color to the matrix for both directions
    edge_color_matrix[i, j] = color
    edge_color_matrix[j, i] = color
end

layout = star_layout(G)

node_labels = [string(i) for i in 1:N_nodes]

p = plot(
    G, 
    layout=star_layout,         
    nodelabel=node_labels,
    nodesize=0.3,
    fontsize=3,
    linewidth=2.0,
    edgecolor = edge_color_matrix  # Pass the new matrix here
) 

output_filename = joinpath(@__DIR__, "star_graph_plot.png")
savefig(p, output_filename)
println("Saved graph plot to $output_filename")