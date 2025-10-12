function generate_random_adjacency_matrix(L::Int, p::Float64)
    """
    Generate the adjacency matrix for a 1D chain of length L, where each node is connected to every other with probability p.
    """
    G = LightGraphs.erdos_renyi(L, p) # start with no edges
    return LightGraphs.adjacency_matrix(G)
end

function plot_adjacency_matrix(adj_mat; layout_type::String = "circular")
    """
    Plot the graph represented by the adjacency matrix.

    layout_type can be "circular", "spring", "spectral", or "random".
    """

    layout_lookup = Dict("circular" => circular_layout,
                         "spring" => spring_layout,
                         "spectral" => spectral_layout,
                         "random" => random_layout)

    G = Graphs.SimpleGraph(adj_mat)
    N = size(adj_mat, 1)

    node_labels = [string(i) for i in 1:N]

    p = GraphPlot.gplot(
        G, 
        layout=layout_lookup[layout_type],
        nodelabel=node_labels,
        nodesize=0.3,
        nodelabelsize=3,
        edgelinewidth=2.0,
    ) # plot the graph with the specified layout
    display(p)
    return p
end

function generate_chain_adjacency_matrix(N::Int)
    """
    Generate adjacency matrix for a 1D chain with N nodes and nearest-neighbor connections only.
    Returns a symmetric N×N matrix where adj_mat[i,j] = 1 if nodes i and j are connected.
    """
    adj_mat = zeros(Int, N, N)
    
    # connect each node to its nearest neighbors
    for i in 1:N-1
        adj_mat[i, i+1] = 1  # Connect i to i+1
        adj_mat[i+1, i] = 1  # Connect i+1 to i (make symmetric)
    end
    
    return adj_mat
end