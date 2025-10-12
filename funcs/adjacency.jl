function generate_random_adjacency_matrix(L::Int, p::Float64)
    """
    Generate the adjacency matrix for a 1D chain of length L, where each node is connected to every other with probability p.
    """
    G = LightGraphs.erdos_renyi(L, p) # start with no edges
    return LightGraphs.adjacency_matrix(G)
end