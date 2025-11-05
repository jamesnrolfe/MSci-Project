#Erdos-Renyi Graph Construction-----------------
using LightGraphs, Random
using Graphs, GraphPlot
using ITensors, LinearAlgebra

function random_weighting(Nodes, adj_mat, μ, σ)
    adj_mat = convert(Matrix{Float64}, adj_mat)
    rng = MersenneTwister()
    for i in 1:Nodes
        for j in i+1:Nodes
            if adj_mat[i,j] == 0
                break
            else
                rand = μ + σ*randn(rng, Float64)
                while rand <= 0
                    rand = μ + σ*randn(rng, Float64)
                end
                adj_mat[i,j] = rand
                adj_mat[j,i] = rand
            end
        end
    end
    return adj_mat
end;

#Due to Joseph Tindall
#Function to create the MPO corresponding to the XXZ model on a graph specified by the L x L adj_mat A.
#H = \sum_{i > j}A_{ij}(-\sigma^{x}_{i}\sigma^{x}_{j} - \sigma^{y}_{i}\sigma^{y}_{j} + \Delta \sigma^{z}_{i}\sigma^{z}_{j})

function create_XXZ_Ham_MPO(Nodes, adj_mat, Δ, sites)
    ampo = OpSum()
    for i=1:Nodes
        for j = i+1:Nodes
            ampo += 4*adj_mat[i,j]*Δ, "Sz",i,"Sz",j
            ampo += -2*adj_mat[i,j],"S+",i,"S-",j
            ampo += -2*adj_mat[i,j],"S-",i,"S+",j
        end
    end
    H = MPO(ampo, sites)
    return H
end

⊗(x,y) = kron(x,y) 

#Creates a 2ᴺ×2ᴺ matrix representation for the XXZ Hamiltonian on the graph specified by adj_mat
function create_XXZ_Ham_matrix(Nodes, adj_mat, J, Δ)
    id = complex([1 0;0 1])
    σˣ = complex([0 1;1 0])
    σʸ = complex([0 -im; im 0])
    σᶻ = complex([1 0;0 -1])
    
    act_pauli_x = fill(id, Nodes)
    act_pauli_y = fill(id, Nodes)
    act_pauli_z = fill(id, Nodes)
    
    H = complex(zeros(Float64, 2^Nodes, 2^Nodes))
    
    for i in 1:Nodes
        for j in i+1:Nodes
            setindex!(act_pauli_x, [σˣ, σˣ], [i, j])
            setindex!(act_pauli_y, [σʸ, σʸ], [i, j])
            setindex!(act_pauli_z, [σᶻ, σᶻ], [i, j])
            
            H -= 2*J*adj_mat[i,j]*foldl(⊗, act_pauli_x)
            H -= 2*J*adj_mat[i,j]*foldl(⊗, act_pauli_y)
            H += 2*Δ*adj_mat[i,j]*foldl(⊗, act_pauli_z)
            
            setindex!(act_pauli_x, [id, id], [i, j])
            setindex!(act_pauli_y, [id, id], [i, j])
            setindex!(act_pauli_z, [id, id], [i, j])
        end
    end
    return H, adj_mat
end

function Find_Groundstate(Hamiltonian)
    vals, vecs = eigen(Hamiltonian)
    #println(vals)
    #pretty_table(vecs)
    groundstate = vecs[:,1]
    return groundstate
end

function main(Nodes, Probability, μ, σ, χ, J, Δ)
    G = LightGraphs.erdos_renyi(Nodes, Probability)
    adj_mat = LightGraphs.adjacency_matrix(G) # A is a sparse matrix
    adj_mat = random_weighting(Nodes, adj_mat, μ, σ) # randomise the weightings on the graph
    G = Graphs.SimpleGraph(adj_mat)
    #GraphPlot.gplot(G)


    #Initialise site inds and initial state as Neel state. Conserve total Sz.
    sites = siteinds("S=1/2", Nodes; conserve_qns = true)
    init_state = [isodd(i) ? "Up" : "Dn" for i = 1:Nodes]

    #Set DMRG sweep parameters
    sweeps = Sweeps(15)
    setmaxdim!(sweeps,χ)
    setcutoff!(sweeps, 1E-10)

    #Create Hamiltonian MPO and run DMRG
    H = create_XXZ_Ham_MPO(Nodes, adj_mat, Δ, sites)
    psi0 = randomMPS(sites, init_state)
    energy, DMRG_psi = dmrg(H, psi0, sweeps)
    println("DRMG Finished Energy Per Site is ", energy/Nodes)

    DMRG_psi_tensor = ITensors.contract(DMRG_psi)
    DMRG_psi_array = array(DMRG_psi_tensor)
    DMRG_psi_array = reshape(DMRG_psi_array, (2^Nodes))
    DMRG_psi_array = abs.(DMRG_psi_array)
    #DMRG_psi_array = abs.(filter(x -> x<-1e-10 || x>1e-10, DMRG_psi_array))
    #print(DMRG_psi_array)
    #print("\n")
    #Main---------------------------

    Hamiltonian, adj_mat = create_XXZ_Ham_matrix(Nodes, adj_mat, J, Δ)
    exact_psi = Find_Groundstate(Hamiltonian)
    exact_psi = real.(exact_psi)
    #exact_psi = filter(x -> x<-1e-10 || x>1e-10, exact_psi)
    exact_psi = abs.(exact_psi)
    #print(exact_psi)
    #print("\n")

    #Difference Comparison-------------

    difference_array = exact_psi - DMRG_psi_array
    #print(difference_array)

    global magnitude = 0
    for L in 1:Nodes
        global magnitude += difference_array[L]^2
    end
    error = 1 - sqrt(magnitude)
    #print("\n")
    #print(error)
    return error
end

#Main---------------------------
χ = 1000 #DMRG maximum bond dimension
J = 1 #Hamiltonian x,y component coefficient
Δ = 1 #Hamiltonian z component coefficient
Min_Nodes = 3
Max_Nodes = 13
Probability = 1.0
μ = 1
σ = 0.002

error_at_each_node_data = zeros(length(Min_Nodes:Max_Nodes))
iteration = 0

for current_nodes in Min_Nodes:Max_Nodes
    global iteration += 1 
    datapoints = 10
    average_error = 0
    for i in 1:datapoints
        output_error = main(current_nodes, Probability, μ, σ, χ, J, Δ)
        average_error += output_error
    end
    average_error = average_error/datapoints
    error_at_each_node_data[iteration] = average_error
end

x_axis = range(Min_Nodes, Max_Nodes)
Plots.plot(x_axis, error_at_each_node_data, label=false)
scatter = Plots.scatter!(x_axis, error_at_each_node_data, label=false)
title!("The total error between DMRG and the exact wavefunction")
xlabel!("Number of Nodes")
ylabel!("Error")
Plots.savefig(scatter,"figures\\DMRG_error_Max_Nodes_$(Max_Nodes)")
