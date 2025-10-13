using DataFrames
using ITensors
using LightGraphs
using LinearAlgebra
using Random
using Tables
using LaTeXStrings
using Plots; gr()
#------------------------------------------------------------------FUNCTIONS------------------------------------------------------------------------------

#Erdos-Renyi Graph Construction-----------------
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

#Functions for DMRG Solution----------------------------------------------------------------------------------------
#Due to Joseph Tindall
#Function to create the MPO corresponding to the XXZ model on a graph specified by the L x L adj_mat A.
#H = \sum_{i > j}A_{ij}(-\sigma^{x}_{i}\sigma^{x}_{j} - \sigma^{y}_{i}\sigma^{y}_{j} + \Delta \sigma^{z}_{i}\sigma^{z}_{j})

function create_XXZ_Ham_MPO(Nodes, adj_mat, Delta, sites)
    ampo = OpSum()
    for i=1:Nodes
        for j = i+1:Nodes
            ampo += 4*adj_mat[i,j]*Delta, "Sz",i,"Sz",j
            ampo += -2*adj_mat[i,j],"S+",i,"S-",j
            ampo += -2*adj_mat[i,j],"S-",i,"S+",j
        end
    end
    H = MPO(ampo, sites)
    return H
end;

#functions to define an Observer to save the current bond dimension as DMRG runs---------------------------------------
mutable struct BondDimObserver <: AbstractObserver
   max_dims::Vector{Int64}

   BondDimObserver() = new(Vector{Int64}())
end;

function ITensors.measure!(o::BondDimObserver; kwargs...)
    psi = kwargs[:psi]
    sweep_is_done = kwargs[:sweep_is_done]

    if sweep_is_done
        max_dim = maxlinkdim(psi)
        push!(o.max_dims, max_dim)
    else
        return nothing
    end
end;

function von_neumann_entropy_on_MPS(psi::MPS, b::Int)
    s = siteinds(psi)
    #print(s)
    orthogonalize!(psi, b)
    if b==1
        _,S = svd(psi[b], s[b])
    else
        _,S = svd(psi[b], (linkind(psi, b-1), s[b]))
    end
    SvN = 0.0
    for n in 1:dim(S, 1)
        p = S[n,n]^2
        #print(p)
        if p == 0
            SvN -= 0
        elseif p == 1
            SvN -= 0
        else
            SvN -= p*log2(p)
        end
    end
    return SvN
end;

function extract_entanglement_spectrum(psi::MPS, b::Int)
    s = siteinds(psi)
    #print(s)
    orthogonalize!(psi, b)
    if b==1
        _,S = svd(psi[b], s[b])
    else
        _,S = svd(psi[b], (linkind(psi, b-1), s[b]))
    end
    #print(S)
    schmidt_rank = Int64(length(1:dim(S, 1)))
    entanglement_spectrum = zeros(schmidt_rank)
    for n in 1:dim(S, 1)
        entanglement_spectrum[n] = S[n,n]^2
    end
    return entanglement_spectrum, schmidt_rank
end;

#---------------------------------------------------------------MAIN-----------------------------------------------------------------------------------
function main(Nodes_array, Probability, μ, σ, Num_Sweeps, χ, J, Δ)
    iterations = length(Nodes_array)
    global entanglement_spectrum_data = zeros(200, iterations)
    global schmidt_rank_data = zeros(Int64, iterations)

    iteration = 0
    for nodes in Nodes_array
        mid_node = nodes ÷ 2
        iteration += 1
        #Generate a random Graph and it's adjacency matrix, adj_mat
        G = LightGraphs.erdos_renyi(nodes, Probability)
        adj_mat = LightGraphs.adjacency_matrix(G) # A is a sparse matrix
        adj_mat = random_weighting(nodes, adj_mat, μ, σ) # randomise the weightings on the graph
        
        #Initialise site inds and initial state as Neel state. Conserve total \S_{z}.
        sites = siteinds("S=1/2", nodes; conserve_qns = true)
        init_state = [isodd(i) ? "Up" : "Dn" for i = 1:nodes]
        psi0 = randomMPS(sites, init_state)
    
        #Create the Hamiltonian for the DMRG solution
        H = create_XXZ_Ham_MPO(nodes, adj_mat, Δ, sites)
    
        #Set DMRG sweep parameters
        sweeps = Sweeps(Num_Sweeps)
        setmaxdim!(sweeps,χ)
        setcutoff!(sweeps, 1E-10)
               
        #Run DMRG
        energy, DMRG_psi = dmrg(H, psi0, sweeps; outputlevel=0)
        println("DRMG Finished over ", nodes ," Nodes and the Energy Per Site is ", energy/nodes)
        
        entanglement_spectrum, schmidt_rank = extract_entanglement_spectrum(DMRG_psi, mid_node)

        for i in 1:schmidt_rank
            entanglement_spectrum_data[i, iteration] = entanglement_spectrum[i]
        end
        schmidt_rank_data[iteration] = schmidt_rank
    end
    return entanglement_spectrum_data, schmidt_rank_data
end;

#Graph variables
Nodes = [90] #minimum number of spin-1/2 particles
nodes = Nodes[1]
Probability = 1.0 #Probability of a given pair of particles interacting
μ = 1 #mean node weighting
σ = 0.002 #standard deviation of node weighting

#Hamiltonian Variables
J = 1 #Hamiltonian x,y component coefficient
Δ = 1 #Hamiltonian z component coefficient

#DMRG variables
Num_Sweeps = 30 #set maximum number of DMRG sweeps
χ = 1000 #Set maximum bond dimension for DMRG

data_points = 10
iterations_outer = length(Nodes)
average_entanglement_array = zeros(200, iterations_outer)
#average_bond_dataframe = DataFrame(Tables.table(average_bond_dataframe))

#s_1 = zeros(iterations_outer)
#s_2 = zeros(iterations_outer)

for i in 1:data_points
    entanglement_spectrum_data, schmidt_rank_data = main(Nodes, Probability, μ, σ, Num_Sweeps, χ, J, Δ)

    #s_1 .+= Vector(DataFrameRow(output_dataframe, Num_Sweeps))
    #s_2 .+= Vector(DataFrameRow(output_dataframe, Num_Sweeps)).^2
    iterations_dummy=0

    for schmidt_rank in schmidt_rank_data

        iterations_dummy += 1
        entanglement_spectrum_data[:, iterations_dummy] = sort(entanglement_spectrum_data[:, iterations_dummy], rev=true)

        for k in 1:schmidt_rank
            average_entanglement_array[k, iterations_dummy] += entanglement_spectrum_data[k, iterations_dummy]
        end
    end

    #println("\n")
end;

average_entanglement_array = average_entanglement_array ./ data_points
average_entanglement_array = filter(!iszero, average_entanglement_array)
#print(average_entanglement_array)

#\sigma = \frac{\sqrt{Ns_2 - s_1^{2}}}{N} used for standard deviation

#std_dataframe = (data_points.*s_2) .- (s_1.^2)
#print(std_dataframe)
#std_dataframe = std_dataframe.^(0.5)
#std_dataframe = std_dataframe./(data_points*(data_points-1))
schmidt_rank= schmidt_rank_data[1]
bins = 1:30
bar(bins, average_entanglement_array[1:30], ylim=(0,0.35), color=:darkorange3)

title!("$(nodes) nodes")
xlabel!("Schmidt Coefficients")
ylabel!("Coefficient Values")

Plots.savefig("figures\\Entanglement_Spectrum_for_N_$(nodes)_and_σ_$(σ).png")
print("Simulation Complete")

#--------------------------------------------------------Plotting-------------------------------------------------

#function bar_graph(schmidt_rank, spectrum_data, nodes)
#    bins = 1:schmidt_rank
    #histogram(spectrum_data, bins=bins, title="$(nodes) nodes", xlabel="Schmidt Coefficient", ylabel="Coefficient Value", color="red")
#end

#sample_size = 5
#average_bond_data = data_spread(sample_size)
#print(average_bond_data)
#print("\n")
#bins = range(10, 30, length=21)

# bins = 1:schmidt_rank

# histogram(average_bond_data, bins=bins, normalize=:pdf, title="one graph", xlabel="Bond Dimension", ylabel="Fraction of Samples")

# title!("Average over $(sample_size) Graph")
# xlabel!("Average Bond Dimension")
# ylabel!("Fraction of Datapoints")
# Plots.savefig("figures\\number_of_graphs_required_for_convergence_N_$(Nodes)_σ_$(σ)_sample_$(sample_size).png")
# print("Simulation Complete")

# final_bond_dimension_vs_num_Nodes = Vector(DataFrameRow(average_bond_dataframe, Num_Sweeps))
# bond_dimension_growth = Vector(average_bond_dataframe[:, iterations_outer])

# y_axis = range(Min_Nodes, Max_Nodes)
# Plots.plot(y_axis, final_bond_dimension_vs_num_Nodes, label=false)
# scatter = Plots.scatter!(y_axis, final_bond_dimension_vs_num_Nodes, label=false)
# title!("saturated bond dimension for \n an average graph with N Nodes")
# xlabel!("Number of Nodes")
# ylabel!("Average Bond Dimension Required")
# Plots.savefig(scatter,"figures\\errorbar_test")

#Plots.plot(bond_dimension_growth, label=false)
#scatter = Plots.scatter!(bond_dimension_growth, label=false)
#title!("Average Bond dimension after each sweep of DMRG")
#xlabel!("Sweeps")
#ylabel!("Bond Dimension")
#Plots.savefig(scatter,"figures\\bond_dimension_at_each_sweep_100_Nodes_sigma_0_complete.png")

#xkcd()
# final_bond_dimension_vs_num_Nodes = Vector(DataFrameRow(average_bond_dataframe, Num_Sweeps))
# bond_growth_demo = PyPlot.plot(final_bond_dimension_vs_num_Nodes);
# PyPlot.title("saturated bond dimension for an average graph with N Nodes");
# PyPlot.xlabel("Number of Nodes");
# PyPlot.ylabel("Bond Dimension Required");
# PyPlot.savefig("C:\\Users\\samba\\OneDrive\\Documents\\Code\\Tensor_Networks\\connected_spin_systems\\figures\\bond_growth_demo.png");

# bond_dimension_growth = Vector(average_bond_dataframe[:, iterations_outer])
# bond_growth_over_sweeps = PyPlot.plot(bond_dimension_growth);
# PyPlot.title("bond dimension after each sweep of DMRG");
# PyPlot.xlabel("Sweeps");
# PyPlot.ylabel("Bond Dimension");
# PyPlot.savefig("C:\\Users\\samba\\OneDrive\\Documents\\Code\\Tensor_Networks\\connected_spin_systems\\figures\\bond_dimension_at_each_sweep_demo.png");