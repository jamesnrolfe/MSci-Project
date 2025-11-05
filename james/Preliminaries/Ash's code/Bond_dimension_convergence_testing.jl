using DataFrames
using ITensors
using LightGraphs
using LinearAlgebra
using Random
using Tables
using LaTeXStrings
using Plots; gr()

#-----------------------------------------------------------FUNCTIONS------------------------------------------------------------------------------

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
    println(H)
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

#---------------------------------------------------------------MAIN-----------------------------------------------------------------------------------
function main(Nodes, Probability, μ, σ, Num_Sweeps, χ, J, Δ)
    #Generate a random Graph and it's adjacency matrix, adj_mat
    G = LightGraphs.erdos_renyi(Nodes, Probability)
    adj_mat = LightGraphs.adjacency_matrix(G) # A is a sparse matrix
    adj_mat = random_weighting(Nodes, adj_mat, μ, σ) # randomise the weightings on the graph
    
    #Initialise site inds and initial state as Neel state. Conserve total \S_{z}.
    sites = siteinds("S=1/2", Nodes; conserve_qns = true)
    init_state = [isodd(i) ? "Up" : "Dn" for i = 1:Nodes]
    psi0 = randomMPS(sites, init_state)
    
    #Create the Hamiltonian for the DMRG solution
    H = create_XXZ_Ham_MPO(Nodes, adj_mat, Δ, sites)
    
    #Set DMRG sweep parameters
    sweeps = Sweeps(Num_Sweeps)
    setmaxdim!(sweeps,χ)
    setcutoff!(sweeps, 1E-10)
    
    #call the observer function
    obs = BondDimObserver()
    
    #Run DMRG
    energy, DMRG_psi = dmrg(H, psi0, sweeps; observer=obs, outputlevel=0)
    #println("DRMG Finished over ", Nodes ," Nodes and the Energy Per Site is ", energy/Nodes)
    
    final_bond_dimension = obs.max_dims[Num_Sweeps]

    return final_bond_dimension
end;
#-------------------------------------------------------Variables----------------------------------------------
#Graph variables
Nodes = 80 #minimum number of spin-1/2 particles
Probability = 1.0 #Probability of a given pair of particles interacting
μ = 1 #mean node weighting
σ = 0.00 #standard deviation of node weighting

#Hamiltonian Variables
J = 1 #Hamiltonian x,y component coefficient
Δ = 1 #Hamiltonian z component coefficient

#DMRG variables
Num_Sweeps = 30 #set maximum number of DMRG sweeps
χ = 1000 #Set maximum bond dimension for DMRG

data_points = 100

function data_spread(sample_size)
    average_bond_data = Vector{Float64}()

    for i in 1:data_points   
        average_max_bond = 0

        for j in 1:sample_size
             output = main(Nodes, Probability, μ, σ, Num_Sweeps, χ, J, Δ)
             average_max_bond += output
        end
        average_max_bond = average_max_bond / sample_size
        print(average_max_bond)
        push!(average_bond_data, average_max_bond)
    end
    print("average max_bond found for $(data_points) datapoints, averaged over $(sample_size) graphs")
    print("\n")
    return average_bond_data
end

sample_size = 5
average_bond_data = data_spread(sample_size)
print(average_bond_data)
print("\n")
#bins = range(10, 30, length=21)

bins = range(1, 50, length=50)

histogram(average_bond_data, bins=bins, normalize=:pdf, title="one graph", xlabel="Bond Dimension", ylabel="Fraction of Samples")

title!("Average over $(sample_size) Graph")
xlabel!("Average Bond Dimension")
ylabel!("Fraction of Datapoints")
Plots.savefig("figures\\number_of_graphs_required_for_convergence_N_$(Nodes)_σ_$(σ)_sample_$(sample_size).png")
println("Simulation Complete")
#--------------------------------------------------------Plotting-------------------------------------------------

# final_bond_dimension_vs_num_Nodes = Vector(DataFrameRow(average_bond_dataframe, Num_Sweeps))
# bond_dimension_growth = Vector(average_bond_dataframe[:, iterations_outer])

# y_axis = range(min_data_points, max_data_points)
# Plots.plot(y_axis, average_bond_data, label=false)
# scatter = Plots.scatter!(y_axis, average_bond_data, label=false)
# title!("saturated bond dimension for \n an average graph with N Nodes")
# xlabel!("Number of Graphs Averaged per Datapoint")
# ylabel!("Average Bond Dimension Required")
# Plots.savefig(scatter,"figures\\number_of_graphs_required_for_convergence_N_$(Nodes)_σ_$(σ).png")

# Plots.plot(bond_dimension_growth, label=false)
# scatter = Plots.scatter!(bond_dimension_growth, label=false)
# title!("Average Bond dimension after each sweep of DMRG")
# xlabel!("Sweeps")
# ylabel!("Bond Dimension")
# Plots.savefig(scatter,"figures\\bond_dimension_at_each_sweep_100_Nodes_sigma_0_complete.png")

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


# for i in length(Min_Nodes:Max_Nodes)
#     Plots.plot!(bond_dimension_data(i,:))
# end
#scatter = scatter!(sweeps, bond_dimensions, label=false)