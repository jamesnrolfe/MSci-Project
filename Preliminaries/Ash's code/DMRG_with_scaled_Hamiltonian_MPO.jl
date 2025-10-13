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
    scaling = 1/(Nodes)
    ampo = OpSum()
    for i=1:Nodes
        for j = i+1:Nodes
            ampo += 4*scaling*adj_mat[i,j]*Delta, "Sz",i,"Sz",j
            ampo += -2*scaling*adj_mat[i,j],"S+",i,"S-",j
            ampo += -2*scaling*adj_mat[i,j],"S-",i,"S+",j
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

#---------------------------------------------------------------MAIN-----------------------------------------------------------------------------------
function main(Min_Nodes, Max_Nodes, Probability, μ, σ, Num_Sweeps, χ, J, Δ)
    iterations = length(Min_Nodes:Max_Nodes)
    bond_dimension_data = zeros(Num_Sweeps, iterations)
    
    for Nodes in Min_Nodes:Max_Nodes
        iteration = Nodes - Min_Nodes + 1
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
        println("DRMG Finished over ", Nodes ," Nodes and the Energy Per Site is ", energy/Nodes)
    
        bond_dimension_data[:, iteration] = obs.max_dims
    end
    bond_dimension_dataframe = DataFrame(Tables.table(bond_dimension_data))
    return bond_dimension_dataframe
end;

#Graph variables
Min_Nodes = 10 #minimum number of spin-1/2 particles
Max_Nodes = 100 #maximum number of spin-1/2 particles
Probability = 1.0 #Probability of a given pair of particles interacting
μ = 1 #mean node weighting
σ = 0.0015 #standard deviation of node weighting

#Hamiltonian Variables
J = 1 #Hamiltonian x,y component coefficient
Δ = 1 #Hamiltonian z component coefficient

#DMRG variables
Num_Sweeps = 30 #set maximum number of DMRG sweeps
χ = 1000 #Set maximum bond dimension for DMRG

data_points = 8
iterations_outer = length(Min_Nodes:Max_Nodes)
average_bond_dataframe = zeros(Num_Sweeps, iterations_outer)
average_bond_dataframe = DataFrame(Tables.table(average_bond_dataframe))

for i in 1:data_points
    output_dataframe = main(Min_Nodes, Max_Nodes, Probability, μ, σ, Num_Sweeps, χ, J, Δ)
    average_bond_dataframe .+= output_dataframe
    println("\n")
end

average_bond_dataframe = average_bond_dataframe ./ data_points


println("Simulation Complete")

#--------------------------------------------------------Plotting-------------------------------------------------

final_bond_dimension_vs_num_Nodes = Vector(DataFrameRow(average_bond_dataframe, Num_Sweeps))
bond_dimension_growth = Vector(average_bond_dataframe[:, iterations_outer])

y_axis = range(Min_Nodes, Max_Nodes)
Plots.plot(y_axis, final_bond_dimension_vs_num_Nodes, label=false)
scatter = Plots.scatter!(y_axis, final_bond_dimension_vs_num_Nodes, label=false)
title!("saturated bond dimension for \n an average graph with N Nodes")
xlabel!("Number of Nodes")
ylabel!("Average Bond Dimension Required")
Plots.savefig(scatter,"figures\\bond_growth_$(Min_Nodes)_to_$(Max_Nodes)_sweeps_$(Num_Sweepsweep)_sigma_$(σ)_complete.png")

Plots.plot(bond_dimension_growth, label=false)
scatter = Plots.scatter!(bond_dimension_growth, label=false)
title!("Average Bond dimension after each sweep of DMRG")
xlabel!("Sweeps")
ylabel!("Bond Dimension")
Plots.savefig(scatter,"figures\\bond_dimension_at_each_sweep_$(Max_Nodes)_Nodes_sigma_$(σ)_complete.png")

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