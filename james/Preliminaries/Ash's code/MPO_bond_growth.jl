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

#---------------------------------------------------------------MAIN-----------------------------------------------------------------------------------
function main(Min_Nodes, Max_Nodes, Probability, μ, σ, Num_Sweeps, χ, J, Δ)
    iterations = length(Min_Nodes:Max_Nodes)
    bond_dimension_data = zeros(iterations)

    for Nodes in Min_Nodes:Max_Nodes
        iteration = Nodes - Min_Nodes + 1
        #Generate a random Graph and it's adjacency matrix, adj_mat
        G = LightGraphs.erdos_renyi(Nodes, Probability)
        adj_mat = LightGraphs.adjacency_matrix(G) # A is a sparse matrix
        adj_mat = random_weighting(Nodes, adj_mat, μ, σ) # randomise the weightings on the graph
        
        #Initialise site inds and initial state as Neel state. Conserve total \S_{z}.
        sites = siteinds("S=1/2", Nodes; conserve_qns = true)
        #init_state = [isodd(i) ? "Up" : "Dn" for i = 1:Nodes]
        #psi0 = randomMPS(sites, init_state)
    
        #Create the Hamiltonian for the DMRG solution
        H = create_XXZ_Ham_MPO(Nodes, adj_mat, Δ, sites)
        link = maxlinkdim(H)

        bond_dimension_data[iteration] = link
    end
    bond_dimension_dataframe = DataFrame(Tables.table(bond_dimension_data))
    return bond_dimension_dataframe
end;

#Graph variables
Min_Nodes = 10 #minimum number of spin-1/2 particles
Max_Nodes = 100 #maximum number of spin-1/2 particles
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
iterations_outer = length(Min_Nodes:Max_Nodes)
average_bond_data = zeros(iterations_outer)
#average_bond_dataframe = DataFrame(Tables.table(average_bond_dataframe))

s_1 = zeros(iterations_outer)
s_2 = zeros(iterations_outer)

for i in 1:data_points
    output_data = main(Min_Nodes, Max_Nodes, Probability, μ, σ, Num_Sweeps, χ, J, Δ)

    s_1 .+= output_data
    s_2 .+= output_data.^2

    average_bond_data .+= output_data
    println("$(i) runs complete for σ=$(σ)")
    println("\n")
end

average_bond_data = average_bond_data ./ data_points

#\sigma = \frac{\sqrt{Ns_2 - s_1^{2}}}{N} used for standard deviation

s_1 = s_1 ./ data_points
s_1 = s_1.^2
s_2 = s_2 ./ data_points
std_dataframe = (abs.(s_2 .- s_1)).^0.5

#----σ=0 case----------------------------------------------------------------
σ = 0
data_points = 3
iterations_outer = length(Min_Nodes:Max_Nodes)
average_bond_data_sigma0  = zeros(iterations_outer)
#average_bond_dataframe = DataFrame(Tables.table(average_bond_dataframe))

s_1_sigma0 = zeros(iterations_outer)
s_2_sigma0  = zeros(iterations_outer)

for i in 1:data_points
    output_data_sigma0  = main(Min_Nodes, Max_Nodes, Probability, μ, σ, Num_Sweeps, χ, J, Δ)

    s_1_sigma0 .+= output_data_sigma0 
    s_2_sigma0 .+= output_data_sigma0.^2

    average_bond_data_sigma0 .+= output_data_sigma0
    println("$(i) runs complete for σ=$(σ)")
    println("\n")
end

average_bond_data_sigma0  = average_bond_data_sigma0 ./ data_points

#\sigma = \frac{\sqrt{Ns_2 - s_1^{2}}}{N} used for standard deviation

s_1_sigma0 = s_1_sigma0 ./ data_points
s_1_sigma0 = s_1_sigma0.^2
s_2_sigma0 = s_2_sigma0 ./ data_points
std_data_sigma0 = (abs.(s_2_sigma0 .- s_1_sigma0)).^0.5

println("Simulation Complete")

#--------------------------------------------------------Plotting-------------------------------------------------

#final_bond_dimension_vs_num_Nodes = Vector(DataFrameRow(average_bond_data))
#bond_dimension_growth = Vector(average_bond_dataframe[:, iterations_outer])

y_axis = range(Min_Nodes, Max_Nodes)
Plots.plot(y_axis, average_bond_data, label=false)
scatter = Plots.scatter!(y_axis, average_bond_data, yerr=std_dataframe, label="σ=0.002")

Plots.plot!(y_axis, average_bond_data_sigma0, label=false)
Plots.scatter!(y_axis, average_bond_data_sigma0, yerr=std_dataframe_sigma0, label="σ=0")

title!("saturated bond dimension for \n an average graph with N Nodes")
xlabel!("Number of Nodes")
ylabel!("Bond Dimension Required")
Plots.savefig(scatter,"figures\\MPO_growth_test")