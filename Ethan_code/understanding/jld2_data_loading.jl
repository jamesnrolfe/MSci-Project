using JLD2

filename = joinpath(@__DIR__, "surface_plot_sigma_data.jld2")

data_dictionary = load(filename)

println("--- All Data in File ---")
println(data_dictionary)


if haskey(data_dictionary, "N_range")
    n_range = data_dictionary["N_range"]
    println("\n--- N_range ---")
    println(n_range)
    
    println("Number of N values: ", length(n_range))
else
    println("\nVariable 'N_range' not found.")
end

if haskey(data_dictionary, "sigma_range")
    sigma_range = data_dictionary["sigma_range"]
    println("\n--- sigma_range ---")
    println(sigma_range)
else
    println("\nVariable 'sigma_range' not found.")
end

if haskey(data_dictionary, "avg_bond_dims")
    avg_dims = data_dictionary["avg_bond_dims"]
    println("\n--- avg_bond_dims ---")
    println(avg_dims)
else
    println("\nVariable 'avg_bond_dims' not found.")
end

zero_elements_array = avg_dims[avg_dims .== 0]
len_of_zero_array = length(zero_elements_array)
println("Length after removing NON-ZERO elements: ", len_of_zero_array)