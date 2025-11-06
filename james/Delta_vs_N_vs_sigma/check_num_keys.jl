using Pkg
Pkg.add("JLD2")

using JLD2

function main()
    # Check if the correct number of arguments is provided
    if length(ARGS) < 1
        println("Usage: julia check_num_keys.jl <path_to_jld2_file>")
        return
    end

    # Get the file path from the arguments
    file_path = ARGS[1]

    # Check if the file exists
    if !isfile(file_path)
        println("Error: File $file_path does not exist.")
        return
    end

    function read_and_print_jld2(file_path::String)
        if !isfile(file_path)
            println("Error: File $file_path does not exist.")
            return
        end
    
        println("Reading JLD2 file: $file_path")
        jldopen(file_path, "r") do file
            println("Keys in the file:")
            for key in keys(file)
                value = read(file, key)  # Read the value associated with the key
                println("Key: $key")
                println("Value: $value")
                println("-----------")
            end
        end
    end

    read_and_print_jld2(file_path)

    # Load the JLD2 file and print the keys
    println("Loading JLD2 file: $file_path")
    data = jldopen(file_path, "r") do file
        file["data"]
    end

    for key in keys(data)
        println(key)
    end

    num_keys = length(data)
    println("\nNumber of keys: $num_keys")
end

main()