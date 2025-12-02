using JLD2

"""
    print_jld2_contents(filename)

Opens a JLD2 file and recursively prints its keys and values.
"""
function print_jld2_contents(filename)
    if !isfile(filename)
        println("Error: File '$filename' not found.")
        return
    end

    println("Opening file: $filename")
    println("="^40)

    jldopen(filename, "r") do file
        function traverse(node, label, depth)
            indent = repeat("    ", depth)
            
            if node isa JLD2.Group
                println("$(indent) [Group] $label")
                for key in keys(node)
                    traverse(node[key], key, depth + 1)
                end

            elseif node isa Dict
                println("$(indent) [Data Set] $label")
                # for (k, v) in node
                #     # If the value is a large array, print a summary instead of the whole thing
                #     if v isa AbstractArray && length(v) > 10
                #         println("$(indent)    🔹 $k: $(typeof(v)) with size $(size(v))")
                #         println("$(indent)       (First 5: $(v[1:min(5, end)])...)")
                #     else
                #         println("$(indent)    🔹 $k: $v")
                #     end
                # end
                println("") 

            else
                println("$(indent)🔹 $label: $node")
            end
        end

        for key in keys(file)
            traverse(file[key], key, 0)
        end
    end
end

filename = joinpath(@__DIR__,"mach_prec_data.jld2")
print_jld2_contents(filename)