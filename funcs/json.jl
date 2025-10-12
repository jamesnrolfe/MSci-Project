using JSON

function load_data_from_json(file_path::String)
    if isfile(file_path)
        open(file_path, "r") do f
            loaded_data = JSON.parse(f)
            data = Dict{Float64, Dict{Int, Tuple{Float64, Float64}}}()  # Initialize empty data structure
            # Convert string keys back to proper types
            for (σ_str, σ_data) in loaded_data
                σ = parse(Float64, σ_str)  # Convert string key to Float64
                data[σ] = Dict{Int, Tuple{Float64, Float64}}()
                
                for (N_str, values) in σ_data
                    N = parse(Int, N_str)  # Convert string key to Int
                    # Values might be arrays, convert to tuple
                    if isa(values, Array)
                        data[σ][N] = (values[1], values[2])
                    else
                        data[σ][N] = values
                    end
                end
            end
        end
    end
    return data
end

function write_data_as_json(file_path::String, data::Dict{Float64, Dict{Int, Tuple{Float64, Float64}}})
    # Convert keys to strings for JSON compatibility
    json_compatible_data = Dict{String, Any}()
    for (σ, σ_data) in data
        σ_str = string(σ)
        json_compatible_data[σ_str] = Dict{String, Any}()
        
        for (N, values) in σ_data
            N_str = string(N)
            json_compatible_data[σ_str][N_str] = values
        end
    end
    
    open(file_path, "w") do f
        JSON.print(f, json_compatible_data)
    end
end