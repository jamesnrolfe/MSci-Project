using JSON

function load_data_from_json(file_path::String)
    if isfile(file_path)
        open(file_path, "r") do f
            loaded_data = JSON.parse(f)
            return loaded_data
        end
    end
    return nothing
end

function write_data_as_json(file_path::String, data::Dict)
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