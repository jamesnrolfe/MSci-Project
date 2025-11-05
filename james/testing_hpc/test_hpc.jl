import Pkg
Pkg.add("JLD2")

using JLD2

test_file = "test_data.jld2"

function square(x)
    return x^2
end

function test_jld2()
    # Create some test data
    nums = 1:10
    data = [square(n) for n in nums]

    # Save the data to a JLD2 file
    jldsave(test_file; data)

    println("JLD2 save and load test passed.")
    println(data)
end

test_jld2()