 using LinearAlgebra, Base.Threads, DelimitedFiles, StatsBase, Plots, CSV, DataFrames

include("s2mpjlib.jl")
include("trustRegionARIES.jl")
include("functions.jl")
include("processingFunctions.jl")

for f in functionsToTest
    loadproblem(f)
end

algos = ["trustRegionDoglegBFGS", "trustRegionARIESBFGS12", "trustRegionARIESBFGSAdaptive10"]

results = compareMethods(functionsToTest, algos)

# Define header
header = ["Function", "gD", "kD", "tD", "gA", "kA", "tA", "gAA", "kAA", "tAA"]

# Combine function names and results into a DataFrame
df = DataFrame([:Function => functionsToTest], copycols=false)
for (i, col) in enumerate(header[2:end])
    df[!, col] = results[:, i]
end

test_name = length(ARGS) >= 1 ? String(ARGS[1]) : "ARIESadaptivevsFixedBFGS"

# Save to CSV
CSV.write("test_results_"*test_name*".csv", df)

println("Results saved to test_results_"*test_name*".csv")

algos_names = ["Dogleg", "ARIES α = 1/4, β = 1/2", "Adaptive ARIES"]

checkConvergence(results, algos_names)
for (i, col) in enumerate(header[2:end])
    df[!, col] = results[:, i]
end
CSV.write("test_results_"*test_name*"_clean.csv", df)
plot_and_save_profile(results, algos_names, "Figures_"*test_name)