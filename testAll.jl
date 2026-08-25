using LinearAlgebra, Base.Threads, DelimitedFiles, StatsBase, Plots, CSV, DataFrames

include("s2mpjlib.jl")
include("trustRegionARIES.jl")
include("functions.jl")
include("processingFunctions.jl")

for f in functionsToTest
    loadproblem(f)
end

algos = ["trustRegionDoglegBFGS", "trustRegionARIESBFGS11", "trustRegionARIESBFGS12","trustRegionARIESBFGS13", "trustRegionARIESBFGS21", "trustRegionARIESBFGS22","trustRegionARIESBFGS23", "trustRegionARIESBFGS31", "trustRegionARIESBFGS32","trustRegionARIESBFGS33", "trustRegionARIESBFGSAdaptive10"]

results = compareMethods(functionsToTest, algos)

# Define header
header = ["Function", "gD", "kD", "tD", "gA11", "kA11", "tA11", "gA12", "kA12", "tA12", "gA13", "kA13", "tA13", "gA21", "kA21", "tA21", "gA22", "kA22", "tA22", "gA23", "kA23", "tA23", "gA31", "kA31", "tA31", "gA32", "kA32", "tA32", "gA33", "kA33", "tA33", "gAA10", "kAA10", "tAA10"]

# Combine function names and results into a DataFrame
df = DataFrame([:Function => functionsToTest], copycols=false)
for (i, col) in enumerate(header[2:end])
    df[!, col] = results[:, i]
end

test_name = "ARIESadaptivevsFixedBFGS"

# Save to CSV
CSV.write("test_results_"*test_name*".csv", df)

println("Results saved to test_results_"*test_name*".csv")

algos_names = ["Dogleg", "α = 1/4, β = 1/4", "α = 1/4, β = 1/2", "α = 1/4, β = 3/4", "α = 1/2, β = 1/4", "α = 1/2, β = 1/2", "α = 1/2, β = 3/4", "α = 3/4, β = 1/4", "α = 3/4, β = 1/2", "α = 3/4, β = 3/4", "Adaptive 10"]

checkConvergence(results, algos_names)
for (i, col) in enumerate(header[2:end])
    df[!, col] = results[:, i]
end
CSV.write("test_results_"*test_name*"_clean.csv", df)
plot_and_save_profile(results, algos_names, "Figures_"*test_name)