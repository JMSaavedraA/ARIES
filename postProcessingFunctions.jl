using CSV, DataFrames, Plots
include("processingFunctions.jl")

test_name = length(ARGS) >= 1 ? String(ARGS[1]) : "ARIESadaptivevsFixedBFGS"


function loadResults(test_name, testname)
    df = CSV.read("test_results_"*test_name*"_clean.csv", DataFrame)
    results = Matrix(df[:, 2:end])
    functionsToTest = df[:, 1]
    return df, results, functionsToTest
end

df, results, functionsToTest = loadResults(test_name, test_name)

algos_names = ["Dogleg", "ARIES α = 1/4, β = 1/2", "Adaptive ARIES"]

plot_and_save_profile(results, algos_names, "Figures_"*test_name)