using LinearAlgebra, Base.Threads, DelimitedFiles, StatsBase, Plots, CSV, DataFrames

include("s2mpjlib.jl")
include("trustRegionARIES.jl")
include("processingFunctions.jl")

funcName = length(ARGS) >= 1 ? uppercase(String(ARGS[1])) : "TOINTPSP"
plotting = length(ARGS) >= 2 ? !(lowercase(String(ARGS[2])) in ("false", "0", "no")) : true

algos = ["trustRegionDoglegBFGS", "trustRegionARIESBFGS12", "trustRegionARIESBFGSAdaptive10"]

loadproblem(funcName)
allGrads, allFVals, allAcc, allTimes, allIters = compareMethodsOneFunc(funcName, algos)

kMax = maximum(allIters)

for i in eachindex(algos)
    allFVals[allIters[i]:kMax, i] .= allFVals[allIters[i], i]
    allGrads[allIters[i]:kMax, i] .= allGrads[allIters[i], i]
end

algos_names = ["Dogleg", "ARIES α = 1/4, β = 1/2", "Adaptive ARIES"]

maxIter = size(allFVals, 1)

if(plotting)
    # Plot gradient norm histories
    plt1 = plot()
    for (i, algo) in enumerate(algos_names)
        plot!(plt1,
            1:kMax,
            allGrads[1:kMax, i],
            label=algo,
            xlabel="Iteration",
            ylabel="Gradient Norm",
            title="Gradient Norm Histories for $funcName",
            titlefont = 14,
            guidefont = 12,
            tickfont = 12,
            linewidth = 2,
            legendfont = 12,
            size = (600, 400),
            palette = :Dark2_8,
            dpi = 600,
            ylims = (0, maximum(allGrads[kMax,:]) * 100)
            )
    end
    # Optionally save the figure
    savefig(plt1, string(funcName, "_gradient_norms.pdf"))

    # Plot function value histories
    plt2 = plot()
    f_final_min = minimum(allFVals[kMax,:])
    f_final_max = maximum(allFVals[kMax,:])

    # Adjust limits to handle positive, mixed, and negative cases
    y_low = f_final_min < 0 ? f_final_min * 1.1 : f_final_min * 0.9
    y_high = f_final_max > 0 ? f_final_max * 4 * 10^0 : f_final_max * 0.9

    for (i, algo) in enumerate(algos_names)
        plot!(plt2,
            1:kMax,
            allFVals[1:kMax, i],
            label=algo,
            xlabel="Iteration",
            ylabel="Function Value",
            title="Function Value Histories for $funcName",
            titlefont = 14,
            guidefont = 12,
            tickfont = 12,
            linewidth = 2,
            legendfont = 12,
            size = (600, 400),
            palette = :Dark2_8,
            dpi = 1200,
            ylims = (y_low, y_high)
            )
    end
    # Optionally save the figure
    savefig(plt2, string(funcName, "_function_values.pdf"))
end

# Create CSV with acceptance percentage, CPU time, and iterations


results_df = DataFrame(
    "Algorithm" => algos,
    "Acceptance Percentage" => allAcc,
    "CPU Time" => allTimes,
    "Iterations" => allIters
)

CSV.write(string(funcName, "_summary.csv"), results_df)

println("Summary CSV file created: ", string(funcName, "_summary.csv"))
println("Done!")