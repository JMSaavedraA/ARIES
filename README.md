# ARIES

This repository contains the Julia implementation and reproducible benchmarks for
"Aries: A Three-Segment Path for Solving Trust-Region Subproblems".

## Requirements

- Julia 1.10 or newer
- The bundled S2MPJ problem files

Install and instantiate the Julia environment:

```powershell
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

## Single-problem benchmark

```powershell
julia --project=. testOne.jl TOINTPSP
julia --project=. testOne.jl TOINTPSP false
```

The optional second argument disables PDF/figure generation.

## Full benchmark

```powershell
julia --project=. testAll.jl ARIESadaptivevsFixedBFGS
```

The output CSV files are written in the repository root and performance-profile PDFs
are written under `Figures_<benchmark-name>/`.

## S2MPJ attribution

The `s2mpjlib.jl` runtime and `julia_problems/` files are distributed from
[S2MPJ](https://github.com/GrattonToint/S2MPJ). They are provided under the BSD 3-Clause
license reproduced in `LICENCE-S2MPJ.txt`.

The ARIES source in this repository is available under the BSD 3-Clause license
in `LICENSE`.
