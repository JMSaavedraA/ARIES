#Jose Miguel Saavedra Aguilar
#CIMAT Master in Applied Mathematics

using LinearAlgebra

include("s2mpjlib.jl")

"""
ARIES trust-region implementations used in
"Aries: A Three-Segment Path for Solving Trust-Region Subproblems".

Provides the Dogleg reference method, fixed-parameter ARIES variants, and
adaptive ARIES variants. Solving functions accept an optional iteration limit.
"""

"""
updateΔ(Δ, ρ, c1, c2, η, ΔMin, ΔMax, d, c)

Compute the updated trust-region radius according to the agreement ratio `ρ`.

Arguments
- `Δ` : current trust-region radius
- `ρ` : ratio of actual to predicted reduction (model agreement)
- `c1`, `c2` : thresholds for contraction / expansion
- `η` : expansion/contraction factor (>1)
- `ΔMin`, `ΔMax` : minimum and maximum allowed radius
- `d` : proposed step (kept for API compatibility)
- `c` : code from `dogleg` indicating which region was chosen ("N","D","C")

Returns
- new `Δ` (Number). Note: the caller should assign the returned value if they want
  to update the radius in-place: `Δ = updateΔ(Δ, ρ, ...)`.
"""
function updateΔ(Δ, ρ, c1, c2, η, ΔMin, ΔMax, d, c)
    if (ρ < c1)
        # Contraction: poor model agreement -> reduce radius
        Δ = max(Δ / η , ΔMin)
    elseif (ρ > c2 && c != "N")
        # Expansion: very good agreement and not a Newton step -> increase radius
        Δ = min(η * Δ, ΔMax)
    #else
        # keep Δ unchanged (alternative: set Δ = norm(d))
    end
    return Δ
end

function safeCosine(a::AbstractVector, b::AbstractVector)
    na = norm(a)
    nb = norm(b)
    if !isfinite(na) || !isfinite(nb) || na <= eps(Float64) || nb <= eps(Float64)
        return 0.0
    end
    value = dot(a, b) / (na * nb)
    return clamp(real(value), -1.0, 1.0)
end

function safeCauchyPoint(g::AbstractVector, B::AbstractMatrix)
    Ng2 = dot(g, g)
    curvature = dot(g, B * g)
    if !isfinite(curvature) || curvature <= eps(Float64)
        return -copy(g)
    end
    pU = -Ng2 / curvature * g
    if !all(isfinite, pU)
        return -copy(g)
    end
    return pU
end

"""
dogleg(Δ, g, B, pU, pN, d; tol=1e-6, dim=length(g))

Solve the trust-region subproblem approximately using the Dogleg method.

This routine computes (in-place) a step `d` following the dogleg path defined by
the Cauchy point `pU` and the (approximate) Newton step `pN`. The chosen step is
written into `d` and a code is returned indicating which region was used:
- "C" : Cauchy (steepest-descent) point clipped to the boundary
- "N" : Full Newton step (inside the trust region)
- "D" : Dogleg interpolation between Cauchy and Newton points

Arguments
- `Δ` : trust-region radius
- `g` : gradient vector
- `B` : (approximate) Hessian matrix
- `pU`, `pN` : preallocated vectors for Cauchy and Newton candidates
- `d` : output step (written in-place)

Returns
- `c` : String code describing which region was selected ("C","N","D").
"""
function dogleg(Δ::Number,g::AbstractVector,B::AbstractMatrix, pU::AbstractVector, pN::AbstractVector, d::AbstractVector, w::AbstractVector)
    # Compute the scaled steepest-descent (Cauchy) point pU
    @inbounds pU .= safeCauchyPoint(g, B)
    NpU = norm(pU)
    if NpU > Δ
        # Cauchy point lies outside the trust region -> scale to the boundary
        @inbounds d .= Δ*pU/NpU
        c = "C"
    else
        # Compute (approximate) Newton step pN by solving B*pN = -g
        @inbounds pN .= - (B \ g)
        if norm(pN) <= Δ
            # Newton step is inside trust region -> accept it
            @inbounds d .= pN
            c = "N"
        else
            # Interpolate along the dogleg path between pU and pN
            @inbounds w .= pN-pU
            p1 = w⋅w
            p2 = pU⋅w
            p3 = (Δ + NpU) * (Δ - NpU)
            t = (-p2 + sqrt(p2^2 + p1*p3)) / p1
            @inbounds d .= pU + t*w
            c = "D"
        end
    end
    return c
end

function ARIES(Δ::Number,g::AbstractVector,B::AbstractMatrix, pU::AbstractVector, pN::AbstractVector, d::AbstractVector, w::AbstractVector, v::AbstractVector, α::Number = 1/2, β::Number = 1/2)
    # Dogleg method for the trust region problem
    if !isfinite(α) || !isfinite(β) || α < 0 || α > 1 || β < 0 || β > 1
        throw(ArgumentError("ARIES parameters α and β must be finite values in [0, 1]"))
    end
    @inbounds pU .= safeCauchyPoint(g, B)
    @inbounds v .= α * pU
    NpU = norm(pU)
    Nv = α * NpU
    if Nv > Δ
        # Case 1: direction is scalled gradient
        @inbounds d .= Δ*pU/NpU
        c = "C"
    else
        @inbounds pN .= - (B \ g)
        nPN = norm(pN)
        if nPN <= Δ
            # Case 4: direction is Newton's step
            @inbounds d .= pN
            c = "N"
        elseif norm(β * (pN - pU) + pU) > Δ
            # Case 2. direction is on the middle path
            @inbounds w .= β * pN + (1 - α - β) * pU
            p1 = w⋅w
            p2 = w⋅v
            p3 = (Δ + Nv) * (Δ - Nv)
            if p1 <= eps(Float64)
                throw(SingularException(1))
            end
            discriminant = p2^2 + p1*p3
            discriminant < 0 && (discriminant = 0.0)
            t = (-p2 + sqrt(discriminant)) / p1
            @inbounds d .= v + t * w
            c = "D1"
        else
            # Case 3. direction is on the dogleg path
            @inbounds w .= pN-pU
            p1 = w⋅w
            p2 = pU⋅w
            p3 = (Δ + NpU) * (Δ - NpU)
            t = (-p2 + sqrt(p2^2 + p1*p3)) / p1
            @inbounds d .= pU + t*w
            c = "D2"
        end
    end
    return c
end

function ARIES2(Δ::Number,g::AbstractVector,B::AbstractMatrix, pU::AbstractVector, pN::AbstractVector, d::AbstractVector, w::AbstractVector, v::AbstractVector, α::Number = 1/2, β::Number = 1/2)
    # Dogleg method for the trust region problem
    if !isfinite(α) || !isfinite(β) || α < 0 || α > 1 || β < 0 || β > 1
        throw(ArgumentError("ARIES parameters α and β must be finite values in [0, 1]"))
    end
    @inbounds v .= α * pU
    NpU = norm(pU)
    Nv = α * NpU
    if Nv > Δ
        # Case 1: direction is scalled gradient
        @inbounds d .= Δ*pU/NpU
        c = "C"
    else
        @inbounds pN .= - (B \ g)
        nPN = norm(pN)
        if nPN <= Δ
            # Case 4: direction is Newton's step
            @inbounds d .= pN
            c = "N"
        elseif norm(β * (pN - pU) + pU) > Δ
            # Case 2. direction is on the middle path
            @inbounds w .= β * pN + (1 - α - β) * pU
            p1 = w⋅w
            p2 = w⋅v
            p3 = (Δ + Nv) * (Δ - Nv)
            p1 <= eps(Float64) && throw(SingularException(1))
            discriminant = p2^2 + p1*p3
            discriminant < 0 && (discriminant = 0.0)
            t = (-p2 + sqrt(discriminant)) / p1
            @inbounds d .= v + t * w
            c = "D1"
        else
            # Case 3. direction is on the dogleg path
            @inbounds w .= pN-pU
            p1 = w⋅w
            p2 = pU⋅w
            p3 = (Δ + NpU) * (Δ - NpU)
            t = (-p2 + sqrt(p2^2 + p1*p3)) / p1
            @inbounds d .= pU + t*w
            c = "D2"
        end
    end
    return c
end

"""
trustRegionDoglegBFGS(pb, pbm, f)

Trust-region method using the Dogleg subproblem solver and a BFGS Hessian update.

Arguments
- `pb`  : `PB` problem descriptor (from `s2mpjlib`)
- `pbm` : `PBM` problem metadata
- `f`   : problem evaluation function (e.g. `probfunc` returning `f`, `g`, `H`)

Returns
- `x`  : final iterate
- `Ng` : final gradient norm
- `k`  : number of iterations performed
- `tol`: gradient tolerance used
- `F`  : history of function values per iteration
- `G`  : history of gradient norms per iteration
- `acc`: acceptance ratio (accepted steps / total iterations)
"""
function trustRegionDoglegBFGS(pb::PB, pbm::PBM, f::Function, maxIter::Int = 100000)
    G = zeros(Float64, maxIter)
    F = copy(G)
    x = copy(pb.x0)
    n = length(x)  # the length of x_k
    # Algorithm parameters: Armijo / radius thresholds / update factors
    w1, c0, c1, c2, η, λ = 1e-4, 0.1, 1/4, 1/2, 2, 0.8
    ΔMax, ΔMin = 10 * sqrt(n), 0.01 * sqrt(n)
    xs = similar(x)
    k = 0
    acc = 0 # Accepted steps counter
    tol = 1e-4
    # Evaluate initial f, g, H/B
    f0, g, B = f("fgHx", pbm, x)
    try
        B = Matrix(B)
        # Ensure B is symmetric positive definite (SPD) via factorization
        S = bunchkaufman(Symmetric(collect(B)))
        minD = minimum(diag(S.D))
        if minD < tol
            B .= S.P'*S.U*abs.(S.D)*S.U'*S.P
        end
        # Regularize if ill-conditioned
        if cond(B) * tol > 1
            B .= 0.8 * B + 0.2 * I
        end
    catch
        # Fallback to identity if Hessian is unavailable or factorization fails
        B = Matrix{Float64}(I, n, n)
    end
    Ng = norm(g)
    y = similar(g)
    pU = similar(g)
    pN = similar(g)
    gs = similar(g)
    c = "I"
    u = similar(g)
    d = similar(g)
    d .= -g
    v = similar(g)
    jter = 0
    Δ = ΔMax
    # Line search for initial Δ
    try
        f1 = f("fx", pbm, x + Δ * d)
        goOn = f1 > f0 + w1 * Δ * dot(g, d)
        while goOn && jter < 30
            Δ .= λ * Δ
            f1 = f("fx", pbm, x + Δ * d)
            goOn = f1 > f0 + w1 * Δ * dot(g, d)
            jter += 1
        end
    catch
        Δ = norm(g)
    end

    t_start = time()  # Start timer
    while Ng > tol && k < maxIter
        # Timeout check
        if time() - t_start > 3600
            println("Timeout: Exceeded 1 hour in trustRegionDogleg.")
            break
        end
        if any(isnan.(B)) || any(isinf.(B))
            # Initial attempt to recompute Hessian if B is invalid
            _, _, B = f("fgHx", pbm, x)
            B = Matrix(B)
        end
        try
            if any(isnan.(B)) || any(isinf.(B))
                # In case B is still invalid, fallback to identity
                B = Matrix{Float64}(I, n, n)
            else
                # Ensure B is symmetric positive definite (SPD) via factorization
                S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
                minD = minimum(diag(S.D))
                if minD < tol
                    B .= S.P'*S.U*abs.(S.D)*S.U'*S.P #B is P.D. iff D has positive entries
                end
                # Regularize if ill-conditioned
                if cond(B) * tol > 1
                    B .= 0.8 * B + 0.2 * I
                end
            end
        catch
            B = Matrix{Float64}(I, n, n)
        end
        c = dogleg(Δ, g, B, pU, pN, d, v)
        @. xs = x + d
        try
            # Evaluate the objective at the candidate step and compute the trust ratio
            f1 = evalgrsum!(xs, pbm, gs)
            denom = 0.5 * dot(d, B*d) + dot(g, d)  # predicted reduction (model)
            if !isfinite(denom) || abs(denom) < 1e-14
                # Avoid division by zero or non-finite model predictions
                ρ = -Inf
            else
                ρ = (f1 - f0) / denom
            end
            if ρ < c0 && Δ > ΔMin
                # Poor agreement -> shrink trust region and reject step
                Δ = max(Δ / η, ΔMin)
            else
                # Accept step: update iterate, gradient and perform BFGS update
                x .= xs
                y .= gs .- g
                # u = B * d (used in BFGS denominator)
                mul!(u, B, d)
                yd = dot(y, d)
                du = dot(d, u)
                # BFGS rank-two update: ensure denominators are safe before update
                if isfinite(yd) && isfinite(du) && abs(yd) > 1e-14 && abs(du) > 1e-14
                    B .+= (y * y') / yd - (u * u') / du
                else
                    # Skip BFGS update to avoid division by zero / numerical issues
                end
                g .= gs
                f0 = f1
                Ng = norm(g)
                # Update trust radius (returns a new Δ; assign if desired)
                Δ = updateΔ(Δ, ρ, c1, c2, η, ΔMin, ΔMax, d, c)
                acc += 1
            end
        catch
            # Evaluation failed: conservatively reduce Δ
            Δ = max(Δ / η, ΔMin)
        end
        k += 1
        G[k] = Ng
        F[k] = f0
    end
    G = G[1:k]  # Truncate the gradient norm history
    F = F[1:k]  # Truncate the function value history
    acc = acc / k  # Compute acceptance ratio
    return x, Ng, k, tol, G, F, acc
end

"""
trustRegionARIESBFGS(pb, pbm, f)

Trust-region method using the ARIES subproblem solver and a BFGS Hessian update.

Arguments
- `pb`  : `PB` problem descriptor (from `s2mpjlib`)
- `pbm` : `PBM` problem metadata
- `f`   : problem evaluation function (e.g. `probfunc` returning `f`, `g`, `H`)
- `α`, `β`: parameters for the ARIES subproblem solver (default 1/2, 1/2)

Returns
- `x`  : final iterate
- `Ng` : final gradient norm
- `k`  : number of iterations performed
- `tol`: gradient tolerance used
- `F`  : history of function values per iteration
- `G`  : history of gradient norms per iteration
- `acc`: acceptance ratio (accepted steps / total iterations)
"""
function trustRegionARIESBFGS(pb::PB, pbm::PBM, f::Function, α::Number=1/2, β::Number=1/2, maxIter::Int = 100000)
    G = zeros(Float64, maxIter)
    F = copy(G)
    x = copy(pb.x0)
    n = length(x)  # the length of x_k
    # Algorithm parameters: Armijo / radius thresholds / update factors
    w1, c0, c1, c2, η, λ = 1e-4, 0.1, 1/4, 1/2, 2, 0.8
    ΔMax, ΔMin = 10 * sqrt(n), 0.01 * sqrt(n)
    xs = similar(x)
    k = 0
    acc = 0 # Accepted steps counter
    tol = 1e-4
    # Evaluate initial f, g, H/B
    f0, g, B = f("fgHx", pbm, x)
    try
        B = Matrix(B)
        # Ensure B is symmetric positive definite (SPD) via factorization
        S = bunchkaufman(Symmetric(collect(B)))
        minD = minimum(diag(S.D))
        if minD < tol
            B .= S.P'*S.U*abs.(S.D)*S.U'*S.P
        end
        # Regularize if ill-conditioned
        if cond(B) * tol > 1
            B .= 0.8 * B + 0.2 * I
        end
    catch
        # Fallback to identity if Hessian is unavailable or factorization fails
        B = Matrix{Float64}(I, n, n)
    end
    Ng = norm(g)
    y = similar(g)
    pU = similar(g)
    pN = similar(g)
    gs = similar(g)
    c = "I"
    u = similar(g)
    d = similar(g)
    d .= -g
    v = similar(g)
    w = similar(g)
    jter = 0
    Δ = ΔMax
    # Line search for initial Δ
    try
        f1 = f("fx", pbm, x + Δ * d)
        goOn = f1 > f0 + w1 * Δ * dot(g, d)
        while goOn && jter < 30
            Δ .= λ * Δ
            f1 = f("fx", pbm, x + Δ * d)
            goOn = f1 > f0 + w1 * Δ * dot(g, d)
            jter += 1
        end
    catch
        Δ = norm(g)
    end

    t_start = time()  # Start timer
    while Ng > tol && k < maxIter
        # Timeout check
        if time() - t_start > 3600
            println("Timeout: Exceeded 1 hour in trustRegionDogleg.")
            break
        end
        if any(isnan.(B)) || any(isinf.(B))
            # Initial attempt to recompute Hessian if B is invalid
            _, _, B = f("fgHx", pbm, x)
            B = Matrix(B)
        end
        try
            if any(isnan.(B)) || any(isinf.(B))
                # In case B is still invalid, fallback to identity
                B = Matrix{Float64}(I, n, n)
            else
                # Ensure B is symmetric positive definite (SPD) via factorization
                S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
                minD = minimum(diag(S.D))
                if minD < tol
                    B .= S.P'*S.U*abs.(S.D)*S.U'*S.P #B is P.D. iff D has positive entries
                end
                # Regularize if ill-conditioned
                if cond(B) * tol > 1
                    B .= 0.8 * B + 0.2 * I
                end
            end
        catch
            B = Matrix{Float64}(I, n, n)
        end
        c = ARIES(Δ, g, B, pU, pN, d, w, v, α, β)
        @. xs = x + d
        try
            # Evaluate the objective at the candidate step and compute the trust ratio
            f1 = evalgrsum!(xs, pbm, gs)
            denom = 0.5 * dot(d, B*d) + dot(g, d)  # predicted reduction (model)
            if !isfinite(denom) || abs(denom) < 1e-14
                # Avoid division by zero or non-finite model predictions
                ρ = -Inf
            else
                ρ = (f1 - f0) / denom
            end
            if ρ < c0 && Δ > ΔMin
                # Poor agreement -> shrink trust region and reject step
                Δ = max(Δ / η, ΔMin)
            else
                # Accept step: update iterate, gradient and perform BFGS update
                x .= xs
                y .= gs .- g
                # u = B * d (used in BFGS denominator)
                mul!(u, B, d)
                yd = dot(y, d)
                du = dot(d, u)
                # BFGS rank-two update: ensure denominators are safe before update
                if isfinite(yd) && isfinite(du) && abs(yd) > 1e-14 && abs(du) > 1e-14
                    B .+= (y * y') / yd - (u * u') / du
                else
                    # Skip BFGS update to avoid division by zero / numerical issues
                end
                g .= gs
                f0 = f1
                Ng = norm(g)
                # Update trust radius (returns a new Δ; assign if desired)
                Δ = updateΔ(Δ, ρ, c1, c2, η, ΔMin, ΔMax, d, c)
                acc += 1
            end
        catch
            # Evaluation failed: conservatively reduce Δ
            Δ = max(Δ / η, ΔMin)
        end
        k += 1
        G[k] = Ng
        F[k] = f0
    end
    G = G[1:k]  # Truncate the gradient norm history
    F = F[1:k]  # Truncate the function value history
    acc = acc / k  # Compute acceptance ratio
    return x, Ng, k, tol, G, F, acc
end

function trustRegionARIESBFGS11(pb::PB, pbm::PBM, f::Function, maxIter::Int = 100000)
    return trustRegionARIESBFGS(pb, pbm, f, 1/4, 1/4, maxIter)
end

function trustRegionARIESBFGS12(pb::PB, pbm::PBM, f::Function, maxIter::Int = 100000)
    return trustRegionARIESBFGS(pb, pbm, f, 1/4, 1/2, maxIter)
end

function trustRegionARIESBFGS13(pb::PB, pbm::PBM, f::Function, maxIter::Int = 100000)
    return trustRegionARIESBFGS(pb, pbm, f, 1/4, 3/4, maxIter)
end

function trustRegionARIESBFGS21(pb::PB, pbm::PBM, f::Function, maxIter::Int = 100000)
    return trustRegionARIESBFGS(pb, pbm, f, 1/2, 1/4, maxIter)
end

function trustRegionARIESBFGS22(pb::PB, pbm::PBM, f::Function, maxIter::Int = 100000)
    return trustRegionARIESBFGS(pb, pbm, f, 1/2, 1/2, maxIter)
end

function trustRegionARIESBFGS23(pb::PB, pbm::PBM, f::Function, maxIter::Int = 100000)
    return trustRegionARIESBFGS(pb, pbm, f, 1/2, 3/4, maxIter)
end

function trustRegionARIESBFGS31(pb::PB, pbm::PBM, f::Function, maxIter::Int = 100000)
    return trustRegionARIESBFGS(pb, pbm, f, 3/4, 1/4, maxIter)
end

function trustRegionARIESBFGS32(pb::PB, pbm::PBM, f::Function, maxIter::Int = 100000)
    return trustRegionARIESBFGS(pb, pbm, f, 3/4, 1/2, maxIter)
end

function trustRegionARIESBFGS33(pb::PB, pbm::PBM, f::Function, maxIter::Int = 100000)
    return trustRegionARIESBFGS(pb, pbm, f, 3/4, 3/4, maxIter)
end





"""
trustRegionARIESBFGSAdaptivek(pb, pbm, f)

Trust-region method using the (k-th version of the) Adaptive ARIES subproblem solver and a BFGS Hessian update.

Arguments
- `pb`  : `PB` problem descriptor (from `s2mpjlib`)
- `pbm` : `PBM` problem metadata
- `f`   : problem evaluation function (e.g. `probfunc` returning `f`, `g`, `H`)

Returns
- `x`  : final iterate
- `Ng` : final gradient norm
- `k`  : number of iterations performed
- `tol`: gradient tolerance used
- `F`  : history of function values per iteration
- `G`  : history of gradient norms per iteration
- `acc`: acceptance ratio (accepted steps / total iterations)
"""

function trustRegionARIESBFGSAdaptive1(pb::PB, pbm::PBM, f::Function, maxIter::Int = 100000)
    G = zeros(Float64, maxIter)
    F = copy(G)
    x = copy(pb.x0)
    n = length(x)  # the length of x_k
    # Algorithm parameters: Armijo / radius thresholds / update factors
    w1, c0, c1, c2, η, λ = 1e-4, 0.1, 1/4, 1/2, 2, 0.8
    ΔMax, ΔMin = 10 * sqrt(n), 0.01 * sqrt(n)
    xs = similar(x)
    k = 0
    acc = 0 # Accepted steps counter
    tol = 1e-4
    # Evaluate initial f, g, H/B
    f0, g, B = f("fgHx", pbm, x)
    try
        B = Matrix(B)
        # Ensure B is symmetric positive definite (SPD) via factorization
        S = bunchkaufman(Symmetric(collect(B)))
        minD = minimum(diag(S.D))
        if minD < tol
            B .= S.P'*S.U*abs.(S.D)*S.U'*S.P
        end
        # Regularize if ill-conditioned
        if cond(B) * tol > 1
            B .= 0.8 * B + 0.2 * I
        end
    catch
        # Fallback to identity if Hessian is unavailable or factorization fails
        B = Matrix{Float64}(I, n, n)
    end
    Ng = norm(g)
    y = similar(g)
    pU = similar(g)
    pN = similar(g)
    gs = similar(g)
    c = "I"
    u = similar(g)
    d = similar(g)
    d .= -g
    v = similar(g)
    w = similar(g)
    α = 1/2
    β = 1/2
    jter = 0
    Δ = ΔMax
    # Line search for initial Δ
    try
        f1 = f("fx", pbm, x + Δ * d)
        goOn = f1 > f0 + w1 * Δ * dot(g, d)
        while goOn && jter < 30
            Δ .= λ * Δ
            f1 = f("fx", pbm, x + Δ * d)
            goOn = f1 > f0 + w1 * Δ * dot(g, d)
            jter += 1
        end
    catch
        Δ = norm(g)
    end

    t_start = time()  # Start timer
    while Ng > tol && k < maxIter
        # Timeout check
        if time() - t_start > 3600
            println("Timeout: Exceeded 1 hour in trustRegionDogleg.")
            break
        end
        if any(isnan.(B)) || any(isinf.(B))
            # Initial attempt to recompute Hessian if B is invalid
            _, _, B = f("fgHx", pbm, x)
            B = Matrix(B)
        end
        try
            if any(isnan.(B)) || any(isinf.(B))
                # In case B is still invalid, fallback to identity
                B = Matrix{Float64}(I, n, n)
            else
                # Ensure B is symmetric positive definite (SPD) via factorization
                S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
                minD = minimum(diag(S.D))
                if minD < tol
                    B .= S.P'*S.U*abs.(S.D)*S.U'*S.P #B is P.D. iff D has positive entries
                end
                # Regularize if ill-conditioned
                if cond(B) * tol > 1
                    B .= 0.8 * B + 0.2 * I
                end
            end
        catch
            B = Matrix{Float64}(I, n, n)
        end
        @inbounds pU .= safeCauchyPoint(g, B)
        @inbounds pN .= - (B \ g)
        α = abs(safeCosine(d, pU))
        β = abs(safeCosine(d, pN))
        c = ARIES2(Δ, g, B, pU, pN, d, w, v, α, β)
        @. xs = x + d
        try
            # Evaluate the objective at the candidate step and compute the trust ratio
            f1 = evalgrsum!(xs, pbm, gs)
            denom = 0.5 * dot(d, B*d) + dot(g, d)  # predicted reduction (model)
            if !isfinite(denom) || abs(denom) < 1e-14
                # Avoid division by zero or non-finite model predictions
                ρ = -Inf
            else
                ρ = (f1 - f0) / denom
            end
            if ρ < c0 && Δ > ΔMin
                # Poor agreement -> shrink trust region and reject step
                Δ = max(Δ / η, ΔMin)
            else
                # Accept step: update iterate, gradient and perform BFGS update
                x .= xs
                y .= gs .- g
                # u = B * d (used in BFGS denominator)
                mul!(u, B, d)
                yd = dot(y, d)
                du = dot(d, u)
                # BFGS rank-two update: ensure denominators are safe before update
                if isfinite(yd) && isfinite(du) && abs(yd) > 1e-14 && abs(du) > 1e-14
                    B .+= (y * y') / yd - (u * u') / du
                else
                    # Skip BFGS update to avoid division by zero / numerical issues
                end
                g .= gs
                f0 = f1
                Ng = norm(g)
                # Update trust radius (returns a new Δ; assign if desired)
                Δ = updateΔ(Δ, ρ, c1, c2, η, ΔMin, ΔMax, d, c)
                acc += 1
            end
        catch
            # Evaluation failed: conservatively reduce Δ
            Δ = max(Δ / η, ΔMin)
        end
        k += 1
        G[k] = Ng
        F[k] = f0
    end
    G = G[1:k]  # Truncate the gradient norm history
    F = F[1:k]  # Truncate the function value history
    acc = acc / k  # Compute acceptance ratio
    return x, Ng, k, tol, G, F, acc
end

function trustRegionARIESBFGSAdaptive2(pb::PB, pbm::PBM, f::Function, maxIter::Int = 100000)
    G = zeros(Float64, maxIter)
    F = copy(G)
    x = copy(pb.x0)
    n = length(x)  # the length of x_k
    # Algorithm parameters: Armijo / radius thresholds / update factors
    w1, c0, c1, c2, η, λ = 1e-4, 0.1, 1/4, 1/2, 2, 0.8
    ΔMax, ΔMin = 10 * sqrt(n), 0.01 * sqrt(n)
    xs = similar(x)
    k = 0
    acc = 0 # Accepted steps counter
    tol = 1e-4
    # Evaluate initial f, g, H/B
    f0, g, B = f("fgHx", pbm, x)
    try
        B = Matrix(B)
        # Ensure B is symmetric positive definite (SPD) via factorization
        S = bunchkaufman(Symmetric(collect(B)))
        minD = minimum(diag(S.D))
        if minD < tol
            B .= S.P'*S.U*abs.(S.D)*S.U'*S.P
        end
        # Regularize if ill-conditioned
        if cond(B) * tol > 1
            B .= 0.8 * B + 0.2 * I
        end
    catch
        # Fallback to identity if Hessian is unavailable or factorization fails
        B = Matrix{Float64}(I, n, n)
    end
    Ng = norm(g)
    y = similar(g)
    pU = similar(g)
    pN = similar(g)
    gs = similar(g)
    c = "I"
    u = similar(g)
    d = similar(g)
    d .= -g
    v = similar(g)
    w = similar(g)
    α = 1/2
    β = 1/2
    jter = 0
    Δ = ΔMax
    # Line search for initial Δ
    try
        f1 = f("fx", pbm, x + Δ * d)
        goOn = f1 > f0 + w1 * Δ * dot(g, d)
        while goOn && jter < 30
            Δ .= λ * Δ
            f1 = f("fx", pbm, x + Δ * d)
            goOn = f1 > f0 + w1 * Δ * dot(g, d)
            jter += 1
        end
    catch
        Δ = norm(g)
    end

    t_start = time()  # Start timer
    while Ng > tol && k < maxIter
        # Timeout check
        if time() - t_start > 3600
            println("Timeout: Exceeded 1 hour in trustRegionDogleg.")
            break
        end
        if any(isnan.(B)) || any(isinf.(B))
            # Initial attempt to recompute Hessian if B is invalid
            _, _, B = f("fgHx", pbm, x)
            B = Matrix(B)
        end
        try
            if any(isnan.(B)) || any(isinf.(B))
                # In case B is still invalid, fallback to identity
                B = Matrix{Float64}(I, n, n)
            else
                # Ensure B is symmetric positive definite (SPD) via factorization
                S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
                minD = minimum(diag(S.D))
                if minD < tol
                    B .= S.P'*S.U*abs.(S.D)*S.U'*S.P #B is P.D. iff D has positive entries
                end
                # Regularize if ill-conditioned
                if cond(B) * tol > 1
                    B .= 0.8 * B + 0.2 * I
                end
            end
        catch
            B = Matrix{Float64}(I, n, n)
        end
        @inbounds pU .= - Ng^2 / (g⋅(B*g)) * g
        @inbounds pN .= - (B \ g)
        α = abs(dot(pN, pU))/(norm(pN)*norm(pU));
        β = 1 - α;
        c = ARIES2(Δ, g, B, pU, pN, d, w, v, α, β)
        @. xs = x + d
        try
            # Evaluate the objective at the candidate step and compute the trust ratio
            f1 = evalgrsum!(xs, pbm, gs)
            denom = 0.5 * dot(d, B*d) + dot(g, d)  # predicted reduction (model)
            if !isfinite(denom) || abs(denom) < 1e-14
                # Avoid division by zero or non-finite model predictions
                ρ = -Inf
            else
                ρ = (f1 - f0) / denom
            end
            if ρ < c0 && Δ > ΔMin
                # Poor agreement -> shrink trust region and reject step
                Δ = max(Δ / η, ΔMin)
            else
                # Accept step: update iterate, gradient and perform BFGS update
                x .= xs
                y .= gs .- g
                # u = B * d (used in BFGS denominator)
                mul!(u, B, d)
                yd = dot(y, d)
                du = dot(d, u)
                # BFGS rank-two update: ensure denominators are safe before update
                if isfinite(yd) && isfinite(du) && abs(yd) > 1e-14 && abs(du) > 1e-14
                    B .+= (y * y') / yd - (u * u') / du
                else
                    # Skip BFGS update to avoid division by zero / numerical issues
                end
                g .= gs
                f0 = f1
                Ng = norm(g)
                # Update trust radius (returns a new Δ; assign if desired)
                Δ = updateΔ(Δ, ρ, c1, c2, η, ΔMin, ΔMax, d, c)
                acc += 1
            end
        catch
            # Evaluation failed: conservatively reduce Δ
            Δ = max(Δ / η, ΔMin)
        end
        k += 1
        G[k] = Ng
        F[k] = f0
    end
    G = G[1:k]  # Truncate the gradient norm history
    F = F[1:k]  # Truncate the function value history
    acc = acc / k  # Compute acceptance ratio
    return x, Ng, k, tol, G, F, acc
end

function trustRegionARIESBFGSAdaptive3(pb::PB, pbm::PBM, f::Function, maxIter::Int = 100000)
    G = zeros(Float64, maxIter)
    F = copy(G)
    x = copy(pb.x0)
    n = length(x)  # the length of x_k
    # Algorithm parameters: Armijo / radius thresholds / update factors
    w1, c0, c1, c2, η, λ = 1e-4, 0.1, 1/4, 1/2, 2, 0.8
    ΔMax, ΔMin = 10 * sqrt(n), 0.01 * sqrt(n)
    xs = similar(x)
    k = 0
    acc = 0 # Accepted steps counter
    tol = 1e-4
    # Evaluate initial f, g, H/B
    f0, g, B = f("fgHx", pbm, x)
    try
        B = Matrix(B)
        # Ensure B is symmetric positive definite (SPD) via factorization
        S = bunchkaufman(Symmetric(collect(B)))
        minD = minimum(diag(S.D))
        if minD < tol
            B .= S.P'*S.U*abs.(S.D)*S.U'*S.P
        end
        # Regularize if ill-conditioned
        if cond(B) * tol > 1
            B .= 0.8 * B + 0.2 * I
        end
    catch
        # Fallback to identity if Hessian is unavailable or factorization fails
        B = Matrix{Float64}(I, n, n)
    end
    Ng = norm(g)
    y = similar(g)
    pU = similar(g)
    pN = similar(g)
    gs = similar(g)
    c = "I"
    u = similar(g)
    d = similar(g)
    d .= -g
    v = similar(g)
    w = similar(g)
    α = 1/2
    β = 1/2
    jter = 0
    Δ = ΔMax
    # Line search for initial Δ
    try
        f1 = f("fx", pbm, x + Δ * d)
        goOn = f1 > f0 + w1 * Δ * dot(g, d)
        while goOn && jter < 30
            Δ .= λ * Δ
            f1 = f("fx", pbm, x + Δ * d)
            goOn = f1 > f0 + w1 * Δ * dot(g, d)
            jter += 1
        end
    catch
        Δ = norm(g)
    end

    t_start = time()  # Start timer
    while Ng > tol && k < maxIter
        # Timeout check
        if time() - t_start > 3600
            println("Timeout: Exceeded 1 hour in trustRegionDogleg.")
            break
        end
        if any(isnan.(B)) || any(isinf.(B))
            # Initial attempt to recompute Hessian if B is invalid
            _, _, B = f("fgHx", pbm, x)
            B = Matrix(B)
        end
        try
            if any(isnan.(B)) || any(isinf.(B))
                # In case B is still invalid, fallback to identity
                B = Matrix{Float64}(I, n, n)
            else
                # Ensure B is symmetric positive definite (SPD) via factorization
                S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
                minD = minimum(diag(S.D))
                if minD < tol
                    B .= S.P'*S.U*abs.(S.D)*S.U'*S.P #B is P.D. iff D has positive entries
                end
                # Regularize if ill-conditioned
                if cond(B) * tol > 1
                    B .= 0.8 * B + 0.2 * I
                end
            end
        catch
            B = Matrix{Float64}(I, n, n)
        end
        @inbounds pN .= - (B \ g)
        γ1 = - Ng^2 / (g⋅(B*g))
        γ2 = Ng^2 / (g⋅pN)
        @inbounds pU .= γ1 * g
        α = max(min(γ1 * γ2, 1.0), 0.0)
        β = 1 - α;
        c = ARIES2(Δ, g, B, pU, pN, d, w, v, α, β)
        @. xs = x + d
        try
            # Evaluate the objective at the candidate step and compute the trust ratio
            f1 = evalgrsum!(xs, pbm, gs)
            denom = 0.5 * dot(d, B*d) + dot(g, d)  # predicted reduction (model)
            if !isfinite(denom) || abs(denom) < 1e-14
                # Avoid division by zero or non-finite model predictions
                ρ = -Inf
            else
                ρ = (f1 - f0) / denom
            end
            if ρ < c0 && Δ > ΔMin
                # Poor agreement -> shrink trust region and reject step
                Δ = max(Δ / η, ΔMin)
            else
                # Accept step: update iterate, gradient and perform BFGS update
                x .= xs
                y .= gs .- g
                # u = B * d (used in BFGS denominator)
                mul!(u, B, d)
                yd = dot(y, d)
                du = dot(d, u)
                # BFGS rank-two update: ensure denominators are safe before update
                if isfinite(yd) && isfinite(du) && abs(yd) > 1e-14 && abs(du) > 1e-14
                    B .+= (y * y') / yd - (u * u') / du
                else
                    # Skip BFGS update to avoid division by zero / numerical issues
                end
                g .= gs
                f0 = f1
                Ng = norm(g)
                # Update trust radius (returns a new Δ; assign if desired)
                Δ = updateΔ(Δ, ρ, c1, c2, η, ΔMin, ΔMax, d, c)
                acc += 1
            end
        catch
            # Evaluation failed: conservatively reduce Δ
            Δ = max(Δ / η, ΔMin)
        end
        k += 1
        G[k] = Ng
        F[k] = f0
    end
    G = G[1:k]  # Truncate the gradient norm history
    F = F[1:k]  # Truncate the function value history
    acc = acc / k  # Compute acceptance ratio
    return x, Ng, k, tol, G, F, acc
end

function trustRegionARIESBFGSAdaptive4(pb::PB, pbm::PBM, f::Function, maxIter::Int = 100000)
    G = zeros(Float64, maxIter)
    F = copy(G)
    x = copy(pb.x0)
    n = length(x)  # the length of x_k
    # Algorithm parameters: Armijo / radius thresholds / update factors
    w1, c0, c1, c2, η, λ = 1e-4, 0.1, 1/4, 1/2, 2, 0.8
    ΔMax, ΔMin = 10 * sqrt(n), 0.01 * sqrt(n)
    xs = similar(x)
    k = 0
    acc = 0 # Accepted steps counter
    tol = 1e-4
    # Evaluate initial f, g, H/B
    f0, g, B = f("fgHx", pbm, x)
    try
        B = Matrix(B)
        # Ensure B is symmetric positive definite (SPD) via factorization
        S = bunchkaufman(Symmetric(collect(B)))
        minD = minimum(diag(S.D))
        if minD < tol
            B .= S.P'*S.U*abs.(S.D)*S.U'*S.P
        end
        # Regularize if ill-conditioned
        if cond(B) * tol > 1
            B .= 0.8 * B + 0.2 * I
        end
    catch
        # Fallback to identity if Hessian is unavailable or factorization fails
        B = Matrix{Float64}(I, n, n)
    end
    Ng = norm(g)
    y = similar(g)
    pU = similar(g)
    pN = similar(g)
    gs = similar(g)
    c = "I"
    u = similar(g)
    d = similar(g)
    d .= -g
    v = similar(g)
    w = similar(g)
    α = 1/2
    β = 1/2
    jter = 0
    Δ = ΔMax
    # Line search for initial Δ
    try
        f1 = f("fx", pbm, x + Δ * d)
        goOn = f1 > f0 + w1 * Δ * dot(g, d)
        while goOn && jter < 30
            Δ .= λ * Δ
            f1 = f("fx", pbm, x + Δ * d)
            goOn = f1 > f0 + w1 * Δ * dot(g, d)
            jter += 1
        end
    catch
        Δ = norm(g)
    end

    t_start = time()  # Start timer
    while Ng > tol && k < maxIter
        # Timeout check
        if time() - t_start > 3600
            println("Timeout: Exceeded 1 hour in trustRegionDogleg.")
            break
        end
        if any(isnan.(B)) || any(isinf.(B))
            # Initial attempt to recompute Hessian if B is invalid
            _, _, B = f("fgHx", pbm, x)
            B = Matrix(B)
        end
        try
            if any(isnan.(B)) || any(isinf.(B))
                # In case B is still invalid, fallback to identity
                B = Matrix{Float64}(I, n, n)
            else
                # Ensure B is symmetric positive definite (SPD) via factorization
                S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
                minD = minimum(diag(S.D))
                if minD < tol
                    B .= S.P'*S.U*abs.(S.D)*S.U'*S.P #B is P.D. iff D has positive entries
                end
                # Regularize if ill-conditioned
                if cond(B) * tol > 1
                    B .= 0.8 * B + 0.2 * I
                end
            end
        catch
            B = Matrix{Float64}(I, n, n)
        end
        @inbounds pU .= - Ng^2 / (g⋅(B*g)) * g
        @inbounds pN .= - (B \ g)
        α = dot(B * g, g)/(norm(B * g)*Ng);
        β = 1 - α;
        c = ARIES2(Δ, g, B, pU, pN, d, w, v, α, β)
        @. xs = x + d
        try
            # Evaluate the objective at the candidate step and compute the trust ratio
            f1 = evalgrsum!(xs, pbm, gs)
            denom = 0.5 * dot(d, B*d) + dot(g, d)  # predicted reduction (model)
            if !isfinite(denom) || abs(denom) < 1e-14
                # Avoid division by zero or non-finite model predictions
                ρ = -Inf
            else
                ρ = (f1 - f0) / denom
            end
            if ρ < c0 && Δ > ΔMin
                # Poor agreement -> shrink trust region and reject step
                Δ = max(Δ / η, ΔMin)
            else
                # Accept step: update iterate, gradient and perform BFGS update
                x .= xs
                y .= gs .- g
                # u = B * d (used in BFGS denominator)
                mul!(u, B, d)
                yd = dot(y, d)
                du = dot(d, u)
                # BFGS rank-two update: ensure denominators are safe before update
                if isfinite(yd) && isfinite(du) && abs(yd) > 1e-14 && abs(du) > 1e-14
                    B .+= (y * y') / yd - (u * u') / du
                else
                    # Skip BFGS update to avoid division by zero / numerical issues
                end
                g .= gs
                f0 = f1
                Ng = norm(g)
                # Update trust radius (returns a new Δ; assign if desired)
                Δ = updateΔ(Δ, ρ, c1, c2, η, ΔMin, ΔMax, d, c)
                acc += 1
            end
        catch
            # Evaluation failed: conservatively reduce Δ
            Δ = max(Δ / η, ΔMin)
        end
        k += 1
        G[k] = Ng
        F[k] = f0
    end
    G = G[1:k]  # Truncate the gradient norm history
    F = F[1:k]  # Truncate the function value history
    acc = acc / k  # Compute acceptance ratio
    return x, Ng, k, tol, G, F, acc
end

function trustRegionARIESBFGSAdaptive5(pb::PB, pbm::PBM, f::Function, maxIter::Int = 100000)
    G = zeros(Float64, maxIter)
    F = copy(G)
    x = copy(pb.x0)
    n = length(x)  # the length of x_k
    # Algorithm parameters: Armijo / radius thresholds / update factors
    w1, c0, c1, c2, η, λ = 1e-4, 0.1, 1/4, 1/2, 2, 0.8
    ΔMax, ΔMin = 10 * sqrt(n), 0.01 * sqrt(n)
    xs = similar(x)
    k = 0
    acc = 0 # Accepted steps counter
    tol = 1e-4
    # Evaluate initial f, g, H/B
    f0, g, B = f("fgHx", pbm, x)
    try
        B = Matrix(B)
        # Ensure B is symmetric positive definite (SPD) via factorization
        S = bunchkaufman(Symmetric(collect(B)))
        minD = minimum(diag(S.D))
        if minD < tol
            B .= S.P'*S.U*abs.(S.D)*S.U'*S.P
        end
        # Regularize if ill-conditioned
        if cond(B) * tol > 1
            B .= 0.8 * B + 0.2 * I
        end
    catch
        # Fallback to identity if Hessian is unavailable or factorization fails
        B = Matrix{Float64}(I, n, n)
    end
    Ng = norm(g)
    y = similar(g)
    pU = similar(g)
    pN = similar(g)
    gs = similar(g)
    c = "I"
    u = similar(g)
    d = similar(g)
    d .= -g
    v = similar(g)
    w = similar(g)
    α = 1/2
    β = 1/2
    jter = 0
    Δ = ΔMax
    # Line search for initial Δ
    try
        f1 = f("fx", pbm, x + Δ * d)
        goOn = f1 > f0 + w1 * Δ * dot(g, d)
        while goOn && jter < 30
            Δ .= λ * Δ
            f1 = f("fx", pbm, x + Δ * d)
            goOn = f1 > f0 + w1 * Δ * dot(g, d)
            jter += 1
        end
    catch
        Δ = norm(g)
    end

    t_start = time()  # Start timer
    while Ng > tol && k < maxIter
        # Timeout check
        if time() - t_start > 3600
            println("Timeout: Exceeded 1 hour in trustRegionDogleg.")
            break
        end
        if any(isnan.(B)) || any(isinf.(B))
            # Initial attempt to recompute Hessian if B is invalid
            _, _, B = f("fgHx", pbm, x)
            B = Matrix(B)
        end
        try
            if any(isnan.(B)) || any(isinf.(B))
                # In case B is still invalid, fallback to identity
                B = Matrix{Float64}(I, n, n)
            else
                # Ensure B is symmetric positive definite (SPD) via factorization
                S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
                minD = minimum(diag(S.D))
                if minD < tol
                    B .= S.P'*S.U*abs.(S.D)*S.U'*S.P #B is P.D. iff D has positive entries
                end
                # Regularize if ill-conditioned
                if cond(B) * tol > 1
                    B .= 0.8 * B + 0.2 * I
                end
            end
        catch
            B = Matrix{Float64}(I, n, n)
        end
        @inbounds pU .= - Ng^2 / (g⋅(B*g)) * g
        @inbounds pN .= - (B \ g)
        α = abs(dot(pN, pU))/(norm(pN)*norm(pU));
        β = α;
        c = ARIES2(Δ, g, B, pU, pN, d, w, v, α, β)
        @. xs = x + d
        try
            # Evaluate the objective at the candidate step and compute the trust ratio
            f1 = evalgrsum!(xs, pbm, gs)
            denom = 0.5 * dot(d, B*d) + dot(g, d)  # predicted reduction (model)
            if !isfinite(denom) || abs(denom) < 1e-14
                # Avoid division by zero or non-finite model predictions
                ρ = -Inf
            else
                ρ = (f1 - f0) / denom
            end
            if ρ < c0 && Δ > ΔMin
                # Poor agreement -> shrink trust region and reject step
                Δ = max(Δ / η, ΔMin)
            else
                # Accept step: update iterate, gradient and perform BFGS update
                x .= xs
                y .= gs .- g
                # u = B * d (used in BFGS denominator)
                mul!(u, B, d)
                yd = dot(y, d)
                du = dot(d, u)
                # BFGS rank-two update: ensure denominators are safe before update
                if isfinite(yd) && isfinite(du) && abs(yd) > 1e-14 && abs(du) > 1e-14
                    B .+= (y * y') / yd - (u * u') / du
                else
                    # Skip BFGS update to avoid division by zero / numerical issues
                end
                g .= gs
                f0 = f1
                Ng = norm(g)
                # Update trust radius (returns a new Δ; assign if desired)
                Δ = updateΔ(Δ, ρ, c1, c2, η, ΔMin, ΔMax, d, c)
                acc += 1
            end
        catch
            # Evaluation failed: conservatively reduce Δ
            Δ = max(Δ / η, ΔMin)
        end
        k += 1
        G[k] = Ng
        F[k] = f0
    end
    G = G[1:k]  # Truncate the gradient norm history
    F = F[1:k]  # Truncate the function value history
    acc = acc / k  # Compute acceptance ratio
    return x, Ng, k, tol, G, F, acc
end

function trustRegionARIESBFGSAdaptive6(pb::PB, pbm::PBM, f::Function, maxIter::Int = 100000)
    G = zeros(Float64, maxIter)
    F = copy(G)
    x = copy(pb.x0)
    n = length(x)  # the length of x_k
    # Algorithm parameters: Armijo / radius thresholds / update factors
    w1, c0, c1, c2, η, λ = 1e-4, 0.1, 1/4, 1/2, 2, 0.8
    ΔMax, ΔMin = 10 * sqrt(n), 0.01 * sqrt(n)
    xs = similar(x)
    k = 0
    acc = 0 # Accepted steps counter
    tol = 1e-4
    # Evaluate initial f, g, H/B
    f0, g, B = f("fgHx", pbm, x)
    try
        B = Matrix(B)
        # Ensure B is symmetric positive definite (SPD) via factorization
        S = bunchkaufman(Symmetric(collect(B)))
        minD = minimum(diag(S.D))
        if minD < tol
            B .= S.P'*S.U*abs.(S.D)*S.U'*S.P
        end
        # Regularize if ill-conditioned
        if cond(B) * tol > 1
            B .= 0.8 * B + 0.2 * I
        end
    catch
        # Fallback to identity if Hessian is unavailable or factorization fails
        B = Matrix{Float64}(I, n, n)
    end
    Ng = norm(g)
    y = similar(g)
    pU = similar(g)
    pN = similar(g)
    gs = similar(g)
    c = "I"
    u = similar(g)
    d = similar(g)
    d .= -g
    v = similar(g)
    w = similar(g)
    α = 1/2
    β = 1/2
    jter = 0
    Δ = ΔMax
    # Line search for initial Δ
    try
        f1 = f("fx", pbm, x + Δ * d)
        goOn = f1 > f0 + w1 * Δ * dot(g, d)
        while goOn && jter < 30
            Δ .= λ * Δ
            f1 = f("fx", pbm, x + Δ * d)
            goOn = f1 > f0 + w1 * Δ * dot(g, d)
            jter += 1
        end
    catch
        Δ = norm(g)
    end

    t_start = time()  # Start timer
    while Ng > tol && k < maxIter
        # Timeout check
        if time() - t_start > 3600
            println("Timeout: Exceeded 1 hour in trustRegionDogleg.")
            break
        end
        if any(isnan.(B)) || any(isinf.(B))
            # Initial attempt to recompute Hessian if B is invalid
            _, _, B = f("fgHx", pbm, x)
            B = Matrix(B)
        end
        try
            if any(isnan.(B)) || any(isinf.(B))
                # In case B is still invalid, fallback to identity
                B = Matrix{Float64}(I, n, n)
            else
                # Ensure B is symmetric positive definite (SPD) via factorization
                S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
                minD = minimum(diag(S.D))
                if minD < tol
                    B .= S.P'*S.U*abs.(S.D)*S.U'*S.P #B is P.D. iff D has positive entries
                end
                # Regularize if ill-conditioned
                if cond(B) * tol > 1
                    B .= 0.8 * B + 0.2 * I
                end
            end
        catch
            B = Matrix{Float64}(I, n, n)
        end
        @inbounds pN .= - (B \ g)
        γ1 = - Ng^2 / (g⋅(B*g))
        γ2 = Ng^2 / (g⋅pN)
        @inbounds pU .= γ1 * g
        α = max(min(γ1 * γ2, 1.0), 0.0)
        β = α;
        c = ARIES2(Δ, g, B, pU, pN, d, w, v, α, β)
        @. xs = x + d
        try
            # Evaluate the objective at the candidate step and compute the trust ratio
            f1 = evalgrsum!(xs, pbm, gs)
            denom = 0.5 * dot(d, B*d) + dot(g, d)  # predicted reduction (model)
            if !isfinite(denom) || abs(denom) < 1e-14
                # Avoid division by zero or non-finite model predictions
                ρ = -Inf
            else
                ρ = (f1 - f0) / denom
            end
            if ρ < c0 && Δ > ΔMin
                # Poor agreement -> shrink trust region and reject step
                Δ = max(Δ / η, ΔMin)
            else
                # Accept step: update iterate, gradient and perform BFGS update
                x .= xs
                y .= gs .- g
                # u = B * d (used in BFGS denominator)
                mul!(u, B, d)
                yd = dot(y, d)
                du = dot(d, u)
                # BFGS rank-two update: ensure denominators are safe before update
                if isfinite(yd) && isfinite(du) && abs(yd) > 1e-14 && abs(du) > 1e-14
                    B .+= (y * y') / yd - (u * u') / du
                else
                    # Skip BFGS update to avoid division by zero / numerical issues
                end
                g .= gs
                f0 = f1
                Ng = norm(g)
                # Update trust radius (returns a new Δ; assign if desired)
                Δ = updateΔ(Δ, ρ, c1, c2, η, ΔMin, ΔMax, d, c)
                acc += 1
            end
        catch
            # Evaluation failed: conservatively reduce Δ
            Δ = max(Δ / η, ΔMin)
        end
        k += 1
        G[k] = Ng
        F[k] = f0
    end
    G = G[1:k]  # Truncate the gradient norm history
    F = F[1:k]  # Truncate the function value history
    acc = acc / k  # Compute acceptance ratio
    return x, Ng, k, tol, G, F, acc
end

function trustRegionARIESBFGSAdaptive7(pb::PB, pbm::PBM, f::Function, maxIter::Int = 100000)
    G = zeros(Float64, maxIter)
    F = copy(G)
    x = copy(pb.x0)
    n = length(x)  # the length of x_k
    # Algorithm parameters: Armijo / radius thresholds / update factors
    w1, c0, c1, c2, η, λ = 1e-4, 0.1, 1/4, 1/2, 2, 0.8
    ΔMax, ΔMin = 10 * sqrt(n), 0.01 * sqrt(n)
    xs = similar(x)
    k = 0
    acc = 0 # Accepted steps counter
    tol = 1e-4
    # Evaluate initial f, g, H/B
    f0, g, B = f("fgHx", pbm, x)
    try
        B = Matrix(B)
        # Ensure B is symmetric positive definite (SPD) via factorization
        S = bunchkaufman(Symmetric(collect(B)))
        minD = minimum(diag(S.D))
        if minD < tol
            B .= S.P'*S.U*abs.(S.D)*S.U'*S.P
        end
        # Regularize if ill-conditioned
        if cond(B) * tol > 1
            B .= 0.8 * B + 0.2 * I
        end
    catch
        # Fallback to identity if Hessian is unavailable or factorization fails
        B = Matrix{Float64}(I, n, n)
    end
    Ng = norm(g)
    y = similar(g)
    pU = similar(g)
    pN = similar(g)
    gs = similar(g)
    c = "I"
    u = similar(g)
    d = similar(g)
    d .= -g
    v = similar(g)
    w = similar(g)
    α = 1/2
    β = 1/2
    jter = 0
    Δ = ΔMax
    # Line search for initial Δ
    try
        f1 = f("fx", pbm, x + Δ * d)
        goOn = f1 > f0 + w1 * Δ * dot(g, d)
        while goOn && jter < 30
            Δ .= λ * Δ
            f1 = f("fx", pbm, x + Δ * d)
            goOn = f1 > f0 + w1 * Δ * dot(g, d)
            jter += 1
        end
    catch
        Δ = norm(g)
    end

    t_start = time()  # Start timer
    while Ng > tol && k < maxIter
        # Timeout check
        if time() - t_start > 3600
            println("Timeout: Exceeded 1 hour in trustRegionDogleg.")
            break
        end
        if any(isnan.(B)) || any(isinf.(B))
            # Initial attempt to recompute Hessian if B is invalid
            _, _, B = f("fgHx", pbm, x)
            B = Matrix(B)
        end
        try
            if any(isnan.(B)) || any(isinf.(B))
                # In case B is still invalid, fallback to identity
                B = Matrix{Float64}(I, n, n)
            else
                # Ensure B is symmetric positive definite (SPD) via factorization
                S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
                minD = minimum(diag(S.D))
                if minD < tol
                    B .= S.P'*S.U*abs.(S.D)*S.U'*S.P #B is P.D. iff D has positive entries
                end
                # Regularize if ill-conditioned
                if cond(B) * tol > 1
                    B .= 0.8 * B + 0.2 * I
                end
            end
        catch
            B = Matrix{Float64}(I, n, n)
        end
        @inbounds pU .= - Ng^2 / (g⋅(B*g)) * g
        @inbounds pN .= - (B \ g)
        α = dot(B * g, g)/(norm(B * g)*Ng);
        β = α;
        c = ARIES2(Δ, g, B, pU, pN, d, w, v, α, β)
        @. xs = x + d
        try
            # Evaluate the objective at the candidate step and compute the trust ratio
            f1 = evalgrsum!(xs, pbm, gs)
            denom = 0.5 * dot(d, B*d) + dot(g, d)  # predicted reduction (model)
            if !isfinite(denom) || abs(denom) < 1e-14
                # Avoid division by zero or non-finite model predictions
                ρ = -Inf
            else
                ρ = (f1 - f0) / denom
            end
            if ρ < c0 && Δ > ΔMin
                # Poor agreement -> shrink trust region and reject step
                Δ = max(Δ / η, ΔMin)
            else
                # Accept step: update iterate, gradient and perform BFGS update
                x .= xs
                y .= gs .- g
                # u = B * d (used in BFGS denominator)
                mul!(u, B, d)
                yd = dot(y, d)
                du = dot(d, u)
                # BFGS rank-two update: ensure denominators are safe before update
                if isfinite(yd) && isfinite(du) && abs(yd) > 1e-14 && abs(du) > 1e-14
                    B .+= (y * y') / yd - (u * u') / du
                else
                    # Skip BFGS update to avoid division by zero / numerical issues
                end
                g .= gs
                f0 = f1
                Ng = norm(g)
                # Update trust radius (returns a new Δ; assign if desired)
                Δ = updateΔ(Δ, ρ, c1, c2, η, ΔMin, ΔMax, d, c)
                acc += 1
            end
        catch
            # Evaluation failed: conservatively reduce Δ
            Δ = max(Δ / η, ΔMin)
        end
        k += 1
        G[k] = Ng
        F[k] = f0
    end
    G = G[1:k]  # Truncate the gradient norm history
    F = F[1:k]  # Truncate the function value history
    acc = acc / k  # Compute acceptance ratio
    return x, Ng, k, tol, G, F, acc
end

function trustRegionARIESBFGSAdaptive8(pb::PB, pbm::PBM, f::Function, maxIter::Int = 100000)
    G = zeros(Float64, maxIter)
    F = copy(G)
    x = copy(pb.x0)
    n = length(x)  # the length of x_k
    # Algorithm parameters: Armijo / radius thresholds / update factors
    w1, c0, c1, c2, η, λ = 1e-4, 0.1, 1/4, 1/2, 2, 0.8
    ΔMax, ΔMin = 10 * sqrt(n), 0.01 * sqrt(n)
    xs = similar(x)
    k = 0
    acc = 0 # Accepted steps counter
    tol = 1e-4
    # Evaluate initial f, g, H/B
    f0, g, B = f("fgHx", pbm, x)
    try
        B = Matrix(B)
        # Ensure B is symmetric positive definite (SPD) via factorization
        S = bunchkaufman(Symmetric(collect(B)))
        minD = minimum(diag(S.D))
        if minD < tol
            B .= S.P'*S.U*abs.(S.D)*S.U'*S.P
        end
        # Regularize if ill-conditioned
        if cond(B) * tol > 1
            B .= 0.8 * B + 0.2 * I
        end
    catch
        # Fallback to identity if Hessian is unavailable or factorization fails
        B = Matrix{Float64}(I, n, n)
    end
    Ng = norm(g)
    y = similar(g)
    pU = similar(g)
    pN = similar(g)
    gs = similar(g)
    c = "I"
    u = similar(g)
    d = similar(g)
    d .= -g
    v = similar(g)
    w = similar(g)
    α = 1/2
    β = 1/2
    jter = 0
    Δ = ΔMax
    # Line search for initial Δ
    try
        f1 = f("fx", pbm, x + Δ * d)
        goOn = f1 > f0 + w1 * Δ * dot(g, d)
        while goOn && jter < 30
            Δ .= λ * Δ
            f1 = f("fx", pbm, x + Δ * d)
            goOn = f1 > f0 + w1 * Δ * dot(g, d)
            jter += 1
        end
    catch
        Δ = norm(g)
    end

    t_start = time()  # Start timer
    while Ng > tol && k < maxIter
        # Timeout check
        if time() - t_start > 3600
            println("Timeout: Exceeded 1 hour in trustRegionDogleg.")
            break
        end
        if any(isnan.(B)) || any(isinf.(B))
            # Initial attempt to recompute Hessian if B is invalid
            _, _, B = f("fgHx", pbm, x)
            B = Matrix(B)
        end
        try
            if any(isnan.(B)) || any(isinf.(B))
                # In case B is still invalid, fallback to identity
                B = Matrix{Float64}(I, n, n)
            else
                # Ensure B is symmetric positive definite (SPD) via factorization
                S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
                minD = minimum(diag(S.D))
                if minD < tol
                    B .= S.P'*S.U*abs.(S.D)*S.U'*S.P #B is P.D. iff D has positive entries
                end
                # Regularize if ill-conditioned
                if cond(B) * tol > 1
                    B .= 0.8 * B + 0.2 * I
                end
            end
        catch
            B = Matrix{Float64}(I, n, n)
        end
        @inbounds pU .= - Ng^2 / (g⋅(B*g)) * g
        @inbounds pN .= - (B \ g)
        α = 1 - abs(dot(pN, pU))/(norm(pN)*norm(pU));
        β = 1 - α;
        c = ARIES2(Δ, g, B, pU, pN, d, w, v, α, β)
        @. xs = x + d
        try
            # Evaluate the objective at the candidate step and compute the trust ratio
            f1 = evalgrsum!(xs, pbm, gs)
            denom = 0.5 * dot(d, B*d) + dot(g, d)  # predicted reduction (model)
            if !isfinite(denom) || abs(denom) < 1e-14
                # Avoid division by zero or non-finite model predictions
                ρ = -Inf
            else
                ρ = (f1 - f0) / denom
            end
            if ρ < c0 && Δ > ΔMin
                # Poor agreement -> shrink trust region and reject step
                Δ = max(Δ / η, ΔMin)
            else
                # Accept step: update iterate, gradient and perform BFGS update
                x .= xs
                y .= gs .- g
                # u = B * d (used in BFGS denominator)
                mul!(u, B, d)
                yd = dot(y, d)
                du = dot(d, u)
                # BFGS rank-two update: ensure denominators are safe before update
                if isfinite(yd) && isfinite(du) && abs(yd) > 1e-14 && abs(du) > 1e-14
                    B .+= (y * y') / yd - (u * u') / du
                else
                    # Skip BFGS update to avoid division by zero / numerical issues
                end
                g .= gs
                f0 = f1
                Ng = norm(g)
                # Update trust radius (returns a new Δ; assign if desired)
                Δ = updateΔ(Δ, ρ, c1, c2, η, ΔMin, ΔMax, d, c)
                acc += 1
            end
        catch
            # Evaluation failed: conservatively reduce Δ
            Δ = max(Δ / η, ΔMin)
        end
        k += 1
        G[k] = Ng
        F[k] = f0
    end
    G = G[1:k]  # Truncate the gradient norm history
    F = F[1:k]  # Truncate the function value history
    acc = acc / k  # Compute acceptance ratio
    return x, Ng, k, tol, G, F, acc
end

function trustRegionARIESBFGSAdaptive9(pb::PB, pbm::PBM, f::Function, maxIter::Int = 100000)
    G = zeros(Float64, maxIter)
    F = copy(G)
    x = copy(pb.x0)
    n = length(x)  # the length of x_k
    # Algorithm parameters: Armijo / radius thresholds / update factors
    w1, c0, c1, c2, η, λ = 1e-4, 0.1, 1/4, 1/2, 2, 0.8
    ΔMax, ΔMin = 10 * sqrt(n), 0.01 * sqrt(n)
    xs = similar(x)
    k = 0
    acc = 0 # Accepted steps counter
    tol = 1e-4
    # Evaluate initial f, g, H/B
    f0, g, B = f("fgHx", pbm, x)
    try
        B = Matrix(B)
        # Ensure B is symmetric positive definite (SPD) via factorization
        S = bunchkaufman(Symmetric(collect(B)))
        minD = minimum(diag(S.D))
        if minD < tol
            B .= S.P'*S.U*abs.(S.D)*S.U'*S.P
        end
        # Regularize if ill-conditioned
        if cond(B) * tol > 1
            B .= 0.8 * B + 0.2 * I
        end
    catch
        # Fallback to identity if Hessian is unavailable or factorization fails
        B = Matrix{Float64}(I, n, n)
    end
    Ng = norm(g)
    y = similar(g)
    pU = similar(g)
    pN = similar(g)
    gs = similar(g)
    c = "I"
    u = similar(g)
    d = similar(g)
    d .= -g
    v = similar(g)
    w = similar(g)
    α = 1/2
    β = 1/2
    jter = 0
    Δ = ΔMax
    # Line search for initial Δ
    try
        f1 = f("fx", pbm, x + Δ * d)
        goOn = f1 > f0 + w1 * Δ * dot(g, d)
        while goOn && jter < 30
            Δ .= λ * Δ
            f1 = f("fx", pbm, x + Δ * d)
            goOn = f1 > f0 + w1 * Δ * dot(g, d)
            jter += 1
        end
    catch
        Δ = norm(g)
    end

    t_start = time()  # Start timer
    while Ng > tol && k < maxIter
        # Timeout check
        if time() - t_start > 3600
            println("Timeout: Exceeded 1 hour in trustRegionDogleg.")
            break
        end
        if any(isnan.(B)) || any(isinf.(B))
            # Initial attempt to recompute Hessian if B is invalid
            _, _, B = f("fgHx", pbm, x)
            B = Matrix(B)
        end
        try
            if any(isnan.(B)) || any(isinf.(B))
                # In case B is still invalid, fallback to identity
                B = Matrix{Float64}(I, n, n)
            else
                # Ensure B is symmetric positive definite (SPD) via factorization
                S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
                minD = minimum(diag(S.D))
                if minD < tol
                    B .= S.P'*S.U*abs.(S.D)*S.U'*S.P #B is P.D. iff D has positive entries
                end
                # Regularize if ill-conditioned
                if cond(B) * tol > 1
                    B .= 0.8 * B + 0.2 * I
                end
            end
        catch
            B = Matrix{Float64}(I, n, n)
        end
        @inbounds pN .= - (B \ g)
        γ1 = - Ng^2 / (g⋅(B*g))
        γ2 = Ng^2 / (g⋅pN)
        @inbounds pU .= γ1 * g
        α = max(min(1 - γ1 * γ2, 1.0), 0.0)
        β = α;
        c = ARIES2(Δ, g, B, pU, pN, d, w, v, α, β)
        @. xs = x + d
        try
            # Evaluate the objective at the candidate step and compute the trust ratio
            f1 = evalgrsum!(xs, pbm, gs)
            denom = 0.5 * dot(d, B*d) + dot(g, d)  # predicted reduction (model)
            if !isfinite(denom) || abs(denom) < 1e-14
                # Avoid division by zero or non-finite model predictions
                ρ = -Inf
            else
                ρ = (f1 - f0) / denom
            end
            if ρ < c0 && Δ > ΔMin
                # Poor agreement -> shrink trust region and reject step
                Δ = max(Δ / η, ΔMin)
            else
                # Accept step: update iterate, gradient and perform BFGS update
                x .= xs
                y .= gs .- g
                # u = B * d (used in BFGS denominator)
                mul!(u, B, d)
                yd = dot(y, d)
                du = dot(d, u)
                # BFGS rank-two update: ensure denominators are safe before update
                if isfinite(yd) && isfinite(du) && abs(yd) > 1e-14 && abs(du) > 1e-14
                    B .+= (y * y') / yd - (u * u') / du
                else
                    # Skip BFGS update to avoid division by zero / numerical issues
                end
                g .= gs
                f0 = f1
                Ng = norm(g)
                # Update trust radius (returns a new Δ; assign if desired)
                Δ = updateΔ(Δ, ρ, c1, c2, η, ΔMin, ΔMax, d, c)
                acc += 1
            end
        catch
            # Evaluation failed: conservatively reduce Δ
            Δ = max(Δ / η, ΔMin)
        end
        k += 1
        G[k] = Ng
        F[k] = f0
    end
    G = G[1:k]  # Truncate the gradient norm history
    F = F[1:k]  # Truncate the function value history
    acc = acc / k  # Compute acceptance ratio
    return x, Ng, k, tol, G, F, acc
end

function trustRegionARIESBFGSAdaptive10(pb::PB, pbm::PBM, f::Function, maxIter::Int = 100000)
    G = zeros(Float64, maxIter)
    F = copy(G)
    x = copy(pb.x0)
    n = length(x)  # the length of x_k
    # Algorithm parameters: Armijo / radius thresholds / update factors
    w1, c0, c1, c2, η, λ = 1e-4, 0.1, 1/4, 1/2, 2, 0.8
    ΔMax, ΔMin = 10 * sqrt(n), 0.01 * sqrt(n)
    xs = similar(x)
    k = 0
    acc = 0 # Accepted steps counter
    tol = 1e-4
    # Evaluate initial f, g, H/B
    f0, g, B = f("fgHx", pbm, x)
    try
        B = Matrix(B)
        # Ensure B is symmetric positive definite (SPD) via factorization
        S = bunchkaufman(Symmetric(collect(B)))
        minD = minimum(diag(S.D))
        if minD < tol
            B .= S.P'*S.U*abs.(S.D)*S.U'*S.P
        end
        # Regularize if ill-conditioned
        if cond(B) * tol > 1
            B .= 0.8 * B + 0.2 * I
        end
    catch
        # Fallback to identity if Hessian is unavailable or factorization fails
        B = Matrix{Float64}(I, n, n)
    end
    Ng = norm(g)
    y = similar(g)
    pU = similar(g)
    pN = similar(g)
    gs = similar(g)
    c = "I"
    u = similar(g)
    d = similar(g)
    d .= -g
    v = similar(g)
    w = similar(g)
    α = 1/2
    β = 1/2
    jter = 0
    Δ = ΔMax
    # Line search for initial Δ
    try
        f1 = f("fx", pbm, x + Δ * d)
        goOn = f1 > f0 + w1 * Δ * dot(g, d)
        while goOn && jter < 30
            Δ .= λ * Δ
            f1 = f("fx", pbm, x + Δ * d)
            goOn = f1 > f0 + w1 * Δ * dot(g, d)
            jter += 1
        end
    catch
        Δ = norm(g)
    end

    t_start = time()  # Start timer
    while Ng > tol && k < maxIter
        # Timeout check
        if time() - t_start > 3600
            println("Timeout: Exceeded 1 hour in trustRegionDogleg.")
            break
        end
        if any(isnan.(B)) || any(isinf.(B))
            # Initial attempt to recompute Hessian if B is invalid
            _, _, B = f("fgHx", pbm, x)
            B = Matrix(B)
        end
        try
            if any(isnan.(B)) || any(isinf.(B))
                # In case B is still invalid, fallback to identity
                B = Matrix{Float64}(I, n, n)
            else
                # Ensure B is symmetric positive definite (SPD) via factorization
                S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
                minD = minimum(diag(S.D))
                if minD < tol
                    B .= S.P'*S.U*abs.(S.D)*S.U'*S.P #B is P.D. iff D has positive entries
                end
                # Regularize if ill-conditioned
                if cond(B) * tol > 1
                    B .= 0.8 * B + 0.2 * I
                end
            end
        catch
            B = Matrix{Float64}(I, n, n)
        end
        @inbounds pU .= safeCauchyPoint(g, B)
        @inbounds pN .= - (B \ g)
        β = safeCosine(B * g, g)
        α = 1 - β;
        c = ARIES2(Δ, g, B, pU, pN, d, w, v, α, β)
        @. xs = x + d
        try
            # Evaluate the objective at the candidate step and compute the trust ratio
            f1 = evalgrsum!(xs, pbm, gs)
            denom = 0.5 * dot(d, B*d) + dot(g, d)  # predicted reduction (model)
            if !isfinite(denom) || abs(denom) < 1e-14
                # Avoid division by zero or non-finite model predictions
                ρ = -Inf
            else
                ρ = (f1 - f0) / denom
            end
            if ρ < c0 && Δ > ΔMin
                # Poor agreement -> shrink trust region and reject step
                Δ = max(Δ / η, ΔMin)
            else
                # Accept step: update iterate, gradient and perform BFGS update
                x .= xs
                y .= gs .- g
                # u = B * d (used in BFGS denominator)
                mul!(u, B, d)
                yd = dot(y, d)
                du = dot(d, u)
                # BFGS rank-two update: ensure denominators are safe before update
                if isfinite(yd) && isfinite(du) && abs(yd) > 1e-14 && abs(du) > 1e-14
                    B .+= (y * y') / yd - (u * u') / du
                else
                    # Skip BFGS update to avoid division by zero / numerical issues
                end
                g .= gs
                f0 = f1
                Ng = norm(g)
                # Update trust radius (returns a new Δ; assign if desired)
                Δ = updateΔ(Δ, ρ, c1, c2, η, ΔMin, ΔMax, d, c)
                acc += 1
            end
        catch
            # Evaluation failed: conservatively reduce Δ
            Δ = max(Δ / η, ΔMin)
        end
        k += 1
        G[k] = Ng
        F[k] = f0
    end
    G = G[1:k]  # Truncate the gradient norm history
    F = F[1:k]  # Truncate the function value history
    acc = acc / k  # Compute acceptance ratio
    return x, Ng, k, tol, G, F, acc
end

function trustRegionARIESBFGSAdaptive11(pb::PB, pbm::PBM, f::Function, maxIter::Int = 100000)
    G = zeros(Float64, maxIter)
    F = copy(G)
    x = copy(pb.x0)
    n = length(x)  # the length of x_k
    # Algorithm parameters: Armijo / radius thresholds / update factors
    w1, c0, c1, c2, η, λ = 1e-4, 0.1, 1/4, 1/2, 2, 0.8
    ΔMax, ΔMin = 10 * sqrt(n), 0.01 * sqrt(n)
    xs = similar(x)
    k = 0
    acc = 0 # Accepted steps counter
    tol = 1e-4
    # Evaluate initial f, g, H/B
    f0, g, B = f("fgHx", pbm, x)
    try
        B = Matrix(B)
        # Ensure B is symmetric positive definite (SPD) via factorization
        S = bunchkaufman(Symmetric(collect(B)))
        minD = minimum(diag(S.D))
        if minD < tol
            B .= S.P'*S.U*abs.(S.D)*S.U'*S.P
        end
        # Regularize if ill-conditioned
        if cond(B) * tol > 1
            B .= 0.8 * B + 0.2 * I
        end
    catch
        # Fallback to identity if Hessian is unavailable or factorization fails
        B = Matrix{Float64}(I, n, n)
    end
    Ng = norm(g)
    y = similar(g)
    pU = similar(g)
    pN = similar(g)
    gs = similar(g)
    c = "I"
    u = similar(g)
    d = similar(g)
    d .= -g
    v = similar(g)
    w = similar(g)
    α = 1/2
    β = 1/2
    jter = 0
    Δ = ΔMax
    # Line search for initial Δ
    try
        f1 = f("fx", pbm, x + Δ * d)
        goOn = f1 > f0 + w1 * Δ * dot(g, d)
        while goOn && jter < 30
            Δ .= λ * Δ
            f1 = f("fx", pbm, x + Δ * d)
            goOn = f1 > f0 + w1 * Δ * dot(g, d)
            jter += 1
        end
    catch
        Δ = norm(g)
    end

    t_start = time()  # Start timer
    while Ng > tol && k < maxIter
        # Timeout check
        if time() - t_start > 3600
            println("Timeout: Exceeded 1 hour in trustRegionDogleg.")
            break
        end
        if any(isnan.(B)) || any(isinf.(B))
            # Initial attempt to recompute Hessian if B is invalid
            _, _, B = f("fgHx", pbm, x)
            B = Matrix(B)
        end
        try
            if any(isnan.(B)) || any(isinf.(B))
                # In case B is still invalid, fallback to identity
                B = Matrix{Float64}(I, n, n)
            else
                # Ensure B is symmetric positive definite (SPD) via factorization
                S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
                minD = minimum(diag(S.D))
                if minD < tol
                    B .= S.P'*S.U*abs.(S.D)*S.U'*S.P #B is P.D. iff D has positive entries
                end
                # Regularize if ill-conditioned
                if cond(B) * tol > 1
                    B .= 0.8 * B + 0.2 * I
                end
            end
        catch
            B = Matrix{Float64}(I, n, n)
        end
        @inbounds pU .= - Ng^2 / (g⋅(B*g)) * g
        @inbounds pN .= - (B \ g)
        α = 1 - abs(dot(pN, pU))/(norm(pN)*norm(pU));
        β = α;
        c = ARIES2(Δ, g, B, pU, pN, d, w, v, α, β)
        @. xs = x + d
        try
            # Evaluate the objective at the candidate step and compute the trust ratio
            f1 = evalgrsum!(xs, pbm, gs)
            denom = 0.5 * dot(d, B*d) + dot(g, d)  # predicted reduction (model)
            if !isfinite(denom) || abs(denom) < 1e-14
                # Avoid division by zero or non-finite model predictions
                ρ = -Inf
            else
                ρ = (f1 - f0) / denom
            end
            if ρ < c0 && Δ > ΔMin
                # Poor agreement -> shrink trust region and reject step
                Δ = max(Δ / η, ΔMin)
            else
                # Accept step: update iterate, gradient and perform BFGS update
                x .= xs
                y .= gs .- g
                # u = B * d (used in BFGS denominator)
                mul!(u, B, d)
                yd = dot(y, d)
                du = dot(d, u)
                # BFGS rank-two update: ensure denominators are safe before update
                if isfinite(yd) && isfinite(du) && abs(yd) > 1e-14 && abs(du) > 1e-14
                    B .+= (y * y') / yd - (u * u') / du
                else
                    # Skip BFGS update to avoid division by zero / numerical issues
                end
                g .= gs
                f0 = f1
                Ng = norm(g)
                # Update trust radius (returns a new Δ; assign if desired)
                Δ = updateΔ(Δ, ρ, c1, c2, η, ΔMin, ΔMax, d, c)
                acc += 1
            end
        catch
            # Evaluation failed: conservatively reduce Δ
            Δ = max(Δ / η, ΔMin)
        end
        k += 1
        G[k] = Ng
        F[k] = f0
    end
    G = G[1:k]  # Truncate the gradient norm history
    F = F[1:k]  # Truncate the function value history
    acc = acc / k  # Compute acceptance ratio
    return x, Ng, k, tol, G, F, acc
end

function trustRegionARIESBFGSAdaptive12(pb::PB, pbm::PBM, f::Function, maxIter::Int = 100000)
    G = zeros(Float64, maxIter)
    F = copy(G)
    x = copy(pb.x0)
    n = length(x)  # the length of x_k
    # Algorithm parameters: Armijo / radius thresholds / update factors
    w1, c0, c1, c2, η, λ = 1e-4, 0.1, 1/4, 1/2, 2, 0.8
    ΔMax, ΔMin = 10 * sqrt(n), 0.01 * sqrt(n)
    xs = similar(x)
    k = 0
    acc = 0 # Accepted steps counter
    tol = 1e-4
    # Evaluate initial f, g, H/B
    f0, g, B = f("fgHx", pbm, x)
    try
        B = Matrix(B)
        # Ensure B is symmetric positive definite (SPD) via factorization
        S = bunchkaufman(Symmetric(collect(B)))
        minD = minimum(diag(S.D))
        if minD < tol
            B .= S.P'*S.U*abs.(S.D)*S.U'*S.P
        end
        # Regularize if ill-conditioned
        if cond(B) * tol > 1
            B .= 0.8 * B + 0.2 * I
        end
    catch
        # Fallback to identity if Hessian is unavailable or factorization fails
        B = Matrix{Float64}(I, n, n)
    end
    Ng = norm(g)
    y = similar(g)
    pU = similar(g)
    pN = similar(g)
    gs = similar(g)
    c = "I"
    u = similar(g)
    d = similar(g)
    d .= -g
    v = similar(g)
    w = similar(g)
    α = 1/2
    β = 1/2
    jter = 0
    Δ = ΔMax
    # Line search for initial Δ
    try
        f1 = f("fx", pbm, x + Δ * d)
        goOn = f1 > f0 + w1 * Δ * dot(g, d)
        while goOn && jter < 30
            Δ .= λ * Δ
            f1 = f("fx", pbm, x + Δ * d)
            goOn = f1 > f0 + w1 * Δ * dot(g, d)
            jter += 1
        end
    catch
        Δ = norm(g)
    end

    t_start = time()  # Start timer
    while Ng > tol && k < maxIter
        # Timeout check
        if time() - t_start > 3600
            println("Timeout: Exceeded 1 hour in trustRegionDogleg.")
            break
        end
        if any(isnan.(B)) || any(isinf.(B))
            # Initial attempt to recompute Hessian if B is invalid
            _, _, B = f("fgHx", pbm, x)
            B = Matrix(B)
        end
        try
            if any(isnan.(B)) || any(isinf.(B))
                # In case B is still invalid, fallback to identity
                B = Matrix{Float64}(I, n, n)
            else
                # Ensure B is symmetric positive definite (SPD) via factorization
                S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
                minD = minimum(diag(S.D))
                if minD < tol
                    B .= S.P'*S.U*abs.(S.D)*S.U'*S.P #B is P.D. iff D has positive entries
                end
                # Regularize if ill-conditioned
                if cond(B) * tol > 1
                    B .= 0.8 * B + 0.2 * I
                end
            end
        catch
            B = Matrix{Float64}(I, n, n)
        end
        @inbounds pN .= - (B \ g)
        γ1 = - Ng^2 / (g⋅(B*g))
        γ2 = Ng^2 / (g⋅pN)
        @inbounds pU .= γ1 * g
        α = max(min(1 - γ1 * γ2, 1.0), 0.0)
        β = α;
        c = ARIES2(Δ, g, B, pU, pN, d, w, v, α, β)
        @. xs = x + d
        try
            # Evaluate the objective at the candidate step and compute the trust ratio
            f1 = evalgrsum!(xs, pbm, gs)
            denom = 0.5 * dot(d, B*d) + dot(g, d)  # predicted reduction (model)
            if !isfinite(denom) || abs(denom) < 1e-14
                # Avoid division by zero or non-finite model predictions
                ρ = -Inf
            else
                ρ = (f1 - f0) / denom
            end
            if ρ < c0 && Δ > ΔMin
                # Poor agreement -> shrink trust region and reject step
                Δ = max(Δ / η, ΔMin)
            else
                # Accept step: update iterate, gradient and perform BFGS update
                x .= xs
                y .= gs .- g
                # u = B * d (used in BFGS denominator)
                mul!(u, B, d)
                yd = dot(y, d)
                du = dot(d, u)
                # BFGS rank-two update: ensure denominators are safe before update
                if isfinite(yd) && isfinite(du) && abs(yd) > 1e-14 && abs(du) > 1e-14
                    B .+= (y * y') / yd - (u * u') / du
                else
                    # Skip BFGS update to avoid division by zero / numerical issues
                end
                g .= gs
                f0 = f1
                Ng = norm(g)
                # Update trust radius (returns a new Δ; assign if desired)
                Δ = updateΔ(Δ, ρ, c1, c2, η, ΔMin, ΔMax, d, c)
                acc += 1
            end
        catch
            # Evaluation failed: conservatively reduce Δ
            Δ = max(Δ / η, ΔMin)
        end
        k += 1
        G[k] = Ng
        F[k] = f0
    end
    G = G[1:k]  # Truncate the gradient norm history
    F = F[1:k]  # Truncate the function value history
    acc = acc / k  # Compute acceptance ratio
    return x, Ng, k, tol, G, F, acc
end

function trustRegionARIESBFGSAdaptive13(pb::PB, pbm::PBM, f::Function, maxIter::Int = 100000)
    G = zeros(Float64, maxIter)
    F = copy(G)
    x = copy(pb.x0)
    n = length(x)  # the length of x_k
    # Algorithm parameters: Armijo / radius thresholds / update factors
    w1, c0, c1, c2, η, λ = 1e-4, 0.1, 1/4, 1/2, 2, 0.8
    ΔMax, ΔMin = 10 * sqrt(n), 0.01 * sqrt(n)
    xs = similar(x)
    k = 0
    acc = 0 # Accepted steps counter
    tol = 1e-4
    # Evaluate initial f, g, H/B
    f0, g, B = f("fgHx", pbm, x)
    try
        B = Matrix(B)
        # Ensure B is symmetric positive definite (SPD) via factorization
        S = bunchkaufman(Symmetric(collect(B)))
        minD = minimum(diag(S.D))
        if minD < tol
            B .= S.P'*S.U*abs.(S.D)*S.U'*S.P
        end
        # Regularize if ill-conditioned
        if cond(B) * tol > 1
            B .= 0.8 * B + 0.2 * I
        end
    catch
        # Fallback to identity if Hessian is unavailable or factorization fails
        B = Matrix{Float64}(I, n, n)
    end
    Ng = norm(g)
    y = similar(g)
    pU = similar(g)
    pN = similar(g)
    gs = similar(g)
    c = "I"
    u = similar(g)
    d = similar(g)
    d .= -g
    v = similar(g)
    w = similar(g)
    α = 1/2
    β = 1/2
    jter = 0
    Δ = ΔMax
    # Line search for initial Δ
    try
        f1 = f("fx", pbm, x + Δ * d)
        goOn = f1 > f0 + w1 * Δ * dot(g, d)
        while goOn && jter < 30
            Δ .= λ * Δ
            f1 = f("fx", pbm, x + Δ * d)
            goOn = f1 > f0 + w1 * Δ * dot(g, d)
            jter += 1
        end
    catch
        Δ = norm(g)
    end

    t_start = time()  # Start timer
    while Ng > tol && k < maxIter
        # Timeout check
        if time() - t_start > 3600
            println("Timeout: Exceeded 1 hour in trustRegionDogleg.")
            break
        end
        if any(isnan.(B)) || any(isinf.(B))
            # Initial attempt to recompute Hessian if B is invalid
            _, _, B = f("fgHx", pbm, x)
            B = Matrix(B)
        end
        try
            if any(isnan.(B)) || any(isinf.(B))
                # In case B is still invalid, fallback to identity
                B = Matrix{Float64}(I, n, n)
            else
                # Ensure B is symmetric positive definite (SPD) via factorization
                S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
                minD = minimum(diag(S.D))
                if minD < tol
                    B .= S.P'*S.U*abs.(S.D)*S.U'*S.P #B is P.D. iff D has positive entries
                end
                # Regularize if ill-conditioned
                if cond(B) * tol > 1
                    B .= 0.8 * B + 0.2 * I
                end
            end
        catch
            B = Matrix{Float64}(I, n, n)
        end
        @inbounds pU .= - Ng^2 / (g⋅(B*g)) * g
        @inbounds pN .= - (B \ g)
        α = 1 - dot(B * g, g)/(norm(B * g)*Ng);
        β = α;
        c = ARIES2(Δ, g, B, pU, pN, d, w, v, α, β)
        @. xs = x + d
        try
            # Evaluate the objective at the candidate step and compute the trust ratio
            f1 = evalgrsum!(xs, pbm, gs)
            denom = 0.5 * dot(d, B*d) + dot(g, d)  # predicted reduction (model)
            if !isfinite(denom) || abs(denom) < 1e-14
                # Avoid division by zero or non-finite model predictions
                ρ = -Inf
            else
                ρ = (f1 - f0) / denom
            end
            if ρ < c0 && Δ > ΔMin
                # Poor agreement -> shrink trust region and reject step
                Δ = max(Δ / η, ΔMin)
            else
                # Accept step: update iterate, gradient and perform BFGS update
                x .= xs
                y .= gs .- g
                # u = B * d (used in BFGS denominator)
                mul!(u, B, d)
                yd = dot(y, d)
                du = dot(d, u)
                # BFGS rank-two update: ensure denominators are safe before update
                if isfinite(yd) && isfinite(du) && abs(yd) > 1e-14 && abs(du) > 1e-14
                    B .+= (y * y') / yd - (u * u') / du
                else
                    # Skip BFGS update to avoid division by zero / numerical issues
                end
                g .= gs
                f0 = f1
                Ng = norm(g)
                # Update trust radius (returns a new Δ; assign if desired)
                Δ = updateΔ(Δ, ρ, c1, c2, η, ΔMin, ΔMax, d, c)
                acc += 1
            end
        catch
            # Evaluation failed: conservatively reduce Δ
            Δ = max(Δ / η, ΔMin)
        end
        k += 1
        G[k] = Ng
        F[k] = f0
    end
    G = G[1:k]  # Truncate the gradient norm history
    F = F[1:k]  # Truncate the function value history
    acc = acc / k  # Compute acceptance ratio
    return x, Ng, k, tol, G, F, acc
end

## Auxiliary functions for evaluating the objective and gradient of a S2MPJ problem updating in-place the gradient vector `gx` and optionally the Hessian `Hx` if provided. It's not included in the S2MPJ library but is heavily based on the `evalgrsum` function from S2MPJ, modified to support in-place updates for efficiency in the trust-region methods, as it avoids unnecessary allocations.

function evalgrsum!(x::Vector{Float64}, pbm::PBM, gx::Vector{Float64})
    glist = pbm.objgrps

    n = length(x)
    fx = 0

    # Gradient initialization
    if !isempty(pbm.objderlvl)
        if pbm.objderlvl >= 1
            gx .= spzeros(n, 1)
        else
            gx .= spzeros(n, 1)
            gx[1] = NaN
        end
    else
        gx .= spzeros(n, 1)
    end

    # Check if pbm.A is not empty
    has_A = !isempty(pbm.A)
    if has_A
        sA1, sA2 = size(pbm.A)
    end

    # Evaluate the quadratic term, if any
    if !isempty(pbm.H)
        Htimesx = pbm.H * x
        gx .+= Htimesx
        fx += 0.5 * x' * Htimesx
    end

    for iig in 1:length(glist)
        ig = glist[iig]

        # Derivative level for the group
        if !isempty(pbm.objderlvl)
            derlvl = pbm.objderlvl
        else
            derlvl = 2
        end
        nout = min(2, derlvl + 1)

        # Group scaling
        gsc = 1.0
        if isdefined(pbm, :gscale)
            if ig <= length(pbm.gscale) && abs(pbm.gscale[ig]) > 1.0e-15
                gsc = pbm.gscale[ig]
            end
        end

        # Linear term

        if isdefined(pbm, :gconst) && ig <= length(pbm.gconst)
            fin = -pbm.gconst[ig]
        else
            fin = 0
        end
        gin = zeros(Float64, n)
        if has_A && ig <= sA1
            gin[1:sA2] = pbm.A[ig, 1:sA2]
            fin += gin[1:sA2]' * x[1:sA2]
        end

        # Nonlinear group elements
        if isdefined(pbm, :grelt) && ig <= length(pbm.grelt) && !isempty(pbm.grelt[ig])
            for iiel in 1:length(pbm.grelt[ig])
                iel = pbm.grelt[ig][iiel]
                irange = pbm.elvar[iel]
                efname = pbm.elftype[iel]
                has_weights = isdefined( pbm, :grelw ) && ig <= length( pbm.grelw ) && !isempty( pbm.grelw[ig] )
                if has_weights
                    wiel        = pbm.grelw[ig][iiel]
                end
                if nout == 1
                    fiel = pbm.call( efname, x[irange], iel, 1, pbm )
                    fin += has_weights ? wiel * fiel : fiel
                else
                    fiel, giel = pbm.call( efname, x[irange], iel, 2, pbm )
                    fin       += has_weights ? wiel * fiel : fiel
                    for ir in 1:length(irange)
                        ii       = irange[ir]
                        gin[ii] += has_weights ? wiel * giel[ir] : giel[ir]
                    end
                end
            end
        end

        # Group function (non-TRIVIAL)
        if isdefined(pbm, :grftype) && ig <= length(pbm.grftype) &&
           isassigned(pbm.grftype, ig) && pbm.grftype[ig] != "TRIVIAL" && pbm.grftype[ig] != ""
            egname = pbm.grftype[ig]
            fa, grada = pbm.call(egname, fin, ig, 2, pbm)
            fx += fa / gsc
            if derlvl >= 1
                gx .+= grada[1] * gin / gsc
            else
                gx .= spzeros(n, 1)
                gx[1] = NaN
            end
        else
            # TRIVIAL case
            fx += fin / gsc
            if derlvl >= 1
                gx .+= gin / gsc
            else
                gx .= spzeros(n, 1)
                gx[1] = NaN
            end
        end
    end

    return fx
end


function evalgrsum!(x::Vector{Float64}, pbm::PBM, gx::Vector{Float64}, Hx::Matrix{Float64})
    glist = pbm.objgrps
    n = length(x)
    fx = 0

    # Initialize gx and Hx
    if !isempty(pbm.objderlvl) && pbm.objderlvl >= 1
        gx .= spzeros(n, 1)
    else
        gx .= spzeros(n, 1)
        gx[1] = NaN
    end
    if !isempty(pbm.objderlvl) && pbm.objderlvl >= 2
        Hx .= spzeros(n, n)
    else
        Hx .= spzeros(n, n)
        Hx[1,1] = NaN
    end

    # Check if pbm.A is not empty
    has_A = !isempty(pbm.A)
    if has_A
        sA1, sA2 = size(pbm.A)
    end

    # Evaluate the quadratic term, if any.
    if !isempty(pbm.H)
        Htimesx = pbm.H * x
        gx .+= Htimesx
        fx += 0.5 * x' * Htimesx
        Hx .+= pbm.H
    end

    for ig in glist
        # Derivative level
        derlvl = !isempty(pbm.objderlvl) ? pbm.objderlvl : 2
        nout = min(3, derlvl + 1)

        # Group scaling
        gsc = 1.0
        if isdefined(pbm, :gscale)
            if ig <= length(pbm.gscale) && abs(pbm.gscale[ig]) > 1.0e-15
                gsc = pbm.gscale[ig]
            end
        end

        # Linear term
        fin = isdefined(pbm, :gconst) && ig <= length(pbm.gconst) ? -pbm.gconst[ig] : 0.0
        gin = zeros(Float64, n)
        if has_A && ig <= sA1
            gin[1:sA2] = pbm.A[ig, 1:sA2]
            fin += gin[1:sA2]' * x[1:sA2]
        end

        Hin = spzeros(n, n)

        # Loop on the group's elements
        if isdefined(pbm, :grelt) && ig <= length(pbm.grelt) && !isempty(pbm.grelt[ig])
            for iiel in 1:length(pbm.grelt[ig])
                iel = pbm.grelt[ig][iiel]
                irange = pbm.elvar[iel]
                efname = pbm.elftype[iel]
                has_weights = isdefined(pbm, :grelw) && ig <= length(pbm.grelw) && !isempty(pbm.grelw[ig])
                if has_weights
                    wiel = pbm.grelw[ig][iiel]
                end

                if nout == 1
                    fiel = pbm.call( efname, x[irange], iel, 1, pbm )
                    fin += has_weights ? wiel * fiel : fiel
                elseif nout == 2
                    fiel, giel = pbm.call( efname, x[irange], iel, 2, pbm )
                    fin       += has_weights ? wiel * fiel : fiel
                    for ir in 1:length(irange)
                        ii       = irange[ir]
                        gin[ii] += has_weights ? wiel * giel[ir] : giel[ir]
                    end
                elseif nout == 3
                    fiel, giel, Hiel = pbm.call( efname, x[irange], iel, 3, pbm )
                    fin += has_weights ? wiel * fiel : fiel
                    for ir in 1:length(irange)
                        ii       = irange[ir]
                        gin[ii] += has_weights ? wiel * giel[ir] : giel[ir]
                        for jr in 1:length(irange)
                           jj          = irange[jr]
                           Hin[ii,jj] += has_weights ? wiel * Hiel[ir,jr] : Hiel[ir,jr]
                        end
                    end
                end
            end
        end

        # Non-TRIVIAL group function
        if isdefined(pbm, :grftype) && ig <= length(pbm.grftype) &&
           isassigned(pbm.grftype, ig) && pbm.grftype[ig] != "TRIVIAL" && pbm.grftype[ig] != ""
            egname = pbm.grftype[ig]
            fa, grada, Hessa = pbm.call(egname, fin, ig, 3, pbm)
            fx += fa / gsc
            if derlvl >= 1
                gx .+= grada[1] * gin / gsc
            else
                gx .= spzeros(n, 1)
                gx[1] = NaN
            end
            if derlvl >= 2
                sgin = sparse(gin)
                Hx .+= (Hessa[1] * sgin * sgin' + grada[1] * Hin) / gsc
            else
                Hx .= spzeros(n, n)
                Hx[1,1] = NaN
            end
        else
            # TRIVIAL case
            fx += fin / gsc
            if derlvl >= 1
                gx .+= gin / gsc
            else
                gx .= spzeros(n, 1)
                gx[1] = NaN
            end
            if derlvl >= 2
                Hx .+= Hin / gsc
            else
                Hx .= spzeros(n, n)
                Hx[1,1] = NaN
            end
        end
    end

    return fx
end