#Jose Miguel Saavedra Aguilar
#CIMAT Master in Applied Mathematics

using LinearAlgebra

include("s2mpjlib.jl")
include("gradienteConjugado.jl")

function zoom(a::Number,b::Number,x::AbstractVector,d::AbstractVector,f::Function,f0::Number,s::Number,c1::Number,c2::Number, pbm::PBM)
    looking = true
    kter = 0
    maxkter = 20    # Maximum number of "Zooms" in the interval
    α = (a+b)/2 # Initialize α
    while looking && kter < maxkter
        c = (a+b)/2 # Midpoint of the interval
        fa = f("fx", pbm, x + a*d) # Value at the left side of interval
        fb = f("fx", pbm, x + b*d) # Value at the right side of the interval
        fc = f("fx", pbm, x + c*d) # Value at the middle of the interval
        if (fa+fb-2*fc) > 1e-6  # Linearity if fa + fb - 2fc = 0
            α = (fa*(a+3*b)+fb*(3*a+b)-4*fc*(a+b))/(4*(fa+fb-2*fc)) # α_min for quadratic interpolation on a, b, (a+b)/2
        else
            α = 0.8*b + 0.2*a   # a + .8(b-a)
        end
        xt = x + α*d    # Trial step
        f1, g1 = f("fgx", pbm, xt)
        if f1 > f0 + α*c1*s # Armijo's condition
            b = α   # Reduce right side of interval
        else
            s1 = d'*g1  # Φ'(α)
            if abs(s1) <= -c2*s # Second strong Wolfe condition
                looking = false
            elseif (b-a)*s1 >= 0
                b = a
                a = α
            else
                a = α
            end
        end
        kter += 1
    end
    xt = x + α*d
    f1, g1 = f("fgx", pbm, xt)
    s1 = d'*g1
    return α, f1, g1, s1
end

function updateΔ(Δ, ρ, c1, c2, η, ΔMin, ΔMax, d, c)
    if (ρ < c1)
        # Reduce the trust radius
        Δ = max(Δ / η , ΔMin)
    elseif (ρ > c2 && c != "N")
        Δ = min(η * Δ, ΔMax) # Increase Δ
    #else
        #Δ = norm(d)
    end
end


function dogleg(Δ::Number,g::AbstractVector,B::AbstractMatrix, pU::AbstractVector, pN::AbstractVector, d::AbstractVector, tol::Number=1e-6,dim=length(g))
    # Dogleg method for the trust region problem
    @inbounds pU .= - (g⋅g) / (g⋅(B*g)) * g
    NpU = norm(pU)
    if NpU > Δ
        @inbounds d .= Δ*pU/NpU
        c = "C"
    else
        @inbounds pN .= - (B \ g)
        if norm(pN) <= Δ
            @inbounds d .= pN
            c = "N"
        else
            w = pN-pU
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

function ARIESDogleg(Δ::Number,g::AbstractVector,B::AbstractMatrix, pU::AbstractVector, pN::AbstractVector, d::AbstractVector, tol::Number=1e-6,dim=length(g), α::Number = 1/2, β::Number = 1/2)
    # Dogleg method for the trust region problem
    @inbounds pU .= - (g⋅g) / (g⋅(B*g)) * g
    v = α * pU
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
            w = β * pN + (1 - α - β) * pU
            p1 = w⋅w
            p2 = w⋅v
            p3 = (Δ + Nv) * (Δ - Nv)
            t = (-p2 + sqrt(p2^2 + p1*p3)) / p1
            @inbounds d .= v + t * w
            c = "D1"
        else
            # Case 3. direction is on the dogleg path
            w = pN-pU
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

function ARIESDogleg2(Δ::Number,g::AbstractVector,B::AbstractMatrix, pU::AbstractVector, pN::AbstractVector, d::AbstractVector, tol::Number=1e-6,dim=length(g), α::Number = 1/2, β::Number = 1/2)
    # Dogleg method for the trust region problem
    v = α * pU
    NpU = norm(pU)
    Nv = α * NpU
    if Nv > Δ
        # Case 1: direction is scalled gradient
        @inbounds d .= Δ*pU/NpU
        c = "C"
    else
        nPN = norm(pN)
        if nPN <= Δ
            # Case 4: direction is Newton's step
            @inbounds d .= pN
            c = "N"
        elseif norm(β * (pN - pU) + pU) > Δ
            # Case 2. direction is on the middle path
            w = β * pN + (1 - α - β) * pU
            p1 = w⋅w
            p2 = w⋅v
            p3 = (Δ + Nv) * (Δ - Nv)
            t = (-p2 + sqrt(p2^2 + p1*p3)) / p1
            @inbounds d .= v + t * w
            c = "D1"
        else
            # Case 3. direction is on the dogleg path
            w = pN-pU
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

function trustRegionDogleg(pb::PB, pbm::PBM, f::Function)
    maxIter = 100000
    G = zeros(Float64, maxIter)
    x = copy(pb.x0)
    n = length(x)  # the length of x_k
    c0, c1, c2, η = 0.1, 1/4, 1/2, 2
    ΔMax, ΔMin = 3, 0.01
    xs = similar(x)
    k = 0
    tol = 1e-4
    f0, g, B = f("fgHx", pbm, x)
    B = Matrix(B)
    if cond(B) * tol > 1
        B = 0.8 * B + 0.2 * I
    end
    Ng = norm(g)
    y = similar(g)
    pU = similar(g)
    pN = similar(g)
    gs = similar(g)
    S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
    minD = minimum(diag(S.D))
    if minD < tol
        B .= S.P'*S.U*abs.(S.D)*S.U'*S.P #B is P.D. iff D has positive entries
    end
    c = "I"
    u = similar(g)
    d = -g
    s = dot(d, g)
    jter = 0
    goOn = true
    α0, α1 = 0.0, 1.0
    Δ = 0.01
    p0 = f0
    # Line search for initial Δ
    try
        while goOn && jter < 30
            @. xs = x + α1 * d
            f1, g1 = f("fgx", pbm, xs)
            if f1 > min(f0 + α1*c1*s, p0)
                Δ, f1, g1, s1 = zoom(α0, α1, x, d, f, f0, s, c1, c2, pbm)
                goOn = false
            else
                s1 = dot(d, g1)
                if abs(s1) <= -c2*s
                    Δ = α1
                    goOn = false
                elseif s1 >= 0
                    Δ, f1, g1, s1 = zoom(α1, 2.0, x, d, f, f0, s, c1, c2, pbm)
                    goOn = false
                else
                    α1 = (α1 + 2.0)/2
                end
            end
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
        c = dogleg(Δ, g, B, pU, pN, d, tol, n)
        
        if any(isnan.(d)) || cond(B) * tol > 1
            _, _, B = f("fgHx", pbm, x) # Recompute B
            B = Matrix(B)
            if cond(B) * tol > 1
                B .= 0.8 * B + 0.2 * I
            end
            S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
            minD = minimum(diag(S.D))
            if minD < tol
                B .= S.P'*S.U*abs.(S.D)*S.U'*S.P #B is P.D. iff D has positive entries
            end
            d .= -g
        end
        nd = norm(d)
        @. xs = x + d
        try
            f1 = evalgrsum!(xs, pbm, gs)
            denom = 0.5 * dot(d, B*d) + dot(g, d)
            ρ = (f1 - f0) / denom
            if ρ < c0 && Δ > ΔMin
                Δ = max(Δ / η, ΔMin)
            else
                x .= xs
                y .= gs .- g
                mul!(u, B, d)
                yd = dot(y, d)
                du = dot(d, u)
                B .+= (y * y') / yd - (u * u') / du
                g .= gs
                f0 = f1
                Ng = norm(g)
                updateΔ(Δ, ρ, c1, c2, η, ΔMin, ΔMax, d, c)
            end
        catch
            Δ = max(Δ / η, ΔMin)
        end
        k += 1
        G[k] = Ng
    end
    G = G[1:k]
    return x, Ng, k, tol, G
end

function trustRegionARIES(pb::PB, pbm::PBM, f::Function, α::Number=1/2, β::Number=1/2)
    maxIter = 100000
    G = zeros(Float64, maxIter)
    x = copy(pb.x0)
    n = length(x)  # the length of x_k
    c0, c1, c2, η = 0.1, 1/4, 1/2, 2
    ΔMax, ΔMin = 8.0*n, 10.0^(-n)
    xs = similar(x)
    k = 0
    tol = 1e-4
    f0, g, B = f("fgHx", pbm, x)
    B = Matrix(B)
    if cond(B) * tol > 1
        B = 0.8 * B + 0.2 * I
    end
    Ng = norm(g)
    y = similar(g)
    pU = similar(g)
    pN = similar(g)
    gs = similar(g)
    S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
    minD = minimum(diag(S.D))
    if minD < tol
        B .= S.P'*S.U*abs.(S.D)*S.U'*S.P #B is P.D. iff D has positive entries
    end
    c = "I"
    u = similar(g)
    d = -g
    s = dot(d, g)
    jter = 0
    goOn = true
    α0, α1 = 0.0, 1.0
    Δ = 0.01
    p0 = f0
    # Line search for initial Δ
    try
        while goOn && jter < 30
            @. xs = x + α1 * d
            f1, g1 = f("fgx", pbm, xs)
            if f1 > min(f0 + α1*c1*s, p0)
                Δ, f1, g1, s1 = zoom(α0, α1, x, d, f, f0, s, c1, c2, pbm)
                goOn = false
            else
                s1 = dot(d, g1)
                if abs(s1) <= -c2*s
                    Δ = α1
                    goOn = false
                elseif s1 >= 0
                    Δ, f1, g1, s1 = zoom(α1, 2.0, x, d, f, f0, s, c1, c2, pbm)
                    goOn = false
                else
                    α1 = (α1 + 2.0)/2
                end
            end
            jter += 1
        end
    catch
        Δ = norm(g)
        println("Initial line search failed, setting Δ = ||g||")
    end
    
    t_start = time()  # Start timer
    while Ng > tol && k < maxIter
        # Timeout check
        if time() - t_start > 3600
            println("Timeout: Exceeded 1 hour in trustRegionARIES.")
            break
        end
        c = ARIESDogleg(Δ, g, B, pU, pN, d, tol, n, α, β) # ARIES dogleg approximation of the quadratic model
        nd = norm(d)
        if any(isnan.(d)) || cond(B) * tol > 1
            _, _, B = f("fgHx", pbm, x) # Recompute B
            B = Matrix(B)
            if cond(B) * tol > 1
                B .= 0.8 * B + 0.2 * I
            end
            S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
            minD = minimum(diag(S.D))
            if minD < tol
                B .= S.P'*S.U*abs.(S.D)*S.U'*S.P #B is P.D. iff D has positive entries
            end
            d .= -g
            nd = norm(d)
        end
        @. xs = x + d
        try
            f1 = evalgrsum!(xs, pbm, gs)
            denom = 0.5 * dot(d, B*d) + dot(g, d)
            ρ = (f1 - f0) / denom
            if ρ < c0 && Δ > ΔMin
                Δ = max(Δ / η, ΔMin)
            else
                x .= xs
                y .= gs .- g
                mul!(u, B, d)
                yd = dot(y, d)
                du = dot(d, u)
                B .+= (y * y') / yd - (u * u') / du
                g .= gs
                f0 = f1
                Ng = norm(g)
                updateΔ(Δ, ρ, c1, c2, η, ΔMin, ΔMax, d, c)
            end
        catch
            Δ = max(Δ / η, ΔMin)
        end
        k += 1
        G[k] = Ng
    end
    G = G[1:k]
    return x, Ng, k, tol, G
end

function trustRegionARIES11(pb::PB, pbm::PBM, f::Function, α::Number=1/4, β::Number=1/4)
    maxIter = 100000
    G = zeros(Float64, maxIter)
    x = copy(pb.x0)
    n = length(x)  # the length of x_k
    c0, c1, c2, η = 0.1, 1/4, 1/2, 2
    ΔMax, ΔMin = 8.0*n, 10.0^(-n)
    xs = similar(x)
    k = 0
    tol = 1e-4
    f0, g, B = f("fgHx", pbm, x)
    B = Matrix(B)
    if cond(B) * tol > 1
        B = 0.8 * B + 0.2 * I
    end
    Ng = norm(g)
    y = similar(g)
    pU = similar(g)
    pN = similar(g)
    gs = similar(g)
    S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
    minD = minimum(diag(S.D))
    if minD < tol
        B .= S.P'*S.U*abs.(S.D)*S.U'*S.P #B is P.D. iff D has positive entries
    end
    c = "I"
    u = similar(g)
    d = -g
    s = dot(d, g)
    jter = 0
    goOn = true
    α0, α1 = 0.0, 1.0
    Δ = 0.01
    p0 = f0
    # Line search for initial Δ
    try
        while goOn && jter < 30
            @. xs = x + α1 * d
            f1, g1 = f("fgx", pbm, xs)
            if f1 > min(f0 + α1*c1*s, p0)
                Δ, f1, g1, s1 = zoom(α0, α1, x, d, f, f0, s, c1, c2, pbm)
                goOn = false
            else
                s1 = dot(d, g1)
                if abs(s1) <= -c2*s
                    Δ = α1
                    goOn = false
                elseif s1 >= 0
                    Δ, f1, g1, s1 = zoom(α1, 2.0, x, d, f, f0, s, c1, c2, pbm)
                    goOn = false
                else
                    α1 = (α1 + 2.0)/2
                end
            end
            jter += 1
        end
    catch
        Δ = norm(g)
        println("Initial line search failed, setting Δ = ||g||")
    end
    
    t_start = time()  # Start timer
    while Ng > tol && k < maxIter
        # Timeout check
        if time() - t_start > 3600
            println("Timeout: Exceeded 1 hour in trustRegionARIES11.")
            break
        end
        c = ARIESDogleg(Δ, g, B, pU, pN, d, tol, n, α, β) # ARIES dogleg approximation of the quadratic model
        nd = norm(d)
        if any(isnan.(d)) || cond(B) * tol > 1
            _, _, B = f("fgHx", pbm, x) # Recompute B
            B = Matrix(B)
            if cond(B) * tol > 1
                B .= 0.8 * B + 0.2 * I
            end
            S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
            minD = minimum(diag(S.D))
            if minD < tol
                B .= S.P'*S.U*abs.(S.D)*S.U'*S.P #B is P.D. iff D has positive entries
            end
            d .= -g
            nd = norm(d)
        end
        @. xs = x + d
        try
            f1 = evalgrsum!(xs, pbm, gs)
            denom = 0.5 * dot(d, B*d) + dot(g, d)
            ρ = (f1 - f0) / denom
            if ρ < c0 && Δ > ΔMin
                Δ = max(Δ / η, ΔMin)
            else
                x .= xs
                y .= gs .- g
                mul!(u, B, d)
                yd = dot(y, d)
                du = dot(d, u)
                B .+= (y * y') / yd - (u * u') / du
                g .= gs
                f0 = f1
                Ng = norm(g)
                updateΔ(Δ, ρ, c1, c2, η, ΔMin, ΔMax, d, c)
            end
        catch
            Δ = max(Δ / η, ΔMin)
        end
        k += 1
        G[k] = Ng
    end
    G = G[1:k]
    return x, Ng, k, tol, G
end

function trustRegionARIES12(pb::PB, pbm::PBM, f::Function, α::Number=1/4, β::Number=1/2)
    maxIter = 100000
    G = zeros(Float64, maxIter)
    x = copy(pb.x0)
    n = length(x)  # the length of x_k
    c0, c1, c2, η = 0.1, 1/4, 1/2, 2
    ΔMax, ΔMin = 8.0*n, 10.0^(-n)
    xs = similar(x)
    k = 0
    tol = 1e-4
    f0, g, B = f("fgHx", pbm, x)
    B = Matrix(B)
    if cond(B) * tol > 1
        B = 0.8 * B + 0.2 * I
    end
    Ng = norm(g)
    y = similar(g)
    pU = similar(g)
    pN = similar(g)
    gs = similar(g)
    S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
    minD = minimum(diag(S.D))
    if minD < tol
        B .= S.P'*S.U*abs.(S.D)*S.U'*S.P #B is P.D. iff D has positive entries
    end
    c = "I"
    u = similar(g)
    d = -g
    s = dot(d, g)
    jter = 0
    goOn = true
    α0, α1 = 0.0, 1.0
    Δ = 0.01
    p0 = f0
    # Line search for initial Δ
    try
        while goOn && jter < 30
            @. xs = x + α1 * d
            f1, g1 = f("fgx", pbm, xs)
            if f1 > min(f0 + α1*c1*s, p0)
                Δ, f1, g1, s1 = zoom(α0, α1, x, d, f, f0, s, c1, c2, pbm)
                goOn = false
            else
                s1 = dot(d, g1)
                if abs(s1) <= -c2*s
                    Δ = α1
                    goOn = false
                elseif s1 >= 0
                    Δ, f1, g1, s1 = zoom(α1, 2.0, x, d, f, f0, s, c1, c2, pbm)
                    goOn = false
                else
                    α1 = (α1 + 2.0)/2
                end
            end
            jter += 1
        end
    catch
        Δ = norm(g)
        println("Initial line search failed, setting Δ = ||g||")
    end
    
    t_start = time()  # Start timer
    while Ng > tol && k < maxIter
        # Timeout check
        if time() - t_start > 3600
            println("Timeout: Exceeded 1 hour in trustRegionARIES12.")
            break
        end
        c = ARIESDogleg(Δ, g, B, pU, pN, d, tol, n, α, β) # ARIES dogleg approximation of the quadratic model
        nd = norm(d)
        if any(isnan.(d)) || cond(B) * tol > 1
            _, _, B = f("fgHx", pbm, x) # Recompute B
            B = Matrix(B)
            if cond(B) * tol > 1
                B .= 0.8 * B + 0.2 * I
            end
            S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
            minD = minimum(diag(S.D))
            if minD < tol
                B .= S.P'*S.U*abs.(S.D)*S.U'*S.P #B is P.D. iff D has positive entries
            end
            d .= -g
            nd = norm(d)
        end
        @. xs = x + d
        try
            f1 = evalgrsum!(xs, pbm, gs)
            denom = 0.5 * dot(d, B*d) + dot(g, d)
            ρ = (f1 - f0) / denom
            if ρ < c0 && Δ > ΔMin
                Δ = max(Δ / η, ΔMin)
            else
                x .= xs
                y .= gs .- g
                mul!(u, B, d)
                yd = dot(y, d)
                du = dot(d, u)
                B .+= (y * y') / yd - (u * u') / du
                g .= gs
                f0 = f1
                Ng = norm(g)
                updateΔ(Δ, ρ, c1, c2, η, ΔMin, ΔMax, d, c)
            end
        catch
            Δ = max(Δ / η, ΔMin)
        end
        k += 1
        G[k] = Ng
    end
    G = G[1:k]
    return x, Ng, k, tol, G
end

function trustRegionARIES13(pb::PB, pbm::PBM, f::Function, α::Number=1/4, β::Number=3/4)
    maxIter = 100000
    G = zeros(Float64, maxIter)
    x = copy(pb.x0)
    n = length(x)  # the length of x_k
    c0, c1, c2, η = 0.1, 1/4, 1/2, 2
    ΔMax, ΔMin = 8.0*n, 10.0^(-n)
    xs = similar(x)
    k = 0
    tol = 1e-4
    f0, g, B = f("fgHx", pbm, x)
    B = Matrix(B)
    if cond(B) * tol > 1
        B = 0.8 * B + 0.2 * I
    end
    Ng = norm(g)
    y = similar(g)
    pU = similar(g)
    pN = similar(g)
    gs = similar(g)
    S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
    minD = minimum(diag(S.D))
    if minD < tol
        B .= S.P'*S.U*abs.(S.D)*S.U'*S.P #B is P.D. iff D has positive entries
    end
    c = "I"
    u = similar(g)
    d = -g
    s = dot(d, g)
    jter = 0
    goOn = true
    α0, α1 = 0.0, 1.0
    Δ = 0.01
    p0 = f0
    # Line search for initial Δ
    try
        while goOn && jter < 30
            @. xs = x + α1 * d
            f1, g1 = f("fgx", pbm, xs)
            if f1 > min(f0 + α1*c1*s, p0)
                Δ, f1, g1, s1 = zoom(α0, α1, x, d, f, f0, s, c1, c2, pbm)
                goOn = false
            else
                s1 = dot(d, g1)
                if abs(s1) <= -c2*s
                    Δ = α1
                    goOn = false
                elseif s1 >= 0
                    Δ, f1, g1, s1 = zoom(α1, 2.0, x, d, f, f0, s, c1, c2, pbm)
                    goOn = false
                else
                    α1 = (α1 + 2.0)/2
                end
            end
            jter += 1
        end
    catch
        Δ = norm(g)
        println("Initial line search failed, setting Δ = ||g||")
    end
    
    t_start = time()  # Start timer
    while Ng > tol && k < maxIter
        # Timeout check
        if time() - t_start > 3600
            println("Timeout: Exceeded 1 hour in trustRegionARIES13.")
            break
        end
        c = ARIESDogleg(Δ, g, B, pU, pN, d, tol, n, α, β) # ARIES dogleg approximation of the quadratic model
        nd = norm(d)
        if any(isnan.(d)) || cond(B) * tol > 1
            _, _, B = f("fgHx", pbm, x) # Recompute B
            B = Matrix(B)
            if cond(B) * tol > 1
                B .= 0.8 * B + 0.2 * I
            end
            S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
            minD = minimum(diag(S.D))
            if minD < tol
                B .= S.P'*S.U*abs.(S.D)*S.U'*S.P #B is P.D. iff D has positive entries
            end
            d .= -g
            nd = norm(d)
        end
        @. xs = x + d
        try
            f1 = evalgrsum!(xs, pbm, gs)
            denom = 0.5 * dot(d, B*d) + dot(g, d)
            ρ = (f1 - f0) / denom
            if ρ < c0 && Δ > ΔMin
                Δ = max(Δ / η, ΔMin)
            else
                x .= xs
                y .= gs .- g
                mul!(u, B, d)
                yd = dot(y, d)
                du = dot(d, u)
                B .+= (y * y') / yd - (u * u') / du
                g .= gs
                f0 = f1
                Ng = norm(g)
                updateΔ(Δ, ρ, c1, c2, η, ΔMin, ΔMax, d, c)
            end
        catch
            Δ = max(Δ / η, ΔMin)
        end
        k += 1
        G[k] = Ng
    end
    G = G[1:k]
    return x, Ng, k, tol, G
end

function trustRegionARIES21(pb::PB, pbm::PBM, f::Function, α::Number=1/2, β::Number=1/4)
    maxIter = 100000
    G = zeros(Float64, maxIter)
    x = copy(pb.x0)
    n = length(x)  # the length of x_k
    c0, c1, c2, η = 0.1, 1/4, 1/2, 2
    ΔMax, ΔMin = 8.0*n, 10.0^(-n)
    xs = similar(x)
    k = 0
    tol = 1e-4
    f0, g, B = f("fgHx", pbm, x)
    B = Matrix(B)
    if cond(B) * tol > 1
        B = 0.8 * B + 0.2 * I
    end
    Ng = norm(g)
    y = similar(g)
    pU = similar(g)
    pN = similar(g)
    gs = similar(g)
    S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
    minD = minimum(diag(S.D))
    if minD < tol
        B .= S.P'*S.U*abs.(S.D)*S.U'*S.P #B is P.D. iff D has positive entries
    end
    c = "I"
    u = similar(g)
    d = -g
    s = dot(d, g)
    jter = 0
    goOn = true
    α0, α1 = 0.0, 1.0
    Δ = 0.01
    p0 = f0
    # Line search for initial Δ
    try
        while goOn && jter < 30
            @. xs = x + α1 * d
            f1, g1 = f("fgx", pbm, xs)
            if f1 > min(f0 + α1*c1*s, p0)
                Δ, f1, g1, s1 = zoom(α0, α1, x, d, f, f0, s, c1, c2, pbm)
                goOn = false
            else
                s1 = dot(d, g1)
                if abs(s1) <= -c2*s
                    Δ = α1
                    goOn = false
                elseif s1 >= 0
                    Δ, f1, g1, s1 = zoom(α1, 2.0, x, d, f, f0, s, c1, c2, pbm)
                    goOn = false
                else
                    α1 = (α1 + 2.0)/2
                end
            end
            jter += 1
        end
    catch
        Δ = norm(g)
        println("Initial line search failed, setting Δ = ||g||")
    end
    
    t_start = time()  # Start timer
    while Ng > tol && k < maxIter
        # Timeout check
        if time() - t_start > 3600
            println("Timeout: Exceeded 1 hour in trustRegionARIES21.")
            break
        end
        c = ARIESDogleg(Δ, g, B, pU, pN, d, tol, n, α, β) # ARIES dogleg approximation of the quadratic model
        nd = norm(d)
        if any(isnan.(d)) || cond(B) * tol > 1
            _, _, B = f("fgHx", pbm, x) # Recompute B
            B = Matrix(B)
            if cond(B) * tol > 1
                B .= 0.8 * B + 0.2 * I
            end
            S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
            minD = minimum(diag(S.D))
            if minD < tol
                B .= S.P'*S.U*abs.(S.D)*S.U'*S.P #B is P.D. iff D has positive entries
            end
            d .= -g
            nd = norm(d)
        end
        @. xs = x + d
        try
            f1 = evalgrsum!(xs, pbm, gs)
            denom = 0.5 * dot(d, B*d) + dot(g, d)
            ρ = (f1 - f0) / denom
            if ρ < c0 && Δ > ΔMin
                Δ = max(Δ / η, ΔMin)
            else
                x .= xs
                y .= gs .- g
                mul!(u, B, d)
                yd = dot(y, d)
                du = dot(d, u)
                B .+= (y * y') / yd - (u * u') / du
                g .= gs
                f0 = f1
                Ng = norm(g)
                updateΔ(Δ, ρ, c1, c2, η, ΔMin, ΔMax, d, c)
            end
        catch
            Δ = max(Δ / η, ΔMin)
        end
        k += 1
        G[k] = Ng
    end
    G = G[1:k]
    return x, Ng, k, tol, G
end

function trustRegionARIES22(pb::PB, pbm::PBM, f::Function, α::Number=1/2, β::Number=1/2)
    maxIter = 100000
    G = zeros(Float64, maxIter)
    x = copy(pb.x0)
    n = length(x)  # the length of x_k
    c0, c1, c2, η = 0.1, 1/4, 1/2, 2
    ΔMax, ΔMin = 8.0*n, 10.0^(-n)
    xs = similar(x)
    k = 0
    tol = 1e-4
    f0, g, B = f("fgHx", pbm, x)
    B = Matrix(B)
    if cond(B) * tol > 1
        B = 0.8 * B + 0.2 * I
    end
    Ng = norm(g)
    y = similar(g)
    pU = similar(g)
    pN = similar(g)
    gs = similar(g)
    S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
    minD = minimum(diag(S.D))
    if minD < tol
        B .= S.P'*S.U*abs.(S.D)*S.U'*S.P #B is P.D. iff D has positive entries
    end
    c = "I"
    u = similar(g)
    d = -g
    s = dot(d, g)
    jter = 0
    goOn = true
    α0, α1 = 0.0, 1.0
    Δ = 0.01
    p0 = f0
    # Line search for initial Δ
    try
        while goOn && jter < 30
            @. xs = x + α1 * d
            f1, g1 = f("fgx", pbm, xs)
            if f1 > min(f0 + α1*c1*s, p0)
                Δ, f1, g1, s1 = zoom(α0, α1, x, d, f, f0, s, c1, c2, pbm)
                goOn = false
            else
                s1 = dot(d, g1)
                if abs(s1) <= -c2*s
                    Δ = α1
                    goOn = false
                elseif s1 >= 0
                    Δ, f1, g1, s1 = zoom(α1, 2.0, x, d, f, f0, s, c1, c2, pbm)
                    goOn = false
                else
                    α1 = (α1 + 2.0)/2
                end
            end
            jter += 1
        end
    catch
        Δ = norm(g)
        println("Initial line search failed, setting Δ = ||g||")
    end
    
    t_start = time()  # Start timer
    while Ng > tol && k < maxIter
        # Timeout check
        if time() - t_start > 3600
            println("Timeout: Exceeded 1 hour in trustRegionARIES22.")
            break
        end
        c = ARIESDogleg(Δ, g, B, pU, pN, d, tol, n, α, β) # ARIES dogleg approximation of the quadratic model
        nd = norm(d)
        if any(isnan.(d)) || cond(B) * tol > 1
            _, _, B = f("fgHx", pbm, x) # Recompute B
            B = Matrix(B)
            if cond(B) * tol > 1
                B .= 0.8 * B + 0.2 * I
            end
            S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
            minD = minimum(diag(S.D))
            if minD < tol
                B .= S.P'*S.U*abs.(S.D)*S.U'*S.P #B is P.D. iff D has positive entries
            end
            d .= -g
            nd = norm(d)
        end
        @. xs = x + d
        try
            f1 = evalgrsum!(xs, pbm, gs)
            denom = 0.5 * dot(d, B*d) + dot(g, d)
            ρ = (f1 - f0) / denom
            if ρ < c0 && Δ > ΔMin
                Δ = max(Δ / η, ΔMin)
            else
                x .= xs
                y .= gs .- g
                mul!(u, B, d)
                yd = dot(y, d)
                du = dot(d, u)
                B .+= (y * y') / yd - (u * u') / du
                g .= gs
                f0 = f1
                Ng = norm(g)
                updateΔ(Δ, ρ, c1, c2, η, ΔMin, ΔMax, d, c)
            end
        catch
            Δ = max(Δ / η, ΔMin)
        end
        k += 1
        G[k] = Ng
    end
    G = G[1:k]
    return x, Ng, k, tol, G
end

function trustRegionARIES23(pb::PB, pbm::PBM, f::Function, α::Number=1/2, β::Number=3/4)
    maxIter = 100000
    G = zeros(Float64, maxIter)
    x = copy(pb.x0)
    n = length(x)  # the length of x_k
    c0, c1, c2, η = 0.1, 1/4, 1/2, 2
    ΔMax, ΔMin = 8.0*n, 10.0^(-n)
    xs = similar(x)
    k = 0
    tol = 1e-4
    f0, g, B = f("fgHx", pbm, x)
    B = Matrix(B)
    if cond(B) * tol > 1
        B = 0.8 * B + 0.2 * I
    end
    Ng = norm(g)
    y = similar(g)
    pU = similar(g)
    pN = similar(g)
    gs = similar(g)
    S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
    minD = minimum(diag(S.D))
    if minD < tol
        B .= S.P'*S.U*abs.(S.D)*S.U'*S.P #B is P.D. iff D has positive entries
    end
    c = "I"
    u = similar(g)
    d = -g
    s = dot(d, g)
    jter = 0
    goOn = true
    α0, α1 = 0.0, 1.0
    Δ = 0.01
    p0 = f0
    # Line search for initial Δ
    try
        while goOn && jter < 30
            @. xs = x + α1 * d
            f1, g1 = f("fgx", pbm, xs)
            if f1 > min(f0 + α1*c1*s, p0)
                Δ, f1, g1, s1 = zoom(α0, α1, x, d, f, f0, s, c1, c2, pbm)
                goOn = false
            else
                s1 = dot(d, g1)
                if abs(s1) <= -c2*s
                    Δ = α1
                    goOn = false
                elseif s1 >= 0
                    Δ, f1, g1, s1 = zoom(α1, 2.0, x, d, f, f0, s, c1, c2, pbm)
                    goOn = false
                else
                    α1 = (α1 + 2.0)/2
                end
            end
            jter += 1
        end
    catch
        Δ = norm(g)
        println("Initial line search failed, setting Δ = ||g||")
    end
    
    t_start = time()  # Start timer
    while Ng > tol && k < maxIter
        # Timeout check
        if time() - t_start > 3600
            println("Timeout: Exceeded 1 hour in trustRegionARIES23.")
            break
        end
        c = ARIESDogleg(Δ, g, B, pU, pN, d, tol, n, α, β) # ARIES dogleg approximation of the quadratic model
        nd = norm(d)
        if any(isnan.(d)) || cond(B) * tol > 1
            _, _, B = f("fgHx", pbm, x) # Recompute B
            B = Matrix(B)
            if cond(B) * tol > 1
                B .= 0.8 * B + 0.2 * I
            end
            S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
            minD = minimum(diag(S.D))
            if minD < tol
                B .= S.P'*S.U*abs.(S.D)*S.U'*S.P #B is P.D. iff D has positive entries
            end
            d .= -g
            nd = norm(d)
        end
        @. xs = x + d
        try
            f1 = evalgrsum!(xs, pbm, gs)
            denom = 0.5 * dot(d, B*d) + dot(g, d)
            ρ = (f1 - f0) / denom
            if ρ < c0 && Δ > ΔMin
                Δ = max(Δ / η, ΔMin)
            else
                x .= xs
                y .= gs .- g
                mul!(u, B, d)
                yd = dot(y, d)
                du = dot(d, u)
                B .+= (y * y') / yd - (u * u') / du
                g .= gs
                f0 = f1
                Ng = norm(g)
                updateΔ(Δ, ρ, c1, c2, η, ΔMin, ΔMax, d, c)
            end
        catch
            Δ = max(Δ / η, ΔMin)
        end
        k += 1
        G[k] = Ng
    end
    G = G[1:k]
    return x, Ng, k, tol, G
end

function trustRegionARIES31(pb::PB, pbm::PBM, f::Function, α::Number=3/4, β::Number=1/4)
    maxIter = 100000
    G = zeros(Float64, maxIter)
    x = copy(pb.x0)
    n = length(x)  # the length of x_k
    c0, c1, c2, η = 0.1, 1/4, 1/2, 2
    ΔMax, ΔMin = 8.0*n, 10.0^(-n)
    xs = similar(x)
    k = 0
    tol = 1e-4
    f0, g, B = f("fgHx", pbm, x)
    B = Matrix(B)
    if cond(B) * tol > 1
        B = 0.8 * B + 0.2 * I
    end
    Ng = norm(g)
    y = similar(g)
    pU = similar(g)
    pN = similar(g)
    gs = similar(g)
    S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
    minD = minimum(diag(S.D))
    if minD < tol
        B .= S.P'*S.U*abs.(S.D)*S.U'*S.P #B is P.D. iff D has positive entries
    end
    c = "I"
    u = similar(g)
    d = -g
    s = dot(d, g)
    jter = 0
    goOn = true
    α0, α1 = 0.0, 1.0
    Δ = 0.01
    p0 = f0
    # Line search for initial Δ
    try
        while goOn && jter < 30
            @. xs = x + α1 * d
            f1, g1 = f("fgx", pbm, xs)
            if f1 > min(f0 + α1*c1*s, p0)
                Δ, f1, g1, s1 = zoom(α0, α1, x, d, f, f0, s, c1, c2, pbm)
                goOn = false
            else
                s1 = dot(d, g1)
                if abs(s1) <= -c2*s
                    Δ = α1
                    goOn = false
                elseif s1 >= 0
                    Δ, f1, g1, s1 = zoom(α1, 2.0, x, d, f, f0, s, c1, c2, pbm)
                    goOn = false
                else
                    α1 = (α1 + 2.0)/2
                end
            end
            jter += 1
        end
    catch
        Δ = norm(g)
        println("Initial line search failed, setting Δ = ||g||")
    end
    
    t_start = time()  # Start timer
    while Ng > tol && k < maxIter
        # Timeout check
        if time() - t_start > 3600
            println("Timeout: Exceeded 1 hour in trustRegionARIES31.")
            break
        end
        c = ARIESDogleg(Δ, g, B, pU, pN, d, tol, n, α, β) # ARIES dogleg approximation of the quadratic model
        nd = norm(d)
        if any(isnan.(d)) || cond(B) * tol > 1
            _, _, B = f("fgHx", pbm, x) # Recompute B
            B = Matrix(B)
            if cond(B) * tol > 1
                B .= 0.8 * B + 0.2 * I
            end
            S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
            minD = minimum(diag(S.D))
            if minD < tol
                B .= S.P'*S.U*abs.(S.D)*S.U'*S.P #B is P.D. iff D has positive entries
            end
            d .= -g
            nd = norm(d)
        end
        @. xs = x + d
        try
            f1 = evalgrsum!(xs, pbm, gs)
            denom = 0.5 * dot(d, B*d) + dot(g, d)
            ρ = (f1 - f0) / denom
            if ρ < c0 && Δ > ΔMin
                Δ = max(Δ / η, ΔMin)
            else
                x .= xs
                y .= gs .- g
                mul!(u, B, d)
                yd = dot(y, d)
                du = dot(d, u)
                B .+= (y * y') / yd - (u * u') / du
                g .= gs
                f0 = f1
                Ng = norm(g)
                updateΔ(Δ, ρ, c1, c2, η, ΔMin, ΔMax, d, c)
            end
        catch
            Δ = max(Δ / η, ΔMin)
        end
        k += 1
        G[k] = Ng
    end
    G = G[1:k]
    return x, Ng, k, tol, G
end

function trustRegionARIES32(pb::PB, pbm::PBM, f::Function, α::Number=3/4, β::Number=1/2)
    maxIter = 100000
    G = zeros(Float64, maxIter)
    x = copy(pb.x0)
    n = length(x)  # the length of x_k
    c0, c1, c2, η = 0.1, 1/4, 1/2, 2
    ΔMax, ΔMin = 8.0*n, 10.0^(-n)
    xs = similar(x)
    k = 0
    tol = 1e-4
    f0, g, B = f("fgHx", pbm, x)
    B = Matrix(B)
    if cond(B) * tol > 1
        B = 0.8 * B + 0.2 * I
    end
    Ng = norm(g)
    y = similar(g)
    pU = similar(g)
    pN = similar(g)
    gs = similar(g)
    S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
    minD = minimum(diag(S.D))
    if minD < tol
        B .= S.P'*S.U*abs.(S.D)*S.U'*S.P #B is P.D. iff D has positive entries
    end
    c = "I"
    u = similar(g)
    d = -g
    s = dot(d, g)
    jter = 0
    goOn = true
    α0, α1 = 0.0, 1.0
    Δ = 0.01
    p0 = f0
    # Line search for initial Δ
    try
        while goOn && jter < 30
            @. xs = x + α1 * d
            f1, g1 = f("fgx", pbm, xs)
            if f1 > min(f0 + α1*c1*s, p0)
                Δ, f1, g1, s1 = zoom(α0, α1, x, d, f, f0, s, c1, c2, pbm)
                goOn = false
            else
                s1 = dot(d, g1)
                if abs(s1) <= -c2*s
                    Δ = α1
                    goOn = false
                elseif s1 >= 0
                    Δ, f1, g1, s1 = zoom(α1, 2.0, x, d, f, f0, s, c1, c2, pbm)
                    goOn = false
                else
                    α1 = (α1 + 2.0)/2
                end
            end
            jter += 1
        end
    catch
        Δ = norm(g)
        println("Initial line search failed, setting Δ = ||g||")
    end
    
    t_start = time()  # Start timer
    while Ng > tol && k < maxIter
        # Timeout check
        if time() - t_start > 3600
            println("Timeout: Exceeded 1 hour in trustRegionARIES32.")
            break
        end
        c = ARIESDogleg(Δ, g, B, pU, pN, d, tol, n, α, β) # ARIES dogleg approximation of the quadratic model
        nd = norm(d)
        if any(isnan.(d)) || cond(B) * tol > 1
            _, _, B = f("fgHx", pbm, x) # Recompute B
            B = Matrix(B)
            if cond(B) * tol > 1
                B .= 0.8 * B + 0.2 * I
            end
            S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
            minD = minimum(diag(S.D))
            if minD < tol
                B .= S.P'*S.U*abs.(S.D)*S.U'*S.P #B is P.D. iff D has positive entries
            end
            d .= -g
            nd = norm(d)
        end
        @. xs = x + d
        try
            f1 = evalgrsum!(xs, pbm, gs)
            denom = 0.5 * dot(d, B*d) + dot(g, d)
            ρ = (f1 - f0) / denom
            if ρ < c0 && Δ > ΔMin
                Δ = max(Δ / η, ΔMin)
            else
                x .= xs
                y .= gs .- g
                mul!(u, B, d)
                yd = dot(y, d)
                du = dot(d, u)
                B .+= (y * y') / yd - (u * u') / du
                g .= gs
                f0 = f1
                Ng = norm(g)
                updateΔ(Δ, ρ, c1, c2, η, ΔMin, ΔMax, d, c)
            end
        catch
            Δ = max(Δ / η, ΔMin)
        end
        k += 1
        G[k] = Ng
    end
    G = G[1:k]
    return x, Ng, k, tol, G
end


function trustRegionARIES33(pb::PB, pbm::PBM, f::Function, α::Number=3/4, β::Number=3/4)
    maxIter = 100000
    G = zeros(Float64, maxIter)
    x = copy(pb.x0)
    n = length(x)  # the length of x_k
    c0, c1, c2, η = 0.1, 1/4, 1/2, 2
    ΔMax, ΔMin = 8.0*n, 10.0^(-n)
    xs = similar(x)
    k = 0
    tol = 1e-4
    f0, g, B = f("fgHx", pbm, x)
    B = Matrix(B)
    if cond(B) * tol > 1
        B = 0.8 * B + 0.2 * I
    end
    Ng = norm(g)
    y = similar(g)
    pU = similar(g)
    pN = similar(g)
    gs = similar(g)
    S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
    minD = minimum(diag(S.D))
    if minD < tol
        B .= S.P'*S.U*abs.(S.D)*S.U'*S.P #B is P.D. iff D has positive entries
    end
    c = "I"
    u = similar(g)
    d = -g
    s = dot(d, g)
    jter = 0
    goOn = true
    α0, α1 = 0.0, 1.0
    Δ = 0.01
    p0 = f0
    # Line search for initial Δ
    try
        while goOn && jter < 30
            @. xs = x + α1 * d
            f1, g1 = f("fgx", pbm, xs)
            if f1 > min(f0 + α1*c1*s, p0)
                Δ, f1, g1, s1 = zoom(α0, α1, x, d, f, f0, s, c1, c2, pbm)
                goOn = false
            else
                s1 = dot(d, g1)
                if abs(s1) <= -c2*s
                    Δ = α1
                    goOn = false
                elseif s1 >= 0
                    Δ, f1, g1, s1 = zoom(α1, 2.0, x, d, f, f0, s, c1, c2, pbm)
                    goOn = false
                else
                    α1 = (α1 + 2.0)/2
                end
            end
            jter += 1
        end
    catch
        Δ = norm(g)
        println("Initial line search failed, setting Δ = ||g||")
    end
    
    t_start = time()  # Start timer
    while Ng > tol && k < maxIter
        # Timeout check
        if time() - t_start > 3600
            println("Timeout: Exceeded 1 hour in trustRegionARIES33.")
            break
        end
        c = ARIESDogleg(Δ, g, B, pU, pN, d, tol, n, α, β) # ARIES dogleg approximation of the quadratic model
        nd = norm(d)
        if any(isnan.(d)) || cond(B) * tol > 1
            _, _, B = f("fgHx", pbm, x) # Recompute B
            B = Matrix(B)
            if cond(B) * tol > 1
                B .= 0.8 * B + 0.2 * I
            end
            S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
            minD = minimum(diag(S.D))
            if minD < tol
                B .= S.P'*S.U*abs.(S.D)*S.U'*S.P #B is P.D. iff D has positive entries
            end
            d .= -g
            nd = norm(d)
        end
        @. xs = x + d
        try
            f1 = evalgrsum!(xs, pbm, gs)
            denom = 0.5 * dot(d, B*d) + dot(g, d)
            ρ = (f1 - f0) / denom
            if ρ < c0 && Δ > ΔMin
                Δ = max(Δ / η, ΔMin)
            else
                x .= xs
                y .= gs .- g
                mul!(u, B, d)
                yd = dot(y, d)
                du = dot(d, u)
                B .+= (y * y') / yd - (u * u') / du
                g .= gs
                f0 = f1
                Ng = norm(g)
                updateΔ(Δ, ρ, c1, c2, η, ΔMin, ΔMax, d, c)
            end
        catch
            Δ = max(Δ / η, ΔMin)
        end
        k += 1
        G[k] = Ng
    end
    G = G[1:k]
    return x, Ng, k, tol, G
end

function trustRegionARIESadaptive1(pb::PB, pbm::PBM, f::Function)
    maxIter = 100000
    G = zeros(Float64, maxIter)
    x = copy(pb.x0)
    n = length(x)  # the length of x_k
    c0, c1, c2, η = 0.1, 1/4, 1/2, 2
    ΔMax, ΔMin = 8.0*n, 10.0^(-n)
    xs = similar(x)
    k = 0
    tol = 1e-4
    f0, g, B = f("fgHx", pbm, x)
    B = Matrix(B)
    if cond(B) * tol > 1
        B = 0.8 * B + 0.2 * I
    end
    Ng = norm(g)
    y = similar(g)
    gs = similar(g)
    S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
    minD = minimum(diag(S.D))
    if minD < tol
        B .= S.P'*S.U*abs.(S.D)*S.U'*S.P #B is P.D. iff D has positive entries
    end
    c = "I"
    u = similar(g)
    d = -g
    s = dot(d, g)
    jter = 0
    goOn = true
    α0, α1 = 0.0, 1.0
    Δ = 0.01
    p0 = f0
    # Line search for initial Δ
    try
        while goOn && jter < 30
            @. xs = x + α1 * d
            f1, g1 = f("fgx", pbm, xs)
            if f1 > min(f0 + α1*c1*s, p0)
                Δ, f1, g1, s1 = zoom(α0, α1, x, d, f, f0, s, c1, c2, pbm)
                goOn = false
            else
                s1 = dot(d, g1)
                if abs(s1) <= -c2*s
                    Δ = α1
                    goOn = false
                elseif s1 >= 0
                    Δ, f1, g1, s1 = zoom(α1, 2.0, x, d, f, f0, s, c1, c2, pbm)
                    goOn = false
                else
                    α1 = (α1 + 2.0)/2
                end
            end
            jter += 1
        end
    catch
        Δ = norm(g)
        println("Initial line search failed, setting Δ = ||g||")
    end
    pN = - (B \ g)
    γ1 = - (Ng^2) / (dot(g, B * g))
    pU = γ1 * g
    α = dot(pN, pU)/(norm(pN)*norm(pU));
    β = 1 - α;
    t_start = time()  # Start timer
    while Ng > tol && k < maxIter
        # Timeout check
        if time() - t_start > 3600
            println("Timeout: Exceeded 1 hour in trustRegionARIESadaptive1.")
            break
        end
        c = ARIESDogleg2(Δ, g, B, pU, pN, d, tol, n, α, β) # ARIES dogleg approximation of the quadratic model
        nd = norm(d)
        if any(isnan.(d)) || cond(B) * tol > 1
            _, _, B = f("fgHx", pbm, x) # Recompute B
            B = Matrix(B)
            if cond(B) * tol > 1
                B .= 0.8 * B + 0.2 * I
            end
            S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
            minD = minimum(diag(S.D))
            if minD < tol
                B .= S.P'*S.U*abs.(S.D)*S.U'*S.P #B is P.D. iff D has positive entries
            end
            d .= -g
            nd = norm(d)
            @inbounds pN .= - (B \ g)
            γ1 = - (Ng^2) / (dot(g, B * g))
            @inbounds pU .= γ1 * g
            α = dot(pN, pU)/(norm(pN)*norm(pU));
            β = 1 - α;
        end
        @. xs = x + d
        try
            f1 = evalgrsum!(xs, pbm, gs)
            denom = 0.5 * dot(d, B*d) + dot(g, d)
            ρ = (f1 - f0) / denom
            if ρ < c0 && Δ > ΔMin
                Δ = max(Δ / η, ΔMin)
            else
                x .= xs
                y .= gs .- g
                mul!(u, B, d)
                yd = dot(y, d)
                du = dot(d, u)
                B .+= (y * y') / yd - (u * u') / du
                g .= gs
                f0 = f1
                Ng = norm(g)
                updateΔ(Δ, ρ, c1, c2, η, ΔMin, ΔMax, d, c)
                @inbounds pN .= - (B \ g)
                γ1 = - (Ng^2) / (dot(g, B * g))
                @inbounds pU .= γ1 * g
                α = dot(pN, pU)/(norm(pN)*norm(pU));
                β = 1 - α;
            end
        catch
            Δ = max(Δ / η, ΔMin)
        end
        k += 1
        G[k] = Ng
    end
    G = G[1:k]
    return x, Ng, k, tol, G
end

function trustRegionARIESadaptive2(pb::PB, pbm::PBM, f::Function)
    maxIter = 100000
    G = zeros(Float64, maxIter)
    x = copy(pb.x0)
    n = length(x)  # the length of x_k
    c0, c1, c2, η = 0.1, 1/4, 1/2, 2
    ΔMax, ΔMin = 8.0*n, 10.0^(-n)
    xs = similar(x)
    k = 0
    tol = 1e-4
    f0, g, B = f("fgHx", pbm, x)
    B = Matrix(B)
    if cond(B) * tol > 1
        B = 0.8 * B + 0.2 * I
    end
    Ng = norm(g)
    y = similar(g)
    gs = similar(g)
    S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
    minD = minimum(diag(S.D))
    if minD < tol
        B .= S.P'*S.U*(abs.(S.D) + tol * I)*S.U'*S.P #B is P.D. iff D has positive entries
    end
    c = "I"
    u = similar(g)
    d = -g
    s = dot(d, g)
    jter = 0
    goOn = true
    α0, α1 = 0.0, 1.0
    Δ = 0.01
    p0 = f0
    # Line search for initial Δ
    try
        while goOn && jter < 30
            @. xs = x + α1 * d
            f1, g1 = f("fgx", pbm, xs)
            if f1 > min(f0 + α1*c1*s, p0)
                Δ, f1, g1, s1 = zoom(α0, α1, x, d, f, f0, s, c1, c2, pbm)
                goOn = false
            else
                s1 = dot(d, g1)
                if abs(s1) <= -c2*s
                    Δ = α1
                    goOn = false
                elseif s1 >= 0
                    Δ, f1, g1, s1 = zoom(α1, 2.0, x, d, f, f0, s, c1, c2, pbm)
                    goOn = false
                else
                    α1 = (α1 + 2.0)/2
                end
            end
            jter += 1
        end
    catch
        Δ = norm(g)
        println("Initial line search failed, setting Δ = ||g||")
    end
    pN = - (B \ g)
    γ1 = - (Ng^2) / (dot(g, B * g))
    γ2 = Ng^2 / dot(g, pN)
    pU = γ1 * g
    α = γ1 * γ2
    β = 1 - α;
    t_start = time()  # Start timer
    while Ng > tol && k < maxIter
        # Timeout check
        if time() - t_start > 3600
            println("Timeout: Exceeded 1 hour in trustRegionARIESadaptive2.")
            break
        end
        c = ARIESDogleg2(Δ, g, B, pU, pN, d, tol, n, α, β) # ARIES dogleg approximation of the quadratic model
        nd = norm(d)
        if any(isnan.(d)) || cond(B) * tol > 1
            _, _, B = f("fgHx", pbm, x) # Recompute B
            B = Matrix(B)
            if cond(B) * tol > 1
                B .= 0.8 * B + 0.2 * I
            end
            S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
            minD = minimum(diag(S.D))
            if minD < tol
                B .= S.P'*S.U*abs.(S.D)*S.U'*S.P #B is P.D. iff D has positive entries
            end
            d .= -g
            nd = norm(d)
            @inbounds pN .= - (B \ g)
            γ1 = - (Ng^2) / (dot(g, B * g))
            γ2 = Ng^2 / dot(g, pN)
            @inbounds pU .= γ1 * g
            α = γ1 * γ2
            β = 1 - α;
        end
        @. xs = x + d
        try
            f1 = evalgrsum!(xs, pbm, gs)
            denom = 0.5 * dot(d, B*d) + dot(g, d)
            ρ = (f1 - f0) / denom
            if ρ < c0 && Δ > ΔMin
                Δ = max(Δ / η, ΔMin)
            else
                x .= xs
                y .= gs .- g
                mul!(u, B, d)
                yd = dot(y, d)
                du = dot(d, u)
                B .+= (y * y') / yd - (u * u') / du
                g .= gs
                f0 = f1
                Ng = norm(g)
                updateΔ(Δ, ρ, c1, c2, η, ΔMin, ΔMax, d, c)
                @inbounds pN .= - (B \ g)
                γ1 = - (Ng^2) / (dot(g, B * g))
                γ2 = Ng^2 / dot(g, pN)
                @inbounds pU .= γ1 * g
                α = γ1 * γ2
                β = 1 - α;
            end
        catch
            Δ = max(Δ / η, ΔMin)
        end
        k += 1
        G[k] = Ng
    end
    G = G[1:k]
    return x, Ng, k, tol, G
end


function trustRegionARIESadaptive3(pb::PB, pbm::PBM, f::Function)
    maxIter = 100000
    G = zeros(Float64, maxIter)
    x = copy(pb.x0)
    n = length(x)  # the length of x_k
    c0, c1, c2, η = 0.1, 1/4, 1/2, 2
    ΔMax, ΔMin = 8.0*n, 10.0^(-n)
    xs = similar(x)
    k = 0
    tol = 1e-4
    f0, g, B = f("fgHx", pbm, x)
    B = Matrix(B)
    if cond(B) * tol > 1
        B = 0.8 * B + 0.2 * I
    end
    Ng = norm(g)
    y = similar(g)
    gs = similar(g)
    S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
    minD = minimum(diag(S.D))
    if minD < tol
        B .= S.P'*S.U*(abs.(S.D) + tol * I)*S.U'*S.P #B is P.D. iff D has positive entries
    end
    c = "I"
    u = similar(g)
    d = -g
    s = dot(d, g)
    jter = 0
    goOn = true
    α0, α1 = 0.0, 1.0
    Δ = 0.01
    p0 = f0
    # Line search for initial Δ
    try
        while goOn && jter < 30
            @. xs = x + α1 * d
            f1, g1 = f("fgx", pbm, xs)
            if f1 > min(f0 + α1*c1*s, p0)
                Δ, f1, g1, s1 = zoom(α0, α1, x, d, f, f0, s, c1, c2, pbm)
                goOn = false
            else
                s1 = dot(d, g1)
                if abs(s1) <= -c2*s
                    Δ = α1
                    goOn = false
                elseif s1 >= 0
                    Δ, f1, g1, s1 = zoom(α1, 2.0, x, d, f, f0, s, c1, c2, pbm)
                    goOn = false
                else
                    α1 = (α1 + 2.0)/2
                end
            end
            jter += 1
        end
    catch
        Δ = norm(g)
        println("Initial line search failed, setting Δ = ||g||")
    end
    pN = - (B \ g)
    γ1 = - (Ng^2) / (dot(g, B * g))
    γ2 = Ng^2 / dot(g, pN)
    pU = γ1 * g
    α = dot(B * g, g)/(norm(B * g)*Ng);
    β = 1 - α;
    t_start = time()  # Start timer
    while Ng > tol && k < maxIter
        # Timeout check
        if time() - t_start > 3600
            println("Timeout: Exceeded 1 hour in trustRegionARIESadaptive3.")
            break
        end
        c = ARIESDogleg2(Δ, g, B, pU, pN, d, tol, n, α, β) # ARIES dogleg approximation of the quadratic model
        nd = norm(d)
        if any(isnan.(d)) || cond(B) * tol > 1
            _, _, B = f("fgHx", pbm, x) # Recompute B
            B = Matrix(B)
            if cond(B) * tol > 1
                B .= 0.8 * B + 0.2 * I
            end
            S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
            minD = minimum(diag(S.D))
            if minD < tol
                B .= S.P'*S.U*abs.(S.D)*S.U'*S.P #B is P.D. iff D has positive entries
            end
            d .= -g
            nd = norm(d)
            @inbounds pN .= - (B \ g)
            γ1 = - (Ng^2) / (dot(g, B * g))
            γ2 = Ng^2 / dot(g, pN)
            @inbounds pU .= γ1 * g
            α = dot(B * g, g)/(norm(B * g)*Ng);
            β = 1 - α;
        end
        @. xs = x + d
        try
            f1 = evalgrsum!(xs, pbm, gs)
            denom = 0.5 * dot(d, B*d) + dot(g, d)
            ρ = (f1 - f0) / denom
            if ρ < c0 && Δ > ΔMin
                Δ = max(Δ / η, ΔMin)
            else
                x .= xs
                y .= gs .- g
                mul!(u, B, d)
                yd = dot(y, d)
                du = dot(d, u)
                B .+= (y * y') / yd - (u * u') / du
                g .= gs
                f0 = f1
                Ng = norm(g)
                updateΔ(Δ, ρ, c1, c2, η, ΔMin, ΔMax, d, c)
                @inbounds pN .= - (B \ g)
                γ1 = - (Ng^2) / (dot(g, B * g))
                γ2 = Ng^2 / dot(g, pN)
                @inbounds pU .= γ1 * g
                α = dot(B * g, g)/(norm(B * g)*Ng);
                β = 1 - α;
            end
        catch
            Δ = max(Δ / η, ΔMin)
        end
        k += 1
        G[k] = Ng
    end
    G = G[1:k]
    return x, Ng, k, tol, G
end



function trustRegionARIESadaptive4(pb::PB, pbm::PBM, f::Function)
    maxIter = 100000
    G = zeros(Float64, maxIter)
    x = copy(pb.x0)
    n = length(x)  # the length of x_k
    c0, c1, c2, η = 0.1, 1/4, 1/2, 2
    ΔMax, ΔMin = 8.0*n, 10.0^(-n)
    xs = similar(x)
    k = 0
    tol = 1e-4
    f0, g, B = f("fgHx", pbm, x)
    B = Matrix(B)
    if cond(B) * tol > 1
        B = 0.8 * B + 0.2 * I
    end
    Ng = norm(g)
    y = similar(g)
    gs = similar(g)
    S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
    minD = minimum(diag(S.D))
    if minD < tol
        B .= S.P'*S.U*abs.(S.D)*S.U'*S.P #B is P.D. iff D has positive entries
    end
    c = "I"
    u = similar(g)
    d = -g
    s = dot(d, g)
    jter = 0
    goOn = true
    α0, α1 = 0.0, 1.0
    Δ = 0.01
    p0 = f0
    # Line search for initial Δ
    try
        while goOn && jter < 30
            @. xs = x + α1 * d
            f1, g1 = f("fgx", pbm, xs)
            if f1 > min(f0 + α1*c1*s, p0)
                Δ, f1, g1, s1 = zoom(α0, α1, x, d, f, f0, s, c1, c2, pbm)
                goOn = false
            else
                s1 = dot(d, g1)
                if abs(s1) <= -c2*s
                    Δ = α1
                    goOn = false
                elseif s1 >= 0
                    Δ, f1, g1, s1 = zoom(α1, 2.0, x, d, f, f0, s, c1, c2, pbm)
                    goOn = false
                else
                    α1 = (α1 + 2.0)/2
                end
            end
            jter += 1
        end
    catch
        Δ = norm(g)
        println("Initial line search failed, setting Δ = ||g||")
    end
    pN = - (B \ g)
    γ1 = - (Ng^2) / (dot(g, B * g))
    pU = γ1 * g
    α = dot(pN, pU)/(norm(pN)*norm(pU));
    β = α;
    t_start = time()  # Start timer
    while Ng > tol && k < maxIter
        # Timeout check
        if time() - t_start > 3600
            println("Timeout: Exceeded 1 hour in trustRegionARIESadaptive4.")
            break
        end
        c = ARIESDogleg2(Δ, g, B, pU, pN, d, tol, n, α, β) # ARIES dogleg approximation of the quadratic model
        nd = norm(d)
        if any(isnan.(d)) || cond(B) * tol > 1
            _, _, B = f("fgHx", pbm, x) # Recompute B
            B = Matrix(B)
            if cond(B) * tol > 1
                B .= 0.8 * B + 0.2 * I
            end
            S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
            minD = minimum(diag(S.D))
            if minD < tol
                B .= S.P'*S.U*abs.(S.D)*S.U'*S.P #B is P.D. iff D has positive entries
            end
            d .= -g
            nd = norm(d)
            @inbounds pN .= - (B \ g)
            γ1 = - (Ng^2) / (dot(g, B * g))
            @inbounds pU .= γ1 * g
            α = dot(pN, pU)/(norm(pN)*norm(pU));
            β = α;
        end
        @. xs = x + d
        try
            f1 = evalgrsum!(xs, pbm, gs)
            denom = 0.5 * dot(d, B*d) + dot(g, d)
            ρ = (f1 - f0) / denom
            if ρ < c0 && Δ > ΔMin
                Δ = max(Δ / η, ΔMin)
            else
                x .= xs
                y .= gs .- g
                mul!(u, B, d)
                yd = dot(y, d)
                du = dot(d, u)
                B .+= (y * y') / yd - (u * u') / du
                g .= gs
                f0 = f1
                Ng = norm(g)
                updateΔ(Δ, ρ, c1, c2, η, ΔMin, ΔMax, d, c)
                @inbounds pN .= - (B \ g)
                γ1 = - (Ng^2) / (dot(g, B * g))
                @inbounds pU .= γ1 * g
                α = dot(pN, pU)/(norm(pN)*norm(pU));
                β = α;
            end
        catch
            Δ = max(Δ / η, ΔMin)
        end
        k += 1
        G[k] = Ng
    end
    G = G[1:k]
    return x, Ng, k, tol, G
end

function trustRegionARIESadaptive5(pb::PB, pbm::PBM, f::Function)
    maxIter = 100000
    G = zeros(Float64, maxIter)
    x = copy(pb.x0)
    n = length(x)  # the length of x_k
    c0, c1, c2, η = 0.1, 1/4, 1/2, 2
    ΔMax, ΔMin = 8.0*n, 10.0^(-n)
    xs = similar(x)
    k = 0
    tol = 1e-4
    f0, g, B = f("fgHx", pbm, x)
    B = Matrix(B)
    if cond(B) * tol > 1
        B = 0.8 * B + 0.2 * I
    end
    Ng = norm(g)
    y = similar(g)
    gs = similar(g)
    S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
    minD = minimum(diag(S.D))
    if minD < tol
        B .= S.P'*S.U*(abs.(S.D) + tol * I)*S.U'*S.P #B is P.D. iff D has positive entries
    end
    c = "I"
    u = similar(g)
    d = -g
    s = dot(d, g)
    jter = 0
    goOn = true
    α0, α1 = 0.0, 1.0
    Δ = 0.01
    p0 = f0
    # Line search for initial Δ
    try
        while goOn && jter < 30
            @. xs = x + α1 * d
            f1, g1 = f("fgx", pbm, xs)
            if f1 > min(f0 + α1*c1*s, p0)
                Δ, f1, g1, s1 = zoom(α0, α1, x, d, f, f0, s, c1, c2, pbm)
                goOn = false
            else
                s1 = dot(d, g1)
                if abs(s1) <= -c2*s
                    Δ = α1
                    goOn = false
                elseif s1 >= 0
                    Δ, f1, g1, s1 = zoom(α1, 2.0, x, d, f, f0, s, c1, c2, pbm)
                    goOn = false
                else
                    α1 = (α1 + 2.0)/2
                end
            end
            jter += 1
        end
    catch
        Δ = norm(g)
        println("Initial line search failed, setting Δ = ||g||")
    end
    pN = - (B \ g)
    γ1 = - (Ng^2) / (dot(g, B * g))
    γ2 = Ng^2 / dot(g, pN)
    pU = γ1 * g
    α = γ1 * γ2
    β = α;
    t_start = time()  # Start timer
    while Ng > tol && k < maxIter
        # Timeout check
        if time() - t_start > 3600
            println("Timeout: Exceeded 1 hour in trustRegionARIESadaptive5.")
            break
        end
        c = ARIESDogleg2(Δ, g, B, pU, pN, d, tol, n, α, β) # ARIES dogleg approximation of the quadratic model
        nd = norm(d)
        if any(isnan.(d)) || cond(B) * tol > 1
            _, _, B = f("fgHx", pbm, x) # Recompute B
            B = Matrix(B)
            if cond(B) * tol > 1
                B .= 0.8 * B + 0.2 * I
            end
            S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
            minD = minimum(diag(S.D))
            if minD < tol
                B .= S.P'*S.U*abs.(S.D)*S.U'*S.P #B is P.D. iff D has positive entries
            end
            d .= -g
            nd = norm(d)
            @inbounds pN .= - (B \ g)
            γ1 = - (Ng^2) / (dot(g, B * g))
            γ2 = Ng^2 / dot(g, pN)
            @inbounds pU .= γ1 * g
            α = γ1 * γ2
            β = α;
        end
        @. xs = x + d
        try
            f1 = evalgrsum!(xs, pbm, gs)
            denom = 0.5 * dot(d, B*d) + dot(g, d)
            ρ = (f1 - f0) / denom
            if ρ < c0 && Δ > ΔMin
                Δ = max(Δ / η, ΔMin)
            else
                x .= xs
                y .= gs .- g
                mul!(u, B, d)
                yd = dot(y, d)
                du = dot(d, u)
                B .+= (y * y') / yd - (u * u') / du
                g .= gs
                f0 = f1
                Ng = norm(g)
                updateΔ(Δ, ρ, c1, c2, η, ΔMin, ΔMax, d, c)
                @inbounds pN .= - (B \ g)
                γ1 = - (Ng^2) / (dot(g, B * g))
                γ2 = Ng^2 / dot(g, pN)
                @inbounds pU .= γ1 * g
                α = γ1 * γ2
                β = α;
            end
        catch
            Δ = max(Δ / η, ΔMin)
        end
        k += 1
        G[k] = Ng
    end
    G = G[1:k]
    return x, Ng, k, tol, G
end


function trustRegionARIESadaptive6(pb::PB, pbm::PBM, f::Function)
    maxIter = 100000
    G = zeros(Float64, maxIter)
    x = copy(pb.x0)
    n = length(x)  # the length of x_k
    c0, c1, c2, η = 0.1, 1/4, 1/2, 2
    ΔMax, ΔMin = 8.0*n, 10.0^(-n)
    xs = similar(x)
    k = 0
    tol = 1e-4
    f0, g, B = f("fgHx", pbm, x)
    B = Matrix(B)
    if cond(B) * tol > 1
        B = 0.8 * B + 0.2 * I
    end
    Ng = norm(g)
    y = similar(g)
    gs = similar(g)
    S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
    minD = minimum(diag(S.D))
    if minD < tol
        B .= S.P'*S.U*(abs.(S.D) + tol * I)*S.U'*S.P #B is P.D. iff D has positive entries
    end
    c = "I"
    u = similar(g)
    d = -g
    s = dot(d, g)
    jter = 0
    goOn = true
    α0, α1 = 0.0, 1.0
    Δ = 0.01
    p0 = f0
    # Line search for initial Δ
    try
        while goOn && jter < 30
            @. xs = x + α1 * d
            f1, g1 = f("fgx", pbm, xs)
            if f1 > min(f0 + α1*c1*s, p0)
                Δ, f1, g1, s1 = zoom(α0, α1, x, d, f, f0, s, c1, c2, pbm)
                goOn = false
            else
                s1 = dot(d, g1)
                if abs(s1) <= -c2*s
                    Δ = α1
                    goOn = false
                elseif s1 >= 0
                    Δ, f1, g1, s1 = zoom(α1, 2.0, x, d, f, f0, s, c1, c2, pbm)
                    goOn = false
                else
                    α1 = (α1 + 2.0)/2
                end
            end
            jter += 1
        end
    catch
        Δ = norm(g)
        println("Initial line search failed, setting Δ = ||g||")
    end
    pN = - (B \ g)
    γ1 = - (Ng^2) / (dot(g, B * g))
    γ2 = Ng^2 / dot(g, pN)
    pU = γ1 * g
    α = dot(B * g, g)/(norm(B * g)*Ng);
    β = α;
    t_start = time()  # Start timer
    while Ng > tol && k < maxIter
        # Timeout check
        if time() - t_start > 3600
            println("Timeout: Exceeded 1 hour in trustRegionARIESadaptive6.")
            break
        end
        c = ARIESDogleg2(Δ, g, B, pU, pN, d, tol, n, α, β) # ARIES dogleg approximation of the quadratic model
        nd = norm(d)
        if any(isnan.(d)) || cond(B) * tol > 1
            _, _, B = f("fgHx", pbm, x) # Recompute B
            B = Matrix(B)
            if cond(B) * tol > 1
                B .= 0.8 * B + 0.2 * I
            end
            S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
            minD = minimum(diag(S.D))
            if minD < tol
                B .= S.P'*S.U*abs.(S.D)*S.U'*S.P #B is P.D. iff D has positive entries
            end
            d .= -g
            nd = norm(d)
            @inbounds pN .= - (B \ g)
            γ1 = - (Ng^2) / (dot(g, B * g))
            γ2 = Ng^2 / dot(g, pN)
            @inbounds pU .= γ1 * g
            α = dot(B * g, g)/(norm(B * g)*Ng);
            β = α;
        end
        @. xs = x + d
        try
            f1 = evalgrsum!(xs, pbm, gs)
            denom = 0.5 * dot(d, B*d) + dot(g, d)
            ρ = (f1 - f0) / denom
            if ρ < c0 && Δ > ΔMin
                Δ = max(Δ / η, ΔMin)
            else
                x .= xs
                y .= gs .- g
                mul!(u, B, d)
                yd = dot(y, d)
                du = dot(d, u)
                B .+= (y * y') / yd - (u * u') / du
                g .= gs
                f0 = f1
                Ng = norm(g)
                updateΔ(Δ, ρ, c1, c2, η, ΔMin, ΔMax, d, c)
                @inbounds pN .= - (B \ g)
                γ1 = - (Ng^2) / (dot(g, B * g))
                γ2 = Ng^2 / dot(g, pN)
                @inbounds pU .= γ1 * g
                α = dot(B * g, g)/(norm(B * g)*Ng);
                β = α;
            end
        catch
            Δ = max(Δ / η, ΔMin)
        end
        k += 1
        G[k] = Ng
    end
    G = G[1:k]
    return x, Ng, k, tol, G
end


function trustRegionARIESadaptive7(pb::PB, pbm::PBM, f::Function)
    maxIter = 100000
    G = zeros(Float64, maxIter)
    x = copy(pb.x0)
    n = length(x)  # the length of x_k
    c0, c1, c2, η = 0.1, 1/4, 1/2, 2
    ΔMax, ΔMin = 8.0*n, 10.0^(-n)
    xs = similar(x)
    k = 0
    tol = 1e-4
    f0, g, B = f("fgHx", pbm, x)
    B = Matrix(B)
    if cond(B) * tol > 1
        B = 0.8 * B + 0.2 * I
    end
    Ng = norm(g)
    y = similar(g)
    gs = similar(g)
    S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
    minD = minimum(diag(S.D))
    if minD < tol
        B .= S.P'*S.U*(abs.(S.D) + tol * I)*S.U'*S.P #B is P.D. iff D has positive entries
    end
    c = "I"
    u = similar(g)
    d = -g
    s = dot(d, g)
    jter = 0
    goOn = true
    α0, α1 = 0.0, 1.0
    Δ = 0.01
    p0 = f0
    # Line search for initial Δ
    try
        while goOn && jter < 30
            @. xs = x + α1 * d
            f1, g1 = f("fgx", pbm, xs)
            if f1 > min(f0 + α1*c1*s, p0)
                Δ, f1, g1, s1 = zoom(α0, α1, x, d, f, f0, s, c1, c2, pbm)
                goOn = false
            else
                s1 = dot(d, g1)
                if abs(s1) <= -c2*s
                    Δ = α1
                    goOn = false
                elseif s1 >= 0
                    Δ, f1, g1, s1 = zoom(α1, 2.0, x, d, f, f0, s, c1, c2, pbm)
                    goOn = false
                else
                    α1 = (α1 + 2.0)/2
                end
            end
            jter += 1
        end
    catch
        Δ = norm(g)
        println("Initial line search failed, setting Δ = ||g||")
    end
    pN = - (B \ g)
    γ1 = - (Ng^2) / (dot(g, B * g))
    γ2 = Ng^2 / dot(g, pN)
    pU = γ1 * g
    α = 1/2;
    β = 1/2;
    t_start = time()  # Start timer
    while Ng > tol && k < maxIter
        # Timeout check
        if time() - t_start > 3600
            println("Timeout: Exceeded 1 hour in trustRegionARIESadaptive7.")
            break
        end
        c = ARIESDogleg2(Δ, g, B, pU, pN, d, tol, n, α, β) # ARIES dogleg approximation of the quadratic model
        nd = norm(d)
        if any(isnan.(d)) || cond(B) * tol > 1
            _, _, B = f("fgHx", pbm, x) # Recompute B
            B = Matrix(B)
            if cond(B) * tol > 1
                B .= 0.8 * B + 0.2 * I
            end
            S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
            minD = minimum(diag(S.D))
            if minD < tol
                B .= S.P'*S.U*abs.(S.D)*S.U'*S.P #B is P.D. iff D has positive entries
            end
            d .= -g
            nd = norm(d)
            @inbounds pN .= - (B \ g)
            γ1 = - (Ng^2) / (dot(g, B * g))
            γ2 = Ng^2 / dot(g, pN)
            @inbounds pU .= γ1 * g
            α = 1/2;
            β = 1/2;
        end
        @. xs = x + d
        try
            f1 = evalgrsum!(xs, pbm, gs)
            denom = 0.5 * dot(d, B*d) + dot(g, d)
            ρ = (f1 - f0) / denom
            if ρ < c0 && Δ > ΔMin
                Δ = max(Δ / η, ΔMin)
            else
                x .= xs
                y .= gs .- g
                mul!(u, B, d)
                yd = dot(y, d)
                du = dot(d, u)
                B .+= (y * y') / yd - (u * u') / du
                g .= gs
                f0 = f1
                Ng = norm(g)
                updateΔ(Δ, ρ, c1, c2, η, ΔMin, ΔMax, d, c)
                @inbounds pN .= - (B \ g)
                γ1 = - (Ng^2) / (dot(g, B * g))
                γ2 = Ng^2 / dot(g, pN)
                @inbounds pU .= γ1 * g
                α = abs(dot(d, pU)) / (norm(d) * norm(pU))
                β = abs(dot(d, pN)) / (norm(d) * norm(pN))
            end
        catch
            Δ = max(Δ / η, ΔMin)
        end
        k += 1
        G[k] = Ng
    end
    G = G[1:k]
    return x, Ng, k, tol, G
end