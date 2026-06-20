# ex7_collocation.jl
#
# Müntz / fractional-Jacobi collocation method for
#
#   u(x) + lambda * ∫_{-1}^x (x-y)^(-mu) K(x,y)u(y)dy = g(x),  x ∈ [-1,1],
#
# with
#   K(x,y) = 1 / ((x+1)^2 + (y-1)^2),
#   g(x)   = 1 + sin(pi*x)/(10^3*x^2 + 1),
#   lambda = 0.5, mu = 0.5.
#
# The equation is mapped by x = 2t - 1, y = 2s - 1 to t,s ∈ [0,1]:
#
#   v(t) + lambda * 2^(1-mu) * ∫_0^t (t-s)^(-mu) K(2t-1,2s-1)v(s)ds = g(2t-1).
#
# The implementation follows the fractional Jacobi collocation construction in
# Hou--Lin--Azaiez--Xu (2019), especially equations (3.2)--(3.5).
#
# Output:
#   data/ex7_collocation_cauchy.csv
#
# Default N-grid:
#   100 linearly spaced integer values from 10 to 1500.
#
# Optional command line examples:
#   julia ex7_collocation.jl --numN=50
#   julia ex7_collocation.jl --Nmin=10 --Nmax=1500 --numN=100 --grid=4001
#   julia ex7_collocation.jl --fraclambda=0.5 --alpha=-0.5 --beta=-0.5

using LinearAlgebra
using Printf
using Dates
using SpecialFunctions: loggamma

# -----------------------------
# User-adjustable defaults
# -----------------------------
Nmin_default = 10
Nmax_default = 1500
numN_default = 100
ngrid_default = 4001

# Fractional Jacobi collocation parameters.
# J_n^{alpha,beta,fraclambda}(t) = J_n^{alpha,beta}(2t^fraclambda - 1).
fraclambda_default = 0.5
alpha_default = -0.5
beta_default = -0.5

# Equation parameters.
coef_lambda = 0.5
mu = 0.5

# -----------------------------
# Basic problem data on [-1,1]
# -----------------------------
@inline function kernel_xy(x::Float64, y::Float64)
    return 1.0 / ((x + 1.0)^2 + (y - 1.0)^2)
end

@inline function rhs_x(x::Float64)
    return 1.0 + sin(pi * x) / (1.0e3 * x^2 + 1.0)
end

# -----------------------------
# Small command-line parser
# -----------------------------
function parse_args()
    opts = Dict{String,String}()
    for arg in ARGS
        if startswith(arg, "--") && occursin("=", arg)
            key, val = split(arg[3:end], "=", limit=2)
            opts[key] = val
        end
    end

    Nmin = parse(Int, get(opts, "Nmin", string(Nmin_default)))
    Nmax = parse(Int, get(opts, "Nmax", string(Nmax_default)))
    numN = parse(Int, get(opts, "numN", string(numN_default)))
    ngrid = parse(Int, get(opts, "grid", string(ngrid_default)))

    fraclambda = parse(Float64, get(opts, "fraclambda", string(fraclambda_default)))
    alpha = parse(Float64, get(opts, "alpha", string(alpha_default)))
    beta = parse(Float64, get(opts, "beta", string(beta_default)))

    if !(0.0 < fraclambda <= 1.0)
        error("fraclambda must satisfy 0 < fraclambda <= 1.")
    end
    if alpha <= -1.0 || beta <= -1.0
        error("alpha and beta must be greater than -1.")
    end
    if Nmin < 1 || Nmax < Nmin || numN < 1
        error("Require 1 <= Nmin <= Nmax and numN >= 1.")
    end
    if ngrid < 2
        error("grid must be at least 2.")
    end

    return (; Nmin, Nmax, numN, ngrid, fraclambda, alpha, beta)
end

# -----------------------------
# Gauss-Jacobi quadrature on [0,1]
#
# Computes nodes z_i and weights w_i for
#   ∫_0^1 (1-z)^alpha z^beta f(z) dz.
#
# This self-contained Golub-Welsch implementation avoids depending on
# FastGaussQuadrature.jl.
# -----------------------------
function gauss_jacobi_shifted(n::Int, alpha::Float64, beta::Float64)
    n >= 1 || error("number of quadrature points must be positive")
    alpha > -1.0 || error("alpha must be greater than -1")
    beta > -1.0 || error("beta must be greater than -1")

    diag = Vector{Float64}(undef, n)
    offdiag = Vector{Float64}(undef, max(n - 1, 0))

    # Jacobi matrix for the orthonormal polynomials on [-1,1]
    diag[1] = (beta - alpha) / (alpha + beta + 2.0)
    for j in 2:n
        k = j - 1.0
        den = (2.0k + alpha + beta) * (2.0k + alpha + beta + 2.0)
        diag[j] = abs(den) < 100eps(Float64) ? 0.0 : (beta^2 - alpha^2) / den
    end

    if n >= 2
        offdiag[1] = sqrt(4.0 * (1.0 + alpha) * (1.0 + beta) /
                          ((alpha + beta + 3.0) * (alpha + beta + 2.0)^2))
        for j in 2:n-1
            k = Float64(j)
            abi = 2.0k + alpha + beta
            val = 4.0 * k * (k + alpha) * (k + beta) * (k + alpha + beta) /
                  ((abi^2 - 1.0) * abi^2)
            offdiag[j] = sqrt(max(val, 0.0))
        end
    end

    F = eigen(SymTridiagonal(diag, offdiag))
    x = F.values                         # nodes on [-1,1]
    Q = F.vectors

    # Shift nodes to [0,1]. The total mass on [0,1] is B(alpha+1,beta+1).
    z = @. (x + 1.0) / 2.0
    logmass = loggamma(alpha + 1.0) + loggamma(beta + 1.0) - loggamma(alpha + beta + 2.0)
    mass = exp(logmass)
    w = mass .* (Q[1, :] .^ 2)

    return z, w
end

# -----------------------------
# Jacobi polynomial recurrence
# P_n^{(alpha,beta)}(x), n=0,...,N.
# -----------------------------
function jacobi_values!(vals::Vector{Float64}, x::Float64, N::Int,
                        alpha::Float64, beta::Float64)
    @inbounds begin
        vals[1] = 1.0
        if N >= 1
            vals[2] = 0.5 * ((alpha - beta) + (alpha + beta + 2.0) * x)
        end
        for n in 2:N
            nf = Float64(n)
            a1 = 2.0 * nf * (nf + alpha + beta) * (2.0 * nf + alpha + beta - 2.0)
            a2 = (2.0 * nf + alpha + beta - 1.0) * (alpha^2 - beta^2)
            a3 = (2.0 * nf + alpha + beta - 2.0) *
                 (2.0 * nf + alpha + beta - 1.0) *
                 (2.0 * nf + alpha + beta)
            a4 = 2.0 * (nf + alpha - 1.0) * (nf + beta - 1.0) *
                 (2.0 * nf + alpha + beta)
            vals[n + 1] = ((a2 + a3 * x) * vals[n] - a4 * vals[n - 1]) / a1
        end
    end
    return vals
end

function jacobi_vandermonde(xnodes::AbstractVector{Float64}, N::Int,
                            alpha::Float64, beta::Float64)
    V = Matrix{Float64}(undef, length(xnodes), N + 1)
    tmp = Vector{Float64}(undef, N + 1)
    @inbounds for i in eachindex(xnodes)
        jacobi_values!(tmp, xnodes[i], N, alpha, beta)
        V[i, :] .= tmp
    end
    return V
end

@inline function stable_ratio(theta::Float64, fraclambda::Float64)
    # ratio = (1 - theta^(1/fraclambda))/(1 - theta), with removable limit 1/fraclambda at theta=1.
    if abs(1.0 - theta) < 1.0e-12
        return 1.0 / fraclambda
    else
        return (1.0 - theta^(1.0 / fraclambda)) / (1.0 - theta)
    end
end

struct CollocationSolution
    N::Int
    coeffs::Vector{Float64}
    build_sec::Float64
    solve_sec::Float64
    total_sec::Float64
end

function solve_collocation(N::Int; fraclambda::Float64, alpha::Float64, beta::Float64)
    n = N + 1
    t0 = time()

    # Collocation nodes: z_i are shifted Jacobi-Gauss nodes, t_i = z_i^(1/fraclambda).
    zcol, _ = gauss_jacobi_shifted(n, alpha, beta)
    tcol = zcol .^ (1.0 / fraclambda)
    xcol_std = @. 2.0 * zcol - 1.0

    # Quadrature in theta with weight (1-theta)^(-mu) theta^(1/fraclambda-1).
    theta, wq = gauss_jacobi_shifted(n, -mu, 1.0 / fraclambda - 1.0)

    # Unknowns are coefficients in J_n^{alpha,beta}(2t^fraclambda-1), n=0,...,N.
    A = jacobi_vandermonde(xcol_std, N, alpha, beta)
    rhs = Vector{Float64}(undef, n)

    # Constant after mapping [-1,1] to [0,1].
    mapped_coef = coef_lambda * 2.0^(1.0 - mu)

    # Add quadrature approximation of the Volterra operator.
    tmp = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        ti = tcol[i]
        zi = zcol[i]
        xorig = 2.0 * ti - 1.0
        rhs[i] = rhs_x(xorig)

        for q in 1:n
            th = theta[q]
            thpow = th^(1.0 / fraclambda)
            s = ti * thpow
            yorig = 2.0 * s - 1.0

            r = stable_ratio(th, fraclambda)
            kbar = ti^(1.0 - mu) / fraclambda * r^(-mu) * kernel_xy(xorig, yorig)
            factor = mapped_coef * wq[q] * kbar

            # u_N(t_i * theta^(1/lambda)) = p_N(z_i * theta), z=t^lambda.
            xeval = 2.0 * zi * th - 1.0
            jacobi_values!(tmp, xeval, N, alpha, beta)
            @simd for m in 1:n
                A[i, m] += factor * tmp[m]
            end
        end
    end

    build_sec = time() - t0
    t1 = time()
    coeffs = A \ rhs
    solve_sec = time() - t1
    total_sec = time() - t0

    return CollocationSolution(N, coeffs, build_sec, solve_sec, total_sec)
end

function eval_jacobi_series_grid(coeffs::Vector{Float64}, tgrid::AbstractVector{Float64},
                                 fraclambda::Float64, alpha::Float64, beta::Float64)
    N = length(coeffs) - 1
    vals = Vector{Float64}(undef, length(tgrid))
    P = Vector{Float64}(undef, N + 1)
    @inbounds for (i, t) in pairs(tgrid)
        z = t^fraclambda
        xstd = 2.0 * z - 1.0
        jacobi_values!(P, xstd, N, alpha, beta)
        s = 0.0
        @simd for k in 1:N+1
            s += coeffs[k] * P[k]
        end
        vals[i] = s
    end
    return vals
end

function coeff_cauchy_inf(cN::Vector{Float64}, cM::Vector{Float64})
    L = max(length(cN), length(cM))
    err = 0.0
    @inbounds for k in 1:L
        a = k <= length(cN) ? cN[k] : 0.0
        b = k <= length(cM) ? cM[k] : 0.0
        err = max(err, abs(a - b))
    end
    return err
end

function make_N_list(Nmin::Int, Nmax::Int, numN::Int)
    if numN == 1
        return [Nmin]
    end
    Ns = unique(round.(Int, range(Nmin, Nmax, length=numN)))
    Ns = filter(N -> Nmin <= N <= Nmax, Ns)
    return Ns
end

function ensure_dir(path::AbstractString)
    isdir(path) || mkpath(path)
    return path
end

function main()
    opts = parse_args()
    script_dir = @__DIR__
    data_dir = ensure_dir(joinpath(script_dir, "data"))
    out_file = joinpath(data_dir, "ex7_collocation_cauchy.csv")

    Ns = make_N_list(opts.Nmin, opts.Nmax, opts.numN)
    tgrid = collect(range(0.0, 1.0, length=opts.ngrid))

    println("ex7_collocation.jl")
    println("start              = ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    println("N range            = ", first(Ns), " ... ", last(Ns), "  (", length(Ns), " values)")
    println("fraclambda         = ", opts.fraclambda)
    println("alpha, beta        = ", opts.alpha, ", ", opts.beta)
    println("grid points        = ", opts.ngrid)
    println("output             = ", out_file)

    open(out_file, "w") do io
        println(io, "N,M,coeff_cauchy_inf,grid_cauchy_inf,build_N_sec,solve_N_sec,total_N_sec,build_M_sec,solve_M_sec,total_M_sec,status")
    end

    for N in Ns
        M = max(N + 1, round(Int, 1.01 * N))
        status = "ok"
        coeff_err = NaN
        grid_err = NaN
        solN = nothing
        solM = nothing

        try
            @printf("\nN = %d, M = %d\n", N, M)
            solN = solve_collocation(N; fraclambda=opts.fraclambda, alpha=opts.alpha, beta=opts.beta)
            solM = solve_collocation(M; fraclambda=opts.fraclambda, alpha=opts.alpha, beta=opts.beta)

            coeff_err = coeff_cauchy_inf(solN.coeffs, solM.coeffs)
            uN_grid = eval_jacobi_series_grid(solN.coeffs, tgrid, opts.fraclambda, opts.alpha, opts.beta)
            uM_grid = eval_jacobi_series_grid(solM.coeffs, tgrid, opts.fraclambda, opts.alpha, opts.beta)
            grid_err = maximum(abs.(uN_grid .- uM_grid))

            @printf("  coeff Cauchy inf = %.6e\n", coeff_err)
            @printf("  grid  Cauchy inf = %.6e\n", grid_err)
            @printf("  time N: build %.3f s, solve %.3f s, total %.3f s\n", solN.build_sec, solN.solve_sec, solN.total_sec)
            @printf("  time M: build %.3f s, solve %.3f s, total %.3f s\n", solM.build_sec, solM.solve_sec, solM.total_sec)
        catch err
            status = "failed: " * sprint(showerror, err)
            @warn "Failed at N=$N, M=$M" exception=(err, catch_backtrace())
        end

        open(out_file, "a") do io
            if solN === nothing || solM === nothing
                @printf(io, "%d,%d,%.17e,%.17e,NaN,NaN,NaN,NaN,NaN,NaN,%s\n",
                        N, M, coeff_err, grid_err, status)
            else
                @printf(io, "%d,%d,%.17e,%.17e,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%s\n",
                        N, M, coeff_err, grid_err,
                        solN.build_sec, solN.solve_sec, solN.total_sec,
                        solM.build_sec, solM.solve_sec, solM.total_sec,
                        status)
            end
        end
        flush(stdout)
    end

    println("\nfinished           = ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    println("csv written        = ", out_file)
end

main()
