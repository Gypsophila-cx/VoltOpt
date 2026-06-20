# ex7_collocation_fast.jl
#
# Fast construction version of the fractional Jacobi / Muentz collocation method
# for the weakly singular Volterra equation
#
#   u(x) + lambda0 * int_{-1}^x (x-y)^(-mu) K(x,y) u(y) dy = g(x),  x in [-1,1],
#
# where
#   K(x,y) = 1 / ((x+1)^2 + (y-1)^2),
#   g(x)   = 1 + sin(pi*x)/(10^3*x^2 + 1),
#   lambda0 = 0.5, mu = 0.5.
#
# Mapping x = 2t - 1, y = 2s - 1 gives
#
#   v(t) + lambda0 * 2^(1-mu) * int_0^t (t-s)^(-mu) K(2t-1,2s-1)v(s) ds = g(2t-1).
#
# Approximation space:
#   span{J_m^{alpha,beta}(2 t^fraclambda - 1), m=0,...,N}.
#
# Fast construction idea:
#   The paper uses N+1 quadrature nodes in the inner theta integral. The slow
#   reference code evaluated all basis functions inside an i-q-m triple loop.
#   This script instead caps the quadrature order by --quad and assembles each
#   weighted Vandermonde block columnwise via a three-term Jacobi recurrence:
#
#       A += diag(row_weight_q) * V_q,
#       (V_q)_{i,m} = J_m^{alpha,beta}(2 z_i theta_q - 1), z_i = t_i^fraclambda.
#
#   With fixed/capped Q, the assembly cost is O(Q N^2), not O(N^3).
#   Set --quad=0 to recover the original paper-style Q=N+1 quadrature, but with
#   the faster columnwise assembly implementation.
#
# Output:
#   data/ex7_collocation_fast_cauchy.csv
#
# Examples:
#   julia ex7_collocation_fast.jl
#   julia ex7_collocation_fast.jl --numN=50 --quad=192
#   julia ex7_collocation_fast.jl --quad=0       # paper-style Q=N+1 quadrature
#   julia ex7_collocation_fast.jl --Nmin=10 --Nmax=1500 --numN=100 --grid=4001

using LinearAlgebra
using Printf
using Dates
using SpecialFunctions: loggamma

# -----------------------------
# Defaults
# -----------------------------
Nmin_default = 10
Nmax_default = 1500
numN_default = 100
ngrid_default = 4001

# Q is min(N+1, quad_cap) when quad_cap > 0.
# Set --quad=0 to use Q=N+1.
quad_cap_default = 256

# Fractional Jacobi parameters.
fraclambda_default = 0.5
alpha_default = -0.5
beta_default = -0.5

# Equation parameters.
coef_lambda = 0.5
mu = 0.5

# Cache Gauss-Jacobi rules because the capped quadrature rule is reused many times.
gauss_cache = Dict{Tuple{Int,Float64,Float64},Tuple{Vector{Float64},Vector{Float64}}}()

# -----------------------------
# Problem data on [-1,1]
# -----------------------------
@inline function kernel_xy(x::Float64, y::Float64)
    return 1.0 / ((x + 1.0)^2 + (y - 1.0)^2)
end

@inline function rhs_x(x::Float64)
    return 1.0 + sin(pi * x) / (1.0e3 * x^2 + 1.0)
end

# -----------------------------
# Command-line parser
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
    quad_cap = parse(Int, get(opts, "quad", string(quad_cap_default)))

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
    if quad_cap < 0
        error("quad must be nonnegative. Use quad=0 for Q=N+1.")
    end

    return (; Nmin, Nmax, numN, ngrid, quad_cap, fraclambda, alpha, beta)
end

# -----------------------------
# Gauss-Jacobi quadrature on [0,1]
#
# Nodes z_i and weights w_i for
#   int_0^1 (1-z)^alpha z^beta f(z) dz.
# -----------------------------
function gauss_jacobi_shifted_uncached(n::Int, alpha::Float64, beta::Float64)
    n >= 1 || error("number of quadrature points must be positive")
    alpha > -1.0 || error("alpha must be greater than -1")
    beta > -1.0 || error("beta must be greater than -1")

    diag = Vector{Float64}(undef, n)
    offdiag = Vector{Float64}(undef, max(n - 1, 0))

    # Jacobi matrix for orthonormal polynomials on [-1,1] with weight
    # (1-x)^alpha (1+x)^beta.
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
    x = F.values
    Q = F.vectors

    z = @. (x + 1.0) / 2.0
    logmass = loggamma(alpha + 1.0) + loggamma(beta + 1.0) - loggamma(alpha + beta + 2.0)
    mass = exp(logmass)
    w = mass .* (Q[1, :] .^ 2)

    return z, w
end

function gauss_jacobi_shifted(n::Int, alpha::Float64, beta::Float64)
    key = (n, alpha, beta)
    if haskey(gauss_cache, key)
        return gauss_cache[key]
    end
    val = gauss_jacobi_shifted_uncached(n, alpha, beta)
    gauss_cache[key] = val
    return val
end

# -----------------------------
# Jacobi polynomial recurrence.
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

# Add diag(roww) * V(xvec) to A, where V[i,m+1] = P_m^{(alpha,beta)}(xvec[i]).
# This is the main fast block assembler. It is column-major and uses only O(n) workspace.
function add_weighted_jacobi_vandermonde!(A::Matrix{Float64},
                                          xvec::Vector{Float64},
                                          roww::Vector{Float64},
                                          N::Int,
                                          alpha::Float64,
                                          beta::Float64,
                                          pnm1::Vector{Float64},
                                          pn::Vector{Float64},
                                          pnext::Vector{Float64})
    nrow = length(xvec)
    @inbounds begin
        # degree 0
        for i in 1:nrow
            pnm1[i] = 1.0
            A[i, 1] += roww[i]
        end

        if N >= 1
            # degree 1
            for i in 1:nrow
                pn[i] = 0.5 * ((alpha - beta) + (alpha + beta + 2.0) * xvec[i])
                A[i, 2] += roww[i] * pn[i]
            end
        end

        for deg in 2:N
            nf = Float64(deg)
            a1 = 2.0 * nf * (nf + alpha + beta) * (2.0 * nf + alpha + beta - 2.0)
            a2 = (2.0 * nf + alpha + beta - 1.0) * (alpha^2 - beta^2)
            a3 = (2.0 * nf + alpha + beta - 2.0) *
                 (2.0 * nf + alpha + beta - 1.0) *
                 (2.0 * nf + alpha + beta)
            a4 = 2.0 * (nf + alpha - 1.0) * (nf + beta - 1.0) *
                 (2.0 * nf + alpha + beta)

            for i in 1:nrow
                pnext[i] = ((a2 + a3 * xvec[i]) * pn[i] - a4 * pnm1[i]) / a1
                A[i, deg + 1] += roww[i] * pnext[i]
            end

            tmp = pnm1
            pnm1 = pn
            pn = pnext
            pnext = tmp
        end
    end
    return A
end

@inline function stable_ratio(theta::Float64, fraclambda::Float64)
    # ratio = (1 - theta^(1/fraclambda))/(1 - theta), removable limit 1/fraclambda at theta=1.
    if abs(1.0 - theta) < 1.0e-12
        return 1.0 / fraclambda
    else
        return (1.0 - theta^(1.0 / fraclambda)) / (1.0 - theta)
    end
end

struct CollocationSolution
    N::Int
    Q::Int
    coeffs::Vector{Float64}
    build_sec::Float64
    solve_sec::Float64
    total_sec::Float64
end

function solve_collocation_fast(N::Int;
                                fraclambda::Float64,
                                alpha::Float64,
                                beta::Float64,
                                quad_cap::Int)
    n = N + 1
    Qinner = quad_cap == 0 ? n : min(n, quad_cap)
    t0 = time()

    # Collocation nodes: z_i are shifted Jacobi-Gauss nodes, t_i = z_i^(1/fraclambda).
    zcol, _ = gauss_jacobi_shifted(n, alpha, beta)
    tcol = zcol .^ (1.0 / fraclambda)
    xcol_std = @. 2.0 * zcol - 1.0
    xorig = @. 2.0 * tcol - 1.0

    # Quadrature in theta with weight (1-theta)^(-mu) theta^(1/fraclambda-1).
    theta, wq = gauss_jacobi_shifted(Qinner, -mu, 1.0 / fraclambda - 1.0)

    A = zeros(Float64, n, n)
    rhs = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        rhs[i] = rhs_x(xorig[i])
    end

    # Work arrays for block assembly.
    roww = Vector{Float64}(undef, n)
    xeval = Vector{Float64}(undef, n)
    p0 = Vector{Float64}(undef, n)
    p1 = Vector{Float64}(undef, n)
    p2 = Vector{Float64}(undef, n)

    # Identity/collocation Vandermonde block.
    fill!(roww, 1.0)
    copyto!(xeval, xcol_std)
    add_weighted_jacobi_vandermonde!(A, xeval, roww, N, alpha, beta, p0, p1, p2)

    mapped_coef = coef_lambda * 2.0^(1.0 - mu)
    inv_fraclambda = 1.0 / fraclambda

    # Fast Volterra block assembly.
    @inbounds for q in 1:Qinner
        th = theta[q]
        thpow = th^inv_fraclambda
        ratio = stable_ratio(th, fraclambda)
        common = mapped_coef * wq[q] * inv_fraclambda * ratio^(-mu)

        for i in 1:n
            ti = tcol[i]
            s = ti * thpow
            yorig = 2.0 * s - 1.0

            # Basis argument at s: 2*s^fraclambda - 1 = 2*(t_i^fraclambda)*theta - 1.
            xeval[i] = 2.0 * zcol[i] * th - 1.0
            roww[i] = common * ti^(1.0 - mu) * kernel_xy(xorig[i], yorig)
        end

        add_weighted_jacobi_vandermonde!(A, xeval, roww, N, alpha, beta, p0, p1, p2)
    end

    build_sec = time() - t0
    t1 = time()
    coeffs = A \ rhs
    solve_sec = time() - t1
    total_sec = time() - t0

    return CollocationSolution(N, Qinner, coeffs, build_sec, solve_sec, total_sec)
end

function eval_jacobi_series_grid(coeffs::Vector{Float64},
                                 tgrid::AbstractVector{Float64},
                                 fraclambda::Float64,
                                 alpha::Float64,
                                 beta::Float64)
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

function clean_csv_status(status::AbstractString)
    return replace(status, ',' => ';', '\n' => ' ', '\r' => ' ')
end

function main()
    opts = parse_args()
    script_dir = @__DIR__
    data_dir = ensure_dir(joinpath(script_dir, "data"))
    out_file = joinpath(data_dir, "ex7_collocation_fast_cauchy.csv")

    Ns = make_N_list(opts.Nmin, opts.Nmax, opts.numN)
    tgrid = collect(range(0.0, 1.0, length=opts.ngrid))

    println("ex7_collocation_fast.jl")
    println("start              = ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    println("N range            = ", first(Ns), " ... ", last(Ns), "  (", length(Ns), " values)")
    println("fraclambda         = ", opts.fraclambda)
    println("alpha, beta        = ", opts.alpha, ", ", opts.beta)
    println("inner quad cap     = ", opts.quad_cap == 0 ? "paper-style Q=N+1" : string(opts.quad_cap))
    println("grid points        = ", opts.ngrid)
    println("output             = ", out_file)

    open(out_file, "w") do io
        println(io, "N,M,Q_N,Q_M,coeff_cauchy_inf,grid_cauchy_inf,build_N_sec,solve_N_sec,total_N_sec,build_M_sec,solve_M_sec,total_M_sec,status")
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
            solN = solve_collocation_fast(N;
                                          fraclambda=opts.fraclambda,
                                          alpha=opts.alpha,
                                          beta=opts.beta,
                                          quad_cap=opts.quad_cap)
            solM = solve_collocation_fast(M;
                                          fraclambda=opts.fraclambda,
                                          alpha=opts.alpha,
                                          beta=opts.beta,
                                          quad_cap=opts.quad_cap)

            coeff_err = coeff_cauchy_inf(solN.coeffs, solM.coeffs)
            uN_grid = eval_jacobi_series_grid(solN.coeffs, tgrid, opts.fraclambda, opts.alpha, opts.beta)
            uM_grid = eval_jacobi_series_grid(solM.coeffs, tgrid, opts.fraclambda, opts.alpha, opts.beta)
            grid_err = maximum(abs.(uN_grid .- uM_grid))

            @printf("  Q_N, Q_M         = %d, %d\n", solN.Q, solM.Q)
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
                @printf(io, "%d,%d,%d,%d,%.17e,%.17e,NaN,NaN,NaN,NaN,NaN,NaN,%s\n",
                        N, M, 0, 0, coeff_err, grid_err, clean_csv_status(status))
            else
                @printf(io, "%d,%d,%d,%d,%.17e,%.17e,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%s\n",
                        N, M, solN.Q, solM.Q,
                        coeff_err, grid_err,
                        solN.build_sec, solN.solve_sec, solN.total_sec,
                        solM.build_sec, solM.solve_sec, solM.total_sec,
                        clean_csv_status(status))
            end
        end
        flush(stdout)
    end

    println("\nfinished           = ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    println("csv written        = ", out_file)
end

main()
