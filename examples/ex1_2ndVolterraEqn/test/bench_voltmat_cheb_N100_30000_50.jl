using BenchmarkTools
using FFTW
using InteractiveUtils
using LinearAlgebra
using Printf
using SparseArrays

# This script is intended to live in examples/ex1_2ndVolterraEqn.
# It is VoltMat_Cheb-only and should not include or call GutlebOlver.
ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(ROOT, "src", "VoltMat", "voltmat_cheb.jl"))

Ns = unique(round.(Int, range(100, 30000; length=50)))
length(Ns) == 50 || error("Expected 50 distinct N values, got $(length(Ns)): $(Ns)")

tol = 1e-14
METHOD = "VoltMat_Cheb"
EQUATION_SIGN = "u + V_K u = f"
OUTPUT_CSV = joinpath(@__DIR__, "bench_voltmat_cheb_N100_30000_50_results.csv")

# ============================================================
# Kernel copied from debug_simple.jl / ex1_aging_maxwell.jl.
# ============================================================

tau_xy(x, y) = 1.5 + x - 0.5 * y

function K(x, y)
    tau = tau_xy(x, y)
    return exp(-(x - y) / tau) / tau
end

# Original ex1 equation, confirmed from ex1_aging_maxwell.jl:
#     u + V_K u = f
# Hence the discrete system is:
#     Aop = I + VK

# ============================================================
# Exact solution and independent RHS copied from ex1_aging_maxwell.jl.
# ============================================================

const x1 = -0.35
const x2 = 0.42
const A1 = 1.0
const A2 = -0.6
const delta1 = 1.0e-3
const delta2 = 1.5e-3

step_layer(x, x0, delta) = 0.5 * (1.0 + tanh((x - x0) / delta))

function u_exact(x)
    return 1.0 + A1 * step_layer(x, x1, delta1) + A2 * step_layer(x, x2, delta2)
end

const RHS_Q = 160
const RHS_SPLIT_L = 40.0

function gausslegendre_rule(n::Int)
    n <= 0 && throw(ArgumentError("n must be positive"))
    if n == 1
        return [0.0], [2.0]
    end

    beta = [k / sqrt(4.0 * k^2 - 1.0) for k in 1:(n - 1)]
    T = SymTridiagonal(zeros(n), beta)
    E = eigen(T)

    x = collect(E.values)
    w = 2.0 .* abs2.(E.vectors[1, :])
    p = sortperm(x)
    return x[p], w[p]
end

const GL_X, GL_W = gausslegendre_rule(RHS_Q)

function gauss_integral_on_interval(f, a::Float64, b::Float64)
    b > a || return 0.0
    c = 0.5 * (a + b)
    h = 0.5 * (b - a)
    s = 0.0
    @inbounds for q in eachindex(GL_X)
        y = c + h * GL_X[q]
        s += GL_W[q] * f(y)
    end
    return h * s
end

function rhs_integral_direct(x::Real)
    xf = Float64(x)
    xf <= -1.0 && return 0.0

    pts = Float64[-1.0, xf]

    a1 = x1 - RHS_SPLIT_L * delta1
    b1 = x1 + RHS_SPLIT_L * delta1
    -1.0 < a1 < xf && push!(pts, a1)
    -1.0 < x1 < xf && push!(pts, x1)
    -1.0 < b1 < xf && push!(pts, b1)

    a2 = x2 - RHS_SPLIT_L * delta2
    b2 = x2 + RHS_SPLIT_L * delta2
    -1.0 < a2 < xf && push!(pts, a2)
    -1.0 < x2 < xf && push!(pts, x2)
    -1.0 < b2 < xf && push!(pts, b2)

    sort!(pts)

    unique_pts = Float64[]
    sizehint!(unique_pts, length(pts))
    lastp = NaN
    for p in pts
        if isempty(unique_pts) || abs(p - lastp) > 10eps(Float64) * max(1.0, abs(p))
            push!(unique_pts, p)
            lastp = p
        end
    end

    total = 0.0
    @inbounds for j in 1:(length(unique_pts) - 1)
        a = unique_pts[j]
        b = unique_pts[j + 1]
        total += gauss_integral_on_interval(y -> K(xf, y) * u_exact(y), a, b)
    end
    return total
end

function f_exact(x::Real)
    return u_exact(x) + rhs_integral_direct(x)
end

function chebpts_lobatto(N::Int)
    n = N - 1
    n == 0 && return [0.0]
    return [-cospi(j / n) for j in 0:n]
end

function chebcoeffs_from_values(vals_increasing::AbstractVector{<:Real})
    N = length(vals_increasing)
    n = N - 1
    vals_cos_order = reverse(collect(vals_increasing))
    n == 0 && return copy(vals_cos_order)

    c = FFTW.r2r(vals_cos_order, FFTW.REDFT00) ./ n
    c[1] *= 0.5
    c[end] *= 0.5
    return c
end

function cheb_eval_many(c::AbstractVector{<:Real}, x::AbstractVector{<:Real})
    M = length(x)
    N = length(c)
    N == 1 && return fill(c[1], M)

    b1 = zeros(Float64, M)
    b2 = zeros(Float64, M)
    b0 = zeros(Float64, M)

    @inbounds for k in N:-1:2
        ck = c[k]
        for i in 1:M
            b0[i] = 2.0 * x[i] * b1[i] - b2[i] + ck
        end
        b2, b1, b0 = b1, b0, b2
    end

    vals = zeros(Float64, M)
    @inbounds for i in 1:M
        vals[i] = x[i] * b1[i] - b2[i] + c[1]
    end
    return vals
end

function make_test_grid(; Mglobal=8001, Mlocal=5001)
    xg = collect(range(-1.0, 1.0; length=Mglobal))
    xl1 = collect(range(max(-1.0, x1 - 0.04), min(1.0, x1 + 0.04); length=Mlocal))
    xl2 = collect(range(max(-1.0, x2 - 0.05), min(1.0, x2 + 0.05); length=Mlocal))
    return sort(unique(vcat(xg, xl1, xl2, [x1, x2])))
end

const XTEST = make_test_grid()
const UTEST_EXACT = u_exact.(XTEST)
const MAX_ABS_UEXACT = maximum(abs, UTEST_EXACT)

# ============================================================
# VoltMat_Cheb-only build and solve helpers.
# ============================================================

function build_operator(N::Int, tol::Real)
    return VoltMat_Cheb.voltmat_cheb_square(K, N - 1; tol=tol)
end

function build_rhs(N::Int)
    xnodes = chebpts_lobatto(N)
    fvals = f_exact.(xnodes)
    return chebcoeffs_from_values(fvals)
end

function build_system(VK, N::Int)
    return spdiagm(0 => ones(N)) + VK
end

solve_system(Aop, rhs) = Aop \ rhs

function end_to_end_solve(N::Int, tol::Real)
    VK = build_operator(N, tol)
    rhs = build_rhs(N)
    Aop = build_system(VK, N)
    uh_coeff = solve_system(Aop, rhs)
    return VK, rhs, Aop, uh_coeff
end

function matrix_nnz(A)
    return isa(A, SparseMatrixCSC) ? nnz(A) : count(!iszero, A)
end

function compute_accuracy(VK, rhs, Aop, uh_coeff)
    residual = Aop * uh_coeff - rhs
    residual_norm = norm(residual)
    rhs_norm = norm(rhs)
    relative_residual = residual_norm / max(rhs_norm, eps(Float64))

    uh = cheb_eval_many(uh_coeff, XTEST)
    err = uh .- UTEST_EXACT
    max_abs_error = maximum(abs, err)
    relative_error = norm(err) / max(norm(UTEST_EXACT), eps(Float64))

    return (
        residual_norm=residual_norm,
        relative_residual=relative_residual,
        max_abs_error=max_abs_error,
        relative_error=relative_error,
        max_abs_uh=maximum(abs, uh),
        max_abs_uexact=MAX_ABS_UEXACT,
    )
end

# ============================================================
# Benchmark helpers.
# ============================================================

function summarize_trial(trial)
    tmin = minimum(trial)
    tmed = median(trial)
    tmean = mean(trial)
    return (
        min_sec=tmin.time / 1e9,
        median_sec=tmed.time / 1e9,
        mean_sec=tmean.time / 1e9,
        samples=length(trial.times),
        memory=tmin.memory,
        allocs=tmin.allocs,
    )
end

function benchmark_one_N(N::Int)
    # Warmup also provides the objects used for system/solve benchmarks and accuracy.
    GC.gc()
    VK, rhs, Aop, uh_coeff = end_to_end_solve(N, tol)
    acc = compute_accuracy(VK, rhs, Aop, uh_coeff)

    operator_type = string(typeof(VK))
    system_type = string(typeof(Aop))
    operator_nnz = matrix_nnz(VK)
    system_nnz = matrix_nnz(Aop)

    GC.gc()
    bench_build = @benchmark build_operator($N, $tol) samples=3 evals=1 seconds=1000000
    build_stats = summarize_trial(bench_build)

    GC.gc()
    bench_rhs = @benchmark build_rhs($N) samples=3 evals=1 seconds=1000000
    rhs_stats = summarize_trial(bench_rhs)

    GC.gc()
    bench_system = @benchmark build_system($VK, $N) samples=3 evals=1 seconds=1000000
    system_stats = summarize_trial(bench_system)

    GC.gc()
    bench_solve = @benchmark solve_system($Aop, $rhs) samples=3 evals=1 seconds=1000000
    solve_stats = summarize_trial(bench_solve)

    GC.gc()
    bench_end2end = @benchmark end_to_end_solve($N, $tol) samples=3 evals=1 seconds=1000000
    end2end_stats = summarize_trial(bench_end2end)

    return (
        method=METHOD,
        N=N,
        tol=tol,
        equation_sign=EQUATION_SIGN,
        operator_type=operator_type,
        system_type=system_type,
        operator_nnz=operator_nnz,
        system_nnz=system_nnz,
        build_min_sec=build_stats.min_sec,
        build_median_sec=build_stats.median_sec,
        build_mean_sec=build_stats.mean_sec,
        build_samples=build_stats.samples,
        rhs_min_sec=rhs_stats.min_sec,
        rhs_median_sec=rhs_stats.median_sec,
        rhs_mean_sec=rhs_stats.mean_sec,
        rhs_samples=rhs_stats.samples,
        system_min_sec=system_stats.min_sec,
        system_median_sec=system_stats.median_sec,
        system_mean_sec=system_stats.mean_sec,
        system_samples=system_stats.samples,
        solve_min_sec=solve_stats.min_sec,
        solve_median_sec=solve_stats.median_sec,
        solve_mean_sec=solve_stats.mean_sec,
        solve_samples=solve_stats.samples,
        end2end_min_sec=end2end_stats.min_sec,
        end2end_median_sec=end2end_stats.median_sec,
        end2end_mean_sec=end2end_stats.mean_sec,
        end2end_samples=end2end_stats.samples,
        residual_norm=acc.residual_norm,
        relative_residual=acc.relative_residual,
        max_abs_error=acc.max_abs_error,
        relative_error=acc.relative_error,
        max_abs_uh=acc.max_abs_uh,
        max_abs_uexact=acc.max_abs_uexact,
        build_memory=build_stats.memory,
        build_allocs=build_stats.allocs,
        rhs_memory=rhs_stats.memory,
        rhs_allocs=rhs_stats.allocs,
        system_memory=system_stats.memory,
        system_allocs=system_stats.allocs,
        solve_memory=solve_stats.memory,
        solve_allocs=solve_stats.allocs,
        end2end_memory=end2end_stats.memory,
        end2end_allocs=end2end_stats.allocs,
        status="ok",
        error_message="",
    )
end

# ============================================================
# CSV helpers.
# ============================================================

const CSV_COLUMNS = [
    :method,
    :N,
    :tol,
    :equation_sign,
    :operator_type,
    :system_type,
    :operator_nnz,
    :system_nnz,
    :build_min_sec,
    :build_median_sec,
    :build_mean_sec,
    :build_samples,
    :rhs_min_sec,
    :rhs_median_sec,
    :rhs_mean_sec,
    :rhs_samples,
    :system_min_sec,
    :system_median_sec,
    :system_mean_sec,
    :system_samples,
    :solve_min_sec,
    :solve_median_sec,
    :solve_mean_sec,
    :solve_samples,
    :end2end_min_sec,
    :end2end_median_sec,
    :end2end_mean_sec,
    :end2end_samples,
    :residual_norm,
    :relative_residual,
    :max_abs_error,
    :relative_error,
    :max_abs_uh,
    :max_abs_uexact,
    :build_memory,
    :build_allocs,
    :rhs_memory,
    :rhs_allocs,
    :system_memory,
    :system_allocs,
    :solve_memory,
    :solve_allocs,
    :end2end_memory,
    :end2end_allocs,
    :status,
    :error_message,
]

function csv_escape(x)
    s = string(x)
    if occursin(",", s) || occursin("\"", s) || occursin("\n", s) || occursin("\r", s)
        return "\"" * replace(s, "\"" => "\"\"") * "\""
    end
    return s
end

function init_csv(path::AbstractString)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(string.(CSV_COLUMNS), ","))
    end
end

function append_csv(path::AbstractString, row)
    open(path, "a") do io
        println(io, join((csv_escape(getproperty(row, col)) for col in CSV_COLUMNS), ","))
        flush(io)
    end
end

function failure_row(N::Int, err)
    return (
        method=METHOD,
        N=N,
        tol=tol,
        equation_sign=EQUATION_SIGN,
        operator_type="",
        system_type="",
        operator_nnz=missing,
        system_nnz=missing,
        build_min_sec=NaN,
        build_median_sec=NaN,
        build_mean_sec=NaN,
        build_samples=0,
        rhs_min_sec=NaN,
        rhs_median_sec=NaN,
        rhs_mean_sec=NaN,
        rhs_samples=0,
        system_min_sec=NaN,
        system_median_sec=NaN,
        system_mean_sec=NaN,
        system_samples=0,
        solve_min_sec=NaN,
        solve_median_sec=NaN,
        solve_mean_sec=NaN,
        solve_samples=0,
        end2end_min_sec=NaN,
        end2end_median_sec=NaN,
        end2end_mean_sec=NaN,
        end2end_samples=0,
        residual_norm=NaN,
        relative_residual=NaN,
        max_abs_error=NaN,
        relative_error=NaN,
        max_abs_uh=NaN,
        max_abs_uexact=MAX_ABS_UEXACT,
        build_memory=missing,
        build_allocs=missing,
        rhs_memory=missing,
        rhs_allocs=missing,
        system_memory=missing,
        system_allocs=missing,
        solve_memory=missing,
        solve_allocs=missing,
        end2end_memory=missing,
        end2end_allocs=missing,
        status="failed",
        error_message=sprint(showerror, err),
    )
end

# ============================================================
# Main.
# ============================================================

println("Benchmark ex1 VoltMat_Cheb only")
println("Ns = ", Ns)
println("length(Ns) = ", length(Ns))
println("tol = ", tol)
println("Original ex1 equation: ", EQUATION_SIGN)
println("method = ", @which VoltMat_Cheb.voltmat_cheb_square(K, first(Ns) - 1; tol=tol))
println("VoltMat_Cheb defined = ", isdefined(Main, :VoltMat_Cheb))
println("GutlebOlver defined = ", isdefined(Main, :GutlebOlver))
println("BenchmarkTools samples per item = 3")
println("CSV = ", OUTPUT_CSV)
println()

init_csv(OUTPUT_CSV)

@printf("%8s %10s %10s %10s %10s %10s %10s %12s %12s %8s\n",
    "N", "build", "rhs", "system", "solve", "end2end", "nnz(op)", "res", "rel_err", "status")

for Nval in Ns
    row = try
        benchmark_one_N(Nval)
    catch err
        failure_row(Nval, err)
    end

    append_csv(OUTPUT_CSV, row)

    @printf("%8d %10.4f %10.4f %10.4f %10.4f %10.4f %10s %12.3e %12.3e %8s\n",
        row.N,
        row.build_median_sec,
        row.rhs_median_sec,
        row.system_median_sec,
        row.solve_median_sec,
        row.end2end_median_sec,
        string(row.operator_nnz),
        row.residual_norm,
        row.relative_error,
        row.status)
end

println()
println("Wrote ", OUTPUT_CSV)
