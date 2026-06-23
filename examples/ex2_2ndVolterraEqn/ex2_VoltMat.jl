using BenchmarkTools
using FFTW
using InteractiveUtils
using LinearAlgebra
using Printf
using SparseArrays

# This script is intended to live in examples/ex2_2ndVolterraEqn.
# It is VoltMat_Cheb-only and should not include or call GutlebOlver.
root = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(root, "src", "VoltMat", "voltmat_cheb.jl"))

# ============================================================
# User-facing configuration.
# Do not declare tol or method/module name variables as const.
# ============================================================

tol = 1e-14
method = "VoltMat_Cheb"
equation_sign = "u + V_K u = f"

data_dir = joinpath(@__DIR__, "data")
error_csv = joinpath(data_dir, "ex2_VoltMat_error.csv")
time_csv = joinpath(data_dir, "ex2_VoltMat_time.csv")

error_Ns = unique(round.(Int, range(100, 30000; length=100)))
length(error_Ns) == 100 || error("Expected 100 distinct linearly spaced N values, got $(length(error_Ns)): $(error_Ns)")

time_Ns = unique(round.(Int, exp.(range(log(100.0), log(30000.0); length=100))))
time_Ns[1] = 100
time_Ns[end] = 30000
time_Ns = unique(time_Ns)
length(time_Ns) == 100 || error("Expected 100 distinct logarithmically spaced N values, got $(length(time_Ns)): $(time_Ns)")

benchmark_samples = 5
benchmark_evals = 1
benchmark_seconds = 1.0e6

# Fixed RHS projection. The RHS is projected once at this length and then
# truncated or zero-padded for every N. This keeps RHS construction independent
# of the current solve size.
rhs_fixed_N = 20000
rhs_projection_Nmax = rhs_fixed_N

# Mark rows as failed if the solution or diagnostics reach this scale.
instability_threshold = 1.0e100

# ============================================================
# Kernel copied from debug_simple.jl / ex1_aging_maxwell.jl.
# ============================================================

tau_xy(x, y) = 1.5 + x - 0.5 * y

function K(x, y)
    tau = tau_xy(x, y)
    return exp(-(x - y) / tau) / tau
end

# Original ex1 equation:
#     u + V_K u = f
# Hence the discrete system is:
#     Aop = I + VK

# ============================================================
# Exact solution and independent RHS copied from ex1_aging_maxwell.jl.
# ============================================================

function ex1_params()
    return (
        x1=-0.35,
        x2=0.42,
        A1=1.0,
        A2=-0.6,
        delta1=1.0e-3,
        delta2=1.5e-3,
        rhs_q=160,
        rhs_split_l=40.0,
    )
end

step_layer(x, x0, delta) = 0.5 * (1.0 + tanh((x - x0) / delta))

function u_exact(x, p)
    return 1.0 + p.A1 * step_layer(x, p.x1, p.delta1) + p.A2 * step_layer(x, p.x2, p.delta2)
end

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

function gauss_integral_on_interval(f, a::Float64, b::Float64, rule)
    b > a || return 0.0
    gl_x, gl_w = rule
    c = 0.5 * (a + b)
    h = 0.5 * (b - a)
    s = 0.0
    @inbounds for q in eachindex(gl_x)
        y = c + h * gl_x[q]
        s += gl_w[q] * f(y)
    end
    return h * s
end

function rhs_integral_direct(x::Real, p, rule)
    xf = Float64(x)
    xf <= -1.0 && return 0.0

    pts = Float64[-1.0, xf]

    a1 = p.x1 - p.rhs_split_l * p.delta1
    b1 = p.x1 + p.rhs_split_l * p.delta1
    -1.0 < a1 < xf && push!(pts, a1)
    -1.0 < p.x1 < xf && push!(pts, p.x1)
    -1.0 < b1 < xf && push!(pts, b1)

    a2 = p.x2 - p.rhs_split_l * p.delta2
    b2 = p.x2 + p.rhs_split_l * p.delta2
    -1.0 < a2 < xf && push!(pts, a2)
    -1.0 < p.x2 < xf && push!(pts, p.x2)
    -1.0 < b2 < xf && push!(pts, b2)

    sort!(pts)

    unique_pts = Float64[]
    sizehint!(unique_pts, length(pts))
    lastp = NaN
    for pnt in pts
        if isempty(unique_pts) || abs(pnt - lastp) > 10eps(Float64) * max(1.0, abs(pnt))
            push!(unique_pts, pnt)
            lastp = pnt
        end
    end

    total = 0.0
    @inbounds for j in 1:(length(unique_pts) - 1)
        a = unique_pts[j]
        b = unique_pts[j + 1]
        total += gauss_integral_on_interval(y -> K(xf, y) * u_exact(y, p), a, b, rule)
    end
    return total
end

function f_exact(x::Real, p, rule)
    return u_exact(x, p) + rhs_integral_direct(x, p, rule)
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

function make_test_grid(p; Mglobal=8001, Mlocal=5001)
    xg = collect(range(-1.0, 1.0; length=Mglobal))
    xl1 = collect(range(max(-1.0, p.x1 - 0.04), min(1.0, p.x1 + 0.04); length=Mlocal))
    xl2 = collect(range(max(-1.0, p.x2 - 0.05), min(1.0, p.x2 + 0.05); length=Mlocal))
    return sort(unique(vcat(xg, xl1, xl2, [p.x1, p.x2])))
end


# ============================================================
# VoltMat_Cheb-only build and solve helpers.
# ============================================================

function build_operator(N::Int, tol::Real)
    return VoltMat_Cheb.voltmat_cheb_square(K, N - 1; tol=tol)
end

function build_rhs_raw(N::Int, p, rule)
    xnodes = chebpts_lobatto(N)
    fvals = f_exact.(xnodes, Ref(p), Ref(rule))
    rhs = chebcoeffs_from_values(fvals)
    assert_all_finite("rhs_raw", rhs)
    return Vector{Float64}(rhs)
end

function precompute_rhs_cache(p, rule; rhs_fixed_N::Int)
    rhs_fixed_N > 0 || throw(ArgumentError("rhs_fixed_N must be positive"))
    rhs_cache = build_rhs_raw(rhs_fixed_N, p, rule)
    assert_all_finite("rhs_cache", rhs_cache)
    return rhs_cache
end

function build_rhs(N::Int, rhs_cache::AbstractVector{<:Real})
    N > 0 || throw(ArgumentError("N must be positive"))
    fixed_N = length(rhs_cache)

    if N <= fixed_N
        rhs = Vector{Float64}(@view rhs_cache[1:N])
    else
        rhs = zeros(Float64, N)
        rhs[1:fixed_N] .= rhs_cache
    end

    assert_all_finite("rhs", rhs)
    return rhs
end

function rhs_diagnostics(N::Int, rhs::AbstractVector{<:Real}, rhs_cache::AbstractVector{<:Real}, rhs_fixed_precompute_sec::Real)
    fixed_N = length(rhs_cache)
    effective_N = min(N, fixed_N)
    return (
        rhs_projection_Nmax=fixed_N,
        rhs_projection_N=fixed_N,
        rhs_fixed_N=fixed_N,
        rhs_effective_N=effective_N,
        rhs_padding_N=max(N - fixed_N, 0),
        rhs_truncation_N=max(fixed_N - N, 0),
        rhs_fixed_precompute_sec=rhs_fixed_precompute_sec,
        rhs_norm=norm(rhs),
        rhs_max_abs=maximum(abs, rhs),
        rhs_isfinite=all(isfinite, rhs),
    )
end

function build_system(VK, N::Int)
    return spdiagm(0 => ones(N)) + VK
end

solve_system(Aop, rhs) = Aop \ rhs

function end_to_end_solve(N::Int, tol::Real, rhs_cache)
    VK = build_operator(N, tol)
    rhs = build_rhs(N, rhs_cache)
    Aop = build_system(VK, N)
    uh_coeff = solve_system(Aop, rhs)
    assert_all_finite("uh_coeff", uh_coeff)
    return VK, rhs, Aop, uh_coeff
end

function matrix_nnz(A)
    return isa(A, SparseMatrixCSC) ? nnz(A) : count(!iszero, A)
end

function assert_all_finite(label::AbstractString, x)
    if x isa Number
        isfinite(x) || error(label * " is not finite: $(x)")
        return nothing
    end

    all(isfinite, x) || error(label * " contains NaN or Inf")
    return nothing
end

function assert_reasonable_scalar(label::AbstractString, x; threshold::Real=instability_threshold)
    assert_all_finite(label, x)
    abs(float(x)) <= threshold || error(label * " is too large: $(x)")
    return nothing
end

function compute_accuracy(VK, rhs, Aop, uh_coeff, xtest, utest_exact)
    assert_all_finite("rhs", rhs)
    assert_all_finite("uh_coeff", uh_coeff)

    residual = Aop * uh_coeff - rhs
    assert_all_finite("residual", residual)

    residual_norm = norm(residual)
    rhs_norm = norm(rhs)
    relative_residual = residual_norm / max(rhs_norm, eps(Float64))

    uh_coeff_norm = norm(uh_coeff)
    uh_coeff_max_abs = maximum(abs, uh_coeff)
    uh_coeff_isfinite = all(isfinite, uh_coeff)

    assert_reasonable_scalar("uh_coeff_norm", uh_coeff_norm)
    assert_reasonable_scalar("uh_coeff_max_abs", uh_coeff_max_abs)
    assert_reasonable_scalar("residual_norm", residual_norm)

    uh = cheb_eval_many(uh_coeff, xtest)
    assert_all_finite("uh values", uh)

    err = uh .- utest_exact
    assert_all_finite("pointwise error", err)

    max_abs_error = maximum(abs, err)
    relative_error = norm(err) / max(norm(utest_exact), eps(Float64))
    max_abs_uh = maximum(abs, uh)
    max_abs_uexact = maximum(abs, utest_exact)

    assert_reasonable_scalar("max_abs_error", max_abs_error)
    assert_reasonable_scalar("relative_error", relative_error)
    assert_reasonable_scalar("max_abs_uh", max_abs_uh)

    return (
        residual_norm=residual_norm,
        relative_residual=relative_residual,
        max_abs_error=max_abs_error,
        relative_error=relative_error,
        max_abs_uh=max_abs_uh,
        max_abs_uexact=max_abs_uexact,
        uh_coeff_norm=uh_coeff_norm,
        uh_coeff_max_abs=uh_coeff_max_abs,
        uh_coeff_isfinite=uh_coeff_isfinite,
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
        memory=BenchmarkTools.memory(trial),
        allocs=BenchmarkTools.allocs(trial),
    )
end

function run_benchmarkable(benchable; samples::Int, evals::Int, seconds::Real)
    GC.gc()
    return run(benchable; samples=samples, evals=evals, seconds=seconds)
end

function benchmark_one_N(N::Int, tol::Real, rhs_cache, rhs_fixed_precompute_sec; samples::Int, evals::Int, seconds::Real)
    # Warmup provides the objects used for system/solve benchmarks.
    VK, rhs, Aop, _ = end_to_end_solve(N, tol, rhs_cache)

    operator_type = string(typeof(VK))
    system_type = string(typeof(Aop))
    operator_nnz = matrix_nnz(VK)
    system_nnz = matrix_nnz(Aop)
    rhs_diag = rhs_diagnostics(N, rhs, rhs_cache, rhs_fixed_precompute_sec)

    bench_build = run_benchmarkable(
        @benchmarkable build_operator($N, $tol);
        samples=samples,
        evals=evals,
        seconds=seconds,
    )
    build_stats = summarize_trial(bench_build)

    bench_rhs = run_benchmarkable(
        @benchmarkable build_rhs($N, $rhs_cache);
        samples=samples,
        evals=evals,
        seconds=seconds,
    )
    rhs_stats = summarize_trial(bench_rhs)

    bench_system = run_benchmarkable(
        @benchmarkable build_system($VK, $N);
        samples=samples,
        evals=evals,
        seconds=seconds,
    )
    system_stats = summarize_trial(bench_system)

    bench_solve = run_benchmarkable(
        @benchmarkable solve_system($Aop, $rhs);
        samples=samples,
        evals=evals,
        seconds=seconds,
    )
    solve_stats = summarize_trial(bench_solve)

    bench_end2end = run_benchmarkable(
        @benchmarkable end_to_end_solve($N, $tol, $rhs_cache);
        samples=samples,
        evals=evals,
        seconds=seconds,
    )
    end2end_stats = summarize_trial(bench_end2end)

    return (
        method=method,
        N=N,
        tol=tol,
        equation_sign=equation_sign,
        grid_type="logspace",
        benchmark_samples_requested=samples,
        benchmark_evals=evals,
        operator_type=operator_type,
        system_type=system_type,
        operator_nnz=operator_nnz,
        system_nnz=system_nnz,
        rhs_projection_Nmax=rhs_diag.rhs_projection_Nmax,
        rhs_projection_N=rhs_diag.rhs_projection_N,
        rhs_fixed_N=rhs_diag.rhs_fixed_N,
        rhs_effective_N=rhs_diag.rhs_effective_N,
        rhs_padding_N=rhs_diag.rhs_padding_N,
        rhs_truncation_N=rhs_diag.rhs_truncation_N,
        rhs_fixed_precompute_sec=rhs_diag.rhs_fixed_precompute_sec,
        rhs_norm=rhs_diag.rhs_norm,
        rhs_max_abs=rhs_diag.rhs_max_abs,
        rhs_isfinite=rhs_diag.rhs_isfinite,
        build_min_sec=build_stats.min_sec,
        build_median_sec=build_stats.median_sec,
        build_mean_sec=build_stats.mean_sec,
        build_samples=build_stats.samples,
        build_memory=build_stats.memory,
        build_allocs=build_stats.allocs,
        rhs_min_sec=rhs_stats.min_sec,
        rhs_median_sec=rhs_stats.median_sec,
        rhs_mean_sec=rhs_stats.mean_sec,
        rhs_samples=rhs_stats.samples,
        rhs_memory=rhs_stats.memory,
        rhs_allocs=rhs_stats.allocs,
        system_min_sec=system_stats.min_sec,
        system_median_sec=system_stats.median_sec,
        system_mean_sec=system_stats.mean_sec,
        system_samples=system_stats.samples,
        system_memory=system_stats.memory,
        system_allocs=system_stats.allocs,
        solve_min_sec=solve_stats.min_sec,
        solve_median_sec=solve_stats.median_sec,
        solve_mean_sec=solve_stats.mean_sec,
        solve_samples=solve_stats.samples,
        solve_memory=solve_stats.memory,
        solve_allocs=solve_stats.allocs,
        end2end_min_sec=end2end_stats.min_sec,
        end2end_median_sec=end2end_stats.median_sec,
        end2end_mean_sec=end2end_stats.mean_sec,
        end2end_samples=end2end_stats.samples,
        end2end_memory=end2end_stats.memory,
        end2end_allocs=end2end_stats.allocs,
        status="ok",
        error_message="",
    )
end

function solve_error_one_N(N::Int, tol::Real, rhs_cache, rhs_fixed_precompute_sec, xtest, utest_exact)
    VK, rhs, Aop, uh_coeff = end_to_end_solve(N, tol, rhs_cache)
    acc = compute_accuracy(VK, rhs, Aop, uh_coeff, xtest, utest_exact)
    rhs_diag = rhs_diagnostics(N, rhs, rhs_cache, rhs_fixed_precompute_sec)

    return (
        method=method,
        N=N,
        tol=tol,
        equation_sign=equation_sign,
        grid_type="linspace",
        operator_type=string(typeof(VK)),
        system_type=string(typeof(Aop)),
        operator_nnz=matrix_nnz(VK),
        system_nnz=matrix_nnz(Aop),
        rhs_projection_Nmax=rhs_diag.rhs_projection_Nmax,
        rhs_projection_N=rhs_diag.rhs_projection_N,
        rhs_fixed_N=rhs_diag.rhs_fixed_N,
        rhs_effective_N=rhs_diag.rhs_effective_N,
        rhs_padding_N=rhs_diag.rhs_padding_N,
        rhs_truncation_N=rhs_diag.rhs_truncation_N,
        rhs_fixed_precompute_sec=rhs_diag.rhs_fixed_precompute_sec,
        rhs_norm=rhs_diag.rhs_norm,
        rhs_max_abs=rhs_diag.rhs_max_abs,
        rhs_isfinite=rhs_diag.rhs_isfinite,
        residual_norm=acc.residual_norm,
        relative_residual=acc.relative_residual,
        max_abs_error=acc.max_abs_error,
        relative_error=acc.relative_error,
        max_abs_uh=acc.max_abs_uh,
        max_abs_uexact=acc.max_abs_uexact,
        uh_coeff_norm=acc.uh_coeff_norm,
        uh_coeff_max_abs=acc.uh_coeff_max_abs,
        uh_coeff_isfinite=acc.uh_coeff_isfinite,
        status="ok",
        error_message="",
    )
end

# ============================================================
# CSV helpers.
# ============================================================

function csv_escape(x)
    s = string(x)
    if occursin(",", s) || occursin("\"", s) || occursin("\n", s) || occursin("\r", s)
        return "\"" * replace(s, "\"" => "\"\"") * "\""
    end
    return s
end

function init_csv(path::AbstractString, columns)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(string.(columns), ","))
    end
end

function append_csv(path::AbstractString, columns, row)
    open(path, "a") do io
        println(io, join((csv_escape(getproperty(row, col)) for col in columns), ","))
        flush(io)
    end
end

function error_failure_row(N::Int, tol::Real, err)
    return (
        method=method,
        N=N,
        tol=tol,
        equation_sign=equation_sign,
        grid_type="linspace",
        operator_type="",
        system_type="",
        operator_nnz=missing,
        system_nnz=missing,
        rhs_projection_Nmax=rhs_projection_Nmax,
        rhs_projection_N=missing,
        rhs_fixed_N=rhs_fixed_N,
        rhs_effective_N=missing,
        rhs_padding_N=missing,
        rhs_truncation_N=missing,
        rhs_fixed_precompute_sec=NaN,
        rhs_norm=NaN,
        rhs_max_abs=NaN,
        rhs_isfinite=false,
        residual_norm=NaN,
        relative_residual=NaN,
        max_abs_error=NaN,
        relative_error=NaN,
        max_abs_uh=NaN,
        max_abs_uexact=NaN,
        uh_coeff_norm=NaN,
        uh_coeff_max_abs=NaN,
        uh_coeff_isfinite=false,
        status="failed",
        error_message=sprint(showerror, err),
    )
end

function time_failure_row(N::Int, tol::Real, err; samples::Int, evals::Int)
    return (
        method=method,
        N=N,
        tol=tol,
        equation_sign=equation_sign,
        grid_type="logspace",
        benchmark_samples_requested=samples,
        benchmark_evals=evals,
        operator_type="",
        system_type="",
        operator_nnz=missing,
        system_nnz=missing,
        rhs_projection_Nmax=rhs_projection_Nmax,
        rhs_projection_N=missing,
        rhs_fixed_N=rhs_fixed_N,
        rhs_effective_N=missing,
        rhs_padding_N=missing,
        rhs_truncation_N=missing,
        rhs_fixed_precompute_sec=NaN,
        rhs_norm=NaN,
        rhs_max_abs=NaN,
        rhs_isfinite=false,
        build_min_sec=NaN,
        build_median_sec=NaN,
        build_mean_sec=NaN,
        build_samples=0,
        build_memory=missing,
        build_allocs=missing,
        rhs_min_sec=NaN,
        rhs_median_sec=NaN,
        rhs_mean_sec=NaN,
        rhs_samples=0,
        rhs_memory=missing,
        rhs_allocs=missing,
        system_min_sec=NaN,
        system_median_sec=NaN,
        system_mean_sec=NaN,
        system_samples=0,
        system_memory=missing,
        system_allocs=missing,
        solve_min_sec=NaN,
        solve_median_sec=NaN,
        solve_mean_sec=NaN,
        solve_samples=0,
        solve_memory=missing,
        solve_allocs=missing,
        end2end_min_sec=NaN,
        end2end_median_sec=NaN,
        end2end_mean_sec=NaN,
        end2end_samples=0,
        end2end_memory=missing,
        end2end_allocs=missing,
        status="failed",
        error_message=sprint(showerror, err),
    )
end

# ============================================================
# Main sweeps.
# ============================================================

function run_error_sweep(error_Ns, tol, rhs_cache, rhs_fixed_precompute_sec, xtest, utest_exact, error_csv)
    error_columns = [
        :method,
        :N,
        :tol,
        :equation_sign,
        :grid_type,
        :operator_type,
        :system_type,
        :operator_nnz,
        :system_nnz,
        :rhs_projection_Nmax,
        :rhs_projection_N,
        :rhs_fixed_N,
        :rhs_effective_N,
        :rhs_padding_N,
        :rhs_truncation_N,
        :rhs_fixed_precompute_sec,
        :rhs_norm,
        :rhs_max_abs,
        :rhs_isfinite,
        :residual_norm,
        :relative_residual,
        :max_abs_error,
        :relative_error,
        :max_abs_uh,
        :max_abs_uexact,
        :uh_coeff_norm,
        :uh_coeff_max_abs,
        :uh_coeff_isfinite,
        :status,
        :error_message,
    ]

    println()
    println("=== Error sweep ===")
    println("N grid type = linear")
    println("Ns = ", error_Ns)
    println("length(Ns) = ", length(error_Ns))
    println("CSV = ", error_csv)
    init_csv(error_csv, error_columns)

    @printf("%8s %12s %12s %12s %12s %12s %8s\n",
        "N", "max_abs_err", "rel_err", "res", "rhs_norm", "nnz(op)", "status")

    for Nval in error_Ns
        row = try
            solve_error_one_N(Nval, tol, rhs_cache, rhs_fixed_precompute_sec, xtest, utest_exact)
        catch err
            error_failure_row(Nval, tol, err)
        end

        append_csv(error_csv, error_columns, row)

        @printf("%8d %12.3e %12.3e %12.3e %12.3e %12s %8s\n",
            row.N,
            row.max_abs_error,
            row.relative_error,
            row.residual_norm,
            row.rhs_norm,
            string(row.operator_nnz),
            row.status)
    end
end

function run_time_sweep(time_Ns, tol, rhs_cache, rhs_fixed_precompute_sec, time_csv; samples::Int, evals::Int, seconds::Real)
    time_columns = [
        :method,
        :N,
        :tol,
        :equation_sign,
        :grid_type,
        :benchmark_samples_requested,
        :benchmark_evals,
        :operator_type,
        :system_type,
        :operator_nnz,
        :system_nnz,
        :rhs_projection_Nmax,
        :rhs_projection_N,
        :rhs_fixed_N,
        :rhs_effective_N,
        :rhs_padding_N,
        :rhs_truncation_N,
        :rhs_fixed_precompute_sec,
        :rhs_norm,
        :rhs_max_abs,
        :rhs_isfinite,
        :build_min_sec,
        :build_median_sec,
        :build_mean_sec,
        :build_samples,
        :build_memory,
        :build_allocs,
        :rhs_min_sec,
        :rhs_median_sec,
        :rhs_mean_sec,
        :rhs_samples,
        :rhs_memory,
        :rhs_allocs,
        :system_min_sec,
        :system_median_sec,
        :system_mean_sec,
        :system_samples,
        :system_memory,
        :system_allocs,
        :solve_min_sec,
        :solve_median_sec,
        :solve_mean_sec,
        :solve_samples,
        :solve_memory,
        :solve_allocs,
        :end2end_min_sec,
        :end2end_median_sec,
        :end2end_mean_sec,
        :end2end_samples,
        :end2end_memory,
        :end2end_allocs,
        :status,
        :error_message,
    ]

    println()
    println("=== Timing sweep ===")
    println("N grid type = logarithmic")
    println("Ns = ", time_Ns)
    println("length(Ns) = ", length(time_Ns))
    println("BenchmarkTools samples per item = ", samples)
    println("BenchmarkTools evals per sample = ", evals)
    println("CSV = ", time_csv)
    init_csv(time_csv, time_columns)

    @printf("%8s %10s %10s %10s %10s %10s %10s %8s\n",
        "N", "build", "rhs", "system", "solve", "end2end", "nnz(op)", "status")

    for Nval in time_Ns
        row = try
            benchmark_one_N(Nval, tol, rhs_cache, rhs_fixed_precompute_sec; samples=samples, evals=evals, seconds=seconds)
        catch err
            time_failure_row(Nval, tol, err; samples=samples, evals=evals)
        end

        append_csv(time_csv, time_columns, row)

        @printf("%8d %10.4f %10.4f %10.4f %10.4f %10.4f %10s %8s\n",
            row.N,
            row.build_median_sec,
            row.rhs_median_sec,
            row.system_median_sec,
            row.solve_median_sec,
            row.end2end_median_sec,
            string(row.operator_nnz),
            row.status)
    end
end

# ============================================================
# Script entry.
# ============================================================

p = ex1_params()
rule = gausslegendre_rule(p.rhs_q)
xtest = make_test_grid(p)
utest_exact = u_exact.(xtest, Ref(p))

println("ex2 VoltMat_Cheb benchmark")
println("Original ex1 equation: ", equation_sign)
println("tol = ", tol)
println("method = ", @which VoltMat_Cheb.voltmat_cheb_square(K, first(error_Ns) - 1; tol=tol))
println("rhs_fixed_N = ", rhs_fixed_N)
println("VoltMat_Cheb defined = ", isdefined(Main, :VoltMat_Cheb))
println("GutlebOlver defined = ", isdefined(Main, :GutlebOlver))
println("Data directory = ", data_dir)

println()
println("Precomputing fixed RHS cache")
rhs_cache_timed = @timed precompute_rhs_cache(p, rule; rhs_fixed_N=rhs_fixed_N)
rhs_cache = rhs_cache_timed.value
rhs_fixed_precompute_sec = rhs_cache_timed.time
println("rhs_fixed_precompute_sec = ", rhs_fixed_precompute_sec)
println("length(rhs_cache) = ", length(rhs_cache))
println("norm(rhs_cache) = ", norm(rhs_cache))
println("maximum(abs, rhs_cache) = ", maximum(abs, rhs_cache))
println("all finite rhs_cache = ", all(isfinite, rhs_cache))

run_error_sweep(error_Ns, tol, rhs_cache, rhs_fixed_precompute_sec, xtest, utest_exact, error_csv)

run_time_sweep(
    time_Ns,
    tol,
    rhs_cache,
    rhs_fixed_precompute_sec,
    time_csv;
    samples=benchmark_samples,
    evals=benchmark_evals,
    seconds=benchmark_seconds,
)

println()
println("Wrote error CSV: ", error_csv)
println("Wrote timing CSV: ", time_csv)
