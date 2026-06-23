using BenchmarkTools
using LinearAlgebra
using Printf
using SparseArrays

# This script is intended to live in examples/ex2_2ndVolterraEqn.
# It is GutlebOlver-only and should not include or call VoltMat_Cheb.
example_dir = @__DIR__
root = normpath(joinpath(example_dir, "..", ".."))
go_project = joinpath(root, "others", "GutlebOlver")
pushfirst!(LOAD_PATH, go_project)

using ApproxFun
using BandedMatrices
using GutlebOlver

# ============================================================
# User-facing configuration.
# Keep tol and method/module-name variables as ordinary bindings.
# ============================================================

tol = 1e-14
method = "GutlebOlver"
equation_sign = "u + V_K u = f"
kerneldepth = 32

# RHS quadrature settings copied from the verified GO debug script.
rhs_q = 160
rhs_split_L = 40.0

# Fixed RHS projection. The RHS is projected once at this length and then
# truncated or zero-padded for every N. This avoids unstable high-order RHS
# projection when N is very large.
rhs_fixed_N = 20000
rhs_projection_Nmax = rhs_fixed_N

# Mark rows as failed if the solution or diagnostics reach this scale.
instability_threshold = 1.0e100

data_dir = joinpath(@__DIR__, "data")
error_csv = joinpath(data_dir, "ex2_GO_error.csv")
time_csv = joinpath(data_dir, "ex2_GO_time.csv")

# Error sweep: 100 linearly spaced N values.
error_Ns = unique(round.(Int, range(100, 30000; length=100)))
length(error_Ns) == 100 || error("Expected 100 distinct linearly spaced N values, got $(length(error_Ns)): $(error_Ns)")

# Timing sweep: 100 logarithmically spaced N values.
time_Ns = unique(round.(Int, exp.(range(log(100.0), log(30000.0); length=100))))
time_Ns[1] = 100
time_Ns[end] = 30000
time_Ns = unique(time_Ns)
length(time_Ns) == 100 || error("Expected 100 distinct logarithmically spaced N values, got $(length(time_Ns)): $(time_Ns)")

# Multiple BenchmarkTools samples reduce timing fluctuations.
benchmark_samples = 5
benchmark_evals = 1
benchmark_seconds = 1.0e6

# ============================================================
# Same aging Maxwell problem as the verified VoltMat_Cheb script.
# Original ex1 equation:
#     u + V_K u = f
# After mapping x = 2s - 1:
#     u_go(s) + int_0^s K_go(s,r) u_go(r) dr = rhs_go(s)
# ============================================================

tau_xy(x, y) = 1.5 + x - 0.5 * y

function K(x, y)
    tau = tau_xy(x, y)
    return exp(-(x - y) / tau) / tau
end

step_layer(x, x0, delta) = 0.5 * (1.0 + tanh((x - x0) / delta))

function ex1_params()
    return (
        x1=-0.35,
        x2=0.42,
        A1=1.0,
        A2=-0.6,
        delta1=1.0e-3,
        delta2=1.5e-3,
        rhs_q=rhs_q,
        rhs_split_L=rhs_split_L,
    )
end

function u_exact(x, p)
    return 1.0 + p.A1 * step_layer(x, p.x1, p.delta1) + p.A2 * step_layer(x, p.x2, p.delta2)
end

x_from_s(s) = 2.0 * s - 1.0
s_from_x(x) = 0.5 * (x + 1.0)
u_go(s, p) = u_exact(x_from_s(s), p)
K_go(s, r) = 2.0 * K(x_from_s(s), x_from_s(r))

# ============================================================
# Independent RHS construction.
# ============================================================

function gausslegendre_rule(n::Int)
    n <= 0 && throw(ArgumentError("n must be positive"))
    if n == 1
        return [0.0], [2.0]
    end

    beta = [k / sqrt(4.0 * k^2 - 1.0) for k in 1:(n - 1)]
    E = eigen(SymTridiagonal(zeros(n), beta))
    perm = sortperm(E.values)
    return collect(E.values[perm]), (2.0 .* abs2.(E.vectors[1, :]))[perm]
end

function gauss_integral_on_interval(f, a::Float64, b::Float64, gl_x, gl_w)
    b > a || return 0.0

    c = 0.5 * (a + b)
    h = 0.5 * (b - a)
    acc = 0.0

    @inbounds for q in eachindex(gl_x)
        acc += gl_w[q] * f(c + h * gl_x[q])
    end

    return h * acc
end

function rhs_integral_go(s::Real, p, gl_x, gl_w)
    sf = Float64(s)
    sf <= 0.0 && return 0.0

    pts = Float64[0.0, sf]
    for (x0, delta) in ((p.x1, p.delta1), (p.x2, p.delta2))
        for xsplit in (x0 - p.rhs_split_L * delta, x0, x0 + p.rhs_split_L * delta)
            ssplit = s_from_x(xsplit)
            0.0 < ssplit < sf && push!(pts, ssplit)
        end
    end

    sort!(pts)

    unique_pts = Float64[]
    sizehint!(unique_pts, length(pts))
    lastp = NaN

    for pt in pts
        if isempty(unique_pts) || abs(pt - lastp) > 10eps(Float64) * max(1.0, abs(pt))
            push!(unique_pts, pt)
            lastp = pt
        end
    end

    total = 0.0
    @inbounds for j in 1:(length(unique_pts) - 1)
        a = unique_pts[j]
        b = unique_pts[j + 1]
        total += gauss_integral_on_interval(r -> K_go(sf, r) * u_go(r, p), a, b, gl_x, gl_w)
    end

    return total
end

function make_rhs_go(p, gl_x, gl_w)
    return s -> u_go(s, p) + rhs_integral_go(s, p, gl_x, gl_w)
end

function make_test_grid(p; Mglobal=8001, Mlocal=5001)
    xg = collect(range(-1.0, 1.0; length=Mglobal))
    xl1 = collect(range(max(-1.0, p.x1 - 0.04), min(1.0, p.x1 + 0.04); length=Mlocal))
    xl2 = collect(range(max(-1.0, p.x2 - 0.05), min(1.0, p.x2 + 0.05); length=Mlocal))
    return s_from_x.(sort(unique(vcat(xg, xl1, xl2, [p.x1, p.x2]))))
end

# ============================================================
# GutlebOlver-only build and solve helpers.
# ============================================================

function build_operator(N::Int, kerneldepth::Int)
    plan = precompute_triangular_kernel_plan(kerneldepth, true)
    coeffs = triangular_kernel_coeffs_from_plan(plan, K_go)
    Vraw = triVolterraFullKernelOpP01_strictband_from_coeffs(coeffs, N, true)
    return build_secondkind_volterra_strictband(Vraw, N)
end

function build_rhs_raw(N::Int, rhs_fn)
    rhs_coeff, _ = build_go_rhs_coefficients(
        rhs_fn,
        N;
        alpha=0,
        beta=1,
        domain=0..1,
        method=:fixed_gauss_jacobi,
        validate=false,
    )
    return Vector{Float64}(rhs_coeff)
end

function precompute_rhs_cache(rhs_fn; rhs_fixed_N::Int)
    rhs_fixed_N > 0 || throw(ArgumentError("rhs_fixed_N must be positive"))
    rhs_cache = build_rhs_raw(rhs_fixed_N, rhs_fn)
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
    Aop = copy(VK)
    @inbounds for i in 1:N
        Aop[i, i] += 1.0
    end
    return Aop
end

solve_system(Aop, rhs) = Aop \ rhs

function end_to_end_solve(N::Int, kerneldepth::Int, rhs_cache)
    VK = build_operator(N, kerneldepth)
    rhs = build_rhs(N, rhs_cache)
    Aop = build_system(VK, N)
    uh_coeff = solve_system(Aop, rhs)
    assert_all_finite("uh_coeff", uh_coeff)
    return VK, rhs, Aop, uh_coeff
end

function matrix_nnz(A)
    try
        return safe_nnz(A)
    catch
        # Continue to generic fallbacks below.
    end

    try
        return nnz(A)
    catch
        # Continue to count fallback below.
    end

    try
        return count(!iszero, A)
    catch
        return missing
    end
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

function compute_accuracy(VK, rhs, Aop, uh_coeff, go_space, stest, utest_exact)
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

    uh = Fun(go_space, uh_coeff).(stest)
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

function benchmark_one_N(N::Int, kerneldepth::Int, rhs_cache, rhs_fixed_precompute_sec; samples::Int, evals::Int, seconds::Real)
    # Warmup provides the objects used for system/solve benchmarks.
    VK, rhs, Aop, _ = end_to_end_solve(N, kerneldepth, rhs_cache)

    operator_type = string(typeof(VK))
    system_type = string(typeof(Aop))
    operator_nnz = matrix_nnz(VK)
    system_nnz = matrix_nnz(Aop)
    rhs_diag = rhs_diagnostics(N, rhs, rhs_cache, rhs_fixed_precompute_sec)

    bench_build = run_benchmarkable(
        @benchmarkable build_operator($N, $kerneldepth);
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
        @benchmarkable end_to_end_solve($N, $kerneldepth, $rhs_cache);
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
        kerneldepth=kerneldepth,
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

function solve_error_one_N(N::Int, kerneldepth::Int, rhs_cache, rhs_fixed_precompute_sec, go_space, stest, utest_exact)
    VK, rhs, Aop, uh_coeff = end_to_end_solve(N, kerneldepth, rhs_cache)
    acc = compute_accuracy(VK, rhs, Aop, uh_coeff, go_space, stest, utest_exact)
    rhs_diag = rhs_diagnostics(N, rhs, rhs_cache, rhs_fixed_precompute_sec)

    return (
        method=method,
        N=N,
        tol=tol,
        equation_sign=equation_sign,
        grid_type="linspace",
        kerneldepth=kerneldepth,
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

function error_failure_row(N::Int, tol::Real, kerneldepth::Int, err)
    return (
        method=method,
        N=N,
        tol=tol,
        equation_sign=equation_sign,
        grid_type="linspace",
        kerneldepth=kerneldepth,
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

function time_failure_row(N::Int, tol::Real, kerneldepth::Int, err; samples::Int, evals::Int)
    return (
        method=method,
        N=N,
        tol=tol,
        equation_sign=equation_sign,
        grid_type="logspace",
        kerneldepth=kerneldepth,
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

function run_error_sweep(error_Ns, tol, kerneldepth, rhs_cache, rhs_fixed_precompute_sec, go_space, stest, utest_exact, error_csv)
    error_columns = [
        :method,
        :N,
        :tol,
        :equation_sign,
        :grid_type,
        :kerneldepth,
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
            solve_error_one_N(Nval, kerneldepth, rhs_cache, rhs_fixed_precompute_sec, go_space, stest, utest_exact)
        catch err
            error_failure_row(Nval, tol, kerneldepth, err)
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

function run_time_sweep(time_Ns, tol, kerneldepth, rhs_cache, rhs_fixed_precompute_sec, time_csv; samples::Int, evals::Int, seconds::Real)
    time_columns = [
        :method,
        :N,
        :tol,
        :equation_sign,
        :grid_type,
        :kerneldepth,
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
            benchmark_one_N(Nval, kerneldepth, rhs_cache, rhs_fixed_precompute_sec; samples=samples, evals=evals, seconds=seconds)
        catch err
            time_failure_row(Nval, tol, kerneldepth, err; samples=samples, evals=evals)
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
gl_x, gl_w = gausslegendre_rule(p.rhs_q)
rhs_fn = make_rhs_go(p, gl_x, gl_w)
go_space = Jacobi(0, 1, 0..1)
stest = make_test_grid(p)
utest_exact = u_go.(stest, Ref(p))

println("ex2 GutlebOlver benchmark")
println("Original ex1 equation: ", equation_sign)
println("tol = ", tol)
println("method = ", method)
println("kerneldepth = ", kerneldepth)
println("rhs_q = ", p.rhs_q)
println("rhs_split_L = ", p.rhs_split_L)
println("rhs_fixed_N = ", rhs_fixed_N)
println("VoltMat_Cheb defined = ", isdefined(Main, :VoltMat_Cheb))
println("GutlebOlver defined = ", isdefined(Main, :GutlebOlver))
println("Data directory = ", data_dir)

println()
println("Precomputing fixed RHS cache")
rhs_cache_timed = @timed precompute_rhs_cache(rhs_fn; rhs_fixed_N=rhs_fixed_N)
rhs_cache = rhs_cache_timed.value
rhs_fixed_precompute_sec = rhs_cache_timed.time
println("rhs_fixed_precompute_sec = ", rhs_fixed_precompute_sec)
println("length(rhs_cache) = ", length(rhs_cache))
println("norm(rhs_cache) = ", norm(rhs_cache))
println("maximum(abs, rhs_cache) = ", maximum(abs, rhs_cache))
println("all finite rhs_cache = ", all(isfinite, rhs_cache))

run_error_sweep(error_Ns, tol, kerneldepth, rhs_cache, rhs_fixed_precompute_sec, go_space, stest, utest_exact, error_csv)

run_time_sweep(
    time_Ns,
    tol,
    kerneldepth,
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
