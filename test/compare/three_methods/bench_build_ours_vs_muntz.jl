# Build-time benchmark for the coefficient-space weak Volterra matrix and
# the Muntz/fractional Jacobi collocation matrix.
#
# This script measures only operator/matrix construction. It does not include
# RHS projection, linear solves, interpolation, error evaluation, or plotting.
#
# Run:
#   julia --project=. test/compare/three_methods/bench_build_ours_vs_muntz.jl
#   julia --project=. test/compare/three_methods/plot_build_ours_vs_muntz.jl
#
# Quick PowerShell run:
#   $env:N_MAX="512"; $env:MUNTZ_N_MAX="256";
#   julia --project=. test/compare/three_methods/bench_build_ours_vs_muntz.jl
#
# Full-range PowerShell run:
#   $env:N_MAX="10000"; $env:OURS_N_MAX="10000"; $env:MUNTZ_N_MAX="2048";
#   julia --project=. test/compare/three_methods/bench_build_ours_vs_muntz.jl

using FastGaussQuadrature
using LinearAlgebra
using Printf
using Statistics

function find_project_root(start_dir::AbstractString = @__DIR__)
    dir = abspath(start_dir)
    while true
        if isfile(joinpath(dir, "Project.toml")) && isdir(joinpath(dir, "src"))
            return dir
        end
        parent = dirname(dir)
        parent == dir && error("Could not find project root from $(start_dir).")
        dir = parent
    end
end

const SCRIPT_DIR = @__DIR__
const PROJECT_ROOT = find_project_root(SCRIPT_DIR)
const DATA_DIR = joinpath(SCRIPT_DIR, "data")
const FIGURES_DIR = joinpath(SCRIPT_DIR, "figures")
mkpath(DATA_DIR)
mkpath(FIGURES_DIR)

include(joinpath(PROJECT_ROOT, "src", "VoltOpt.jl"))
using .VoltOpt

const WeakCheb = VoltOpt.Weak_VoltMat_Cheb

const MU = parse(Float64, get(ENV, "MU", "0.5"))
const BETA = parse(Float64, get(ENV, "BETA", "0.5"))
const SIGMA = parse(Float64, get(ENV, "SIGMA", "0.0"))
const MUNTZ_LAMBDA = parse(Float64, get(ENV, "MUNTZ_LAMBDA", "0.5"))
const MUNTZ_ALPHA = parse(Float64, get(ENV, "MUNTZ_ALPHA", "-0.5"))
const MUNTZ_BETA = parse(Float64, get(ENV, "MUNTZ_BETA", "-0.5"))

const N_MAX = parse(Int, get(ENV, "N_MAX", "10000"))
const MUNTZ_N_MAX = parse(Int, get(ENV, "MUNTZ_N_MAX", "2048"))
const OURS_N_MAX = parse(Int, get(ENV, "OURS_N_MAX", string(N_MAX)))
const OURS_DENSE_N_MAX = parse(Int, get(ENV, "OURS_DENSE_N_MAX", string(min(128, OURS_N_MAX))))
const FULL_OURS_RANGE = lowercase(strip(get(ENV, "FULL_OURS_RANGE", "0"))) in ("1", "true", "yes", "on")
const MAX_MUNTZ_SECONDS_PER_N = parse(Float64, get(ENV, "MAX_MUNTZ_SECONDS_PER_N", "120.0"))
const MAX_MEMORY_GB = parse(Float64, get(ENV, "MAX_MEMORY_GB", "8.0"))
const REPEATS_SMALL = parse(Int, get(ENV, "REPEATS_SMALL", "5"))
const REPEATS_LARGE = parse(Int, get(ENV, "REPEATS_LARGE", "3"))
const LARGE_N_THRESHOLD = parse(Int, get(ENV, "LARGE_N_THRESHOLD", "256"))

const KERNEL_C_ROWS = parse(Int, get(ENV, "KERNEL_C_ROWS", "61"))
const KERNEL_C_COLS = parse(Int, get(ENV, "KERNEL_C_COLS", "41"))
const KERNEL_I = KERNEL_C_ROWS - 1
const KERNEL_J = KERNEL_C_COLS - 1
const OUR_OUTPUT_FORMATS_RAW = get(ENV, "OURS_OUTPUT_FORMATS", "packed_lower_hessenberg,dense")

const MUNTZ_QUAD_MODE = lowercase(strip(get(ENV, "MUNTZ_QUAD_MODE", "N")))
const MUNTZ_QUAD_ORDER_FIXED = parse(Int, get(ENV, "MUNTZ_QUAD_ORDER", "192"))

const RESULTS_CSV = joinpath(DATA_DIR, "build_ours_vs_muntz_results.csv")
const SUMMARY_CSV = joinpath(DATA_DIR, "build_ours_vs_muntz_summary.csv")

Kx(x, y) = inv((Float64(x) + 1.0)^2 + (Float64(y) - 1.0)^2)
Kt(t, s) = Kx(2.0 * Float64(t) - 1.0, 2.0 * Float64(s) - 1.0)

function parse_int_list(raw::AbstractString)
    vals = Int[]
    for piece in split(raw, ",")
        s = strip(piece)
        isempty(s) && continue
        v = tryparse(Int, s)
        v === nothing && error("Invalid integer list entry: $(s)")
        push!(vals, v)
    end
    isempty(vals) && error("List did not contain any integer values.")
    return vals
end

function parse_symbol_list(raw::AbstractString)
    vals = Symbol[]
    for piece in split(raw, ",")
        s = strip(piece)
        isempty(s) && continue
        if s in ("packed", "packed_lower_hessenberg")
            push!(vals, :packed_lower_hessenberg)
        elseif s == "dense"
            push!(vals, :dense)
        else
            error("Unsupported output format $(s). Use packed_lower_hessenberg or dense.")
        end
    end
    return unique(vals)
end

function default_sample_Ns(nmax::Int)
    nmax < 1 && error("N_MAX must be positive.")
    small = collect(1:min(64, nmax))
    if nmax <= 80
        return small
    end
    nlog = unique(round.(Int, exp10.(range(log10(80.0), log10(Float64(nmax)), length = 35))))
    return sort(unique(vcat(small, [n for n in nlog if 1 <= n <= nmax])))
end

function benchmark_Ns()
    explicit = get(ENV, "N_LIST", nothing)
    base = explicit === nothing ? default_sample_Ns(N_MAX) : sort(unique(parse_int_list(explicit)))
    base = sort(unique(vcat(base, [N_MAX, min(OURS_N_MAX, N_MAX), min(MUNTZ_N_MAX, N_MAX)])))
    ours = [n for n in base if n <= OURS_N_MAX]
    if FULL_OURS_RANGE
        ours = sort(unique(vcat(ours, collect(1:OURS_N_MAX))))
    end
    muntz = [n for n in base if n <= MUNTZ_N_MAX]
    return ours, muntz
end

function shifted_gaussjacobi(n::Int, alpha::Real, beta::Real)
    r, w = gaussjacobi(n, Float64(alpha), Float64(beta))
    z = (r .+ 1.0) ./ 2.0
    wshift = w ./ 2.0^(Float64(alpha) + Float64(beta) + 1.0)
    return z, wshift
end

function barycentric_weights(nodes::AbstractVector{<:Real})
    n = length(nodes)
    logs = zeros(Float64, n)
    signs = ones(Float64, n)
    @inbounds for j in 1:n
        xj = Float64(nodes[j])
        logabs = 0.0
        signv = 1.0
        for k in 1:n
            k == j && continue
            d = xj - Float64(nodes[k])
            signv *= sign(d)
            logabs += log(abs(d))
        end
        logs[j] = -logabs
        signs[j] = signv
    end
    maxlog = maximum(logs)
    return signs .* exp.(logs .- maxlog)
end

function barycentric_basis_values!(basis, nodes, weights, x)
    fill!(basis, 0.0)
    xf = Float64(x)
    tol = 256eps(Float64) * max(1.0, abs(xf))
    @inbounds for j in eachindex(nodes)
        if abs(xf - nodes[j]) <= tol
            basis[j] = 1.0
            return basis
        end
    end
    denom = 0.0
    @inbounds for j in eachindex(nodes)
        v = weights[j] / (xf - nodes[j])
        basis[j] = v
        denom += v
    end
    @inbounds for j in eachindex(nodes)
        basis[j] /= denom
    end
    return basis
end

function build_kernel_coefficients()
    stat = @timed begin
        values = WeakCheb.sample_duffy_kernel(Kx, KERNEL_I, KERNEL_J, BETA)
        WeakCheb.cheb_coeffs_lobatto_2d(values)
    end
    return stat.value, stat.time
end

function weak_build(C, N::Int, output_format::Symbol)
    return VoltOpt.weak_voltmat_cheb(
        C;
        C_is_coeffs = true,
        mu = MU,
        sigma = SIGMA,
        beta = BETA,
        N = N,
        output_format = output_format,
        diagnostics = false,
        H_recurrence_mode = :block_i,
        return_info = true,
    )
end

function muntz_qorder(N::Int)
    if MUNTZ_QUAD_MODE in ("n", "dof", "nplus1")
        return N + 1
    elseif MUNTZ_QUAD_MODE == "fixed"
        return MUNTZ_QUAD_ORDER_FIXED
    else
        error("Unsupported MUNTZ_QUAD_MODE=$(MUNTZ_QUAD_MODE). Use N or fixed.")
    end
end

function build_muntz_matrix(N::Int)
    dof = N + 1
    lambda_frac = MUNTZ_LAMBDA
    qorder = muntz_qorder(N)
    z_nodes, _ = shifted_gaussjacobi(dof, MUNTZ_ALPHA, MUNTZ_BETA)
    tnodes = z_nodes .^ inv(lambda_frac)
    bwz = barycentric_weights(z_nodes)
    theta, wq = shifted_gaussjacobi(qorder, -MU, inv(lambda_frac) - 1.0)
    Aop = zeros(Float64, dof, dof)
    basis = zeros(Float64, dof)

    @inbounds for i in 1:dof
        ti = tnodes[i]
        factor = ti^(1.0 - MU) / lambda_frac
        for k in eachindex(theta)
            th = theta[k]
            thpow = th^inv(lambda_frac)
            s = ti * thpow
            denom = 1.0 - th
            ratio = abs(denom) <= sqrt(eps(Float64)) ? inv(lambda_frac) :
                    (1.0 - thpow) / denom
            z = s^lambda_frac
            barycentric_basis_values!(basis, z_nodes, bwz, z)
            coeff = wq[k] * factor * ratio^(-MU) * Kt(ti, s)
            for j in 1:dof
                Aop[i, j] += coeff * basis[j]
            end
        end
    end
    return Aop
end

function repeats_for_N(N::Int)
    return N <= LARGE_N_THRESHOLD ? REPEATS_SMALL : REPEATS_LARGE
end

function bench_call(f; repeats::Int)
    GC.gc()
    times = Float64[]
    bytes = Float64[]
    gctimes = Float64[]
    for _ in 1:repeats
        stat = @timed f()
        push!(times, stat.time)
        push!(bytes, Float64(stat.bytes))
        push!(gctimes, Float64(stat.gctime))
    end
    return (
        time_min = minimum(times),
        time_median = median(times),
        time_mean = mean(times),
        alloc_bytes = minimum(bytes),
        gc_time = mean(gctimes),
        samples = length(times),
    )
end

function dense_matrix_bytes(N::Int)
    return 8.0 * (Float64(N) + 1.0)^2
end

function estimated_ours_bytes(N::Int, output_format::Symbol, info)
    if output_format === :packed_lower_hessenberg && hasproperty(info, :stored_entries)
        return 8.0 * Float64(info.stored_entries)
    end
    return dense_matrix_bytes(N)
end

function estimated_muntz_total_bytes(N::Int)
    return 3.0 * dense_matrix_bytes(N)
end

function csv_cell(x)
    if x isa AbstractFloat
        return isfinite(x) ? @sprintf("%.17e", x) : string(x)
    elseif x isa Symbol
        return String(x)
    else
        return replace(string(x), "," => ";")
    end
end

function write_csv(path::AbstractString, headers, rows)
    open(path, "w") do io
        println(io, join(headers, ","))
        for r in rows
            println(io, join((csv_cell(getfield(r, Symbol(h))) for h in headers), ","))
        end
    end
end

const RESULTS_HEADERS = [
    "method", "N", "dof", "mu", "muntz_lambda",
    "kernel_coeff_rows", "kernel_coeff_cols", "output_format",
    "build_time_min", "build_time_median", "build_time_mean",
    "alloc_bytes", "gc_time", "estimated_matrix_bytes", "estimated_total_bytes",
    "included_kernel_coeff_build", "kernel_coeff_build_time", "notes",
]

const SUMMARY_HEADERS = [
    "method", "fit_N_min", "fit_N_max", "num_fit_points",
    "slope_p", "intercept_a", "r2", "notes",
]

function skipped_row(method, N, output_format, kernel_time, notes;
    estimated_matrix_bytes = dense_matrix_bytes(N),
    estimated_total_bytes = dense_matrix_bytes(N))
    return (
        method = method,
        N = N,
        dof = N + 1,
        mu = MU,
        muntz_lambda = method == "muntz_collocation" ? MUNTZ_LAMBDA : NaN,
        kernel_coeff_rows = KERNEL_C_ROWS,
        kernel_coeff_cols = KERNEL_C_COLS,
        output_format = output_format,
        build_time_min = NaN,
        build_time_median = NaN,
        build_time_mean = NaN,
        alloc_bytes = NaN,
        gc_time = NaN,
        estimated_matrix_bytes = estimated_matrix_bytes,
        estimated_total_bytes = estimated_total_bytes,
        included_kernel_coeff_build = false,
        kernel_coeff_build_time = kernel_time,
        notes = notes,
    )
end

function result_row(method, N, output_format, stats, kernel_time, notes;
    estimated_matrix_bytes = dense_matrix_bytes(N),
    estimated_total_bytes = dense_matrix_bytes(N))
    return (
        method = method,
        N = N,
        dof = N + 1,
        mu = MU,
        muntz_lambda = method == "muntz_collocation" ? MUNTZ_LAMBDA : NaN,
        kernel_coeff_rows = KERNEL_C_ROWS,
        kernel_coeff_cols = KERNEL_C_COLS,
        output_format = output_format,
        build_time_min = stats.time_min,
        build_time_median = stats.time_median,
        build_time_mean = stats.time_mean,
        alloc_bytes = stats.alloc_bytes,
        gc_time = stats.gc_time,
        estimated_matrix_bytes = estimated_matrix_bytes,
        estimated_total_bytes = estimated_total_bytes,
        included_kernel_coeff_build = false,
        kernel_coeff_build_time = kernel_time,
        notes = notes,
    )
end

function fit_slope(rows, method; fit_N_min::Int = 256)
    mrows = [r for r in rows if r.method == method &&
             isfinite(r.build_time_min) && r.build_time_min > 0 && r.N >= fit_N_min]
    if length(mrows) < 2
        return (
            method = method,
            fit_N_min = fit_N_min,
            fit_N_max = isempty(mrows) ? 0 : maximum(r.N for r in mrows),
            num_fit_points = length(mrows),
            slope_p = NaN,
            intercept_a = NaN,
            r2 = NaN,
            notes = "too_few_fit_points",
        )
    end
    sort!(mrows; by = r -> r.N)
    x = log.(Float64[r.N for r in mrows])
    y = log.(Float64[r.build_time_min for r in mrows])
    xbar = mean(x)
    ybar = mean(y)
    denom = sum((xi - xbar)^2 for xi in x)
    p = sum((x[i] - xbar) * (y[i] - ybar) for i in eachindex(x)) / denom
    a = ybar - p * xbar
    yhat = a .+ p .* x
    ss_res = sum((y[i] - yhat[i])^2 for i in eachindex(y))
    ss_tot = sum((yi - ybar)^2 for yi in y)
    r2 = ss_tot == 0.0 ? 1.0 : 1.0 - ss_res / ss_tot
    return (
        method = method,
        fit_N_min = minimum(r.N for r in mrows),
        fit_N_max = maximum(r.N for r in mrows),
        num_fit_points = length(mrows),
        slope_p = p,
        intercept_a = a,
        r2 = r2,
        notes = "",
    )
end

function main()
    println("============================================================")
    println("Build-time benchmark: our coefficient matrix vs Muntz matrix")
    @printf("mu = %.6f, beta = %.6f, sigma = %.6f\n", MU, BETA, SIGMA)
    @printf("kernel coefficient size = %d x %d\n", KERNEL_C_ROWS, KERNEL_C_COLS)
    @printf("Muntz lambda = %.6f, alpha = %.6f, beta = %.6f, quadrature mode = %s\n",
        MUNTZ_LAMBDA, MUNTZ_ALPHA, MUNTZ_BETA, MUNTZ_QUAD_MODE)
    @printf("N_MAX=%d, OURS_N_MAX=%d, MUNTZ_N_MAX=%d, MAX_MEMORY_GB=%.3f\n",
        N_MAX, OURS_N_MAX, MUNTZ_N_MAX, MAX_MEMORY_GB)
    println("Only matrix/operator construction is timed.")
    println("============================================================")

    ours_Ns, muntz_Ns = benchmark_Ns()
    formats = parse_symbol_list(OUR_OUTPUT_FORMATS_RAW)
    @printf("our N count = %d, Muntz N count = %d\n", length(ours_Ns), length(muntz_Ns))
    println("our output formats = ", formats)

    C, kernel_time = build_kernel_coefficients()
    @printf("Fixed kernel coefficients built once in %.6e s; excluded from per-N build timings.\n", kernel_time)

    # JIT warmup outside measured timings.
    try
        weak_build(C, 8, :packed_lower_hessenberg)
        if :dense in formats
            weak_build(C, 8, :dense)
        end
        build_muntz_matrix(8)
    catch err
        @warn "Warmup failed; continuing so failures are visible in rows." exception = (err, catch_backtrace())
    end

    rows = NamedTuple[]

    println()
    println("Running our_coeffspace matrix builds...")
    for fmt in formats
        method = fmt === :packed_lower_hessenberg ? "our_coeffspace_packed" : "our_coeffspace_dense"
        for N in ours_Ns
            dof = N + 1
            if fmt === :dense && N > OURS_DENSE_N_MAX
                push!(rows, skipped_row(method, N, fmt, kernel_time, "skipped_dense_N_guard";
                    estimated_matrix_bytes = dense_matrix_bytes(N),
                    estimated_total_bytes = dense_matrix_bytes(N)))
                continue
            end
            est = dense_matrix_bytes(N)
            if est / 1024.0^3 > MAX_MEMORY_GB
                push!(rows, skipped_row(method, N, fmt, kernel_time, "skipped_memory_guard";
                    estimated_matrix_bytes = est,
                    estimated_total_bytes = est))
                @printf("  %s N=%5d skipped by memory guard\n", method, N)
                continue
            end

            # Per-N warmup, as requested.
            Vwarm, info = weak_build(C, N, fmt)
            est_bytes = estimated_ours_bytes(N, fmt, info)
            stats = bench_call(() -> weak_build(C, N, fmt); repeats = repeats_for_N(N))
            push!(rows, result_row(method, N, fmt, stats, kernel_time, "";
                estimated_matrix_bytes = est_bytes,
                estimated_total_bytes = est_bytes))
            @printf("  %-24s N=%5d dof=%5d min=%.6e median=%.6e alloc=%.3e est=%.3e bytes\n",
                method, N, dof, stats.time_min, stats.time_median, stats.alloc_bytes, est_bytes)
            Vwarm === nothing && println()
        end
    end

    println()
    println("Running Muntz collocation matrix builds...")
    muntz_time_guard = false
    last_muntz_time = 0.0
    for N in muntz_Ns
        est_matrix = dense_matrix_bytes(N)
        est_total = estimated_muntz_total_bytes(N)
        if est_total / 1024.0^3 > MAX_MEMORY_GB
            push!(rows, skipped_row("muntz_collocation", N, "dense", kernel_time, "skipped_memory_guard";
                estimated_matrix_bytes = est_matrix,
                estimated_total_bytes = est_total))
            @printf("  Muntz N=%5d skipped by memory guard\n", N)
            continue
        end
        if muntz_time_guard
            push!(rows, skipped_row("muntz_collocation", N, "dense", kernel_time, "skipped_time_guard";
                estimated_matrix_bytes = est_matrix,
                estimated_total_bytes = est_total))
            continue
        end
        if last_muntz_time > MAX_MUNTZ_SECONDS_PER_N
            muntz_time_guard = true
            push!(rows, skipped_row("muntz_collocation", N, "dense", kernel_time, "skipped_time_guard";
                estimated_matrix_bytes = est_matrix,
                estimated_total_bytes = est_total))
            @printf("  Muntz N=%5d skipped after previous time %.3f s\n", N, last_muntz_time)
            continue
        end

        build_muntz_matrix(N)
        stats = bench_call(() -> build_muntz_matrix(N); repeats = repeats_for_N(N))
        last_muntz_time = stats.time_min
        qorder = muntz_qorder(N)
        notes = "qorder=$(qorder)"
        push!(rows, result_row("muntz_collocation", N, "dense", stats, kernel_time, notes;
            estimated_matrix_bytes = est_matrix,
            estimated_total_bytes = est_total))
        @printf("  %-24s N=%5d dof=%5d q=%5d min=%.6e median=%.6e alloc=%.3e est=%.3e bytes\n",
            "muntz_collocation", N, N + 1, qorder, stats.time_min, stats.time_median,
            stats.alloc_bytes, est_total)
    end

    summary = [
        fit_slope(rows, "our_coeffspace_packed"),
        fit_slope(rows, "our_coeffspace_dense"),
        fit_slope(rows, "muntz_collocation"),
    ]

    write_csv(RESULTS_CSV, RESULTS_HEADERS, rows)
    write_csv(SUMMARY_CSV, SUMMARY_HEADERS, summary)
    println()
    println("Saved CSV:")
    println("  ", RESULTS_CSV)
    println("  ", SUMMARY_CSV)
    println()
    println("Log-log slope fits:")
    for r in summary
        @printf("  %-24s p=%.4f, R2=%.4f, N=[%d,%d], points=%d, notes=%s\n",
            r.method, r.slope_p, r.r2, r.fit_N_min, r.fit_N_max, r.num_fit_points, r.notes)
    end

    return (rows = rows, summary = summary)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
