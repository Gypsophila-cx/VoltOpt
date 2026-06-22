# ex7_FracCollocation.jl
#
# Solve the weakly singular Volterra integral equation used by the current
# Cauchy-test scripts in examples/ex7_weaksingular:
#
#   u(x) + lambda0 * int_{-1}^x (x-y)^(-mu) K(x,y) u(y) dy = g(x),
#   x in [-1, 1],
#
# where
#
#   K(x,y)  = 1 / ((x+1)^2 + (y-1)^2),
#   g(x)    = 1 + sin(pi*x) / (1000*x^2 + 1),
#   mu      = 0.5,
#   lambda0 = 0.5.
#
# The FracCollocation package solves equations on [0, 1] in the form
#
#   v(t) = ghat(t) + sigma * int_0^t (t-s)^(-mu) Khat(t,s) v(s) ds.
#
# This script maps x = 2t - 1 and y = 2s - 1, so
#
#   Khat(t,s) = K(2t-1, 2s-1),
#   ghat(t)   = g(2t-1),
#   sigma     = -lambda0 * 2^(1-mu),
#
# because the original equation has the Volterra term on the left-hand side.
# No exact solution is available for this simple-RHS variant, so the reported
# errors are Cauchy errors against N_ref = max(N+1, round(Int, 1.01*N)).
# Note: the older ex7_weaksingular.jl also contains a manufactured RHS with an
# exact solution. This script follows the simple-RHS Cauchy-test variant used by
# ex7_collocation.jl, ex7_collocation_fast.jl, ex7_highorder.jl, and
# ex7_weaksingular1.jl for comparability with the current data files here.
#
# Method:
#   Fractional Jacobi / Muntz collocation method implemented in
#   others/FracCollocation.
#
# Convention:
#   In this script, N denotes the number of collocation nodes, the solution
#   vector length, and the quadrature node number.
#
# Fractional collocation parameters:
#   lambda = 0.5 is the Muntz/fractional approximation-space parameter, and
#   alpha = beta = -0.5 are the Jacobi parameters.
#
# Output:
#   examples/ex7_weaksingular/data/ex7_FracCollocation_error.csv
#   examples/ex7_weaksingular/data/ex7_FracCollocation_timing.csv
#
# Run from the VoltOpt project root:
#   julia --project=. examples/ex7_weaksingular/ex7_FracCollocation.jl
#
# Or run from this folder:
#   julia --project=../.. ex7_FracCollocation.jl
#
# Run modes:
#   EX7_RUN_MODE=error   only refresh the error CSV
#   EX7_RUN_MODE=timing  only refresh the timing CSV
#   EX7_RUN_MODE=both    refresh error first, then timing
#
# Timing / benchmarking:
#   The timing mode uses BenchmarkTools to estimate build and solve times by
#   repeated sampling on log-spaced N values. Timing CSV columns report min,
#   median, and mean times in seconds. Error CSV files do not contain timing
#   statistics.
#
#   Environment variables:
#     EX7_BENCH_SAMPLES=5        number of benchmark samples
#     EX7_BENCH_EVALS=1          evaluations per sample
#
#   Example quick benchmark run:
#     EX7_RUN_MODE=timing EX7_BENCH_SAMPLES=3 EX7_TIMING_NMAX=100 julia --project=. examples/ex7_weaksingular/ex7_FracCollocation.jl
#
#   Windows PowerShell example:
#     $env:EX7_BENCH_SAMPLES="3"
#     $env:EX7_RUN_MODE="timing"
#     $env:EX7_TIMING_NMAX="100"
#     julia --project=. examples/ex7_weaksingular/ex7_FracCollocation.jl
#     Remove-Item Env:\EX7_BENCH_SAMPLES
#     Remove-Item Env:\EX7_RUN_MODE
#     Remove-Item Env:\EX7_TIMING_NMAX
#
# Only compute errors:
#   $env:EX7_RUN_MODE="error"
#   julia --project=. examples/ex7_weaksingular/ex7_FracCollocation.jl
#
# Only benchmark timing:
#   $env:EX7_RUN_MODE="timing"
#   $env:EX7_TIMING_NMIN="10"
#   $env:EX7_TIMING_NMAX="1500"
#   $env:EX7_TIMING_NPTS="100"
#   $env:EX7_BENCH_SAMPLES="5"
#   julia --project=. examples/ex7_weaksingular/ex7_FracCollocation.jl
#
# Compute both:
#   $env:EX7_RUN_MODE="both"
#   julia --project=. examples/ex7_weaksingular/ex7_FracCollocation.jl
#
# Environment controls:
#   EX7_FRAC_NMAX   controls the largest N, default 1500.
#   EX7_FRAC_STEP   controls the N step, default 10.
#   EX7_FRAC_COND   controls whether to compute cond(sys.A), default false.
#   EX7_FRAC_RESUME controls whether to skip existing successful N, default true.
#   EX7_FRAC_GRID_M controls the fixed check-grid size, default 2001.

using LinearAlgebra
using Printf
using BenchmarkTools
using Statistics: mean, median

script_dir = @__DIR__
project_root = normpath(joinpath(script_dir, "..", ".."))
frac_src = joinpath(project_root, "others", "FracCollocation", "src", "FracCollocation.jl")
include(frac_src)
using .FracCollocation

function kernel_xy(x::Real, y::Real)
    xf = Float64(x)
    yf = Float64(y)
    return 1.0 / ((xf + 1.0)^2 + (yf - 1.0)^2)
end

function rhs_x(x::Real)
    xf = Float64(x)
    return 1.0 + sin(pi * xf) / (1000.0 * xf^2 + 1.0)
end

function mapped_kernel(t::Real, s::Real)
    return kernel_xy(2.0 * Float64(t) - 1.0, 2.0 * Float64(s) - 1.0)
end

function mapped_rhs(t::Real)
    return rhs_x(2.0 * Float64(t) - 1.0)
end

function make_problem(mu, lambda, alpha, beta, lambda0)
    return FracVIEProblem(
        mu = mu,
        lambda = lambda,
        alpha = alpha,
        beta = beta,
        K = mapped_kernel,
        g = mapped_rhs,
        sigma = -lambda0 * 2.0^(1.0 - mu),
    )
end

function build_and_solve(prob::FracVIEProblem, N::Int)
    sys = nothing
    build_time = @elapsed begin
        sys = build_system(prob, N)
    end

    values = nothing
    solve_time = @elapsed begin
        values = sys.A \ sys.rhs
    end

    sol = FracCollocationSolution(
        sys.problem,
        sys,
        sys.x,
        sys.z,
        Vector{eltype(sys.x)}(values),
        sys.bary,
    )

    return sys, sol, build_time, solve_time
end

function csv_field(x::AbstractString)
    return "\"" * replace(x, "\"" => "\"\"") * "\""
end

csv_field(x::Integer) = string(x)

function csv_field(x::AbstractFloat)
    return isfinite(x) ? @sprintf("%.17e", x) : string(x)
end

csv_field(x::Real) = csv_field(Float64(x))

function write_csv_row(io, row)
    println(io, join(csv_field.(row), ","))
    flush(io)
    return nothing
end

function write_error_header(io)
    println(io,
        "method,N,N_ref,mu,lambda,alpha,beta," *
        "err_nodes_inf,err_grid_inf," *
        "residual_inf,linear_residual_inf," *
        "cond_A,maxabs_V,maxabs_A,maxabs_u,status",
    )
    flush(io)
    return nothing
end

function write_timing_header(io)
    println(io,
        "method,N,mu,lambda,alpha,beta," *
        "build_time_min,build_time_median,build_time_mean," *
        "solve_time_min,solve_time_median,solve_time_mean," *
        "total_time_min,total_time_median,total_time_mean," *
        "bench_samples,bench_evals,status",
    )
    flush(io)
    return nothing
end

function clean_status(status::AbstractString)
    return replace(status, '\n' => ' ', '\r' => ' ')
end

parse_bool_env(name::AbstractString, default::AbstractString) =
    lowercase(get(ENV, name, default)) in ("1", "true", "yes")

function trial_seconds(trial)
    times = trial.times ./ 1.0e9
    return (
        min = minimum(times),
        median = median(times),
        mean = mean(times),
    )
end

elapsed_seconds(t::Real) = (
    min = Float64(t),
    median = Float64(t),
    mean = Float64(t),
)

function benchmark_main_times(
    prob::FracVIEProblem,
    N::Int,
    sys::FracCollocationSystem,
    bench_samples::Int,
    bench_evals::Int,
)
    trial_build = @benchmark build_system($prob, $N) samples=bench_samples evals=bench_evals
    build_stats = trial_seconds(trial_build)

    A_bench = copy(sys.A)
    rhs_bench = copy(sys.rhs)
    trial_solve = @benchmark $A_bench \ $rhs_bench samples=bench_samples evals=bench_evals
    solve_stats = trial_seconds(trial_solve)

    return build_stats, solve_stats
end

function logspace_int_unique(nmin::Int, nmax::Int, npts::Int)
    nmin > 0 || throw(ArgumentError("timing nmin must be positive"))
    nmax >= nmin || throw(ArgumentError("timing nmax must be at least nmin"))
    npts >= 1 || throw(ArgumentError("timing npts must be positive"))

    raw = round.(Int, exp.(range(log(nmin), log(nmax); length = npts)))
    Ns = sort(unique(raw))
    Ns = Ns[(Ns .>= nmin) .& (Ns .<= nmax)]
    isempty(Ns) && (Ns = Int[])
    if isempty(Ns) || first(Ns) != nmin
        pushfirst!(Ns, nmin)
    end
    if last(Ns) != nmax
        push!(Ns, nmax)
    end
    return unique(sort(Ns))
end

function split_csv_line(line::AbstractString)
    fields = String[]
    buf = IOBuffer()
    inquote = false
    i = firstindex(line)
    while i <= lastindex(line)
        c = line[i]
        if c == '"'
            j = nextind(line, i)
            if inquote && j <= lastindex(line) && line[j] == '"'
                print(buf, '"')
                i = nextind(line, j)
            else
                inquote = !inquote
                i = j
            end
        elseif c == ',' && !inquote
            push!(fields, String(take!(buf)))
            i = nextind(line, i)
        else
            print(buf, c)
            i = nextind(line, i)
        end
    end
    push!(fields, String(take!(buf)))
    return fields
end

function csv_column_map(header_line::AbstractString)
    header = split_csv_line(header_line)
    return Dict(name => idx for (idx, name) in pairs(header))
end

function parse_float_field(fields::Vector{String}, colmap, name::AbstractString)
    idx = get(colmap, name, 0)
    if idx == 0 || idx > length(fields)
        return NaN
    end
    return try
        parse(Float64, fields[idx])
    catch
        NaN
    end
end

function read_existing_success_rows(
    csv_path::AbstractString,
    Ns::AbstractVector{Int},
    required_cols,
)
    requested = Set(Ns)
    rows_by_N = Dict{Int,String}()
    success_count = 0
    failed_count = 0
    duplicate_success = 0

    if !isfile(csv_path) || filesize(csv_path) == 0
        return rows_by_N, success_count, failed_count, duplicate_success
    end

    open(csv_path, "r") do io
        eof(io) && return
        colmap = csv_column_map(readline(io))
        n_col = get(colmap, "N", 0)
        status_col = get(colmap, "status", 0)
        (n_col == 0 || status_col == 0) && return
        all(haskey(colmap, c) for c in required_cols) || return

        for line in eachline(io)
            isempty(strip(line)) && continue
            fields = split_csv_line(line)
            length(fields) >= max(n_col, status_col) || continue
            n = try
                parse(Int, fields[n_col])
            catch
                continue
            end
            n in requested || continue

            status = fields[status_col]
            if status == "success"
                success_count += 1
                if haskey(rows_by_N, n)
                    duplicate_success += 1
                else
                    rows_by_N[n] = line
                end
            else
                failed_count += 1
            end
        end
    end

    return rows_by_N, success_count, failed_count, duplicate_success
end

function prepare_csv(
    csv_path::AbstractString,
    Ns::AbstractVector{Int},
    resume::Bool,
    header_writer,
    required_cols,
)
    if !resume
        open(csv_path, "w") do io
            header_writer(io)
        end
        return Set{Int}(), 0, 0, 0
    end

    rows_by_N, success_count, failed_count, duplicate_success =
        read_existing_success_rows(csv_path, Ns, required_cols)

    # Keep one successful row per requested N. Failed rows are intentionally
    # dropped in resume mode so recomputation does not create duplicate N rows.
    open(csv_path, "w") do io
        header_writer(io)
        for N in Ns
            if haskey(rows_by_N, N)
                println(io, rows_by_N[N])
            end
        end
        flush(io)
    end

    return Set(keys(rows_by_N)), success_count, failed_count, duplicate_success
end

function error_failure_row(N, N_ref, mu, lambda, alpha, beta, status)
    return (
        "FracCollocation",
        N,
        N_ref,
        mu,
        lambda,
        alpha,
        beta,
        NaN,
        NaN,
        NaN,
        NaN,
        NaN,
        NaN,
        NaN,
        NaN,
        clean_status(status),
    )
end

function timing_failure_row(N, mu, lambda, alpha, beta, bench_samples, bench_evals, status)
    return (
        "FracCollocation",
        N,
        mu,
        lambda,
        alpha,
        beta,
        NaN,
        NaN,
        NaN,
        NaN,
        NaN,
        NaN,
        NaN,
        NaN,
        NaN,
        bench_samples,
        bench_evals,
        clean_status(status),
    )
end

function csv_summary(csv_path::AbstractString; expect_cond_nan::Bool)
    if !isfile(csv_path) || filesize(csv_path) == 0
        return (
            rows = 0,
            completed_success = 0,
            failed = 0,
            nonfinite = 0,
            duplicate_N = Int[],
            max_success_N = 0,
            bad_success_err_grid = Int[],
            bad_success_residual = Int[],
            bad_success_cond = Int[],
            statuses = String[],
        )
    end

    rows = 0
    completed_success = 0
    failed = 0
    nonfinite = 0
    max_success_N = 0
    seen = Set{Int}()
    duplicate_N = Int[]
    bad_success_err_grid = Int[]
    bad_success_residual = Int[]
    bad_success_cond = Int[]
    statuses = String[]

    open(csv_path, "r") do io
        eof(io) && return
        colmap = csv_column_map(readline(io))
        n_col = get(colmap, "N", 0)
        status_col = get(colmap, "status", 0)
        err_col = get(colmap, "err_grid_inf", 0)
        residual_col = get(colmap, "residual_inf", 0)
        cond_col = get(colmap, "cond_A", 0)

        for line in eachline(io)
            isempty(strip(line)) && continue
            fields = split_csv_line(line)
            length(fields) >= max(n_col, status_col, err_col, residual_col, cond_col) || continue
            rows += 1

            N = try
                parse(Int, fields[n_col])
            catch
                0
            end
            if N != 0
                if N in seen
                    push!(duplicate_N, N)
                else
                    push!(seen, N)
                end
            end

            status = fields[status_col]
            status in statuses || push!(statuses, status)

            if status == "success"
                completed_success += 1
                max_success_N = max(max_success_N, N)

                err_grid = parse_float_field(fields, colmap, "err_grid_inf")
                residual = parse_float_field(fields, colmap, "residual_inf")
                cond_A = parse_float_field(fields, colmap, "cond_A")

                isfinite(err_grid) || push!(bad_success_err_grid, N)
                isfinite(residual) || push!(bad_success_residual, N)
                if expect_cond_nan
                    isnan(cond_A) || push!(bad_success_cond, N)
                else
                    isfinite(cond_A) || push!(bad_success_cond, N)
                end
            elseif startswith(status, "failed:")
                failed += 1
            elseif status == "nonfinite"
                nonfinite += 1
            end
        end
    end

    return (
        rows = rows,
        completed_success = completed_success,
        failed = failed,
        nonfinite = nonfinite,
        duplicate_N = duplicate_N,
        max_success_N = max_success_N,
        bad_success_err_grid = bad_success_err_grid,
        bad_success_residual = bad_success_residual,
        bad_success_cond = bad_success_cond,
        statuses = statuses,
    )
end

function print_summary(summary, csv_path::AbstractString)
    println("------------------------------------------------------------")
    println("CSV summary")
    println("completed_success = ", summary.completed_success)
    println("failed = ", summary.failed)
    println("nonfinite = ", summary.nonfinite)
    println("max_success_N = ", summary.max_success_N)
    println("duplicate_N_count = ", length(summary.duplicate_N))
    println("bad_success_err_grid_count = ", length(summary.bad_success_err_grid))
    println("bad_success_residual_count = ", length(summary.bad_success_residual))
    println("bad_success_cond_count = ", length(summary.bad_success_cond))
    println("statuses = ", join(summary.statuses, "; "))
    println("csv_path = ", csv_path)
    println("------------------------------------------------------------")
    return nothing
end

function timing_summary(csv_path::AbstractString)
    rows = 0
    completed_success = 0
    failed = 0
    nonfinite = 0
    max_success_N = 0
    seen = Set{Int}()
    duplicate_N = Int[]
    bad_time = Int[]
    statuses = String[]

    if !isfile(csv_path) || filesize(csv_path) == 0
        return (; rows, completed_success, failed, nonfinite, max_success_N,
            duplicate_N, bad_time, statuses)
    end

    open(csv_path, "r") do io
        eof(io) && return
        colmap = csv_column_map(readline(io))
        n_col = get(colmap, "N", 0)
        status_col = get(colmap, "status", 0)
        for line in eachline(io)
            isempty(strip(line)) && continue
            fields = split_csv_line(line)
            length(fields) >= max(n_col, status_col) || continue
            rows += 1
            N = try
                parse(Int, fields[n_col])
            catch
                0
            end
            if N != 0
                if N in seen
                    push!(duplicate_N, N)
                else
                    push!(seen, N)
                end
            end

            status = fields[status_col]
            status in statuses || push!(statuses, status)
            if status == "success"
                completed_success += 1
                max_success_N = max(max_success_N, N)
                vals = (
                    parse_float_field(fields, colmap, "build_time_min"),
                    parse_float_field(fields, colmap, "build_time_median"),
                    parse_float_field(fields, colmap, "build_time_mean"),
                    parse_float_field(fields, colmap, "solve_time_min"),
                    parse_float_field(fields, colmap, "solve_time_median"),
                    parse_float_field(fields, colmap, "solve_time_mean"),
                    parse_float_field(fields, colmap, "total_time_min"),
                    parse_float_field(fields, colmap, "total_time_median"),
                    parse_float_field(fields, colmap, "total_time_mean"),
                )
                all(t -> isfinite(t) && t > 0.0, vals) || push!(bad_time, N)
            elseif startswith(status, "failed:")
                failed += 1
            elseif status == "nonfinite"
                nonfinite += 1
            end
        end
    end

    return (; rows, completed_success, failed, nonfinite, max_success_N,
        duplicate_N, bad_time, statuses)
end

function print_timing_summary(summary, csv_path::AbstractString)
    println("------------------------------------------------------------")
    println("Timing CSV summary")
    println("completed_success = ", summary.completed_success)
    println("failed = ", summary.failed)
    println("nonfinite = ", summary.nonfinite)
    println("max_success_N = ", summary.max_success_N)
    println("duplicate_N_count = ", length(summary.duplicate_N))
    println("bad_time_count = ", length(summary.bad_time))
    println("statuses = ", join(summary.statuses, "; "))
    println("csv_path = ", csv_path)
    println("------------------------------------------------------------")
    return nothing
end

function run_error(
    prob,
    Ns_error,
    xgrid,
    csv_path::AbstractString,
    legacy_results_path::AbstractString,
    resume::Bool,
    compute_cond::Bool,
    mu,
    lambda,
    alpha,
    beta,
)
    required_cols = (
        "method", "N", "N_ref", "mu", "lambda", "alpha", "beta",
        "err_nodes_inf", "err_grid_inf", "residual_inf", "linear_residual_inf",
        "cond_A", "maxabs_V", "maxabs_A", "maxabs_u", "status",
    )
    completed_Ns, existing_success, existing_failed, existing_dup_success =
        prepare_csv(csv_path, Ns_error, resume, write_error_header, required_cols)

    println("Error CSV = ", csv_path)
    if resume
        println("error resume existing success rows = ", existing_success,
            ", existing failed/non-success rows dropped = ", existing_failed,
            ", duplicate success rows dropped = ", existing_dup_success)
        println("error remaining cases to compute = ", count(N -> !(N in completed_Ns), Ns_error))
    end

    open(csv_path, "a") do io
        for (idx, N) in enumerate(Ns_error)
            if N in completed_Ns
                @printf("[%3d/%3d] N=%5d skipped: existing error success row\n",
                    idx, length(Ns_error), N)
                continue
            end

            N_ref = max(N + 1, round(Int, 1.01 * N))
            try
                println("Error N = ", N, ", N_ref = ", N_ref)
                sys, sol, build_time, solve_time = build_and_solve(prob, N)

                err_nodes_inf = NaN
                err_grid_inf = NaN
                residual_inf = NaN
                linear_residual_inf = NaN
                cond_A = NaN
                maxabs_V = NaN
                maxabs_A = NaN
                maxabs_u = NaN
                diag_time = @elapsed begin
                    sol_ref = solve(prob, N_ref)
                    err_nodes_inf = maximum(abs.(evaluate(sol, sol_ref.x) .- sol_ref.values))
                    err_grid_inf = maximum(abs.(evaluate(sol, xgrid) .- evaluate(sol_ref, xgrid)))

                    residual_inf = maximum(abs.(collocation_residual(sys, sol.values)))
                    linear_residual_inf = maximum(abs.(sys.A * sol.values .- sys.rhs))
                    cond_A = compute_cond ? cond(sys.A) : NaN
                    maxabs_V = maximum(abs.(sys.V))
                    maxabs_A = maximum(abs.(sys.A))
                    maxabs_u = maximum(abs.(sol.values))
                end

                required_diagnostics = (
                    err_nodes_inf,
                    err_grid_inf,
                    residual_inf,
                    linear_residual_inf,
                    maxabs_V,
                    maxabs_A,
                    maxabs_u,
                )
                cond_ok = !compute_cond || isfinite(cond_A)
                status = all(isfinite, required_diagnostics) && cond_ok ? "success" : "nonfinite"

                write_csv_row(io, (
                    "FracCollocation",
                    N,
                    N_ref,
                    mu,
                    lambda,
                    alpha,
                    beta,
                    err_nodes_inf,
                    err_grid_inf,
                    residual_inf,
                    linear_residual_inf,
                    cond_A,
                    maxabs_V,
                    maxabs_A,
                    maxabs_u,
                    status,
                ))

                @printf("  single build_time = %.6e\n", build_time)
                @printf("  single solve_time = %.6e\n", solve_time)
                @printf("  diag_time = %.6e\n", diag_time)
                @printf("  err_grid_inf = %.6e\n", err_grid_inf)
                @printf("  residual_inf = %.6e\n", residual_inf)
                println("  status = ", status)
            catch err
                status = "failed: " * sprint(showerror, err)
                write_csv_row(io, error_failure_row(N, N_ref, mu, lambda, alpha, beta, status))
                @printf("[%3d/%3d] N=%5d N_ref=%5d ERROR FAILED: %s\n",
                    idx, length(Ns_error), N, N_ref, clean_status(status))
            end
        end
    end

    cp(csv_path, legacy_results_path; force = true)
    println("wrote ", csv_path)
    println("wrote legacy error copy ", legacy_results_path)
    summary = csv_summary(csv_path; expect_cond_nan = !compute_cond)
    print_summary(summary, csv_path)
    return summary
end

function run_timing(
    prob,
    Ns_timing,
    csv_path::AbstractString,
    resume::Bool,
    mu,
    lambda,
    alpha,
    beta,
    bench_samples::Int,
    bench_evals::Int,
)
    required_cols = (
        "method", "N", "mu", "lambda", "alpha", "beta",
        "build_time_min", "build_time_median", "build_time_mean",
        "solve_time_min", "solve_time_median", "solve_time_mean",
        "total_time_min", "total_time_median", "total_time_mean",
        "bench_samples", "bench_evals", "status",
    )
    completed_Ns, existing_success, existing_failed, existing_dup_success =
        prepare_csv(csv_path, Ns_timing, resume, write_timing_header, required_cols)

    println("Timing CSV = ", csv_path)
    println("timing N count = ", length(Ns_timing))
    if resume
        println("timing resume existing success rows = ", existing_success,
            ", existing failed/non-success rows dropped = ", existing_failed,
            ", duplicate success rows dropped = ", existing_dup_success)
        println("timing remaining cases to compute = ", count(N -> !(N in completed_Ns), Ns_timing))
    end

    open(csv_path, "a") do io
        for (idx, N) in enumerate(Ns_timing)
            if N in completed_Ns
                @printf("[%3d/%3d] N=%5d skipped: existing timing success row\n",
                    idx, length(Ns_timing), N)
                continue
            end

            try
                println("Timing N = ", N)
                sys_for_solve = build_system(prob, N)
                build_stats, solve_stats = benchmark_main_times(
                    prob,
                    N,
                    sys_for_solve,
                    bench_samples,
                    bench_evals,
                )
                total_min = build_stats.min + solve_stats.min
                total_median = build_stats.median + solve_stats.median
                total_mean = build_stats.mean + solve_stats.mean
                time_ok = all(t -> isfinite(t) && t > 0.0, (
                    build_stats.min,
                    build_stats.median,
                    build_stats.mean,
                    solve_stats.min,
                    solve_stats.median,
                    solve_stats.mean,
                    total_min,
                    total_median,
                    total_mean,
                ))
                status = time_ok ? "success" : "nonfinite"

                write_csv_row(io, (
                    "FracCollocation",
                    N,
                    mu,
                    lambda,
                    alpha,
                    beta,
                    build_stats.min,
                    build_stats.median,
                    build_stats.mean,
                    solve_stats.min,
                    solve_stats.median,
                    solve_stats.mean,
                    total_min,
                    total_median,
                    total_mean,
                    bench_samples,
                    bench_evals,
                    status,
                ))

                @printf("  build median = %.6e\n", build_stats.median)
                @printf("  solve median = %.6e\n", solve_stats.median)
                @printf("  total median = %.6e\n", total_median)
                println("  status = ", status)
            catch err
                status = "failed: " * sprint(showerror, err)
                write_csv_row(io, timing_failure_row(
                    N,
                    mu,
                    lambda,
                    alpha,
                    beta,
                    bench_samples,
                    bench_evals,
                    status,
                ))
                @printf("[%3d/%3d] N=%5d TIMING FAILED: %s\n",
                    idx, length(Ns_timing), N, clean_status(status))
            end
        end
    end

    println("wrote ", csv_path)
    summary = timing_summary(csv_path)
    print_timing_summary(summary, csv_path)
    return summary
end

function main()
    mu = 0.5
    lambda0 = 0.5
    lambda = 0.5
    alpha = -0.5
    beta = -0.5

    run_mode = lowercase(get(ENV, "EX7_RUN_MODE", "both"))
    run_mode in ("error", "timing", "both") ||
        throw(ArgumentError("EX7_RUN_MODE must be error, timing, or both"))

    nmax = parse(Int, get(ENV, "EX7_FRAC_NMAX", "1500"))
    nstep = parse(Int, get(ENV, "EX7_FRAC_STEP", "10"))
    compute_cond = parse_bool_env("EX7_FRAC_COND", "false")
    resume = parse_bool_env("EX7_FRAC_RESUME", "true")
    grid_m = parse(Int, get(ENV, "EX7_FRAC_GRID_M", "2001"))
    timing_nmin = parse(Int, get(ENV, "EX7_TIMING_NMIN", "10"))
    timing_nmax = parse(Int, get(ENV, "EX7_TIMING_NMAX", "1500"))
    timing_npts = parse(Int, get(ENV, "EX7_TIMING_NPTS", "100"))
    bench_samples = parse(Int, get(ENV, "EX7_BENCH_SAMPLES", "5"))
    bench_evals = parse(Int, get(ENV, "EX7_BENCH_EVALS", "1"))
    nmax >= 10 || throw(ArgumentError("EX7_FRAC_NMAX must be at least 10"))
    nstep > 0 || throw(ArgumentError("EX7_FRAC_STEP must be positive"))
    grid_m >= 2 || throw(ArgumentError("EX7_FRAC_GRID_M must be at least 2"))
    bench_samples > 0 || throw(ArgumentError("EX7_BENCH_SAMPLES must be positive"))
    bench_evals > 0 || throw(ArgumentError("EX7_BENCH_EVALS must be positive"))
    Ns_error = collect(10:nstep:nmax)
    Ns_timing = logspace_int_unique(timing_nmin, timing_nmax, timing_npts)

    data_dir = joinpath(script_dir, "data")
    mkpath(data_dir)
    error_csv_path = joinpath(data_dir, "ex7_FracCollocation_error.csv")
    timing_csv_path = joinpath(data_dir, "ex7_FracCollocation_timing.csv")
    legacy_results_path = joinpath(data_dir, "ex7_FracCollocation_results.csv")
    xgrid = collect(range(0.0, 1.0; length = grid_m))

    prob = make_problem(mu, lambda, alpha, beta, lambda0)

    println("ex7_FracCollocation.jl")
    println("run mode = ", run_mode)
    @printf("mu = %.16g, equation lambda0 = %.16g\n", mu, lambda0)
    @printf("fractional collocation lambda = %.16g, alpha = %.16g, beta = %.16g\n",
        lambda, alpha, beta)
    println("error N range = ", first(Ns_error), ":", nstep, ":", last(Ns_error),
        " (", length(Ns_error), " cases)")
    println("timing N range = ", first(Ns_timing), " to ", last(Ns_timing),
        " (", length(Ns_timing), " log-spaced unique cases)")
    println("grid points = ", grid_m)
    println("compute cond_A = ", compute_cond)
    println("benchmark samples = ", bench_samples, ", evals = ", bench_evals)
    println("resume = ", resume)

    if run_mode in ("error", "both")
        run_error(
            prob,
            Ns_error,
            xgrid,
            error_csv_path,
            legacy_results_path,
            resume,
            compute_cond,
            mu,
            lambda,
            alpha,
            beta,
        )
    end

    if run_mode in ("timing", "both")
        run_timing(
            prob,
            Ns_timing,
            timing_csv_path,
            resume,
            mu,
            lambda,
            alpha,
            beta,
            bench_samples,
            bench_evals,
        )
    end

    return (error_csv = error_csv_path, timing_csv = timing_csv_path)
end

main()
