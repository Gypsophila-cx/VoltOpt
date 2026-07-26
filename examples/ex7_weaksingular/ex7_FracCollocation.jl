# ex7_FracCollocation.jl
#
# Solve the weakly singular Volterra integral equation used by the current
# Cauchy-test scripts in examples/ex7_weaksingular:
#
#   u(x) + 0.5 * int_{-1}^x (x-y)^(-0.5) K(x,y) u(y) dy = g(x),
#   x in [-1, 1],
#
# where
#
#   K(x,y)  = 1 / ((x+1)^2 + (y-1)^2),
#   g(x)    = ((1+x)/2)^(1/3) * (1 + sin(pi*x)/(1000*x^2 + 1)).
#
# The FracCollocation package solves equations on [0, 1] in the form
#
#   v(t) = ghat(t) + sigma * int_0^t (t-s)^(-mu) Khat(t,s) v(s) ds.
#
# This script maps x = 2t - 1 and y = 2s - 1, so
#
#   Khat(t,s) = K(2t-1, 2s-1),
#   ghat(t)   = g(2t-1),
#   sigma     = -0.5 * 2^(1-mu),
#
# because the original equation has the Volterra term on the left-hand side.
# No exact solution is available for this simple-RHS variant, so the reported
# errors are Cauchy errors against Nplus = max(N+1, round(Int, 1.01*N)),
# evaluated on the shared composite physical grid: at least 10001 uniform x
# points plus 5001 uniform z=s^(1/6) points mapped to x.
#
# Method:
#   Muntz collocation method cited as the Hou method in the paper and implemented
#   in others/FracCollocation. Its existing numerical implementation is retained.
#
# Convention:
#   In the Example 7 CSV, N is the highest Muntz power index, so the space is
#   span(1, s^(1/6), ..., s^(N/6)). The FracCollocation core uses its integer
#   argument as the node count, hence this driver consistently passes N+1.
#
# Muntz collocation parameters:
#   The Muntz fractional power is muntz_lambda = 1/6 and both Jacobi parameters
#   are -0.5. The Jacobi weights remain method-specific and are not replaced by
#   the Weak Duffy--CFP basis parameters alpha=1/3 and beta=1/2.
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
#   median, q25, and q75 times in seconds. Error CSV files do not contain
#   timing statistics.
#
#   Environment variables:
#     EX7_BENCH_SAMPLES=5        number of benchmark samples
#     EX7_BENCH_EVALS=1          evaluations per sample
#     EX7_BENCH_SECONDS=120      maximum seconds per benchmark
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
#   EX7_ERROR_NMIN        smallest error N, default 10.
#   EX7_ERROR_NMAX        largest error N, default 2000.
#   EX7_ERROR_STEP        uniform error step, default 10.
#   EX7_ERROR_UNIFORM_POINTS   uniform physical-grid count, default 10001.
#   EX7_ERROR_FRACTIONAL_POINTS uniform z=s^(1/6) count, default 5001.
#   EX7_FRAC_RESUME controls whether to skip existing successful N, default true.

using LinearAlgebra
using Printf
using BenchmarkTools
using Statistics: median, quantile

script_dir = @__DIR__
project_root = normpath(joinpath(script_dir, "..", ".."))
frac_src = joinpath(project_root, "others", "FracCollocation", "src", "FracCollocation.jl")
include(frac_src)
using .FracCollocation
include(joinpath(script_dir, "ex7_common.jl"))
using .Ex7Common

function kernel_xy(x::Real, y::Real)
    xf = Float64(x)
    yf = Float64(y)
    return 1.0 / ((xf + 1.0)^2 + (yf - 1.0)^2)
end

function rhs_x(x::Real)
    xf = Float64(x)
    return ((1.0 + xf) / 2.0)^(1.0 / 3.0) *
           (1.0 + sin(pi * xf) / (1000.0 * xf^2 + 1.0))
end

function mapped_kernel(t::Real, s::Real)
    # Evaluate the algebraically mapped kernel directly in [0,1]
    # coordinates.  At the extreme fractional nodes, forming x=2t-1 and
    # then x+1 loses the small nonzero value of 2t by cancellation.
    tf = Float64(t)
    sf = Float64(s)
    return 1.0 / ((2.0 * tf)^2 + (2.0 * sf - 2.0)^2)
end

function mapped_rhs(t::Real)
    # Keep t itself when evaluating t^(1/3).  For standard Muntz nodes
    # t=z^6 can be below eps(Float64); the old t -> x=2t-1 -> (1+x)/2
    # round trip rounded those nonzero nodes to zero (40 nodes at N=1500),
    # corrupting the endpoint data and producing the apparent error plateau.
    tf = Float64(t)
    x = 2.0 * tf - 1.0
    return tf^(1.0 / 3.0) *
           (1.0 + sin(pi * x) / (1000.0 * x^2 + 1.0))
end

function stable_coordinate_regression_check()
    # This positive Float64 is too small to survive the round trip through
    # x=2s-1.  The stable mapped RHS must nevertheless retain s^(1/3).
    s = ldexp(1.0, -60)
    x = 2.0 * s - 1.0
    @assert s > 0.0
    @assert x == -1.0
    @assert (1.0 + x) / 2.0 == 0.0
    value = mapped_rhs(s)
    @assert isfinite(value) && value > 0.0
    return (s = s, value = value)
end

function make_problem(mu, muntz_lambda, muntz_jacobi_a, muntz_jacobi_b)
    # Use the generated positional constructor to keep the method's exponent
    # clearly separate from the equation, whose integral coefficient is 0.5.
    # The core then uses this same fractional power in its s=z^(1/power) nodes,
    # fractional interpolation, transformed quadrature, and solution evaluation.
    return FracVIEProblem(
        Float64(mu),
        Float64(muntz_lambda),
        Float64(muntz_jacobi_a),
        Float64(muntz_jacobi_b),
        mapped_kernel,
        mapped_rhs,
        -0.5 * 2.0^(1.0 - Float64(mu)),
    )
end

function build_and_solve(prob::FracVIEProblem, N::Int)
    sys = nothing
    build_time = @elapsed begin
        sys = build_system(prob, N + 1)
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
        "method,N,Nplus,mu,alpha,beta,cauchy_error,max_error_x,test_grid_size," *
        "muntz_lambda,muntz_jacobi_a,muntz_jacobi_b," *
        "residual_inf,linear_residual_inf," *
        "maxabs_V,maxabs_A,maxabs_u,status",
    )
    flush(io)
    return nothing
end

function write_timing_header(io)
    println(io,
        "method,N,mu,alpha,beta," *
        "muntz_lambda,muntz_jacobi_a,muntz_jacobi_b," *
        "build_min,build_median,build_q25,build_q75," *
        "solve_min,solve_median,solve_q25,solve_q75," *
        "total_min,total_median,build_samples,solve_samples," *
        "bench_evals,bench_seconds,status",
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
        q25 = quantile(times, 0.25),
        q75 = quantile(times, 0.75),
        samples = length(times),
    )
end

function benchmark_main_times(
    prob::FracVIEProblem,
    N::Int,
    sys::FracCollocationSystem,
    bench_samples::Int,
    bench_evals::Int,
    bench_seconds::Float64,
)
    trial_build = @benchmark build_system($prob, $(N + 1)) samples=bench_samples evals=bench_evals seconds=bench_seconds
    build_stats = trial_seconds(trial_build)

    A_bench = copy(sys.A)
    rhs_bench = copy(sys.rhs)
    trial_solve = @benchmark $A_bench \ $rhs_bench samples=bench_samples evals=bench_evals seconds=bench_seconds
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

function error_failure_row(
    N, Nplus, mu, alpha, beta,
    muntz_lambda, muntz_jacobi_a, muntz_jacobi_b, test_grid_size, status,
)
    return (
        "Müntz--collocation",
        N,
        Nplus,
        mu,
        alpha,
        beta,
        NaN,
        NaN,
        test_grid_size,
        muntz_lambda,
        muntz_jacobi_a,
        muntz_jacobi_b,
        NaN,
        NaN,
        NaN,
        NaN,
        NaN,
        clean_status(status),
    )
end

function timing_failure_row(
    N, mu, alpha, beta,
    muntz_lambda, muntz_jacobi_a, muntz_jacobi_b,
    bench_evals, bench_seconds, status,
)
    return (
        "Müntz--collocation",
        N,
        mu,
        alpha,
        beta,
        muntz_lambda,
        muntz_jacobi_a,
        muntz_jacobi_b,
        NaN,
        NaN,
        NaN,
        NaN,
        NaN,
        NaN,
        NaN,
        NaN,
        NaN,
        NaN,
        0,
        0,
        bench_evals,
        bench_seconds,
        clean_status(status),
    )
end

function csv_summary(csv_path::AbstractString)
    if !isfile(csv_path) || filesize(csv_path) == 0
        return (
            rows = 0,
            completed_success = 0,
            failed = 0,
            nonfinite = 0,
            duplicate_N = Int[],
            max_success_N = 0,
            bad_success_cauchy = Int[],
            bad_success_residual = Int[],
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
    bad_success_cauchy = Int[]
    bad_success_residual = Int[]
    statuses = String[]

    open(csv_path, "r") do io
        eof(io) && return
        colmap = csv_column_map(readline(io))
        n_col = get(colmap, "N", 0)
        status_col = get(colmap, "status", 0)
        err_col = get(colmap, "cauchy_error", 0)
        residual_col = get(colmap, "residual_inf", 0)

        for line in eachline(io)
            isempty(strip(line)) && continue
            fields = split_csv_line(line)
            length(fields) >= max(n_col, status_col, err_col, residual_col) || continue
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

                cauchy_error = parse_float_field(fields, colmap, "cauchy_error")
                residual = parse_float_field(fields, colmap, "residual_inf")

                isfinite(cauchy_error) || push!(bad_success_cauchy, N)
                isfinite(residual) || push!(bad_success_residual, N)
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
        bad_success_cauchy = bad_success_cauchy,
        bad_success_residual = bad_success_residual,
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
    println("bad_success_cauchy_count = ", length(summary.bad_success_cauchy))
    println("bad_success_residual_count = ", length(summary.bad_success_residual))
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
                    parse_float_field(fields, colmap, "build_min"),
                    parse_float_field(fields, colmap, "build_median"),
                    parse_float_field(fields, colmap, "build_q25"),
                    parse_float_field(fields, colmap, "build_q75"),
                    parse_float_field(fields, colmap, "solve_min"),
                    parse_float_field(fields, colmap, "solve_median"),
                    parse_float_field(fields, colmap, "solve_q25"),
                    parse_float_field(fields, colmap, "solve_q75"),
                    parse_float_field(fields, colmap, "total_min"),
                    parse_float_field(fields, colmap, "total_median"),
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
    grid,
    csv_path::AbstractString,
    resume::Bool,
    mu,
    alpha,
    beta,
    muntz_lambda,
    muntz_jacobi_a,
    muntz_jacobi_b,
)
    required_cols = (
        "method", "N", "Nplus", "mu", "alpha", "beta", "cauchy_error",
        "max_error_x", "test_grid_size",
        "muntz_lambda", "muntz_jacobi_a", "muntz_jacobi_b",
        "residual_inf", "linear_residual_inf",
        "maxabs_V", "maxabs_A", "maxabs_u", "status",
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

            Nplus = max(N + 1, round(Int, 1.01 * N))
            try
                println("Error N = ", N, ", Nplus = ", Nplus)
                sys, sol, build_time, solve_time = build_and_solve(prob, N)

                cauchy_error = NaN
                max_error_x = NaN
                residual_inf = NaN
                linear_residual_inf = NaN
                maxabs_V = NaN
                maxabs_A = NaN
                maxabs_u = NaN
                diag_time = @elapsed begin
                    sol_ref = solve(prob, Nplus + 1)
                    # Both methods use this same composite physical grid.  Its
                    # retained s-coordinates avoid a lossy x -> s round trip.
                    differences = abs.(
                        evaluate(sol, grid.s) .- evaluate(sol_ref, grid.s)
                    )
                    cauchy_error, max_idx = findmax(differences)
                    max_error_x = grid.x[max_idx]

                    residual_inf = maximum(abs.(collocation_residual(sys, sol.values)))
                    linear_residual_inf = maximum(abs.(sys.A * sol.values .- sys.rhs))
                    maxabs_V = maximum(abs.(sys.V))
                    maxabs_A = maximum(abs.(sys.A))
                    maxabs_u = maximum(abs.(sol.values))
                end

                required_diagnostics = (
                    cauchy_error,
                    max_error_x,
                    residual_inf,
                    linear_residual_inf,
                    maxabs_V,
                    maxabs_A,
                    maxabs_u,
                )
                status = all(isfinite, required_diagnostics) ? "success" : "nonfinite"

                write_csv_row(io, (
                    "Müntz--collocation",
                    N,
                    Nplus,
                    mu,
                    alpha,
                    beta,
                    cauchy_error,
                    max_error_x,
                    grid.total_count,
                    muntz_lambda,
                    muntz_jacobi_a,
                    muntz_jacobi_b,
                    residual_inf,
                    linear_residual_inf,
                    maxabs_V,
                    maxabs_A,
                    maxabs_u,
                    status,
                ))

                @printf("  single build_time = %.6e\n", build_time)
                @printf("  single solve_time = %.6e\n", solve_time)
                @printf("  diag_time = %.6e\n", diag_time)
                @printf("  cauchy_error = %.6e\n", cauchy_error)
                @printf("  max_error_x = %.17e\n", max_error_x)
                @printf("  residual_inf = %.6e\n", residual_inf)
                println("  status = ", status)
            catch err
                status = "failed: " * sprint(showerror, err)
                write_csv_row(io, error_failure_row(
                    N, Nplus, mu, alpha, beta,
                    muntz_lambda, muntz_jacobi_a, muntz_jacobi_b,
                    grid.total_count, status,
                ))
                @printf("[%3d/%3d] N=%5d Nplus=%5d ERROR FAILED: %s\n",
                    idx, length(Ns_error), N, Nplus, clean_status(status))
            end
        end
    end

    println("wrote ", csv_path)
    summary = csv_summary(csv_path)
    print_summary(summary, csv_path)
    return summary
end

function run_timing(
    prob,
    Ns_timing,
    csv_path::AbstractString,
    resume::Bool,
    mu,
    alpha,
    beta,
    muntz_lambda,
    muntz_jacobi_a,
    muntz_jacobi_b,
    bench_samples::Int,
    bench_evals::Int,
    bench_seconds::Float64,
)
    required_cols = (
        "method", "N", "mu", "alpha", "beta",
        "muntz_lambda", "muntz_jacobi_a", "muntz_jacobi_b",
        "build_min", "build_median", "build_q25", "build_q75",
        "solve_min", "solve_median", "solve_q25", "solve_q75",
        "total_min", "total_median", "build_samples", "solve_samples",
        "bench_evals", "bench_seconds", "status",
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

    warm_N = first(Ns_timing)
    println("precompile/warm-up N = ", warm_N)
    warm = build_system(prob, warm_N + 1)
    _ = warm.A \ warm.rhs

    open(csv_path, "a") do io
        for (idx, N) in enumerate(Ns_timing)
            if N in completed_Ns
                @printf("[%3d/%3d] N=%5d skipped: existing timing success row\n",
                    idx, length(Ns_timing), N)
                continue
            end

            try
                println("Timing N = ", N)
                # Per-N build and solve warm-up are outside the formal trials.
                sys_for_solve = build_system(prob, N + 1)
                _ = sys_for_solve.A \ sys_for_solve.rhs
                build_stats, solve_stats = benchmark_main_times(
                    prob,
                    N,
                    sys_for_solve,
                    bench_samples,
                    bench_evals,
                    bench_seconds,
                )
                total_min = build_stats.min + solve_stats.min
                total_median = build_stats.median + solve_stats.median
                time_ok = all(t -> isfinite(t) && t > 0.0, (
                    build_stats.min,
                    build_stats.median,
                    build_stats.q25,
                    build_stats.q75,
                    solve_stats.min,
                    solve_stats.median,
                    solve_stats.q25,
                    solve_stats.q75,
                    total_min,
                    total_median,
                ))
                status = time_ok ? "success" : "nonfinite"

                write_csv_row(io, (
                    "Müntz--collocation",
                    N,
                    mu,
                    alpha,
                    beta,
                    muntz_lambda,
                    muntz_jacobi_a,
                    muntz_jacobi_b,
                    build_stats.min,
                    build_stats.median,
                    build_stats.q25,
                    build_stats.q75,
                    solve_stats.min,
                    solve_stats.median,
                    solve_stats.q25,
                    solve_stats.q75,
                    total_min,
                    total_median,
                    build_stats.samples,
                    solve_stats.samples,
                    bench_evals,
                    bench_seconds,
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
                    alpha,
                    beta,
                    muntz_lambda,
                    muntz_jacobi_a,
                    muntz_jacobi_b,
                    bench_evals,
                    bench_seconds,
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
    alpha = 1.0 / 3.0
    beta = 0.5
    muntz_lambda = 1.0 / 6.0
    muntz_jacobi_a = -0.5
    muntz_jacobi_b = -0.5

    run_mode = lowercase(get(ENV, "EX7_RUN_MODE", "both"))
    run_mode in ("error", "timing", "both") ||
        throw(ArgumentError("EX7_RUN_MODE must be error, timing, or both"))

    error_nmin = parse(Int, get(ENV, "EX7_ERROR_NMIN", "10"))
    error_nmax = parse(Int, get(ENV, "EX7_ERROR_NMAX",
        get(ENV, "EX7_FRAC_NMAX", "2000")))
    error_step = parse(Int, get(ENV, "EX7_ERROR_STEP",
        get(ENV, "EX7_FRAC_STEP", "10")))
    uniform_grid_points = parse(Int, get(ENV, "EX7_ERROR_UNIFORM_POINTS", "10001"))
    fractional_grid_points = parse(Int, get(ENV, "EX7_ERROR_FRACTIONAL_POINTS", "5001"))
    resume = parse_bool_env("EX7_FRAC_RESUME", "true")
    timing_nmin = parse(Int, get(ENV, "EX7_TIMING_NMIN", "10"))
    timing_nmax = parse(Int, get(ENV, "EX7_TIMING_NMAX", "1500"))
    timing_npts = parse(Int, get(ENV, "EX7_TIMING_NPTS", "100"))
    bench_samples = parse(Int, get(ENV, "EX7_BENCH_SAMPLES", "5"))
    bench_evals = parse(Int, get(ENV, "EX7_BENCH_EVALS", "1"))
    bench_seconds = parse(Float64, get(ENV, "EX7_BENCH_SECONDS", "120"))
    bench_samples > 0 || throw(ArgumentError("EX7_BENCH_SAMPLES must be positive"))
    bench_evals > 0 || throw(ArgumentError("EX7_BENCH_EVALS must be positive"))
    bench_seconds > 0 || throw(ArgumentError("EX7_BENCH_SECONDS must be positive"))
    Ns_error = error_degrees(error_nmax;
        nmin = error_nmin,
        step = error_step,
    )
    Ns_timing = logspace_int_unique(timing_nmin, timing_nmax, timing_npts)

    data_dir = joinpath(script_dir, "data")
    mkpath(data_dir)
    error_csv_path = joinpath(data_dir, "ex7_FracCollocation_error.csv")
    timing_csv_path = joinpath(data_dir, "ex7_FracCollocation_timing.csv")
    grid = composite_test_grid(uniform_grid_points, fractional_grid_points)

    regression = stable_coordinate_regression_check()
    prob = make_problem(mu, muntz_lambda, muntz_jacobi_a, muntz_jacobi_b)

    println("ex7_FracCollocation.jl")
    println("run mode = ", run_mode)
    @printf("mu = %.16g, alpha = %.16g, beta = %.16g\n", mu, alpha, beta)
    println("linear system = (I + 0.5 V_K) u = g")
    @printf("muntz_lambda = %.16g, Jacobi parameters = (%.16g, %.16g)\n",
        muntz_lambda, muntz_jacobi_a, muntz_jacobi_b)
    println("error N rule = ", error_nmin, ":", error_step, ":", error_nmax,
        " (", length(Ns_error), " cases)")
    println("timing N range = ", first(Ns_timing), " to ", last(Ns_timing),
        " (", length(Ns_timing), " log-spaced unique cases)")
    println("composite grid = ", grid.uniform_count, " uniform x + ",
        grid.fractional_count, " uniform z=s^(1/6); merged unique x = ",
        grid.total_count)
    @printf("stable-coordinate regression: s=%.6e, mapped_rhs(s)=%.6e\n",
        regression.s, regression.value)
    println("Julia threads = ", Threads.nthreads(), ", BLAS threads = ", BLAS.get_num_threads())
    println("benchmark samples = ", bench_samples, ", evals = ", bench_evals,
        ", seconds = ", bench_seconds)
    println("resume = ", resume)

    if run_mode in ("error", "both")
        run_error(
            prob,
            Ns_error,
            grid,
            error_csv_path,
            resume,
            mu,
            alpha,
            beta,
            muntz_lambda,
            muntz_jacobi_a,
            muntz_jacobi_b,
        )
    end

    if run_mode in ("timing", "both")
        run_timing(
            prob,
            Ns_timing,
            timing_csv_path,
            resume,
            mu,
            alpha,
            beta,
            muntz_lambda,
            muntz_jacobi_a,
            muntz_jacobi_b,
            bench_samples,
            bench_evals,
            bench_seconds,
        )
    end

    return (error_csv = error_csv_path, timing_csv = timing_csv_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
