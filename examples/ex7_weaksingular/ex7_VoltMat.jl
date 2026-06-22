# ex7_VoltMat.jl
#
# Solve the weakly singular Volterra equation
#
#   u(x) + 0.5 * int_{-1}^x (x-y)^(-0.5) K(x,y) u(y) dy = g(x),
#   x in [-1, 1],
#
# where
#
#   K(x,y) = 1 / ((x+1)^2 + (y-1)^2),
#   g(x)   = 1 + sin(pi*x)/(1000*x^2 + 1).
#
# Method:
#   Weak_VoltMat_Cheb from src/ is used to assemble the weakly singular
#   Volterra operator in a JFP/fractional-Chebyshev coefficient
#   representation. With beta = 1 - mu = 0.5, the approximation is evaluated
#   as a Chebyshev series in z = 2*((1+x)/2)^beta - 1. The operator is requested
#   in packed lower-Hessenberg form and converted to a sparse matrix for the
#   linear solve.
#
# Linear system:
#   Weak_VoltMat_Cheb constructs V_K for
#     (V_K u)(x) = int_{-1}^x (x-y)^(-mu) K(x,y) u(y) dy.
#   Therefore this script solves
#     (I + 0.5 V_K) u = g,
#   not I - 0.5 V_K.
#
# Error:
#   No exact solution is used. The script reports Cauchy errors by comparing
#   the N approximation with the N_ref approximation, where
#   N_ref = max(N+1, round(Int, 1.01*N)). err_coeff_inf is the internal
#   JFP/Cheb coefficient-vector Cauchy difference; err_grid_inf on the fixed
#   x-grid is the primary point-value error.
#
# Output:
#   examples/ex7_weaksingular/data/ex7_VoltMat_error.csv
#   examples/ex7_weaksingular/data/ex7_VoltMat_timing.csv
#
# Run from the VoltOpt project root:
#   julia --project=. examples/ex7_weaksingular/ex7_VoltMat.jl
#
# Or from this folder:
#   julia --project=../.. ex7_VoltMat.jl
#
# Run modes:
#   EX7_RUN_MODE=error   only refresh the error CSV
#   EX7_RUN_MODE=timing  only refresh the timing CSV
#   EX7_RUN_MODE=both    refresh error first, then timing
#
# Timing / benchmarking:
#   Timing mode uses BenchmarkTools to estimate build and solve times by
#   repeated sampling on log-spaced N values. Timing CSV columns report min,
#   median, and mean times in seconds. Error CSV files do not contain timing
#   statistics.
#
#   Environment variables:
#     EX7_BENCH_SAMPLES=5        number of benchmark samples
#     EX7_BENCH_EVALS=1          evaluations per sample
#
#   Example quick benchmark run:
#     EX7_RUN_MODE=timing EX7_BENCH_SAMPLES=3 EX7_TIMING_NMAX=100 julia --project=. examples/ex7_weaksingular/ex7_VoltMat.jl
#
#   Windows PowerShell example:
#     $env:EX7_BENCH_SAMPLES="3"
#     $env:EX7_RUN_MODE="timing"
#     $env:EX7_TIMING_NMAX="100"
#     julia --project=. examples/ex7_weaksingular/ex7_VoltMat.jl
#     Remove-Item Env:\EX7_BENCH_SAMPLES
#     Remove-Item Env:\EX7_RUN_MODE
#     Remove-Item Env:\EX7_TIMING_NMAX
#
# Only compute errors:
#   $env:EX7_RUN_MODE="error"
#   julia --project=. examples/ex7_weaksingular/ex7_VoltMat.jl
#
# Only benchmark timing:
#   $env:EX7_RUN_MODE="timing"
#   $env:EX7_TIMING_NMIN="10"
#   $env:EX7_TIMING_NMAX="1500"
#   $env:EX7_TIMING_NPTS="100"
#   $env:EX7_BENCH_SAMPLES="5"
#   julia --project=. examples/ex7_weaksingular/ex7_VoltMat.jl
#
# Compute both:
#   $env:EX7_RUN_MODE="both"
#   julia --project=. examples/ex7_weaksingular/ex7_VoltMat.jl
#
# Environment controls:
#   EX7_VOLTMAT_NMAX   controls the largest error N, default 1500.
#   EX7_VOLTMAT_STEP   controls the error N step, default 10.
#   EX7_VOLTMAT_COND   controls whether to compute cond(A), default false.
#   EX7_VOLTMAT_RESUME controls whether to skip existing successful N, default true.
#   EX7_VOLTMAT_GRID_M controls the fixed check-grid size, default 2001.

using LinearAlgebra
using Printf
using SparseArrays
using BenchmarkTools
using Statistics: mean, median

script_dir = @__DIR__
project_root = normpath(joinpath(script_dir, "..", ".."))
include(joinpath(project_root, "src", "VoltOpt.jl"))
using .VoltOpt

function kernel_xy(x::Real, y::Real)
    xf = Float64(x)
    yf = Float64(y)
    return 1.0 / ((xf + 1.0)^2 + (yf - 1.0)^2)
end

function rhs_x(x::Real)
    xf = Float64(x)
    return 1.0 + sin(pi * xf) / (1000.0 * xf^2 + 1.0)
end

phi_beta_local(x::Real, beta::Float64) =
    2.0 * ((1.0 + Float64(x)) / 2.0)^beta - 1.0

invphi_beta_local(z::Real, beta::Float64) =
    2.0 * ((1.0 + Float64(z)) / 2.0)^(1.0 / beta) - 1.0

function cheb_eval(c::AbstractVector{<:Real}, z::Real)
    n = length(c) - 1
    n < 0 && throw(ArgumentError("coefficient vector must be nonempty"))
    n == 0 && return Float64(c[1])

    zz = Float64(z)
    bkp1 = 0.0
    bkp2 = 0.0
    @inbounds for k in n:-1:1
        bk = 2.0 * zz * bkp1 - bkp2 + Float64(c[k + 1])
        bkp2 = bkp1
        bkp1 = bk
    end
    return zz * bkp1 - bkp2 + Float64(c[1])
end

eval_jfp(c::AbstractVector{<:Real}, x::Real, beta::Float64) =
    cheb_eval(c, phi_beta_local(x, beta))

function cheb_project_lobatto(f, coeff_degree::Int, sample_degree::Int)
    coeff_degree <= sample_degree ||
        throw(ArgumentError("coeff_degree must not exceed sample_degree"))
    sample_degree >= 1 ||
        throw(ArgumentError("sample_degree must be positive"))

    M = sample_degree
    vals = Vector{Float64}(undef, M + 1)
    @inbounds for j in 0:M
        z = cos(pi * j / M)
        vals[j + 1] = f(z)
    end

    c = zeros(Float64, coeff_degree + 1)
    @inbounds for k in 0:coeff_degree
        s = 0.0
        for j in 0:M
            wj = (j == 0 || j == M) ? 0.5 : 1.0
            s += wj * vals[j + 1] * cos(pi * k * j / M)
        end
        c[k + 1] = 2.0 * s / M
    end
    c[1] *= 0.5

    return c
end

function build_rhs_coefficients(max_degree::Int, beta::Float64)
    sample_degree = max(8192, 4 * max_degree)
    rhs_z = z -> rhs_x(invphi_beta_local(z, beta))

    coeffs = nothing
    rhs_time = @elapsed begin
        coeffs = cheb_project_lobatto(rhs_z, max_degree, sample_degree)
    end
    return coeffs, rhs_time, sample_degree
end

function build_kernel_coefficients(beta::Float64)
    kernel_i = 80
    kernel_j = 64

    C = nothing
    kernel_time = @elapsed begin
        values = VoltOpt.Weak_VoltMat_Cheb.sample_duffy_kernel(
            kernel_xy,
            kernel_i,
            kernel_j,
            beta,
        )
        C = VoltOpt.Weak_VoltMat_Cheb.cheb_coeffs_lobatto_2d(values)
    end
    tail = VoltOpt.Weak_VoltMat_Cheb.coefficient_tail_info(C)
    return C, kernel_time, tail, kernel_i, kernel_j
end

function packed_to_sparse(A::VoltOpt.PackedLowerHessenberg)
    n = A.n
    nnz_band = A.colptr[end] - 1
    rows = Vector{Int}(undef, nnz_band)
    cols = Vector{Int}(undef, nnz_band)
    vals = Vector{Float64}(undef, nnz_band)

    p = 1
    @inbounds for col in 1:n
        len = min(n, col + A.lower_bw)
        src = A.colptr[col]
        for row in 1:len
            rows[p] = row
            cols[p] = col
            vals[p] = A.values[src + row - 1]
            p += 1
        end
    end

    S = sparse(rows, cols, vals, n, n)
    dropzeros!(S)
    return S
end

function operator_to_sparse(V, n::Int)
    if V isa VoltOpt.PackedLowerHessenberg
        return packed_to_sparse(V)
    elseif V isa SparseMatrixCSC
        S = copy(V)
        dropzeros!(S)
        return S
    elseif V isa AbstractMatrix
        size(V) == (n, n) || throw(DimensionMismatch("operator matrix has wrong size"))
        S = sparse(V)
        dropzeros!(S)
        return S
    else
        D = VoltOpt.packed_to_dense(V)
        S = sparse(D)
        dropzeros!(S)
        return S
    end
end

function system_matrix(VK::SparseMatrixCSC, lambda_vie::Float64)
    n = size(VK, 1)
    size(VK, 2) == n || throw(DimensionMismatch("operator matrix must be square"))
    A = lambda_vie .* copy(VK)
    @inbounds for i in 1:n
        A[i, i] += 1.0
    end
    dropzeros!(A)
    return A
end

function maxabs_matrix(A)
    if A isa SparseMatrixCSC
        return nnz(A) == 0 ? 0.0 : maximum(abs, nonzeros(A))
    else
        return maximum(abs, A)
    end
end

function build_voltmat_system(
    N::Int,
    C::AbstractMatrix,
    rhs_coeffs::AbstractVector,
    mu::Float64,
    beta::Float64,
    lambda_vie::Float64,
)
    rhsN = rhs_coeffs[1:(N + 1)]
    Vraw, _info = VoltOpt.weak_voltmat_cheb(
        C;
        C_is_coeffs = true,
        mu = mu,
        sigma = 0.0,
        beta = beta,
        N = N,
        diagnostics = false,
        output_format = :packed_lower_hessenberg,
        return_info = true,
    )
    VK = operator_to_sparse(Vraw, N + 1)
    A = system_matrix(VK, lambda_vie)
    return (VK = VK, A = A, rhs = rhsN)
end

function solve_degree(
    N::Int,
    C::AbstractMatrix,
    rhs_coeffs::AbstractVector,
    mu::Float64,
    beta::Float64,
    lambda_vie::Float64;
    compute_cond::Bool,
)
    built = nothing
    build_time = @elapsed begin
        built = build_voltmat_system(N, C, rhs_coeffs, mu, beta, lambda_vie)
    end

    coeffs = nothing
    solve_time = @elapsed begin
        coeffs = built.A \ built.rhs
    end

    linear_residual_inf = norm(built.A * coeffs - built.rhs, Inf)
    # Same discrete equation residual for (I + lambda_vie*V_K)c = rhs.
    residual_inf = linear_residual_inf
    cond_A = compute_cond ? cond(Matrix(built.A)) : NaN
    maxabs_V = maxabs_matrix(built.VK)
    maxabs_A = maxabs_matrix(built.A)
    # maxabs_u is the maximum absolute value of the solved coefficient vector.
    maxabs_u = maximum(abs, coeffs)

    return (
        N = N,
        coeffs = coeffs,
        VK = built.VK,
        A = built.A,
        rhs = built.rhs,
        build_time = build_time,
        solve_time = solve_time,
        residual_inf = residual_inf,
        linear_residual_inf = linear_residual_inf,
        cond_A = cond_A,
        maxabs_V = maxabs_V,
        maxabs_A = maxabs_A,
        maxabs_u = maxabs_u,
    )
end

function trial_seconds(trial)
    times = trial.times ./ 1.0e9
    return (
        min = minimum(times),
        median = median(times),
        mean = mean(times),
    )
end

function benchmark_main_times(
    N::Int,
    C::AbstractMatrix,
    rhs_coeffs::AbstractVector,
    mu::Float64,
    beta::Float64,
    lambda_vie::Float64,
    A,
    rhs,
    bench_samples::Int,
    bench_evals::Int,
)
    trial_build = @benchmark build_voltmat_system(
        $N,
        $C,
        $rhs_coeffs,
        $mu,
        $beta,
        $lambda_vie,
    ) samples=bench_samples evals=bench_evals
    build_stats = trial_seconds(trial_build)

    A_bench = copy(A)
    rhs_bench = copy(rhs)
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

function coeff_cauchy(cN::AbstractVector, c_ref::AbstractVector)
    nN = length(cN)
    nR = length(c_ref)
    nmin = min(nN, nR)
    err = norm(@view(cN[1:nmin]) .- @view(c_ref[1:nmin]), Inf)
    if nN > nmin
        err = max(err, norm(@view(cN[(nmin + 1):nN]), Inf))
    end
    if nR > nmin
        err = max(err, norm(@view(c_ref[(nmin + 1):nR]), Inf))
    end
    return err
end

function grid_cauchy(cN::AbstractVector, c_ref::AbstractVector, xgrid, beta::Float64)
    err = 0.0
    @inbounds for x in xgrid
        e = abs(eval_jfp(cN, x, beta) - eval_jfp(c_ref, x, beta))
        err = max(err, e)
    end
    return err
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
        "method,N,N_ref,mu," *
        "err_coeff_inf,err_grid_inf," *
        "residual_inf,linear_residual_inf," *
        "cond_A,maxabs_V,maxabs_A,maxabs_u,status",
    )
    flush(io)
    return nothing
end

function write_timing_header(io)
    println(io,
        "method,N,mu," *
        "build_time_min,build_time_median,build_time_mean," *
        "solve_time_min,solve_time_median,solve_time_mean," *
        "total_time_min,total_time_median,total_time_mean," *
        "bench_samples,bench_evals,status",
    )
    flush(io)
    return nothing
end

parse_bool_env(name::AbstractString, default::AbstractString) =
    lowercase(get(ENV, name, default)) in ("1", "true", "yes")

function clean_status(status::AbstractString)
    return replace(status, '\n' => ' ', '\r' => ' ')
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

csv_column_map(header_line::AbstractString) =
    Dict(name => idx for (idx, name) in pairs(split_csv_line(header_line)))

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

function error_failure_row(N, N_ref, mu, status)
    return (
        "VoltMat",
        N,
        N_ref,
        mu,
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

function timing_failure_row(N, mu, bench_samples, bench_evals, status)
    return (
        "VoltMat",
        N,
        mu,
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

    if !isfile(csv_path) || filesize(csv_path) == 0
        return (; rows, completed_success, failed, nonfinite, max_success_N,
            duplicate_N, bad_success_err_grid, bad_success_residual,
            bad_success_cond, statuses)
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

    return (; rows, completed_success, failed, nonfinite, max_success_N,
        duplicate_N, bad_success_err_grid, bad_success_residual,
        bad_success_cond, statuses)
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

function print_kernel_rhs_info(C, kernel_time, kernel_tail, kernel_i, kernel_j,
        rhs_coeffs, rhs_time, rhs_degree, rhs_sample_degree)
    @printf("kernel coeffs = %d x %d, build_time = %.6e s\n",
        kernel_i, kernel_j, kernel_time)
    @printf("kernel tail row_rel = %.3e, col_rel = %.3e, max_coeff = %.3e\n",
        kernel_tail.row_rel, kernel_tail.col_rel, kernel_tail.max_coeff)
    @printf("rhs coeff degree = %d, sample_degree = %d, build_time = %.6e s\n",
        rhs_degree, rhs_sample_degree, rhs_time)
    println("rhs coeff length = ", length(rhs_coeffs), ", kernel size = ", size(C))
    return nothing
end

function run_error(
    Ns_error,
    xgrid,
    csv_path::AbstractString,
    resume::Bool,
    compute_cond::Bool,
    mu::Float64,
    beta::Float64,
    lambda_vie::Float64,
)
    required_cols = (
        "method", "N", "N_ref", "mu",
        "err_coeff_inf", "err_grid_inf", "residual_inf", "linear_residual_inf",
        "cond_A", "maxabs_V", "maxabs_A", "maxabs_u", "status",
    )
    completed_Ns, existing_success, existing_failed, existing_dup_success =
        prepare_csv(csv_path, Ns_error, resume, write_error_header, required_cols)

    max_ref = maximum(max(N + 1, round(Int, 1.01 * N)) for N in Ns_error)
    println("Error CSV = ", csv_path)
    println("max N_ref = ", max_ref)
    if resume
        println("error resume existing success rows = ", existing_success,
            ", existing failed/non-success rows dropped = ", existing_failed,
            ", duplicate success rows dropped = ", existing_dup_success)
        println("error remaining cases to compute = ", count(N -> !(N in completed_Ns), Ns_error))
    end

    C, kernel_time, kernel_tail, kernel_i, kernel_j = build_kernel_coefficients(beta)
    rhs_coeffs, rhs_time, rhs_sample_degree = build_rhs_coefficients(max_ref, beta)
    print_kernel_rhs_info(C, kernel_time, kernel_tail, kernel_i, kernel_j,
        rhs_coeffs, rhs_time, max_ref, rhs_sample_degree)

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
                solN = solve_degree(
                    N,
                    C,
                    rhs_coeffs,
                    mu,
                    beta,
                    lambda_vie;
                    compute_cond = compute_cond,
                )

                err_coeff_inf = NaN
                err_grid_inf = NaN
                ref_error_time = @elapsed begin
                    sol_ref = solve_degree(
                        N_ref,
                        C,
                        rhs_coeffs,
                        mu,
                        beta,
                        lambda_vie;
                        compute_cond = false,
                    )
                    err_coeff_inf = coeff_cauchy(solN.coeffs, sol_ref.coeffs)
                    err_grid_inf = grid_cauchy(solN.coeffs, sol_ref.coeffs, xgrid, beta)
                end

                required_diagnostics = (
                    err_coeff_inf,
                    err_grid_inf,
                    solN.residual_inf,
                    solN.linear_residual_inf,
                    solN.maxabs_V,
                    solN.maxabs_A,
                    solN.maxabs_u,
                )
                cond_ok = !compute_cond || isfinite(solN.cond_A)
                status = all(isfinite, required_diagnostics) && cond_ok ? "success" : "nonfinite"

                write_csv_row(io, (
                    "VoltMat",
                    N,
                    N_ref,
                    mu,
                    err_coeff_inf,
                    err_grid_inf,
                    solN.residual_inf,
                    solN.linear_residual_inf,
                    solN.cond_A,
                    solN.maxabs_V,
                    solN.maxabs_A,
                    solN.maxabs_u,
                    status,
                ))

                @printf("  single build_time = %.6e\n", solN.build_time)
                @printf("  single solve_time = %.6e\n", solN.solve_time)
                @printf("  ref_error_time = %.6e\n", ref_error_time)
                @printf("  err_grid_inf = %.6e\n", err_grid_inf)
                @printf("  residual_inf = %.6e\n", solN.residual_inf)
                @printf("  cond_A = %s\n",
                    isnan(solN.cond_A) ? "NaN" : @sprintf("%.6e", solN.cond_A))
                println("  status = ", status)
            catch err
                status = "failed: " * sprint(showerror, err)
                write_csv_row(io, error_failure_row(N, N_ref, mu, status))
                @printf("[%3d/%3d] N=%5d N_ref=%5d ERROR FAILED: %s\n",
                    idx, length(Ns_error), N, N_ref, clean_status(status))
            end
        end
    end

    println("wrote ", csv_path)
    summary = csv_summary(csv_path; expect_cond_nan = !compute_cond)
    print_summary(summary, csv_path)
    return summary
end

function run_timing(
    Ns_timing,
    csv_path::AbstractString,
    resume::Bool,
    mu::Float64,
    beta::Float64,
    lambda_vie::Float64,
    bench_samples::Int,
    bench_evals::Int,
)
    required_cols = (
        "method", "N", "mu",
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

    max_timing_N = maximum(Ns_timing)
    C, kernel_time, kernel_tail, kernel_i, kernel_j = build_kernel_coefficients(beta)
    rhs_coeffs, rhs_time, rhs_sample_degree = build_rhs_coefficients(max_timing_N, beta)
    print_kernel_rhs_info(C, kernel_time, kernel_tail, kernel_i, kernel_j,
        rhs_coeffs, rhs_time, max_timing_N, rhs_sample_degree)

    open(csv_path, "a") do io
        for (idx, N) in enumerate(Ns_timing)
            if N in completed_Ns
                @printf("[%3d/%3d] N=%5d skipped: existing timing success row\n",
                    idx, length(Ns_timing), N)
                continue
            end

            try
                println("Timing N = ", N)
                built = build_voltmat_system(N, C, rhs_coeffs, mu, beta, lambda_vie)
                build_stats, solve_stats = benchmark_main_times(
                    N,
                    C,
                    rhs_coeffs,
                    mu,
                    beta,
                    lambda_vie,
                    built.A,
                    built.rhs,
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
                    "VoltMat",
                    N,
                    mu,
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
                write_csv_row(io, timing_failure_row(N, mu, bench_samples, bench_evals, status))
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
    beta = 1.0 - mu
    lambda_vie = 0.5

    run_mode = lowercase(get(ENV, "EX7_RUN_MODE", "both"))
    run_mode in ("error", "timing", "both") ||
        throw(ArgumentError("EX7_RUN_MODE must be error, timing, or both"))

    nmax = parse(Int, get(ENV, "EX7_VOLTMAT_NMAX", "1500"))
    nstep = parse(Int, get(ENV, "EX7_VOLTMAT_STEP", "10"))
    resume = parse_bool_env("EX7_VOLTMAT_RESUME", "true")
    grid_m = parse(Int, get(ENV, "EX7_VOLTMAT_GRID_M", "2001"))
    compute_cond = parse_bool_env("EX7_VOLTMAT_COND", "false")
    timing_nmin = parse(Int, get(ENV, "EX7_TIMING_NMIN", "10"))
    timing_nmax = parse(Int, get(ENV, "EX7_TIMING_NMAX", "1500"))
    timing_npts = parse(Int, get(ENV, "EX7_TIMING_NPTS", "100"))
    bench_samples = parse(Int, get(ENV, "EX7_BENCH_SAMPLES", "5"))
    bench_evals = parse(Int, get(ENV, "EX7_BENCH_EVALS", "1"))

    nmax >= 10 || throw(ArgumentError("EX7_VOLTMAT_NMAX must be at least 10"))
    nstep > 0 || throw(ArgumentError("EX7_VOLTMAT_STEP must be positive"))
    grid_m >= 2 || throw(ArgumentError("EX7_VOLTMAT_GRID_M must be at least 2"))
    bench_samples > 0 || throw(ArgumentError("EX7_BENCH_SAMPLES must be positive"))
    bench_evals > 0 || throw(ArgumentError("EX7_BENCH_EVALS must be positive"))

    Ns_error = collect(10:nstep:nmax)
    Ns_timing = logspace_int_unique(timing_nmin, timing_nmax, timing_npts)
    xgrid = collect(range(-1.0, 1.0; length = grid_m))

    data_dir = joinpath(script_dir, "data")
    mkpath(data_dir)
    error_csv_path = joinpath(data_dir, "ex7_VoltMat_error.csv")
    timing_csv_path = joinpath(data_dir, "ex7_VoltMat_timing.csv")

    println("ex7_VoltMat.jl")
    println("run mode = ", run_mode)
    @printf("mu = %.16g, beta = %.16g, equation lambda = %.16g\n", mu, beta, lambda_vie)
    println("linear system = (I + 0.5 V_K) u = g")
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
            Ns_error,
            xgrid,
            error_csv_path,
            resume,
            compute_cond,
            mu,
            beta,
            lambda_vie,
        )
    end

    if run_mode in ("timing", "both")
        run_timing(
            Ns_timing,
            timing_csv_path,
            resume,
            mu,
            beta,
            lambda_vie,
            bench_samples,
            bench_evals,
        )
    end

    return (error_csv = error_csv_path, timing_csv = timing_csv_path)
end

main()
