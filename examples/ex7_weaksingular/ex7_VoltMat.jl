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
#   g(x)   = ((1+x)/2)^(1/3) * (1 + sin(pi*x)/(1000*x^2 + 1)).
#
# Method:
#   Weak_VoltMat_Cheb from src/ is used to assemble the weakly singular
#   Volterra operator in a JFP/fractional-Chebyshev coefficient
#   representation. The weighted fractional-Chebyshev basis is
#     ((1+x)/2)^alpha T_n(2*((1+x)/2)^beta - 1),
#   with alpha = 1/3 and beta = 1/2. The operator is requested in packed
#   lower-Hessenberg form and converted to a sparse matrix for the linear solve.
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
#   the N approximation with the Nplus approximation, where
#   Nplus = max(N+1, round(Int, 1.01*N)). The Cauchy error is the maximum
#   point-value difference on the shared, N-independent composite physical
#   grid: at least 10001 uniform x points plus 5001 uniform z=s^(1/6) points
#   mapped to x.  Both solutions are first differenced in their common weighted
#   Chebyshev basis and the difference is then evaluated by Clenshaw recurrence,
#   avoiding cancellation between two separately evaluated function values.
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
#   median, q25, and q75 times in seconds. Error CSV files do not contain
#   timing statistics.
#
#   Environment variables:
#     EX7_BENCH_SAMPLES=5        number of benchmark samples
#     EX7_BENCH_EVALS=1          evaluations per sample
#     EX7_BENCH_SECONDS=120      maximum seconds per benchmark
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
#   EX7_ERROR_NMIN        smallest error N, default 10.
#   EX7_ERROR_NMAX        largest error N, default 2000.
#   EX7_ERROR_STEP        uniform error step, default 10.
#   EX7_ERROR_UNIFORM_POINTS   uniform physical-grid count, default 10001.
#   EX7_ERROR_FRACTIONAL_POINTS uniform z=s^(1/6) count, default 5001.
#   EX7_VOLTMAT_RESUME controls whether to skip existing successful N, default true.

using LinearAlgebra
using Printf
using SparseArrays
using BenchmarkTools
using Statistics: median, quantile

script_dir = @__DIR__
project_root = normpath(joinpath(script_dir, "..", ".."))
include(joinpath(project_root, "src", "VoltOpt.jl"))
using .VoltOpt
include(joinpath(script_dir, "ex7_common.jl"))
using .Ex7Common

function kernel_xy(x::Real, y::Real)
    xf = Float64(x)
    yf = Float64(y)
    return 1.0 / ((xf + 1.0)^2 + (yf - 1.0)^2)
end

function rhs_factor_x(x::Real)
    xf = Float64(x)
    return 1.0 + sin(pi * xf) / (1000.0 * xf^2 + 1.0)
end

function rhs_x(x::Real)
    xf = Float64(x)
    return ((1.0 + xf) / 2.0)^(1.0 / 3.0) * rhs_factor_x(xf)
end

function rhs_basis_factor_x(x::Real, _alpha::Float64)
    xf = Float64(x)
    # Analytically cancel the basis weight instead of multiplying by s^alpha
    # in rhs_x and dividing it back out.  The latter injects pointwise
    # roundoff into the high Chebyshev modes of this smooth factor.
    return rhs_factor_x(xf)
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

eval_jfp(c::AbstractVector{<:Real}, x::Real, alpha::Float64, beta::Float64) =
    ((1.0 + Float64(x)) / 2.0)^alpha * cheb_eval(c, phi_beta_local(x, beta))

eval_jfp_s(c::AbstractVector{<:Real}, s::Real, alpha::Float64, beta::Float64) =
    Float64(s)^alpha * cheb_eval(c, 2.0 * Float64(s)^beta - 1.0)

function cheb_project_lobatto(f, coeff_degree::Int, sample_degree::Int)
    coeff_degree <= sample_degree ||
        throw(ArgumentError("coeff_degree must not exceed sample_degree"))
    sample_degree >= 1 ||
        throw(ArgumentError("sample_degree must be positive"))

    M = sample_degree
    vals = Vector{Float64}(undef, M + 1)
    @inbounds for j in 0:M
        z = cospi(j / M)
        vals[j + 1] = f(z)
    end

    # Reuse the project's DCT-I coefficient transform.  The former explicit
    # oscillatory sums lost several digits once the true high-order
    # coefficients fell below machine precision, producing a noisy RHS tail.
    all_coeffs = VoltOpt.Weak_VoltMat_Cheb.cheb_coeffs_lobatto_1d(vals)
    return all_coeffs[1:(coeff_degree + 1)]
end

function build_rhs_coefficients(max_degree::Int, alpha::Float64, beta::Float64)
    sample_degree = max(8192, 4 * max_degree)
    # The weighted basis already contains ((1+x)/2)^alpha, so project only
    # the remaining smooth factor. This also avoids a 0/0 division at x=-1.
    rhs_z = z -> rhs_basis_factor_x(invphi_beta_local(z, beta), alpha)

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

function system_matrix(VK::SparseMatrixCSC)
    n = size(VK, 1)
    size(VK, 2) == n || throw(DimensionMismatch("operator matrix must be square"))
    A = 0.5 .* copy(VK)
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
    alpha::Float64,
    beta::Float64,
)
    rhsN = rhs_coeffs[1:(N + 1)]
    Vraw, _info = VoltOpt.weak_voltmat_cheb(
        C;
        C_is_coeffs = true,
        mu = mu,
        sigma = alpha,
        beta = beta,
        N = N,
        diagnostics = false,
        output_format = :packed_lower_hessenberg,
        H_recurrence_mode = :block_i,
        return_info = true,
    )
    _info.H_recurrence_mode === :block_i ||
        error("Example 7 did not select block-i moment recurrence")
    _info.moment_recurrence === :four_point ||
        error("Example 7 did not select the four-point recurrence")
    _info.contraction_mode === :precombined_j ||
        error("Example 7 did not select precombined-j contraction")
    _info.assembly_mode === :precombined_ellmajor_serial ||
        error("Example 7 did not select ell-major assembly")
    _info.multiplier_mode === :deferred_eta1_column ||
        error("Example 7 did not select the deferred eta=1 multiplier")
    VK = operator_to_sparse(Vraw, N + 1)
    # For nonzero sigma, the current core interface returns columns with the
    # unnormalised endpoint factor (1+x)^sigma, while this example and the
    # public basis use ((1+x)/2)^alpha. Restore the physical operator scaling
    # here without changing the core algorithm. The equation coefficient in
    # system_matrix remains exactly 0.5.
    VK .*= 2.0^alpha
    A = system_matrix(VK)
    return (VK = VK, A = A, rhs = rhsN)
end

function solve_degree(
    N::Int,
    C::AbstractMatrix,
    rhs_coeffs::AbstractVector,
    mu::Float64,
    alpha::Float64,
    beta::Float64,
)
    built = nothing
    build_time = @elapsed begin
        built = build_voltmat_system(N, C, rhs_coeffs, mu, alpha, beta)
    end

    coeffs = nothing
    solve_time = @elapsed begin
        factor = lu(built.A)
        coeffs = factor \ built.rhs
    end

    linear_residual_inf = norm(built.A * coeffs - built.rhs, Inf)
    # Same discrete equation residual for (I + 0.5*V_K)c = rhs.
    residual_inf = linear_residual_inf
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
        q25 = quantile(times, 0.25),
        q75 = quantile(times, 0.75),
        samples = length(times),
    )
end

function benchmark_main_times(
    N::Int,
    C::AbstractMatrix,
    rhs_coeffs::AbstractVector,
    mu::Float64,
    alpha::Float64,
    beta::Float64,
    A,
    rhs,
    bench_samples::Int,
    bench_evals::Int,
    bench_seconds::Float64,
)
    trial_build = @benchmark build_voltmat_system(
        $N,
        $C,
        $rhs_coeffs,
        $mu,
        $alpha,
        $beta,
    ) samples=bench_samples evals=bench_evals seconds=bench_seconds
    build_stats = trial_seconds(trial_build)

    A_bench = copy(A)
    rhs_bench = copy(rhs)
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

function grid_cauchy(cN::AbstractVector, c_ref::AbstractVector, grid,
        alpha::Float64, beta::Float64)
    length(cN) <= length(c_ref) ||
        throw(DimensionMismatch("reference coefficient vector must not be shorter"))

    # Both approximations use the same normalized weighted-Chebyshev basis.
    # Form their padded coefficient difference first, then evaluate that small
    # series once.  This avoids subtracting two O(1), separately rounded
    # Clenshaw values when the Cauchy error is near machine precision.
    coefficient_difference = -Float64.(c_ref)
    @views coefficient_difference[1:length(cN)] .+= cN

    err = 0.0
    max_error_x = grid.x[1]
    @inbounds for j in eachindex(grid.x, grid.s)
        e = abs(eval_jfp_s(coefficient_difference, grid.s[j], alpha, beta))
        if e > err
            err = e
            max_error_x = grid.x[j]
        end
    end
    return err, max_error_x
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
        "residual_inf,linear_residual_inf," *
        "maxabs_V,maxabs_A,maxabs_u,status",
    )
    flush(io)
    return nothing
end

function write_timing_header(io)
    println(io,
        "method,N,mu,alpha,beta," *
        "build_min,build_median,build_q25,build_q75," *
        "solve_min,solve_median,solve_q25,solve_q75," *
        "total_min,total_median,build_samples,solve_samples," *
        "bench_evals,bench_seconds,status",
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

function error_failure_row(N, Nplus, mu, alpha, beta, test_grid_size, status)
    return (
        "Weak Duffy--CFP",
        N,
        Nplus,
        mu,
        alpha,
        beta,
        NaN,
        NaN,
        test_grid_size,
        NaN,
        NaN,
        NaN,
        NaN,
        NaN,
        clean_status(status),
    )
end

function timing_failure_row(N, mu, alpha, beta, bench_evals, bench_seconds, status)
    return (
        "Weak Duffy--CFP",
        N,
        mu,
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
        NaN,
        0,
        0,
        bench_evals,
        bench_seconds,
        clean_status(status),
    )
end

function csv_summary(csv_path::AbstractString)
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

    if !isfile(csv_path) || filesize(csv_path) == 0
        return (; rows, completed_success, failed, nonfinite, max_success_N,
            duplicate_N, bad_success_cauchy, bad_success_residual, statuses)
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

    return (; rows, completed_success, failed, nonfinite, max_success_N,
        duplicate_N, bad_success_cauchy, bad_success_residual, statuses)
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
    grid,
    csv_path::AbstractString,
    resume::Bool,
    mu::Float64,
    alpha::Float64,
    beta::Float64,
)
    required_cols = (
        "method", "N", "Nplus", "mu", "alpha", "beta", "cauchy_error",
        "max_error_x", "test_grid_size",
        "residual_inf", "linear_residual_inf",
        "maxabs_V", "maxabs_A", "maxabs_u", "status",
    )
    completed_Ns, existing_success, existing_failed, existing_dup_success =
        prepare_csv(csv_path, Ns_error, resume, write_error_header, required_cols)

    max_ref = maximum(max(N + 1, round(Int, 1.01 * N)) for N in Ns_error)
    println("Error CSV = ", csv_path)
    println("max Nplus = ", max_ref)
    if resume
        println("error resume existing success rows = ", existing_success,
            ", existing failed/non-success rows dropped = ", existing_failed,
            ", duplicate success rows dropped = ", existing_dup_success)
        println("error remaining cases to compute = ", count(N -> !(N in completed_Ns), Ns_error))
    end

    C, kernel_time, kernel_tail, kernel_i, kernel_j = build_kernel_coefficients(beta)
    rhs_coeffs, rhs_time, rhs_sample_degree = build_rhs_coefficients(max_ref, alpha, beta)
    print_kernel_rhs_info(C, kernel_time, kernel_tail, kernel_i, kernel_j,
        rhs_coeffs, rhs_time, max_ref, rhs_sample_degree)

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
                solN = solve_degree(
                    N,
                    C,
                    rhs_coeffs,
                    mu,
                    alpha,
                    beta,
                )

                cauchy_error = NaN
                max_error_x = NaN
                ref_error_time = @elapsed begin
                    sol_ref = solve_degree(
                        Nplus,
                        C,
                        rhs_coeffs,
                        mu,
                        alpha,
                        beta,
                    )
                    cauchy_error, max_error_x = grid_cauchy(
                        solN.coeffs,
                        sol_ref.coeffs,
                        grid,
                        alpha,
                        beta,
                    )
                end

                required_diagnostics = (
                    cauchy_error,
                    max_error_x,
                    solN.residual_inf,
                    solN.linear_residual_inf,
                    solN.maxabs_V,
                    solN.maxabs_A,
                    solN.maxabs_u,
                )
                status = all(isfinite, required_diagnostics) ? "success" : "nonfinite"

                write_csv_row(io, (
                    "Weak Duffy--CFP",
                    N,
                    Nplus,
                    mu,
                    alpha,
                    beta,
                    cauchy_error,
                    max_error_x,
                    grid.total_count,
                    solN.residual_inf,
                    solN.linear_residual_inf,
                    solN.maxabs_V,
                    solN.maxabs_A,
                    solN.maxabs_u,
                    status,
                ))

                @printf("  single build_time = %.6e\n", solN.build_time)
                @printf("  single solve_time = %.6e\n", solN.solve_time)
                @printf("  ref_error_time = %.6e\n", ref_error_time)
                @printf("  cauchy_error = %.6e\n", cauchy_error)
                @printf("  max_error_x = %.17e\n", max_error_x)
                @printf("  residual_inf = %.6e\n", solN.residual_inf)
                println("  status = ", status)
            catch err
                status = "failed: " * sprint(showerror, err)
                write_csv_row(io, error_failure_row(
                    N, Nplus, mu, alpha, beta, grid.total_count, status,
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
    Ns_timing,
    csv_path::AbstractString,
    resume::Bool,
    mu::Float64,
    alpha::Float64,
    beta::Float64,
    bench_samples::Int,
    bench_evals::Int,
    bench_seconds::Float64,
)
    required_cols = (
        "method", "N", "mu", "alpha", "beta",
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

    max_timing_N = maximum(Ns_timing)
    C, kernel_time, kernel_tail, kernel_i, kernel_j = build_kernel_coefficients(beta)
    rhs_coeffs, rhs_time, rhs_sample_degree = build_rhs_coefficients(max_timing_N, alpha, beta)
    print_kernel_rhs_info(C, kernel_time, kernel_tail, kernel_i, kernel_j,
        rhs_coeffs, rhs_time, max_timing_N, rhs_sample_degree)

    warm_N = first(Ns_timing)
    println("precompile/warm-up N = ", warm_N)
    warm = build_voltmat_system(warm_N, C, rhs_coeffs, mu, alpha, beta)
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
                built = build_voltmat_system(N, C, rhs_coeffs, mu, alpha, beta)
                _ = built.A \ built.rhs
                build_stats, solve_stats = benchmark_main_times(
                    N,
                    C,
                    rhs_coeffs,
                    mu,
                    alpha,
                    beta,
                    built.A,
                    built.rhs,
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
                    "Weak Duffy--CFP",
                    N,
                    mu,
                    alpha,
                    beta,
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
                    N, mu, alpha, beta, bench_evals, bench_seconds, status,
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

    run_mode = lowercase(get(ENV, "EX7_RUN_MODE", "both"))
    run_mode in ("error", "timing", "both") ||
        throw(ArgumentError("EX7_RUN_MODE must be error, timing, or both"))

    error_nmin = parse(Int, get(ENV, "EX7_ERROR_NMIN", "10"))
    error_nmax = parse(Int, get(ENV, "EX7_ERROR_NMAX",
        get(ENV, "EX7_VOLTMAT_NMAX", "2000")))
    error_step = parse(Int, get(ENV, "EX7_ERROR_STEP",
        get(ENV, "EX7_VOLTMAT_STEP", "10")))
    uniform_grid_points = parse(Int, get(ENV, "EX7_ERROR_UNIFORM_POINTS", "10001"))
    fractional_grid_points = parse(Int, get(ENV, "EX7_ERROR_FRACTIONAL_POINTS", "5001"))
    resume = parse_bool_env("EX7_VOLTMAT_RESUME", "true")
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
    grid = composite_test_grid(uniform_grid_points, fractional_grid_points)

    data_dir = joinpath(script_dir, "data")
    mkpath(data_dir)
    error_csv_path = joinpath(data_dir, "ex7_VoltMat_error.csv")
    timing_csv_path = joinpath(data_dir, "ex7_VoltMat_timing.csv")

    println("ex7_VoltMat.jl")
    println("run mode = ", run_mode)
    @printf("mu = %.16g, alpha = %.16g, beta = %.16g\n", mu, alpha, beta)
    println("linear system = (I + 0.5 V_K) u = g")
    println("error N rule = ", error_nmin, ":", error_step, ":", error_nmax,
        " (", length(Ns_error), " cases)")
    println("timing N range = ", first(Ns_timing), " to ", last(Ns_timing),
        " (", length(Ns_timing), " log-spaced unique cases)")
    println("composite grid = ", grid.uniform_count, " uniform x + ",
        grid.fractional_count, " uniform z=s^(1/6); merged unique x = ",
        grid.total_count)
    println("Julia threads = ", Threads.nthreads(), ", BLAS threads = ", BLAS.get_num_threads())
    println("benchmark samples = ", bench_samples, ", evals = ", bench_evals,
        ", seconds = ", bench_seconds)
    println("resume = ", resume)

    if run_mode in ("error", "both")
        run_error(
            Ns_error,
            grid,
            error_csv_path,
            resume,
            mu,
            alpha,
            beta,
        )
    end

    if run_mode in ("timing", "both")
        run_timing(
            Ns_timing,
            timing_csv_path,
            resume,
            mu,
            alpha,
            beta,
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
