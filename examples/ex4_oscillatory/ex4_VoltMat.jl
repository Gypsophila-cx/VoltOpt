# -------------------------------------------------------------------------
# ex4_VoltMat.jl
#
# This script solves the second-kind Volterra integral equation
#
#     u(x) + int_0^x cos(omega*x*y) u(y) dy = f(x),      x in [0,1],
#
# with
#
#     omega = 20,
#     u_exact(x) = cos(50*x).
#
# The right-hand side is evaluated analytically using
#
#     f(x) = cos(50*x)
#            + x/2 * [
#                  sinc1((omega*x - 50)*x)
#                + sinc1((omega*x + 50)*x)
#              ],
#
# where sinc1(z) = sin(z)/z and sinc1(0) = 1. The implementation uses a
# stable small-z sinc form and does not generate f by numerical integration.
#
# Method:
#   The equation is mapped from x in [0,1] to xi in [-1,1] by
#
#       x = (xi + 1)/2.
#
#   Under this map,
#
#       int_0^x cos(omega*x*y) u(y) dy
#       = 1/2 int_-1^xi cos(omega*(xi+1)*(eta+1)/4) U(eta) deta,
#
#   with
#
#       U(xi) = u((xi+1)/2).
#
#   Therefore, the actual mapped Volterra kernel used to build the coefficient
#   matrix is
#
#       1/2 * cos(omega*(xi+1)*(eta+1)/4),
#
#   where the factor 1/2 comes from dy = deta/2. In the code this is the
#   mapped_kernel/voltmat_absorbed_mapped_kernel path used to generate C.
#
#   The Volterra operator is constructed by the VoltMat-Cheb method from src/.
#   The kernel coefficient matrix C is square, with C_rows = C_cols = depth.
#
# Parameters:
#   basis_len = 200.
#   depth ranges from 2 to 60 in the full sweep.
#
# Output:
#   The main sweep writes
#
#       examples/ex4_oscillatory/data/ex4_VoltMat_depth_sweep_N200.csv
#
#   The optional depth-50 diagnostic writes
#
#       examples/ex4_oscillatory/data/ex4_VoltMat_depth50_rhs_diagnostic.csv
#
# Usage:
#   Error sweep:
#
#       julia --project=. examples/ex4_oscillatory/ex4_VoltMat.jl
#
#   BenchmarkTools timing sweep:
#
#       julia --project=. examples/ex4_oscillatory/ex4_VoltMat.jl --benchmark
#
#   Optional depth-50 RHS diagnostic:
#
#       julia --project=. examples/ex4_oscillatory/ex4_VoltMat.jl --diagnose-depth50
#
# Benchmark mode:
#   The benchmark mode uses BenchmarkTools and is separated from the error
#   computation. It does not evaluate the dense grid error. It updates the
#   timing columns in the same CSV file while preserving existing error data.
#   The environment variables EX4_BENCH_BUILD_SAMPLES, EX4_BENCH_SOLVE_SAMPLES,
#   EX4_BENCH_BUILD_SECONDS, EX4_BENCH_SOLVE_SECONDS, and
#   EX4_BENCH_PRIMARY_STAT can be used to tune the timing run.
#
# Notes:
#   This script uses only the VoltMat-Cheb construction from src/.
#   It does not use the Gutleb--Olver code path.
#   The equation is a pure second-kind Volterra integral equation and does not
#   contain a derivative term.
#   Do not remove the factor 1/2 in the mapped Volterra kernel.
# -------------------------------------------------------------------------

using FFTW
using BenchmarkTools
using LinearAlgebra
using Printf
using SparseArrays
using Statistics

function find_project_root(start_dir::AbstractString=@__DIR__)
    dir = abspath(start_dir)
    while true
        if isfile(joinpath(dir, "Project.toml")) && isdir(joinpath(dir, "src"))
            return dir
        end

        parent = dirname(dir)
        parent == dir && error("Could not find VoltOpt project root from $(start_dir).")
        dir = parent
    end
end

const PROJECT_ROOT = find_project_root()
include(joinpath(PROJECT_ROOT, "src", "VoltOpt.jl"))
using .VoltOpt

const OMEGA = 20.0
const BASIS_LEN = 200
const DEGREE_USED = BASIS_LEN - 1
matrix_tol = 0.0
const GRID_LEN = 20_001
const TAIL_LEN = 32

const DATA_DIR = joinpath(@__DIR__, "data")
const SWEEP_CSV = joinpath(DATA_DIR, "ex4_VoltMat_depth_sweep_N200.csv")
const DIAGNOSTIC_CSV = joinpath(DATA_DIR, "ex4_VoltMat_depth50_rhs_diagnostic.csv")

evaluation_grid_cache = Ref{Any}(nothing)

struct ChebCoeffData
    coeffs::Vector{Float64}
    source_coeffs::Vector{Float64}
    method::String
    source_len::Int
    tail_inf::Float64
    tail_l2::Float64
end

const DIAGNOSTIC_COLUMNS = (
    :N,
    :depth,
    :rhs_method,
    :rhs_coeff_len,
    :rhs_tail_inf,
    :rhs_tail_l2,
    :exact_solution_coeff_tail_inf,
    :residual_inf_on_coeff,
    :residual_relative_on_coeff,
    :max_error_grid,
    :solve_residual_relative,
    :status,
)

const SWEEP_COLUMNS = (
    :depth,
    :C_rows,
    :C_cols,
    :basis_len,
    :degree_used,
    :omega,
    :rhs_method,
    :rhs_coeff_len,
    :rhs_tail_inf,
    :exact_residual_inf,
    :exact_residual_relative,
    :max_error,
    :residual_relative,
    :operator_build_time,
    :solve_time,
    :benchmark_build_min_time,
    :benchmark_build_median_time,
    :benchmark_build_q25_time,
    :benchmark_build_q75_time,
    :benchmark_build_iqr_time,
    :benchmark_solve_min_time,
    :benchmark_solve_median_time,
    :benchmark_solve_q25_time,
    :benchmark_solve_q75_time,
    :benchmark_solve_iqr_time,
    :benchmark_build_samples,
    :benchmark_solve_samples,
    :benchmark_build_allocs,
    :benchmark_solve_allocs,
    :maxabs_V,
    :maxabs_A,
    :nnz_V,
    :nnz_A,
    :status,
    :error_message,
)

@inline function sinc1(z::Float64)
    az = abs(z)
    if az < 1.0e-4
        z2 = z * z
        return 1.0 - z2 / 6.0 + z2 * z2 / 120.0 - z2 * z2 * z2 / 5040.0
    end
    return sin(z) / z
end

@inline function sinc1_big(z::BigFloat)
    iszero(z) && return one(BigFloat)
    return sin(z) / z
end

@inline u_exact(x::Float64) = cos(50.0 * x)
@inline u_exact_mapped(xi::Float64) = cos(25.0 * (xi + 1.0))

function rhs_physical(x::Float64)
    z_minus = (OMEGA * x - 50.0) * x
    z_plus = (OMEGA * x + 50.0) * x
    return u_exact(x) + 0.5 * x * (sinc1(z_minus) + sinc1(z_plus))
end

function rhs_physical_big(x::BigFloat)
    omega = BigFloat(OMEGA)
    fifty = BigFloat(50)
    z_minus = (omega * x - fifty) * x
    z_plus = (omega * x + fifty) * x
    return cos(fifty * x) + x / BigFloat(2) * (sinc1_big(z_minus) + sinc1_big(z_plus))
end

@inline function rhs_mapped(xi::Float64)
    x = 0.5 * (xi + 1.0)
    return rhs_physical(x)
end

function rhs_mapped_big(xi::BigFloat)
    x = (xi + one(BigFloat)) / BigFloat(2)
    return rhs_physical_big(x)
end

function u_exact_mapped_big(xi::BigFloat)
    return cos(BigFloat(25) * (xi + one(BigFloat)))
end

@inline function mapped_kernel(xi::Float64, eta::Float64)
    return 0.5 * cos(OMEGA * (xi + 1.0) * (eta + 1.0) / 4.0)
end

@inline function duffy_eta(xi::Float64, q::Float64)
    return (1.0 + xi) * (1.0 + q) / 2.0 - 1.0
end

@inline function voltmat_absorbed_mapped_kernel(xi::Float64, q::Float64)
    # VoltMat_Cheb's C interface expects coefficients of K(xi, eta(xi, q)).
    return mapped_kernel(xi, duffy_eta(xi, q))
end

function cheb_lobatto_nodes(npts::Int)
    npts >= 2 || throw(ArgumentError("npts must be at least 2"))
    return [cospi((j - 1) / (npts - 1)) for j in 1:npts]
end

function cheb_coeffs_lobatto(f, ncoeffs::Int)
    ncoeffs >= 2 || throw(ArgumentError("ncoeffs must be at least 2"))

    nodes = cheb_lobatto_nodes(ncoeffs)
    vals = [Float64(f(x)) for x in nodes]

    coeffs = FFTW.r2r(vals, FFTW.REDFT00)
    coeffs ./= ncoeffs - 1
    coeffs[1] *= 0.5
    coeffs[end] *= 0.5

    return Vector{Float64}(coeffs)
end

function cheb_coeffs_bigfloat_gauss(f, ncoeffs::Int, nquad::Int; precision_bits::Int=256)
    ncoeffs >= 1 || throw(ArgumentError("ncoeffs must be positive"))
    nquad >= ncoeffs || throw(ArgumentError("nquad must be at least ncoeffs"))

    return setprecision(BigFloat, precision_bits) do
        sums = zeros(BigFloat, ncoeffs)
        pi_big = big(pi)
        two = BigFloat(2)

        for j in 0:(nquad - 1)
            theta = pi_big * (BigFloat(j) + BigFloat(0.5)) / BigFloat(nquad)
            x = cos(theta)
            fx = f(x)

            if ncoeffs >= 1
                sums[1] += fx
            end
            if ncoeffs >= 2
                tkm2 = one(BigFloat)
                tkm1 = x
                sums[2] += fx * tkm1

                for k in 3:ncoeffs
                    tk = two * x * tkm1 - tkm2
                    sums[k] += fx * tk
                    tkm2, tkm1 = tkm1, tk
                end
            end
        end

        coeffs = Vector{Float64}(undef, ncoeffs)
        coeffs[1] = Float64(sums[1] / BigFloat(nquad))
        for k in 2:ncoeffs
            coeffs[k] = Float64(two * sums[k] / BigFloat(nquad))
        end
        coeffs
    end
end

function tail_stats(source_coeffs::AbstractVector{<:Real}, basis_len::Int)
    if length(source_coeffs) > basis_len
        tail = @view source_coeffs[(basis_len + 1):end]
    else
        first_tail = max(1, length(source_coeffs) - TAIL_LEN + 1)
        tail = @view source_coeffs[first_tail:end]
    end
    return Float64(norm(tail, Inf)), Float64(norm(tail))
end

function coeff_data(method::String, source_coeffs::Vector{Float64}, basis_len::Int)
    length(source_coeffs) >= basis_len ||
        throw(ArgumentError("source coefficient vector is shorter than basis_len"))
    tail_inf, tail_l2 = tail_stats(source_coeffs, basis_len)
    return ChebCoeffData(
        Vector{Float64}(@view source_coeffs[1:basis_len]),
        source_coeffs,
        method,
        length(source_coeffs),
        tail_inf,
        tail_l2,
    )
end

function rhs_coeff_data_current()
    source = cheb_coeffs_lobatto(rhs_mapped, BASIS_LEN)
    return coeff_data("lobatto_float64_len$(BASIS_LEN)", source, BASIS_LEN)
end

function rhs_coeff_data_oversampled(source_len::Int)
    source = cheb_coeffs_lobatto(rhs_mapped, source_len)
    return coeff_data("lobatto_float64_len$(source_len)_trunc$(BASIS_LEN)", source, BASIS_LEN)
end

function rhs_coeff_data_bigfloat(; source_len::Int=800, nquad::Int=4096, precision_bits::Int=256)
    source = cheb_coeffs_bigfloat_gauss(
        rhs_mapped_big,
        source_len,
        nquad;
        precision_bits=precision_bits,
    )
    return coeff_data(
        "cheb_gauss_bigfloat_len$(source_len)_nquad$(nquad)_p$(precision_bits)",
        source,
        BASIS_LEN,
    )
end

function exact_solution_coeff_data(; source_len::Int=800, nquad::Int=4096, precision_bits::Int=256)
    source = cheb_coeffs_bigfloat_gauss(
        u_exact_mapped_big,
        source_len,
        nquad;
        precision_bits=precision_bits,
    )
    return coeff_data(
        "exact_cheb_gauss_bigfloat_len$(source_len)_nquad$(nquad)_p$(precision_bits)",
        source,
        BASIS_LEN,
    )
end

function cheb2_coeffmat_lobatto(f, nrows::Int, ncols::Int)
    nrows >= 2 || throw(ArgumentError("nrows must be at least 2"))
    ncols >= 2 || throw(ArgumentError("ncols must be at least 2"))

    xnodes = cheb_lobatto_nodes(nrows)
    ynodes = cheb_lobatto_nodes(ncols)
    vals = Matrix{Float64}(undef, nrows, ncols)

    @inbounds for j in 1:ncols
        y = ynodes[j]
        for i in 1:nrows
            vals[i, j] = Float64(f(xnodes[i], y))
        end
    end

    coeffs = FFTW.r2r(vals, FFTW.REDFT00)
    coeffs ./= (nrows - 1) * (ncols - 1)
    coeffs[1, :] ./= 2.0
    coeffs[end, :] ./= 2.0
    coeffs[:, 1] ./= 2.0
    coeffs[:, end] ./= 2.0

    return Matrix{Float64}(coeffs)
end

function cheb_eval_many(coeffs::AbstractVector{<:Real}, xs::AbstractVector{<:Real})
    m = length(xs)
    n = length(coeffs)
    n == 1 && return fill(Float64(coeffs[1]), m)

    b1 = zeros(Float64, m)
    b2 = zeros(Float64, m)
    b0 = zeros(Float64, m)

    @inbounds for k in n:-1:2
        ck = Float64(coeffs[k])
        for i in 1:m
            x = Float64(xs[i])
            b0[i] = 2.0 * x * b1[i] - b2[i] + ck
        end
        b2, b1, b0 = b1, b0, b2
    end

    vals = Vector{Float64}(undef, m)
    @inbounds for i in 1:m
        x = Float64(xs[i])
        vals[i] = x * b1[i] - b2[i] + Float64(coeffs[1])
    end
    return vals
end

function evaluation_grid_data()
    cached = evaluation_grid_cache[]
    cached !== nothing && return cached

    xi_grid = collect(range(-1.0, 1.0; length=GRID_LEN))
    x_grid = @. 0.5 * (xi_grid + 1.0)
    u_exact_grid = @. cos(50.0 * x_grid)
    cached = (xi_grid=xi_grid, u_exact_grid=u_exact_grid)
    evaluation_grid_cache[] = cached
    return cached
end

function kernel_coeffs_for_depth(depth::Int)
    C = cheb2_coeffmat_lobatto(
        voltmat_absorbed_mapped_kernel,
        depth,
        depth,
    )
    @assert size(C) == (depth, depth)
    return C
end

function construct_voltmat_from_coeffs(C::Matrix{Float64})
    Vraw = VoltOpt.voltmat_square_from_coeffmat_fft_convolution(
        C,
        DEGREE_USED;
        tol=matrix_tol,
        use_fft_top=true,
    )
    V = sparse(Float64.(Vraw))
    dropzeros!(V)
    return V
end

function assemble_system_from_operator(V)
    A = spdiagm(0 => ones(Float64, BASIS_LEN)) + V
    dropzeros!(A)
    return A
end

function solve_system(A, rhs_coeff::Vector{Float64})
    return A \ rhs_coeff
end

function build_voltmat_system(depth::Int)
    local C
    local V
    operator_build_time = @elapsed begin
        C = kernel_coeffs_for_depth(depth)
        V = construct_voltmat_from_coeffs(C)
    end

    A = assemble_system_from_operator(V)
    return C, V, A, operator_build_time
end

function maxabs_sparse(A::SparseMatrixCSC)
    return nnz(A) == 0 ? 0.0 : maximum(abs, nonzeros(A))
end

function csv_cell(x)
    if x === missing || x === nothing
        return ""
    elseif x isa AbstractFloat
        xf = Float64(x)
        return isfinite(xf) ? @sprintf("%.17e", xf) : string(xf)
    elseif x isa Integer
        return string(x)
    else
        text = string(x)
        if occursin(",", text) || occursin("\"", text) ||
           occursin("\n", text) || occursin("\r", text)
            return "\"" * replace(text, "\"" => "\"\"") * "\""
        end
        return text
    end
end

function parse_csv_line(line::AbstractString)
    fields = String[]
    buf = IOBuffer()
    in_quotes = false
    i = firstindex(line)
    while i <= lastindex(line)
        c = line[i]
        if c == '"'
            if in_quotes && i < lastindex(line) && line[nextind(line, i)] == '"'
                write(buf, '"')
                i = nextind(line, i)
            else
                in_quotes = !in_quotes
            end
        elseif c == ',' && !in_quotes
            push!(fields, String(take!(buf)))
        else
            write(buf, c)
        end
        i = nextind(line, i)
    end
    push!(fields, String(take!(buf)))
    return fields
end

function read_csv_by_depth(path::AbstractString)
    rows = Dict{Int,Dict{Symbol,Any}}()
    isfile(path) || return rows

    lines = readlines(path)
    isempty(lines) && return rows

    header = Symbol.(parse_csv_line(lines[1]))
    for line in lines[2:end]
        isempty(strip(line)) && continue
        fields = parse_csv_line(line)
        row = Dict{Symbol,Any}()
        for (j, col) in enumerate(header)
            row[col] = j <= length(fields) ? fields[j] : ""
        end
        if haskey(row, :depth) && !isempty(string(row[:depth]))
            rows[parse(Int, string(row[:depth]))] = row
        end
    end
    return rows
end

function blank_sweep_row(depth::Int)
    row = Dict{Symbol,Any}(col => "" for col in SWEEP_COLUMNS)
    row[:depth] = depth
    row[:C_rows] = depth
    row[:C_cols] = depth
    row[:basis_len] = BASIS_LEN
    row[:degree_used] = DEGREE_USED
    row[:omega] = OMEGA
    row[:status] = "missing"
    return row
end

function row_dict(row)
    return Dict{Symbol,Any}(col => getproperty(row, col) for col in SWEEP_COLUMNS)
end

function write_sweep_rows(path::AbstractString, rows_by_depth::Dict{Int,Dict{Symbol,Any}})
    mkpath(dirname(path))
    open(path, "w") do io
        write_csv_header(io, SWEEP_COLUMNS)
        for depth in 2:60
            row = get(rows_by_depth, depth, blank_sweep_row(depth))
            println(io, join((csv_cell(get(row, col, "")) for col in SWEEP_COLUMNS), ","))
        end
        flush(io)
    end
    return path
end

function existing_or_nan(old_row, col::Symbol)
    if old_row !== nothing && haskey(old_row, col) && !isempty(string(old_row[col]))
        return old_row[col]
    end
    return NaN
end

function preserve_benchmark_columns!(new_row::Dict{Symbol,Any}, old_row)
    for col in (
        :operator_build_time,
        :solve_time,
        :benchmark_build_min_time,
        :benchmark_build_median_time,
        :benchmark_build_q25_time,
        :benchmark_build_q75_time,
        :benchmark_build_iqr_time,
        :benchmark_solve_min_time,
        :benchmark_solve_median_time,
        :benchmark_solve_q25_time,
        :benchmark_solve_q75_time,
        :benchmark_solve_iqr_time,
        :benchmark_build_samples,
        :benchmark_solve_samples,
        :benchmark_build_allocs,
        :benchmark_solve_allocs,
    )
        new_row[col] = existing_or_nan(old_row, col)
    end
    return new_row
end

function write_csv_header(io, columns)
    println(io, join(string.(columns), ","))
    flush(io)
    return nothing
end

function write_csv_row(io, columns, row)
    println(io, join((csv_cell(getproperty(row, col)) for col in columns), ","))
    flush(io)
    return nothing
end

function env_int(name::AbstractString, default::Int)
    raw = get(ENV, name, "")
    isempty(strip(raw)) && return default
    return parse(Int, raw)
end

function env_float(name::AbstractString, default::Float64)
    raw = get(ENV, name, "")
    isempty(strip(raw)) && return default
    return parse(Float64, raw)
end

function benchmark_summary(trial)
    times_sec = sort!(Float64.(trial.times) .* 1.0e-9)
    q25_time = quantile(times_sec, 0.25)
    q75_time = quantile(times_sec, 0.75)
    return (
        min_time=minimum(times_sec),
        median_time=median(times_sec),
        q25_time=q25_time,
        q75_time=q75_time,
        iqr_time=q75_time - q25_time,
        samples=length(trial.times),
        allocs=BenchmarkTools.allocs(trial),
    )
end

function primary_stat_value(stat, primary::Symbol)
    primary === :median && return stat.median_time
    primary === :min && return stat.min_time
    primary === :q25 && return stat.q25_time
    throw(ArgumentError("EX4_BENCH_PRIMARY_STAT must be median, min, or q25"))
end

function primary_stat_symbol()
    raw = lowercase(strip(get(ENV, "EX4_BENCH_PRIMARY_STAT", "median")))
    raw == "median" && return :median
    raw == "min" && return :min
    raw == "q25" && return :q25
    throw(ArgumentError("EX4_BENCH_PRIMARY_STAT must be median, min, or q25"))
end

function adaptive_build_samples(requested::Int, warmup_time::Float64, target_seconds::Float64)
    requested = max(requested, 8)
    warmup_time <= 0.0 && return requested
    estimate = floor(Int, target_seconds / warmup_time)
    return clamp(max(estimate, 8), 8, requested)
end

function adjusted_seconds(base_seconds::Float64, warmup_time::Float64, samples::Int)
    return max(base_seconds, 1.25 * max(warmup_time, eps(Float64)) * samples)
end

function diagnostic_rhs_cases()
    cases = ChebCoeffData[]
    push!(cases, rhs_coeff_data_current())
    for source_len in (400, 800, 1200, 2000)
        push!(cases, rhs_coeff_data_oversampled(source_len))
    end
    push!(cases, rhs_coeff_data_bigfloat())
    return cases
end

function solve_and_measure(A, rhs_coeff::Vector{Float64}, exact_coeff::Vector{Float64})
    solution_coeff = zeros(Float64, BASIS_LEN)
    solve_time = @elapsed begin
        factor = lu(A)
        solution_coeff = factor \ rhs_coeff
    end

    solve_residual = A * solution_coeff - rhs_coeff
    solve_residual_relative = norm(solve_residual) / max(norm(rhs_coeff), eps(Float64))

    exact_residual = A * exact_coeff - rhs_coeff
    exact_residual_inf = norm(exact_residual, Inf)
    exact_residual_relative = norm(exact_residual) / max(norm(rhs_coeff), eps(Float64))

    grid_data = evaluation_grid_data()
    u_num = cheb_eval_many(solution_coeff, grid_data.xi_grid)
    max_error = maximum(abs.(u_num .- grid_data.u_exact_grid))

    return (
        solution_coeff=solution_coeff,
        solve_time=solve_time,
        solve_residual_relative=solve_residual_relative,
        exact_residual_inf=exact_residual_inf,
        exact_residual_relative=exact_residual_relative,
        max_error=max_error,
    )
end

function diagnostic_row(
    rhs_data::ChebCoeffData,
    exact_data::ChebCoeffData,
    A,
    depth::Int,
)
    metrics = solve_and_measure(A, rhs_data.coeffs, exact_data.coeffs)
    status = metrics.max_error <= 2.0e-15 ? "ok_1e-15" : "limited"

    return (
        N=BASIS_LEN,
        depth=depth,
        rhs_method=rhs_data.method,
        rhs_coeff_len=rhs_data.source_len,
        rhs_tail_inf=rhs_data.tail_inf,
        rhs_tail_l2=rhs_data.tail_l2,
        exact_solution_coeff_tail_inf=exact_data.tail_inf,
        residual_inf_on_coeff=metrics.exact_residual_inf,
        residual_relative_on_coeff=metrics.exact_residual_relative,
        max_error_grid=metrics.max_error,
        solve_residual_relative=metrics.solve_residual_relative,
        status=status,
    )
end

function run_depth50_diagnostic()
    mkpath(DATA_DIR)
    depth = 50

    println("VoltMat depth=50 RHS diagnostic")
    println("basis_len   = ", BASIS_LEN)
    println("degree_used = ", DEGREE_USED)
    println("omega       = ", OMEGA)
    println("output_csv  = ", DIAGNOSTIC_CSV)
    println()

    println("Building depth=50 VoltMat system ...")
    _, _, A, operator_build_time = build_voltmat_system(depth)
    @printf("operator_build_time = %.6e s\n", operator_build_time)

    println("Building high-precision exact solution coefficients ...")
    exact_data = exact_solution_coeff_data()
    @printf(
        "exact method = %s, source_len = %d, tail_inf = %.6e, tail_l2 = %.6e\n",
        exact_data.method,
        exact_data.source_len,
        exact_data.tail_inf,
        exact_data.tail_l2,
    )

    rows = NamedTuple[]
    open(DIAGNOSTIC_CSV, "w") do io
        write_csv_header(io, DIAGNOSTIC_COLUMNS)

        for rhs_data in diagnostic_rhs_cases()
            row = diagnostic_row(rhs_data, exact_data, A, depth)
            write_csv_row(io, DIAGNOSTIC_COLUMNS, row)
            push!(rows, row)

            @printf(
                "rhs=%-48s len=%4d rhs_tail=%.3e exact_res=%.3e coeff_rel=%.3e max_err=%.3e solve_rel=%.3e status=%s\n",
                row.rhs_method,
                row.rhs_coeff_len,
                row.rhs_tail_inf,
                row.residual_inf_on_coeff,
                row.residual_relative_on_coeff,
                row.max_error_grid,
                row.solve_residual_relative,
                row.status,
            )
        end
    end

    best = rows[argmin([r.max_error_grid for r in rows])]
    println()
    @printf(
        "Diagnostic best RHS method: %s, max_error = %.17e, exact_residual_inf = %.17e\n",
        best.rhs_method,
        best.max_error_grid,
        best.residual_inf_on_coeff,
    )
    if best.max_error_grid > 2.0e-15
        @printf(
            "Limiting factor at depth=50 appears to be exact/system residual %.3e and Float64 solve/evaluation roundoff %.3e, not RHS tail %.3e.\n",
            best.residual_inf_on_coeff,
            best.solve_residual_relative,
            best.rhs_tail_inf,
        )
    end
    println("Diagnostic CSV written to: ", DIAGNOSTIC_CSV)
    return rows
end

function sweep_failure_row(depth::Int, rhs_data::ChebCoeffData, err)
    return (
        depth=depth,
        C_rows=depth,
        C_cols=depth,
        basis_len=BASIS_LEN,
        degree_used=DEGREE_USED,
        omega=OMEGA,
        rhs_method=rhs_data.method,
        rhs_coeff_len=rhs_data.source_len,
        rhs_tail_inf=rhs_data.tail_inf,
        exact_residual_inf=NaN,
        exact_residual_relative=NaN,
        max_error=NaN,
        residual_relative=NaN,
        operator_build_time=NaN,
        solve_time=NaN,
        benchmark_build_min_time=NaN,
        benchmark_build_median_time=NaN,
        benchmark_build_q25_time=NaN,
        benchmark_build_q75_time=NaN,
        benchmark_build_iqr_time=NaN,
        benchmark_solve_min_time=NaN,
        benchmark_solve_median_time=NaN,
        benchmark_solve_q25_time=NaN,
        benchmark_solve_q75_time=NaN,
        benchmark_solve_iqr_time=NaN,
        benchmark_build_samples=0,
        benchmark_solve_samples=0,
        benchmark_build_allocs=NaN,
        benchmark_solve_allocs=NaN,
        maxabs_V=NaN,
        maxabs_A=NaN,
        nnz_V=0,
        nnz_A=0,
        status="failed",
        error_message=sprint(showerror, err),
    )
end

function run_one_depth(depth::Int, rhs_data::ChebCoeffData, exact_data::ChebCoeffData)
    C, V, A, operator_build_time = build_voltmat_system(depth)
    metrics = solve_and_measure(A, rhs_data.coeffs, exact_data.coeffs)

    return (
        depth=depth,
        C_rows=size(C, 1),
        C_cols=size(C, 2),
        basis_len=BASIS_LEN,
        degree_used=DEGREE_USED,
        omega=OMEGA,
        rhs_method=rhs_data.method,
        rhs_coeff_len=rhs_data.source_len,
        rhs_tail_inf=rhs_data.tail_inf,
        exact_residual_inf=metrics.exact_residual_inf,
        exact_residual_relative=metrics.exact_residual_relative,
        max_error=metrics.max_error,
        residual_relative=metrics.solve_residual_relative,
        operator_build_time=operator_build_time,
        solve_time=metrics.solve_time,
        benchmark_build_min_time=NaN,
        benchmark_build_median_time=NaN,
        benchmark_build_q25_time=NaN,
        benchmark_build_q75_time=NaN,
        benchmark_build_iqr_time=NaN,
        benchmark_solve_min_time=NaN,
        benchmark_solve_median_time=NaN,
        benchmark_solve_q25_time=NaN,
        benchmark_solve_q75_time=NaN,
        benchmark_solve_iqr_time=NaN,
        benchmark_build_samples=0,
        benchmark_solve_samples=0,
        benchmark_build_allocs=NaN,
        benchmark_solve_allocs=NaN,
        maxabs_V=maxabs_sparse(V),
        maxabs_A=maxabs_sparse(A),
        nnz_V=nnz(V),
        nnz_A=nnz(A),
        status="success",
        error_message="",
    )
end

function selected_rhs_data_for_sweep()
    return rhs_coeff_data_bigfloat()
end

function main()
    mkpath(DATA_DIR)
    existing_rows = read_csv_by_depth(SWEEP_CSV)

    println("VoltMat N=200 depth sweep for oscillatory second-kind Volterra equation")
    println("project_root = ", PROJECT_ROOT)
    println("output_csv   = ", SWEEP_CSV)
    println("basis_len    = ", BASIS_LEN)
    println("degree_used  = ", DEGREE_USED)
    println("omega        = ", OMEGA)
    println("depths       = 2:60")

    println("Building selected high-precision RHS coefficients ...")
    rhs_data = selected_rhs_data_for_sweep()
    @printf(
        "rhs_method = %s, source_len = %d, discarded_tail_inf = %.6e, discarded_tail_l2 = %.6e\n",
        rhs_data.method,
        rhs_data.source_len,
        rhs_data.tail_inf,
        rhs_data.tail_l2,
    )

    println("Building high-precision exact solution coefficients ...")
    exact_data = exact_solution_coeff_data()
    @printf(
        "exact_solution_source_len = %d, discarded_tail_inf = %.6e\n",
        exact_data.source_len,
        exact_data.tail_inf,
    )

    best_depth = missing
    best_error = Inf
    rows_by_depth = copy(existing_rows)

    for depth in 2:60
        row = try
            run_one_depth(depth, rhs_data, exact_data)
        catch err
            sweep_failure_row(depth, rhs_data, err)
        end

        row_out = row_dict(row)
        preserve_benchmark_columns!(row_out, get(existing_rows, depth, nothing))
        rows_by_depth[depth] = row_out
        write_sweep_rows(SWEEP_CSV, rows_by_depth)

        if row.status == "success" && isfinite(row.max_error) && row.max_error < best_error
            best_depth = depth
            best_error = row.max_error
        end

        @printf(
            "depth = %2d, status = %-7s, max_error = %.6e, exact_res = %.6e, residual_relative = %.6e\n",
            depth,
            row.status,
            row.max_error,
            row.exact_residual_inf,
            row.residual_relative,
        )
        flush(stdout)
    end

    if best_depth === missing
        println("No successful depth completed.")
    else
        @printf(
            "Best depth = %d, max_error = %.17e\n",
            best_depth,
            best_error,
        )
    end

    println("CSV written to: ", SWEEP_CSV)
    return nothing
end

function benchmark_failure_row(depth::Int, rhs_data::ChebCoeffData, err)
    row = blank_sweep_row(depth)
    row[:rhs_method] = rhs_data.method
    row[:rhs_coeff_len] = rhs_data.source_len
    row[:rhs_tail_inf] = rhs_data.tail_inf
    row[:operator_build_time] = NaN
    row[:solve_time] = NaN
    row[:benchmark_build_min_time] = NaN
    row[:benchmark_build_median_time] = NaN
    row[:benchmark_build_q25_time] = NaN
    row[:benchmark_build_q75_time] = NaN
    row[:benchmark_build_iqr_time] = NaN
    row[:benchmark_solve_min_time] = NaN
    row[:benchmark_solve_median_time] = NaN
    row[:benchmark_solve_q25_time] = NaN
    row[:benchmark_solve_q75_time] = NaN
    row[:benchmark_solve_iqr_time] = NaN
    row[:benchmark_build_samples] = 0
    row[:benchmark_solve_samples] = 0
    row[:benchmark_build_allocs] = NaN
    row[:benchmark_solve_allocs] = NaN
    row[:status] = "benchmark_failed"
    row[:error_message] = sprint(showerror, err)
    return row
end

function benchmark_one_depth(depth::Int, rhs_data::ChebCoeffData;
                             build_samples::Int,
                             solve_samples::Int,
                             build_seconds::Float64,
                             solve_seconds::Float64,
                             primary::Symbol)
    C = kernel_coeffs_for_depth(depth)

    local V_warm
    warmup_build_time = @elapsed begin
        V_warm = construct_voltmat_from_coeffs(C)
    end
    A_warm = assemble_system_from_operator(V_warm)
    rhs = rhs_data.coeffs
    warmup_solve_time = @elapsed begin
        solve_system(A_warm, rhs)
    end
    GC.gc()

    effective_build_samples = adaptive_build_samples(
        build_samples,
        warmup_build_time,
        build_seconds,
    )
    effective_build_seconds = adjusted_seconds(
        build_seconds,
        warmup_build_time,
        effective_build_samples,
    )

    build_trial = run(
        @benchmarkable construct_voltmat_from_coeffs($C) evals=1;
        samples=effective_build_samples,
        seconds=effective_build_seconds,
    )
    build_stat = benchmark_summary(build_trial)

    V = construct_voltmat_from_coeffs(C)
    A = assemble_system_from_operator(V)
    effective_solve_seconds = adjusted_seconds(
        solve_seconds,
        warmup_solve_time,
        solve_samples,
    )

    solve_trial = run(
        @benchmarkable solve_system($A, $rhs) evals=1;
        samples=solve_samples,
        seconds=effective_solve_seconds,
    )
    solve_stat = benchmark_summary(solve_trial)
    primary_build_time = primary_stat_value(build_stat, primary)
    primary_solve_time = primary_stat_value(solve_stat, primary)

    row = blank_sweep_row(depth)
    row[:C_rows] = size(C, 1)
    row[:C_cols] = size(C, 2)
    row[:rhs_method] = rhs_data.method
    row[:rhs_coeff_len] = rhs_data.source_len
    row[:rhs_tail_inf] = rhs_data.tail_inf
    row[:operator_build_time] = primary_build_time
    row[:solve_time] = primary_solve_time
    row[:benchmark_build_min_time] = build_stat.min_time
    row[:benchmark_build_median_time] = primary_build_time
    row[:benchmark_build_q25_time] = build_stat.q25_time
    row[:benchmark_build_q75_time] = build_stat.q75_time
    row[:benchmark_build_iqr_time] = build_stat.iqr_time
    row[:benchmark_solve_min_time] = solve_stat.min_time
    row[:benchmark_solve_median_time] = primary_solve_time
    row[:benchmark_solve_q25_time] = solve_stat.q25_time
    row[:benchmark_solve_q75_time] = solve_stat.q75_time
    row[:benchmark_solve_iqr_time] = solve_stat.iqr_time
    row[:benchmark_build_samples] = build_stat.samples
    row[:benchmark_solve_samples] = solve_stat.samples
    row[:benchmark_build_allocs] = build_stat.allocs
    row[:benchmark_solve_allocs] = solve_stat.allocs
    row[:maxabs_V] = maxabs_sparse(V)
    row[:maxabs_A] = maxabs_sparse(A)
    row[:nnz_V] = nnz(V)
    row[:nnz_A] = nnz(A)
    row[:status] = "success"
    row[:error_message] = ""
    return row
end

function merge_benchmark_row!(rows_by_depth, depth::Int, bench_row::Dict{Symbol,Any})
    row = get(rows_by_depth, depth, blank_sweep_row(depth))
    for (key, value) in bench_row
        if key in (
            :operator_build_time,
            :solve_time,
            :benchmark_build_min_time,
            :benchmark_build_median_time,
            :benchmark_build_q25_time,
            :benchmark_build_q75_time,
            :benchmark_build_iqr_time,
            :benchmark_solve_min_time,
            :benchmark_solve_median_time,
            :benchmark_solve_q25_time,
            :benchmark_solve_q75_time,
            :benchmark_solve_iqr_time,
            :benchmark_build_samples,
            :benchmark_solve_samples,
            :benchmark_build_allocs,
            :benchmark_solve_allocs,
            :maxabs_V,
            :maxabs_A,
            :nnz_V,
            :nnz_A,
            :C_rows,
            :C_cols,
            :basis_len,
            :degree_used,
            :omega,
            :rhs_method,
            :rhs_coeff_len,
            :rhs_tail_inf,
            :status,
            :error_message,
        )
            row[key] = value
        end
    end
    rows_by_depth[depth] = row
    return row
end

function run_benchmark_depth_sweep()
    mkpath(DATA_DIR)
    build_samples = env_int("EX4_BENCH_BUILD_SAMPLES", 20)
    solve_samples = env_int("EX4_BENCH_SOLVE_SAMPLES", 100)
    build_seconds = env_float("EX4_BENCH_BUILD_SECONDS", 2.0)
    solve_seconds = env_float("EX4_BENCH_SOLVE_SECONDS", 2.0)
    primary = primary_stat_symbol()
    blas_threads_before = BLAS.get_num_threads()
    BLAS.set_num_threads(1)

    println("VoltMat BenchmarkTools depth sweep")
    println("output_csv           = ", SWEEP_CSV)
    println("basis_len            = ", BASIS_LEN)
    println("degree_used          = ", DEGREE_USED)
    println("build benchmark      = precomputed C -> VoltMat_Cheb V only")
    println("solve benchmark      = prebuilt A \\ rhs only")
    println("Julia threads        = ", Threads.nthreads())
    println("BLAS threads before  = ", blas_threads_before)
    println("BLAS threads active  = ", BLAS.get_num_threads())
    println("build_samples        = ", build_samples)
    println("solve_samples        = ", solve_samples)
    println("build_seconds        = ", build_seconds)
    println("solve_seconds        = ", solve_seconds)
    println("primary benchmark statistic = ", primary)

    rhs_data = selected_rhs_data_for_sweep()
    rows_by_depth = read_csv_by_depth(SWEEP_CSV)

    for depth in 2:60
        bench_row = try
            benchmark_one_depth(
                depth,
                rhs_data;
                build_samples=build_samples,
                solve_samples=solve_samples,
                build_seconds=build_seconds,
                solve_seconds=solve_seconds,
                primary=primary,
            )
        catch err
            benchmark_failure_row(depth, rhs_data, err)
        end

        merged = merge_benchmark_row!(rows_by_depth, depth, bench_row)
        write_sweep_rows(SWEEP_CSV, rows_by_depth)

        @printf(
            "depth = %2d, status = %-16s\n",
            depth,
            string(get(merged, :status, "")),
        )
        @printf(
            "  build: min=%.6e median=%.6e q25=%.6e q75=%.6e iqr=%.6e samples=%s allocs=%s\n",
            parse(Float64, string(get(merged, :benchmark_build_min_time, "NaN"))),
            parse(Float64, string(get(merged, :benchmark_build_median_time, "NaN"))),
            parse(Float64, string(get(merged, :benchmark_build_q25_time, "NaN"))),
            parse(Float64, string(get(merged, :benchmark_build_q75_time, "NaN"))),
            parse(Float64, string(get(merged, :benchmark_build_iqr_time, "NaN"))),
            string(get(merged, :benchmark_build_samples, "")),
            string(get(merged, :benchmark_build_allocs, "")),
        )
        @printf(
            "  solve: min=%.6e median=%.6e q25=%.6e q75=%.6e iqr=%.6e samples=%s allocs=%s\n",
            parse(Float64, string(get(merged, :benchmark_solve_min_time, "NaN"))),
            parse(Float64, string(get(merged, :benchmark_solve_median_time, "NaN"))),
            parse(Float64, string(get(merged, :benchmark_solve_q25_time, "NaN"))),
            parse(Float64, string(get(merged, :benchmark_solve_q75_time, "NaN"))),
            parse(Float64, string(get(merged, :benchmark_solve_iqr_time, "NaN"))),
            string(get(merged, :benchmark_solve_samples, "")),
            string(get(merged, :benchmark_solve_allocs, "")),
        )
        println("  primary = ", primary)
        flush(stdout)
    end

    println("Benchmark CSV written to: ", SWEEP_CSV)
    return nothing
end

function requested_diagnostic()
    return any(arg -> arg == "--diagnose-depth50", ARGS)
end

function requested_benchmark()
    return any(arg -> arg == "--benchmark", ARGS)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    if requested_benchmark()
        run_benchmark_depth_sweep()
    elseif requested_diagnostic()
        run_depth50_diagnostic()
    else
        main()
    end
end
