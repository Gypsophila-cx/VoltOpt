using Pkg
using BenchmarkTools
using Printf
using SparseArrays

function find_project_root(start_dir::AbstractString)
    dir = abspath(start_dir)
    while true
        if isfile(joinpath(dir, "Project.toml")) && isdir(joinpath(dir, "src"))
            return dir
        end
        parent = dirname(dir)
        parent == dir && error("Could not locate VoltOpt project root from $start_dir")
        dir = parent
    end
end

const SCRIPT_DIR = @__DIR__
const PROJECT_ROOT = find_project_root(SCRIPT_DIR)
const GUTLEB_OLVER_PROJECT = joinpath(PROJECT_ROOT, "others", "GutlebOlver")
const OUTPUT_CSV = joinpath(SCRIPT_DIR, "bench_end2end_fixedC_varyN.csv")

include(joinpath(PROJECT_ROOT, "src", "VoltOpt.jl"))
using .VoltOpt

Pkg.activate(GUTLEB_OLVER_PROJECT; io=devnull)
using GutlebOlver

const DEFAULT_CSIZE = 20
const DEFAULT_N_VALUES = [100, 200, 400, 800, 1200, 1600, 2400, 3200, 4800, 6400, 9600, 12800]
const DEFAULT_SAMPLES = parse(Int, get(ENV, "SMOOTH_BENCH_SAMPLES", "5"))
const DEFAULT_SECONDS = parse(Float64, get(ENV, "SMOOTH_BENCH_SECONDS", "2.0"))
const DEFAULT_CHEB_COEFF_TOL = 64eps(Float64)
const DEFAULT_MATRIX_TOL = 64eps(Float64)

Kexp(x, y) = exp(-x * y)
Khat_cheb(x, y) = Kexp(x, (1 + x) * (1 + y) / 2 - 1)
Kgo(s, r) = 2.0 * Kexp(2.0s - 1.0, 2.0r - 1.0)

function parse_int_list(text::AbstractString, default::Vector{Int})
    stripped = strip(text)
    isempty(stripped) && return copy(default)
    return [parse(Int, strip(part)) for part in split(stripped, ',') if !isempty(strip(part))]
end

function fixed_size_coeffmat(C::AbstractMatrix{<:Real}, nrow::Int, ncol::Int)
    nrow > 0 || throw(ArgumentError("nrow must be positive"))
    ncol > 0 || throw(ArgumentError("ncol must be positive"))
    Cout = zeros(Float64, nrow, ncol)
    nr = min(nrow, size(C, 1))
    nc = min(ncol, size(C, 2))
    Cout[1:nr, 1:nc] .= @view C[1:nr, 1:nc]
    return Cout
end

function build_cheb_coeffs(Csize::Int; cheb_coeff_tol::Float64=DEFAULT_CHEB_COEFF_TOL)
    Craw = VoltOpt.adaptive_cheb2_coeffmat(Khat_cheb; tol=cheb_coeff_tol)
    return fixed_size_coeffmat(Craw, Csize, Csize)
end

function build_voltmat_cheb_end2end(
    N::Int,
    Csize::Int;
    cheb_coeff_tol::Float64=DEFAULT_CHEB_COEFF_TOL,
    matrix_tol::Float64=DEFAULT_MATRIX_TOL,
    use_fft_top::Bool=true,
    tail_half_bandwidth::Union{Nothing,Int}=nothing,
)
    C = build_cheb_coeffs(Csize; cheb_coeff_tol=cheb_coeff_tol)
    return VoltOpt.voltmat_square_from_coeffmat_fft_convolution(
        C,
        N;
        tol=matrix_tol,
        use_fft_top=use_fft_top,
        tail_half_bandwidth=tail_half_bandwidth,
    )
end

function build_gutleb_olver_end2end(N::Int, kerneldepth::Int)
    order = N + 1
    kdepth = max(1, min(kerneldepth, order))
    plan = precompute_triangular_kernel_plan(kdepth, true)
    coeffs = triangular_kernel_coeffs_from_plan(plan, Kgo)
    Vraw = triVolterraFullKernelOpP01_strictband_from_coeffs(coeffs, order, true)
    return build_secondkind_volterra_strictband(Vraw, order)
end

function median_sorted(values::Vector{Float64})
    n = length(values)
    n == 0 && throw(ArgumentError("cannot take median of an empty vector"))
    return isodd(n) ? values[div(n + 1, 2)] : 0.5 * (values[div(n, 2)] + values[div(n, 2) + 1])
end

function benchmark_call(f; samples::Int=DEFAULT_SAMPLES, seconds::Float64=DEFAULT_SECONDS)
    GC.gc()
    bench = @benchmarkable ($f)() evals=1
    all_times = Float64[]
    memory = 0
    allocs = 0
    passes = 0

    while length(all_times) < samples && passes < max(samples, 1)
        passes += 1
        remaining = samples - length(all_times)
        trial = run(bench; samples=remaining, seconds=seconds)
        append!(all_times, Float64.(trial.times))
        memory = max(memory, Int(trial.memory))
        allocs = max(allocs, Int(trial.allocs))
        isempty(trial.times) && break
    end

    isempty(all_times) && error("BenchmarkTools returned no samples")
    sort!(all_times)
    return (
        min_ms=first(all_times) * 1.0e-6,
        median_ms=median_sorted(all_times) * 1.0e-6,
        mean_ms=(sum(all_times) / length(all_times)) * 1.0e-6,
        memory_MiB=memory / 1024.0^2,
        allocs=allocs,
        samples_run=length(all_times),
    )
end

function sparse_bandwidths(A::SparseMatrixCSC)
    lower = 0
    upper = 0
    @inbounds for col in 1:size(A, 2)
        for p in A.colptr[col]:(A.colptr[col + 1] - 1)
            row = A.rowval[p]
            if row >= col
                lower = max(lower, row - col)
            else
                upper = max(upper, col - row)
            end
        end
    end
    return lower, upper
end

function structural_nnz(A)
    A isa SparseMatrixCSC && return nnz(A)
    try
        return GutlebOlver.safe_nnz(A)
    catch
        return count(!iszero, A)
    end
end

function matrix_bandwidths(A)
    A isa SparseMatrixCSC && return sparse_bandwidths(A)
    try
        bw = GutlebOlver.safe_bandwidth(A)
        return Int(bw[1]), Int(bw[2])
    catch
        S = sparse(A)
        return sparse_bandwidths(S)
    end
end

function success_method_result(build_function, samples::Int, seconds::Float64)
    A = build_function()
    lower_bw, upper_bw = matrix_bandwidths(A)
    stat = benchmark_call(build_function; samples=samples, seconds=seconds)
    return (
        ok=true,
        min_ms=stat.min_ms,
        median_ms=stat.median_ms,
        mean_ms=stat.mean_ms,
        memory_MiB=stat.memory_MiB,
        allocs=stat.allocs,
        samples_run=stat.samples_run,
        nnz=structural_nnz(A),
        rows=size(A, 1),
        cols=size(A, 2),
        bw_lower=lower_bw,
        bw_upper=upper_bw,
        error_message="",
    )
end

function failed_method_result(err)
    return (
        ok=false,
        min_ms="",
        median_ms="",
        mean_ms="",
        memory_MiB="",
        allocs="",
        samples_run=0,
        nnz="",
        rows="",
        cols="",
        bw_lower="",
        bw_upper="",
        error_message=sprint(showerror, err),
    )
end

function method_result(build_function, samples::Int, seconds::Float64)
    try
        return success_method_result(build_function, samples, seconds)
    catch err
        return failed_method_result(err)
    end
end

function row_status(cheb, gutleb, bandwidth_match::Bool)
    cheb.ok && gutleb.ok && bandwidth_match && return "success"
    cheb.ok && gutleb.ok && !bandwidth_match && return "bandwidth_mismatch"
    cheb.ok && !gutleb.ok && return "gutleb_failed"
    !cheb.ok && gutleb.ok && return "cheb_failed"
    return "both_failed"
end

function row_error_message(cheb, gutleb, bandwidth_match::Bool)
    parts = String[]
    cheb.ok || push!(parts, "Cheb: $(cheb.error_message)")
    gutleb.ok || push!(parts, "GutlebOlver: $(gutleb.error_message)")
    cheb.ok && gutleb.ok && !bandwidth_match &&
        push!(parts, "lower bandwidth mismatch: cheb=$(cheb.bw_lower), gutleb=$(gutleb.bw_lower)")
    return join(parts, " | ")
end

function benchmark_row(N::Int, Csize::Int, samples::Int, seconds::Float64)
    cheb = method_result(
        () -> build_voltmat_cheb_end2end(N, Csize),
        samples,
        seconds,
    )
    gutleb = method_result(
        () -> build_gutleb_olver_end2end(N, Csize),
        samples,
        seconds,
    )
    bandwidth_match = cheb.ok && gutleb.ok && cheb.bw_lower == gutleb.bw_lower
    status = row_status(cheb, gutleb, bandwidth_match)

    @printf("N=%d, C=%d\n", N, Csize)
    @printf("Cheb:   lower bw = %s, upper bw = %s\n", cheb.bw_lower, cheb.bw_upper)
    @printf("Gutleb: lower bw = %s, upper bw = %s\n", gutleb.bw_lower, gutleb.bw_upper)
    bandwidth_match || @warn "Lower bandwidth mismatch or failed bandwidth check" N Csize cheb_bw_lower=cheb.bw_lower gutleb_bw_lower=gutleb.bw_lower status

    return (
        benchmark="fixedC_varyN",
        N=N,
        C_rows=Csize,
        C_cols=Csize,
        cheb_min_ms=cheb.min_ms,
        cheb_median_ms=cheb.median_ms,
        cheb_mean_ms=cheb.mean_ms,
        cheb_memory_MiB=cheb.memory_MiB,
        cheb_allocs=cheb.allocs,
        cheb_samples_run=cheb.samples_run,
        cheb_nnz=cheb.nnz,
        cheb_rows=cheb.rows,
        cheb_cols=cheb.cols,
        cheb_bw_lower=cheb.bw_lower,
        cheb_bw_upper=cheb.bw_upper,
        gutleb_min_ms=gutleb.min_ms,
        gutleb_median_ms=gutleb.median_ms,
        gutleb_mean_ms=gutleb.mean_ms,
        gutleb_memory_MiB=gutleb.memory_MiB,
        gutleb_allocs=gutleb.allocs,
        gutleb_samples_run=gutleb.samples_run,
        gutleb_nnz=gutleb.nnz,
        gutleb_rows=gutleb.rows,
        gutleb_cols=gutleb.cols,
        gutleb_bw_lower=gutleb.bw_lower,
        gutleb_bw_upper=gutleb.bw_upper,
        bandwidth_match=bandwidth_match,
        samples_requested=samples,
        seconds=seconds,
        status=status,
        error_message=row_error_message(cheb, gutleb, bandwidth_match),
    )
end

const CSV_HEADER = (
    :benchmark,
    :N,
    :C_rows,
    :C_cols,
    :cheb_min_ms,
    :cheb_median_ms,
    :cheb_mean_ms,
    :cheb_memory_MiB,
    :cheb_allocs,
    :cheb_samples_run,
    :cheb_nnz,
    :cheb_rows,
    :cheb_cols,
    :cheb_bw_lower,
    :cheb_bw_upper,
    :gutleb_min_ms,
    :gutleb_median_ms,
    :gutleb_mean_ms,
    :gutleb_memory_MiB,
    :gutleb_allocs,
    :gutleb_samples_run,
    :gutleb_nnz,
    :gutleb_rows,
    :gutleb_cols,
    :gutleb_bw_lower,
    :gutleb_bw_upper,
    :bandwidth_match,
    :samples_requested,
    :seconds,
    :status,
    :error_message,
)

function csv_escape(x)
    text = x === nothing ? "" : string(x)
    if occursin(',', text) || occursin('"', text) || occursin('\n', text) || occursin('\r', text)
        return "\"" * replace(text, "\"" => "\"\"") * "\""
    end
    return text
end

function write_csv(path::AbstractString, rows::Vector{<:NamedTuple})
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(string.(CSV_HEADER), ","))
        for row in rows
            println(io, join((csv_escape(getproperty(row, col)) for col in CSV_HEADER), ","))
        end
    end
    return path
end

function bench_end2end_fixedC_varyN(;
    Csize::Int=DEFAULT_CSIZE,
    N_values::Vector{Int}=parse_int_list(get(ENV, "SMOOTH_BENCH_NS", ""), DEFAULT_N_VALUES),
    samples::Int=DEFAULT_SAMPLES,
    seconds::Float64=DEFAULT_SECONDS,
    output_csv::String=OUTPUT_CSV,
)
    rows = NamedTuple[]
    @printf("Smooth Volterra end-to-end benchmark: fixed C, varying N\n")
    @printf("Csize=%d, N_values=%s\n", Csize, N_values)
    @printf("samples=%d, seconds=%.3f\n", samples, seconds)
    @printf("output_csv=%s\n", output_csv)

    for N in N_values
        row = benchmark_row(N, Csize, samples, seconds)
        push!(rows, row)
        @printf("  Cheb median ms   = %s, samples = %s\n", row.cheb_median_ms, row.cheb_samples_run)
        @printf("  Gutleb median ms = %s, samples = %s\n", row.gutleb_median_ms, row.gutleb_samples_run)
        @printf("  bandwidth_match  = %s, status = %s\n\n", row.bandwidth_match, row.status)
        write_csv(output_csv, rows)
    end

    written = write_csv(output_csv, rows)
    @printf("CSV saved to: %s\n", written)
    return rows
end

bench_end2end_fixedC_varyN()
