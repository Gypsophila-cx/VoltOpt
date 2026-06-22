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

# ============================================================
# Paths
# ============================================================

scriptDir = @__DIR__
projectRoot = find_project_root(scriptDir)
gutlebOlverProject = joinpath(projectRoot, "others", "GutlebOlver")
dataDir = joinpath(@__DIR__, "data")
mkpath(dataDir)

include(joinpath(projectRoot, "src", "VoltOpt.jl"))
using .VoltOpt

Pkg.activate(gutlebOlverProject; io=devnull)
using GutlebOlver

defaultCsize = 20
defaultNValues = [100, 200, 400, 800, 1200, 1600, 2400, 3200, 4800, 6400, 9600, 12800]
defaultSamples = parse(Int, get(ENV, "SMOOTH_BENCH_SAMPLES", "5"))
defaultSeconds = parse(Float64, get(ENV, "SMOOTH_BENCH_SECONDS", "2.0"))
defaultChebCoeffTol = 64eps(Float64)
defaultMatrixTol = 64eps(Float64)

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

function build_cheb_coeffs(Csize::Int; cheb_coeff_tol::Float64=defaultChebCoeffTol)
    Craw = VoltOpt.adaptive_cheb2_coeffmat(Khat_cheb; tol=cheb_coeff_tol)
    return fixed_size_coeffmat(Craw, Csize, Csize)
end

function build_voltmat_cheb_end2end(
    N::Int,
    Csize::Int;
    cheb_coeff_tol::Float64=defaultChebCoeffTol,
    matrix_tol::Float64=defaultMatrixTol,
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

function benchmark_call(f; samples::Int=defaultSamples, seconds::Float64=defaultSeconds)
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
        time_min_sec=first(all_times) * 1.0e-9,
        time_median_sec=median_sorted(all_times) * 1.0e-9,
        time_mean_sec=(sum(all_times) / length(all_times)) * 1.0e-9,
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

function success_method_result(method::AbstractString, build_function, samples::Int, seconds::Float64)
    A = build_function()
    lower_bw, upper_bw = matrix_bandwidths(A)
    stat = benchmark_call(build_function; samples=samples, seconds=seconds)
    entries = structural_nnz(A)
    return (
        ok=true,
        method=method,
        time_min_sec=stat.time_min_sec,
        time_median_sec=stat.time_median_sec,
        time_mean_sec=stat.time_mean_sec,
        memory_MiB=stat.memory_MiB,
        allocs=stat.allocs,
        samples_run=stat.samples_run,
        nnz=entries,
        stored_entries=entries,
        packed_size=missing,
        rows=size(A, 1),
        cols=size(A, 2),
        bw_lower=lower_bw,
        bw_upper=upper_bw,
        output_format=string(typeof(A)),
        method_status="success",
        error_message="",
    )
end

function failed_method_result(method::AbstractString, err)
    return (
        ok=false,
        method=method,
        time_min_sec=missing,
        time_median_sec=missing,
        time_mean_sec=missing,
        memory_MiB=missing,
        allocs=missing,
        samples_run=0,
        nnz=missing,
        stored_entries=missing,
        packed_size=missing,
        rows=missing,
        cols=missing,
        bw_lower=missing,
        bw_upper=missing,
        output_format=missing,
        method_status="failed",
        error_message=sprint(showerror, err),
    )
end

function method_result(method::AbstractString, build_function, samples::Int, seconds::Float64)
    try
        return success_method_result(method, build_function, samples, seconds)
    catch err
        return failed_method_result(method, err)
    end
end

function row_status(cheb, gutleb)
    cheb.ok && gutleb.ok && return "success"
    cheb.ok && !gutleb.ok && return "gutleb_failed"
    !cheb.ok && gutleb.ok && return "cheb_failed"
    return "both_failed"
end

function row_error_message(cheb, gutleb)
    parts = String[]
    cheb.ok || push!(parts, "Cheb: $(cheb.error_message)")
    gutleb.ok || push!(parts, "GutlebOlver: $(gutleb.error_message)")
    return join(parts, " | ")
end

function bandwidth_status(cheb, gutleb)
    cheb.ok && gutleb.ok || return "unavailable"
    return cheb.bw_lower == gutleb.bw_lower ? "match" : "mismatch"
end

function bandwidth_message(cheb, gutleb)
    cheb.ok && gutleb.ok || return "bandwidth comparison unavailable because one or more methods failed"
    cheb.bw_lower == gutleb.bw_lower && return ""
    return "lower bandwidth mismatch: cheb=$(cheb.bw_lower), gutleb=$(gutleb.bw_lower)"
end

function csv_row(benchmark::AbstractString, N::Int, Csize::Int, result,
                 status::AbstractString, error_message::AbstractString,
                 bandwidth_status_value::AbstractString,
                 bandwidth_message_value::AbstractString,
                 samples::Int, seconds::Float64)
    return (
        benchmark=benchmark,
        N=N,
        C_rows=Csize,
        C_cols=Csize,
        C_size=Csize * Csize,
        method=result.method,
        matrix_rows=result.rows,
        matrix_cols=result.cols,
        nnz=result.nnz,
        stored_entries=result.stored_entries,
        packed_size=result.packed_size,
        output_format=result.output_format,
        lower_bw=result.bw_lower,
        upper_bw=result.bw_upper,
        t_build=result.time_min_sec,
        t_build_median=result.time_median_sec,
        t_build_mean=result.time_mean_sec,
        t_total=result.time_min_sec,
        t_total_median=result.time_median_sec,
        t_total_mean=result.time_mean_sec,
        memory_MiB=result.memory_MiB,
        allocs=result.allocs,
        samples_requested=samples,
        samples_run=result.samples_run,
        seconds=seconds,
        bandwidth_match=bandwidth_status_value == "match",
        bandwidth_status=bandwidth_status_value,
        bandwidth_message=bandwidth_message_value,
        requested_bandwidth=Csize,
        actual_bandwidth=result.bw_lower,
        method_status=result.method_status,
        status=status,
        error_message=error_message,
        method_error_message=result.error_message,
    )
end

function benchmark_rows(N::Int, Csize::Int, samples::Int, seconds::Float64)
    cheb = method_result(
        "smooth_cheb_fft_convolution",
        () -> build_voltmat_cheb_end2end(N, Csize),
        samples,
        seconds,
    )
    gutleb = method_result(
        "gutleb_olver_strictband",
        () -> build_gutleb_olver_end2end(N, Csize),
        samples,
        seconds,
    )
    bandwidth_match = cheb.ok && gutleb.ok && cheb.bw_lower == gutleb.bw_lower
    bw_status = bandwidth_status(cheb, gutleb)
    bw_message = bandwidth_message(cheb, gutleb)
    status = row_status(cheb, gutleb)
    err = row_error_message(cheb, gutleb)

    @printf("N=%d, C=%d\n", N, Csize)
    @printf("Cheb:   lower bw = %s, upper bw = %s\n", cheb.bw_lower, cheb.bw_upper)
    @printf("Gutleb: lower bw = %s, upper bw = %s\n", gutleb.bw_lower, gutleb.bw_upper)
    bw_status == "match" || @warn "Lower bandwidth mismatch or unavailable bandwidth check" N Csize cheb_bw_lower=cheb.bw_lower gutleb_bw_lower=gutleb.bw_lower status bandwidth_status=bw_status

    return (
        csv_row("fixedC_varyN", N, Csize, cheb, status, err, bw_status, bw_message, samples, seconds),
        csv_row("fixedC_varyN", N, Csize, gutleb, status, err, bw_status, bw_message, samples, seconds),
    )
end

csvHeader = (
    :benchmark,
    :N,
    :C_rows,
    :C_cols,
    :C_size,
    :method,
    :matrix_rows,
    :matrix_cols,
    :nnz,
    :stored_entries,
    :packed_size,
    :output_format,
    :lower_bw,
    :upper_bw,
    :t_build,
    :t_build_median,
    :t_build_mean,
    :t_total,
    :t_total_median,
    :t_total_mean,
    :memory_MiB,
    :allocs,
    :samples_requested,
    :samples_run,
    :seconds,
    :bandwidth_match,
    :bandwidth_status,
    :bandwidth_message,
    :requested_bandwidth,
    :actual_bandwidth,
    :method_status,
    :status,
    :error_message,
    :method_error_message,
)

function csv_cell(x)
    (x === nothing || x === missing) && return ""
    text = string(x)
    if occursin(',', text) || occursin('"', text) || occursin('\n', text) || occursin('\r', text)
        return "\"" * replace(text, "\"" => "\"\"") * "\""
    end
    return text
end

function write_csv_header(io)
    println(io, join(string.(csvHeader), ","))
    flush(io)
end

function write_csv_row(io, row::NamedTuple)
    println(io, join((csv_cell(getproperty(row, col)) for col in csvHeader), ","))
    flush(io)
end

function bench_end2end_fixedC_varyN(;
    Csize::Int=defaultCsize,
    N_values::Vector{Int}=parse_int_list(get(ENV, "SMOOTH_BENCH_NS", ""), defaultNValues),
    samples::Int=defaultSamples,
    seconds::Float64=defaultSeconds,
    output_csv::String=joinpath(dataDir, "bench_smooth_fixedC_varyN.csv"),
)
    results = NamedTuple[]
    @printf("Smooth Volterra end-to-end benchmark: fixed C, varying N\n")
    @printf("Csize=%d, N_values=%s\n", Csize, N_values)
    @printf("samples=%d, seconds=%.3f\n", samples, seconds)
    @printf("scriptDir=%s\n", scriptDir)
    @printf("projectRoot=%s\n", projectRoot)
    @printf("dataDir=%s\n", dataDir)
    @printf("output_csv=%s\n", output_csv)

    mkpath(dirname(output_csv))
    open(output_csv, "w") do io
        write_csv_header(io)
        for N in N_values
            rows = benchmark_rows(N, Csize, samples, seconds)
            for row in rows
                push!(results, row)
                write_csv_row(io, row)
                @printf(
                    "  %-28s min = %s s, median = %s s, samples = %s, status = %s\n",
                    row.method,
                    row.t_build,
                    row.t_build_median,
                    row.samples_run,
                    row.status,
                )
            end
            @printf("  bandwidth_status = %s\n\n", rows[1].bandwidth_status)
        end
    end

    @printf("CSV saved to: %s\n", output_csv)
    return results
end

bench_end2end_fixedC_varyN()
