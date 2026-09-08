# bench_weak_fixedN_varyC.jl
#
# Purpose:
#   Benchmark weakly singular Volterra matrix construction with fixed
#   truncation dimension N and varying Duffy coefficient size.
#
# Benchmark:
#   Uses Weak_VoltMat_Cheb on K(x,y) = exp(-x*y) with mu = 0.5, sigma = 0.0,
#   and beta = 0.5.
#
# Method:
#   Keeps N fixed, sweeps Duffy-Chebyshev coefficient sizes, and uses
#   BenchmarkTools to record coefficient/build time, memory, allocations,
#   sparsity diagnostics, and status rows.
#
# Output files:
#   data/bench_weak_fixedN_varyC.csv
#
# Run from the repository root:
#   julia --project=. test/weaksingular_bench/bench_weak_fixedN_varyC.jl

using BenchmarkTools
using Printf

# ============================================================
# Paths
# ============================================================

scriptDir = @__DIR__
projectRoot = dirname(dirname(scriptDir))
dataDir = joinpath(@__DIR__, "data")
mkpath(dataDir)

include(joinpath(projectRoot, "src", "VoltOpt.jl"))
using .VoltOpt

WeakCheb = VoltOpt.Weak_VoltMat_Cheb

# ============================================================
# Test problem
# ============================================================

Kexp(x, y) = exp(-x * y)

# ============================================================
# Helpers
# ============================================================

function median_sorted(values::Vector{Float64})
    n = length(values)
    n == 0 && throw(ArgumentError("cannot take median of an empty vector"))
    return isodd(n) ? values[div(n + 1, 2)] : 0.5 * (values[div(n, 2)] + values[div(n, 2) + 1])
end

function bench_call(f; samples::Int = 3, seconds::Float64 = 2.0)
    GC.gc()
    bench = @benchmarkable ($f)() evals=1
    all_times = Float64[]
    memory = 0
    allocs = 0
    passes = 0

    while length(all_times) < samples && passes < max(samples, 1)
        passes += 1
        remaining = samples - length(all_times)
        trial = run(bench; samples = remaining, seconds = seconds)
        append!(all_times, Float64.(trial.times))
        memory = max(memory, Int(trial.memory))
        allocs = max(allocs, Int(trial.allocs))
        isempty(trial.times) && break
    end

    isempty(all_times) && error("BenchmarkTools returned no samples")
    sort!(all_times)
    return (
        time_min_sec = first(all_times) * 1.0e-9,
        time_median_sec = median_sorted(all_times) * 1.0e-9,
        time_mean_sec = (sum(all_times) / length(all_times)) * 1.0e-9,
        memory_MiB = memory / 1024.0^2,
        allocs = allocs,
        samples_run = length(all_times),
    )
end

function duffy_coefficients(K, C_rows::Int, C_cols::Int, beta::Real)
    C_rows > 1 || throw(ArgumentError("C_rows must be at least 2"))
    C_cols > 1 || throw(ArgumentError("C_cols must be at least 2"))

    F = WeakCheb.sample_duffy_kernel(K, C_rows - 1, C_cols - 1, beta)
    return WeakCheb.cheb_coeffs_lobatto_2d(F)
end

function build_duffy_coefficients(K, C_rows::Int, C_cols::Int, beta::Real;
                                  samples::Int = 3, seconds::Float64 = 2.0)
    C = duffy_coefficients(K, C_rows, C_cols, beta)
    coeff_stat = bench_call(
        () -> duffy_coefficients(K, C_rows, C_cols, beta);
        samples = samples,
        seconds = seconds,
    )
    return C, coeff_stat
end

function weak_build(C, N::Int; mu::Real, sigma::Real, beta::Real)
    return VoltOpt.weak_voltmat_cheb(
        C;
        C_is_coeffs = true,
        mu = mu,
        sigma = sigma,
        beta = beta,
        N = N,
        output_format = :packed_lower_hessenberg,
        diagnostics = false,
        H_recurrence_mode = :block_i,
        return_info = true,
    )
end

function structural_nnz(V)
    return V isa VoltOpt.PackedLowerHessenberg ? length(V.values) : count(!iszero, V)
end

function packed_size(V)
    return V isa VoltOpt.PackedLowerHessenberg ? length(V.values) : missing
end

function upper_bandwidth(V)
    V isa VoltOpt.PackedLowerHessenberg && return size(V, 2) == 1 ? 0 : 1

    upper = 0
    for col in axes(V, 2), row in axes(V, 1)
        if row < col && !iszero(V[row, col])
            upper = max(upper, col - row)
        end
    end
    return upper
end

function csv_cell(x)
    (x === nothing || x === missing) && return ""
    text = string(x)
    if occursin(',', text) || occursin('"', text) || occursin('\n', text) || occursin('\r', text)
        return "\"" * replace(text, "\"" => "\"\"") * "\""
    end
    return text
end

csvHeader = (
    :benchmark,
    :N,
    :C_rows,
    :C_cols,
    :C_size,
    :mu,
    :sigma,
    :beta,
    :eta,
    :bandwidth,
    :lower_bw,
    :upper_bw,
    :top_rows,
    :matrix_rows,
    :matrix_cols,
    :nnz,
    :stored_entries,
    :packed_size,
    :output_format,
    :assembly_mode,
    :H_recurrence_mode,
    :moment_recurrence,
    :contraction_mode,
    :method,
    :t_build,
    :t_build_median,
    :t_build_mean,
    :t_coeff,
    :t_coeff_median,
    :t_coeff_mean,
    :t_coeff_samples_run,
    :t_solve,
    :t_total,
    :memory_MiB,
    :allocs,
    :samples_requested,
    :samples_run,
    :timing_interpolation,
    :timing_initial_moments,
    :timing_precombine,
    :timing_assembly,
    :timing_multiplier,
    :timing_diagnostics,
    :timing_total_reported,
    :status,
    :error_message,
)

function write_csv_header(io)
    println(io, join(string.(csvHeader), ","))
    flush(io)
end

function write_csv_row(io, row::NamedTuple)
    println(io, join((csv_cell(getproperty(row, col)) for col in csvHeader), ","))
    flush(io)
end

function success_row(benchmark::AbstractString, N::Int, C_rows::Int, C_cols::Int,
                     mu::Real, sigma::Real, beta::Real, coeff_stat,
                     build_stat, V, info, samples::Int, seconds::Float64)
    lower_bw = info.lower_bandwidth_theory
    return (
        benchmark = benchmark,
        N = N,
        C_rows = C_rows,
        C_cols = C_cols,
        C_size = C_rows * C_cols,
        mu = mu,
        sigma = sigma,
        beta = beta,
        eta = info.eta,
        bandwidth = lower_bw,
        lower_bw = lower_bw,
        upper_bw = upper_bandwidth(V),
        top_rows = missing,
        matrix_rows = size(V, 1),
        matrix_cols = size(V, 2),
        nnz = structural_nnz(V),
        stored_entries = info.stored_entries,
        packed_size = packed_size(V),
        output_format = info.output_format,
        assembly_mode = info.assembly_mode,
        H_recurrence_mode = info.H_recurrence_mode,
        moment_recurrence = info.moment_recurrence,
        contraction_mode = info.contraction_mode,
        method = "weak_cheb_packed_lower_hessenberg",
        t_build = build_stat.time_min_sec,
        t_build_median = build_stat.time_median_sec,
        t_build_mean = build_stat.time_mean_sec,
        t_coeff = coeff_stat.time_min_sec,
        t_coeff_median = coeff_stat.time_median_sec,
        t_coeff_mean = coeff_stat.time_mean_sec,
        t_coeff_samples_run = coeff_stat.samples_run,
        t_solve = missing,
        t_total = coeff_stat.time_min_sec + build_stat.time_min_sec,
        memory_MiB = build_stat.memory_MiB,
        allocs = build_stat.allocs,
        samples_requested = samples,
        samples_run = build_stat.samples_run,
        timing_interpolation = info.timings.interpolation,
        timing_initial_moments = info.timings.initial_moments,
        timing_precombine = info.timings.precombine,
        timing_assembly = info.timings.assembly,
        timing_multiplier = info.timings.multiplier,
        timing_diagnostics = info.timings.diagnostics,
        timing_total_reported = info.timings.total,
        status = "success",
        error_message = "",
    )
end

function failed_row(benchmark::AbstractString, N::Int, C_rows::Int, C_cols::Int,
                    mu::Real, sigma::Real, beta::Real, samples::Int,
                    seconds::Float64, err; t_coeff = missing)
    return (
        benchmark = benchmark,
        N = N,
        C_rows = C_rows,
        C_cols = C_cols,
        C_size = C_rows * C_cols,
        mu = mu,
        sigma = sigma,
        beta = beta,
        eta = missing,
        bandwidth = missing,
        lower_bw = missing,
        upper_bw = missing,
        top_rows = missing,
        matrix_rows = missing,
        matrix_cols = missing,
        nnz = missing,
        stored_entries = missing,
        packed_size = missing,
        output_format = "packed_lower_hessenberg",
        assembly_mode = missing,
        H_recurrence_mode = "block_i",
        moment_recurrence = "four_point",
        contraction_mode = "precombined_j",
        method = "weak_cheb_packed_lower_hessenberg",
        t_build = missing,
        t_build_median = missing,
        t_build_mean = missing,
        t_coeff = t_coeff,
        t_coeff_median = missing,
        t_coeff_mean = missing,
        t_coeff_samples_run = 0,
        t_solve = missing,
        t_total = missing,
        memory_MiB = missing,
        allocs = missing,
        samples_requested = samples,
        samples_run = 0,
        timing_interpolation = missing,
        timing_initial_moments = missing,
        timing_precombine = missing,
        timing_assembly = missing,
        timing_multiplier = missing,
        timing_diagnostics = missing,
        timing_total_reported = missing,
        status = "failed",
        error_message = sprint(showerror, err),
    )
end

# ============================================================
# Main benchmark
# ============================================================

function bench_weak_fixedN_varyC(;
    N::Int = 1000,
    C_sizes::Vector{Tuple{Int,Int}} = [(4, 4), (8, 8), (16, 16), (32, 32), (64, 64)],
    mu::Float64 = 0.5,
    sigma::Float64 = 0.0,
    beta::Float64 = 0.5,
    samples::Int = 3,
    seconds::Float64 = 2.0,
    output_csv::String = joinpath(dataDir, "bench_weak_fixedN_varyC.csv"),
)
    results = NamedTuple[]

    println("============================================================")
    println("Benchmark: weakly singular Chebyshev/JFP Volterra matrix")
    println("case                    = fixed N, varying C/bandwidth")
    println("kernel                  = exp(-x*y)")
    println("weak kernel factor      = (x-y)^(-mu)")
    println("N                       = $N")
    println("C sizes                 = $C_sizes")
    println("mu, sigma, beta         = ($mu, $sigma, $beta)")
    println("output_format           = packed_lower_hessenberg")
    println("samples                 = $samples")
    println("seconds                 = $seconds")
    println("scriptDir               = $scriptDir")
    println("projectRoot             = $projectRoot")
    println("dataDir                 = $dataDir")
    println("output_csv              = $output_csv")
    println("============================================================")

    # Compile the main path outside the measured loops.
    Cwarm = duffy_coefficients(Kexp, 4, 4, beta)
    weak_build(Cwarm, min(N, 64); mu = mu, sigma = sigma, beta = beta)

    mkpath(dirname(output_csv))
    open(output_csv, "w") do io
        write_csv_header(io)
        for (C_rows, C_cols) in C_sizes
            println()
            println("------------------------------------------------------------")
            @printf("N = %d, C = %d x %d\n", N, C_rows, C_cols)

            coeff_stat = missing
            r = try
                C, coeff_stat = build_duffy_coefficients(
                    Kexp, C_rows, C_cols, beta;
                    samples = samples,
                    seconds = seconds,
                )
                Vwarm, info = weak_build(C, N; mu = mu, sigma = sigma, beta = beta)
                build_stat = bench_call(
                    () -> weak_build(C, N; mu = mu, sigma = sigma, beta = beta);
                    samples = samples,
                    seconds = seconds,
                )
                success_row(
                    "fixedN_varyC", N, C_rows, C_cols, mu, sigma, beta,
                    coeff_stat, build_stat, Vwarm, info, samples, seconds,
                )
            catch err
                t_coeff = coeff_stat === missing ? missing : coeff_stat.time_min_sec
                failed_row(
                    "fixedN_varyC", N, C_rows, C_cols, mu, sigma, beta,
                    samples, seconds, err; t_coeff = t_coeff,
                )
            end

            push!(results, r)
            write_csv_row(io, r)

            if r.status == "success"
                @printf(
                    "Weak | build = %10.6f s | coeff = %10.6f s | nnz = %10d | lower_bw = %4d | upper_bw = %4d | mem = %8.2f MiB | status = %s\n",
                    r.t_build,
                    r.t_coeff,
                    r.nnz,
                    r.lower_bw,
                    r.upper_bw,
                    r.memory_MiB,
                    r.status,
                )
            else
                @warn "Benchmark point failed" N C_rows C_cols status=r.status error=r.error_message
            end
        end
    end

    println()
    println("CSV saved to: ", output_csv)

    return results
end

# ============================================================
# Run
# ============================================================

results = bench_weak_fixedN_varyC(
    N = 1000,
    C_sizes = [(2,2), (6,6), (14, 14), (30, 30), (62, 62), (126, 126), (254, 254), (510, 510)],
    mu = 0.5,
    sigma = 0.0,
    beta = 0.5,
    samples = 3,
    seconds = 2.0,
    output_csv = joinpath(dataDir, "bench_weak_fixedN_varyC.csv"),
)
