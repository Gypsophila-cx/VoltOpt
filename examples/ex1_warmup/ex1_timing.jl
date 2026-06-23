#!/usr/bin/env julia
#
# ex1_timing.jl
#
# Purpose:
#   Generate the warmup end-to-end timing table for the ex1 method comparison.
#
# Problem:
#   Same low-order Volterra integro-differential equation and method setup as
#   ex1_error.jl, with N sampled on a logarithmic grid from 10 to 1000.
#
# Method:
#   Uses BenchmarkTools to time the VoltMat / Chebyshev, VoltMat / Legendre,
#   and Gutleb--Olver / shifted Jacobi P^(1,0) solve paths.
#
# Inputs:
#   ex1_common.jl
#
# Output files:
#   data/ex1_timing_N10_1000_loggrid.csv
#
# Run from the repository root:
#   julia --project=. examples/ex1_warmup/ex1_timing.jl

using BenchmarkTools
using Statistics

include(joinpath(@__DIR__, "ex1_common.jl"))

const TIMING_NS = unique(round.(Int, exp10.(range(log10(10), log10(1000), length=25))))
const BENCHMARK_SAMPLES = 10
const BENCHMARK_SECONDS = 5.0
const BENCHMARK_EVALS = 1
const TIMING_SANITY_REPEATS = 3
const TIMING_CSV = joinpath(DATA_DIR, "ex1_timing_N10_1000_loggrid.csv")

const TIMING_HEADER = (
    :method,
    :basis,
    :N,
    :kerneldepth,
    :C_rows,
    :C_cols,
    :target_kernel_bandwidth,
    :benchmark_samples,
    :benchmark_evals,
    :benchmark_seconds,
    :min_elapsed_time,
    :median_elapsed_time,
    :mean_elapsed_time,
    :std_elapsed_time,
    :median_max_abs_error,
    :min_max_abs_error,
    :max_max_abs_error,
    :solver,
    :solver_backend,
    :fast_matrix_lband,
    :fast_matrix_uband,
    :fast_matrix_fill_rows,
    :raw_matrix_lband,
    :raw_matrix_uband,
    :raw_structure,
    :final_matrix_lband,
    :final_matrix_uband,
    :final_structure,
    :status,
    :error_message,
)

function finite_values(rows, field::Symbol)
    values = Float64[]
    for row in rows
        value = getproperty(row, field)
        if value isa Real
            fv = Float64(value)
            !isnan(fv) && push!(values, fv)
        end
    end
    return values
end

function sorted_median(values::Vector{Float64})
    isempty(values) && return NaN
    sorted = sort(values)
    n = length(sorted)
    return isodd(n) ? sorted[div(n + 1, 2)] :
           0.5 * (sorted[div(n, 2)] + sorted[div(n, 2) + 1])
end

function benchmark_times(spec, N::Int)
    spec_local = spec
    N_local = N
    bench = @benchmarkable run_method_case($spec_local, $N_local) evals=BENCHMARK_EVALS

    times_ns = Float64[]
    passes = 0
    while length(times_ns) < BENCHMARK_SAMPLES && passes < BENCHMARK_SAMPLES
        passes += 1
        remaining = BENCHMARK_SAMPLES - length(times_ns)
        trial = run(bench; samples=remaining, seconds=BENCHMARK_SECONDS)
        append!(times_ns, Float64.(trial.times))
        isempty(trial.times) && break
    end

    isempty(times_ns) && error("BenchmarkTools returned no samples")
    return times_ns .* 1.0e-9
end

function sanity_rows(spec, N::Int)
    rows = NamedTuple[]
    errors = String[]
    for _ in 1:TIMING_SANITY_REPEATS
        try
            row = run_method_case(spec, N)
            row.status == "success" ? push!(rows, row) :
                push!(errors, row.error_message)
        catch err
            push!(errors, sprint(showerror, err))
        end
    end
    return rows, errors
end

function timing_failed_row(spec, N::Int, message::AbstractString)
    return (
        method=spec.method,
        basis=spec.basis,
        N=N,
        kerneldepth=spec.kerneldepth,
        C_rows=spec.C_rows,
        C_cols=spec.C_cols,
        target_kernel_bandwidth=spec.runner === :go ? KERNELDEPTH : TARGET_KERNEL_BANDWIDTH,
        benchmark_samples=0,
        benchmark_evals=BENCHMARK_EVALS,
        benchmark_seconds=BENCHMARK_SECONDS,
        min_elapsed_time=NaN,
        median_elapsed_time=NaN,
        mean_elapsed_time=NaN,
        std_elapsed_time=NaN,
        median_max_abs_error=NaN,
        min_max_abs_error=NaN,
        max_max_abs_error=NaN,
        solver="fastalmostbanded",
        solver_backend="failed",
        fast_matrix_lband="",
        fast_matrix_uband="",
        fast_matrix_fill_rows="",
        raw_matrix_lband="",
        raw_matrix_uband="",
        raw_structure="",
        final_matrix_lband="",
        final_matrix_uband="",
        final_structure="",
        status="failed",
        error_message=message,
    )
end

function benchmark_method_case(spec, N::Int)
    sanity_successes, sanity_errors = sanity_rows(spec, N)
    isempty(sanity_successes) &&
        return timing_failed_row(spec, N, join(unique(sanity_errors), " | "))

    times_s = try
        benchmark_times(spec, N)
    catch err
        return timing_failed_row(
            spec,
            N,
            join(vcat(unique(sanity_errors), ["BenchmarkTools: " * sprint(showerror, err)]), " | "),
        )
    end

    representative = sanity_successes[end]
    max_abs_errors = finite_values(sanity_successes, :max_abs_error)
    solver_backends = unique(string(getproperty(row, :solver_backend)) for row in sanity_successes)
    solver_backend = length(solver_backends) == 1 ? first(solver_backends) : join(solver_backends, "|")
    solvers = unique(string(getproperty(row, :solver)) for row in sanity_successes)
    solver = length(solvers) == 1 ? first(solvers) : join(solvers, "|")

    message_parts = unique(sanity_errors)
    status = isempty(sanity_errors) && length(sanity_successes) == TIMING_SANITY_REPEATS ?
             "success" : "partial_success"
    if solver_backend != "fastalmostbanded"
        status = status == "success" ? "solver_backend_warning" : status
        push!(message_parts, "solver_backend=$(solver_backend)")
    end

    return (
        method=representative.method,
        basis=representative.basis,
        N=N,
        kerneldepth=representative.kerneldepth,
        C_rows=representative.C_rows,
        C_cols=representative.C_cols,
        target_kernel_bandwidth=representative.target_kernel_bandwidth,
        benchmark_samples=length(times_s),
        benchmark_evals=BENCHMARK_EVALS,
        benchmark_seconds=BENCHMARK_SECONDS,
        min_elapsed_time=minimum(times_s),
        median_elapsed_time=sorted_median(times_s),
        mean_elapsed_time=mean(times_s),
        std_elapsed_time=length(times_s) == 1 ? 0.0 : std(times_s),
        median_max_abs_error=sorted_median(max_abs_errors),
        min_max_abs_error=isempty(max_abs_errors) ? NaN : minimum(max_abs_errors),
        max_max_abs_error=isempty(max_abs_errors) ? NaN : maximum(max_abs_errors),
        solver=solver,
        solver_backend=solver_backend,
        fast_matrix_lband=representative.fast_matrix_lband,
        fast_matrix_uband=representative.fast_matrix_uband,
        fast_matrix_fill_rows=representative.fast_matrix_fill_rows,
        raw_matrix_lband=representative.raw_matrix_lband,
        raw_matrix_uband=representative.raw_matrix_uband,
        raw_structure=representative.raw_structure,
        final_matrix_lband=representative.final_matrix_lband,
        final_matrix_uband=representative.final_matrix_uband,
        final_structure=representative.final_structure,
        status=status,
        error_message=join(filter(!isempty, message_parts), " | "),
    )
end

function run_timing_sweep()
    rows = NamedTuple[]
    @printf("Running low-order VIDE BenchmarkTools timing sweep on N grid: %s\n",
            join(TIMING_NS, ", "))
    @printf("BenchmarkTools settings: samples=%d, evals=%d, seconds=%.2f\n",
            BENCHMARK_SAMPLES, BENCHMARK_EVALS, BENCHMARK_SECONDS)
    @printf("Sanity repeats per method/N: %d\n", TIMING_SANITY_REPEATS)
    @printf("Output CSV: %s\n", TIMING_CSV)
    @printf("Validating FastAlmostBanded solve consistency on N=10,20,50,100...\n")
    validate_fast_solver_consistency()

    for N in TIMING_NS
        for spec in METHOD_SPECS
            push!(rows, benchmark_method_case(spec, N))
        end
        recent = rows[(length(rows) - 2):length(rows)]
        status_summary = join(("$(row.method)/$(row.basis):$(row.status)" for row in recent), ", ")
        @printf("N=%4d  %s\n", N, status_summary)
        write_csv_rows(rows, TIMING_CSV, TIMING_HEADER)
    end

    written_path = write_csv_rows(rows, TIMING_CSV, TIMING_HEADER)
    successes = count(row -> row.status == "success", rows)
    partials = count(row -> row.status == "partial_success", rows)
    warnings = count(row -> row.status == "solver_backend_warning", rows)
    failures = length(rows) - successes - partials - warnings
    @printf("Wrote %d rows (%d success, %d partial, %d warnings, %d failed) to %s.\n",
            length(rows), successes, partials, warnings, failures, written_path)
    return rows
end

run_timing_sweep()
