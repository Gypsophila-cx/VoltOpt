# Adaptive compVolt diagnostic for increasingly oscillatory smooth kernels.
#
# Run from any working directory with:
#   julia --project=. test/compvolt_adaptive/run_adaptive_oscillation.jl

using BenchmarkTools
using Printf
using Statistics
using VoltOpt

include(joinpath(@__DIR__, "oscillation_common.jl"))
using .CompVoltAdaptiveOscillation

const SCRIPT_DIR = @__DIR__
const DATA_DIR = joinpath(SCRIPT_DIR, "data")
const OUTPUT_CSV = joinpath(DATA_DIR, "adaptive_oscillation.csv")
const BENCHMARK_SAMPLES = 7
const BENCHMARK_SECONDS = 10.0
const RELATIVE_ERROR_LIMIT = 5.0e-10

function benchmark_adaptive_evaluation(omega::Real, cu::AbstractVector)
    evaluate = () -> CompVoltAdaptiveOscillation.adaptive_solution(omega, cu)

    # Complete compilation and one full execution before collecting timings.
    warm_result = evaluate()
    benchmark = @benchmarkable ($evaluate)() evals=1
    trial = run(
        benchmark;
        samples=BENCHMARK_SAMPLES,
        seconds=BENCHMARK_SECONDS,
    )
    times = sort!(Float64.(trial.times) .* 1.0e-9)
    isempty(times) && error("BenchmarkTools returned no samples for omega=$omega.")

    return warm_result, (
        time_min=minimum(times),
        time_q25=quantile(times, 0.25),
        time_median=median(times),
        samples_run=length(times),
    )
end

function csv_value(value)
    value isa Bool && return lowercase(string(value))
    return string(value)
end

function write_results_csv(path::AbstractString, rows::Vector{<:NamedTuple})
    headers = [
        "case",
        "omega",
        "N",
        "input_length",
        "M0",
        "accepted_M",
        "accepted_npts",
        "Mref",
        "eta",
        "theta",
        "refinements",
        "converged",
        "tol",
        "Mmax",
        "evaluation_time",
        "time_min",
        "time_q25",
        "time_median",
        "samples_run",
        "error_inf",
        "relative_error",
    ]

    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(headers, ','))
        for row in rows
            println(io, join((csv_value(getfield(row, Symbol(name))) for name in headers), ','))
        end
    end
    return path
end

function validate_results(rows::Vector{<:NamedTuple})
    length(rows) == length(CompVoltAdaptiveOscillation.OMEGAS) || error(
        "Result count $(length(rows)) does not match frequency count " *
        "$(length(CompVoltAdaptiveOscillation.OMEGAS)).",
    )
    all(row -> row.converged, rows) || error("At least one adaptive case did not converge.")
    all(row -> row.accepted_npts == row.accepted_M + 1, rows) || error(
        "accepted_npts is inconsistent with accepted_M.",
    )
    all(row -> max(row.eta, row.theta) <= row.tol, rows) || error(
        "At least one accepted case violates the requested tolerance.",
    )
    all(row -> row.relative_error <= RELATIVE_ERROR_LIMIT, rows) || error(
        "At least one adaptive/reference comparison exceeds $RELATIVE_ERROR_LIMIT.",
    )
    all(
        row -> all(
            isfinite,
            (
                row.eta,
                row.theta,
                row.evaluation_time,
                row.time_min,
                row.time_q25,
                row.time_median,
                row.error_inf,
                row.relative_error,
            ),
        ),
        rows,
    ) || error("A non-finite diagnostic value was produced.")

    accepted = [row.accepted_M for row in rows]
    length(unique(accepted)) >= 2 || error("The experiment found only one accepted grid level.")
    accepted[end] > accepted[1] || error(
        "The highest-frequency grid is not larger than the lowest-frequency grid.",
    )
    any(row -> row.refinements > 1, rows) || error(
        "No medium/high-frequency case required multiple refinements.",
    )
    return nothing
end

function run_adaptive_oscillation(; output_csv::AbstractString=OUTPUT_CSV)
    cu = CompVoltAdaptiveOscillation.input_coefficients()
    length(cu) == CompVoltAdaptiveOscillation.INPUT_LENGTH || error(
        "Unexpected input coefficient length $(length(cu)).",
    )

    rows = NamedTuple[]
    println("Adaptive compVolt oscillation diagnostic")
    println("input u(x)       = exp(x)")
    println("kernel           = cos(omega*(x-t))*u(t)")
    println("input degree N   = ", CompVoltAdaptiveOscillation.INPUT_DEGREE)
    println("input length     = ", length(cu))
    println("M0               = ", CompVoltAdaptiveOscillation.INITIAL_DEGREE)
    println("tol              = ", CompVoltAdaptiveOscillation.TOLERANCE)
    println("Mmax             = ", CompVoltAdaptiveOscillation.MAX_DEGREE)
    println("timing samples   = ", BENCHMARK_SAMPLES)
    println("output CSV       = ", output_csv)

    for omega in CompVoltAdaptiveOscillation.OMEGAS
        adaptive, timing = benchmark_adaptive_evaluation(omega, cu)
        Mref, reference = CompVoltAdaptiveOscillation.fixed_reference(
            omega, adaptive.M, cu,
        )
        error_inf, relative_error = CompVoltAdaptiveOscillation.comparison_error(
            adaptive.coefficients,
            reference;
            evaluation_degree=CompVoltAdaptiveOscillation.EVALUATION_DEGREE,
        )

        row = (
            case="oscillation_omega_$(omega)",
            omega=omega,
            N=CompVoltAdaptiveOscillation.INPUT_DEGREE,
            input_length=length(cu),
            M0=CompVoltAdaptiveOscillation.INITIAL_DEGREE,
            accepted_M=adaptive.M,
            accepted_npts=adaptive.M + 1,
            Mref=Mref,
            eta=adaptive.eta,
            theta=adaptive.theta,
            refinements=adaptive.refinements,
            converged=adaptive.converged,
            tol=CompVoltAdaptiveOscillation.TOLERANCE,
            Mmax=CompVoltAdaptiveOscillation.MAX_DEGREE,
            evaluation_time=timing.time_min,
            time_min=timing.time_min,
            time_q25=timing.time_q25,
            time_median=timing.time_median,
            samples_run=timing.samples_run,
            error_inf=error_inf,
            relative_error=relative_error,
        )
        push!(rows, row)

        @printf(
            "omega=%3d  M=%4d  npts=%4d  refs=%d  eta=%.3e  theta=%.3e  error=%.3e  rel=%.3e  time[min/q25/med]=%.4f/%.4f/%.4f s\n",
            omega,
            row.accepted_M,
            row.accepted_npts,
            row.refinements,
            row.eta,
            row.theta,
            row.error_inf,
            row.relative_error,
            row.time_min,
            row.time_q25,
            row.time_median,
        )
    end

    validate_results(rows)
    write_results_csv(output_csv, rows)
    println("CSV saved to: ", output_csv)
    return rows
end

run_adaptive_oscillation()
