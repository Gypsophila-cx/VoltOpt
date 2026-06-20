using Pkg

const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const GUTLEB_OLVER_PROJECT = joinpath(ROOT, "others", "GutlebOlver")
const OUTPUT_CSV = joinpath(@__DIR__, "data", "section522_k100_sparse_vary_kerneldepth_results.csv")
const N = 128
const DEFAULT_KERNELDEPTHS = (4, 6, 8, 10, 12, 14, 16, 20, 24)
const REPEATS = max(5, parse(Int, get(ENV, "GUTLEB_OLVER_REPEATS", "5")))

function parse_kerneldepths(text::AbstractString)
    stripped = strip(text)
    isempty(stripped) && return collect(DEFAULT_KERNELDEPTHS)
    return [parse(Int, strip(part)) for part in split(stripped, ',') if !isempty(strip(part))]
end

const KERNELDEPTHS = parse_kerneldepths(
    get(ENV, "GUTLEB_OLVER_KERNELDEPTHS", join(string.(DEFAULT_KERNELDEPTHS), ",")),
)

Pkg.activate(joinpath(ROOT, "others", "GutlebOlver"))

using GutlebOlver

const CSV_COLUMNS = (
    :N,
    :kerneldepth,
    :repeats,
    :warmup_done,
    :min_time,
    :median_time,
    :mean_time,
    :rhs_validation_error,
    :tau_residual,
    :body_residual_relative,
    :max_abs_error,
    :raw_storage,
    :raw_nnz,
    :raw_lower_bw,
    :raw_upper_bw,
    :body_storage,
    :body_nnz,
    :body_lower_bw,
    :body_upper_bw,
    :final_storage,
    :final_nnz,
    :final_lower_bw,
    :final_upper_bw,
    :status,
    :elapsed_times,
    :error_message,
)

bandwidth_part(bw, index::Integer) =
    bw === missing ? missing : Tuple(bw)[index]

function median_time(values)
    sorted = sort(values)
    n = length(sorted)
    n == 0 && throw(ArgumentError("cannot compute median of an empty vector"))
    return isodd(n) ? sorted[div(n + 1, 2)] : (sorted[div(n, 2)] + sorted[div(n, 2) + 1]) / 2
end

mean_time(values) = sum(values) / length(values)

function timed_solve(kerneldepth::Integer)
    result = Ref{Any}()
    elapsed = @elapsed begin
        result[] = solve_section522_k100_sparse(N=N, kerneldepth=kerneldepth)
    end
    return elapsed, result[]
end

function aggregate_status(results)
    statuses = unique(string(result.status) for result in results)
    return length(statuses) == 1 ? only(statuses) : join(statuses, ";")
end

function success_row(kerneldepth::Integer, times, results)
    row = results[end]
    return (
        N=row.N,
        kerneldepth=row.kerneldepth,
        repeats=REPEATS,
        warmup_done=true,
        min_time=minimum(times),
        median_time=median_time(times),
        mean_time=mean_time(times),
        rhs_validation_error=row.rhs_validation_error,
        tau_residual=row.tau_residual,
        body_residual_relative=row.body_residual_relative,
        max_abs_error=row.max_abs_error,
        raw_storage=row.raw_operator_type,
        raw_nnz=row.raw_operator_nnz,
        raw_lower_bw=bandwidth_part(row.raw_operator_bandwidth, 1),
        raw_upper_bw=bandwidth_part(row.raw_operator_bandwidth, 2),
        body_storage=row.body_operator_type,
        body_nnz=row.body_operator_nnz,
        body_lower_bw=bandwidth_part(row.body_operator_bandwidth, 1),
        body_upper_bw=bandwidth_part(row.body_operator_bandwidth, 2),
        final_storage=row.final_system_type,
        final_nnz=row.final_system_nnz,
        final_lower_bw=bandwidth_part(row.full_bandwidth, 1),
        final_upper_bw=bandwidth_part(row.full_bandwidth, 2),
        status=aggregate_status(results),
        elapsed_times=join(string.(times), ";"),
        error_message="",
    )
end

function error_text(err)
    io = IOBuffer()
    showerror(io, err)
    return String(take!(io))
end

function error_row(kerneldepth::Integer, err; warmup_done::Bool=false)
    return (
        N=N,
        kerneldepth=kerneldepth,
        repeats=REPEATS,
        warmup_done=warmup_done,
        min_time=missing,
        median_time=missing,
        mean_time=missing,
        rhs_validation_error=missing,
        tau_residual=missing,
        body_residual_relative=missing,
        max_abs_error=missing,
        raw_storage=missing,
        raw_nnz=missing,
        raw_lower_bw=missing,
        raw_upper_bw=missing,
        body_storage=missing,
        body_nnz=missing,
        body_lower_bw=missing,
        body_upper_bw=missing,
        final_storage=missing,
        final_nnz=missing,
        final_lower_bw=missing,
        final_upper_bw=missing,
        status="error",
        elapsed_times="",
        error_message=error_text(err),
    )
end

function benchmark_one(kerneldepth::Integer)
    println("kerneldepth=", kerneldepth, " warmup...")
    warmup_done = false
    try
        warmup = solve_section522_k100_sparse(N=N, kerneldepth=kerneldepth)
        warmup_done = true
        println("kerneldepth=", kerneldepth, " warmup_status=", warmup.status, " timed_repeats=", REPEATS)

        times = Float64[]
        results = Any[]
        for repeat = 1:REPEATS
            elapsed, result = timed_solve(kerneldepth)
            push!(times, elapsed)
            push!(results, result)
            println(
                "kerneldepth=", kerneldepth,
                " repeat=", repeat,
                " elapsed_time=", elapsed,
                " status=", result.status,
            )
        end
        return success_row(kerneldepth, times, results)
    catch err
        println("kerneldepth=", kerneldepth, " error=", error_text(err))
        return error_row(kerneldepth, err; warmup_done=warmup_done)
    end
end

function csv_escape(value)
    text = value === missing ? "" : string(value)
    if occursin(',', text) || occursin('"', text) || occursin('\n', text) || occursin('\r', text)
        return "\"" * replace(text, "\"" => "\"\"") * "\""
    end
    return text
end

function write_header(path::AbstractString)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(string.(CSV_COLUMNS), ","))
    end
    return path
end

function append_csv_row(path::AbstractString, row::NamedTuple)
    open(path, "a") do io
        println(io, join((csv_escape(getproperty(row, col)) for col in CSV_COLUMNS), ","))
    end
    return path
end

function print_row_summary(row)
    println(
        "kerneldepth=", row.kerneldepth,
        " min_time=", row.min_time,
        " median_time=", row.median_time,
        " mean_time=", row.mean_time,
        " raw_nnz=", row.raw_nnz,
        " raw_bw=(", row.raw_lower_bw, ",", row.raw_upper_bw, ")",
        " body_nnz=", row.body_nnz,
        " body_bw=(", row.body_lower_bw, ",", row.body_upper_bw, ")",
        " final_nnz=", row.final_nnz,
        " final_bw=(", row.final_lower_bw, ",", row.final_upper_bw, ")",
        " status=", row.status,
    )
end

function main()
    println("Gutleb-Olver Section 5.2.2 k=100 sparse varying-kerneldepth benchmark")
    println("project = ", GUTLEB_OLVER_PROJECT)
    println("N = ", N)
    println("kerneldepths = ", collect(KERNELDEPTHS))
    println("repeats = ", REPEATS)

    write_header(OUTPUT_CSV)
    rows = NamedTuple[]
    for kerneldepth in KERNELDEPTHS
        row = benchmark_one(kerneldepth)
        append_csv_row(OUTPUT_CSV, row)
        push!(rows, row)
        print_row_summary(row)
    end

    failed = [row.kerneldepth for row in rows if row.status == "error"]
    println("csv = ", OUTPUT_CSV)
    println("failed_kerneldepths = ", failed)
    return rows
end

main()
