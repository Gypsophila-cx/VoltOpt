using Pkg

const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const GUTLEB_OLVER_PROJECT = joinpath(ROOT, "others", "GutlebOlver")
const OUTPUT_CSV = joinpath(@__DIR__, "data", "section522_k100_sparse_repeated_results.csv")
const REPEATS = max(5, parse(Int, get(ENV, "GUTLEB_OLVER_REPEATS", "5")))

Pkg.activate(joinpath(ROOT, "others", "GutlebOlver"))

using SparseGOWorkflow111

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
)

bandwidth_part(bw, index::Integer) =
    bw === missing ? missing : Tuple(bw)[index]

function median_time(values)
    sorted = sort(values)
    n = length(sorted)
    n == 0 && throw(ArgumentError("cannot compute median of an empty vector"))
    return isodd(n) ? sorted[(n + 1) ÷ 2] : (sorted[n ÷ 2] + sorted[n ÷ 2 + 1]) / 2
end

mean_time(values) = sum(values) / length(values)

function timed_solve()
    result = Ref{Any}()
    elapsed = @elapsed begin
        result[] = solve_section522_k100_sparse(N=128, kerneldepth=12)
    end
    return elapsed, result[]
end

function benchmark_row()
    warmup = solve_section522_k100_sparse(N=128, kerneldepth=12)
    warmup.status == "pass" ||
        error("warmup solve did not pass; status=$(warmup.status)")

    times = Float64[]
    results = Any[]
    for _ = 1:REPEATS
        elapsed, result = timed_solve()
        push!(times, elapsed)
        push!(results, result)
    end

    row = results[end]
    all_passed = all(result -> result.status == "pass", results)

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
        status=all_passed ? row.status : "fail",
        elapsed_times=join(string.(times), ";"),
    )
end

function csv_escape(value)
    text = value === missing ? "" : string(value)
    if occursin(',', text) || occursin('"', text) || occursin('\n', text) || occursin('\r', text)
        return "\"" * replace(text, "\"" => "\"\"") * "\""
    end
    return text
end

function write_csv(path::AbstractString, row::NamedTuple)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(string.(CSV_COLUMNS), ","))
        println(io, join((csv_escape(getproperty(row, col)) for col in CSV_COLUMNS), ","))
    end
    return path
end

function print_summary(row::NamedTuple, csv_path::AbstractString)
    println("Gutleb-Olver Section 5.2.2 k=100 repeated sparse benchmark")
    println("project = ", GUTLEB_OLVER_PROJECT)
    println("N = ", row.N)
    println("kerneldepth = ", row.kerneldepth)
    println("repeats = ", row.repeats)
    println("warmup_done = ", row.warmup_done)
    println("elapsed_times = ", row.elapsed_times)
    println("min_time = ", row.min_time)
    println("median_time = ", row.median_time)
    println("mean_time = ", row.mean_time)
    println("rhs_validation_error = ", row.rhs_validation_error)
    println("tau_residual = ", row.tau_residual)
    println("body_residual_relative = ", row.body_residual_relative)
    println("max_abs_error = ", row.max_abs_error)
    println("raw operator = ", row.raw_storage,
            " nnz=", row.raw_nnz,
            " bw=(", row.raw_lower_bw, ",", row.raw_upper_bw, ")")
    println("body operator = ", row.body_storage,
            " nnz=", row.body_nnz,
            " bw=(", row.body_lower_bw, ",", row.body_upper_bw, ")")
    println("final system = ", row.final_storage,
            " nnz=", row.final_nnz,
            " bw=(", row.final_lower_bw, ",", row.final_upper_bw, ")")
    println("status = ", row.status)
    println("csv = ", csv_path)
end

function main()
    row = benchmark_row()
    csv_path = write_csv(OUTPUT_CSV, row)
    print_summary(row, csv_path)
    return row
end

main()
