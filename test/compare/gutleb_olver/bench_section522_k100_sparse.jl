using Pkg

const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const GUTLEB_OLVER_PROJECT = joinpath(ROOT, "others", "GutlebOlver")
const OUTPUT_CSV = joinpath(@__DIR__, "data", "section522_k100_sparse_results.csv")

Pkg.activate(joinpath(ROOT, "others", "GutlebOlver"))

using GutlebOlver

const CSV_COLUMNS = (
    :N,
    :kerneldepth,
    :elapsed_time,
    :rhs_validation_error,
    :tau_residual,
    :body_residual_relative,
    :max_abs_error,
    :raw_operator_type,
    :raw_operator_nnz,
    :raw_lower_bandwidth,
    :raw_upper_bandwidth,
    :body_operator_type,
    :body_operator_nnz,
    :body_lower_bandwidth,
    :body_upper_bandwidth,
    :final_system_type,
    :final_system_nnz,
    :final_lower_bandwidth,
    :final_upper_bandwidth,
    :density,
    :status,
)

bandwidth_part(bw, index::Integer) =
    bw === missing ? missing : Tuple(bw)[index]

function benchmark_row()
    result = Ref{Any}()
    elapsed_time = @elapsed begin
        result[] = solve_section522_k100_sparse(N=128, kerneldepth=12)
    end
    row = result[]

    return (
        N=row.N,
        kerneldepth=row.kerneldepth,
        elapsed_time=elapsed_time,
        rhs_validation_error=row.rhs_validation_error,
        tau_residual=row.tau_residual,
        body_residual_relative=row.body_residual_relative,
        max_abs_error=row.max_abs_error,
        raw_operator_type=row.raw_operator_type,
        raw_operator_nnz=row.raw_operator_nnz,
        raw_lower_bandwidth=bandwidth_part(row.raw_operator_bandwidth, 1),
        raw_upper_bandwidth=bandwidth_part(row.raw_operator_bandwidth, 2),
        body_operator_type=row.body_operator_type,
        body_operator_nnz=row.body_operator_nnz,
        body_lower_bandwidth=bandwidth_part(row.body_operator_bandwidth, 1),
        body_upper_bandwidth=bandwidth_part(row.body_operator_bandwidth, 2),
        final_system_type=row.final_system_type,
        final_system_nnz=row.final_system_nnz,
        final_lower_bandwidth=bandwidth_part(row.full_bandwidth, 1),
        final_upper_bandwidth=bandwidth_part(row.full_bandwidth, 2),
        density=row.density,
        status=row.status,
    )
end

function csv_value(value)
    value === missing && return ""
    return string(value)
end

function write_csv(path::AbstractString, row::NamedTuple)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(string.(CSV_COLUMNS), ","))
        println(io, join((csv_value(getproperty(row, col)) for col in CSV_COLUMNS), ","))
    end
    return path
end

function print_summary(row::NamedTuple, csv_path::AbstractString)
    println("Gutleb-Olver Section 5.2.2 k=100 sparse benchmark")
    println("project = ", GUTLEB_OLVER_PROJECT)
    println("N = ", row.N)
    println("kerneldepth = ", row.kerneldepth)
    println("elapsed_time = ", row.elapsed_time)
    println("rhs_validation_error = ", row.rhs_validation_error)
    println("tau_residual = ", row.tau_residual)
    println("body_residual_relative = ", row.body_residual_relative)
    println("max_abs_error = ", row.max_abs_error)
    println("raw operator = ", row.raw_operator_type,
            " nnz=", row.raw_operator_nnz,
            " bw=(", row.raw_lower_bandwidth, ",", row.raw_upper_bandwidth, ")")
    println("body operator = ", row.body_operator_type,
            " nnz=", row.body_operator_nnz,
            " bw=(", row.body_lower_bandwidth, ",", row.body_upper_bandwidth, ")")
    println("final system = ", row.final_system_type,
            " nnz=", row.final_system_nnz,
            " bw=(", row.final_lower_bandwidth, ",", row.final_upper_bandwidth, ")")
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
