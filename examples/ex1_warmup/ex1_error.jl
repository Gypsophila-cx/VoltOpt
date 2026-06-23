#!/usr/bin/env julia

include(joinpath(@__DIR__, "ex1_common.jl"))

const ERROR_NS = collect(1:100)
const ERROR_CSV = joinpath(DATA_DIR, "ex1_error_N1_100.csv")

const ERROR_HEADER = (
    :method,
    :basis,
    :N,
    :kerneldepth,
    :C_rows,
    :C_cols,
    :target_kernel_bandwidth,
    :solver,
    :solver_backend,
    :fast_matrix_type,
    :fast_matrix_lband,
    :fast_matrix_uband,
    :fast_matrix_fill_rows,
    :max_abs_error,
    :residual_relative,
    :body_residual_relative,
    :elapsed_time, # Single-run sanity timing only; not a benchmark statistic.
    :raw_matrix_type,
    :raw_matrix_lband,
    :raw_matrix_uband,
    :raw_structure,
    :final_matrix_type,
    :final_matrix_lband,
    :final_matrix_uband,
    :final_structure,
    :status,
    :error_message,
)

function run_error_sweep()
    rows = NamedTuple[]
    @printf("Running low-order VIDE error sweep for N=%d:%d\n", first(ERROR_NS), last(ERROR_NS))
    @printf("Output CSV: %s\n", ERROR_CSV)
    @printf("Validating FastAlmostBanded solve consistency on N=10,20,50,100...\n")
    validate_fast_solver_consistency()

    for N in ERROR_NS
        for spec in METHOD_SPECS
            row = try
                run_method_case(spec, N)
            catch err
                failed_case_row(spec, N, err)
            end
            push!(rows, row)
        end

        recent = rows[(length(rows) - 2):length(rows)]
        status_summary = join(("$(row.method)/$(row.basis):$(row.status)" for row in recent), ", ")
        @printf("N=%3d  %s\n", N, status_summary)
    end

    written_path = write_csv_rows(rows, ERROR_CSV, ERROR_HEADER)
    successes = count(row -> row.status == "success", rows)
    @printf("Wrote %d rows (%d success, %d failed) to %s.\n",
            length(rows), successes, length(rows) - successes, written_path)
    return rows
end

run_error_sweep()
