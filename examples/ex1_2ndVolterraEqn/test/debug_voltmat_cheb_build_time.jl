include(joinpath(@__DIR__, "ex1_2nd_volterra_common.jl"))

const DEBUG_CHEB_NS = ex1_parse_int_list_env("EX1_DEBUG_CHEB_NS", [1000, 2000, 4000, 8000, 16000, 30000])
const DEBUG_CHEB_C_SIZES = ex1_parse_int_list_env("EX1_DEBUG_CHEB_C_SIZES", [32, 64, 100])
const DEBUG_CHEB_ASSEMBLY = Symbol(get(ENV, "EX1_DEBUG_CHEB_ASSEMBLY", string(EX1_DEFAULT_CHEB_ASSEMBLY)))
const DEBUG_CHEB_OUTPUT_CSV = joinpath(@__DIR__, "debug_voltmat_cheb_build_time.csv")

const DEBUG_CHEB_COLUMNS = [
    :N,
    :kernel_coeff_rows,
    :kernel_coeff_cols,
    :kernel_coeff_time,
    :cached_kernel_coeff,
    :nnz,
    :matrix_assembly_time,
    :time_per_nnz,
    :rhs_time,
    :solve_time,
    :max_abs_error,
    :status,
    :notes,
]

function debug_cheb_write_csv(path::AbstractString, rows)
    open(path, "w") do io
        println(io, join(string.(DEBUG_CHEB_COLUMNS), ","))
        for row in rows
            vals = [ex1_csv_escape(getproperty(row, col)) for col in DEBUG_CHEB_COLUMNS]
            println(io, join(vals, ","))
        end
    end
    return path
end

function debug_cheb_failure_row(N::Int, rows::Int, cols::Int, coeff_time::Float64, cached::Bool, err)
    return (
        N=N,
        kernel_coeff_rows=rows,
        kernel_coeff_cols=cols,
        kernel_coeff_time=coeff_time,
        cached_kernel_coeff=cached,
        nnz=missing,
        matrix_assembly_time=NaN,
        time_per_nnz=NaN,
        rhs_time=NaN,
        solve_time=NaN,
        max_abs_error=NaN,
        status="failed",
        notes=sprint(showerror, err),
    )
end

function debug_cheb_row_from_solve(row)
    time_per_nnz =
        row.nnz isa Integer && row.nnz > 0 ?
        row.matrix_assembly_time / row.nnz :
        NaN
    return (
        N=row.N,
        kernel_coeff_rows=row.kernel_coeff_rows,
        kernel_coeff_cols=row.kernel_coeff_cols,
        kernel_coeff_time=row.kernel_coeff_time,
        cached_kernel_coeff=row.cached_kernel_coeff,
        nnz=row.nnz,
        matrix_assembly_time=row.matrix_assembly_time,
        time_per_nnz=time_per_nnz,
        rhs_time=row.rhs_time,
        solve_time=row.solve_time,
        max_abs_error=row.max_abs_error,
        status=row.status,
        notes="assembly=$(DEBUG_CHEB_ASSEMBLY); to_fast_almost_banded_time=$(row.to_fast_almost_banded_time); factor_time=$(row.factor_time)",
    )
end

function run_debug_voltmat_cheb_build_time()
    rows = NamedTuple[]
    println("VoltMat_Cheb build-time diagnostics for ex1 second-kind Volterra")
    println("N values: ", DEBUG_CHEB_NS)
    println("C sizes: ", DEBUG_CHEB_C_SIZES)
    println("assembly: ", DEBUG_CHEB_ASSEMBLY)
    println("output: ", DEBUG_CHEB_OUTPUT_CSV)

    for csize in DEBUG_CHEB_C_SIZES
        GC.gc()
        cache = ex1_build_cheb_kernel_coeff_cache(csize)
        @printf(
            "\nC = %dx%d, kernel_coeff_time = %.6e s\n",
            cache.kernel_coeff_rows,
            cache.kernel_coeff_cols,
            cache.kernel_coeff_time,
        )

        for (idx, N) in pairs(DEBUG_CHEB_NS)
            GC.gc()
            charge_coeff_time = idx == firstindex(DEBUG_CHEB_NS)
            row = try
                solve_row = ex1_solve_cheb(
                    N;
                    kernel_cache=cache,
                    charge_kernel_coeff_time=charge_coeff_time,
                    assembly=DEBUG_CHEB_ASSEMBLY,
                )
                debug_cheb_row_from_solve(solve_row)
            catch err
                @warn "VoltMat_Cheb debug solve failed for C=$csize, N=$N" exception=(err, catch_backtrace())
                debug_cheb_failure_row(
                    N,
                    cache.kernel_coeff_rows,
                    cache.kernel_coeff_cols,
                    charge_coeff_time ? cache.kernel_coeff_time : 0.0,
                    !charge_coeff_time,
                    err,
                )
            end
            push!(rows, row)
            debug_cheb_write_csv(DEBUG_CHEB_OUTPUT_CSV, rows)
            @printf(
                "  N=%d cached=%s nnz=%s matrix=%.6e s per_nnz=%.6e rhs=%.6e solve=%.6e err=%.6e status=%s\n",
                row.N,
                string(row.cached_kernel_coeff),
                string(row.nnz),
                row.matrix_assembly_time,
                row.time_per_nnz,
                row.rhs_time,
                row.solve_time,
                row.max_abs_error,
                row.status,
            )
        end
    end

    println("\nWrote ", DEBUG_CHEB_OUTPUT_CSV)
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_debug_voltmat_cheb_build_time()
end
