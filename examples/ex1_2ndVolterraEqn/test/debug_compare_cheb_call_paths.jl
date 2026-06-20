include(joinpath(@__DIR__, "ex1_2nd_volterra_common.jl"))

const DEBUG_CALL_PATH_NS = ex1_parse_int_list_env("EX1_DEBUG_CALL_PATH_NS", [8000, 16000, 30000, 40000])
const DEBUG_CALL_PATHS = Symbol.(strip.(split(get(ENV, "EX1_DEBUG_CALL_PATHS", "aging_maxwell_fft,manual_C_fft,compressed_rankupdate"), ",")))
const DEBUG_CALL_PATH_OUTPUT_CSV = joinpath(@__DIR__, "debug_compare_cheb_call_paths.csv")
const DEBUG_CALL_PATH_NREF = parse(Int, get(ENV, "EX1_DEBUG_CALL_PATH_NREF", string(max(80000, 2 * maximum(DEBUG_CALL_PATH_NS)))))

const DEBUG_CALL_PATH_COLUMNS = [
    :N,
    :cheb_path,
    :cheb_assembly,
    :kernel_coeff_rows,
    :kernel_coeff_cols,
    :nnz,
    :top_rows,
    :lower_bw,
    :upper_bw,
    :t_rhs,
    :t_VK,
    :t_factor,
    :t_solve,
    :total,
    :err_inf,
    :rel_inf,
    :coeff_err,
    :res_inf,
    :status,
    :notes,
]

function debug_call_path_write_csv(path::AbstractString, rows)
    open(path, "w") do io
        println(io, join(string.(DEBUG_CALL_PATH_COLUMNS), ","))
        for row in rows
            vals = [ex1_csv_escape(getproperty(row, col)) for col in DEBUG_CALL_PATH_COLUMNS]
            println(io, join(vals, ","))
        end
    end
    return path
end

function debug_call_path_cache(path::Symbol, fixed_cache, adaptive_metadata)
    return path === :aging_maxwell_fft ? adaptive_metadata : fixed_cache
end

function debug_call_path_failure_row(N::Int, path::Symbol, cache, err)
    return (
        N=N,
        cheb_path=string(path),
        cheb_assembly=path === :aging_maxwell_fft || path === :manual_C_fft ? "fft" :
                      path === :compressed_rankupdate ? "compressed_rankupdate" : "",
        kernel_coeff_rows=hasproperty(cache, :kernel_coeff_rows) ? cache.kernel_coeff_rows : missing,
        kernel_coeff_cols=hasproperty(cache, :kernel_coeff_cols) ? cache.kernel_coeff_cols : missing,
        nnz=missing,
        top_rows=missing,
        lower_bw=missing,
        upper_bw=missing,
        t_rhs=NaN,
        t_VK=NaN,
        t_factor=NaN,
        t_solve=NaN,
        total=NaN,
        err_inf=NaN,
        rel_inf=NaN,
        coeff_err=NaN,
        res_inf=NaN,
        status="failed",
        notes=sprint(showerror, err),
    )
end

function debug_call_path_row(result, coeff_ref)
    row = result.row
    N = row.N
    coeff_err = maximum(abs.(result.sol .- @view(coeff_ref[1:N])))
    return (
        N=N,
        cheb_path=row.cheb_path,
        cheb_assembly=row.cheb_assembly,
        kernel_coeff_rows=row.kernel_coeff_rows,
        kernel_coeff_cols=row.kernel_coeff_cols,
        nnz=row.nnz,
        top_rows=row.top_rows,
        lower_bw=row.lower_bw,
        upper_bw=row.upper_bw,
        t_rhs=row.rhs_time,
        t_VK=row.t_VK,
        t_factor=row.factor_time,
        t_solve=row.solve_time,
        total=row.total_time,
        err_inf=row.max_abs_error,
        rel_inf=row.relative_error,
        coeff_err=coeff_err,
        res_inf=row.residual_inf,
        status=row.status,
        notes=row.notes,
    )
end

function run_debug_compare_cheb_call_paths()
    println("Cheb call path comparison for ex1 second-kind Volterra")
    println("N values: ", DEBUG_CALL_PATH_NS)
    println("paths: ", DEBUG_CALL_PATHS)
    println("output: ", DEBUG_CALL_PATH_OUTPUT_CSV)

    println("Building coefficient reference, Nref = ", DEBUG_CALL_PATH_NREF)
    local coeff_ref
    t_ref = @elapsed begin
        xref = ex1_chebpts_lobatto(DEBUG_CALL_PATH_NREF)
        coeff_ref = ex1_chebcoeffs_from_values(ex1_u_exact.(xref))
    end
    @printf("coefficient reference time = %.6e s\n", t_ref)

    fixed_cache = ex1_build_cheb_kernel_coeff_cache(EX1_DEFAULT_CHEB_KERNEL_COEFF_SIZE)
    adaptive_metadata = ex1_build_adaptive_cheb_kernel_coeff_metadata()
    @printf(
        "fixed C metadata: %dx%d; adaptive C metadata for aging path: %dx%d\n",
        fixed_cache.kernel_coeff_rows,
        fixed_cache.kernel_coeff_cols,
        adaptive_metadata.kernel_coeff_rows,
        adaptive_metadata.kernel_coeff_cols,
    )

    rows = NamedTuple[]
    for N in DEBUG_CALL_PATH_NS
        println()
        @printf("N = %d\n", N)
        for path in DEBUG_CALL_PATHS
            GC.gc()
            cache = debug_call_path_cache(path, fixed_cache, adaptive_metadata)
            row = try
                result = ex1_solve_cheb(
                    N;
                    kernel_cache=cache,
                    charge_kernel_coeff_time=false,
                    cheb_path=path,
                    return_work=true,
                )
                debug_call_path_row(result, coeff_ref)
            catch err
                @warn "Cheb call path failed" N path exception=(err, catch_backtrace())
                debug_call_path_failure_row(N, path, cache, err)
            end
            push!(rows, row)
            debug_call_path_write_csv(DEBUG_CALL_PATH_OUTPUT_CSV, rows)
            @printf(
                "  %-22s assembly=%-22s nnz=%s top/l/u=%s/%s/%s t_rhs=%.4f tVK=%.4f tfac=%.4f tsolve=%.4f total=%.4f err=%.3e coeff=%.3e res=%.3e status=%s\n",
                row.cheb_path,
                row.cheb_assembly,
                string(row.nnz),
                string(row.top_rows),
                string(row.lower_bw),
                string(row.upper_bw),
                row.t_rhs,
                row.t_VK,
                row.t_factor,
                row.t_solve,
                row.total,
                row.err_inf,
                row.coeff_err,
                row.res_inf,
                row.status,
            )
        end
    end

    println("\nWrote ", DEBUG_CALL_PATH_OUTPUT_CSV)
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_debug_compare_cheb_call_paths()
end
