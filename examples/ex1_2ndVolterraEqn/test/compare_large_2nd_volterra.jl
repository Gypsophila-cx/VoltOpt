include(joinpath(@__DIR__, "ex1_2nd_volterra_common.jl"))

const EX1_DEFAULT_NS = [500, 1000, 2000, 4000, 8000, 16000, 30000]
const EX1_N_VALUES = ex1_parse_int_list_env("EX1_COMPARE_NS", EX1_DEFAULT_NS)
const EX1_OUTPUT_CSV = joinpath(@__DIR__, "ex1_2nd_volterra_large_compare.csv")
const EX1_COMPARE_GO_RHS_METHOD = Symbol(get(ENV, "EX1_GO_RHS_METHOD", "auto"))
const EX1_COMPARE_CHEB_KERNEL_COEFF_SIZE = parse(Int, get(ENV, "EX1_CHEB_KERNEL_COEFF_SIZE", string(EX1_DEFAULT_CHEB_KERNEL_COEFF_SIZE)))
const EX1_COMPARE_CHEB_PATH = Symbol(get(ENV, "EX1_CHEB_PATH", string(EX1_DEFAULT_CHEB_PATH)))

function ex1_run_compare()
    rows = NamedTuple[]
    println("ex1 second-kind Volterra large comparison")
    println("N values: ", EX1_N_VALUES)
    println("output: ", EX1_OUTPUT_CSV)
    println("RHS: split Gauss-Legendre quadrature around both smoothed steps")
    println("Gutleb--Olver RHS projection method: ", EX1_COMPARE_GO_RHS_METHOD)
    println("VoltMat_Cheb path: ", EX1_COMPARE_CHEB_PATH)

    cheb_kernel_cache =
        EX1_COMPARE_CHEB_PATH === :aging_maxwell_fft ?
        ex1_build_adaptive_cheb_kernel_coeff_metadata() :
        ex1_build_cheb_kernel_coeff_cache(EX1_COMPARE_CHEB_KERNEL_COEFF_SIZE)
    if EX1_COMPARE_CHEB_PATH === :aging_maxwell_fft
        println(
            "VoltMat_Cheb adaptive C metadata: ",
            cheb_kernel_cache.kernel_coeff_rows, "x", cheb_kernel_cache.kernel_coeff_cols,
            " inspected in ", @sprintf("%.6f", cheb_kernel_cache.kernel_coeff_time), " s; VK construction follows ex1_aging_maxwell.jl and recomputes C internally.",
        )
    else
        println(
            "VoltMat_Cheb fixed kernel C: ",
            cheb_kernel_cache.kernel_coeff_rows, "x", cheb_kernel_cache.kernel_coeff_cols,
            " built in ", @sprintf("%.6f", cheb_kernel_cache.kernel_coeff_time), " s",
        )
    end

    for (idx, N) in pairs(EX1_N_VALUES)
        println()
        @printf("N = %d\n", N)

        GC.gc()
        cheb_row = try
            row = ex1_solve_cheb(
                N;
                kernel_cache=cheb_kernel_cache,
                charge_kernel_coeff_time=(idx == firstindex(EX1_N_VALUES) && EX1_COMPARE_CHEB_PATH !== :aging_maxwell_fft),
                cheb_path=EX1_COMPARE_CHEB_PATH,
            )
            @printf("  VoltMat_Cheb: path=%s assembly=%s C=%s x %s status=%s, err=%.3e, rel=%.3e, residual=%.3e, tVK=%.3f s, solve=%.3f s, total=%.3f s, top/l/u=%s/%s/%s\n",
                row.cheb_path, row.cheb_assembly,
                string(row.kernel_coeff_rows), string(row.kernel_coeff_cols),
                row.status, row.max_abs_error, row.relative_error, row.residual_rel,
                row.t_VK, row.solve_time, row.total_time,
                string(row.top_rows), string(row.lower_bw), string(row.upper_bw))
            row
        catch err
            @warn "VoltMat_Cheb failed for N=$N" exception=(err, catch_backtrace())
            ex1_failure_row("VoltMat_Cheb", N, err)
        end
        push!(rows, cheb_row)
        ex1_write_csv(EX1_OUTPUT_CSV, rows)

        GC.gc()
        go_row = try
            row = ex1_solve_gutleb_olver(N; rhs_projection_method=EX1_COMPARE_GO_RHS_METHOD)
            @printf("  GutlebOlver: status=%s, rhs_status=%s, rhs_valerr=%.3e, err=%.3e, rel=%.3e, residual=%.3e, total=%.3f s, bw=%s/%s\n",
                row.status, row.rhs_projection_status,
                row.rhs_projection_validation_error,
                row.max_abs_error, row.relative_error, row.residual_rel,
                row.total_time, string(row.bw_lower), string(row.bw_upper))
            row
        catch err
            @warn "GutlebOlver failed for N=$N" exception=(err, catch_backtrace())
            ex1_failure_row("GutlebOlver_P10", N, err)
        end
        push!(rows, go_row)
        ex1_write_csv(EX1_OUTPUT_CSV, rows)
    end

    println()
    println("Wrote ", EX1_OUTPUT_CSV)
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    ex1_run_compare()
end
