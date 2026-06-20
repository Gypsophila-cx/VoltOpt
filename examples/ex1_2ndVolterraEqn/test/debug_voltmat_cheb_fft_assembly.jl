include(joinpath(@__DIR__, "ex1_2nd_volterra_common.jl"))

const DEBUG_FFT_NS = ex1_parse_int_list_env("EX1_DEBUG_FFT_NS", [1000, 2000, 4000, 8000, 16000, 30000])
const DEBUG_FFT_ASSEMBLIES = Symbol.(strip.(split(get(ENV, "EX1_DEBUG_FFT_ASSEMBLIES", "compressed_rankupdate,fft,compressed"), ",")))
const DEBUG_FFT_C_SIZE = parse(Int, get(ENV, "EX1_DEBUG_FFT_C_SIZE", string(EX1_DEFAULT_CHEB_KERNEL_COEFF_SIZE)))
const DEBUG_FFT_MATRIX_DIFF_MAX_N = parse(Int, get(ENV, "EX1_DEBUG_FFT_MATRIX_DIFF_MAX_N", "4000"))
const DEBUG_FFT_OUTPUT_CSV = joinpath(@__DIR__, "debug_voltmat_cheb_fft_assembly.csv")

const DEBUG_FFT_COLUMNS = [
    :N,
    :cheb_assembly,
    :kernel_coeff_rows,
    :kernel_coeff_cols,
    :cached_kernel_coeff,
    :matrix_assembly_time,
    :to_fast_almost_banded_time,
    :factor_time,
    :solve_time,
    :total_time,
    :nnz,
    :max_abs_error,
    :max_abs_solution_diff_vs_compressed_rankupdate,
    :max_abs_matrix_diff_vs_compressed_rankupdate,
    :status,
    :notes,
]

function debug_fft_write_csv(path::AbstractString, rows)
    open(path, "w") do io
        println(io, join(string.(DEBUG_FFT_COLUMNS), ","))
        for row in rows
            vals = [ex1_csv_escape(getproperty(row, col)) for col in DEBUG_FFT_COLUMNS]
            println(io, join(vals, ","))
        end
    end
    return path
end

function debug_fft_sparse_maxabs(A::SparseMatrixCSC)
    isempty(nonzeros(A)) && return 0.0
    return maximum(abs, nonzeros(A))
end

function debug_fft_solve_cheb(N::Int, kernel_cache, assembly::Symbol; tol=eps())
    C = ex1_kernel_cache_matrix(kernel_cache)
    kernel_coeff_rows, kernel_coeff_cols = size(C)
    local xnodes, fvals, rhs_coeff, VK, Aop, sol, residual
    local structure, top_rows, lower_bw, upper_bw, solver_name

    t_rhs = @elapsed begin
        xnodes = ex1_chebpts_lobatto(N)
        fvals = ex1_f_exact.(xnodes)
        rhs_coeff = ex1_chebcoeffs_from_values(fvals)
    end
    t_matrix = @elapsed begin
        VK = VoltMat_Cheb.voltmat_cheb_square(C, N - 1; assembly=assembly, tol=tol)
    end
    t_system = @elapsed begin
        Aop = spdiagm(0 => ones(N)) + VK
    end

    t_convert = 0.0
    t_factor = 0.0
    t_solve = 0.0
    try
        if EX1_HAS_FAST_AB[]
            local Aab, F
            t_convert = @elapsed begin
                structure = ex1_estimate_almostbanded_structure(Aop)
                top_rows = structure.top_rows
                lower_bw = structure.lower_bw
                upper_bw = structure.upper_bw
                Aab = ex1_sparse_to_almostbanded(Aop; top_rows=top_rows, lower_bw=lower_bw, upper_bw=upper_bw)
            end
            t_factor = @elapsed begin
                F = qr(Aab)
            end
            t_solve = @elapsed begin
                sol = F \ rhs_coeff
            end
            solver_name = "fast_almost_banded_qr"
        else
            throw(ArgumentError("FastAlmostBandedMatrices unavailable"))
        end
    catch err
        local Fsp
        @warn "Cheb debug fast almost-banded solve failed; falling back to sparse LU." assembly exception=(err, catch_backtrace())
        t_factor = @elapsed begin
            Fsp = lu(Aop)
        end
        t_solve = @elapsed begin
            sol = Fsp \ rhs_coeff
        end
        solver_name = "sparse_lu_fallback"
    end

    local max_abs_error, rel_error, residual_inf, residual_rel
    t_error = @elapsed begin
        uvals = ex1_cheb_eval_many(sol, EX1_XTEST)
        max_abs_error = maximum(abs.(uvals .- EX1_UTEST_EXACT))
        rel_error = max_abs_error / EX1_UTEST_SCALE
        residual = Aop * sol - rhs_coeff
        residual_inf = norm(residual, Inf)
        residual_rel = residual_inf / max(norm(rhs_coeff, Inf), eps(Float64))
    end

    total_time = t_rhs + t_matrix + t_system + t_convert + t_factor + t_solve + t_error
    row = (
        N=N,
        cheb_assembly=string(assembly),
        kernel_coeff_rows=kernel_coeff_rows,
        kernel_coeff_cols=kernel_coeff_cols,
        cached_kernel_coeff=true,
        matrix_assembly_time=t_matrix,
        to_fast_almost_banded_time=t_convert,
        factor_time=t_factor,
        solve_time=t_solve,
        total_time=total_time,
        nnz=nnz(Aop),
        max_abs_error=max_abs_error,
        max_abs_solution_diff_vs_compressed_rankupdate=NaN,
        max_abs_matrix_diff_vs_compressed_rankupdate=NaN,
        status="success",
        notes="solver=$(solver_name); rhs_time=$(t_rhs); system_time=$(t_system); error_time=$(t_error); relative_error=$(rel_error); residual_rel=$(residual_rel)",
    )
    return (row=row, sol=sol, Aop=Aop)
end

function debug_fft_failure_row(N::Int, kernel_cache, assembly::Symbol, err)
    status =
        assembly === :fft && err isa MethodError ?
        "fft_not_available_for_C_interface" :
        "failed"
    return (
        N=N,
        cheb_assembly=string(assembly),
        kernel_coeff_rows=kernel_cache.kernel_coeff_rows,
        kernel_coeff_cols=kernel_cache.kernel_coeff_cols,
        cached_kernel_coeff=true,
        matrix_assembly_time=NaN,
        to_fast_almost_banded_time=NaN,
        factor_time=NaN,
        solve_time=NaN,
        total_time=NaN,
        nnz=missing,
        max_abs_error=NaN,
        max_abs_solution_diff_vs_compressed_rankupdate=NaN,
        max_abs_matrix_diff_vs_compressed_rankupdate=NaN,
        status=status,
        notes=sprint(showerror, err),
    )
end

function run_debug_voltmat_cheb_fft_assembly()
    rows = NamedTuple[]
    kernel_cache = ex1_build_cheb_kernel_coeff_cache(DEBUG_FFT_C_SIZE)

    println("VoltMat_Cheb FFT assembly diagnostics for ex1 second-kind Volterra")
    println("N values: ", DEBUG_FFT_NS)
    println("assemblies: ", DEBUG_FFT_ASSEMBLIES)
    println("output: ", DEBUG_FFT_OUTPUT_CSV)
    @printf(
        "kernel C: %dx%d built once in %.6e s\n",
        kernel_cache.kernel_coeff_rows,
        kernel_cache.kernel_coeff_cols,
        kernel_cache.kernel_coeff_time,
    )

    for N in DEBUG_FFT_NS
        println()
        @printf("N = %d\n", N)

        baseline = nothing
        baseline_failed = false

        for assembly in DEBUG_FFT_ASSEMBLIES
            GC.gc()
            result = try
                debug_fft_solve_cheb(N, kernel_cache, assembly)
            catch err
                @warn "VoltMat_Cheb assembly debug failed" N assembly exception=(err, catch_backtrace())
                baseline_failed |= assembly === :compressed_rankupdate
                row = debug_fft_failure_row(N, kernel_cache, assembly, err)
                push!(rows, row)
                debug_fft_write_csv(DEBUG_FFT_OUTPUT_CSV, rows)
                @printf("  %-23s status=%s notes=%s\n", string(assembly), row.status, row.notes)
                continue
            end

            row = result.row
            if assembly === :compressed_rankupdate
                baseline = result
                row = merge(row, (
                    max_abs_solution_diff_vs_compressed_rankupdate=0.0,
                    max_abs_matrix_diff_vs_compressed_rankupdate=0.0,
                ))
            elseif baseline !== nothing
                sol_diff = maximum(abs.(result.sol .- baseline.sol))
                mat_diff =
                    N <= DEBUG_FFT_MATRIX_DIFF_MAX_N ?
                    debug_fft_sparse_maxabs(result.Aop - baseline.Aop) :
                    NaN
                row = merge(row, (
                    max_abs_solution_diff_vs_compressed_rankupdate=sol_diff,
                    max_abs_matrix_diff_vs_compressed_rankupdate=mat_diff,
                ))
            elseif baseline_failed
                row = merge(row, (
                    notes=row.notes * "; compressed_rankupdate baseline failed",
                ))
            end

            push!(rows, row)
            debug_fft_write_csv(DEBUG_FFT_OUTPUT_CSV, rows)
            @printf(
                "  %-23s matrix=%.6e total=%.6e solve=%.6e err=%.6e sol_diff=%.6e mat_diff=%.6e status=%s\n",
                row.cheb_assembly,
                row.matrix_assembly_time,
                row.total_time,
                row.solve_time,
                row.max_abs_error,
                row.max_abs_solution_diff_vs_compressed_rankupdate,
                row.max_abs_matrix_diff_vs_compressed_rankupdate,
                row.status,
            )
        end
    end

    println("\nWrote ", DEBUG_FFT_OUTPUT_CSV)
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_debug_voltmat_cheb_fft_assembly()
end
