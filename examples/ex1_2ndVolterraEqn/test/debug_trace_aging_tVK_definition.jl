using LinearAlgebra
using Printf
using SparseArrays

include(joinpath(@__DIR__, "ex1_2nd_volterra_common.jl"))

const TRACE_AGING_NS = ex1_parse_int_list_env("EX1_TRACE_AGING_NS", [810, 1000, 1214])
const TRACE_AGING_OUTPUT_CSV = joinpath(@__DIR__, "debug_trace_aging_tVK_definition.csv")
const TRACE_AGING_OUTPUT_MD = joinpath(@__DIR__, "debug_trace_aging_tVK_definition.md")
const TRACE_AGING_SOURCE = joinpath(@__DIR__, "ex1_aging_maxwell.jl")
const TRACE_AGING_TOL = eps()

const TRACE_AGING_COLUMNS = [
    :N,
    :timing_type,
    :t,
    :nnz_VK,
    :top_rows,
    :lower_bw,
    :upper_bw,
    :err_inf_if_available,
    :source_code_location,
    :status,
    :notes,
]

# Direct copy of ex1_aging_maxwell.jl's kernel definition.
aging_trace_tau_xy(x, y) = 1.5 + x - 0.5 * y

function aging_trace_K(x, y)
    tau = aging_trace_tau_xy(x, y)
    return exp(-(x - y) / tau) / tau
end

function trace_write_csv(path::AbstractString, rows)
    open(path, "w") do io
        println(io, join(string.(TRACE_AGING_COLUMNS), ","))
        for row in rows
            vals = [ex1_csv_escape(getproperty(row, col)) for col in TRACE_AGING_COLUMNS]
            println(io, join(vals, ","))
        end
    end
    return path
end

function trace_source_lines()
    return collect(eachline(TRACE_AGING_SOURCE))
end

function trace_matching_lines(lines, patterns)
    hits = Tuple{Int,String}[]
    for (idx, line) in enumerate(lines)
        for pat in patterns
            if occursin(pat, line)
                push!(hits, (idx, line))
                break
            end
        end
    end
    return hits
end

function trace_source_excerpt(lines, center::Int; radius::Int=3)
    lo = max(1, center - radius)
    hi = min(length(lines), center + radius)
    return [(i, lines[i]) for i in lo:hi]
end

function trace_structure(A::SparseMatrixCSC)
    structure = ex1_estimate_almostbanded_structure(A)
    return structure.top_rows, structure.lower_bw, structure.upper_bw
end

function trace_solve_error(Aop::SparseMatrixCSC, rhs_coeff::AbstractVector)
    local sol
    try
        if EX1_HAS_FAST_AB[]
            structure = ex1_estimate_almostbanded_structure(Aop)
            Aab = ex1_sparse_to_almostbanded(
                Aop;
                top_rows=structure.top_rows,
                lower_bw=structure.lower_bw,
                upper_bw=structure.upper_bw,
            )
            sol = qr(Aab) \ rhs_coeff
        else
            sol = lu(Aop) \ rhs_coeff
        end
        uvals = ex1_cheb_eval_many(sol, EX1_XTEST)
        return maximum(abs.(uvals .- EX1_UTEST_EXACT))
    catch err
        @warn "error solve failed in trace script" exception=(err, catch_backtrace())
        return NaN
    end
end

function trace_table_style_row(N::Int)
    local xnodes, fvals, rhs_coeff
    # This mirrors ex1_aging_maxwell.jl's solve_volterra_once ordering around
    # t_VK: coefficient/RHS work first, then GC.gc(), then only VK construction.
    xnodes = ex1_chebpts_lobatto(N)
    fvals = ex1_f_exact.(xnodes)
    rhs_coeff = ex1_chebcoeffs_from_values(fvals)

    local VK
    GC.gc()
    t_vk = @elapsed begin
        VK = VoltMat_Cheb.voltmat_cheb_square(aging_trace_K, N - 1; tol=TRACE_AGING_TOL)
    end

    Aop = spdiagm(0 => ones(N)) + VK
    top_rows, lower_bw, upper_bw = trace_structure(Aop)
    err_inf = trace_solve_error(Aop, rhs_coeff)
    return (
        N=N,
        timing_type="table_style_tVK",
        t=t_vk,
        nnz_VK=nnz(VK),
        top_rows=top_rows,
        lower_bw=lower_bw,
        upper_bw=upper_bw,
        err_inf_if_available=err_inf,
        source_code_location="ex1_aging_maxwell.jl:540-541",
        status="success",
        notes="Current workspace summary-table t_VK definition: GC.gc(); @elapsed VK = VoltMat_Cheb.voltmat_cheb_square(K, N - 1; tol = tol). This includes function-kernel adaptive coefficient construction inside voltmat_cheb_square.",
    )
end

function trace_raw_function_call_row(N::Int)
    local VK
    GC.gc()
    t_vk = @elapsed begin
        VK = VoltMat_Cheb.voltmat_cheb_square(aging_trace_K, N - 1; tol=TRACE_AGING_TOL)
    end
    Aop = spdiagm(0 => ones(N)) + VK
    top_rows, lower_bw, upper_bw = trace_structure(Aop)
    return (
        N=N,
        timing_type="raw_function_call_time",
        t=t_vk,
        nnz_VK=nnz(VK),
        top_rows=top_rows,
        lower_bw=lower_bw,
        upper_bw=upper_bw,
        err_inf_if_available=NaN,
        source_code_location="debug_trace_aging_tVK_definition.jl raw call",
        status="success",
        notes="Only @elapsed VK = VoltMat_Cheb.voltmat_cheb_square(K, N - 1; tol=tol); no RHS, bandwidth scan, conversion, factorization, solve, Matrix(VK), or metadata in t.",
    )
end

function trace_cached_or_coeff_row(N::Int, adaptive_metadata)
    return (
        N=N,
        timing_type="cached_or_coeff_assembly_time",
        t=NaN,
        nnz_VK=missing,
        top_rows=missing,
        lower_bw=missing,
        upper_bw=missing,
        err_inf_if_available=NaN,
        source_code_location="ex1_aging_maxwell.jl current workspace",
        status="not_available",
        notes="No precomputed C/plan/cache is present in the current ex1_aging_maxwell.jl t_VK path. Separately inspected adaptive C metadata is $(adaptive_metadata.kernel_coeff_rows)x$(adaptive_metadata.kernel_coeff_cols), built in $(@sprintf("%.6e", adaptive_metadata.kernel_coeff_time)) s, but that object is not used by the current summary-table t_VK assignment.",
    )
end

function trace_write_md(path::AbstractString, lines, hits, rows, adaptive_metadata)
    t_vk_hits = [h for h in hits if occursin("t_VK", h[2]) || occursin("voltmat_cheb_square", h[2]) || occursin("VK =", h[2])]
    open(path, "w") do io
        println(io, "# debug_trace_aging_tVK_definition")
        println(io)
        println(io, "This file was generated by `debug_trace_aging_tVK_definition.jl`.")
        println(io)
        println(io, "## Current Source Conclusion")
        println(io)
        println(io, "- In the current workspace, `ex1_aging_maxwell.jl` assigns `t_VK` at lines 540-541 as:")
        println(io)
        println(io, "```julia")
        for (ln, text) in trace_source_excerpt(lines, 540; radius=2)
            println(io, lpad(string(ln), 4), ": ", text)
        end
        println(io, "```")
        println(io)
        println(io, "- Therefore current-source `table_style_tVK` is `@elapsed VoltMat_Cheb.voltmat_cheb_square(K, N - 1; tol = tol)`.")
        println(io, "- No precomputed `C`, plan, kerneldata, cache, or metadata object is passed to `voltmat_cheb_square` in that assignment.")
        println(io, "- Separately inspecting `adaptive_cheb2_coeffmat` gives C size `$(adaptive_metadata.kernel_coeff_rows)x$(adaptive_metadata.kernel_coeff_cols)` in `$(@sprintf("%.6e", adaptive_metadata.kernel_coeff_time))` s, but current `t_VK` does not reuse that object.")
        println(io)
        println(io, "## Relevant Source Hits")
        println(io)
        println(io, "```text")
        for (ln, text) in hits
            println(io, lpad(string(ln), 4), ": ", text)
        end
        println(io, "```")
        println(io)
        println(io, "## Timing Results")
        println(io)
        println(io, "| N | timing_type | t | nnz_VK | top_rows | lower_bw | upper_bw | status |")
        println(io, "|---:|---|---:|---:|---:|---:|---:|---|")
        for row in rows
            println(io, "| $(row.N) | $(row.timing_type) | $(@sprintf("%.6e", row.t)) | $(row.nnz_VK) | $(row.top_rows) | $(row.lower_bw) | $(row.upper_bw) | $(row.status) |")
        end
        println(io)
        println(io, "## Interpretation")
        println(io)
        println(io, "- The current workspace does not reproduce an `N≈1000` table-style `t_VK` of a few 1e-2 seconds.")
        println(io, "- If an attachment reports `N=810 t_VK≈0.0279s` and `N=1214 t_VK≈0.0406s`, that timing boundary is not the current visible `ex1_aging_maxwell.jl` lines 540-541, or the attachment came from a different code revision/runtime path.")
        println(io, "- The `manual_C_fft` diagnostic remains a different path and should not be used to infer the cost of the function-kernel default path.")
    end
    return path
end

function run_debug_trace_aging_tVK_definition()
    println("Tracing ex1_aging_maxwell.jl t_VK definition near N=1000")
    println("Ns: ", TRACE_AGING_NS)
    println("CSV: ", TRACE_AGING_OUTPUT_CSV)
    println("MD: ", TRACE_AGING_OUTPUT_MD)

    lines = trace_source_lines()
    patterns = [
        "t_VK", "time_VK", "VK =", "voltmat_cheb_square", "@elapsed",
        "@timed", "time_ns", "kernel", "C", "coeff", "plan", "cache", "metadata",
    ]
    hits = trace_matching_lines(lines, patterns)
    adaptive_metadata = ex1_build_adaptive_cheb_kernel_coeff_metadata()

    rows = NamedTuple[]
    for N in TRACE_AGING_NS
        println()
        @printf("N = %d\n", N)
        table_row = trace_table_style_row(N)
        raw_row = trace_raw_function_call_row(N)
        cached_row = trace_cached_or_coeff_row(N, adaptive_metadata)
        append!(rows, (table_row, raw_row, cached_row))
        trace_write_csv(TRACE_AGING_OUTPUT_CSV, rows)
        @printf("  table_style_tVK          %.6e s\n", table_row.t)
        @printf("  raw_function_call_time   %.6e s\n", raw_row.t)
        @printf("  cached/coeff assembly    %s\n", cached_row.status)
    end

    trace_write_md(TRACE_AGING_OUTPUT_MD, lines, hits, rows, adaptive_metadata)
    println()
    println("Wrote ", TRACE_AGING_OUTPUT_CSV)
    println("Wrote ", TRACE_AGING_OUTPUT_MD)
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_debug_trace_aging_tVK_definition()
end
