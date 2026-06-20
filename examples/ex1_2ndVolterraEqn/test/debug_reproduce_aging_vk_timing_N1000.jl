using Printf
using SparseArrays

include(joinpath(@__DIR__, "ex1_2nd_volterra_common.jl"))

const DEBUG_REPRO_N = 1000
const DEBUG_REPRO_WARMUP_N = 256
const DEBUG_REPRO_TOL = eps()
const DEBUG_REPRO_OUTPUT_CSV = joinpath(@__DIR__, "debug_reproduce_aging_vk_timing_N1000.csv")

# Direct copy of the kernel formula used by ex1_aging_maxwell.jl.
aging_tau_xy(x, y) = 1.5 + x - 0.5 * y

function aging_K(x, y)
    tau = aging_tau_xy(x, y)
    return exp(-(x - y) / tau) / tau
end

const DEBUG_REPRO_COLUMNS = [
    :source,
    :N,
    :tol,
    :K_type,
    :which_voltmat_cheb_square,
    :t_raw_VK,
    :typeof_VK,
    :size_VK,
    :nnz_VK,
    :status,
    :notes,
]

function debug_repro_write_csv(path::AbstractString, rows)
    open(path, "w") do io
        println(io, join(string.(DEBUG_REPRO_COLUMNS), ","))
        for row in rows
            vals = [ex1_csv_escape(getproperty(row, col)) for col in DEBUG_REPRO_COLUMNS]
            println(io, join(vals, ","))
        end
    end
    return path
end

function debug_repro_raw_vk_row(source::AbstractString, Kfun, K_label::AbstractString)
    local VK
    t_raw = NaN
    status = "success"
    notes = String[]
    try
        GC.gc()
        t_raw = @elapsed begin
            VK = VoltMat_Cheb.voltmat_cheb_square(Kfun, DEBUG_REPRO_N - 1; tol=DEBUG_REPRO_TOL)
        end
    catch err
        status = "failed"
        push!(notes, sprint(showerror, err))
        return (
            source=source,
            N=DEBUG_REPRO_N,
            tol=DEBUG_REPRO_TOL,
            K_type=string(typeof(Kfun)),
            which_voltmat_cheb_square="VoltMat_Cheb.voltmat_cheb_square($(K_label), N - 1; tol=tol)",
            t_raw_VK=t_raw,
            typeof_VK=missing,
            size_VK=missing,
            nnz_VK=missing,
            status=status,
            notes=join(notes, "; "),
        )
    end
    return (
        source=source,
        N=DEBUG_REPRO_N,
        tol=DEBUG_REPRO_TOL,
        K_type=string(typeof(Kfun)),
        which_voltmat_cheb_square="VoltMat_Cheb.voltmat_cheb_square($(K_label), N - 1; tol=tol)",
        t_raw_VK=t_raw,
        typeof_VK=string(typeof(VK)),
        size_VK=string(size(VK)),
        nnz_VK=nnz(VK),
        status=status,
        notes="",
    )
end

function debug_repro_kernel_sample_maxdiff()
    pts = ((-1.0, -1.0), (-0.2, -0.6), (0.3, -0.1), (1.0, 0.4), (1.0, 1.0))
    return maximum(abs(aging_K(x, y) - ex1_kernel(x, y)) for (x, y) in pts)
end

function run_debug_reproduce_aging_vk_timing_N1000()
    println("Reproduce ex1_aging_maxwell raw VK timing at N=1000")
    println("output: ", DEBUG_REPRO_OUTPUT_CSV)
    println("N = ", DEBUG_REPRO_N, ", tol = ", DEBUG_REPRO_TOL)

    local VKwarm
    GC.gc()
    t_warmup = @elapsed begin
        VKwarm = VoltMat_Cheb.voltmat_cheb_square(aging_K, DEBUG_REPRO_WARMUP_N - 1; tol=DEBUG_REPRO_TOL)
    end
    warmup_note = @sprintf(
        "warmup follows ex1_aging_maxwell style with raw VK only: N=%d, t=%.6e, nnz=%d",
        DEBUG_REPRO_WARMUP_N,
        t_warmup,
        nnz(VKwarm),
    )

    rows = [
        debug_repro_raw_vk_row("original_aging_style", aging_K, "K"),
        debug_repro_raw_vk_row("current_wrapper_aging_maxwell_fft", ex1_kernel, "ex1_kernel"),
    ]

    kdiff = debug_repro_kernel_sample_maxdiff()
    t_original = rows[1].t_raw_VK
    t_wrapper = rows[2].t_raw_VK
    ratio = isfinite(t_original) && t_original > 0.0 ? t_wrapper / t_original : NaN
    alignment =
        isfinite(ratio) && 0.75 <= ratio <= 1.25 ?
        "aligned_within_25pct" :
        "not_aligned_within_25pct"
    common_notes = join((
        warmup_note,
        @sprintf("sample_max_abs_diff_between_aging_K_and_ex1_kernel=%.3e", kdiff),
        "tol matches eps(); both rows pass a function K, not C",
        "no assembly keyword is passed",
        "t_raw_VK excludes RHS, bandwidth detection, almost-banded conversion, factorization, solve, residual/error checks, Matrix(VK), and metadata extraction",
        @sprintf("t_ratio_current_wrapper_over_original=%.6e", ratio),
        alignment,
    ), "; ")

    rows = [merge(row, (notes=isempty(row.notes) ? common_notes : row.notes * "; " * common_notes,)) for row in rows]
    debug_repro_write_csv(DEBUG_REPRO_OUTPUT_CSV, rows)

    for row in rows
        @printf(
            "%-35s t_raw_VK=%.6e nnz=%s status=%s\n",
            row.source,
            row.t_raw_VK,
            string(row.nnz_VK),
            row.status,
        )
    end
    println("notes: ", common_notes)
    println("Wrote ", DEBUG_REPRO_OUTPUT_CSV)
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_debug_reproduce_aging_vk_timing_N1000()
end
