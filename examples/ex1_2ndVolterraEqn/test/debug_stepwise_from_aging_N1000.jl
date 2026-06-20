using FFTW
using InteractiveUtils
using LinearAlgebra
using Printf
using Profile
using SparseArrays

const DBG_DIR = @__DIR__
const DBG_N = 1000
const DBG_TOL = eps(Float64)
const DBG_CSV = joinpath(DBG_DIR, "debug_stepwise_from_aging_N1000.csv")
const DBG_MD = joinpath(DBG_DIR, "debug_stepwise_from_aging_N1000.md")
const DBG_PROFILE_BEFORE = joinpath(DBG_DIR, "debug_stepwise_from_aging_N1000_profile_before.txt")
const DBG_PROFILE_AFTER = joinpath(DBG_DIR, "debug_stepwise_from_aging_N1000_profile_after.txt")

function dbg_find_project_root(start_dir::AbstractString=DBG_DIR)
    dir = abspath(start_dir)
    while true
        if isfile(joinpath(dir, "Project.toml")) && isdir(joinpath(dir, "src"))
            return dir
        end
        parent = dirname(dir)
        parent == dir && error("Could not find project root from $(start_dir).")
        dir = parent
    end
end

const DBG_PROJECT_ROOT = dbg_find_project_root()
include(joinpath(DBG_PROJECT_ROOT, "src", "VoltMat", "voltmat_cheb.jl"))
using .VoltMat_Cheb

# Minimal copy of the kernel used by ex1_aging_maxwell.jl.
dbg_aging_tau_xy(x, y) = 1.5 + x - 0.5 * y

function dbg_aging_K(x, y)
    tau = dbg_aging_tau_xy(x, y)
    return exp(-(x - y) / tau) / tau
end

const DBG_SAMPLE_PAIRS = [
    (-0.8, -1.0),
    (-0.2, -0.7),
    (0.0, -0.1),
    (0.5, 0.2),
    (0.9, 0.8),
]

const DBG_COLUMNS = [
    :step_id,
    :step_name,
    :N,
    :tol,
    :uses_common_include,
    :K_source,
    :K_type,
    :sample_max_abs_diff_vs_aging_K,
    :assembly,
    :uses_precomputed_C,
    :C_rows,
    :C_cols,
    :t_C_build,
    :t_VK,
    :t_bandwidth_detect,
    :t_to_fast_almost_banded,
    :t_factor,
    :t_solve,
    :t_error,
    :t_rhs,
    :t_system,
    :t_metadata,
    :nnz,
    :size_rows,
    :size_cols,
    :typeof_VK,
    :top_rows,
    :lower_bw,
    :upper_bw,
    :which_method,
    :status,
    :notes,
]

function dbg_csv_escape(x)
    if x === missing
        return ""
    end
    s = string(x)
    if occursin(",", s) || occursin("\"", s) || occursin("\n", s) || occursin("\r", s)
        return "\"" * replace(s, "\"" => "\"\"") * "\""
    end
    return s
end

function dbg_write_csv(path::AbstractString, rows)
    open(path, "w") do io
        println(io, join(string.(DBG_COLUMNS), ","))
        for row in rows
            println(io, join((dbg_csv_escape(getproperty(row, col)) for col in DBG_COLUMNS), ","))
        end
    end
end

function dbg_row(; kwargs...)
    data = Dict{Symbol,Any}(
        col => missing for col in DBG_COLUMNS
    )
    data[:N] = DBG_N
    data[:tol] = DBG_TOL
    data[:sample_max_abs_diff_vs_aging_K] = NaN
    data[:t_C_build] = NaN
    data[:t_VK] = NaN
    data[:t_bandwidth_detect] = NaN
    data[:t_to_fast_almost_banded] = NaN
    data[:t_factor] = NaN
    data[:t_solve] = NaN
    data[:t_error] = NaN
    data[:t_rhs] = NaN
    data[:t_system] = NaN
    data[:t_metadata] = NaN
    data[:nnz] = missing
    data[:size_rows] = missing
    data[:size_cols] = missing
    data[:status] = "not_run"
    data[:notes] = ""
    for (key, value) in kwargs
        data[key] = value
    end
    return NamedTuple{Tuple(DBG_COLUMNS)}(Tuple(data[col] for col in DBG_COLUMNS))
end

function dbg_method_location(arg)
    method = which(VoltMat_Cheb.voltmat_cheb_square, Tuple{typeof(arg), Int})
    return "$(method.file):$(method.line) $(method.sig)"
end

function dbg_sample_diff(Kfun)
    diffs = Float64[]
    for (x, y) in DBG_SAMPLE_PAIRS
        push!(diffs, abs(Float64(Base.invokelatest(Kfun, x, y)) - Float64(dbg_aging_K(x, y))))
    end
    return maximum(diffs)
end

function dbg_call_vk(arg, assembly::Union{Nothing,Symbol})
    if assembly === nothing
        return Base.invokelatest(VoltMat_Cheb.voltmat_cheb_square, arg, DBG_N - 1; tol=DBG_TOL)
    end
    return Base.invokelatest(VoltMat_Cheb.voltmat_cheb_square, arg, DBG_N - 1; tol=DBG_TOL, assembly=assembly)
end

const DBG_PROFILE_BUILDERS = Dict{Int,Function}()

function dbg_run_vk_step!(
    rows::Vector{NamedTuple},
    step_id::Int,
    step_name::AbstractString,
    arg;
    uses_common_include::Bool,
    K_source::AbstractString,
    assembly::Union{Nothing,Symbol}=nothing,
    uses_precomputed_C::Bool=false,
    C_rows=missing,
    C_cols=missing,
    t_C_build=NaN,
    sample_diff=NaN,
    notes::AbstractString="",
)
    assembly_text = assembly === nothing ? "default" : string(assembly)
    k_type = sprint(show, typeof(arg))
    which_method = ""
    try
        which_method = dbg_method_location(arg)
    catch err
        which_method = "which_failed: $(sprint(showerror, err))"
    end

    builder = () -> dbg_call_vk(arg, assembly)
    DBG_PROFILE_BUILDERS[step_id] = builder

    local VK
    try
        GC.gc()
        t_vk = @elapsed begin
            VK = builder()
        end
        push!(rows, dbg_row(
            step_id=step_id,
            step_name=step_name,
            uses_common_include=uses_common_include,
            K_source=K_source,
            K_type=k_type,
            sample_max_abs_diff_vs_aging_K=sample_diff,
            assembly=assembly_text,
            uses_precomputed_C=uses_precomputed_C,
            C_rows=C_rows,
            C_cols=C_cols,
            t_C_build=t_C_build,
            t_VK=t_vk,
            nnz=nnz(VK),
            size_rows=size(VK, 1),
            size_cols=size(VK, 2),
            typeof_VK=sprint(show, typeof(VK)),
            which_method=which_method,
            status="success",
            notes=notes,
        ))
        return VK
    catch err
        push!(rows, dbg_row(
            step_id=step_id,
            step_name=step_name,
            uses_common_include=uses_common_include,
            K_source=K_source,
            K_type=k_type,
            sample_max_abs_diff_vs_aging_K=sample_diff,
            assembly=assembly_text,
            uses_precomputed_C=uses_precomputed_C,
            C_rows=C_rows,
            C_cols=C_cols,
            t_C_build=t_C_build,
            which_method=which_method,
            status="failed",
            notes="$(notes) error=$(sprint(showerror, err))",
        ))
        return nothing
    end
end

function dbg_profile_to_file(path::AbstractString, builder::Function, label::AbstractString)
    open(path, "w") do io
        println(io, label)
        println(io, "N=$(DBG_N), tol=$(DBG_TOL)")
        try
            GC.gc()
            Profile.clear()
            @profile builder()
            Profile.print(io; format=:flat, maxdepth=20)
        catch err
            println(io, "profile_failed: ", sprint(showerror, err))
        end
    end
end

function dbg_maybe_profile(rows)
    vk_rows = [row for row in rows if !ismissing(row.t_VK) && row.status == "success" && isfinite(row.t_VK)]
    if isempty(vk_rows)
        return "No successful VK row to profile."
    end

    first_row = vk_rows[1]
    if first_row.step_id == 0 && first_row.t_VK > 0.5
        open(DBG_PROFILE_BEFORE, "w") do io
            println(io, "No before profile: Step 0 aging_minimal_copy is already slow in the current workspace.")
            println(io, "Step 0 t_VK = $(first_row.t_VK)")
        end
        dbg_profile_to_file(
            DBG_PROFILE_AFTER,
            DBG_PROFILE_BUILDERS[Int(first_row.step_id)],
            "Profile for Step 0 aging_minimal_copy; there is no earlier fast step in this run.",
        )
        return "Step 0 already slow; profiled Step 0 only."
    end

    for i in 2:length(vk_rows)
        prev = vk_rows[i - 1]
        curr = vk_rows[i]
        if curr.t_VK > max(2.0 * prev.t_VK, prev.t_VK + 0.5)
            dbg_profile_to_file(
                DBG_PROFILE_BEFORE,
                DBG_PROFILE_BUILDERS[Int(prev.step_id)],
                "Profile before slowdown: step $(prev.step_id) $(prev.step_name)",
            )
            dbg_profile_to_file(
                DBG_PROFILE_AFTER,
                DBG_PROFILE_BUILDERS[Int(curr.step_id)],
                "Profile after slowdown: step $(curr.step_id) $(curr.step_name)",
            )
            return "Profiled first adjacent slowdown from step $(prev.step_id) to step $(curr.step_id)."
        end
    end

    open(DBG_PROFILE_BEFORE, "w") do io
        println(io, "No adjacent slowdown detected by the script threshold.")
    end
    open(DBG_PROFILE_AFTER, "w") do io
        println(io, "No adjacent slowdown detected by the script threshold.")
    end
    return "No adjacent slowdown detected."
end

function dbg_run_postprocess!(rows::Vector{NamedTuple}, step_id::Int, step_name::AbstractString, VK)
    if VK === nothing
        push!(rows, dbg_row(
            step_id=step_id,
            step_name=step_name,
            uses_common_include=true,
            K_source="postprocess_only",
            assembly="n/a",
            status="skipped",
            notes="VK input was not available.",
        ))
        return nothing
    end

    local Aop, structure, top_rows, lower_bw, upper_bw, Aab, factor, sol
    local rhs_coeff, xnodes, fvals, max_abs_error, residual
    local t_system, t_bandwidth, t_to_fast, t_factor, t_solve, t_error, t_rhs
    local status = "success"
    local notes = "Postprocessing is timed separately from VK construction."

    try
        t_system = @elapsed begin
            Aop = spdiagm(0 => ones(DBG_N)) + VK
        end

        t_bandwidth = @elapsed begin
            structure = Base.invokelatest(ex1_estimate_almostbanded_structure, Aop)
            top_rows = structure.top_rows
            lower_bw = structure.lower_bw
            upper_bw = structure.upper_bw
        end

        t_to_fast = NaN
        if EX1_HAS_FAST_AB[]
            t_to_fast = @elapsed begin
                Aab = Base.invokelatest(
                    ex1_sparse_to_almostbanded,
                    Aop;
                    top_rows=top_rows,
                    lower_bw=lower_bw,
                    upper_bw=upper_bw,
                )
            end
        else
            status = "fast_ab_unavailable_sparse_lu_fallback"
            notes *= " FastAlmostBandedMatrices unavailable."
        end

        t_rhs = @elapsed begin
            xnodes = Base.invokelatest(ex1_chebpts_lobatto, DBG_N)
            fvals = [Base.invokelatest(ex1_f_exact, x) for x in xnodes]
            rhs_coeff = Base.invokelatest(ex1_chebcoeffs_from_values, fvals)
        end

        if EX1_HAS_FAST_AB[]
            t_factor = @elapsed begin
                factor = Base.invokelatest(A -> qr(A), Aab)
            end
            t_solve = @elapsed begin
                sol = Base.invokelatest((F, b) -> F \ b, factor, rhs_coeff)
            end
        else
            t_factor = @elapsed begin
                factor = Base.invokelatest(lu, Aop)
            end
            t_solve = @elapsed begin
                sol = Base.invokelatest((F, b) -> F \ b, factor, rhs_coeff)
            end
        end

        t_error = @elapsed begin
            uvals = Base.invokelatest(ex1_cheb_eval_many, sol, EX1_XTEST)
            max_abs_error = maximum(abs.(uvals .- EX1_UTEST_EXACT))
            residual = Aop * sol - rhs_coeff
        end

        push!(rows, dbg_row(
            step_id=step_id,
            step_name=step_name,
            uses_common_include=true,
            K_source="postprocess_from_current_compressed_rankupdate_VK",
            K_type="n/a",
            assembly="n/a",
            uses_precomputed_C=true,
            t_bandwidth_detect=t_bandwidth,
            t_to_fast_almost_banded=t_to_fast,
            t_factor=t_factor,
            t_solve=t_solve,
            t_error=t_error,
            t_rhs=t_rhs,
            t_system=t_system,
            t_metadata=t_system,
            nnz=nnz(Aop),
            size_rows=size(Aop, 1),
            size_cols=size(Aop, 2),
            typeof_VK=sprint(show, typeof(VK)),
            top_rows=top_rows,
            lower_bw=lower_bw,
            upper_bw=upper_bw,
            status=status,
            notes="$(notes) max_abs_error=$(max_abs_error); residual_inf=$(norm(residual, Inf))",
        ))
    catch err
        push!(rows, dbg_row(
            step_id=step_id,
            step_name=step_name,
            uses_common_include=true,
            K_source="postprocess_only",
            assembly="n/a",
            status="failed",
            notes="postprocess_failed: $(sprint(showerror, err))",
        ))
    end

    return nothing
end

function dbg_write_md(path::AbstractString, rows, profile_note::AbstractString)
    vk_rows = [row for row in rows if !ismissing(row.t_VK) && row.status == "success" && isfinite(row.t_VK)]
    first_slow_note = "No adjacent slowdown detected."
    if !isempty(vk_rows)
        if vk_rows[1].step_id == 0 && vk_rows[1].t_VK > 0.5
            first_slow_note = "Step 0 is already slow, so the slowdown is not introduced by common/include/wrapper steps."
        else
            for i in 2:length(vk_rows)
                prev = vk_rows[i - 1]
                curr = vk_rows[i]
                if curr.t_VK > max(2.0 * prev.t_VK, prev.t_VK + 0.5)
                    first_slow_note = "First adjacent slowdown: step $(prev.step_id) -> step $(curr.step_id)."
                    break
                end
            end
        end
    end

    open(path, "w") do io
        println(io, "# debug_stepwise_from_aging_N1000")
        println(io)
        println(io, "- N: $(DBG_N)")
        println(io, "- tol: $(DBG_TOL)")
        println(io, "- CSV: $(DBG_CSV)")
        println(io, "- Profile note: $(profile_note)")
        println(io, "- Slowdown note: $(first_slow_note)")
        println(io)
        println(io, "## VK timings")
        println(io)
        println(io, "| step | name | assembly | K source | C | t_VK | status |")
        println(io, "|---:|---|---|---|---|---:|---|")
        for row in rows
            if !ismissing(row.t_VK)
                ctext = row.uses_precomputed_C === true ? "$(row.C_rows)x$(row.C_cols)" : ""
                println(io, "| $(row.step_id) | $(row.step_name) | $(row.assembly) | $(row.K_source) | $(ctext) | $(@sprintf("%.6f", row.t_VK)) | $(row.status) |")
            end
        end
        println(io)
        println(io, "## Notes")
        for row in rows
            println(io, "- step $(row.step_id) $(row.step_name): $(row.notes)")
        end
    end
end

function main()
    rows = NamedTuple[]

    println("debug_stepwise_from_aging_N1000")
    println("N = ", DBG_N)

    # Step 0: no common include, direct aging K copy.
    vk0 = dbg_run_vk_step!(
        rows,
        0,
        "aging_minimal_copy",
        dbg_aging_K;
        uses_common_include=false,
        K_source="direct_copy_from_ex1_aging_maxwell",
        sample_diff=0.0,
        notes="No common include; no RHS/solve/error/bandwidth conversion.",
    )

    # Step 1: include common, but keep using the direct aging K copy.
    include(joinpath(DBG_DIR, "ex1_2nd_volterra_common.jl"))

    vk1 = dbg_run_vk_step!(
        rows,
        1,
        "common_include_aging_K",
        dbg_aging_K;
        uses_common_include=true,
        K_source="direct_copy_from_ex1_aging_maxwell_after_common_include",
        sample_diff=0.0,
        notes="Only common include was added; VK still uses the direct aging K copy.",
    )

    # Step 2: replace K with common's ex1_kernel.
    vk2 = dbg_run_vk_step!(
        rows,
        2,
        "common_ex1_kernel_default",
        ex1_kernel;
        uses_common_include=true,
        K_source="ex1_2nd_volterra_common.ex1_kernel",
        sample_diff=dbg_sample_diff(ex1_kernel),
        notes="K is replaced by ex1_kernel from common.",
    )

    # Step 3: test the absorbed-kernel wrapper as a function-K input. This is a
    # diagnostic only; common uses this absorbed kernel primarily to build C.
    vk3 = dbg_run_vk_step!(
        rows,
        3,
        "absorbed_kernel_as_function_K_default",
        ex1_absorbed_kernel;
        uses_common_include=true,
        K_source="ex1_absorbed_kernel_passed_to_function_K_interface",
        sample_diff=dbg_sample_diff(ex1_absorbed_kernel),
        notes="Diagnostic wrapper test. Passing the absorbed kernel to the function-K interface applies VoltMat_Cheb's internal absorption again.",
    )

    # Step 4: explicit assembly keyword on the same common ex1_kernel.
    vk4a = dbg_run_vk_step!(
        rows,
        40,
        "common_ex1_kernel_explicit_fft",
        ex1_kernel;
        uses_common_include=true,
        K_source="ex1_2nd_volterra_common.ex1_kernel",
        sample_diff=dbg_sample_diff(ex1_kernel),
        assembly=:fft,
        notes="Same as common ex1_kernel, but assembly=:fft is explicit.",
    )

    vk4b = dbg_run_vk_step!(
        rows,
        41,
        "common_ex1_kernel_explicit_compressed_rankupdate",
        ex1_kernel;
        uses_common_include=true,
        K_source="ex1_2nd_volterra_common.ex1_kernel",
        sample_diff=dbg_sample_diff(ex1_kernel),
        assembly=:compressed_rankupdate,
        notes="Function-K interface with internal C construction and compressed rank-update assembly.",
    )

    vk4c = dbg_run_vk_step!(
        rows,
        42,
        "common_ex1_kernel_explicit_compressed",
        ex1_kernel;
        uses_common_include=true,
        K_source="ex1_2nd_volterra_common.ex1_kernel",
        sample_diff=dbg_sample_diff(ex1_kernel),
        assembly=:compressed,
        notes="Function-K interface with internal C construction and compressed no-FFT Clenshaw tail.",
    )

    # Step 5: current compare/common VK-only builders.
    adaptive_metadata = Base.invokelatest(ex1_build_adaptive_cheb_kernel_coeff_metadata)
    vk5a = dbg_run_vk_step!(
        rows,
        50,
        "current_cheb_builder_VK_only_aging_maxwell_fft",
        ex1_kernel;
        uses_common_include=true,
        K_source="ex1_solve_cheb aging_maxwell_fft VK branch",
        sample_diff=dbg_sample_diff(ex1_kernel),
        C_rows=adaptive_metadata.kernel_coeff_rows,
        C_cols=adaptive_metadata.kernel_coeff_cols,
        t_C_build=adaptive_metadata.kernel_coeff_time,
        notes="Matches ex1_solve_cheb(:aging_maxwell_fft) VK-only branch. Adaptive C metadata is built outside but not passed to VK.",
    )

    fixed_cache = Base.invokelatest(ex1_build_cheb_kernel_coeff_cache, EX1_DEFAULT_CHEB_KERNEL_COEFF_SIZE)
    C_fixed = Base.invokelatest(ex1_kernel_cache_matrix, fixed_cache)
    c_rows, c_cols = size(C_fixed)

    vk5b = dbg_run_vk_step!(
        rows,
        51,
        "current_cheb_builder_VK_only_manual_C_fft",
        C_fixed;
        uses_common_include=true,
        K_source="ex1_solve_cheb manual_C_fft VK branch",
        assembly=:fft,
        uses_precomputed_C=true,
        C_rows=c_rows,
        C_cols=c_cols,
        t_C_build=fixed_cache.kernel_coeff_time,
        notes="Matches ex1_solve_cheb(:manual_C_fft) VK-only branch with fixed 32x32 absorbed-kernel C.",
    )

    vk5c = dbg_run_vk_step!(
        rows,
        52,
        "current_cheb_builder_VK_only_compressed_rankupdate",
        C_fixed;
        uses_common_include=true,
        K_source="ex1_solve_cheb compressed_rankupdate VK branch",
        assembly=:compressed_rankupdate,
        uses_precomputed_C=true,
        C_rows=c_rows,
        C_cols=c_cols,
        t_C_build=0.0,
        notes="Matches ex1_solve_cheb(:compressed_rankupdate) VK-only branch and reuses the fixed 32x32 C.",
    )

    # Step 6: postprocessing from the current compressed-rankupdate VK.
    dbg_run_postprocess!(
        rows,
        60,
        "postprocess_from_current_compressed_rankupdate",
        vk5c,
    )

    profile_note = dbg_maybe_profile(rows)
    dbg_write_csv(DBG_CSV, rows)
    dbg_write_md(DBG_MD, rows, profile_note)

    println()
    println("Wrote ", DBG_CSV)
    println("Wrote ", DBG_MD)
    println(profile_note)
    println()
    println("VK timing summary:")
    for row in rows
        if !ismissing(row.t_VK)
            @printf("  step %-3s %-52s t_VK = %.6f s status=%s\n",
                string(row.step_id), row.step_name, row.t_VK, row.status)
        end
    end

    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
