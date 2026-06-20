using InteractiveUtils
using LinearAlgebra
using Printf
using Profile
using SparseArrays

include(joinpath(@__DIR__, "ex1_2nd_volterra_common.jl"))

const FIND_FAST_NS = ex1_parse_int_list_env("EX1_FIND_FAST_NS", [810, 1000, 1214])
const FIND_FAST_TOL = eps()
const FIND_FAST_CSV = joinpath(@__DIR__, "debug_find_fast_cheb_path_N1000.csv")
const FIND_FAST_MD = joinpath(@__DIR__, "debug_find_fast_cheb_path_N1000.md")
const FIND_FAST_PROFILE = joinpath(@__DIR__, "debug_find_fast_cheb_path_N1000_profile.txt")

const FIND_FAST_COLUMNS = [
    :N,
    :path,
    :assembly,
    :uses_function_K,
    :uses_precomputed_C,
    :C_rows,
    :C_cols,
    :t_C_build,
    :t_VK,
    :nnz_VK,
    :top_rows,
    :lower_bw,
    :upper_bw,
    :err_inf_if_available,
    :status,
    :notes,
]

# Direct copy of ex1_aging_maxwell.jl's kernel.
find_fast_tau_xy(x, y) = 1.5 + x - 0.5 * y

function find_fast_K(x, y)
    tau = find_fast_tau_xy(x, y)
    return exp(-(x - y) / tau) / tau
end

function find_fast_git(cmd)
    try
        return readchomp(cmd)
    catch err
        return "FAILED: $(sprint(showerror, err))"
    end
end

function find_fast_write_csv(path::AbstractString, rows)
    open(path, "w") do io
        println(io, join(string.(FIND_FAST_COLUMNS), ","))
        for row in rows
            vals = [ex1_csv_escape(getproperty(row, col)) for col in FIND_FAST_COLUMNS]
            println(io, join(vals, ","))
        end
    end
    return path
end

function find_fast_structure(A::SparseMatrixCSC)
    s = ex1_estimate_almostbanded_structure(A)
    return s.top_rows, s.lower_bw, s.upper_bw
end

function find_fast_err(Aop::SparseMatrixCSC, N::Int)
    try
        xnodes = ex1_chebpts_lobatto(N)
        rhs_coeff = ex1_chebcoeffs_from_values(ex1_f_exact.(xnodes))
        sol = if EX1_HAS_FAST_AB[]
            s = ex1_estimate_almostbanded_structure(Aop)
            Aab = ex1_sparse_to_almostbanded(Aop; top_rows=s.top_rows, lower_bw=s.lower_bw, upper_bw=s.upper_bw)
            qr(Aab) \ rhs_coeff
        else
            lu(Aop) \ rhs_coeff
        end
        uvals = ex1_cheb_eval_many(sol, EX1_XTEST)
        return maximum(abs.(uvals .- EX1_UTEST_EXACT))
    catch err
        @warn "find_fast error solve failed" exception=(err, catch_backtrace())
        return NaN
    end
end

function find_fast_row(;
    N::Int,
    path::String,
    assembly,
    uses_function_K::Bool,
    uses_precomputed_C::Bool,
    C,
    t_C_build::Float64,
    builder,
    notes::String,
)
    local VK
    t_vk = NaN
    status = "success"
    err_note = ""
    try
        GC.gc()
        t_vk = @elapsed begin
            VK = builder()
        end
    catch err
        status = "failed"
        err_note = sprint(showerror, err)
        return (
            N=N,
            path=path,
            assembly=string(assembly),
            uses_function_K=uses_function_K,
            uses_precomputed_C=uses_precomputed_C,
            C_rows=C === nothing ? missing : size(C, 1),
            C_cols=C === nothing ? missing : size(C, 2),
            t_C_build=t_C_build,
            t_VK=t_vk,
            nnz_VK=missing,
            top_rows=missing,
            lower_bw=missing,
            upper_bw=missing,
            err_inf_if_available=NaN,
            status=status,
            notes=isempty(notes) ? err_note : notes * "; " * err_note,
        )
    end

    Aop = spdiagm(0 => ones(N)) + VK
    top_rows, lower_bw, upper_bw = find_fast_structure(Aop)
    err_inf = find_fast_err(Aop, N)
    return (
        N=N,
        path=path,
        assembly=string(assembly),
        uses_function_K=uses_function_K,
        uses_precomputed_C=uses_precomputed_C,
        C_rows=C === nothing ? missing : size(C, 1),
        C_cols=C === nothing ? missing : size(C, 2),
        t_C_build=t_C_build,
        t_VK=t_vk,
        nnz_VK=nnz(VK),
        top_rows=top_rows,
        lower_bw=lower_bw,
        upper_bw=upper_bw,
        err_inf_if_available=err_inf,
        status=status,
        notes=notes,
    )
end

function find_fast_profile_raw_default(N::Int)
    Profile.clear()
    GC.gc()
    @profile VoltMat_Cheb.voltmat_cheb_square(find_fast_K, N - 1; tol=FIND_FAST_TOL)
    open(FIND_FAST_PROFILE, "w") do io
        println(io, "Profile for raw_function_default_fft at N=$N")
        println(io, "Call: VoltMat_Cheb.voltmat_cheb_square(K, N - 1; tol=tol)")
        println(io)
        Profile.print(io; format=:flat, maxdepth=20)
    end
    return FIND_FAST_PROFILE
end

function find_fast_write_md(path::AbstractString, rows, git_status, src_diff, aging_diff, method_function, method_C, t_C_build, C)
    fastest_by_N = Dict{Int,Any}()
    for N in unique(row.N for row in rows)
        ok = [row for row in rows if row.N == N && row.status == "success" && isfinite(row.t_VK)]
        isempty(ok) || (fastest_by_N[N] = ok[argmin([row.t_VK for row in ok])])
    end
    open(path, "w") do io
        println(io, "# debug_find_fast_cheb_path_N1000")
        println(io)
        println(io, "## Git Status")
        println(io, "```text")
        println(io, isempty(git_status) ? "(clean status output)" : git_status)
        println(io, "```")
        println(io)
        println(io, "## git diff -- src/VoltMat/voltmat_cheb.jl")
        println(io, "```diff")
        println(io, isempty(src_diff) ? "(no diff)" : src_diff)
        println(io, "```")
        println(io)
        println(io, "## git diff -- examples/ex1_2ndVolterraEqn/ex1_aging_maxwell.jl")
        println(io, "```diff")
        println(io, isempty(aging_diff) ? "(no diff)" : aging_diff)
        println(io, "```")
        println(io)
        println(io, "## Loaded Methods")
        println(io, "- function-K method: `$(method_function)`")
        println(io, "- precomputed-C method: `$(method_C)`")
        println(io)
        println(io, "## Precomputed C")
        println(io, "- C size: `$(size(C))`")
        println(io, "- t_C_build: `$(@sprintf("%.6e", t_C_build))` s")
        println(io, "- C was built with `VoltMat_Cheb.adaptive_cheb2_coeffmat((x,y)->K(x,(1+x)*(1+y)/2-1))` via `ex1_build_adaptive_cheb_kernel_coeff_metadata()`.")
        println(io)
        println(io, "## Timing Results")
        println(io, "| N | path | assembly | uses K | uses C | C size | t_C_build | t_VK | nnz | top/l/u | err | status |")
        println(io, "|---:|---|---|---:|---:|---|---:|---:|---:|---|---:|---|")
        for row in rows
            csize = row.C_rows === missing ? "" : "$(row.C_rows)x$(row.C_cols)"
            top = row.top_rows === missing ? "" : "$(row.top_rows)/$(row.lower_bw)/$(row.upper_bw)"
            println(io, "| $(row.N) | $(row.path) | $(row.assembly) | $(row.uses_function_K) | $(row.uses_precomputed_C) | $(csize) | $(@sprintf("%.6e", row.t_C_build)) | $(@sprintf("%.6e", row.t_VK)) | $(row.nnz_VK) | $(top) | $(@sprintf("%.3e", row.err_inf_if_available)) | $(row.status) |")
        end
        println(io)
        println(io, "## Fastest Path By N")
        for N in sort(collect(keys(fastest_by_N)))
            row = fastest_by_N[N]
            println(io, "- N=$N: `$(row.path)` with `t_VK=$(@sprintf("%.6e", row.t_VK))` s")
        end
        println(io)
        println(io, "## Profile")
        println(io, "Raw default profile saved to `$(basename(FIND_FAST_PROFILE))`.")
        println(io)
        println(io, "## Interpretation")
        println(io, "- The historical attachment's tens-of-milliseconds `t_VK` near N=1000 cannot correspond to the current raw function-K default FFT path if this run's timings are representative.")
        println(io, "- If a precomputed-C compressed path lands near that scale, it is a candidate for the historical timing boundary: fixed kernel coefficients outside the N loop, then VK assembly only.")
        println(io, "- Do not use the slow `precomputed_C_fft` path with an untrimmed or incompatible C-interface as evidence that all FFT assembly is slow.")
    end
    return path
end

function run_debug_find_fast_cheb_path_N1000()
    println("Finding fast Cheb path near N=1000")
    println("Ns: ", FIND_FAST_NS)
    println("CSV: ", FIND_FAST_CSV)
    println("MD: ", FIND_FAST_MD)
    println("Profile: ", FIND_FAST_PROFILE)

    git_status = find_fast_git(`git status --short`)
    src_diff = find_fast_git(`git diff -- src/VoltMat/voltmat_cheb.jl`)
    aging_diff = find_fast_git(`git diff -- examples/ex1_2ndVolterraEqn/ex1_aging_maxwell.jl`)

    method_function = which(VoltMat_Cheb.voltmat_cheb_square, (typeof(find_fast_K), Int))
    adaptive = ex1_build_adaptive_cheb_kernel_coeff_metadata()
    C = adaptive.C
    t_C_build = adaptive.kernel_coeff_time
    method_C = which(VoltMat_Cheb.voltmat_cheb_square, (typeof(C), Int))

    rows = NamedTuple[]
    for N in FIND_FAST_NS
        println()
        @printf("N = %d\n", N)
        path_specs = (
            (
                path="raw_function_default_fft",
                assembly=:default_fft,
                uses_function_K=true,
                uses_precomputed_C=false,
                C=nothing,
                t_C_build=0.0,
                notes="Function-K interface with no assembly keyword; current default is :fft.",
                builder=() -> VoltMat_Cheb.voltmat_cheb_square(find_fast_K, N - 1; tol=FIND_FAST_TOL),
            ),
            (
                path="raw_function_explicit_fft",
                assembly=:fft,
                uses_function_K=true,
                uses_precomputed_C=false,
                C=nothing,
                t_C_build=0.0,
                notes="Function-K interface with explicit assembly=:fft.",
                builder=() -> VoltMat_Cheb.voltmat_cheb_square(find_fast_K, N - 1; tol=FIND_FAST_TOL, assembly=:fft),
            ),
            (
                path="raw_function_compressed_rankupdate",
                assembly=:compressed_rankupdate,
                uses_function_K=true,
                uses_precomputed_C=false,
                C=nothing,
                t_C_build=0.0,
                notes="Function-K interface still builds adaptive C internally, then uses compressed rank-update assembly.",
                builder=() -> VoltMat_Cheb.voltmat_cheb_square(find_fast_K, N - 1; tol=FIND_FAST_TOL, assembly=:compressed_rankupdate),
            ),
            (
                path="raw_function_compressed",
                assembly=:compressed,
                uses_function_K=true,
                uses_precomputed_C=false,
                C=nothing,
                t_C_build=0.0,
                notes="Function-K interface still builds adaptive C internally, then uses compressed assembly.",
                builder=() -> VoltMat_Cheb.voltmat_cheb_square(find_fast_K, N - 1; tol=FIND_FAST_TOL, assembly=:compressed),
            ),
            (
                path="precomputed_C_fft",
                assembly=:fft,
                uses_function_K=false,
                uses_precomputed_C=true,
                C=C,
                t_C_build=t_C_build,
                notes="Adaptive absorbed-kernel C is built once outside per-N timing, then C-interface assembly=:fft is timed.",
                builder=() -> VoltMat_Cheb.voltmat_cheb_square(C, N - 1; tol=FIND_FAST_TOL, assembly=:fft),
            ),
            (
                path="precomputed_C_compressed_rankupdate",
                assembly=:compressed_rankupdate,
                uses_function_K=false,
                uses_precomputed_C=true,
                C=C,
                t_C_build=t_C_build,
                notes="Adaptive absorbed-kernel C is built once outside per-N timing, then C-interface assembly=:compressed_rankupdate is timed.",
                builder=() -> VoltMat_Cheb.voltmat_cheb_square(C, N - 1; tol=FIND_FAST_TOL, assembly=:compressed_rankupdate),
            ),
            (
                path="ex1_aging_current",
                assembly=:default_fft,
                uses_function_K=true,
                uses_precomputed_C=false,
                C=nothing,
                t_C_build=0.0,
                notes="Current ex1_aging_maxwell.jl table-style t_VK: GC.gc(); @elapsed VK = voltmat_cheb_square(K, N - 1; tol=tol).",
                builder=() -> VoltMat_Cheb.voltmat_cheb_square(find_fast_K, N - 1; tol=FIND_FAST_TOL),
            ),
        )
        for spec in path_specs
            row = find_fast_row(;
                N=N,
                path=spec.path,
                assembly=spec.assembly,
                uses_function_K=spec.uses_function_K,
                uses_precomputed_C=spec.uses_precomputed_C,
                C=spec.C,
                t_C_build=spec.t_C_build,
                builder=spec.builder,
                notes=spec.notes,
            )
            push!(rows, row)
            find_fast_write_csv(FIND_FAST_CSV, rows)
            @printf(
                "  %-38s tVK=%.6e nnz=%s top/l/u=%s/%s/%s err=%.3e status=%s\n",
                row.path,
                row.t_VK,
                string(row.nnz_VK),
                string(row.top_rows),
                string(row.lower_bw),
                string(row.upper_bw),
                row.err_inf_if_available,
                row.status,
            )
        end
    end

    find_fast_profile_raw_default(1000)
    find_fast_write_md(FIND_FAST_MD, rows, git_status, src_diff, aging_diff, method_function, method_C, t_C_build, C)
    println()
    println("Wrote ", FIND_FAST_CSV)
    println("Wrote ", FIND_FAST_MD)
    println("Wrote ", FIND_FAST_PROFILE)
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_debug_find_fast_cheb_path_N1000()
end
