using BenchmarkTools
using LinearAlgebra
using Printf
using SparseArrays

ex1_dir = @__DIR__
root = normpath(joinpath(ex1_dir, "..", ".."))
go_project = joinpath(root, "others", "GutlebOlver")

# Load GutlebOlver through its own project path.  Do not include or load
# VoltMat_Cheb in this script.
pushfirst!(LOAD_PATH, go_project)

using ApproxFun
using BandedMatrices
using GutlebOlver

# ============================================================
# Same aging Maxwell problem as the verified VoltMat_Cheb script.
# ============================================================

tau_xy(x, y) = 1.5 + x - 0.5 * y

function K(x, y)
    tau = tau_xy(x, y)
    return exp(-(x - y) / tau) / tau
end

step_layer(x, x0, delta) = 0.5 * (1.0 + tanh((x - x0) / delta))

function u_exact(x)
    return 1.0 +
           1.0 * step_layer(x, -0.35, 1.0e-3) -
           0.6 * step_layer(x, 0.42, 1.5e-3)
end

x_from_s(s) = 2.0 * s - 1.0
s_from_x(x) = 0.5 * (x + 1.0)
u_go(s) = u_exact(x_from_s(s))

# Transformed equation:
#     u_go(s) + int_0^s K_go(s,r) u_go(r) dr = rhs_go(s)
K_go(s, r) = 2.0 * K(x_from_s(s), x_from_s(r))

function gausslegendre_rule(n::Int)
    n <= 0 && throw(ArgumentError("n must be positive"))
    n == 1 && return [0.0], [2.0]

    beta = [k / sqrt(4.0 * k^2 - 1.0) for k in 1:(n - 1)]
    E = eigen(SymTridiagonal(zeros(n), beta))
    p = sortperm(E.values)
    return collect(E.values[p]), (2.0 .* abs2.(E.vectors[1, :]))[p]
end

function gauss_integral_on_interval(f, a::Float64, b::Float64, gl_x, gl_w)
    b > a || return 0.0

    c = 0.5 * (a + b)
    h = 0.5 * (b - a)
    acc = 0.0

    @inbounds for q in eachindex(gl_x)
        acc += gl_w[q] * f(c + h * gl_x[q])
    end

    return h * acc
end

function rhs_integral_go(s::Real, gl_x, gl_w; split_L=40.0)
    sf = Float64(s)
    sf <= 0.0 && return 0.0

    pts = Float64[0.0, sf]
    for (x0, delta) in ((-0.35, 1.0e-3), (0.42, 1.5e-3))
        for xsplit in (x0 - split_L * delta, x0, x0 + split_L * delta)
            ssplit = s_from_x(xsplit)
            0.0 < ssplit < sf && push!(pts, ssplit)
        end
    end

    sort!(pts)

    unique_pts = Float64[]
    sizehint!(unique_pts, length(pts))
    lastp = NaN

    for p in pts
        if isempty(unique_pts) || abs(p - lastp) > 10eps(Float64) * max(1.0, abs(p))
            push!(unique_pts, p)
            lastp = p
        end
    end

    total = 0.0
    @inbounds for j in 1:(length(unique_pts) - 1)
        a = unique_pts[j]
        b = unique_pts[j + 1]
        total += gauss_integral_on_interval(r -> K_go(sf, r) * u_go(r), a, b, gl_x, gl_w)
    end

    return total
end

function make_rhs_go(gl_x, gl_w; split_L=40.0)
    return s -> u_go(s) + rhs_integral_go(s, gl_x, gl_w; split_L=split_L)
end

function make_test_grid(; Mglobal=8001, Mlocal=5001)
    xg = collect(range(-1.0, 1.0; length=Mglobal))
    xl1 = collect(range(max(-1.0, -0.35 - 0.04), min(1.0, -0.35 + 0.04); length=Mlocal))
    xl2 = collect(range(max(-1.0, 0.42 - 0.05), min(1.0, 0.42 + 0.05); length=Mlocal))
    return s_from_x.(sort(unique(vcat(xg, xl1, xl2, [-0.35, 0.42]))))
end

# ============================================================
# Gutleb--Olver-only construction.
# ============================================================

function build_VK_go(N::Int, kerneldepth::Int)
    plan = precompute_triangular_kernel_plan(kerneldepth, true)
    coeffs = triangular_kernel_coeffs_from_plan(plan, K_go)
    Vraw = triVolterraFullKernelOpP01_strictband_from_coeffs(coeffs, N, true)
    return build_secondkind_volterra_strictband(Vraw, N)
end

function build_rhs_go(N::Int, rhs_fn)
    rhs_coeff, _ = build_go_rhs_coefficients(
        rhs_fn,
        N;
        alpha=0,
        beta=1,
        domain=0..1,
        method=:fixed_gauss_jacobi,
        validate=false,
    )
    return rhs_coeff
end

function build_system_go(VK, N::Int)
    Aop = copy(VK)
    @inbounds for i in 1:N
        Aop[i, i] += 1.0
    end
    return Aop
end

solve_system(Aop, rhs) = Aop \ rhs

function end_to_end_go(N::Int, kerneldepth::Int, rhs_fn)
    VK = build_VK_go(N, kerneldepth)
    rhs = build_rhs_go(N, rhs_fn)
    Aop = build_system_go(VK, N)
    return solve_system(Aop, rhs)
end

function matrix_nnz(A)
    try
        return safe_nnz(A)
    catch
        # Continue to generic fallbacks below.
    end

    try
        return nnz(A)
    catch
        # Continue to count fallback below.
    end

    try
        return count(!iszero, A)
    catch
        return missing
    end
end

function matrix_size(A)
    try
        return size(A)
    catch
        return (missing, missing)
    end
end

function benchmark_summary(f)
    GC.gc()

    bench = @benchmark $f() samples=3 evals=1 seconds=1000000

    return (
        min_sec=minimum(bench).time / 1e9,
        median_sec=median(bench).time / 1e9,
        mean_sec=mean(bench).time / 1e9,
        samples=length(bench.times),
        memory=BenchmarkTools.memory(bench),
        allocs=BenchmarkTools.allocs(bench),
    )
end

function csv_escape(x)
    s = string(x)
    if occursin(",", s) || occursin("\"", s) || occursin("\n", s) || occursin("\r", s)
        return "\"" * replace(s, "\"" => "\"\"") * "\""
    end
    return s
end

function init_csv(path, columns)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(string.(columns), ","))
        flush(io)
    end
end

function append_csv(path, columns, row::Dict{Symbol, Any})
    open(path, "a") do io
        println(io, join((csv_escape(get(row, col, "")) for col in columns), ","))
        flush(io)
    end
end

function make_ok_row(;
    method,
    N,
    tol,
    equation_sign,
    VK,
    Aop,
    build_stats,
    rhs_stats,
    system_stats,
    solve_stats,
    end2end_stats,
    residual_norm,
    relative_residual,
    max_abs_error,
    relative_error,
    max_abs_uh,
    max_abs_uexact,
)
    return Dict{Symbol, Any}(
        :status => "ok",
        :error_message => "",
        :method => method,
        :N => N,
        :tol => tol,
        :equation_sign => equation_sign,
        :operator_type => string(typeof(VK)),
        :system_type => string(typeof(Aop)),
        :operator_nnz => matrix_nnz(VK),
        :system_nnz => matrix_nnz(Aop),

        :build_min_sec => build_stats.min_sec,
        :build_median_sec => build_stats.median_sec,
        :build_mean_sec => build_stats.mean_sec,
        :build_samples => build_stats.samples,
        :build_memory => build_stats.memory,
        :build_allocs => build_stats.allocs,

        :rhs_min_sec => rhs_stats.min_sec,
        :rhs_median_sec => rhs_stats.median_sec,
        :rhs_mean_sec => rhs_stats.mean_sec,
        :rhs_samples => rhs_stats.samples,
        :rhs_memory => rhs_stats.memory,
        :rhs_allocs => rhs_stats.allocs,

        :system_min_sec => system_stats.min_sec,
        :system_median_sec => system_stats.median_sec,
        :system_mean_sec => system_stats.mean_sec,
        :system_samples => system_stats.samples,
        :system_memory => system_stats.memory,
        :system_allocs => system_stats.allocs,

        :solve_min_sec => solve_stats.min_sec,
        :solve_median_sec => solve_stats.median_sec,
        :solve_mean_sec => solve_stats.mean_sec,
        :solve_samples => solve_stats.samples,
        :solve_memory => solve_stats.memory,
        :solve_allocs => solve_stats.allocs,

        :end2end_min_sec => end2end_stats.min_sec,
        :end2end_median_sec => end2end_stats.median_sec,
        :end2end_mean_sec => end2end_stats.mean_sec,
        :end2end_samples => end2end_stats.samples,
        :end2end_memory => end2end_stats.memory,
        :end2end_allocs => end2end_stats.allocs,

        :residual_norm => residual_norm,
        :relative_residual => relative_residual,
        :max_abs_error => max_abs_error,
        :relative_error => relative_error,
        :max_abs_uh => max_abs_uh,
        :max_abs_uexact => max_abs_uexact,
    )
end

function make_failed_row(columns, method, N, tol, equation_sign, err, bt)
    row = Dict{Symbol, Any}(col => NaN for col in columns)

    row[:status] = "failed"
    row[:error_message] = sprint(showerror, err, bt)
    row[:method] = method
    row[:N] = N
    row[:tol] = tol
    row[:equation_sign] = equation_sign

    return row
end

function solve_one_N(N::Int, tol, kerneldepth::Int, rhs_fn, go_space, stest, utest_exact, equation_sign)
    GC.gc()
    println()
    println("N = ", N)

    local VK, rhs, Aop, uh_coeff

    first_total = @elapsed begin
        VK = build_VK_go(N, kerneldepth)
        rhs = build_rhs_go(N, rhs_fn)
        Aop = build_system_go(VK, N)
        uh_coeff = solve_system(Aop, rhs)
    end

    println("first-call/end-to-end timing includes compilation and cache effects: ", first_total, " sec")

    residual = Aop * uh_coeff - rhs
    residual_norm = norm(residual)
    relative_residual = residual_norm / max(norm(rhs), eps(Float64))

    uh = Fun(go_space, uh_coeff).(stest)
    err = uh .- utest_exact
    max_abs_error = maximum(abs, err)
    relative_error = norm(err) / max(norm(utest_exact), eps(Float64))
    max_abs_uh = maximum(abs, uh)
    max_abs_uexact = maximum(abs, utest_exact)

    println("VK / operator type = ", typeof(VK))
    println("Aop type = ", typeof(Aop))
    println("size(Aop) = ", matrix_size(Aop))
    println("nnz(Aop) = ", matrix_nnz(Aop))
    println("residual_norm = ", residual_norm)
    println("relative_residual = ", relative_residual)
    println("max_abs_error = ", max_abs_error)
    println("relative_error = ", relative_error)

    build_stats = benchmark_summary(() -> build_VK_go(N, kerneldepth))
    rhs_stats = benchmark_summary(() -> build_rhs_go(N, rhs_fn))
    system_stats = benchmark_summary(() -> build_system_go(VK, N))
    solve_stats = benchmark_summary(() -> solve_system(Aop, rhs))
    end2end_stats = benchmark_summary(() -> end_to_end_go(N, kerneldepth, rhs_fn))

    return make_ok_row(
        method="GutlebOlver",
        N=N,
        tol=tol,
        equation_sign=equation_sign,
        VK=VK,
        Aop=Aop,
        build_stats=build_stats,
        rhs_stats=rhs_stats,
        system_stats=system_stats,
        solve_stats=solve_stats,
        end2end_stats=end2end_stats,
        residual_norm=residual_norm,
        relative_residual=relative_residual,
        max_abs_error=max_abs_error,
        relative_error=relative_error,
        max_abs_uh=max_abs_uh,
        max_abs_uexact=max_abs_uexact,
    )
end

function make_linear_Ns()
    Ns = unique(round.(Int, range(100, 30000; length=50)))
    length(Ns) == 50 || error("Expected 50 distinct N values, got $(length(Ns)): $(Ns)")
    return Ns
end

function main()
    Ns = make_linear_Ns()
    tol = 1e-14
    kerneldepth = 32
    rhs_q = 160
    rhs_split_L = 40.0
    equation_sign = "u + V_K u = f"
    output_csv = joinpath(ex1_dir, "bench_gutleb_olver_N100_30000_50_results.csv")

    gl_x, gl_w = gausslegendre_rule(rhs_q)
    rhs_fn = make_rhs_go(gl_x, gl_w; split_L=rhs_split_L)

    go_space = Jacobi(0, 1, 0..1)
    stest = make_test_grid()
    utest_exact = u_go.(stest)

    columns = [
        :status,
        :error_message,
        :method,
        :N,
        :tol,
        :equation_sign,
        :operator_type,
        :system_type,
        :operator_nnz,
        :system_nnz,

        :build_min_sec,
        :build_median_sec,
        :build_mean_sec,
        :build_samples,
        :rhs_min_sec,
        :rhs_median_sec,
        :rhs_mean_sec,
        :rhs_samples,
        :system_min_sec,
        :system_median_sec,
        :system_mean_sec,
        :system_samples,
        :solve_min_sec,
        :solve_median_sec,
        :solve_mean_sec,
        :solve_samples,
        :end2end_min_sec,
        :end2end_median_sec,
        :end2end_mean_sec,
        :end2end_samples,

        :residual_norm,
        :relative_residual,
        :max_abs_error,
        :relative_error,
        :max_abs_uh,
        :max_abs_uexact,

        :build_memory,
        :build_allocs,
        :rhs_memory,
        :rhs_allocs,
        :system_memory,
        :system_allocs,
        :solve_memory,
        :solve_allocs,
        :end2end_memory,
        :end2end_allocs,
    ]

    println("Benchmark GutlebOlver ex1 aging Maxwell")
    println("Ns = ", Ns)
    println("number of N values = ", length(Ns))
    println("tol = ", tol)
    println("kerneldepth = ", kerneldepth)
    println("rhs_q = ", rhs_q)
    println("equation_sign = ", equation_sign)
    println("VoltMat_Cheb defined = ", isdefined(Main, :VoltMat_Cheb))
    println("GutlebOlver defined = ", isdefined(Main, :GutlebOlver))
    println("CSV = ", output_csv)

    init_csv(output_csv, columns)

    @printf(
        "%8s %10s %10s %10s %10s %10s %12s %12s %8s\n",
        "N",
        "build",
        "rhs",
        "system",
        "solve",
        "end2end",
        "residual",
        "rel_error",
        "status",
    )

    for Nval in Ns
        try
            row = solve_one_N(
                Nval,
                tol,
                kerneldepth,
                rhs_fn,
                go_space,
                stest,
                utest_exact,
                equation_sign,
            )

            append_csv(output_csv, columns, row)

            @printf(
                "%8d %10.4f %10.4f %10.4f %10.4f %10.4f %12.3e %12.3e %8s\n",
                row[:N],
                row[:build_median_sec],
                row[:rhs_median_sec],
                row[:system_median_sec],
                row[:solve_median_sec],
                row[:end2end_median_sec],
                row[:residual_norm],
                row[:relative_error],
                row[:status],
            )
        catch err
            bt = catch_backtrace()
            row = make_failed_row(columns, "GutlebOlver", Nval, tol, equation_sign, err, bt)
            append_csv(output_csv, columns, row)

            @printf(
                "%8d %10s %10s %10s %10s %10s %12s %12s %8s\n",
                Nval,
                "NaN",
                "NaN",
                "NaN",
                "NaN",
                "NaN",
                "NaN",
                "NaN",
                "failed",
            )

            println("Failed N = ", Nval)
            showerror(stdout, err, bt)
            println()
        end
    end

    println()
    println("Wrote ", output_csv)
end

main()
