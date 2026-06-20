using BenchmarkTools
using LinearAlgebra
using Printf
using SparseArrays

const EX1_DIR = @__DIR__
const ROOT = normpath(joinpath(EX1_DIR, "..", ".."))
const GO_PROJECT = joinpath(ROOT, "others", "GutlebOlver")

# Load GutlebOlver as a package through its own project without including any
# VoltMat_Cheb source or mixed compare/common scripts.
pushfirst!(LOAD_PATH, GO_PROJECT)

using ApproxFun
using BandedMatrices
using GutlebOlver

const Ns = [1000, 4000, 8000, 16000]
const tol = 1e-14
const KERNELDEPTH = 32
const OUTPUT_CSV = joinpath(EX1_DIR, "debug_gutleb_olver_Ns_results.csv")
const EQUATION_SIGN = "u + V_K u = f"
const GO_SPACE = Jacobi(0, 1, 0..1)  # Gutleb--Olver shifted Jacobi P^(1,0) representation.

# ============================================================
# Same aging Maxwell problem as debug_voltmat_cheb_N1000.jl.
# ============================================================

tau_xy(x, y) = 1.5 + x - 0.5 * y

function K(x, y)
    tau = tau_xy(x, y)
    return exp(-(x - y) / tau) / tau
end

const x1 = -0.35
const x2 = 0.42
const A1 = 1.0
const A2 = -0.6
const delta1 = 1.0e-3
const delta2 = 1.5e-3

step_layer(x, x0, delta) = 0.5 * (1.0 + tanh((x - x0) / delta))
u_exact(x) = 1.0 + A1 * step_layer(x, x1, delta1) + A2 * step_layer(x, x2, delta2)

x_from_s(s) = 2.0 * s - 1.0
s_from_x(x) = 0.5 * (x + 1.0)
u_go(s) = u_exact(x_from_s(s))

# Transformed equation:
# u_go(s) + int_0^s K_go(s,r) u_go(r) dr = rhs_go(s)
K_go(s, r) = 2.0 * K(x_from_s(s), x_from_s(r))

const RHS_Q = 160
const RHS_SPLIT_L = 40.0

function gausslegendre_rule(n::Int)
    n <= 0 && throw(ArgumentError("n must be positive"))
    n == 1 && return [0.0], [2.0]
    beta = [k / sqrt(4.0 * k^2 - 1.0) for k in 1:(n - 1)]
    E = eigen(SymTridiagonal(zeros(n), beta))
    p = sortperm(E.values)
    return collect(E.values[p]), (2.0 .* abs2.(E.vectors[1, :]))[p]
end

const GL_X, GL_W = gausslegendre_rule(RHS_Q)

function gauss_integral_on_interval(f, a::Float64, b::Float64)
    b > a || return 0.0
    c = 0.5 * (a + b)
    h = 0.5 * (b - a)
    acc = 0.0
    @inbounds for q in eachindex(GL_X)
        acc += GL_W[q] * f(c + h * GL_X[q])
    end
    return h * acc
end

function rhs_integral_go(s::Real)
    sf = Float64(s)
    sf <= 0.0 && return 0.0

    pts = Float64[0.0, sf]
    for (x0, delta) in ((x1, delta1), (x2, delta2))
        for xsplit in (x0 - RHS_SPLIT_L * delta, x0, x0 + RHS_SPLIT_L * delta)
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
        total += gauss_integral_on_interval(r -> K_go(sf, r) * u_go(r), a, b)
    end
    return total
end

rhs_go(s) = u_go(s) + rhs_integral_go(s)

function make_test_grid(; Mglobal=8001, Mlocal=5001)
    xg = collect(range(-1.0, 1.0; length=Mglobal))
    xl1 = collect(range(max(-1.0, x1 - 0.04), min(1.0, x1 + 0.04); length=Mlocal))
    xl2 = collect(range(max(-1.0, x2 - 0.05), min(1.0, x2 + 0.05); length=Mlocal))
    return s_from_x.(sort(unique(vcat(xg, xl1, xl2, [x1, x2]))))
end

const STEST = make_test_grid()
const UTEST_EXACT = u_go.(STEST)

# ============================================================
# Gutleb--Olver-only construction.
# ============================================================

function build_VK_go(N::Int)
    plan = precompute_triangular_kernel_plan(KERNELDEPTH, true)
    coeffs = triangular_kernel_coeffs_from_plan(plan, K_go)
    Vraw = triVolterraFullKernelOpP01_strictband_from_coeffs(coeffs, N, true)
    return build_secondkind_volterra_strictband(Vraw, N)
end

function build_rhs_go(N::Int)
    rhs_coeff, _ = build_go_rhs_coefficients(
        rhs_go,
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

function end_to_end_go(N::Int)
    VK = build_VK_go(N)
    rhs = build_rhs_go(N)
    Aop = build_system_go(VK, N)
    return solve_system(Aop, rhs)
end

function matrix_nnz(A)
    if A isa BandedMatrix || A isa SparseMatrixCSC
        return safe_nnz(A)
    end
    return count(!iszero, A)
end

function matrix_size(A)
    try
        return size(A)
    catch
        return (missing, missing)
    end
end

function matrix_norm(A)
    try
        return norm(A)
    catch
        return NaN
    end
end

function run_benchmark(f; seconds=3, samples=3)
    bench = @benchmark $f() seconds=3 samples=3 evals=1
    return (
        min=minimum(bench).time / 1e9,
        median=median(bench).time / 1e9,
        mean=mean(bench).time / 1e9,
    )
end

function csv_escape(x)
    s = string(x)
    if occursin(",", s) || occursin("\"", s) || occursin("\n", s) || occursin("\r", s)
        return "\"" * replace(s, "\"" => "\"\"") * "\""
    end
    return s
end

const CSV_COLUMNS = [
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
    :rhs_min_sec,
    :rhs_median_sec,
    :rhs_mean_sec,
    :system_min_sec,
    :system_median_sec,
    :system_mean_sec,
    :solve_min_sec,
    :solve_median_sec,
    :solve_mean_sec,
    :end2end_min_sec,
    :end2end_median_sec,
    :end2end_mean_sec,
    :residual_norm,
    :relative_residual,
    :max_abs_error,
    :relative_error,
    :max_abs_uh,
    :max_abs_uexact,
]

function init_csv(path)
    open(path, "w") do io
        println(io, join(string.(CSV_COLUMNS), ","))
    end
end

function append_csv(path, row)
    open(path, "a") do io
        println(io, join((csv_escape(getproperty(row, col)) for col in CSV_COLUMNS), ","))
    end
end

function solve_one_N(N::Int)
    GC.gc()
    println()
    println("N = ", N)

    local VK, rhs, Aop, uh_coeff
    first_total = @elapsed begin
        VK = build_VK_go(N)
        rhs = build_rhs_go(N)
        Aop = build_system_go(VK, N)
        uh_coeff = solve_system(Aop, rhs)
    end
    println("first-call/end-to-end timing includes compilation and cache effects: ", first_total, " sec")

    residual = Aop * uh_coeff - rhs
    residual_norm = norm(residual)
    relative_residual = residual_norm / max(norm(rhs), eps(Float64))

    uh = Fun(GO_SPACE, uh_coeff).(STEST)
    err = uh .- UTEST_EXACT
    max_abs_error = maximum(abs, err)
    relative_error = norm(err) / max(norm(UTEST_EXACT), eps(Float64))

    println("VK / operator type = ", typeof(VK))
    println("Aop type = ", typeof(Aop))
    println("size(Aop) = ", matrix_size(Aop))
    println("nnz(Aop) = ", matrix_nnz(Aop))
    println("residual_norm = ", residual_norm)
    println("relative_residual = ", relative_residual)
    println("max_abs_error = ", max_abs_error)
    println("relative_error = ", relative_error)

    build_stats = run_benchmark(() -> build_VK_go(N))
    rhs_stats = run_benchmark(() -> build_rhs_go(N))
    system_stats = run_benchmark(() -> build_system_go(VK, N))
    solve_stats = run_benchmark(() -> solve_system(Aop, rhs))
    end2end_stats = run_benchmark(() -> end_to_end_go(N))

    return (
        method="GutlebOlver",
        N=N,
        tol=tol,
        equation_sign=EQUATION_SIGN,
        operator_type=string(typeof(VK)),
        system_type=string(typeof(Aop)),
        operator_nnz=matrix_nnz(VK),
        system_nnz=matrix_nnz(Aop),
        build_min_sec=build_stats.min,
        build_median_sec=build_stats.median,
        build_mean_sec=build_stats.mean,
        rhs_min_sec=rhs_stats.min,
        rhs_median_sec=rhs_stats.median,
        rhs_mean_sec=rhs_stats.mean,
        system_min_sec=system_stats.min,
        system_median_sec=system_stats.median,
        system_mean_sec=system_stats.mean,
        solve_min_sec=solve_stats.min,
        solve_median_sec=solve_stats.median,
        solve_mean_sec=solve_stats.mean,
        end2end_min_sec=end2end_stats.min,
        end2end_median_sec=end2end_stats.median,
        end2end_mean_sec=end2end_stats.mean,
        residual_norm=residual_norm,
        relative_residual=relative_residual,
        max_abs_error=max_abs_error,
        relative_error=relative_error,
        max_abs_uh=maximum(abs, uh),
        max_abs_uexact=maximum(abs, UTEST_EXACT),
    )
end

println("Debug GutlebOlver ex1 aging Maxwell")
println("Ns = ", Ns)
println("tol = ", tol)
println("kerneldepth = ", KERNELDEPTH)
println("equation_sign = ", EQUATION_SIGN)
println("VoltMat_Cheb defined = ", isdefined(Main, :VoltMat_Cheb))
println("GutlebOlver defined = ", isdefined(Main, :GutlebOlver))
println("CSV = ", OUTPUT_CSV)

init_csv(OUTPUT_CSV)

@printf("%8s %10s %10s %10s %10s %10s %12s %12s\n",
    "N", "build", "rhs", "system", "solve", "end2end", "residual", "rel_error")

for Nval in Ns
    row = solve_one_N(Nval)
    append_csv(OUTPUT_CSV, row)
    @printf("%8d %10.4f %10.4f %10.4f %10.4f %10.4f %12.3e %12.3e\n",
        row.N,
        row.build_median_sec,
        row.rhs_median_sec,
        row.system_median_sec,
        row.solve_median_sec,
        row.end2end_median_sec,
        row.residual_norm,
        row.relative_error)
end

println()
println("Wrote ", OUTPUT_CSV)
