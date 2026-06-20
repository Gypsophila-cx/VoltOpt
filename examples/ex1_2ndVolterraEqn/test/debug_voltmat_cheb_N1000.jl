using FFTW
using InteractiveUtils
using LinearAlgebra
using Printf
using SparseArrays

# Match debug_simple.jl: path is relative to examples/ex1_2ndVolterraEqn.
ROOT = normpath(joinpath(@__DIR__, "..", ".."))

include(joinpath(ROOT, "src", "VoltMat", "voltmat_cheb.jl"))

const Ns = [1000, 4000, 8000, 16000]
const tol = 1e-14
const OUTPUT_CSV = joinpath(@__DIR__, "debug_voltmat_cheb_Ns_results.csv")

# ============================================================
# Kernel copied from debug_simple.jl / ex1_aging_maxwell.jl.
# ============================================================

tau_xy(x, y) = 1.5 + x - 0.5 * y

function K(x, y)
    tau = tau_xy(x, y)
    return exp(-(x - y) / tau) / tau
end

# Original ex1 equation, confirmed from ex1_aging_maxwell.jl:
# u + V_K u = f
# Aop = spdiagm(0 => ones(N)) + VK
const EQUATION_SIGN = "I_plus_VK"

# ============================================================
# Exact solution and independent RHS copied from ex1_aging_maxwell.jl.
# ============================================================

const x1 = -0.35
const x2 = 0.42
const A1 = 1.0
const A2 = -0.6
const delta1 = 1.0e-3
const delta2 = 1.5e-3

step_layer(x, x0, delta) = 0.5 * (1.0 + tanh((x - x0) / delta))

function u_exact(x)
    return 1.0 + A1 * step_layer(x, x1, delta1) + A2 * step_layer(x, x2, delta2)
end

const RHS_Q = 160
const RHS_SPLIT_L = 40.0

function gausslegendre_rule(n::Int)
    n <= 0 && throw(ArgumentError("n must be positive"))
    if n == 1
        return [0.0], [2.0]
    end

    beta = [k / sqrt(4.0 * k^2 - 1.0) for k in 1:(n - 1)]
    T = SymTridiagonal(zeros(n), beta)
    E = eigen(T)

    x = collect(E.values)
    w = 2.0 .* abs2.(E.vectors[1, :])
    p = sortperm(x)
    return x[p], w[p]
end

const GL_X, GL_W = gausslegendre_rule(RHS_Q)

function gauss_integral_on_interval(f, a::Float64, b::Float64)
    b > a || return 0.0
    c = 0.5 * (a + b)
    h = 0.5 * (b - a)
    s = 0.0
    @inbounds for q in eachindex(GL_X)
        y = c + h * GL_X[q]
        s += GL_W[q] * f(y)
    end
    return h * s
end

function rhs_integral_direct(x::Real)
    xf = Float64(x)
    xf <= -1.0 && return 0.0

    pts = Float64[-1.0, xf]

    a1 = x1 - RHS_SPLIT_L * delta1
    b1 = x1 + RHS_SPLIT_L * delta1
    -1.0 < a1 < xf && push!(pts, a1)
    -1.0 < x1 < xf && push!(pts, x1)
    -1.0 < b1 < xf && push!(pts, b1)

    a2 = x2 - RHS_SPLIT_L * delta2
    b2 = x2 + RHS_SPLIT_L * delta2
    -1.0 < a2 < xf && push!(pts, a2)
    -1.0 < x2 < xf && push!(pts, x2)
    -1.0 < b2 < xf && push!(pts, b2)

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
        total += gauss_integral_on_interval(y -> K(xf, y) * u_exact(y), a, b)
    end
    return total
end

function f_exact(x::Real)
    return u_exact(x) + rhs_integral_direct(x)
end

function chebpts_lobatto(N::Int)
    n = N - 1
    n == 0 && return [0.0]
    return [-cospi(j / n) for j in 0:n]
end

function chebcoeffs_from_values(vals_increasing::AbstractVector{<:Real})
    N = length(vals_increasing)
    n = N - 1
    vals_cos_order = reverse(collect(vals_increasing))
    n == 0 && return copy(vals_cos_order)

    c = FFTW.r2r(vals_cos_order, FFTW.REDFT00) ./ n
    c[1] *= 0.5
    c[end] *= 0.5
    return c
end

function cheb_eval_many(c::AbstractVector{<:Real}, x::AbstractVector{<:Real})
    M = length(x)
    N = length(c)
    N == 1 && return fill(c[1], M)

    b1 = zeros(Float64, M)
    b2 = zeros(Float64, M)
    b0 = zeros(Float64, M)

    @inbounds for k in N:-1:2
        ck = c[k]
        for i in 1:M
            b0[i] = 2.0 * x[i] * b1[i] - b2[i] + ck
        end
        b2, b1, b0 = b1, b0, b2
    end

    vals = zeros(Float64, M)
    @inbounds for i in 1:M
        vals[i] = x[i] * b1[i] - b2[i] + c[1]
    end
    return vals
end

function make_test_grid(; Mglobal=8001, Mlocal=5001)
    xg = collect(range(-1.0, 1.0; length=Mglobal))
    xl1 = collect(range(max(-1.0, x1 - 0.04), min(1.0, x1 + 0.04); length=Mlocal))
    xl2 = collect(range(max(-1.0, x2 - 0.05), min(1.0, x2 + 0.05); length=Mlocal))
    return sort(unique(vcat(xg, xl1, xl2, [x1, x2])))
end

const XTEST = make_test_grid()
const UTEST_EXACT = u_exact.(XTEST)

# ============================================================
# VoltMat_Cheb-only build and solve helpers.
# ============================================================

function build_VK(K, N, tol)
    return VoltMat_Cheb.voltmat_cheb_square(K, N - 1; tol=tol)
end

function build_rhs(N)
    xnodes = chebpts_lobatto(N)
    fvals = f_exact.(xnodes)
    return chebcoeffs_from_values(fvals)
end

function build_system(VK, N)
    return spdiagm(0 => ones(N)) + VK
end

function solve_system(A, rhs)
    return A \ rhs
end

function matrix_nnz(A)
    return isa(A, SparseMatrixCSC) ? nnz(A) : count(!iszero, A)
end

function csv_escape(x)
    s = string(x)
    if occursin(",", s) || occursin("\"", s) || occursin("\n", s) || occursin("\r", s)
        return "\"" * replace(s, "\"" => "\"\"") * "\""
    end
    return s
end

const CSV_COLUMNS = [
    :N,
    :tol,
    :equation_sign,
    :VK_build_time,
    :RHS_build_time,
    :system_build_time,
    :solve_time,
    :end_to_end_time,
    :VK_type,
    :VK_rows,
    :VK_cols,
    :VK_nnz,
    :VK_norm,
    :residual_norm,
    :relative_residual,
    :max_abs_error,
    :relative_error,
    :max_abs_uh,
    :max_abs_uexact,
    :status,
    :notes,
]

function init_csv(path::AbstractString)
    open(path, "w") do io
        println(io, join(string.(CSV_COLUMNS), ","))
    end
end

function append_csv(path::AbstractString, row)
    open(path, "a") do io
        println(io, join((csv_escape(getproperty(row, col)) for col in CSV_COLUMNS), ","))
    end
end

function failure_row(N::Int, err)
    return (
        N=N,
        tol=tol,
        equation_sign=EQUATION_SIGN,
        VK_build_time=NaN,
        RHS_build_time=NaN,
        system_build_time=NaN,
        solve_time=NaN,
        end_to_end_time=NaN,
        VK_type="",
        VK_rows=missing,
        VK_cols=missing,
        VK_nnz=missing,
        VK_norm=NaN,
        residual_norm=NaN,
        relative_residual=NaN,
        max_abs_error=NaN,
        relative_error=NaN,
        max_abs_uh=NaN,
        max_abs_uexact=maximum(abs, UTEST_EXACT),
        status="failed",
        notes=sprint(showerror, err),
    )
end

function solve_one_N(N::Int)
    local VK, rhs, Aop, uh_coeff
    local t_vk, t_rhs, t_system, t_solve

    GC.gc()
    t0 = time_ns()

    t_vk = @elapsed begin
        VK = build_VK(K, N, tol)
    end

    t_rhs = @elapsed begin
        rhs = build_rhs(N)
    end

    t_system = @elapsed begin
        Aop = build_system(VK, N)
    end

    t_solve = @elapsed begin
        uh_coeff = solve_system(Aop, rhs)
    end

    end_to_end_time = (time_ns() - t0) / 1e9

    residual = Aop * uh_coeff - rhs
    residual_norm = norm(residual)
    rhs_norm = norm(rhs)
    relative_residual = residual_norm / max(rhs_norm, eps(Float64))

    uh = cheb_eval_many(uh_coeff, XTEST)
    uexact = UTEST_EXACT
    err = uh .- uexact
    max_abs_error = maximum(abs, err)
    relative_error = norm(err) / max(norm(uexact), eps(Float64))

    return (
        N=N,
        tol=tol,
        equation_sign=EQUATION_SIGN,
        VK_build_time=t_vk,
        RHS_build_time=t_rhs,
        system_build_time=t_system,
        solve_time=t_solve,
        end_to_end_time=end_to_end_time,
        VK_type=string(typeof(VK)),
        VK_rows=size(VK, 1),
        VK_cols=size(VK, 2),
        VK_nnz=matrix_nnz(VK),
        VK_norm=norm(VK),
        residual_norm=residual_norm,
        relative_residual=relative_residual,
        max_abs_error=max_abs_error,
        relative_error=relative_error,
        max_abs_uh=maximum(abs, uh),
        max_abs_uexact=maximum(abs, uexact),
        status="success",
        notes="",
    )
end

println("Benchmark ex1 VoltMat_Cheb only")
println("Ns = ", Ns)
println("tol = ", tol)
println("Original ex1 equation: u + V_K u = f")
println("equation_sign = ", EQUATION_SIGN)
println("method = ", @which VoltMat_Cheb.voltmat_cheb_square(K, first(Ns) - 1; tol=tol))
println("GutlebOlver included = ", isdefined(Main, :GutlebOlver))
println("CSV = ", OUTPUT_CSV)
println()

init_csv(OUTPUT_CSV)

@printf("%8s %10s %10s %10s %10s %10s %10s %12s %12s %10s\n",
    "N", "tVK", "tRHS", "tSys", "tSolve", "tTotal", "nnz(VK)", "res", "rel_err", "status")

for Nval in Ns
    row = try
        solve_one_N(Nval)
    catch err
        failure_row(Nval, err)
    end

    append_csv(OUTPUT_CSV, row)

    @printf("%8d %10.4f %10.4f %10.4f %10.4f %10.4f %10s %12.3e %12.3e %10s\n",
        row.N,
        row.VK_build_time,
        row.RHS_build_time,
        row.system_build_time,
        row.solve_time,
        row.end_to_end_time,
        string(row.VK_nnz),
        row.residual_norm,
        row.relative_error,
        row.status)
end

println()
println("Wrote ", OUTPUT_CSV)
