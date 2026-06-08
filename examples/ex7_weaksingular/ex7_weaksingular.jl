using LinearAlgebra
using Printf
using Plots

const THIS_DIR = @__DIR__

function find_project_root(start_dir::AbstractString = @__DIR__)
    dir = abspath(start_dir)
    while true
        if isfile(joinpath(dir, "Project.toml")) && isdir(joinpath(dir, "src"))
            return dir
        end
        parent = dirname(dir)
        parent == dir && error("Could not find the VoltOpt project root from $(start_dir).")
        dir = parent
    end
end

const PROJECT_ROOT = find_project_root()
const FIGURES_DIR = joinpath(PROJECT_ROOT, "examples", "figures")
const DATA_DIR = joinpath(PROJECT_ROOT, "examples", "data")
mkpath(FIGURES_DIR)
mkpath(DATA_DIR)

function savefig_pdf_png(fig, pdf_path::AbstractString)
    png_path = replace(pdf_path, r"\.pdf$" => ".png")
    savefig(fig, pdf_path)
    savefig(fig, png_path)
    return png_path
end

include(joinpath(PROJECT_ROOT, "src", "VoltOpt.jl"))
using .VoltOpt

const WeakCheb = VoltOpt.Weak_VoltMat_Cheb

# ============================================================
## Problem data
# ============================================================

const MU = 0.5
const BETA = 1.0 - MU
const SIGMA = 0.0
const LAMBDA = 0.5
const KERNEL_I = 60
const KERNEL_J = 40
const MAX_N = 100
const NS = collect(2:2:MAX_N)

const RHS_PRECISION = 256
const RHS_SAMPLE_DEGREE = 512
const RHS_CHECK_SAMPLE_DEGREE = 256
const RHS_COEFF_DEGREE = 180
const TS_H_STR = "0.125"
const TS_ATOL_STR = "1e-62"
const TS_MAX_TERMS = 180

K_rational(x, y) = 1.0 / ((x + 1.0)^2 + (y - 1.0)^2)

function u_exact_float(x::Real)
    t = Float64(x) + 1.0
    return 1.0 -
           0.5 * LAMBDA * sqrt(t) +
           (LAMBDA^2 * pi / 16.0) * t -
           (LAMBDA / 3.0) * (1.0 + LAMBDA^2 * pi / 16.0) * t^(1.5)
end

function u_exact_big(x::BigFloat)
    lambda = BigFloat(LAMBDA)
    t = x + one(BigFloat)
    return one(BigFloat) -
           lambda / BigFloat(2) * sqrt(t) +
           (lambda^2 * big(pi) / BigFloat(16)) * t -
           lambda / BigFloat(3) *
           (one(BigFloat) + lambda^2 * big(pi) / BigFloat(16)) *
           t * sqrt(t)
end

kernel_big(x::BigFloat, y::BigFloat) =
    inv((x + one(BigFloat))^2 + (y - one(BigFloat))^2)

phi_beta_float(x::Real) =
    2.0 * ((1.0 + Float64(x)) / 2.0)^BETA - 1.0

invphi_beta_float(z::Real) =
    2.0 * ((1.0 + Float64(z)) / 2.0)^(1.0 / BETA) - 1.0

function invphi_beta_big(z::BigFloat)
    return BigFloat(2) * ((one(BigFloat) + z) / BigFloat(2))^(one(BigFloat) / BigFloat(BETA)) -
           one(BigFloat)
end

# ============================================================
# BigFloat tanh-sinh RHS generation
# ============================================================

mutable struct TanhSinhStats
    max_terms::Int
    max_last_term::BigFloat
    calls::Int
end

TanhSinhStats() = TanhSinhStats(0, zero(BigFloat), 0)

function tanh_sinh_q_dq(s::BigFloat)
    half = BigFloat(0.5)
    a = (big(pi) / BigFloat(2)) * sinh(s)
    q = half * (one(BigFloat) + tanh(a))
    sech2 = inv(cosh(a)^2)
    dq = half * (big(pi) / BigFloat(2)) * cosh(s) * sech2
    return q, dq
end

function tanh_sinh_integrate_01_big(
    f;
    h::BigFloat = parse(BigFloat, TS_H_STR),
    atol::BigFloat = parse(BigFloat, TS_ATOL_STR),
    max_terms::Int = TS_MAX_TERMS,
    stats::Union{Nothing,TanhSinhStats} = nothing,
)
    total = zero(BigFloat)
    max_term = zero(BigFloat)
    last_pair = typemax(BigFloat)
    small_count = 0
    terms_used = 0

    for k in 0:max_terms
        pair_sum = zero(BigFloat)

        if k == 0
            q, dq = tanh_sinh_q_dq(zero(BigFloat))
            if zero(BigFloat) < q < one(BigFloat)
                pair_sum += f(q) * dq
            end
        else
            for sgn in (-1, 1)
                q, dq = tanh_sinh_q_dq(BigFloat(sgn * k) * h)
                if zero(BigFloat) < q < one(BigFloat)
                    pair_sum += f(q) * dq
                end
            end
        end

        total += pair_sum
        last_pair = abs(pair_sum)
        max_term = max(max_term, last_pair)
        terms_used = k

        scale = max(one(BigFloat), abs(total), max_term)
        if k >= 8 && last_pair <= atol * scale
            small_count += 1
            small_count >= 8 && break
        else
            small_count = 0
        end
    end

    if stats !== nothing
        stats.calls += 1
        stats.max_terms = max(stats.max_terms, terms_used)
        stats.max_last_term = max(stats.max_last_term, last_pair)
    end

    return h * total
end

function volterra_exact_big(x::BigFloat, stats::TanhSinhStats)
    x <= -one(BigFloat) && return zero(BigFloat)

    xp1 = x + one(BigFloat)
    xp1 == 0 && return zero(BigFloat)

    # Public convention: the weak kernel factor is (x-y)^(-MU).
    integral = tanh_sinh_integrate_01_big(stats = stats) do xi
        y = -one(BigFloat) + xp1 * xi
        return inv(sqrt(one(BigFloat) - xi)) * kernel_big(x, y) * u_exact_big(y)
    end

    return sqrt(xp1) * integral
end

function rhs_big_at_x(x::BigFloat, stats::TanhSinhStats)
    return u_exact_big(x) + BigFloat(LAMBDA) * volterra_exact_big(x, stats)
end

function cheb_project_lobatto_big(f, coeff_degree::Int, sample_degree::Int; precision::Int)
    coeff_degree <= sample_degree ||
        throw(ArgumentError("coeff_degree must not exceed sample_degree."))
    sample_degree >= 1 ||
        throw(ArgumentError("sample_degree must be positive."))

    stats = TanhSinhStats()

    coeffs, info = setprecision(BigFloat, precision) do
        M = sample_degree
        vals = Vector{BigFloat}(undef, M + 1)

        @inbounds for j in 0:M
            z = cos(big(pi) * BigFloat(j) / BigFloat(M))
            vals[j + 1] = f(z, stats)
        end

        c = zeros(BigFloat, coeff_degree + 1)

        @inbounds for k in 0:coeff_degree
            s = zero(BigFloat)
            for j in 0:M
                wj = (j == 0 || j == M) ? BigFloat(0.5) : one(BigFloat)
                s += wj * vals[j + 1] * cos(big(pi) * BigFloat(k * j) / BigFloat(M))
            end
            c[k + 1] = BigFloat(2) * s / BigFloat(M)
        end
        c[1] /= BigFloat(2)

        tail_count = min(20, coeff_degree + 1)
        tail_inf = maximum(abs, @view c[(end - tail_count + 1):end])
        info = (
            sample_degree = M,
            coeff_degree = coeff_degree,
            tail_count = tail_count,
            tail_inf = Float64(tail_inf),
            ts_calls = stats.calls,
            ts_max_terms = stats.max_terms,
            ts_max_last_term = Float64(stats.max_last_term),
        )

        Float64.(c), info
    end

    return coeffs, info
end

function compute_rhs_coefficients()
    f_rhs = function (z::BigFloat, stats::TanhSinhStats)
        x = invphi_beta_big(z)
        return rhs_big_at_x(x, stats)
    end

    t_hi = @elapsed begin
        coeffs_hi, info_hi = cheb_project_lobatto_big(
            f_rhs,
            RHS_COEFF_DEGREE,
            RHS_SAMPLE_DEGREE;
            precision = RHS_PRECISION,
        )
    end

    t_check = @elapsed begin
        coeffs_check, info_check = cheb_project_lobatto_big(
            f_rhs,
            RHS_COEFF_DEGREE,
            RHS_CHECK_SAMPLE_DEGREE;
            precision = RHS_PRECISION,
        )
    end

    diff_inf = maximum(abs.(coeffs_hi .- coeffs_check))
    diff_rel = diff_inf / max(norm(coeffs_hi, Inf), floatmin(Float64))

    info = (
        high = info_hi,
        check = info_check,
        coeff_diff_inf = diff_inf,
        coeff_diff_rel = diff_rel,
        high_time = t_hi,
        check_time = t_check,
    )

    return coeffs_hi, info
end

# ============================================================
# Chebyshev/JFP utilities
# ============================================================

function cheb_eval(c::AbstractVector{<:Real}, z::Real)
    n = length(c) - 1
    n < 0 && throw(ArgumentError("coefficient vector must be nonempty."))
    n == 0 && return Float64(c[1])

    zz = Float64(z)
    bkp1 = 0.0
    bkp2 = 0.0

    @inbounds for k in n:-1:1
        bk = 2.0 * zz * bkp1 - bkp2 + Float64(c[k + 1])
        bkp2 = bkp1
        bkp1 = bk
    end

    return zz * bkp1 - bkp2 + Float64(c[1])
end

eval_jfp(c::AbstractVector{<:Real}, x::Real) = cheb_eval(c, phi_beta_float(x))

function exact_solution_coefficients(maxN::Int)
    coeffs = zeros(Float64, maxN + 1)
    a1 = -LAMBDA / (2.0 * sqrt(2.0))
    a2 = LAMBDA^2 * pi / 32.0
    a3 = -LAMBDA * (1.0 + LAMBDA^2 * pi / 16.0) / (6.0 * sqrt(2.0))

    # u(z) = 1 + a1*(z+1) + a2*(z+1)^2 + a3*(z+1)^3.
    coeffs[1] = 1.0 + a1 + 1.5 * a2 + 2.5 * a3
    maxN >= 1 && (coeffs[2] = a1 + 2.0 * a2 + 3.75 * a3)
    maxN >= 2 && (coeffs[3] = 0.5 * a2 + 1.5 * a3)
    maxN >= 3 && (coeffs[4] = 0.25 * a3)
    return coeffs
end

function evaluation_grid()
    x_uniform = collect(range(-1.0, 1.0; length = 5001))
    z_uniform = collect(range(-1.0, 1.0; length = 5001))
    x_jfp = invphi_beta_float.(z_uniform)
    return sort(unique(vcat(x_uniform, x_jfp)))
end

function dense_matrix(V)
    return V isa VoltOpt.PackedLowerHessenberg ? VoltOpt.packed_to_dense(V) : Matrix(V)
end

function write_error_csv(path::AbstractString, rows::Vector{<:NamedTuple})
    open(path, "w") do io
        println(io, "N,err_inf,coeff_err,residual_inf,rhs_time,matrix_time,solve_time")
        for r in rows
            @printf(io, "%d,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e\n",
                r.N,
                r.err_inf,
                r.coeff_err,
                r.residual_inf,
                r.rhs_time,
                r.matrix_time,
                r.solve_time,
            )
        end
    end
end

function build_kernel_coefficients()
    stat = @timed begin
        values = WeakCheb.sample_duffy_kernel(K_rational, KERNEL_I, KERNEL_J, BETA)
        WeakCheb.cheb_coeffs_lobatto_2d(values)
    end

    C = stat.value
    tail = WeakCheb.coefficient_tail_info(C)
    return C, stat.time, tail
end

function solve_for_N(N::Int, C::AbstractMatrix, rhs_coeffs::AbstractVector, exact_coeffs::AbstractVector, xgrid)
    rhsN = rhs_coeffs[1:(N + 1)]

    matrix_stat = @timed begin
        V, info = VoltOpt.weak_voltmat_cheb(
            C;
            C_is_coeffs = true,
            mu = MU,
            sigma = SIGMA,
            beta = BETA,
            N = N,
            diagnostics = false,
            output_format = :dense,
            return_info = true,
        )
        dense_matrix(V), info
    end

    Vdense, info = matrix_stat.value
    A = Matrix{Float64}(LinearAlgebra.I, N + 1, N + 1)
    A .+= LAMBDA .* Vdense

    solve_stat = @timed begin
        A \ rhsN
    end
    c = solve_stat.value

    u_num = [eval_jfp(c, x) for x in xgrid]
    u_ex = u_exact_float.(xgrid)
    err_inf = maximum(abs.(u_num .- u_ex))
    coeff_err = norm(c .- exact_coeffs[1:(N + 1)], Inf)
    residual_inf = norm(A * c - rhsN, Inf)

    return (
        N = N,
        coeffs = c,
        info = info,
        err_inf = err_inf,
        coeff_err = coeff_err,
        residual_inf = residual_inf,
        matrix_time = matrix_stat.time,
        solve_time = solve_stat.time,
    )
end

function main()
    println("============================================================")
    println("Example 7: weakly singular Volterra equation")
    @printf("mu     = %.16g\n", MU)
    @printf("beta   = %.16g\n", BETA)
    println("kernel factor = (x-y)^(-mu)")
    @printf("lambda = %.16g\n", LAMBDA)
    println("exact solution:")
    println("u_*(x) = 1 - lambda/2*(x+1)^(1/2) + lambda^2*pi/16*(x+1)")
    println("         - lambda/3*(1 + lambda^2*pi/16)*(x+1)^(3/2)")
    println("============================================================")

    C, coeff_time, kernel_tail = build_kernel_coefficients()
    @printf("Duffy-Chebyshev kernel coefficient matrix size = %d x %d\n", size(C, 1), size(C, 2))
    @printf("kernel coefficient build time = %.6f s\n", coeff_time)
    @printf("kernel tail row_rel = %.3e, col_rel = %.3e, max_coeff = %.3e\n",
        kernel_tail.row_rel,
        kernel_tail.col_rel,
        kernel_tail.max_coeff,
    )
    println("------------------------------------------------------------")

    rhs_coeffs, rhs_info = compute_rhs_coefficients()
    @printf("RHS BigFloat precision = %d bits\n", RHS_PRECISION)
    @printf("rhs_stab high sample degree = %d, check sample degree = %d\n",
        rhs_info.high.sample_degree,
        rhs_info.check.sample_degree,
    )
    @printf("rhs_stab coeff diff inf = %.6e, rel = %.6e\n",
        rhs_info.coeff_diff_inf,
        rhs_info.coeff_diff_rel,
    )
    @printf("rhs_stab tail last %d modes = %.6e\n",
        rhs_info.high.tail_count,
        rhs_info.high.tail_inf,
    )
    @printf("rhs_stab tanh-sinh calls = %d, max k = %d, max last pair = %.6e\n",
        rhs_info.high.ts_calls,
        rhs_info.high.ts_max_terms,
        rhs_info.high.ts_max_last_term,
    )
    @printf("rhs BigFloat projection time = %.6f s (check %.6f s)\n",
        rhs_info.high_time,
        rhs_info.check_time,
    )
    println("------------------------------------------------------------")

    xgrid = evaluation_grid()
    exact_coeffs = exact_solution_coefficients(MAX_N)
    rows = NamedTuple[]
    solution_N100 = nothing

    for N in NS
        result = solve_for_N(N, C, rhs_coeffs, exact_coeffs, xgrid)
        push!(rows, (
            N = N,
            err_inf = result.err_inf,
            coeff_err = result.coeff_err,
            residual_inf = result.residual_inf,
            rhs_time = N == first(NS) ? rhs_info.high_time : 0.0,
            matrix_time = result.matrix_time,
            solve_time = result.solve_time,
        ))

        @printf("N = %3d  err_inf = %.6e  coeff_err = %.6e  residual_inf = %.6e\n",
            N,
            result.err_inf,
            result.coeff_err,
            result.residual_inf,
        )

        if N == MAX_N
            solution_N100 = result
        end
    end

    final_err = rows[end].err_inf
    @printf("Final N = %d error_inf = %.6e\n", MAX_N, final_err)
    println("------------------------------------------------------------")

    solution_pdf = joinpath(FIGURES_DIR, "ex7_weak_singular_rational_kernel_solution.pdf")
    error_pdf = joinpath(FIGURES_DIR, "ex7_weak_singular_rational_kernel_error_vs_N.pdf")
    csv_path = joinpath(DATA_DIR, "ex7_weak_singular_rational_kernel_error_vs_N.csv")

    xplot = collect(range(-1.0, 1.0; length = 1600))
    u_num = [eval_jfp(solution_N100.coeffs, x) for x in xplot]
    u_ex = u_exact_float.(xplot)

    p_solution = plot(
        xplot,
        u_num;
        label = "numerical",
        framestyle=:box,
        xlabel = "x",
        ylabel = "u(x)",
        linewidth = 2.0,
        legend = :best,
        grid = false,
        size = (760, 460),
    )
    plot!(p_solution, xplot, u_ex; label = "exact", linewidth = 2.0, linestyle = :dash)
    solution_png = savefig_pdf_png(p_solution, solution_pdf)

    Ns_vec = [r.N for r in rows]
    errs = [r.err_inf for r in rows]
    ytick_values = [1e-15, 1e-10, 1e-5, 1e0]
    ytick_labels = ["10⁻¹⁵", "10⁻¹⁰", "10⁻⁵", "10⁰"]
    p_error = plot(
        Ns_vec,
        errs;
        label = nothing,
        xlabel = "N",
        ylabel = "error",
        yscale = :log10,
        framestyle=:box,
        #ylims = (1e-15, 1e1),
        #yticks = (ytick_values, ytick_labels),
        linewidth = 2.0,
        marker = false,
        # markersize = 3.5,
        legend = :topright,
        grid = false,
        size = (760, 460),
        color = :black,
    )
    error_png = savefig_pdf_png(p_error, error_pdf)

    write_error_csv(csv_path, rows)

    println("Saved:")
    println(solution_pdf)
    println(solution_png)
    println(error_pdf)
    println(error_png)
    println(csv_path)
    println("============================================================")
end

main()
