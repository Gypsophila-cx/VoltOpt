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

# You can increase MAX_N if needed.
const MAX_N = 2000

# Avoid too small N, because round(1.01N) may equal N.
const NS = collect(2:10:1500)
const MS = [max(N + 1, round(Int, 1.01 * N)) for N in NS]
const MAX_SOLVE_N = maximum(MS)

# RHS projection degree must cover the largest solve size.
const RHS_COEFF_DEGREE = MAX_SOLVE_N
const RHS_SAMPLE_DEGREE = max(4096, 4 * MAX_SOLVE_N)

K_rational(x, y) = 1.0 / ((x + 1.0)^2 + (y - 1.0)^2)

g_rhs(x) = 1.0 + sin(pi * x)/(10^3*x^2+1)

phi_beta_float(x::Real) =
    2.0 * ((1.0 + Float64(x)) / 2.0)^BETA - 1.0

invphi_beta_float(z::Real) =
    2.0 * ((1.0 + Float64(z)) / 2.0)^(1.0 / BETA) - 1.0

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

function evaluation_grid()
    x_uniform = collect(range(-1.0, 1.0; length = 5001))
    z_uniform = collect(range(-1.0, 1.0; length = 5001))
    x_jfp = invphi_beta_float.(z_uniform)
    return sort(unique(vcat(x_uniform, x_jfp)))
end

function dense_matrix(V)
    return V isa VoltOpt.PackedLowerHessenberg ? VoltOpt.packed_to_dense(V) : Matrix(V)
end

# ============================================================
# Float64 Chebyshev-Lobatto projection for g in JFP variable
# ============================================================

function cheb_project_lobatto_float(f, coeff_degree::Int, sample_degree::Int)
    coeff_degree <= sample_degree ||
        throw(ArgumentError("coeff_degree must not exceed sample_degree."))
    sample_degree >= 1 ||
        throw(ArgumentError("sample_degree must be positive."))

    M = sample_degree
    vals = Vector{Float64}(undef, M + 1)

    @inbounds for j in 0:M
        z = cos(pi * j / M)
        vals[j + 1] = f(z)
    end

    c = zeros(Float64, coeff_degree + 1)

    @inbounds for k in 0:coeff_degree
        s = 0.0
        for j in 0:M
            wj = (j == 0 || j == M) ? 0.5 : 1.0
            s += wj * vals[j + 1] * cos(pi * k * j / M)
        end
        c[k + 1] = 2.0 * s / M
    end
    c[1] /= 2.0

    tail_count = min(20, coeff_degree + 1)
    tail_inf = maximum(abs, @view c[(end - tail_count + 1):end])

    info = (
        sample_degree = M,
        coeff_degree = coeff_degree,
        tail_count = tail_count,
        tail_inf = tail_inf,
    )

    return c, info
end

function compute_rhs_coefficients()
    f_z = z -> begin
        x = invphi_beta_float(z)
        g_rhs(x)
    end

    t = @elapsed begin
        coeffs, info = cheb_project_lobatto_float(
            f_z,
            RHS_COEFF_DEGREE,
            RHS_SAMPLE_DEGREE,
        )
    end

    return coeffs, merge(info, (time = t,))
end

# ============================================================
# Kernel coefficients and solver
# ============================================================

function build_kernel_coefficients()
    stat = @timed begin
        values = WeakCheb.sample_duffy_kernel(K_rational, KERNEL_I, KERNEL_J, BETA)
        WeakCheb.cheb_coeffs_lobatto_2d(values)
    end

    C = stat.value
    tail = WeakCheb.coefficient_tail_info(C)
    return C, stat.time, tail
end

function solve_for_N(N::Int, C::AbstractMatrix, rhs_coeffs::AbstractVector)
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

    residual_inf = norm(A * c - rhsN, Inf)

    return (
        N = N,
        coeffs = c,
        info = info,
        residual_inf = residual_inf,
        matrix_time = matrix_stat.time,
        solve_time = solve_stat.time,
    )
end

# ============================================================
# Cauchy comparison
# ============================================================

function write_cauchy_csv(path::AbstractString, rows::Vector{<:NamedTuple})
    open(path, "w") do io
        println(io, "N,M,coeff_cauchy_inf,coeff_cauchy_rel,value_cauchy_inf,residual_N,residual_M,matrix_time_N,solve_time_N,matrix_time_M,solve_time_M")
        for r in rows
            @printf(io, "%d,%d,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e\n",
                r.N,
                r.M,
                r.coeff_cauchy_inf,
                r.coeff_cauchy_rel,
                r.value_cauchy_inf,
                r.residual_N,
                r.residual_M,
                r.matrix_time_N,
                r.solve_time_N,
                r.matrix_time_M,
                r.solve_time_M,
            )
        end
    end
end

function main()
    println("============================================================")
    println("Cauchy test for weakly singular Volterra equation")
    @printf("mu       = %.16g\n", MU)
    @printf("beta     = %.16g\n", BETA)
    @printf("lambda   = %.16g\n", LAMBDA)
    println("kernel factor = (x-y)^(-mu)")
    println("K(x,y) = 1 / ((x+1)^2 + (y-1)^2)")
    println("g(x)   = 1 + sin(pi*x)")
    println("equation:")
    println("u(x) + lambda * ∫_{-1}^x (x-y)^(-mu)K(x,y)u(y)dy = g(x)")
    println("Cauchy error:")
    println("compare N with M = max(N+1, round(1.01*N))")
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
    @printf("RHS coefficient degree = %d, sample degree = %d\n",
        rhs_info.coeff_degree,
        rhs_info.sample_degree,
    )
    @printf("RHS tail last %d modes = %.6e\n",
        rhs_info.tail_count,
        rhs_info.tail_inf,
    )
    @printf("RHS projection time = %.6f s\n", rhs_info.time)
    println("------------------------------------------------------------")

    xgrid = evaluation_grid()

    solutions = Dict{Int,Any}()

    function get_solution(N::Int)
        if !haskey(solutions, N)
            solutions[N] = solve_for_N(N, C, rhs_coeffs)
        end
        return solutions[N]
    end

    rows = NamedTuple[]

    for (N, M) in zip(NS, MS)
        solN = get_solution(N)
        solM = get_solution(M)

        cN = solN.coeffs
        cM_trunc = solM.coeffs[1:(N + 1)]

        coeff_cauchy_inf = norm(cN .- cM_trunc, Inf)
        coeff_cauchy_rel = coeff_cauchy_inf / max(norm(cM_trunc, Inf), floatmin(Float64))

        uN = [eval_jfp(cN, x) for x in xgrid]
        uM = [eval_jfp(solM.coeffs, x) for x in xgrid]
        value_cauchy_inf = maximum(abs.(uN .- uM))

        push!(rows, (
            N = N,
            M = M,
            coeff_cauchy_inf = coeff_cauchy_inf,
            coeff_cauchy_rel = coeff_cauchy_rel,
            value_cauchy_inf = value_cauchy_inf,
            residual_N = solN.residual_inf,
            residual_M = solM.residual_inf,
            matrix_time_N = solN.matrix_time,
            solve_time_N = solN.solve_time,
            matrix_time_M = solM.matrix_time,
            solve_time_M = solM.solve_time,
        ))

        @printf("N = %5d, M = %5d  coeff_cauchy = %.6e  value_cauchy = %.6e  residual_N = %.6e\n",
            N,
            M,
            coeff_cauchy_inf,
            value_cauchy_inf,
            solN.residual_inf,
        )
    end

    println("------------------------------------------------------------")

    # Use the largest M solution as a visual reference.
    Nref = maximum(MS)
    solref = get_solution(Nref)

    solution_pdf = joinpath(
        FIGURES_DIR,
        "ex7_weak_singular_rational_kernel_simple_rhs_solution_ref.pdf",
    )
    cauchy_pdf = joinpath(
        FIGURES_DIR,
        "ex7_weak_singular_rational_kernel_simple_rhs_cauchy_error.pdf",
    )
    value_cauchy_pdf = joinpath(
        FIGURES_DIR,
        "ex7_weak_singular_rational_kernel_simple_rhs_value_cauchy_error.pdf",
    )
    csv_path = joinpath(
        DATA_DIR,
        "ex7_weak_singular_rational_kernel_simple_rhs_cauchy_error.csv",
    )

    xplot = collect(range(-1.0, 1.0; length = 2000))
    uref = [eval_jfp(solref.coeffs, x) for x in xplot]
    gplot = g_rhs.(xplot)

    p_solution = plot(
        xplot,
        uref;
        label = nothing,
        framestyle = :box,
        xlabel = "x",
        ylabel = "u(x)",
        linewidth = 2.0,
        legend = :best,
        grid = false,
        size = (760, 460),
        color = :black,
    )
    solution_png = savefig_pdf_png(p_solution, solution_pdf)

    Ns_vec = [r.N for r in rows]
    coeff_errs = [r.coeff_cauchy_inf for r in rows]
    value_errs = [r.value_cauchy_inf for r in rows]

    p_cauchy = plot(
        Ns_vec,
        coeff_errs;
        label = nothing,
        xlabel = "N",
        ylabel = "Cauchy error",
        yscale = :log10,
        framestyle = :box,
        linewidth = 2.0,
        marker = false,
        markersize = 3.0,
        legend = false,
        grid = false,
        size = (760, 460),
        color = :black,
    )
    cauchy_png = savefig_pdf_png(p_cauchy, cauchy_pdf)

    p_value_cauchy = plot(
        Ns_vec,
        value_errs;
        label = nothing,
        xlabel = "N",
        ylabel = "value Cauchy error",
        yscale = :log10,
        framestyle = :box,
        linewidth = 2.0,
        marker = false,
        markersize = 3.0,
        legend = false,
        grid = false,
        size = (760, 460),
        color = :black,
    )
    value_cauchy_png = savefig_pdf_png(p_value_cauchy, value_cauchy_pdf)

    write_cauchy_csv(csv_path, rows)

    println("Saved:")
    println(solution_pdf)
    println(solution_png)
    println(cauchy_pdf)
    println(cauchy_png)
    println(value_cauchy_pdf)
    println(value_cauchy_png)
    println(csv_path)
    println("============================================================")
end

main()