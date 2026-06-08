using LinearAlgebra
using SparseArrays
using FFTW
using Printf

const THIS_DIR = @__DIR__

function find_project_root(start_dir::AbstractString=@__DIR__)
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

const VOLTERRA_ROOT = find_project_root()
const SRC_DIR = joinpath(VOLTERRA_ROOT, "src")
const FIGURES_DIR = joinpath(VOLTERRA_ROOT, "examples", "figures")
const DATA_DIR = joinpath(VOLTERRA_ROOT, "examples", "data")

# Optional fast solver.
# If FastAlmostBandedMatrices.jl is unavailable, the script falls back to sparse LU.
const HAS_FAST_AB = Ref(false)
try
    @eval using FastAlmostBandedMatrices
    @eval using BandedMatrices
    HAS_FAST_AB[] = true
catch err
    @warn "FastAlmostBandedMatrices/BandedMatrices not available; fallback to sparse LU." exception=err
end

# Load the current Chebyshev Volterra matrix constructor.
include(joinpath(SRC_DIR, "VoltMat", "voltmat_cheb.jl"))
using .VoltMat_Cheb

# ============================================================
# 1. Chebyshev tools
# ============================================================

function chebpts_lobatto(N::Int)
    n = N - 1
    return [-cospi(j / n) for j in 0:n]
end

function chebcoeffs_from_values(vals_increasing::AbstractVector{<:Real})
    N = length(vals_increasing)
    n = N - 1

    vals_cos_order = reverse(collect(vals_increasing))

    if n == 0
        return copy(vals_cos_order)
    end

    C = FFTW.r2r(vals_cos_order, FFTW.REDFT00)

    c = C ./ n
    c[1] *= 0.5
    c[end] *= 0.5

    return c
end

function cheb_eval_many(c::AbstractVector{<:Real}, x::AbstractVector{<:Real})
    M = length(x)
    N = length(c)

    if N == 1
        return fill(c[1], M)
    end

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

function tail_ratio(c; m=20)
    lo = max(1, length(c) - m + 1)
    return maximum(abs.(c[lo:end])) / maximum(abs.(c))
end

function integer_power_ticks(values; step::Int=1)
    positives = Float64[]

    for v in values
        av = abs(float(v))
        if isfinite(av) && av > 0.0
            push!(positives, av)
        end
    end

    isempty(positives) && return :auto

    vmin = minimum(positives)
    vmax = maximum(positives)

    kmin = floor(Int, log10(vmin))
    kmax = ceil(Int, log10(vmax))

    ks = collect(kmin:step:kmax)
    vals = 10.0 .^ ks
    labels = ["10^{$k}" for k in ks]

    return vals, labels
end

function fixed_power_ticks(; kmin=-15, kmax=0, step=5)
    ks = collect(kmin:step:kmax)
    vals = 10.0 .^ ks
    labels = [k == 0 ? "1" : "10^{$k}" for k in ks]
    return vals, labels
end

function save_coefficients_csv(filename::AbstractString, c::AbstractVector)
    open(filename, "w") do io
        println(io, "n,c_n,abs_c_n")
        for (j, cj) in enumerate(c)
            @printf(io, "%d,%.17e,%.17e\n", j - 1, cj, abs(cj))
        end
    end
end

function save_results_csv(filename::AbstractString, results)
    open(filename, "w") do io
        println(io, "N,solver,top_rows,lower_bw,upper_bw,nnzVK,densityVK,err_inf,rel_inf,coeff_err,coeff_rel,res_inf,t_coeff,t_rhs,t_VK,t_Aop,t_convert,t_factor,t_solve,t_error,total_time")
        for r in results
            @printf(io,
                "%d,%s,%d,%d,%d,%d,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e\n",
                r.N, String(r.solver), r.top_rows, r.lower_bw, r.upper_bw,
                r.nnzVK, r.densityVK,
                r.err_inf, r.rel_inf, r.coeff_err, r.coeff_rel, r.res_inf,
                r.t_coeff, r.t_rhs, r.t_VK, r.t_Aop,
                r.t_convert, r.t_factor, r.t_solve, r.t_error, r.total_time
            )
        end
    end
end

function save_absolute_error_csv(filename::AbstractString, results)
    open(filename, "w") do io
        println(io, "N,abs_error_inf,residual_inf,total_time")
        for r in results
            @printf(io, "%d,%.17e,%.17e,%.17e\n",
                r.N,
                r.err_inf,
                r.res_inf,
                r.total_time,
            )
        end
    end
    return filename
end

function evenly_spaced_integer_nodes(lo::Int, hi::Int, count::Int)
    lo <= hi || error("lo must not exceed hi.")
    count >= 2 || error("count must be at least 2.")

    nodes = round.(Int, range(lo, hi; length=count))
    length(unique(nodes)) == count || error("Rounded node list contains duplicates.")

    return nodes
end

# ============================================================
# 2. Problem definition: smoothed step exact solution
# ============================================================

tol = eps()

x1 = -0.35
x2 =  0.42

A1 =  1.0
A2 = -0.6

δ1 = 1.0e-3
δ2 = 1.5e-3

Sδ(x, x0, δ) = 0.5 * (1.0 + tanh((x - x0) / δ))

function u_exact(x)
    return 1.0 + A1 * Sδ(x, x1, δ1) + A2 * Sδ(x, x2, δ2)
end

# Aging Maxwell-type Volterra kernel:
# tau(x,y) = 3/2 + x - y/2
tau_xy(x, y) = 1.5 + x - 0.5 * y

function K(x, y)
    τ = tau_xy(x, y)
    return exp(-(x - y) / τ) / τ
end

# ============================================================
# 3. Independent high-order RHS generation by split Gauss quadrature
# ============================================================

# Right-hand side:
#
#   f(x) = u_*(x) + ∫_{-1}^x K(x,y) u_*(y) dy.
#
# This version computes the integral directly by high-order 1D
# Gauss--Legendre quadrature, split around the narrow transition layers.
#
# It avoids the previous Heaviside + local correction formula and does not
# use the discrete matrix VK.

const RHS_Q = 160
const RHS_SPLIT_L = 40.0

function gausslegendre_rule(n::Int)
    n <= 0 && error("n must be positive.")
    if n == 1
        return [0.0], [2.0]
    end

    β = [k / sqrt(4.0 * k^2 - 1.0) for k in 1:n-1]
    T = SymTridiagonal(zeros(n), β)
    E = eigen(T)

    x = collect(E.values)
    w = 2.0 .* abs2.(E.vectors[1, :])

    p = sortperm(x)
    return x[p], w[p]
end

const GL_X, GL_W = gausslegendre_rule(RHS_Q)

function gauss_integral_on_interval(f, a::Float64, b::Float64)
    if !(b > a)
        return 0.0
    end

    c = 0.5 * (a + b)
    h = 0.5 * (b - a)

    s = 0.0
    @inbounds for q in eachindex(GL_X)
        y = c + h * GL_X[q]
        s += GL_W[q] * f(y)
    end

    return h * s
end

"""
    rhs_integral_direct(x)

Compute

    ∫_{-1}^x K(x,y) u_*(y) dy

by split high-order Gauss quadrature.

The interval [-1,x] is split at
    x_i - RHS_SPLIT_L*δ_i,
    x_i,
    x_i + RHS_SPLIT_L*δ_i
whenever those points lie inside [-1,x].
"""
function rhs_integral_direct(x::Real)
    xf = Float64(x)

    if xf <= -1.0
        return 0.0
    end

    pts = Float64[]
    sizehint!(pts, 10)

    push!(pts, -1.0)
    push!(pts, xf)

    # Split near the first transition layer.
    a1 = x1 - RHS_SPLIT_L * δ1
    b1 = x1 + RHS_SPLIT_L * δ1

    if -1.0 < a1 < xf
        push!(pts, a1)
    end
    if -1.0 < x1 < xf
        push!(pts, x1)
    end
    if -1.0 < b1 < xf
        push!(pts, b1)
    end

    # Split near the second transition layer.
    a2 = x2 - RHS_SPLIT_L * δ2
    b2 = x2 + RHS_SPLIT_L * δ2

    if -1.0 < a2 < xf
        push!(pts, a2)
    end
    if -1.0 < x2 < xf
        push!(pts, x2)
    end
    if -1.0 < b2 < xf
        push!(pts, b2)
    end

    sort!(pts)

    # Remove near-duplicate split points.
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

    @inbounds for j in 1:length(unique_pts)-1
        a = unique_pts[j]
        b = unique_pts[j+1]

        if b > a
            total += gauss_integral_on_interval(a, b) do y
                K(xf, y) * u_exact(y)
            end
        end
    end

    return total
end

"""
    f_exact(x)

Independent manufactured RHS.

No use of VK * cu_exact.
"""
function f_exact(x::Real)
    return u_exact(x) + rhs_integral_direct(x)
end
# ============================================================
# 4. Test grids and high-order coefficient reference
# ============================================================

function make_test_grid(; Mglobal=8001, Mlocal=5001, radius_factor=80.0)
    δmax = max(δ1, δ2)
    r = radius_factor * δmax

    xg = collect(range(-1.0, 1.0; length=Mglobal))

    xl1 = collect(range(max(-1.0, x1 - r), min(1.0, x1 + r); length=Mlocal))
    xl2 = collect(range(max(-1.0, x2 - r), min(1.0, x2 + r); length=Mlocal))

    x = sort(unique(vcat(xg, xl1, xl2, [x1, x2])))
    return x
end

xtest = make_test_grid()
utest_exact = u_exact.(xtest)

xplot = sort(unique(vcat(
    collect(range(-1.0, 1.0; length=5001)),
    collect(range(max(-1.0, x1 - 0.04), min(1.0, x1 + 0.04); length=3001)),
    collect(range(max(-1.0, x2 - 0.05), min(1.0, x2 + 0.05); length=3001)),
    [x1, x2]
)))
xplot = xplot[(xplot .>= -1.0) .& (xplot .<= 1.0)]

const NREF_COEFF = 80000

@printf("Building high-order exact coefficient reference, Nref = %d...\n", NREF_COEFF)
xnodes_ref = chebpts_lobatto(NREF_COEFF)
cu_exact_ref = chebcoeffs_from_values(u_exact.(xnodes_ref))

# ============================================================
# 5. FastAlmostBandedMatrices conversion utilities
# ============================================================

function estimate_almostbanded_structure(
    A::SparseMatrixCSC;
    dense_frac::Float64 = 0.02,
    droptol::Float64 = 0.0,
)
    n, m = size(A)
    @assert n == m

    I, J, V = findnz(A)

    row_nnz = zeros(Int, n)

    @inbounds for k in eachindex(V)
        if abs(V[k]) > droptol
            row_nnz[I[k]] += 1
        end
    end

    dense_threshold = max(64, ceil(Int, dense_frac * n))
    dense_rows = findall(row_nnz .>= dense_threshold)

    p = isempty(dense_rows) ? 1 : maximum(dense_rows)
    p = max(p, 1)

    lower_bw = 0
    upper_bw = 0

    @inbounds for k in eachindex(V)
        v = V[k]
        if abs(v) > droptol
            i = I[k]
            j = J[k]

            if i > p
                lower_bw = max(lower_bw, i - j)
                upper_bw = max(upper_bw, j - i)
            end
        end
    end

    lower_bw = max(lower_bw, p - 1)

    return (
        top_rows = p,
        lower_bw = lower_bw,
        upper_bw = upper_bw,
        row_nnz = row_nnz,
        dense_threshold = dense_threshold,
    )
end

function sparse_to_almostbanded(
    A::SparseMatrixCSC;
    top_rows::Int,
    lower_bw::Int,
    upper_bw::Int,
    droptol::Float64 = 0.0,
)
    n, m = size(A)
    @assert n == m

    T = eltype(A)

    p = max(top_rows, 1)
    l = max(lower_bw, p - 1)
    u = max(upper_bw, 0)

    bands = BandedMatrix{T}(undef, (n, n), (l, u))
    fill!(bands, zero(T))

    fillpart = zeros(T, p, n)

    dropped = 0
    max_dropped = 0.0

    I, J, V = findnz(A)

    @inbounds for k in eachindex(V)
        i = I[k]
        j = J[k]
        v = V[k]

        if abs(v) <= droptol
            continue
        end

        if i <= p
            fillpart[i, j] = v
        elseif (i - j <= l) && (j - i <= u)
            bands[i, j] = v
        else
            dropped += 1
            max_dropped = max(max_dropped, abs(v))
        end
    end

    if dropped > 0
        error("""
        sparse_to_almostbanded dropped $dropped entries, max dropped = $max_dropped.
        Increase top_rows/lower_bw/upper_bw or lower droptol.
        """)
    end

    return AlmostBandedMatrix(bands, fillpart)
end

# ============================================================
# 6. Solve once for a given N
# ============================================================

function solve_volterra_once(
    N::Int;
    tol = eps(),
    keep_solution = false,
    use_fastalmostbanded = true,
    top_rows = nothing,
    lower_bw = nothing,
    upper_bw = nothing,
    detect_droptol = 0.0,
    convert_droptol = 0.0,
)
    GC.gc()

    local xnodes, uvals_exact, cu_interp_exact
    t_coeff = @elapsed begin
        xnodes = chebpts_lobatto(N)
        uvals_exact = u_exact.(xnodes)
        cu_interp_exact = chebcoeffs_from_values(uvals_exact)
    end

    local fvals, cf
    t_rhs = @elapsed begin
        # Independent high-order RHS.
        # Do NOT use cf = cu_exact + VK * cu_exact.
        fvals = f_exact.(xnodes)
        cf = chebcoeffs_from_values(fvals)
    end

    local VK
    GC.gc()
    t_VK = @elapsed begin
        VK = VoltMat_Cheb.voltmat_cheb_square(K, N - 1; tol = tol)
    end

    local Aop
    t_Aop = @elapsed begin
        Aop = spdiagm(0 => ones(N)) + VK
    end

    local cu_num, solver_info
    t_convert = 0.0
    t_factor = 0.0
    t_solve = 0.0

    if use_fastalmostbanded && HAS_FAST_AB[]
        local Aab, F, tmp_info

        try
            t_convert = @elapsed begin
                structure = estimate_almostbanded_structure(
                    Aop;
                    droptol = detect_droptol,
                )

                p_use = top_rows === nothing ? structure.top_rows : top_rows
                l_use = lower_bw === nothing ? structure.lower_bw : lower_bw
                u_use = upper_bw === nothing ? structure.upper_bw : upper_bw

                p_use = max(Int(p_use), 1)
                l_use = max(Int(l_use), p_use - 1)
                u_use = max(Int(u_use), 0)

                @printf("  FastAlmostBanded: top_rows = %d, lower_bw = %d, upper_bw = %d\n",
                    p_use, l_use, u_use)

                Aab = sparse_to_almostbanded(
                    Aop;
                    top_rows = p_use,
                    lower_bw = l_use,
                    upper_bw = u_use,
                    droptol = convert_droptol,
                )

                tmp_info = (
                    solver = :fast_almost_banded_qr,
                    top_rows = p_use,
                    lower_bw = l_use,
                    upper_bw = u_use,
                )
            end

            GC.gc()
            t_factor = @elapsed begin
                F = qr(Aab)
            end

            t_solve = @elapsed begin
                cu_num = F \ cf
            end

            solver_info = tmp_info

        catch err
            @warn "FastAlmostBandedMatrices solver failed; falling back to sparse lu(Aop)." exception=(err, catch_backtrace())

            local Fsp
            GC.gc()
            t_factor = @elapsed begin
                Fsp = lu(Aop)
            end
            t_solve = @elapsed begin
                cu_num = Fsp \ cf
            end

            solver_info = (
                solver = :sparse_lu_fallback,
                top_rows = -1,
                lower_bw = -1,
                upper_bw = -1,
            )
        end
    else
        local Fsp
        GC.gc()
        t_factor = @elapsed begin
            Fsp = lu(Aop)
        end
        t_solve = @elapsed begin
            cu_num = Fsp \ cf
        end

        solver_info = (
            solver = :sparse_lu,
            top_rows = -1,
            lower_bw = -1,
            upper_bw = -1,
        )
    end

    local unum_test, err_inf, rel_inf, coeff_err, coeff_rel, res_inf
    t_error = @elapsed begin
        unum_test = cheb_eval_many(cu_num, xtest)

        err_inf = maximum(abs.(unum_test .- utest_exact))
        rel_inf = err_inf / maximum(abs.(utest_exact))

        coeff_err = maximum(abs.(cu_num .- cu_exact_ref[1:N]))
        coeff_rel = coeff_err / maximum(abs.(cu_exact_ref[1:N]))

        res_inf = norm(Aop * cu_num - cf, Inf)
    end

    total_time = t_coeff + t_rhs + t_VK + t_Aop + t_convert + t_factor + t_solve + t_error

    @printf("N = %-6d solver = %-24s nnz(VK) = %-10d err_inf = %.3e rel = %.3e coeff_err = %.3e res = %.3e tRHS = %.3f tVK = %.3f tfac = %.3f tsolve = %.3f total = %.3f\n",
        N, String(solver_info.solver), nnz(VK),
        err_inf, rel_inf, coeff_err, res_inf,
        t_rhs, t_VK, t_factor, t_solve, total_time
    )

    base = (
        N = N,
        solver = solver_info.solver,
        top_rows = solver_info.top_rows,
        lower_bw = solver_info.lower_bw,
        upper_bw = solver_info.upper_bw,
        err_inf = err_inf,
        rel_inf = rel_inf,
        coeff_err = coeff_err,
        coeff_rel = coeff_rel,
        res_inf = res_inf,
        nnzVK = nnz(VK),
        densityVK = nnz(VK) / N^2,
        t_coeff = t_coeff,
        t_rhs = t_rhs,
        t_VK = t_VK,
        t_Aop = t_Aop,
        t_convert = t_convert,
        t_factor = t_factor,
        t_solve = t_solve,
        t_error = t_error,
        total_time = total_time,
    )

    if keep_solution
        return merge(base, (
            cu_num = cu_num,
            cu_interp_exact = cu_interp_exact,
            cf = cf,
            VK = VK,
        ))
    else
        return base
    end
end

# ============================================================
## 7. Run convergence experiment
# ============================================================

@printf("Warm-up run...\n")
solve_volterra_once(
    256;
    tol = tol,
    keep_solution = false,
    use_fastalmostbanded = true,
)

Ns = evenly_spaced_integer_nodes(2, 40000, 100)

results = Vector{Any}()

@printf("\n============================================================\n")
@printf("Convergence test for aging Maxwell Volterra equation\n")
@printf("Exact solution: two smoothed step layers\n")
@printf("Independent RHS: split Gauss quadrature\n")
@printf("x1 = %.4f, x2 = %.4f, A1 = %.2f, A2 = %.2f\n", x1, x2, A1, A2)
@printf("δ1 = %.3e, δ2 = %.3e, tol = %.1e\n", δ1, δ2, tol)
@printf("RHS_SPLIT_L = %.1f, RHS_Q = %d, Nref_coeff = %d\n", RHS_SPLIT_L, RHS_Q, NREF_COEFF)
@printf("Ns: %d integer nodes in [%d, %d]\n", length(Ns), first(Ns), last(Ns))
@printf("============================================================\n")

for N in Ns
    keep = (N == maximum(Ns))

    push!(results, solve_volterra_once(
        N;
        tol = tol,
        keep_solution = keep,
        use_fastalmostbanded = true,

        # 默认自动检测 almost-banded 结构。
        # 如果你想手动固定结构，可以改成例如：
        # top_rows = 24,
        # lower_bw = 28,
        # upper_bw = 700,
        top_rows = nothing,
        lower_bw = nothing,
        upper_bw = nothing,

        detect_droptol = 0.0,
        convert_droptol = 0.0,
    ))
end

result_max = results[end]
Nmax = result_max.N
cu_max = result_max.cu_num
cu_interp_exact_max = result_max.cu_interp_exact
cf_max = result_max.cf

@printf("\nLargest-N result:\n")
@printf("Nmax                       = %d\n", Nmax)
@printf("solver                     = %s\n", String(result_max.solver))
@printf("top_rows/lower_bw/upper_bw = %d / %d / %d\n",
    result_max.top_rows, result_max.lower_bw, result_max.upper_bw)
@printf("tail ratio num coeffs      = %.3e\n", tail_ratio(cu_max))
@printf("tail ratio interp exact    = %.3e\n", tail_ratio(cu_interp_exact_max))
@printf("err_inf                    = %.3e\n", result_max.err_inf)
@printf("rel_inf                    = %.3e\n", result_max.rel_inf)
@printf("coeff_err                  = %.3e\n", result_max.coeff_err)
@printf("coeff_rel                  = %.3e\n", result_max.coeff_rel)
@printf("res_inf                    = %.3e\n", result_max.res_inf)

# ============================================================
# 8. Absolute-error CSV data
# ============================================================

datadir = DATA_DIR
mkpath(datadir)

abs_error_csv = save_absolute_error_csv(
    joinpath(datadir, "ex1_aging_maxwell_abs_error.csv"),
    results,
)

# ============================================================
# 9. Print summary table
# ============================================================

@printf("\nSummary table\n")
@printf(" N        solver                    top  l_bw  u_bw  nnz(VK)       err_inf      rel_inf      coeff_err    res_inf      t_rhs     t_VK      t_factor   t_solve    total\n")
@printf("---------------------------------------------------------------------------------------------------------------------------------------------------------\n")
for r in results
    @printf("%-8d %-25s %-4d %-5d %-5d %-13d %.3e   %.3e   %.3e   %.3e   %-9.4f %-9.4f %-10.4f %-10.4f %.4f\n",
        r.N,
        String(r.solver),
        r.top_rows,
        r.lower_bw,
        r.upper_bw,
        r.nnzVK,
        r.err_inf,
        r.rel_inf,
        r.coeff_err,
        r.res_inf,
        r.t_rhs,
        r.t_VK,
        r.t_factor,
        r.t_solve,
        r.total_time,
    )
end

println("\nAbsolute-error CSV saved to $(abs_error_csv)")
println("Run examples/ex1_2ndVolterraEqn/ex1_aging_maxwell_plot_abs_error.jl to draw the absolute-error figure.")
