############################################################
# compare_firstkind_direct_vs_differentiated.jl
############################################################

using LinearAlgebra
using SparseArrays
using FFTW
using Printf
using Plots

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

include(joinpath(SRC_DIR, "VoltMat", "voltmat_cheb.jl"))
using .VoltMat_Cheb

############################################################
# 0. Wrapper: build Volterra matrix
############################################################

function build_volterra_matrix(K, N; tol=1e-13)
    M = VoltMat_Cheb.voltmat_cheb_square(K, N; tol=tol)
    if M isa Tuple
        M = M[1]
    end

    M = sparse(M)
    n = N + 1
    return M[1:n, 1:n]
end

############################################################
# 1. Chebyshev utilities
############################################################

chebpts_lobatto(N) = cospi.(collect(0:N) ./ N)

function vals_to_chebcoeffs(v::AbstractVector{T}) where {T}
    N = length(v) - 1
    if N == 0
        return copy(v)
    end
    y = FFTW.r2r(collect(v), FFTW.REDFT00)
    c = y ./ N
    c[1] /= 2
    c[end] /= 2
    return c
end

function cheb_eval(c::AbstractVector, x::Real)
    N = length(c) - 1
    if N == 0
        return c[1]
    end
    b1 = zero(eltype(c))
    b2 = zero(eltype(c))
    for k = N:-1:1
        b0 = 2*x*b1 - b2 + c[k+1]
        b2 = b1
        b1 = b0
    end
    return c[1] + x*b1 - b2
end

cheb_eval(c::AbstractVector, x::AbstractVector) = [cheb_eval(c, xx) for xx in x]

############################################################
# 2. Gauss-Legendre quadrature
############################################################

function gausslegendre(n::Integer)
    n <= 0 && error("n must be positive.")
    if n == 1
        return [0.0], [2.0]
    end
    β = [k / sqrt(4k^2 - 1) for k in 1:n-1]
    J = SymTridiagonal(zeros(n), β)
    F = eigen(J)
    x = F.values
    w = 2 .* F.vectors[1, :].^2
    return x, w
end

############################################################
# 3. Exact solution and kernels
############################################################

u_exact(x) = 1 / (25x^2 + 1)

# K1(x,y)=exp(-20(x-y)^2)
K1(x, y) = exp(-20 * (x - y)^2)

# d/dx K1
K1x(x, y) = begin
    d = x - y
    -40d * exp(-20d^2)
end

# K2(x,y)=exp(-20(x-y)^2 - x(x-y))
K2(x, y) = begin
    d = x - y
    exp(-20d^2 - x*d)
end

# d/dx K2
K2x(x, y) = begin
    d = x - y
    E = -20d^2 - x*d
    (-40d - (2x - y)) * exp(E)
end

const TEST_CASES = [
    (
        name = "K1",
        label = "K1(x,y)=exp(-20(x-y)^2)",
        K = K1,
        Kx = K1x,
    ),
    (
        name = "K2",
        label = "K2(x,y)=exp(-20(x-y)^2-x(x-y))",
        K = K2,
        Kx = K2x,
    ),
]

############################################################
# 4. Build f and f'
############################################################

# f(x) = ∫_{-1}^x K(x,y) u_exact(y) dy
function f_value(x, K, xg, wg)
    h = (x + 1) / 2
    m = (x - 1) / 2
    if h == 0
        return 0.0
    end
    acc = 0.0
    @inbounds for q in eachindex(xg)
        y = h * xg[q] + m
        acc += wg[q] * K(x, y) * u_exact(y)
    end
    return h * acc
end

# f'(x) = u(x) + ∫_{-1}^x ∂_xK(x,y)u(y)dy
# because K(x,x)=1 for both kernels
function fprime_value(x, Kx, xg, wg)
    h = (x + 1) / 2
    m = (x - 1) / 2
    if h == 0
        return u_exact(x)
    end
    acc = 0.0
    @inbounds for q in eachindex(xg)
        y = h * xg[q] + m
        acc += wg[q] * Kx(x, y) * u_exact(y)
    end
    return u_exact(x) + h * acc
end

function build_rhs_coeffs_f(K, N; nq=600)
    xcheb = chebpts_lobatto(N)
    xg, wg = gausslegendre(nq)
    vals = [f_value(x, K, xg, wg) for x in xcheb]
    return vals_to_chebcoeffs(vals)
end

function build_rhs_coeffs_fprime(Kx, N; nq=600)
    xcheb = chebpts_lobatto(N)
    xg, wg = gausslegendre(nq)
    vals = [fprime_value(x, Kx, xg, wg) for x in xcheb]
    return vals_to_chebcoeffs(vals)
end

############################################################
# 5. Two solution methods
############################################################

# Method A: directly solve first-kind equation
#     V_K u = f
function solve_direct_first_kind(case, N; tol=1e-13, nq=700)
    VK = build_volterra_matrix(case.K, N; tol=tol)
    b = build_rhs_coeffs_f(case.K, N; nq=nq)
    c = VK \ b
    relres = norm(VK*c - b) / max(norm(b), eps())
    return c, relres
end

# Method B: differentiate to second-kind equation
#     (I + V_{Kx}) u = f'
function solve_differentiated_second_kind(case, N; tol=1e-13, nq=700)
    VKx = build_volterra_matrix(case.Kx, N; tol=tol)
    n = N + 1
    A = spdiagm(0 => ones(n)) + VKx
    b = build_rhs_coeffs_fprime(case.Kx, N; nq=nq)
    c = A \ b
    relres = norm(A*c - b) / max(norm(b), eps())
    return c, relres
end

############################################################
# 6. Error computation
############################################################

function max_error_on_grid(c; ngrid=2001)
    x = collect(range(-1, 1; length=ngrid))
    uh = cheb_eval(c, x)
    ue = u_exact.(x)
    err = abs.(uh .- ue)
    return maximum(err), x, err
end

############################################################
# 7. Manual y-axis ticks
############################################################

function log10_ticks_inrange(ymin, ymax; step=1)
    @assert ymin > 0
    @assert ymax > 0
    @assert ymin < ymax

    kmin = ceil(Int, log10(ymin))
    kmax = floor(Int, log10(ymax))

    ks = collect(kmin:step:kmax)
    vals = 10.0 .^ ks
    labels = ["10^{$k}" for k in ks]
    return vals, labels
end

function write_compare_csv(path, Ns, direct_errs, diff_errs)
    open(path, "w") do io
        println(io, "N,direct_first_kind_err,differentiated_second_kind_err")
        for i in eachindex(Ns)
            @printf(io, "%d,%.17e,%.17e\n", Ns[i], direct_errs[i], diff_errs[i])
        end
    end
    return path
end

############################################################
# 8. Plot one kernel per figure
############################################################

function plot_compare_for_one_kernel(case;
        Ns = [16, 24, 32, 48, 64, 96, 128, 192, 256, 384],
        tol = 1e-13,
        nq = 700,
        ngrid = 3001,
        ymin = 1e-15,
        ymax = 1e0,
        ytick_step = 2,
        outdir = FIGURES_DIR)

    mkpath(outdir)

    direct_errs = Float64[]
    diff_errs   = Float64[]

    println("============================================================")
    println("Kernel: ", case.label)

    for N in Ns
        println("N = $N")

        c1, r1 = solve_direct_first_kind(case, N; tol=tol, nq=nq)
        e1, _, _ = max_error_on_grid(c1; ngrid=ngrid)
        push!(direct_errs, e1)

        c2, r2 = solve_differentiated_second_kind(case, N; tol=tol, nq=nq)
        e2, _, _ = max_error_on_grid(c2; ngrid=ngrid)
        push!(diff_errs, e2)

        println(@sprintf("  direct first-kind:        relres = %.3e, maxerr = %.3e", r1, e1))
        println(@sprintf("  differentiated 2nd-kind:  relres = %.3e, maxerr = %.3e", r2, e2))
    end

    ytk = log10_ticks_inrange(ymin, ymax; step=ytick_step)

    plt = plot(
        xlabel = "N",
        ylabel = "max absolute error",
        yscale = :log10,
        ylims = (ymin, ymax),
        yticks = ytk,
        grid = false,
        minorgrid = false,
        linewidth = 2,
        marker = :circle,
        title = "",
    )

    plot!(
        plt,
        Ns,
        direct_errs;
        label = "direct first-kind",
        linewidth = 2,
        marker = :circle,
    )

    plot!(
        plt,
        Ns,
        diff_errs;
        label = "differentiate -> second-kind",
        linewidth = 2,
        marker = :diamond,
    )

    outfile = joinpath(outdir, "ex2_compare_$(case.name)_direct_vs_differentiated.pdf")
    csvfile = joinpath(outdir, "ex2_compare_$(case.name)_direct_vs_differentiated.csv")
    write_compare_csv(csvfile, Ns, direct_errs, diff_errs)
    savefig(plt, outfile)
    println("Saved: ", outfile)
    println("Saved: ", csvfile)

    return (
        Ns = Ns,
        direct_errs = direct_errs,
        diff_errs = diff_errs,
        outfile = outfile,
        csvfile = csvfile,
    )
end

############################################################
# 9. Main
############################################################

println("Current output folder: ", FIGURES_DIR)

results = Dict()

for case in TEST_CASES
    results[case.name] = plot_compare_for_one_kernel(
        case;
        Ns = [16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024],
        tol = eps()/10,
        nq = 700,
        ngrid = 3001,
        ymin = 1e-15,
        ymax = 1e0,
        ytick_step = 2,
        outdir = FIGURES_DIR,
    )
end
