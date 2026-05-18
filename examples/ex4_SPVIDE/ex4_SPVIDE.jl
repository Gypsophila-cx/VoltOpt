############################################################
# solve_spvide_cheb_ultra_default_bandwidth.jl
#
# Solve
#
#   2eps*u'(x) + (x+3)/2*u(x)
#     + 1/2 ∫_{-1}^x exp(mu*(x+1)*(y+1)/4) u(y) dy
#     = f((x+1)/2),
#
#   u(-1) = 1.
#
# Exact solution:
#
#   u(x) = sin((x+1)/2) + exp(-(x+1)/(2eps)).
#
# Important:
#   This script DOES NOT pass tail_half_bandwidth.
#   voltmat_cheb_square chooses its default tail bandwidth internally.
############################################################

using LinearAlgebra
using SparseArrays
using Printf
using FFTW
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


# ============================================================
# 0. Parameters
# ============================================================

# Use 1e-3 for debugging, then change to 1e-7.
const EPS = 1.0e-7
# const EPS = 1.0e-7

const MU = -1.0

const N_FINAL = 32_000

const N_LIST = [
    100,
    1_000,
    2_000,
    3_000,
    4_000,
    6_000,
    8_000,
    10_000,
    12_000,
    15_000,
    18_000,
    20_000,
    22_000,
    24_000,
    26_000,
    28_000,
    30_000,
    32_000
]

const DROP_TOL = eps()
const RHS_COEFF_METHOD = :dct
const NERR_FACTOR = 4


# ============================================================
# 1. Rectangular ultraspherical matrices
# ============================================================

function diffmat_cheb_to_ultra1_rect(N::Int)
    I = Int[]
    J = Int[]
    V = Float64[]

    sizehint!(I, N)
    sizehint!(J, N)
    sizehint!(V, N)

    for k in 1:N
        push!(I, k)
        push!(J, k + 1)
        push!(V, Float64(k))
    end

    return sparse(I, J, V, N, N + 1)
end


function convertmat_cheb_to_ultra1_rect(nrows::Int, ncols::Int)
    I = Int[]
    J = Int[]
    V = Float64[]

    sizehint!(I, 2ncols)
    sizehint!(J, 2ncols)
    sizehint!(V, 2ncols)

    # T_0 = C_0^(1)
    if nrows >= 1 && ncols >= 1
        push!(I, 1)
        push!(J, 1)
        push!(V, 1.0)
    end

    # T_1 = 1/2 C_1^(1)
    if nrows >= 2 && ncols >= 2
        push!(I, 2)
        push!(J, 2)
        push!(V, 0.5)
    end

    # T_k = 1/2 C_k^(1) - 1/2 C_{k-2}^(1), k >= 2.
    for k in 2:(ncols - 1)
        if k <= nrows - 1
            push!(I, k + 1)
            push!(J, k + 1)
            push!(V, 0.5)
        end

        if k - 2 <= nrows - 1
            push!(I, k - 1)
            push!(J, k + 1)
            push!(V, -0.5)
        end
    end

    return sparse(I, J, V, nrows, ncols)
end


function cheb_mult_affine_a_rect(N::Int)
    # a(x) = (x+3)/2 = 3/2 T_0 + 1/2 T_1.
    # Input modes:  0,...,N.
    # Output modes: 0,...,N+1.

    I = Int[]
    J = Int[]
    V = Float64[]

    sizehint!(I, 3N + 4)
    sizehint!(J, 3N + 4)
    sizehint!(V, 3N + 4)

    # 3/2 T_0 * u
    for j in 0:N
        push!(I, j + 1)
        push!(J, j + 1)
        push!(V, 1.5)
    end

    # 1/2 T_1 * T_0 = 1/2 T_1
    push!(I, 2)
    push!(J, 1)
    push!(V, 0.5)

    # 1/2 T_1 * T_j = 1/4 T_{j-1} + 1/4 T_{j+1}
    for j in 1:N
        push!(I, j)
        push!(J, j + 1)
        push!(V, 0.25)

        push!(I, j + 2)
        push!(J, j + 1)
        push!(V, 0.25)
    end

    return sparse(I, J, V, N + 2, N + 1)
end


# ============================================================
# 2. Chebyshev transforms and evaluation
# ============================================================

function cheb_lobatto_nodes(M::Int)
    return cos.(pi .* (0:M) ./ M)
end


function cheb_coeffs_dct_values(vals::AbstractVector{<:Real})
    N = length(vals) - 1

    y = FFTW.r2r(Float64.(vals), FFTW.REDFT00)

    c = y ./ N
    c[1]   *= 0.5
    c[end] *= 0.5

    return c
end


function chebcoeffs_fixedN(f, N::Int; method=:dct)
    if method == :dct
        x = cheb_lobatto_nodes(N)
        vals = f.(x)
        return cheb_coeffs_dct_values(vals)
    else
        error("Only method=:dct is used in this script.")
    end
end


function cheb_values_on_lobatto_grid(c::AbstractVector{<:Real}, M::Int)
    N = length(c) - 1
    @assert M >= N

    cp = zeros(Float64, M + 1)
    cp[1:N + 1] .= c

    tmp = copy(cp)
    tmp[1]   *= 2.0
    tmp[end] *= 2.0

    vals = FFTW.r2r(tmp, FFTW.REDFT00) ./ 2.0
    return vals
end


# ============================================================
# 3. Exact solution, RHS, kernel
# ============================================================

@inline t_of_x(x) = 0.5 * (x + 1.0)

@inline function u_exact_x(x)
    t = t_of_x(x)
    return sin(t) + exp(-t / EPS)
end


@inline function phi1(z::Float64)
    if abs(z) < 1.0e-8
        return 1.0 + z / 2.0 + z^2 / 6.0 + z^3 / 24.0 + z^4 / 120.0
    else
        return expm1(z) / z
    end
end


function f_phys(t)
    eps = EPS
    μ = MU

    exp_layer = exp(-t / eps)
    v = sin(t) + exp_layer

    # eps*v'(t)
    diff_part = eps * cos(t) - exp_layer

    # (1+t)v(t)
    local_part = (1.0 + t) * v

    # ∫_0^t exp(μ*t*s) sin(s) ds
    α = μ * t
    I_sin = (exp(α * t) * (α * sin(t) - cos(t)) + 1.0) / (α^2 + 1.0)

    # ∫_0^t exp(μ*t*s) exp(-s/eps) ds
    z = μ * t^2 - t / eps
    I_layer = t * phi1(z)

    return diff_part + local_part + I_sin + I_layer
end


@inline f_mapped_x(x) = f_phys(t_of_x(x))


@inline function K_mapped(x, y)
    return 0.5 * exp(MU * (x + 1.0) * (y + 1.0) / 4.0)
end


# ============================================================
# 4. Volterra matrix construction
# ============================================================

function build_volterra_matrix_square_default(N::Int; tol=DROP_TOL)
    # Important:
    #   Do not pass tail_half_bandwidth.
    #   voltmat_cheb_square will use its internal default tail bandwidth.
    VK = VoltMat_Cheb.voltmat_cheb_square(
        K_mapped,
        N;
        tol=tol,
        use_fft_top=true,
    )

    VKs = sparse(Float64.(VK))

    @printf("  square VK: size = %d x %d, nnz = %d\n",
            size(VKs, 1), size(VKs, 2), nnz(VKs))

    return VKs
end


function build_volterra_matrix_rect_default(N::Int; tol=DROP_TOL)
    # Unknown u_N has input modes 0,...,N.
    # Non-derivative term must be kept to output modes 0,...,N+1.
    #
    # voltmat_cheb_square only returns square matrices, so build degree N+1
    # and keep columns 0,...,N.

    VK_big = build_volterra_matrix_square_default(N + 1; tol=tol)

    VK = sparse(VK_big[:, 1:(N + 1)])

    @printf("  rectangular VK: size = %d x %d, nnz = %d\n",
            size(VK, 1), size(VK, 2), nnz(VK))

    return VK
end


# ============================================================
# 5. Assemble top-tau ultraspherical system
# ============================================================

function assemble_tau_system_rect_default(
    N::Int;
    rhs_coeff_method=RHS_COEFF_METHOD,
    tol=DROP_TOL,
)
    @printf("  Building rectangular D0, S0, Ma ...\n")

    # Residual rows: C_0^(1),...,C_{N-1}^(1)
    D0 = diffmat_cheb_to_ultra1_rect(N)

    # Convert Chebyshev output modes 0,...,N+1 to C^(1) rows 0,...,N-1.
    S0 = convertmat_cheb_to_ultra1_rect(N, N + 2)

    # Multiplication by a(x), input 0:N -> output 0:N+1.
    Ma = cheb_mult_affine_a_rect(N)

    @printf("  Building Volterra matrix with voltmat_cheb_square default bandwidth ...\n")
    VK = build_volterra_matrix_rect_default(N; tol=tol)

    @printf("  Assembling residual operator ...\n")
    A_res = sparse(2.0 * EPS * D0 + S0 * (Ma + VK))

    @printf("  Computing RHS coefficients to degree N+1 ...\n")
    f_cheb = chebcoeffs_fixedN(
        f_mapped_x,
        N + 1;
        method=rhs_coeff_method,
    )

    rhs_res = Vector{Float64}(S0 * f_cheb)

    # Boundary row u(-1)=1:
    #   sum_k (-1)^k c_k = 1.
    bc_vals = [isodd(k) ? -1.0 : 1.0 for k in 0:N]

    bc_row = sparse(
        ones(Int, N + 1),
        collect(1:N + 1),
        bc_vals,
        1,
        N + 1,
    )

    A_tau = vcat(bc_row, A_res)

    rhs_tau = Vector{Float64}(undef, N + 1)
    rhs_tau[1] = 1.0
    rhs_tau[2:end] .= rhs_res

    return A_tau, rhs_tau, VK
end


# ============================================================
# 6. Solve one N
# ============================================================

function solve_one_N_default(
    N::Int;
    rhs_coeff_method=RHS_COEFF_METHOD,
    tol=DROP_TOL,
)
    @printf("\n============================================================\n")
    @printf("Solving N = %d\n", N)
    @printf("eps = %.3e, mu = %.3e\n", EPS, MU)

    A_tau, rhs_tau, VK = assemble_tau_system_rect_default(
        N;
        rhs_coeff_method=rhs_coeff_method,
        tol=tol,
    )

    @printf("  size(A_tau) = %d x %d\n", size(A_tau, 1), size(A_tau, 2))
    @printf("  nnz(A_tau)  = %d\n", nnz(A_tau))

    @printf("  Sparse LU factorization and solve ...\n")
    F = lu(A_tau)
    c = F \ rhs_tau

    res_inf = norm(A_tau * c - rhs_tau, Inf)
    @printf("  residual_inf = %.6e\n", res_inf)

    return (; N, coeffs=c, res_inf, nnzA=nnz(A_tau), nnzVK=nnz(VK))
end


# ============================================================
# 7. Error diagnostics
# ============================================================

function diagnostic_grid_size(N::Int)
    return max(20_000, NERR_FACTOR * N)
end


function true_error_inf(c::AbstractVector{<:Real}; Ndiag::Int)
    N = length(c) - 1
    M = max(diagnostic_grid_size(Ndiag), N)

    xgrid = cheb_lobatto_nodes(M)
    unum = cheb_values_on_lobatto_grid(c, M)
    uex = u_exact_x.(xgrid)

    return maximum(abs.(unum .- uex))
end


function cauchy_coeff_error_inf(cN::AbstractVector{<:Real},
                                cP::AbstractVector{<:Real})
    N1 = length(cN)
    N2 = length(cP)
    L = max(N1, N2)

    cN_pad = zeros(Float64, L)
    cP_pad = zeros(Float64, L)

    cN_pad[1:N1] .= cN
    cP_pad[1:N2] .= cP

    return norm(cN_pad .- cP_pad, Inf)
end


# ============================================================
# 8. Run sequence
# ============================================================

function run_error_sequence_default(;
    N_list=N_LIST,
    rhs_coeff_method=RHS_COEFF_METHOD,
    tol=DROP_TOL,
)
    coeffs_cache = Dict{Int, Vector{Float64}}()
    residuals = Dict{Int, Float64}()

    needed_N = Set{Int}()

    for N in N_list
        push!(needed_N, N)
        push!(needed_N, round(Int, 1.01 * N))
    end

    push!(needed_N, N_FINAL)

    needed_N_sorted = sort(collect(needed_N))

    @printf("\nDegrees to solve:\n")
    @printf("%s\n", string(needed_N_sorted))

    for N in needed_N_sorted
        sol = solve_one_N_default(
            N;
            rhs_coeff_method=rhs_coeff_method,
            tol=tol,
        )

        coeffs_cache[N] = sol.coeffs
        residuals[N] = sol.res_inf

        GC.gc()
    end

    Ns = collect(N_list)
    true_errors = Float64[]
    cauchy_errors = Float64[]

    @printf("\n============================================================\n")
    @printf("Computing errors ...\n")

    for N in Ns
        Np = round(Int, 1.01 * N)

        cN = coeffs_cache[N]
        cP = coeffs_cache[Np]

        e_true = true_error_inf(cN; Ndiag=Np)
        e_cauchy = cauchy_coeff_error_inf(cN, cP)

        push!(true_errors, e_true)
        push!(cauchy_errors, e_cauchy)

        @printf(
            "N = %7d, Np = %7d, true_err = %.6e, cauchy_err = %.6e, residual = %.6e\n",
            N,
            Np,
            e_true,
            e_cauchy,
            residuals[N],
        )
    end

    return (; Ns, true_errors, cauchy_errors, coeffs_cache, residuals)
end


# ============================================================
# 9. Plotting
# ============================================================

function write_error_history_csv(filename, Ns, true_errors, cauchy_errors, residuals)
    open(filename, "w") do io
        println(io, "N,true_error_inf,cauchy_coeff_error_inf,residual_inf")
        for i in eachindex(Ns)
            N = Ns[i]
            @printf(io, "%d,%.17e,%.17e,%.17e\n",
                N, true_errors[i], cauchy_errors[i], residuals[N])
        end
    end
    return filename
end


function write_coefficients_csv(filename, c)
    open(filename, "w") do io
        println(io, "n,c_n,abs_c_n")
        for (j, cj) in enumerate(c)
            @printf(io, "%d,%.17e,%.17e\n", j - 1, cj, abs(cj))
        end
    end
    return filename
end


@inline function v_exact_t(t)
    return sin(t) + exp(-t / EPS)
end


function plot_solution_default(
    c::AbstractVector{<:Real};
    filename=joinpath(FIGURES_DIR, "ex4_SPVIDE_solution.pdf"),
)
    N = length(c) - 1

    # 在 x 上用 Chebyshev--Lobatto 网格评估数值解。
    M = max(100_000, 5N)

    x_desc = cheb_lobatto_nodes(M)          # 从 x=1 到 x=-1
    u_desc = cheb_values_on_lobatto_grid(c, M)

    # 拉回到原变量 t=(x+1)/2。
    t_desc = 0.5 .* (x_desc .+ 1.0)
    v_exact_desc = v_exact_t.(t_desc)

    # 为了图像从 t=0 到 t=1，从左到右显示，需要 reverse。
    t_plot = reverse(t_desc)
    v_num_plot = reverse(u_desc)
    v_exact_plot = reverse(v_exact_desc)

    p = plot(
        t_plot,
        v_num_plot;
        lw=2,
        label="numerical",
        xlabel="t",
        ylabel="v(t)",
        legend=:bottomright,
        grid=false,
        minorgrid=false,
    )

    plot!(
        p,
        t_plot,
        v_exact_plot;
        lw=1.5,
        ls=:dash,
        label="exact",
    )

    savefig(p, filename)
    return p
end


function dense_log10_ticks(vmin, vmax; minor=true)
    @assert vmin > 0
    @assert vmax > 0

    kmin = floor(Int, log10(vmin))
    kmax = ceil(Int, log10(vmax))

    vals = Float64[]
    labels = String[]

    for k in kmin:2:kmax+2
        # 主刻度：10^k
        push!(vals, 10.0^k)
        push!(labels, "10^{$k}")

        if minor && k < kmax
            # 次刻度：2,3,...,9 × 10^k
            for a in 2:9
                push!(vals, a * 10.0^k)
                push!(labels, "")
            end
        end
    end

    keep = (vals .>= vmin) .& (vals .<= vmax)
    return vals[keep], labels[keep]
end


function plot_true_error_default(
    Ns,
    true_errors;
    filename=joinpath(FIGURES_DIR, "ex4_SPVIDE_inferror.pdf"),
)
    ymin = minimum(true_errors)
    ymax = maximum(true_errors)

    # 稍微扩展范围，避免点贴着边界。
    ymin_plot = 10.0^(floor(log10(ymin)) - 0.2)
    ymax_plot = 10.0^(ceil(log10(ymax)) + 0.2)

    yt = dense_log10_ticks(ymin_plot, ymax_plot; minor=false)

    p = plot(
        Ns,
        true_errors;
        yscale=:log10,
        yticks=yt,
        marker=:circle,
        lw=2,
        xlabel="N",
        ylabel="||u_N - u_exact||_∞",
        label="true error",
        legend=:topright,
        color = :black,
    )

    savefig(p, filename)
    return p
end


function fixed_power_ticks(kmin::Int, kmax::Int)
    # Return ticks 10^k for k = kmin:kmax.
    vals = [10.0^k for k in kmin:2:kmax]
    labels = ["10^{$k}" for k in kmin:2:kmax]
    return vals, labels
end


function plot_cauchy_error_default(
    Ns,
    cauchy_errors;
    filename=joinpath(FIGURES_DIR, "ex4_SPVIDE_cauchy_error.pdf"),
)
    # y-axis ticks from 1e-15 to 1e-2.
    yt = fixed_power_ticks(-16, -1)

    p = plot(
        Ns,
        cauchy_errors;
        # x-axis is linear by default.
        yscale=:log10,
        ylim=(1e-16, 1e-1),
        yticks=yt,

        marker=:circle,
        lw=2,
        xlabel="N",
        ylabel="Caucy Error",
        legend=false,
        color=:black,
    )

    savefig(p, filename)
    return p
end


# ============================================================
# 10. Main
# ============================================================

out = run_error_sequence_default(
    N_list=N_LIST;
    rhs_coeff_method=RHS_COEFF_METHOD,
    tol=DROP_TOL,
)

c_final = out.coeffs_cache[N_FINAL]
mkpath(FIGURES_DIR)

tag = @sprintf("eps%.0e_mu%.0f", EPS, MU)
tag = replace(tag, "+" => "", "-" => "minus")

history_csv = joinpath(FIGURES_DIR, "ex4_SPVIDE_error_history_$(tag).csv")
coeff_csv = joinpath(FIGURES_DIR, "ex4_SPVIDE_coefficients_N$(N_FINAL)_$(tag).csv")
solution_pdf = joinpath(FIGURES_DIR, "ex4_SPVIDE_solution_$(tag).pdf")
error_pdf = joinpath(FIGURES_DIR, "ex4_SPVIDE_error_$(tag).pdf")
cauchy_pdf = joinpath(FIGURES_DIR, "ex4_SPVIDE_cauchy_$(tag).pdf")

write_error_history_csv(history_csv, out.Ns, out.true_errors, out.cauchy_errors, out.residuals)
write_coefficients_csv(coeff_csv, c_final)

p1 = plot_solution_default(
    c_final;
    filename=solution_pdf,
)

p2 = plot_true_error_default(
    out.Ns,
    out.true_errors;
    filename=error_pdf,
)

p3 = plot_cauchy_error_default(
    out.Ns,
    out.cauchy_errors;
    filename=cauchy_pdf,
)

display(p1)
display(p2)
display(p3)

@printf("\nDone.\n")
@printf("Saved outputs:\n")
@printf("  %s\n", solution_pdf)
@printf("  %s\n", error_pdf)
@printf("  %s\n", cauchy_pdf)
@printf("  %s\n", history_csv)
@printf("  %s\n", coeff_csv)
