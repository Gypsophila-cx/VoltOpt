############################################################
# ex4_VoltMat_cos1000.jl
#
# Solve the manufactured oscillatory Volterra integro-differential equation
#
#   u'(x) + u(x) + ∫_0^x cos(omega*x*y) u(y) dy = f(x),
#   x ∈ [0,1],  u(0)=1.
#
# Exact solution:
#
#   u(x) = cos(kappa*x),   kappa = 1000.
#
# Integral term:
#
#   I(x) = ∫_0^x cos(omega*x*y) cos(kappa*y) dy
#
#        = 1/2 [
#             sin((omega*x-kappa)*x)/(omega*x-kappa)
#           + sin((omega*x+kappa)*x)/(omega*x+kappa)
#          ].
#
# Stable form:
#
#   sin(a*x)/a = x*sinc(a*x),  sinc(z)=sin(z)/z.
#
# Therefore
#
#   f(x) = -kappa*sin(kappa*x)
#          + cos(kappa*x)
#          + I(x).
#
# Run:
#
#   julia --project=. examples/ex4_SPVIDE/ex4_VoltMat_cos1000.jl
#
# Optional PowerShell variables:
#
#   $env:EX4_N = "3000"
#   $env:EX4_OMEGA = "50.0"
#   $env:EX4_KAPPA = "1000.0"
#   $env:EX4_GRID_M = "20000"
#
############################################################

using LinearAlgebra
using SparseArrays
using Printf
using FFTW

script_dir = @__DIR__

function find_project_root(start_dir::AbstractString=@__DIR__)
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

project_root = find_project_root()
src_dir = joinpath(project_root, "src")
data_dir = joinpath(script_dir, "data")
mkpath(data_dir)

include(joinpath(src_dir, "VoltMat", "voltmat_cheb.jl"))
using .VoltMat_Cheb

# ============================================================
# 0. Parameters
# ============================================================

function env_float(name::AbstractString, default::Float64)
    return parse(Float64, get(ENV, name, string(default)))
end

function env_int(name::AbstractString, default::Int)
    return parse(Int, get(ENV, name, string(default)))
end

omega = env_float("EX4_OMEGA", 50.0)
kappa = env_float("EX4_KAPPA", 1000.0)

N = env_int("EX4_N", 3000)
N_ref = max(N + 1, round(Int, 1.01 * N))

GRID_M = env_int("EX4_GRID_M", max(20_000, 4 * N_ref))
SHAPE_POINTS = env_int("EX4_SHAPE_POINTS", 4000)

drop_tol = eps(Float64)

# ============================================================
# 1. Chebyshev utilities on [-1,1]
# ============================================================

function cheb_lobatto_nodes(M::Int)
    M >= 1 || error("M must be positive.")
    return cos.(pi .* (0:M) ./ M)
end

function cheb_coeffs_dct_values(vals::AbstractVector{<:Real})
    M = length(vals) - 1
    y = FFTW.r2r(Float64.(vals), FFTW.REDFT00)
    c = y ./ M
    c[1] *= 0.5
    c[end] *= 0.5
    return c
end

function chebcoeffs_fixedN(f, M::Int)
    ξ = cheb_lobatto_nodes(M)
    vals = f.(ξ)
    return cheb_coeffs_dct_values(vals)
end

function cheb_values_on_lobatto_grid(c::AbstractVector{<:Real}, M::Int)
    Nloc = length(c) - 1
    M >= Nloc || error("Evaluation grid degree M must satisfy M >= degree(c).")

    cp = zeros(Float64, M + 1)
    cp[1:Nloc + 1] .= c

    tmp = copy(cp)
    tmp[1] *= 2.0
    tmp[end] *= 2.0

    vals = FFTW.r2r(tmp, FFTW.REDFT00) ./ 2.0
    return vals
end

function cheb_eval_clenshaw(c::AbstractVector{<:Real}, x)
    n = length(c) - 1
    n == 0 && return c[1]

    b1 = 0.0
    b2 = 0.0

    for k in n:-1:1
        b0 = 2.0 * x * b1 - b2 + c[k + 1]
        b2 = b1
        b1 = b0
    end

    return x * b1 - b2 + c[1]
end

# ============================================================
# 2. Ultraspherical rectangular matrices
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

function identity_cheb_rect(N::Int)
    I = collect(1:(N + 1))
    J = collect(1:(N + 1))
    V = ones(Float64, N + 1)
    return sparse(I, J, V, N + 2, N + 1)
end

# ============================================================
# 3. Physical problem on x ∈ [0,1]
# ============================================================

@inline function xphys_of_xi(ξ)
    return 0.5 * (ξ + 1.0)
end

@inline function xi_of_xphys(x)
    return 2.0 * x - 1.0
end

@inline function sinc_unpi(z)
    az = abs(z)
    if az < 1.0e-8
        z2 = z * z
        return 1.0 - z2 / 6.0 + z2 * z2 / 120.0 - z2 * z2 * z2 / 5040.0
    else
        return sin(z) / z
    end
end

@inline function u_exact_phys(x)
    return cos(kappa * x)
end

@inline function du_exact_phys(x)
    return -kappa * sin(kappa * x)
end

@inline function I_exact_phys(x)
    a = omega * x - kappa
    b = omega * x + kappa

    Ia = x * sinc_unpi(a * x)
    Ib = x * sinc_unpi(b * x)

    return 0.5 * (Ia + Ib)
end

@inline function f_phys(x)
    return du_exact_phys(x) + u_exact_phys(x) + I_exact_phys(x)
end

@inline function u_exact_xi(ξ)
    return u_exact_phys(xphys_of_xi(ξ))
end

@inline function f_mapped_xi(ξ)
    return f_phys(xphys_of_xi(ξ))
end

# Mapped Volterra kernel:
#
#   ∫_0^x cos(omega*x*y)u(y)dy
#   = ∫_{-1}^ξ 1/2 cos(omega*x(ξ)*x(η)) U(η)dη.
#
@inline function K_mapped(ξ, η)
    x = xphys_of_xi(ξ)
    y = xphys_of_xi(η)
    return 0.5 * cos(omega * x * y)
end

u0 = u_exact_phys(0.0)

# ============================================================
# 4. Build Volterra matrix
# ============================================================

function build_volterra_matrix_square(Nsquare::Int; tol=drop_tol)
    VK = VoltMat_Cheb.voltmat_cheb_square(
        K_mapped,
        Nsquare;
        tol = tol,
        use_fft_top = true,
    )

    VKs = sparse(Float64.(VK))

    @printf("  square VK: size = %d x %d, nnz = %d\n",
            size(VKs, 1), size(VKs, 2), nnz(VKs))

    return VKs
end

function build_volterra_matrix_rect(N::Int; tol=drop_tol)
    # Unknown modes: 0,...,N.
    # Residual modes after multiplication/conversion: 0,...,N-1.
    # The square Volterra matrix with Nsquare=N+1 gives output rows 0,...,N+1.
    VK_big = build_volterra_matrix_square(N + 1; tol=tol)
    VK = sparse(VK_big[:, 1:(N + 1)])

    @printf("  rectangular VK: size = %d x %d, nnz = %d\n",
            size(VK, 1), size(VK, 2), nnz(VK))

    return VK
end

# ============================================================
# 5. Assemble tau system
# ============================================================

function assemble_tau_system(N::Int; tol=drop_tol)
    @printf("  Building D0, S0, identity, and VK ...\n")

    D0 = diffmat_cheb_to_ultra1_rect(N)
    S0 = convertmat_cheb_to_ultra1_rect(N, N + 2)
    Irect = identity_cheb_rect(N)

    VK = build_volterra_matrix_rect(N; tol=tol)

    # Mapping x=(ξ+1)/2:
    #
    #   u_x = 2 U_ξ.
    #
    # The equation becomes
    #
    #   2 U'(ξ) + U(ξ) + V_K U(ξ) = f(x(ξ)).
    #
    A_res = sparse(2.0 * D0 + S0 * (Irect + VK))

    @printf("  Computing RHS Chebyshev coefficients to degree N+1 ...\n")
    f_cheb = chebcoeffs_fixedN(f_mapped_xi, N + 1)
    rhs_res = Vector{Float64}(S0 * f_cheb)

    # Boundary condition U(-1)=u(0)=1.
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
    rhs_tau[1] = u0
    rhs_tau[2:end] .= rhs_res

    return A_tau, rhs_tau, VK
end

function solve_one_N(N::Int; tol=drop_tol)
    @printf("\n============================================================\n")
    @printf("VoltMat solve, N = %d\n", N)
    @printf("omega = %.6e, kappa = %.6e, u0 = %.17e\n", omega, kappa, u0)

    local A_tau, rhs_tau, VK, c

    build_sec = @elapsed begin
        A_tau, rhs_tau, VK = assemble_tau_system(N; tol=tol)
    end

    @printf("  size(A_tau) = %d x %d\n", size(A_tau, 1), size(A_tau, 2))
    @printf("  nnz(A_tau)  = %d\n", nnz(A_tau))

    solve_sec = @elapsed begin
        F = lu(A_tau)
        c = F \ rhs_tau
    end

    residual_inf = norm(A_tau * c - rhs_tau, Inf)

    @printf("  build_sec    = %.6e\n", build_sec)
    @printf("  solve_sec    = %.6e\n", solve_sec)
    @printf("  residual_inf = %.6e\n", residual_inf)

    return (;
        N = N,
        coeffs = Vector{Float64}(c),
        residual_inf = residual_inf,
        nnzA = nnz(A_tau),
        nnzVK = nnz(VK),
        build_sec = build_sec,
        solve_sec = solve_sec,
        total_sec = build_sec + solve_sec,
    )
end

# ============================================================
# 6. Diagnostics
# ============================================================

function cauchy_coeff_error_inf(cN::AbstractVector{<:Real},
                                cP::AbstractVector{<:Real})
    L = max(length(cN), length(cP))
    a = zeros(Float64, L)
    b = zeros(Float64, L)

    a[1:length(cN)] .= cN
    b[1:length(cP)] .= cP

    return norm(a .- b, Inf)
end

function grid_true_error_inf(c::AbstractVector{<:Real}; Mgrid::Int=GRID_M)
    ξgrid = cheb_lobatto_nodes(Mgrid)
    unum = cheb_values_on_lobatto_grid(c, Mgrid)
    uex = u_exact_xi.(ξgrid)
    return maximum(abs.(unum .- uex))
end

function cauchy_grid_error_inf(cN::AbstractVector{<:Real},
                               cP::AbstractVector{<:Real};
                               Mgrid::Int=GRID_M)
    uN = cheb_values_on_lobatto_grid(cN, Mgrid)
    uP = cheb_values_on_lobatto_grid(cP, Mgrid)
    return maximum(abs.(uN .- uP))
end

function exact_solution_coeffs(N::Int)
    return chebcoeffs_fixedN(u_exact_xi, N)
end

function exact_discrete_residual_check(Ncheck::Int)
    @printf("\n============================================================\n")
    @printf("Exact solution discrete tau residual check, N = %d\n", Ncheck)

    A_tau, rhs_tau, VK = assemble_tau_system(Ncheck)
    cex = exact_solution_coeffs(Ncheck)

    r = A_tau * cex - rhs_tau

    res_inf = norm(r, Inf)
    rhs_inf = norm(rhs_tau, Inf)
    rel_res = res_inf / max(rhs_inf, 1.0)

    @printf("  exact_residual_inf = %.6e\n", res_inf)
    @printf("  exact_residual_rel = %.6e\n", rel_res)

    return (; N=Ncheck, res_inf, rel_res, rhs_inf)
end

function write_coefficients_csv(filename::AbstractString, c)
    open(filename, "w") do io
        println(io, "n,c_n,abs_c_n")
        for (j, cj) in enumerate(c)
            @printf(io, "%d,%.17e,%.17e\n", j - 1, cj, abs(cj))
        end
        flush(io)
    end
end

function write_shape_csv(filename::AbstractString, c; Mshape::Int=SHAPE_POINTS)
    open(filename, "w") do io
        println(io, "x,xi,u_num,u_exact,abs_error")
        for j in 0:Mshape
            x = j / Mshape
            ξ = xi_of_xphys(x)
            un = cheb_eval_clenshaw(c, ξ)
            ue = u_exact_phys(x)
            @printf(io, "%.17e,%.17e,%.17e,%.17e,%.17e\n",
                    x, ξ, un, ue, abs(un - ue))
        end
        flush(io)
    end
end

# ============================================================
# 7. Main
# ============================================================

function main()
    summary_csv = joinpath(data_dir, "ex4_VoltMat_cos1000_summary.csv")
    coeff_csv = joinpath(data_dir, "ex4_VoltMat_cos1000_coefficients_N$(N).csv")
    shape_csv = joinpath(data_dir, "ex4_VoltMat_cos1000_shape_N$(N).csv")

    @printf("\n============================================================\n")
    @printf("Manufactured oscillatory VIDE: u(x)=cos(kappa*x)\n")
    @printf("omega  = %.17e\n", omega)
    @printf("kappa  = %.17e\n", kappa)
    @printf("N      = %d\n", N)
    @printf("N_ref  = %d\n", N_ref)
    @printf("GRID_M = %d\n", GRID_M)
    @printf("u0     = %.17e\n", u0)

    # A moderate residual check.  It is useful for catching mapping mistakes.
    Ncheck = min(N, 1000)
    exact_check = exact_discrete_residual_check(Ncheck)

    @printf("\nSolving pair N = %d and N_ref = %d\n", N, N_ref)

    solN = solve_one_N(N)
    GC.gc()
    solP = solve_one_N(N_ref)

    true_err = grid_true_error_inf(solN.coeffs; Mgrid=GRID_M)
    cauchy_coeff = cauchy_coeff_error_inf(solN.coeffs, solP.coeffs)
    cauchy_grid = cauchy_grid_error_inf(solN.coeffs, solP.coeffs; Mgrid=GRID_M)

    open(summary_csv, "w") do io
        println(io, "method,N,N_ref,omega,kappa,true_grid_error,cauchy_coeff_error,cauchy_grid_error,residual_inf,nnzA,nnzVK,build_sec,solve_sec,total_sec,exact_check_N,exact_check_res_inf,exact_check_rel_res,status")
        row = (
            "VoltMat",
            N,
            N_ref,
            omega,
            kappa,
            true_err,
            cauchy_coeff,
            cauchy_grid,
            solN.residual_inf,
            solN.nnzA,
            solN.nnzVK,
            solN.build_sec,
            solN.solve_sec,
            solN.total_sec,
            exact_check.N,
            exact_check.res_inf,
            exact_check.rel_res,
            "ok",
        )
        println(io, join(row, ","))
        flush(io)
    end

    write_coefficients_csv(coeff_csv, solN.coeffs)
    write_shape_csv(shape_csv, solN.coeffs)

    @printf("\n============================================================\n")
    @printf("Summary\n")
    @printf("N                      = %d\n", N)
    @printf("N_ref                  = %d\n", N_ref)
    @printf("true_grid_error        = %.6e\n", true_err)
    @printf("cauchy_coeff_error     = %.6e\n", cauchy_coeff)
    @printf("cauchy_grid_error      = %.6e\n", cauchy_grid)
    @printf("residual_inf           = %.6e\n", solN.residual_inf)
    @printf("total_sec(N only)      = %.6e\n", solN.total_sec)
    @printf("exact_check_res_inf    = %.6e\n", exact_check.res_inf)
    @printf("exact_check_rel_res    = %.6e\n", exact_check.rel_res)
    @printf("summary_csv            = %s\n", summary_csv)
    @printf("coeff_csv              = %s\n", coeff_csv)
    @printf("shape_csv              = %s\n", shape_csv)
end

main()