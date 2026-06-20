############################################################
# ex4_VoltMat_manufactured_omega50.jl
#
# Manufactured oscillatory VIDE solved by VoltMat_Cheb.
#
# Problem on t in [0,T]:
#
#   u'(t) + u(t) + ∫_0^t cos(omega*t*s) u(s) ds = f(t),
#   u(0) = u0.
#
# Default parameters:
#
#   T      = 1
#   omega  = 50
#   delta  = 0.001842
#   c      = T/2
#
# Exact solution:
#
#   u(t) = delta^2 / ((t-c)^2 + delta^2).
#
# The integral term
#
#   I(t) = ∫_0^t cos(omega*t*s) u(s) ds
#
# is evaluated analytically by Ei:
#
#   Ei(z) = -Gamma(0,-z) ± i*pi.
#
# Diagnostics:
#
#   1. Float64 RHS vs BigFloat RHS consistency check.
#   2. Ei integral formula vs independent piecewise Gauss--Legendre check.
#   3. Exact manufactured solution inserted into the discrete tau system.
#   4. N vs N_ref Cauchy error.
#   5. Grid error against exact solution.
#
############################################################

using LinearAlgebra
using SparseArrays
using Printf
using FFTW
using SpecialFunctions

script_dir = @__DIR__

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

volterra_root = find_project_root()
src_dir = joinpath(volterra_root, "src")
data_dir = joinpath(script_dir, "data")

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

function env_bool(name::AbstractString, default::Bool)
    s = lowercase(strip(get(ENV, name, default ? "true" : "false")))
    return s in ("1", "true", "yes", "y", "on")
end

Tphys = env_float("EX4_T", 1.0)
omega = env_float("EX4_OMEGA", 50.0)

# For T=1, delta=0.001842 gives mapped singularity distance
# 2*delta/T ≈ 0.003684, hence about 10000 Chebyshev modes to
# reach machine precision.
delta = env_float("EX4_DELTA", 0.001842)
c0 = Tphys / 2.0

N = env_int("EX4_N", 10_000)
N_ref = max(N + 1, round(Int, 1.01 * N))

grid_factor = env_int("EX4_GRID_FACTOR", 4)
shape_points = env_int("EX4_SHAPE_POINTS", 2000)

drop_tol = eps(Float64)

do_bigfloat_check = env_bool("EX4_RHS_BIGFLOAT_CHECK", true)
bigfloat_precision = env_int("EX4_BIGFLOAT_PRECISION", 256)
rhs_check_points = env_int("EX4_RHS_CHECK_POINTS", 41)

do_I_quad_check = env_bool("EX4_I_QUAD_CHECK", true)
I_quad_check_points = env_int("EX4_I_QUAD_CHECK_POINTS", 201)
I_quad_nq = env_int("EX4_I_QUAD_NQ", 128)

do_exact_residual_check = env_bool("EX4_EXACT_RESIDUAL_CHECK", true)
exact_residual_N = env_int("EX4_EXACT_RESIDUAL_N", min(N, 2000))

mkpath(data_dir)

# ============================================================
# 1. Chebyshev utilities
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
    x = cheb_lobatto_nodes(M)
    vals = f.(x)
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
    if n == 0
        return c[1]
    end

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
# 3. Exact solution and mapping
# ============================================================

@inline function t_of_x(x)
    return 0.5 * Tphys * (x + 1.0)
end

@inline function x_of_t(t)
    return 2.0 * t / Tphys - 1.0
end

@inline function u_exact_t(t)
    return delta^2 / ((t - c0)^2 + delta^2)
end

@inline function du_exact_t(t)
    den = (t - c0)^2 + delta^2
    return -2.0 * delta^2 * (t - c0) / den^2
end

@inline function u_exact_x(x)
    return u_exact_t(t_of_x(x))
end

u0 = u_exact_t(0.0)

# ============================================================
# 4. Complex Ei through incomplete Gamma, with branch handling
# ============================================================

# SpecialFunctions.gamma(0,-z) is essentially E1(-z).
# For the principal branch of Ei with branch cut on the negative real axis:
#
#   Ei(z) = -Gamma(0,-z) + i*pi,   Im(z) > 0,
#   Ei(z) = -Gamma(0,-z) - i*pi,   Im(z) < 0.
#
# On the negative real axis we select the side according to the
# integration path.

function Ei_offcut_float(z::ComplexF64)
    G = -gamma(0.0 + 0.0im, -z)

    if imag(z) > 0.0
        return G + pi * im
    elseif imag(z) < 0.0
        return G - pi * im
    else
        return ComplexF64(real(G), 0.0)
    end
end

function Ei_offcut_big(z::Complex{BigFloat})
    z0 = Complex{BigFloat}(big(0), big(0))
    G = -gamma(z0, -z)
    pi_b = big(pi)

    if imag(z) > 0
        return G + Complex{BigFloat}(0, pi_b)
    elseif imag(z) < 0
        return G - Complex{BigFloat}(0, pi_b)
    else
        return Complex{BigFloat}(real(G), big(0))
    end
end

function is_on_negative_real_axis_float(z::ComplexF64; atol::Float64=1.0e-14)
    return real(z) < 0.0 && abs(imag(z)) <= atol * max(1.0, abs(real(z)))
end

function is_on_negative_real_axis_big(z::Complex{BigFloat})
    return real(z) < 0 && abs(imag(z)) <= big"1.0e-60" * max(big(1), abs(real(z)))
end

function Ei_on_cut_float(z::ComplexF64, side::Int)
    G = -gamma(0.0 + 0.0im, -ComplexF64(real(z), 0.0))
    base = ComplexF64(real(G), 0.0)

    if side > 0
        return base + pi * im
    elseif side < 0
        return base - pi * im
    else
        return base
    end
end

function Ei_on_cut_big(z::Complex{BigFloat}, side::Int)
    z0 = Complex{BigFloat}(big(0), big(0))
    zr = Complex{BigFloat}(real(z), big(0))
    G = -gamma(z0, -zr)
    base = Complex{BigFloat}(real(G), big(0))
    pi_b = big(pi)

    if side > 0
        return base + Complex{BigFloat}(0, pi_b)
    elseif side < 0
        return base - Complex{BigFloat}(0, pi_b)
    else
        return base
    end
end

function Ei_value_for_segment_endpoint_float(z::ComplexF64, side::Int)
    if is_on_negative_real_axis_float(z)
        return Ei_on_cut_float(z, side)
    else
        return Ei_offcut_float(z)
    end
end

function Ei_value_for_segment_endpoint_big(z::Complex{BigFloat}, side::Int)
    if is_on_negative_real_axis_big(z)
        return Ei_on_cut_big(z, side)
    else
        return Ei_offcut_big(z)
    end
end

function segment_cut_sides_float(w0::ComplexF64, w1::ComplexF64)
    # side = +1: upper side of negative real cut
    # side = -1: lower side of negative real cut
    dw = w1 - w0

    side0 = 0
    side1 = 0

    if is_on_negative_real_axis_float(w0)
        # Leaving w0 in direction dw.
        side0 = imag(dw) >= 0.0 ? +1 : -1
    end

    if is_on_negative_real_axis_float(w1)
        # Approaching w1 from direction dw, so the side is determined by dw.
        side1 = imag(dw) >= 0.0 ? +1 : -1
    end

    return side0, side1
end

function segment_cut_sides_big(w0::Complex{BigFloat}, w1::Complex{BigFloat})
    dw = w1 - w0

    side0 = 0
    side1 = 0

    if is_on_negative_real_axis_big(w0)
        side0 = imag(dw) >= 0 ? +1 : -1
    end

    if is_on_negative_real_axis_big(w1)
        side1 = imag(dw) >= 0 ? +1 : -1
    end

    return side0, side1
end

function crosses_negative_real_cut_interior_float(w0::ComplexF64, w1::ComplexF64)
    if !(real(w0) < 0.0 && real(w1) < 0.0)
        return 0
    end

    if imag(w0) < 0.0 && imag(w1) > 0.0
        return +1
    elseif imag(w0) > 0.0 && imag(w1) < 0.0
        return -1
    else
        return 0
    end
end

function crosses_negative_real_cut_interior_big(w0::Complex{BigFloat}, w1::Complex{BigFloat})
    if !(real(w0) < 0 && real(w1) < 0)
        return 0
    end

    if imag(w0) < 0 && imag(w1) > 0
        return +1
    elseif imag(w0) > 0 && imag(w1) < 0
        return -1
    else
        return 0
    end
end

function Ei_difference_along_segment(w0::ComplexF64, w1::ComplexF64)
    side0, side1 = segment_cut_sides_float(w0, w1)

    E0 = Ei_value_for_segment_endpoint_float(w0, side0)
    E1 = Ei_value_for_segment_endpoint_float(w1, side1)

    d = E1 - E0

    cross = crosses_negative_real_cut_interior_float(w0, w1)

    # Principal Ei jumps by +2*pi*i when crossing from below to above.
    # To obtain the continuous antiderivative along the segment, remove it.
    if cross == +1
        d -= 2.0 * pi * im
    elseif cross == -1
        d += 2.0 * pi * im
    end

    return d
end

function Ei_difference_along_segment(w0::Complex{BigFloat}, w1::Complex{BigFloat})
    side0, side1 = segment_cut_sides_big(w0, w1)

    E0 = Ei_value_for_segment_endpoint_big(w0, side0)
    E1 = Ei_value_for_segment_endpoint_big(w1, side1)

    d = E1 - E0

    cross = crosses_negative_real_cut_interior_big(w0, w1)
    two_pi_i = Complex{BigFloat}(0, 2 * big(pi))

    if cross == +1
        d -= two_pi_i
    elseif cross == -1
        d += two_pi_i
    end

    return d
end

function J_expint_float(t::Float64, z::ComplexF64)
    q = omega * t
    if abs(q) < 1.0e-14
        return 0.0 + 0.0im
    end

    w0 = ComplexF64(-im * q * z)
    w1 = ComplexF64(im * q * (t - z))

    return exp(im * q * z) * Ei_difference_along_segment(w0, w1)
end

function J_expint_big(t_in, z_in)
    t = BigFloat(t_in)
    om = BigFloat(omega)
    q = om * t

    if abs(q) < big"1.0e-60"
        return Complex{BigFloat}(0, 0)
    end

    z = Complex{BigFloat}(BigFloat(real(z_in)), BigFloat(imag(z_in)))

    im_b = Complex{BigFloat}(0, 1)
    w0 = -im_b * q * z
    w1 = im_b * q * (Complex{BigFloat}(t, 0) - z)

    return exp(im_b * q * z) * Ei_difference_along_segment(w0, w1)
end

function I_exact_t(t::Float64)
    if abs(t) < 1.0e-14
        return 0.0
    end

    zp = ComplexF64(c0, delta)
    zm = ComplexF64(c0, -delta)

    val = delta / (2.0im) * (J_expint_float(t, zp) - J_expint_float(t, zm))
    return real(val)
end

function I_exact_t_big(t_in)
    return setprecision(BigFloat, bigfloat_precision) do
        t = BigFloat(t_in)
        c = BigFloat(c0)
        dlt = BigFloat(delta)

        if abs(t) < big"1.0e-60"
            return BigFloat(0)
        end

        zp = Complex{BigFloat}(c, dlt)
        zm = Complex{BigFloat}(c, -dlt)

        im_b = Complex{BigFloat}(0, 1)
        val = dlt / (2 * im_b) * (J_expint_big(t, zp) - J_expint_big(t, zm))
        return real(val)
    end
end

function f_phys(t::Float64)
    return du_exact_t(t) + u_exact_t(t) + I_exact_t(t)
end

function f_phys_big(t_in)
    return setprecision(BigFloat, bigfloat_precision) do
        t = BigFloat(t_in)
        c = BigFloat(c0)
        dlt = BigFloat(delta)

        den = (t - c)^2 + dlt^2
        u = dlt^2 / den
        du = -2 * dlt^2 * (t - c) / den^2
        return du + u + I_exact_t_big(t)
    end
end

@inline function f_mapped_x(x)
    return f_phys(t_of_x(x))
end

@inline function K_mapped(x, y)
    tx = t_of_x(x)
    ty = t_of_x(y)
    return 0.5 * Tphys * cos(omega * tx * ty)
end

# ============================================================
# 5. BigFloat RHS validation
# ============================================================

function validate_rhs_bigfloat()
    if !do_bigfloat_check
        @printf("\nBigFloat RHS check skipped.\n")
        return (; max_abs=NaN, max_rel=NaN, worst_t=NaN, status="skipped")
    end

    @printf("\n============================================================\n")
    @printf("BigFloat RHS validation\n")
    @printf("precision = %d bits, points = %d\n", bigfloat_precision, rhs_check_points)

    ts = collect(range(0.0, Tphys; length=rhs_check_points))

    max_abs = 0.0
    max_rel = 0.0
    worst_t = 0.0

    for t in ts
        ff = f_phys(t)
        fb = Float64(f_phys_big(t))
        abs_err = abs(ff - fb)
        rel_err = abs_err / max(abs(fb), 1.0)

        if abs_err > max_abs
            max_abs = abs_err
            worst_t = t
        end
        max_rel = max(max_rel, rel_err)
    end

    status = isfinite(max_abs) && max_abs <= 1.0e-10 ? "ok" : "warning"

    @printf("max_abs_rhs_float64_minus_bigfloat = %.6e\n", max_abs)
    @printf("max_rel_rhs_float64_minus_bigfloat = %.6e\n", max_rel)
    @printf("worst_t                            = %.17e\n", worst_t)
    @printf("rhs_check_status                   = %s\n", status)

    return (; max_abs, max_rel, worst_t, status)
end

function write_rhs_check_csv(filename::AbstractString, check)
    open(filename, "w") do io
        println(io, "T,omega,delta,bigfloat_precision,check_points,max_abs_diff,max_rel_diff,worst_t,status")
        @printf(io, "%.17e,%.17e,%.17e,%d,%d,%.17e,%.17e,%.17e,%s\n",
                Tphys, omega, delta, bigfloat_precision, rhs_check_points,
                check.max_abs, check.max_rel, check.worst_t, check.status)
        flush(io)
    end
end

# ============================================================
# 6. Independent quadrature check for I(t)
# ============================================================

function gausslegendre_nodes_weights(n::Int)
    n >= 1 || error("n must be positive.")
    if n == 1
        return [0.0], [2.0]
    end

    beta = [k / sqrt(4.0 * k^2 - 1.0) for k in 1:(n - 1)]
    J = SymTridiagonal(zeros(Float64, n), beta)
    F = eigen(J)

    x = F.values
    w = 2.0 .* abs2.(F.vectors[1, :])
    return x, w
end

function quad_on_interval(f, a::Float64, b::Float64, xg, wg)
    if b == a
        return 0.0
    end

    mid = 0.5 * (a + b)
    rad = 0.5 * (b - a)
    s = mid .+ rad .* xg
    return rad * sum(wg .* f.(s))
end

function local_quadrature_breakpoints(t::Float64)
    pts = Float64[0.0, t]

    offsets = Float64[
        -256.0, -128.0, -64.0, -32.0, -16.0, -8.0, -4.0, -2.0, -1.0,
        -0.5, -0.25, 0.0, 0.25, 0.5,
        1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0,
    ]

    for a in offsets
        p = c0 + a * delta
        if 0.0 < p < t
            push!(pts, p)
        end
    end

    sort!(unique!(pts))
    return pts
end

function I_quad_t(t::Float64; nq::Int=I_quad_nq)
    if abs(t) < 1.0e-14
        return 0.0
    end

    xg, wg = gausslegendre_nodes_weights(nq)
    pts = local_quadrature_breakpoints(t)

    integrand(s) = cos(omega * t * s) * u_exact_t(s)

    total = 0.0
    for j in 1:(length(pts) - 1)
        total += quad_on_interval(integrand, pts[j], pts[j + 1], xg, wg)
    end

    return total
end

function validate_I_against_quadrature()
    if !do_I_quad_check
        @printf("\nIndependent I(t) quadrature check skipped.\n")
        return (; max_abs=NaN, max_rel=NaN, worst_t=NaN, status="skipped")
    end

    @printf("\n============================================================\n")
    @printf("Independent quadrature check for I(t)\n")
    @printf("points = %d, Gauss points per panel = %d\n", I_quad_check_points, I_quad_nq)

    ts = collect(range(0.0, Tphys; length=I_quad_check_points))

    max_abs = 0.0
    max_rel = 0.0
    worst_t = 0.0

    for t in ts
        Ie = I_exact_t(t)
        Iq = I_quad_t(t; nq=I_quad_nq)

        abs_err = abs(Ie - Iq)
        rel_err = abs_err / max(abs(Iq), 1.0)

        if abs_err > max_abs
            max_abs = abs_err
            worst_t = t
        end
        max_rel = max(max_rel, rel_err)
    end

    status = isfinite(max_abs) && max_abs <= 1.0e-10 ? "ok" : "warning"

    @printf("max_abs_I_Ei_minus_quad = %.6e\n", max_abs)
    @printf("max_rel_I_Ei_minus_quad = %.6e\n", max_rel)
    @printf("worst_t                 = %.17e\n", worst_t)
    @printf("I_quad_check_status     = %s\n", status)

    return (; max_abs, max_rel, worst_t, status)
end

function write_I_quad_check_csv(filename::AbstractString, check)
    open(filename, "w") do io
        println(io, "T,omega,delta,check_points,nq,max_abs_diff,max_rel_diff,worst_t,status")
        @printf(io, "%.17e,%.17e,%.17e,%d,%d,%.17e,%.17e,%.17e,%s\n",
                Tphys, omega, delta, I_quad_check_points, I_quad_nq,
                check.max_abs, check.max_rel, check.worst_t, check.status)
        flush(io)
    end
end

# ============================================================
# 7. Volterra matrix construction
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
    VK_big = build_volterra_matrix_square(N + 1; tol=tol)
    VK = sparse(VK_big[:, 1:(N + 1)])

    @printf("  rectangular VK: size = %d x %d, nnz = %d\n",
            size(VK, 1), size(VK, 2), nnz(VK))

    return VK
end

# ============================================================
# 8. Assemble and solve tau system
# ============================================================

function assemble_tau_system(N::Int; tol=drop_tol)
    @printf("  Building D0, S0, identity, and VK ...\n")

    D0 = diffmat_cheb_to_ultra1_rect(N)
    S0 = convertmat_cheb_to_ultra1_rect(N, N + 2)
    Irect = identity_cheb_rect(N)

    VK = build_volterra_matrix_rect(N; tol=tol)

    # Mapping:
    #
    #   t = T*(x+1)/2,
    #   u_t = (2/T) U_x,
    #
    #   (2/T) U'(x) + U(x)
    #     + ∫_{-1}^x (T/2) cos(omega*t(x)*t(y)) U(y) dy
    #     = f(t(x)).
    #
    A_res = sparse((2.0 / Tphys) * D0 + S0 * (Irect + VK))

    @printf("  Computing RHS Chebyshev coefficients to degree N+1 ...\n")
    f_cheb = chebcoeffs_fixedN(f_mapped_x, N + 1)
    rhs_res = Vector{Float64}(S0 * f_cheb)

    # Boundary condition U(-1)=u(0)=u0.
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
    @printf("VoltMat manufactured oscillatory VIDE, N = %d\n", N)
    @printf("T = %.6g, omega = %.6g, delta = %.6g, u0 = %.17e\n",
            Tphys, omega, delta, u0)

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

    res_inf = norm(A_tau * c - rhs_tau, Inf)

    @printf("  build_sec    = %.6e\n", build_sec)
    @printf("  solve_sec    = %.6e\n", solve_sec)
    @printf("  residual_inf = %.6e\n", res_inf)

    return (;
        N = N,
        coeffs = Vector{Float64}(c),
        res_inf = res_inf,
        nnzA = nnz(A_tau),
        nnzVK = nnz(VK),
        build_sec = build_sec,
        solve_sec = solve_sec,
        total_sec = build_sec + solve_sec,
    )
end

# ============================================================
# 9. Exact solution discrete residual check
# ============================================================

function exact_solution_coeffs(N::Int)
    return chebcoeffs_fixedN(u_exact_x, N)
end

function check_exact_solution_discrete_residual(Ncheck::Int)
    if !do_exact_residual_check
        @printf("\nExact solution discrete residual check skipped.\n")
        return (; N=Ncheck, res_inf=NaN, rel_res=NaN, rhs_inf=NaN,
                  build_sec=NaN, status="skipped")
    end

    @printf("\n============================================================\n")
    @printf("Exact solution discrete tau residual check, N = %d\n", Ncheck)

    local A_tau, rhs_tau, VK
    build_sec = @elapsed begin
        A_tau, rhs_tau, VK = assemble_tau_system(Ncheck)
    end

    cex = exact_solution_coeffs(Ncheck)
    r = A_tau * cex - rhs_tau

    res_inf = norm(r, Inf)
    rhs_inf = norm(rhs_tau, Inf)
    rel_res = res_inf / max(rhs_inf, 1.0)

    # This check is strict only when Ncheck is large enough to resolve
    # the sharp exact solution.
    status = isfinite(res_inf) && rel_res <= 1.0e-6 ? "ok" : "warning"

    @printf("exact_discrete_residual_inf = %.6e\n", res_inf)
    @printf("exact_discrete_residual_rel = %.6e\n", rel_res)
    @printf("exact_residual_rhs_inf      = %.6e\n", rhs_inf)
    @printf("exact_residual_build_sec    = %.6e\n", build_sec)
    @printf("exact_residual_status       = %s\n", status)

    return (; N=Ncheck, res_inf, rel_res, rhs_inf, build_sec, status)
end

function write_exact_residual_csv(filename::AbstractString, check)
    open(filename, "w") do io
        println(io, "N,T,omega,delta,res_inf,rel_res,rhs_inf,build_sec,status")
        @printf(io, "%d,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%s\n",
                check.N, Tphys, omega, delta,
                check.res_inf, check.rel_res, check.rhs_inf,
                check.build_sec, check.status)
        flush(io)
    end
end

# ============================================================
# 10. Error diagnostics and output
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

function grid_error_inf(c::AbstractVector{<:Real}; Ndiag::Int)
    M = max(grid_factor * Ndiag, Ndiag, 20_000)

    xgrid = cheb_lobatto_nodes(M)
    unum = cheb_values_on_lobatto_grid(c, M)
    uex = u_exact_x.(xgrid)

    return maximum(abs.(unum .- uex))
end

function cauchy_grid_error_inf(cN::AbstractVector{<:Real},
                               cP::AbstractVector{<:Real};
                               Ndiag::Int)
    M = max(grid_factor * Ndiag, Ndiag, 20_000)

    uN = cheb_values_on_lobatto_grid(cN, M)
    uP = cheb_values_on_lobatto_grid(cP, M)

    return maximum(abs.(uN .- uP))
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

function write_shape_csv(filename::AbstractString, c; Mshape::Int=shape_points)
    open(filename, "w") do io
        println(io, "t,x,u_num,u_exact,abs_error")
        for j in 0:Mshape
            t = Tphys * j / Mshape
            x = x_of_t(t)
            un = cheb_eval_clenshaw(c, x)
            ue = u_exact_t(t)
            @printf(io, "%.17e,%.17e,%.17e,%.17e,%.17e\n",
                    t, x, un, ue, abs(un - ue))
        end
        flush(io)
    end
end

# ============================================================
# 11. Main
# ============================================================

function main()
    rhs_check_csv = joinpath(data_dir, "ex4_VoltMat_manufactured_omega50_rhs_check.csv")
    I_quad_check_csv = joinpath(data_dir, "ex4_VoltMat_manufactured_omega50_I_quad_check.csv")
    exact_residual_csv = joinpath(data_dir, "ex4_VoltMat_manufactured_omega50_exact_residual_check.csv")
    summary_csv = joinpath(data_dir, "ex4_VoltMat_manufactured_omega50_summary.csv")
    coeff_csv = joinpath(data_dir, "ex4_VoltMat_manufactured_omega50_coefficients_N$(N).csv")
    shape_csv = joinpath(data_dir, "ex4_VoltMat_manufactured_omega50_shape_N$(N).csv")

    @printf("\n============================================================\n")
    @printf("Manufactured oscillatory VIDE by VoltMat\n")
    @printf("T       = %.17e\n", Tphys)
    @printf("omega   = %.17e\n", omega)
    @printf("delta   = %.17e\n", delta)
    @printf("c       = %.17e\n", c0)
    @printf("N       = %d\n", N)
    @printf("N_ref   = %d\n", N_ref)
    @printf("u0      = %.17e\n", u0)

    rhs_check = validate_rhs_bigfloat()
    write_rhs_check_csv(rhs_check_csv, rhs_check)

    I_quad_check = validate_I_against_quadrature()
    write_I_quad_check_csv(I_quad_check_csv, I_quad_check)

    exact_res_check = check_exact_solution_discrete_residual(exact_residual_N)
    write_exact_residual_csv(exact_residual_csv, exact_res_check)

    if rhs_check.status == "warning"
        @warn "RHS Float64 vs BigFloat check is not small. Accuracy may be RHS-limited." rhs_check.max_abs rhs_check.max_rel
    end

    if I_quad_check.status == "warning"
        @warn "Ei integral formula and independent quadrature disagree. Check branch correction or quadrature resolution." I_quad_check.max_abs I_quad_check.max_rel
    end

    if exact_res_check.status == "warning"
        @warn "The exact manufactured solution does not satisfy the discrete tau system to high accuracy at the check degree." exact_res_check.N exact_res_check.res_inf exact_res_check.rel_res
    end

    @printf("\nSolving pair N = %d and N_ref = %d\n", N, N_ref)

    solN = solve_one_N(N)
    GC.gc()
    solP = solve_one_N(N_ref)

    err_true = grid_error_inf(solN.coeffs; Ndiag=N_ref)
    err_cauchy_coeff = cauchy_coeff_error_inf(solN.coeffs, solP.coeffs)
    err_cauchy_grid = cauchy_grid_error_inf(solN.coeffs, solP.coeffs; Ndiag=N_ref)

    open(summary_csv, "w") do io
        println(io, "method,N,N_ref,T,omega,delta,true_grid_error,cauchy_coeff_error,cauchy_grid_error,residual_inf,nnzA,nnzVK,build_sec,solve_sec,total_sec,rhs_bigfloat_max_abs,rhs_bigfloat_max_rel,rhs_check_status,I_quad_max_abs,I_quad_max_rel,I_quad_check_status,exact_residual_N,exact_residual_inf,exact_residual_rel,exact_residual_status,status")
        row = (
            "VoltMat",
            N,
            N_ref,
            Tphys,
            omega,
            delta,
            err_true,
            err_cauchy_coeff,
            err_cauchy_grid,
            solN.res_inf,
            solN.nnzA,
            solN.nnzVK,
            solN.build_sec,
            solN.solve_sec,
            solN.total_sec,
            rhs_check.max_abs,
            rhs_check.max_rel,
            rhs_check.status,
            I_quad_check.max_abs,
            I_quad_check.max_rel,
            I_quad_check.status,
            exact_res_check.N,
            exact_res_check.res_inf,
            exact_res_check.rel_res,
            exact_res_check.status,
            "ok",
        )
        println(io, join(row, ","))
        flush(io)
    end

    write_coefficients_csv(coeff_csv, solN.coeffs)
    write_shape_csv(shape_csv, solN.coeffs)

    @printf("\n============================================================\n")
    @printf("Summary\n")
    @printf("N                         = %d\n", N)
    @printf("N_ref                     = %d\n", N_ref)
    @printf("true_grid_error           = %.6e\n", err_true)
    @printf("cauchy_coeff_error        = %.6e\n", err_cauchy_coeff)
    @printf("cauchy_grid_error         = %.6e\n", err_cauchy_grid)
    @printf("residual_inf              = %.6e\n", solN.res_inf)
    @printf("total_sec(N only)         = %.6e\n", solN.total_sec)
    @printf("rhs_bigfloat_max_abs      = %.6e\n", rhs_check.max_abs)
    @printf("rhs_bigfloat_max_rel      = %.6e\n", rhs_check.max_rel)
    @printf("rhs_check_status          = %s\n", rhs_check.status)
    @printf("I_quad_max_abs            = %.6e\n", I_quad_check.max_abs)
    @printf("I_quad_max_rel            = %.6e\n", I_quad_check.max_rel)
    @printf("I_quad_check_status       = %s\n", I_quad_check.status)
    @printf("exact_residual_N          = %d\n", exact_res_check.N)
    @printf("exact_residual_inf        = %.6e\n", exact_res_check.res_inf)
    @printf("exact_residual_rel        = %.6e\n", exact_res_check.rel_res)
    @printf("exact_residual_status     = %s\n", exact_res_check.status)
    @printf("rhs_check_csv             = %s\n", rhs_check_csv)
    @printf("I_quad_check_csv          = %s\n", I_quad_check_csv)
    @printf("exact_residual_csv        = %s\n", exact_residual_csv)
    @printf("summary_csv               = %s\n", summary_csv)
    @printf("coeff_csv                 = %s\n", coeff_csv)
    @printf("shape_csv                 = %s\n", shape_csv)
end

main()