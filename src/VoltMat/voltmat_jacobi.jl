module Voltmat_Jacobi

using LinearAlgebra
using SparseArrays

export voltmat_jacobi,
       jacobi_interp_coeffs_Khat,
       adaptive_jacobi_interp_coeffs_Khat,
       jacobi_values,
       jacobi_values!,
       gaussjacobi_nodes,
       jacobi_volterra_topdense_from_C,
       voltmat_jacobi_full

# =============================================================================
# Voltmat_Jacobi -- top-dense/band-tail Jacobi Volterra matrix constructor
#
# General Jacobi-basis Volterra matrix constructor.
#
# Main interfaces:
#
#     B = voltmat_jacobi(K, N, alpha, beta)
#     B = voltmat_jacobi(C, N, alpha, beta)
#
# The output is a square SparseMatrixCSC of size (N+1) x (N+1), containing only
# the symmetric band -bandwidth <= ell-k <= bandwidth.  Default bandwidth is
# size(C,1).  In the adaptive constructor the returned coefficient matrix C
# is forced to satisfy size(C,1) >= size(C,2).
#
# C[i+1,j+1] is the coefficient of P_i^(alpha,beta)(x) P_j^(alpha,beta)(t)
# in Khat(x,t)=K(x,(x+1)/2*t+(x-1)/2).
# =============================================================================

const _LOG2 = log(2.0)
const _LOG2PI_HALF = 0.91893853320467274178032973640562

function _check_ab(alpha::Real, beta::Real)
    alpha > -1 || throw(ArgumentError("alpha must be > -1."))
    beta > -1 || throw(ArgumentError("beta must be > -1."))
    return Float64(alpha), Float64(beta)
end


@inline function _is_legendre_case(alpha::Float64, beta::Float64)
    return alpha == 0.0 && beta == 0.0
end

function _check_C_n_ge_m(C::AbstractMatrix)
    nrow, mcol = size(C)
    nrow >= mcol || throw(ArgumentError(
        "the coefficient matrix C must satisfy size(C,1) >= size(C,2); got size(C) = $(size(C))."
    ))
    return nrow, mcol
end

@inline function _default_top_rows(nrow::Int, mcol::Int,
                                  alpha::Float64, beta::Float64,
                                  N::Int)
    # For C in R^{n x m}, the non-Legendre Jacobi construction only needs
    # top n-m+1 dense rows; the remaining rows are covered by the symmetric
    # band with half-bandwidth n.  In the Legendre case the matrix is strictly
    # banded and no dense top block is assembled.
    if _is_legendre_case(alpha, beta)
        return 0
    end
    return min(max(nrow - mcol + 1, 0), N + 1)
end

@inline function _resolve_init_layer_backend(backend::Symbol,
                                             alpha::Float64,
                                             beta::Float64)
    backend in (:auto, :recurrence, :loggamma) ||
        throw(ArgumentError("init_layer_backend must be :auto, :recurrence, or :loggamma."))

    lambda = alpha + beta
    if backend == :auto
        return lambda > 2.0 ? :loggamma : :recurrence
    elseif backend == :loggamma
        lambda > 2.0 || throw(ArgumentError(
            "init_layer_backend=:loggamma is intended for alpha + beta > 2; got alpha + beta = $lambda."
        ))
        return :loggamma
    else
        return :recurrence
    end
end

# Lanczos log-gamma for positive real arguments. This avoids requiring
# SpecialFunctions.jl.
function _loggamma_lanczos(z::Float64)
    p = (
        0.99999999999980993,
        676.5203681218851,
       -1259.1392167224028,
        771.32342877765313,
       -176.61502916214059,
        12.507343278686905,
       -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    )

    if z < 0.5
        return log(pi) - log(sin(pi*z)) - _loggamma_lanczos(1.0 - z)
    end

    z -= 1.0
    x = p[1]

    @inbounds for i in 2:length(p)
        x += p[i] / (z + i - 1)
    end

    t = z + 7.5
    return _LOG2PI_HALF + (z + 0.5)*log(t) - t + log(x)
end

# -----------------------------------------------------------------------------
# Jacobi recurrence coefficients:
#     x P_n = A_n P_{n+1} + B_n P_n + C_n P_{n-1}.
# -----------------------------------------------------------------------------

@inline function _jacobi_x_coeffs(n::Int, alpha::Float64, beta::Float64)
    if n == 0
        A = 2.0 / (alpha + beta + 2.0)
        B = (beta - alpha) / (alpha + beta + 2.0)
        C = 0.0
        return A, B, C
    end

    s = alpha + beta

    A = 2.0 * (n + 1.0) * (n + s + 1.0) /
        ((2.0*n + s + 1.0) * (2.0*n + s + 2.0))

    B = (beta^2 - alpha^2) /
        ((2.0*n + s) * (2.0*n + s + 2.0))

    C = 2.0 * (n + alpha) * (n + beta) /
        ((2.0*n + s) * (2.0*n + s + 1.0))

    return A, B, C
end

@inline function _jacobi_b_orth(n::Int, alpha::Float64, beta::Float64)
    if n == 0
        return (beta - alpha) / (alpha + beta + 2.0)
    else
        s = alpha + beta
        return (beta^2 - alpha^2) / ((2.0*n + s) * (2.0*n + s + 2.0))
    end
end

@inline function _jacobi_a_orth(n::Int, alpha::Float64, beta::Float64)
    s = alpha + beta
    num = n * (n + alpha) * (n + beta) * (n + s)
    den = (2.0*n + s - 1.0) * (2.0*n + s + 1.0)
    return 2.0 / (2.0*n + s) * sqrt(num / den)
end

function gaussjacobi_nodes(Q::Integer, alpha::Real, beta::Real)
    alpha, beta = _check_ab(alpha, beta)
    Q <= 0 && throw(ArgumentError("Q must be positive."))

    diag = [_jacobi_b_orth(n, alpha, beta) for n in 0:Q-1]

    if Q == 1
        return diag
    end

    off = [_jacobi_a_orth(n, alpha, beta) for n in 1:Q-1]
    J = SymTridiagonal(diag, off)
    return eigen(J).values
end

# -----------------------------------------------------------------------------
# Jacobi values.
# -----------------------------------------------------------------------------

function jacobi_values!(p::AbstractVector{Float64}, x::Real, N::Integer,
                        alpha::Real, beta::Real)
    alpha, beta = _check_ab(alpha, beta)
    length(p) >= N + 1 || throw(ArgumentError("length(p) must be at least N+1."))
    N < 0 && throw(ArgumentError("N must be nonnegative."))

    xf = Float64(x)
    p[1] = 1.0

    if N >= 1
        p[2] = 0.5 * ((alpha - beta) + (alpha + beta + 2.0) * xf)
    end

    @inbounds for n in 1:N-1
        A, B, C = _jacobi_x_coeffs(n, alpha, beta)
        p[n+2] = ((xf - B) * p[n+1] - C * p[n]) / A
    end

    return p
end

function jacobi_values(x::Real, N::Integer, alpha::Real, beta::Real)
    p = zeros(Float64, N + 1)
    return jacobi_values!(p, x, N, alpha, beta)
end

function _jacobi_vandermonde(x::AbstractVector{<:Real}, deg::Integer,
                             alpha::Float64, beta::Float64)
    deg < 0 && throw(ArgumentError("deg must be nonnegative."))
    V = zeros(Float64, length(x), deg + 1)
    p = zeros(Float64, deg + 1)

    @inbounds for a in eachindex(x)
        jacobi_values!(p, x[a], deg, alpha, beta)
        for j in 1:deg+1
            V[a, j] = p[j]
        end
    end

    return V
end

# -----------------------------------------------------------------------------
# Interpolation of Khat in tensor-product Jacobi basis.
# -----------------------------------------------------------------------------

function jacobi_interp_coeffs_Khat(
    K::Function,
    nrow::Integer,
    mcol::Integer,
    alpha::Real,
    beta::Real;
    return_nodes::Bool=false,
)
    alpha, beta = _check_ab(alpha, beta)
    nrow <= 0 && throw(ArgumentError("nrow must be positive."))
    mcol <= 0 && throw(ArgumentError("mcol must be positive."))

    xs = gaussjacobi_nodes(nrow, alpha, beta)
    ts = gaussjacobi_nodes(mcol, alpha, beta)

    Vx = _jacobi_vandermonde(xs, nrow - 1, alpha, beta)
    Vt = _jacobi_vandermonde(ts, mcol - 1, alpha, beta)

    F = zeros(Float64, nrow, mcol)

    @inbounds for c in 1:mcol
        t = ts[c]
        for a in 1:nrow
            x = xs[a]
            z = 0.5 * (x + 1.0)
            b = 0.5 * (x - 1.0)
            F[a, c] = K(x, z * t + b)
        end
    end

    C = Vx \ F / transpose(Vt)
    return return_nodes ? (C, xs, ts) : C
end

# -----------------------------------------------------------------------------
# Chebfun-like adaptive coefficient chopping.
# -----------------------------------------------------------------------------

function _coeff_profiles(C::AbstractMatrix{<:Real})
    nrow, mcol = size(C)
    rp = zeros(Float64, nrow)
    cp = zeros(Float64, mcol)

    @inbounds for i in 1:nrow
        mx = 0.0
        for j in 1:mcol
            mx = max(mx, abs(Float64(C[i, j])))
        end
        rp[i] = mx
    end

    @inbounds for j in 1:mcol
        mx = 0.0
        for i in 1:nrow
            mx = max(mx, abs(Float64(C[i, j])))
        end
        cp[j] = mx
    end

    return rp, cp
end

function _monotone_envelope(v::AbstractVector{<:Real})
    n = length(v)
    env = zeros(Float64, n)

    if n == 0
        return env, 0.0
    end

    env[n] = abs(Float64(v[n]))

    @inbounds for k in n-1:-1:1
        env[k] = max(abs(Float64(v[k])), env[k+1])
    end

    scale = max(env[1], eps(Float64))

    @inbounds for k in 1:n
        env[k] /= scale
    end

    return env, scale
end

function _standard_chop_like(coeff_profile::AbstractVector{<:Real};
                             tol::Float64,
                             plateau_tol::Union{Nothing,Float64},
                             plateau_window::Integer,
                             plateau_drop::Float64)
    env, scale = _monotone_envelope(coeff_profile)
    n = length(env)

    if n == 0 || scale <= eps(Float64)
        return (cutoff=1, happy=true, tail=0.0, plateau=true, scale=scale)
    end

    ptol = plateau_tol === nothing ? tol^(2/3) : Float64(plateau_tol)

    direct = findfirst(x -> x <= tol, env)
    if direct !== nothing
        return (cutoff=max(1, direct), happy=true, tail=env[end],
                plateau=false, scale=scale)
    end

    w = min(Int(plateau_window), max(n - 1, 1))

    if n >= w + 2
        @inbounds for j in 2:(n - w)
            if env[j] <= ptol
                ratio = env[j+w] / max(env[j], eps(Float64))
                if ratio >= plateau_drop
                    return (cutoff=j, happy=true, tail=env[end],
                            plateau=true, scale=scale)
                end
            end
        end
    end

    return (cutoff=n, happy=false, tail=env[end], plateau=false, scale=scale)
end

function _chop_coefficient_matrix(C::AbstractMatrix{<:Real};
                                  tol::Float64,
                                  plateau_tol::Union{Nothing,Float64},
                                  plateau_window::Integer,
                                  plateau_drop::Float64,
                                  chop_guard::Integer)
    rp, cp = _coeff_profiles(C)

    rchop = _standard_chop_like(rp; tol=tol, plateau_tol=plateau_tol,
                                plateau_window=plateau_window,
                                plateau_drop=plateau_drop)
    cchop = _standard_chop_like(cp; tol=tol, plateau_tol=plateau_tol,
                                plateau_window=plateau_window,
                                plateau_drop=plateau_drop)

    nrow, mcol = size(C)
    guard = max(Int(chop_guard), 0)
    row_safe = rchop.cutoff <= max(nrow - guard, 1)
    col_safe = cchop.cutoff <= max(mcol - guard, 1)

    return (
        ncut=clamp(rchop.cutoff, 1, nrow),
        mcut=clamp(cchop.cutoff, 1, mcol),
        happy=rchop.happy && cchop.happy && row_safe && col_safe,
        row_happy=rchop.happy,
        col_happy=cchop.happy,
        row_safe=row_safe,
        col_safe=col_safe,
        row_tail=rchop.tail,
        col_tail=cchop.tail,
        row_plateau=rchop.plateau,
        col_plateau=cchop.plateau,
        row_scale=rchop.scale,
        col_scale=cchop.scale,
    )
end

function adaptive_jacobi_interp_coeffs_Khat(
    K::Function,
    alpha::Real,
    beta::Real;
    tol::Float64=1.0e-12,
    nmin::Integer=8,
    mmin::Integer=8,
    nmax::Integer=128,
    mmax::Integer=128,
    growth::Real=1.5,
    maxiters::Integer=50,
    warn_exceed_max::Bool=true,
    plateau_tol::Union{Nothing,Float64}=nothing,
    plateau_window::Integer=8,
    plateau_drop::Float64=0.1,
    chop_guard::Integer=4,
    enforce_n_ge_m::Bool=true,
    verbose::Bool=false,
)
    alpha, beta = _check_ab(alpha, beta)
    tol <= 0 && throw(ArgumentError("tol must be positive."))
    nmin <= 0 && throw(ArgumentError("nmin must be positive."))
    mmin <= 0 && throw(ArgumentError("mmin must be positive."))
    nmax < nmin && throw(ArgumentError("nmax must be >= nmin."))
    mmax < mmin && throw(ArgumentError("mmax must be >= mmin."))
    growth <= 1 && throw(ArgumentError("growth must be > 1."))
    maxiters <= 0 && throw(ArgumentError("maxiters must be positive."))

    nrow = enforce_n_ge_m ? max(Int(nmin), Int(mmin)) : Int(nmin)
    mcol = Int(mmin)
    last_C = zeros(Float64, 0, 0)
    converged = false
    niters = 0
    warned_n = false
    warned_m = false
    last_info = nothing

    while true
        niters += 1
        C = jacobi_interp_coeffs_Khat(K, nrow, mcol, alpha, beta)
        chop = _chop_coefficient_matrix(C; tol=tol, plateau_tol=plateau_tol,
                                        plateau_window=plateau_window,
                                        plateau_drop=plateau_drop,
                                        chop_guard=chop_guard)

        ncut_eff = chop.ncut
        mcut_eff = chop.mcut
        shape_happy = true
        if enforce_n_ge_m
            ncut_eff = max(ncut_eff, mcut_eff)
            shape_happy = ncut_eff <= nrow
        end

        if verbose
            @info "adaptive Jacobi Khat interpolation" nrow mcol chop.ncut chop.mcut ncut_eff mcut_eff chop.row_tail chop.col_tail chop.happy shape_happy
        end

        last_C = C
        last_info = chop

        if chop.happy && shape_happy
            converged = true
            last_C = Matrix{Float64}(@view C[1:ncut_eff, 1:mcut_eff])
            break
        end

        if niters >= maxiters
            @warn "adaptive Jacobi interpolation reached maxiters before meeting tolerance" niters nrow mcol chop.ncut chop.mcut ncut_eff mcut_eff chop.row_tail chop.col_tail tol
            ncut_last = min(ncut_eff, nrow)
            mcut_last = min(mcut_eff, mcol)
            last_C = Matrix{Float64}(@view C[1:ncut_last, 1:mcut_last])
            break
        end

        new_nrow = chop.row_happy && chop.row_safe && shape_happy ? nrow : max(nrow + 1, ceil(Int, growth * nrow))
        new_mcol = chop.col_happy && chop.col_safe ? mcol : max(mcol + 1, ceil(Int, growth * mcol))

        if enforce_n_ge_m
            new_nrow = max(new_nrow, new_mcol)
        end

        if warn_exceed_max
            if !warned_n && new_nrow > nmax
                @warn "adaptive Jacobi interpolation exceeded nmax; continuing because nmax is only a warning threshold" nmax new_nrow
                warned_n = true
            end
            if !warned_m && new_mcol > mmax
                @warn "adaptive Jacobi interpolation exceeded mmax; continuing because mmax is only a warning threshold" mmax new_mcol
                warned_m = true
            end
        end

        nrow, mcol = new_nrow, new_mcol
    end

    info = (
        nrow=size(last_C, 1),
        mcol=size(last_C, 2),
        tol=tol,
        converged=converged,
        iterations=niters,
        ncut=last_info.ncut,
        mcut=last_info.mcut,
        row_tail=last_info.row_tail,
        col_tail=last_info.col_tail,
        row_plateau=last_info.row_plateau,
        col_plateau=last_info.col_plateau,
        nmax_warning_threshold=Int(nmax),
        mmax_warning_threshold=Int(mmax),
        exceeded_nmax=size(last_C, 1) > nmax,
        exceeded_mmax=size(last_C, 2) > mmax,
        enforce_n_ge_m=enforce_n_ge_m,
        maxiters=Int(maxiters),
    )

    return last_C, info
end

# -----------------------------------------------------------------------------
# Norms and projection constants.
# -----------------------------------------------------------------------------

@inline function _jacobi_norm_log(n::Int, alpha::Float64, beta::Float64)
    return (alpha + beta + 1.0) * _LOG2 -
           log(2.0*n + alpha + beta + 1.0) +
           _loggamma_lanczos(n + alpha + 1.0) +
           _loggamma_lanczos(n + beta + 1.0) -
           _loggamma_lanczos(n + 1.0) -
           _loggamma_lanczos(n + alpha + beta + 1.0)
end

function _jacobi_norms(maxn::Int, alpha::Float64, beta::Float64)
    h = zeros(Float64, maxn + 1)
    @inbounds for n in 0:maxn
        h[n+1] = exp(_jacobi_norm_log(n, alpha, beta))
    end
    return h
end

@inline function _projection_scale(s::Int, r::Int, alpha::Float64, beta::Float64,
                                  norms::Vector{Float64})
    if r == 0
        return norms[s+1]
    else
        k = s + r
        logv = (alpha + beta) * _LOG2 +
               _loggamma_lanczos(alpha + s + 1.0) +
               _loggamma_lanczos(beta + k + 1.0) -
               _loggamma_lanczos(s + 1.0) -
               log(Float64(r)) -
               _loggamma_lanczos(alpha + beta + k + 1.0)
        return exp(logv)
    end
end

# -----------------------------------------------------------------------------
# Polynomial coefficients in z=(t+1)/2 for P_n^(alpha,beta)(2z-1).
# -----------------------------------------------------------------------------

function _poly_mul_t_minus_B(poly::Vector{Float64}, B::Float64)
    n = length(poly)
    out = zeros(Float64, n + 1)
    @inbounds for q in 0:n-1
        out[q+1] += -(1.0 + B) * poly[q+1]
        out[q+2] += 2.0 * poly[q+1]
    end
    return out
end

function _poly_axpy!(out::Vector{Float64}, a::Float64, p::Vector{Float64})
    @inbounds for i in eachindex(p)
        out[i] += a * p[i]
    end
    return nothing
end

function _jacobi_poly_coeff_z(n::Int, alpha::Float64, beta::Float64)
    if n == 0
        return [1.0]
    end
    P0 = [1.0]
    P1 = [-(beta + 1.0), alpha + beta + 2.0]
    if n == 1
        return P1
    end
    Pnm1 = P0
    Pn = P1
    for k in 1:n-1
        A, B, C = _jacobi_x_coeffs(k, alpha, beta)
        out = _poly_mul_t_minus_B(Pn, B)
        _poly_axpy!(out, -C, Pnm1)
        @inbounds for i in eachindex(out)
            out[i] /= A
        end
        Pnm1 = Pn
        Pn = out
    end
    return Pn
end

function _psi_coeff_z(s::Int, r::Int, alpha::Float64, beta::Float64)
    # Psi_{s,0}=z^s; for r>=1,
    # Psi_{s,r}=(t-1)z^s P_{r-1}^{(1, alpha+beta+2s+1)}(t).
    # Return coefficients c_nu of z^s sum c_nu z^nu.
    if r == 0
        return [1.0]
    end
    eta = alpha + beta + 2.0*s + 1.0
    q = _jacobi_poly_coeff_z(r - 1, 1.0, eta)
    coeff = zeros(Float64, r + 1)
    @inbounds for nu in 0:r
        qm1 = nu >= 1 ? q[nu] : 0.0
        q0 = nu <= r - 1 ? q[nu+1] : 0.0
        coeff[nu+1] = 2.0 * (qm1 - q0)
    end
    return coeff
end

@inline function _jacobi_x_coeffs_big(n::Int, alpha::BigFloat, beta::BigFloat)
    if n == 0
        A = BigFloat(2) / (alpha + beta + BigFloat(2))
        B = (beta - alpha) / (alpha + beta + BigFloat(2))
        C = BigFloat(0)
        return A, B, C
    end

    nb = BigFloat(n)
    s = alpha + beta

    A = BigFloat(2) * (nb + BigFloat(1)) * (nb + s + BigFloat(1)) /
        ((BigFloat(2)*nb + s + BigFloat(1)) * (BigFloat(2)*nb + s + BigFloat(2)))

    B = (beta^2 - alpha^2) /
        ((BigFloat(2)*nb + s) * (BigFloat(2)*nb + s + BigFloat(2)))

    C = BigFloat(2) * (nb + alpha) * (nb + beta) /
        ((BigFloat(2)*nb + s) * (BigFloat(2)*nb + s + BigFloat(1)))

    return A, B, C
end


function _poly_mul_t_minus_B_big(poly::Vector{BigFloat}, B::BigFloat)
    n = length(poly)
    out = fill(BigFloat(0), n + 1)

    @inbounds for q in 0:n-1
        out[q+1] += -(BigFloat(1) + B) * poly[q+1]
        out[q+2] += BigFloat(2) * poly[q+1]
    end

    return out
end


function _poly_axpy_big!(out::Vector{BigFloat}, a::BigFloat, p::Vector{BigFloat})
    @inbounds for i in eachindex(p)
        out[i] += a * p[i]
    end
    return nothing
end


function _jacobi_poly_coeff_z_big(n::Int, alpha::BigFloat, beta::BigFloat)
    if n == 0
        return BigFloat[1]
    end

    P0 = BigFloat[1]
    P1 = BigFloat[-(beta + BigFloat(1)), alpha + beta + BigFloat(2)]

    if n == 1
        return P1
    end

    Pnm1 = P0
    Pn = P1

    for k in 1:n-1
        A, B, C = _jacobi_x_coeffs_big(k, alpha, beta)
        out = _poly_mul_t_minus_B_big(Pn, B)
        _poly_axpy_big!(out, -C, Pnm1)

        @inbounds for i in eachindex(out)
            out[i] /= A
        end

        Pnm1 = Pn
        Pn = out
    end

    return Pn
end


function _psi_coeff_z_big(s::Int, r::Int, alpha::BigFloat, beta::BigFloat)
    if r == 0
        return BigFloat[1]
    end

    eta = alpha + beta + BigFloat(2)*BigFloat(s) + BigFloat(1)
    q = _jacobi_poly_coeff_z_big(r - 1, BigFloat(1), eta)

    coeff = fill(BigFloat(0), r + 1)

    @inbounds for nu in 0:r
        qm1 = nu >= 1 ? q[nu] : BigFloat(0)
        q0 = nu <= r - 1 ? q[nu+1] : BigFloat(0)
        coeff[nu+1] = BigFloat(2) * (qm1 - q0)
    end

    return coeff
end


function _initial_J_rows!(J0::Vector{Float64}, J1::Vector{Float64},
                          s::Int, R0::Int, alpha::Float64, beta::Float64)
    # Fast stable initialization of the first two j-layers for the band-tail
    # G recurrence.
    #
    # Ψ_{s,0}(t) = z^s,
    # Ψ_{s,r}(t) = (t-1)z^s P_{r-1}^{(1,α+β+2s+1)}(t), r>=1,
    # z=(t+1)/2.
    #
    # J0[r+1] = ∫ Ψ_{s,r}(t) dt,
    # J1[r+1] = ∫ P_1^{(α,β)}(t) Ψ_{s,r}(t) dt.
    #
    # Since P_1^{(α,β)}(t)=(α+β+2)z-(β+1), set
    # M[r+1] = ∫ z Ψ_{s,r}(t) dt and use
    # J1 = (α+β+2)M - (β+1)J0.
    #
    # The scalar moments J0 and M satisfy stable adjacent-ratio recurrences,
    # avoiding monomial expansions and BigFloat.
    lambda = alpha + beta

    fill!(J0, 0.0)
    fill!(J1, 0.0)

    # r = 0
    J0_r = 2.0 / (s + 1.0)
    M_r  = 2.0 / (s + 2.0)

    J0[1] = J0_r
    J1[1] = (lambda + 2.0) * M_r - (beta + 1.0) * J0_r

    if R0 == 0
        return nothing
    end

    # r = 1
    J0_r = -4.0 / ((s + 1.0) * (s + 2.0))
    M_r  = -4.0 / ((s + 2.0) * (s + 3.0))

    J0[2] = J0_r
    J1[2] = (lambda + 2.0) * M_r - (beta + 1.0) * J0_r

    # r >= 2
    @inbounds for r in 2:R0
        rr = Float64(r)

        ratio_J0 =
            - (rr / (rr - 1.0)) *
            ((lambda + s + rr - 1.0) / (s + rr + 1.0))

        ratio_M =
            - (rr / (rr - 1.0)) *
            ((lambda + s + rr - 2.0) / (s + rr + 2.0))

        J0_r *= ratio_J0
        M_r  *= ratio_M

        J0[r+1] = J0_r
        J1[r+1] = (lambda + 2.0) * M_r - (beta + 1.0) * J0_r
    end

    return nothing
end

function _initial_J_rows_loggamma!(J0::Vector{Float64}, J1::Vector{Float64},
                                   s::Int, R0::Int,
                                   alpha::Float64, beta::Float64)
    # Same quantities as _initial_J_rows!, but the r>=1 entries are evaluated
    # from closed-form gamma ratios.  This is useful for alpha+beta>2, where
    # repeatedly multiplying adjacent ratios may unnecessarily amplify
    # intermediate roundoff in long initial layers.
    lambda = alpha + beta
    lambda > 2.0 || throw(ArgumentError(
        "log-gamma initialization requires alpha + beta > 2; got alpha + beta = $lambda."
    ))

    fill!(J0, 0.0)
    fill!(J1, 0.0)

    J0[1] = 2.0 / (s + 1.0)
    M0 = 2.0 / (s + 2.0)
    J1[1] = (lambda + 2.0) * M0 - (beta + 1.0) * J0[1]

    @inbounds for r in 1:R0
        rr = Float64(r)
        sign = isodd(r) ? -1.0 : 1.0

        logJ0 = log(4.0 * rr) +
                _loggamma_lanczos(s + 1.0) +
                _loggamma_lanczos(lambda + s + rr) -
                _loggamma_lanczos(lambda + s + 1.0) -
                _loggamma_lanczos(s + rr + 2.0)

        logM = log(4.0 * rr) +
               _loggamma_lanczos(s + 2.0) +
               _loggamma_lanczos(lambda + s + rr - 1.0) -
               _loggamma_lanczos(lambda + s) -
               _loggamma_lanczos(s + rr + 3.0)

        J0r = sign * exp(logJ0)
        Mr = sign * exp(logM)
        J0[r+1] = J0r
        J1[r+1] = (lambda + 2.0) * Mr - (beta + 1.0) * J0r
    end

    return nothing
end

function _initial_J_rows_dispatch!(J0::Vector{Float64}, J1::Vector{Float64},
                                   s::Int, R0::Int,
                                   alpha::Float64, beta::Float64,
                                   init_layer_backend::Symbol)
    if init_layer_backend == :loggamma
        return _initial_J_rows_loggamma!(J0, J1, s, R0, alpha, beta)
    elseif init_layer_backend == :recurrence
        return _initial_J_rows!(J0, J1, s, R0, alpha, beta)
    else
        throw(ArgumentError("resolved init_layer_backend must be :recurrence or :loggamma."))
    end
end

@inline function _psi_x_coeffs(s::Int, r::Int, alpha::Float64, beta::Float64)
    if r == 0
        return 1.0, 1.0, 0.0
    else
        rho = r - 1
        eta = alpha + beta + 2.0*s + 1.0
        A, B, C = _jacobi_x_coeffs(rho, 1.0, eta)
        return A, B, C
    end
end

function _fill_G_for_s!(G::Matrix{Float64},
                        Jm1::Vector{Float64},
                        Jcur::Vector{Float64},
                        Jnext::Vector{Float64},
                        s::Int,
                        jmax::Int,
                        Rneed::Int,
                        alpha::Float64,
                        beta::Float64,
                        norms::Vector{Float64},
                        init_layer_backend::Symbol)
    R0 = Rneed + jmax
    _initial_J_rows_dispatch!(Jm1, Jcur, s, R0, alpha, beta, init_layer_backend)
    @inbounds for r in 0:Rneed
        G[1, r+1] = _projection_scale(s, r, alpha, beta, norms) * Jm1[r+1]
    end
    if jmax == 0
        return nothing
    end
    @inbounds for r in 0:Rneed
        G[2, r+1] = _projection_scale(s, r, alpha, beta, norms) * Jcur[r+1]
    end
    @inbounds for j in 1:jmax-1
        Aleft, Bleft, Cleft = _jacobi_x_coeffs(j, alpha, beta)
        nextmax = R0 - j - 1
        for r in 0:nextmax
            Ar, Br, Dr = _psi_x_coeffs(s, r, alpha, beta)
            rhs = Ar * Jcur[r+2] + (Br - Bleft) * Jcur[r+1]
            if r >= 1
                rhs += Dr * Jcur[r]
            end
            rhs -= Cleft * Jm1[r+1]
            Jnext[r+1] = rhs / Aleft
        end
        for r in 0:Rneed
            G[j+2, r+1] = _projection_scale(s, r, alpha, beta, norms) * Jnext[r+1]
        end
        Jm1, Jcur, Jnext = Jcur, Jnext, Jm1
    end
    return nothing
end


# -----------------------------------------------------------------------------
# Stable moments backend for G.
#
# This backend avoids the forward j-recurrence.  It computes
#
#   G[j+1,r+1] = γ_{s,r} ∫ P_j^{(α,β)}(t) Ψ_{s,r}(t) dt
#
# directly by moments in z=(t+1)/2.  If
#
#   P_j^{(α,β)}(2z-1) = sum_μ d_μ z^μ,
#   Ψ_{s,r}(2z-1) = z^s sum_ν e_ν z^ν,
#
# then
#
#   ∫ P_j Ψ dt = sum_{μ,ν} d_μ e_ν 2/(s+μ+ν+1).
# -----------------------------------------------------------------------------

function _precompute_jacobi_poly_coeffs_z(mdeg::Int, alpha::Float64, beta::Float64)
    return [_jacobi_poly_coeff_z(j, alpha, beta) for j in 0:mdeg]
end


function _fill_G_for_s_moments!(
    G::Matrix{Float64},
    pj_coeffs::Vector{Vector{Float64}},
    s::Int,
    jmax::Int,
    Rneed::Int,
    alpha::Float64,
    beta::Float64,
    norms::Vector{Float64},
)
    fill!(G, 0.0)

    @inbounds for r in 0:Rneed
        psi = _psi_coeff_z(s, r, alpha, beta)
        scale = _projection_scale(s, r, alpha, beta, norms)

        for j in 0:jmax
            pj = pj_coeffs[j+1]

            acc = 0.0
            comp = 0.0

            for mu in 0:length(pj)-1
                dμ = pj[mu+1]

                for nu in 0:length(psi)-1
                    eν = psi[nu+1]
                    term = dμ * eν * 2.0 / (s + mu + nu + 1.0)

                    # Kahan compensation.
                    y = term - comp
                    tmp = acc + y
                    comp = (tmp - acc) - y
                    acc = tmp
                end
            end

            G[j+1, r+1] = scale * acc
        end
    end

    return nothing
end


# -----------------------------------------------------------------------------
# Compressed CG tables: CG_s(i,r)=sum_j C_ij G_{j,s+r}^s.
# -----------------------------------------------------------------------------

struct CGCache
    tables::Vector{Matrix{Float64}}
    N::Int
end

@inline function _cg_get(cache::CGCache, i::Int, r::Int, s::Int)
    if 0 <= s <= cache.N
        T = cache.tables[s+1]
        if 0 <= r < size(T, 2)
            return T[i+1, r+1]
        end
    end
    return 0.0
end

function _precompute_CG_tables(C::AbstractMatrix{<:Real},
                               N::Int,
                               alpha::Float64,
                               beta::Float64,
                               mdeg::Int,
                               Rneed::Int,
                               norms::Vector{Float64};
                               G_backend::Symbol=:recurrence,
                               init_layer_backend::Symbol=:recurrence)
    G_backend in (:moments, :recurrence) ||
        throw(ArgumentError("G_backend must be :moments or :recurrence."))

    Cmat = Matrix{Float64}(C)
    nrow = size(Cmat, 1)

    tables = Vector{Matrix{Float64}}(undef, N + 1)
    Gbuf = zeros(Float64, mdeg + 1, Rneed + 1)

    # Workspace for recurrence backend.
    R0max = Rneed + mdeg
    Jm1 = zeros(Float64, R0max + 2)
    Jcur = zeros(Float64, R0max + 2)
    Jnext = zeros(Float64, R0max + 2)

    # Workspace for moments backend.
    pj_coeffs = _precompute_jacobi_poly_coeffs_z(mdeg, alpha, beta)

    @inbounds for s in 0:N
        Rs = min(Rneed, N - s)

        if G_backend == :moments
            _fill_G_for_s_moments!(
                Gbuf,
                pj_coeffs,
                s,
                mdeg,
                Rs,
                alpha,
                beta,
                norms,
            )
        else
            _fill_G_for_s!(
                Gbuf,
                Jm1,
                Jcur,
                Jnext,
                s,
                mdeg,
                Rs,
                alpha,
                beta,
                norms,
                init_layer_backend,
            )
        end

        Gview = @view Gbuf[:, 1:Rs+1]
        CG = Matrix{Float64}(undef, nrow, Rs + 1)
        mul!(CG, Cmat, Gview)
        tables[s+1] = CG
    end

    return CGCache(tables, N)
end

# -----------------------------------------------------------------------------
# Product linearization p_i*p_ell = sum_s L_s p_s by recurrence.
# -----------------------------------------------------------------------------

struct LocalPoly
    offset::Int
    coeff::Vector{Float64}
end

struct LinTerms
    s::Vector{Int}
    c::Vector{Float64}
end

function _apply_x_local(poly::LocalPoly, alpha::Float64, beta::Float64)
    lo = poly.offset
    n = length(poly.coeff)
    outlo = max(lo - 1, 0)
    out = zeros(Float64, n + 2)
    @inbounds for idx in 1:n
        q = lo + idx - 1
        val = poly.coeff[idx]
        A, B, C = _jacobi_x_coeffs(q, alpha, beta)
        out[q + 1 - outlo + 1] += A * val
        out[q - outlo + 1] += B * val
        if q >= 1
            out[q - 1 - outlo + 1] += C * val
        end
    end
    return LocalPoly(outlo, out)
end

function _combine_next(cur::LocalPoly, prev::LocalPoly,
                       i::Int, alpha::Float64, beta::Float64)
    A, B, C = _jacobi_x_coeffs(i, alpha, beta)
    xcur = _apply_x_local(cur, alpha, beta)
    lo = min(xcur.offset, cur.offset, prev.offset)
    hi = max(xcur.offset + length(xcur.coeff) - 1,
             cur.offset + length(cur.coeff) - 1,
             prev.offset + length(prev.coeff) - 1)
    out = zeros(Float64, hi - lo + 1)
    @inbounds for idx in eachindex(xcur.coeff)
        out[xcur.offset + idx - 1 - lo + 1] += xcur.coeff[idx]
    end
    @inbounds for idx in eachindex(cur.coeff)
        out[cur.offset + idx - 1 - lo + 1] -= B * cur.coeff[idx]
    end
    @inbounds for idx in eachindex(prev.coeff)
        out[prev.offset + idx - 1 - lo + 1] -= C * prev.coeff[idx]
    end
    @inbounds for idx in eachindex(out)
        out[idx] /= A
    end
    return LocalPoly(lo, out)
end

function _to_linterms(poly::LocalPoly; droptol::Float64=0.0)
    svec = Int[]
    cvec = Float64[]
    @inbounds for idx in eachindex(poly.coeff)
        val = poly.coeff[idx]
        if abs(val) > droptol
            push!(svec, poly.offset + idx - 1)
            push!(cvec, val)
        end
    end
    return LinTerms(svec, cvec)
end

function _linearization_cache(maxell::Int, ndeg::Int,
                              alpha::Float64, beta::Float64;
                              droptol::Float64=0.0)
    cache = Vector{Vector{LinTerms}}(undef, maxell + 1)
    @inbounds for ell in 0:maxell
        cell = Vector{LinTerms}(undef, ndeg + 1)
        prod_prev = LocalPoly(ell, [1.0])
        cell[1] = _to_linterms(prod_prev; droptol=droptol)
        if ndeg >= 1
            A0, B0, C0 = _jacobi_x_coeffs(0, alpha, beta)
            xprod = _apply_x_local(prod_prev, alpha, beta)
            coeff = copy(xprod.coeff)
            for idx in eachindex(prod_prev.coeff)
                q = prod_prev.offset + idx - 1
                coeff[q - xprod.offset + 1] -= B0 * prod_prev.coeff[idx]
            end
            for idx in eachindex(coeff)
                coeff[idx] /= A0
            end
            prod_cur = LocalPoly(xprod.offset, coeff)
            cell[2] = _to_linterms(prod_cur; droptol=droptol)
            for i in 1:ndeg-1
                prod_next = _combine_next(prod_cur, prod_prev, i, alpha, beta)
                cell[i+2] = _to_linterms(prod_next; droptol=droptol)
                prod_prev, prod_cur = prod_cur, prod_next
            end
        end
        cache[ell+1] = cell
    end
    return cache
end

# -----------------------------------------------------------------------------
# Phi and assembly.
# -----------------------------------------------------------------------------

@inline function _phi_entry_jacobi(CG::CGCache,
                                   lin_cache::Vector{Vector{LinTerms}},
                                   norms::Vector{Float64},
                                   k::Int,
                                   dphi::Int,
                                   ndeg::Int,
                                   maxell::Int)
    ell = k + dphi
    if ell < 0 || ell > maxell
        return 0.0
    end
    val = 0.0
    @inbounds for i in 0:ndeg
        terms = lin_cache[ell+1][i+1]
        for idx in eachindex(terms.s)
            s = terms.s[idx]
            r = k - s
            if r >= 0
                val += terms.c[idx] * _cg_get(CG, i, r, s)
            end
        end
    end
    return val / norms[ell+1]
end

@inline function _get_phival(phivals::Vector{Float64}, d::Int,
                             dphi_min::Int, dphi_max::Int)
    return (d >= dphi_min && d <= dphi_max) ? phivals[d - dphi_min + 1] : 0.0
end

function _assemble_symband_jacobi(CG::CGCache,
                                  lin_cache::Vector{Vector{LinTerms}},
                                  norms::Vector{Float64},
                                  N::Int,
                                  bw::Int,
                                  dphi_min::Int,
                                  dphi_max::Int,
                                  maxell_phi::Int,
                                  alpha::Float64,
                                  beta::Float64,
                                  dropzeros_tol::Float64)
    ndphi = dphi_max - dphi_min + 1
    rowval = Int[]
    nzval = Float64[]
    colptr = Vector{Int}(undef, N + 2)
    colptr[1] = 1
    sizehint!(rowval, (N + 1) * (2bw + 1))
    sizehint!(nzval, (N + 1) * (2bw + 1))
    phivals = zeros(Float64, ndphi)
    ndeg = length(lin_cache[1]) - 1
    @inbounds for k in 0:N
        fill!(phivals, 0.0)
        for dphi in dphi_min:dphi_max
            ellphi = k + dphi
            if 0 <= ellphi <= maxell_phi
                phivals[dphi - dphi_min + 1] =
                    _phi_entry_jacobi(CG, lin_cache, norms, k, dphi, ndeg, maxell_phi)
            end
        end
        d_start = max(-bw, -k)
        d_stop = min(bw, N - k)
        for d in d_start:d_stop
            ell = k + d
            val = 0.5 * _get_phival(phivals, d, dphi_min, dphi_max)
            if ell >= 1
                Aprev, Bprev, Cprev = _jacobi_x_coeffs(ell - 1, alpha, beta)
                val += 0.5 * Aprev * _get_phival(phivals, d - 1, dphi_min, dphi_max)
            end
            Aell, Bell, Cell = _jacobi_x_coeffs(ell, alpha, beta)
            val += 0.5 * Bell * _get_phival(phivals, d, dphi_min, dphi_max)
            Aup, Bup, Cup = _jacobi_x_coeffs(ell + 1, alpha, beta)
            val += 0.5 * Cup * _get_phival(phivals, d + 1, dphi_min, dphi_max)
            if abs(val) > dropzeros_tol
                push!(rowval, ell + 1)
                push!(nzval, val)
            end
        end
        colptr[k+2] = length(rowval) + 1
    end
    return SparseMatrixCSC(N + 1, N + 1, colptr, rowval, nzval)
end


# -----------------------------------------------------------------------------
# Dense top rows by the (s,k)-recurrence.
#
# This constructs the dense top block of the Jacobi-Volterra matrix.  The number
# of top rows defaults to size(C,1), matching the default symmetric bandwidth.
#
# We use the unscaled auxiliary quantity
#
#   T_{j,k}^s = ∫ P_j^{(α,β)}(t)
#                  ((1+t)/2)^s P_{k-s}^{(-1,α+β+2s+1)}(t) dt.
#
# It satisfies, with λ=α+β,
#
#   (k+s+λ+1) T_{j,k}^{s+1} + (k-s-2)T_{j,k-1}^{s+1}
#     = (k-s)T_{j,k}^{s} + (k+s+λ+1)T_{j,k-1}^{s}.
#
# The first layer T^0 is computed explicitly from monomial moments in BigFloat.
# The actual projected quantity is
#
#   H_{j,k}^s = ρ_{s,k} T_{j,k}^s,
#
# where ρ_{s,k} converts the (-1,α+β+2s+1) representation to the Jacobi
# projection V_{s,k}^{(α,β)}.
# -----------------------------------------------------------------------------

@inline function _rho_tilde_scale(
    s::Int,
    k::Int,
    alpha::Float64,
    beta::Float64,
    norms::Vector{Float64},
)
    k < s && return 0.0

    r = k - s

    if r == 0
        return norms[s+1]
    end

    eta = alpha + beta + 2.0*s + 1.0

    # _projection_scale is the coefficient of
    #   (t-1) z^s P_{r-1}^{(1,eta)}.
    # Since
    #   P_r^{(-1,eta)} = (r+eta)/(2r) (t-1)P_{r-1}^{(1,eta)},
    # the coefficient of z^s P_r^{(-1,eta)} is
    #   _projection_scale * 2r/(r+eta).
    return _projection_scale(s, r, alpha, beta, norms) * (2.0*r) / (r + eta)
end


@inline function _q_x_coeffs_minus1(k::Int, lambda::Float64)
    # Q_k(t) = P_k^{(-1,lambda+1)}(t).
    #
    # t Q_k = A Q_{k+1} + B Q_k + C Q_{k-1}.
    if k == 0
        A = 2.0 / (lambda + 2.0)
        B = 1.0
        C = 0.0
        return A, B, C
    end

    kk = Float64(k)

    A =
        2.0 * (kk + 1.0) * (kk + lambda + 1.0) /
        ((2.0*kk + lambda + 1.0) * (2.0*kk + lambda + 2.0))

    B =
        ((lambda + 1.0)^2 - 1.0) /
        ((2.0*kk + lambda) * (2.0*kk + lambda + 2.0))

    C =
        2.0 * (kk - 1.0) * (kk + lambda + 1.0) /
        ((2.0*kk + lambda) * (2.0*kk + lambda + 1.0))

    return A, B, C
end


function _tilde_H0_mixed!(
    H0::Matrix{Float64},
    jmax::Int,
    N::Int,
    alpha::Float64,
    beta::Float64;
    init_layer_backend::Symbol=:recurrence,
)
    # Computes
    #
    #   H0[j+1,k+1] = ∫_{-1}^1 P_j^{(α,β)}(t)
    #                         P_k^{(-1,α+β+1)}(t) dt
    #
    # for j=0,...,jmax and k=0,...,N in O(jmax*N) time.
    #
    # The recurrence uses
    #   ∫(t P_j)Q_k dt = ∫P_j(t Q_k)dt,
    # where Q_k=P_k^{(-1,α+β+1)}.
    #
    # Extra columns are needed because row j+1 depends on row j at k+1.
    lambda = alpha + beta
    Kmax = N + jmax

    T = zeros(Float64, jmax + 1, Kmax + 1)

    # ------------------------------------------------------------
    # Row j = 0: I_k = ∫ Q_k(t) dt.
    # ------------------------------------------------------------
    I = @view T[1, :]

    I[1] = 2.0

    if init_layer_backend == :loggamma
        lambda > 2.0 || throw(ArgumentError(
            "log-gamma initialization requires alpha + beta > 2; got alpha + beta = $lambda."
        ))
        @inbounds for k in 1:Kmax
            kk = Float64(k)
            logIk = log(2.0) +
                    _loggamma_lanczos(kk + lambda + 2.0) -
                    log(kk + lambda) -
                    _loggamma_lanczos(lambda + 1.0) -
                    _loggamma_lanczos(kk + 2.0)
            I[k+1] = (isodd(k) ? -1.0 : 1.0) * exp(logIk)
        end
    elseif init_layer_backend == :recurrence
        if Kmax >= 1
            I[2] = -(lambda + 2.0)
        end

        @inbounds for k in 2:Kmax
            kk = Float64(k)
            I[k+1] =
                -((kk - 1.0 + lambda) * (kk + 1.0 + lambda)) /
                 ((kk + lambda) * (kk + 1.0)) * I[k]
        end
    else
        throw(ArgumentError("resolved init_layer_backend must be :recurrence or :loggamma."))
    end

    # ------------------------------------------------------------
    # Row j = 1.
    # P_1^{(α,β)}(t) = ((α+β+2)/2)t + ((α-β)/2).
    # ------------------------------------------------------------
    if jmax >= 1
        @inbounds for k in 0:(Kmax - 1)
            Aq, Bq, Cq = _q_x_coeffs_minus1(k, lambda)

            Im1 = k >= 1 ? I[k] : 0.0
            Ik  = I[k+1]
            Ip1 = I[k+2]

            T[2, k+1] =
                0.5 * (lambda + 2.0) *
                (Aq * Ip1 + Bq * Ik + Cq * Im1) +
                0.5 * (alpha - beta) * Ik
        end
    end

    # ------------------------------------------------------------
    # Rows j >= 2 by mixed recurrence.
    # ------------------------------------------------------------
    if jmax >= 2
        @inbounds for j in 1:(jmax - 1)
            Aj, Bj, Cj = _jacobi_x_coeffs(j, alpha, beta)

            # Row j+1 only needs k up to Kmax-j-1.
            kmax_row = Kmax - j - 1

            for k in 0:kmax_row
                Aq, Bq, Cq = _q_x_coeffs_minus1(k, lambda)

                left_km1 = k >= 1 ? T[j+1, k] : 0.0

                T[j+2, k+1] =
                    (
                        Aq * T[j+1, k+2] +
                        (Bq - Bj) * T[j+1, k+1] +
                        Cq * left_km1 -
                        Cj * T[j, k+1]
                    ) / Aj
            end
        end
    end

    # Copy only k=0,...,N to output.
    @inbounds for j in 0:jmax
        for k in 0:N
            H0[j+1, k+1] = T[j+1, k+1]
        end
    end

    return nothing
end


function _tilde_H0_big!(
    H0::Matrix{Float64},
    jmax::Int,
    N::Int,
    alpha::Float64,
    beta::Float64;
    init_precision::Integer=256,
    init_layer_backend::Symbol=:recurrence,
)
    # H0[j+1,k+1] = ∫ P_j^{(α,β)}(t) P_k^{(-1,α+β+1)}(t) dt.
    #
    # The monomial coefficients can be strongly cancelling, so this whole
    # initialization is done in BigFloat.
    setprecision(BigFloat, Int(init_precision)) do
        abig = BigFloat(alpha)
        bbig = BigFloat(beta)
        lambdabig = abig + bbig

        pj_coeffs = [_jacobi_poly_coeff_z_big(j, abig, bbig) for j in 0:jmax]

        @inbounds for k in 0:N
            qk = _jacobi_poly_coeff_z_big(k, -BigFloat(1), lambdabig + BigFloat(1))

            for j in 0:jmax
                pj = pj_coeffs[j+1]

                acc = BigFloat(0)

                for mu in 0:length(pj)-1
                    dμ = pj[mu+1]

                    for nu in 0:length(qk)-1
                        eν = qk[nu+1]
                        acc += dμ * eν * BigFloat(2) / BigFloat(mu + nu + 1)
                    end
                end

                H0[j+1, k+1] = Float64(acc)
            end
        end
    end

    return nothing
end


function _build_tilde_H_layers_sk(
    jmax::Int,
    N::Int,
    Smax::Int,
    alpha::Float64,
    beta::Float64;
    init_precision::Integer=256,
    init_layer_backend::Symbol=:recurrence,
)
    lambda = alpha + beta

    layers = Vector{Matrix{Float64}}(undef, Smax + 1)

    Hcur = zeros(Float64, jmax + 1, N + 1)
    _tilde_H0_mixed!(Hcur, jmax, N, alpha, beta;
                     init_layer_backend=init_layer_backend)
    layers[1] = copy(Hcur)

    for s in 0:Smax-1
        Hnext = zeros(Float64, jmax + 1, N + 1)

        @inbounds for k in s+1:N
            A = k + s + lambda + 1.0

            for j in 0:jmax
                Hnext[j+1, k+1] =
                    ((k - s) * Hcur[j+1, k+1] +
                     A * Hcur[j+1, k] -
                     (k - s - 2) * Hnext[j+1, k]) / A
            end
        end

        layers[s+2] = Hnext
        Hcur = Hnext
    end

    return layers
end


function _precompute_CH_top_sk(
    C::AbstractMatrix{<:Real},
    N::Int,
    alpha::Float64,
    beta::Float64,
    Smax::Int,
    norms::Vector{Float64};
    init_precision::Integer=256,
    init_layer_backend::Symbol=:recurrence,
)
    Cmat = Matrix{Float64}(C)
    mdeg = size(Cmat, 2) - 1

    tilde_layers = _build_tilde_H_layers_sk(
        mdeg,
        N,
        Smax,
        alpha,
        beta;
        init_precision=init_precision,
        init_layer_backend=init_layer_backend,
    )

    CH = Vector{Matrix{Float64}}(undef, Smax + 1)

    @inbounds for s in 0:Smax
        T = tilde_layers[s+1]
        Hfull = zeros(Float64, size(T, 1), size(T, 2))

        for k in s:N
            scale = _rho_tilde_scale(s, k, alpha, beta, norms)

            for j in 0:mdeg
                Hfull[j+1, k+1] = scale * T[j+1, k+1]
            end
        end

        CH[s+1] = Cmat * Hfull
    end

    return CH
end


function jacobi_volterra_topdense_from_C(
    C::AbstractMatrix{<:Real},
    N::Integer,
    alpha::Real=0.0,
    beta::Real=0.0;
    n_top::Union{Nothing,Integer}=nothing,
    init_precision::Integer=256,  # retained for API compatibility
    lin_droptol::Float64=0.0,
    init_layer_backend::Symbol=:auto,
    return_phi::Bool=false,
)
    alpha, beta = _check_ab(alpha, beta)
    N < 0 && throw(ArgumentError("N must be nonnegative."))

    Nint = Int(N)
    nrow, mcol = _check_C_n_ge_m(C)
    ndeg = nrow - 1

    ntop = _is_legendre_case(alpha, beta) ? 0 :
           (n_top === nothing ? _default_top_rows(nrow, mcol, alpha, beta, Nint) : Int(n_top))
    ntop < 0 && throw(ArgumentError("n_top must be nonnegative."))
    ntop = min(ntop, Nint + 1)

    if ntop == 0
        Btop = zeros(Float64, 0, Nint + 1)
        Phi = zeros(Float64, 0, Nint + 1)
        return return_phi ? (Btop, Phi) : Btop
    end

    init_backend = _resolve_init_layer_backend(init_layer_backend, alpha, beta)

    # To compute top rows ell=0,...,ntop-1 of B=MΦ, we need Φ rows
    # ell=0,...,ntop because M is tridiagonal.
    phimax = ntop

    # Product P_i P_ell with i<=nrow-1 and ell<=phimax needs
    # s<=i+ell<=ndeg+phimax.
    Smax = ndeg + phimax

    maxnorm = max(Nint, phimax, Smax) + 2
    norms = _jacobi_norms(maxnorm, alpha, beta)

    CH = _precompute_CH_top_sk(
        C,
        Nint,
        alpha,
        beta,
        Smax,
        norms;
        init_precision=init_precision,
        init_layer_backend=init_backend,
    )

    lin_cache = _linearization_cache(phimax, ndeg, alpha, beta; droptol=lin_droptol)

    Phi = zeros(Float64, phimax + 1, Nint + 1)

    @inbounds for ell in 0:phimax
        for k in 0:Nint
            val = 0.0

            for i in 0:ndeg
                terms = lin_cache[ell+1][i+1]

                for idx in eachindex(terms.s)
                    s = terms.s[idx]

                    if 0 <= s <= Smax
                        val += terms.c[idx] * CH[s+1][i+1, k+1]
                    end
                end
            end

            Phi[ell+1, k+1] = val / norms[ell+1]
        end
    end

    Btop = zeros(Float64, ntop, Nint + 1)

    @inbounds for ell in 0:ntop-1
        Aell, Bell, Cell = _jacobi_x_coeffs(ell, alpha, beta)

        for k in 0:Nint
            val = 0.5 * (1.0 + Bell) * Phi[ell+1, k+1]

            if ell >= 1
                Aprev, Bprev, Cprev = _jacobi_x_coeffs(ell - 1, alpha, beta)
                val += 0.5 * Aprev * Phi[ell, k+1]
            end

            Aup, Bup, Cup = _jacobi_x_coeffs(ell + 1, alpha, beta)
            val += 0.5 * Cup * Phi[ell+2, k+1]

            Btop[ell+1, k+1] = val
        end
    end

    return return_phi ? (Btop, Phi) : Btop
end


# -----------------------------------------------------------------------------
# Main constructor from known C.
# -----------------------------------------------------------------------------

function voltmat_jacobi(C::AbstractMatrix{<:Real},
                        N::Integer,
                        alpha::Real=0.0,
                        beta::Real=0.0;
                        bandwidth::Union{Nothing,Integer}=nothing,
                        dropzeros_tol::Float64=0.0,
                        lin_droptol::Float64=0.0,
                        G_backend::Symbol=:recurrence,
                        init_layer_backend::Symbol=:auto,
                        return_info::Bool=false)
    alpha, beta = _check_ab(alpha, beta)
    N < 0 && throw(ArgumentError("N must be nonnegative."))
    Nint = Int(N)
    nrow, mcol = _check_C_n_ge_m(C)
    ndeg = nrow - 1
    mdeg = mcol - 1
    bw = bandwidth === nothing ? nrow : Int(bandwidth)
    bw < 0 && throw(ArgumentError("bandwidth must be nonnegative."))
    bw = min(bw, Nint)
    dphi_min = -bw - 1
    dphi_max = min(bw + 1, ndeg)
    Rneed = ndeg - dphi_min
    maxell_phi = min(Nint + ndeg, Nint + 1)
    maxnorm = max(maxell_phi, Nint)
    norms = _jacobi_norms(maxnorm, alpha, beta)
    init_backend = _resolve_init_layer_backend(init_layer_backend, alpha, beta)
    CG = _precompute_CG_tables(C, Nint, alpha, beta, mdeg, Rneed, norms;
                               G_backend=G_backend,
                               init_layer_backend=init_backend)
    lin_cache = _linearization_cache(maxell_phi, ndeg, alpha, beta; droptol=lin_droptol)
    B = _assemble_symband_jacobi(CG, lin_cache, norms, Nint, bw,
                                  dphi_min, dphi_max, maxell_phi,
                                  alpha, beta, dropzeros_tol)
    info = (
        alpha=alpha,
        beta=beta,
        nrow=nrow,
        mcol=mcol,
        bandwidth=bw,
        Rneed=Rneed,
        G_backend=G_backend,
        init_layer_backend=init_backend,
        matrix_size=size(B),
        nnz=nnz(B),
    )
    return return_info ? (B, info) : B
end

# -----------------------------------------------------------------------------
# Main constructor from analytic K.
# -----------------------------------------------------------------------------

function voltmat_jacobi(K::Function,
                        N::Integer,
                        alpha::Real=0.0,
                        beta::Real=0.0;
                        tol::Float64=1.0e-12,
                        nmin::Integer=8,
                        mmin::Integer=8,
                        nmax::Integer=128,
                        mmax::Integer=128,
                        growth::Real=1.5,
                        maxiters::Integer=50,
                        warn_exceed_max::Bool=true,
                        plateau_tol::Union{Nothing,Float64}=nothing,
                        plateau_window::Integer=8,
                        plateau_drop::Float64=0.1,
                        chop_guard::Integer=4,
                        bandwidth::Union{Nothing,Integer}=nothing,
                        dropzeros_tol::Float64=0.0,
                        lin_droptol::Float64=0.0,
                        G_backend::Symbol=:recurrence,
                        init_layer_backend::Symbol=:auto,
                        enforce_n_ge_m::Bool=true,
                        return_C::Bool=false,
                        return_info::Bool=false,
                        verbose::Bool=false)
    alpha, beta = _check_ab(alpha, beta)
    N < 0 && throw(ArgumentError("N must be nonnegative."))
    C, interp_info = adaptive_jacobi_interp_coeffs_Khat(
        K,
        alpha,
        beta;
        tol=tol,
        nmin=nmin,
        mmin=mmin,
        nmax=nmax,
        mmax=mmax,
        growth=growth,
        maxiters=maxiters,
        warn_exceed_max=warn_exceed_max,
        plateau_tol=plateau_tol,
        plateau_window=plateau_window,
        plateau_drop=plateau_drop,
        chop_guard=chop_guard,
        enforce_n_ge_m=enforce_n_ge_m,
        verbose=verbose,
    )
    B, assembly_info = voltmat_jacobi(C, N, alpha, beta;
                                      bandwidth=bandwidth,
                                      dropzeros_tol=dropzeros_tol,
                                      lin_droptol=lin_droptol,
                                      G_backend=G_backend,
                                      init_layer_backend=init_layer_backend,
                                      return_info=true)
    info = (; interp_info..., assembly_info...)
    if return_C && return_info
        return B, C, info
    elseif return_C
        return B, C
    elseif return_info
        return B, info
    else
        return B
    end
end

# -----------------------------------------------------------------------------
# Integrated final Volterra matrix:
#
#   first n_top rows are dense,
#   remaining rows are taken from the symmetric band constructor.
#
# By default, n_top = bandwidth = size(C,1).  The output is a SparseMatrixCSC
# whose top rows store all numerically nonzero entries and whose tail rows store
# only the banded part.
# -----------------------------------------------------------------------------

function _merge_topdense_and_tail(
    Btail::SparseMatrixCSC{Float64,Int},
    Btop::AbstractMatrix{<:Real};
    dropzeros_tol::Float64=0.0,
)
    nrows, ncols = size(Btail)
    size(Btop, 2) == ncols ||
        throw(DimensionMismatch("Btop and Btail must have the same number of columns."))

    ntop = size(Btop, 1)
    ntop <= nrows || throw(DimensionMismatch("Btop has too many rows."))

    colptr = Vector{Int}(undef, ncols + 1)
    rowval = Int[]
    nzval = Float64[]

    # Crude but useful size hint: dense top + tail nnz.
    sizehint!(rowval, ntop*ncols + nnz(Btail))
    sizehint!(nzval, ntop*ncols + nnz(Btail))

    colptr[1] = 1

    @inbounds for col in 1:ncols
        # Dense top block.
        for row in 1:ntop
            val = Float64(Btop[row, col])

            if abs(val) > dropzeros_tol
                push!(rowval, row)
                push!(nzval, val)
            end
        end

        # Tail band entries, excluding rows already overwritten by Btop.
        for p in Btail.colptr[col]:(Btail.colptr[col+1]-1)
            row = Btail.rowval[p]

            if row > ntop
                val = Btail.nzval[p]

                if abs(val) > dropzeros_tol
                    push!(rowval, row)
                    push!(nzval, val)
                end
            end
        end

        colptr[col+1] = length(rowval) + 1
    end

    return SparseMatrixCSC(nrows, ncols, colptr, rowval, nzval)
end


function voltmat_jacobi_full(
    C::AbstractMatrix{<:Real},
    N::Integer,
    alpha::Real=0.0,
    beta::Real=0.0;
    n_top::Union{Nothing,Integer}=nothing,
    bandwidth::Union{Nothing,Integer}=nothing,
    G_backend::Symbol=:recurrence,
    init_layer_backend::Symbol=:auto,
    init_precision::Integer=256,
    dropzeros_tol::Float64=0.0,
    lin_droptol::Float64=0.0,
    return_info::Bool=false,
)
    alpha, beta = _check_ab(alpha, beta)
    N < 0 && throw(ArgumentError("N must be nonnegative."))

    Nint = Int(N)
    nrow, mcol = _check_C_n_ge_m(C)

    ntop = _is_legendre_case(alpha, beta) ? 0 :
           (n_top === nothing ? _default_top_rows(nrow, mcol, alpha, beta, Nint) : Int(n_top))
    ntop < 0 && throw(ArgumentError("n_top must be nonnegative."))
    ntop = min(ntop, Nint + 1)

    bw = bandwidth === nothing ? nrow : Int(bandwidth)
    bw < 0 && throw(ArgumentError("bandwidth must be nonnegative."))
    bw = min(bw, Nint)

    Btail, tail_info = voltmat_jacobi(
        C,
        Nint,
        alpha,
        beta;
        bandwidth=bw,
        dropzeros_tol=dropzeros_tol,
        lin_droptol=lin_droptol,
        G_backend=G_backend,
        init_layer_backend=init_layer_backend,
        return_info=true,
    )

    Btop = jacobi_volterra_topdense_from_C(
        C,
        Nint,
        alpha,
        beta;
        n_top=ntop,
        init_precision=init_precision,
        lin_droptol=lin_droptol,
        init_layer_backend=init_layer_backend,
    )

    B = ntop == 0 ? Btail : _merge_topdense_and_tail(
        Btail,
        Btop;
        dropzeros_tol=dropzeros_tol,
    )

    info = (
        alpha=alpha,
        beta=beta,
        nrow=nrow,
        mcol=mcol,
        n_top=ntop,
        bandwidth=bw,
        G_backend=G_backend,
        init_layer_backend=tail_info.init_layer_backend,
        init_precision=Int(init_precision),
        matrix_size=size(B),
        top_nnz=count(!iszero, Btop),
        tail_nnz=tail_info.nnz,
        final_nnz=nnz(B),
        structure=(ntop == 0 ? :strict_band : :topdense_bandtail),
    )

    return return_info ? (B, info) : B
end


function voltmat_jacobi_full(
    K::Function,
    N::Integer,
    alpha::Real=0.0,
    beta::Real=0.0;
    tol::Float64=1.0e-12,
    nmin::Integer=8,
    mmin::Integer=8,
    nmax::Integer=128,
    mmax::Integer=128,
    growth::Real=1.5,
    maxiters::Integer=50,
    warn_exceed_max::Bool=true,
    plateau_tol::Union{Nothing,Float64}=nothing,
    plateau_window::Integer=8,
    plateau_drop::Float64=0.1,
    chop_guard::Integer=4,
    n_top::Union{Nothing,Integer}=nothing,
    bandwidth::Union{Nothing,Integer}=nothing,
    G_backend::Symbol=:recurrence,
    init_layer_backend::Symbol=:auto,
    init_precision::Integer=256,
    dropzeros_tol::Float64=0.0,
    lin_droptol::Float64=0.0,
    enforce_n_ge_m::Bool=true,
    return_C::Bool=false,
    return_info::Bool=false,
    verbose::Bool=false,
)
    alpha, beta = _check_ab(alpha, beta)
    N < 0 && throw(ArgumentError("N must be nonnegative."))

    C, interp_info = adaptive_jacobi_interp_coeffs_Khat(
        K,
        alpha,
        beta;
        tol=tol,
        nmin=nmin,
        mmin=mmin,
        nmax=nmax,
        mmax=mmax,
        growth=growth,
        maxiters=maxiters,
        warn_exceed_max=warn_exceed_max,
        plateau_tol=plateau_tol,
        plateau_window=plateau_window,
        plateau_drop=plateau_drop,
        chop_guard=chop_guard,
        enforce_n_ge_m=enforce_n_ge_m,
        verbose=verbose,
    )

    B, assembly_info = voltmat_jacobi_full(
        C,
        N,
        alpha,
        beta;
        n_top=n_top,
        bandwidth=bandwidth,
        G_backend=G_backend,
        init_layer_backend=init_layer_backend,
        init_precision=init_precision,
        dropzeros_tol=dropzeros_tol,
        lin_droptol=lin_droptol,
        return_info=true,
    )

    info = (;
        interp_info...,
        assembly_info...,
    )

    if return_C && return_info
        return B, C, info
    elseif return_C
        return B, C
    elseif return_info
        return B, info
    else
        return B
    end
end


end # module Voltmat_Jacobi
