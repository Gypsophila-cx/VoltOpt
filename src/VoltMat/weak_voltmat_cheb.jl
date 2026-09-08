module Weak_VoltMat_Cheb

export weak_voltmat_cheb
export PackedLowerHessenberg
export packed_to_dense
export phi_beta, invphi_beta
export mul_transpose!
export validate_weak_voltmat_columns

# Current weak-singular implementation: contract the kernel's second
# Chebyshev index first, then stream the four-point moment recurrence in two
# block-i buffers. Hybrid integer-eta Chebyshev multipliers are fused where
# possible.
using FFTW
using FastGaussQuadrature
using LinearAlgebra
using Random
using SpecialFunctions: loggamma
using Base.Threads
# The block-i recurrence advances all precombined kernel-z moments together in
# two (I+1) x (N+1) buffers; no k>0 H[j,k,l] tensor is formed.

const _DEFAULT_INTERP_TOL = eps()

"""
    phi_beta(x, beta)

JFP map `2*((1+x)/2)^beta - 1`.
"""
phi_beta(x::Real, beta::Real) = 2.0 * ((1.0 + x) / 2.0)^beta - 1.0

"""
    invphi_beta(z, beta)

Inverse JFP map `2*((1+z)/2)^(1/beta) - 1`.
"""
invphi_beta(z::Real, beta::Real) = 2.0 * ((1.0 + z) / 2.0)^(1.0 / beta) - 1.0

cheb_h(n::Integer) = n == 0 ? pi : pi / 2.0

_is_nonnegative_integer(x::Real; tol::Float64 = 128eps(Float64)) =
    x >= -tol && abs(x - round(x)) <= tol * max(1.0, abs(x))

"""
    cheb_lobatto_nodes(N)

Return Chebyshev-Lobatto nodes `cos(pi*j/N)`, `j=0,...,N`.
"""
function cheb_lobatto_nodes(N::Int)
    N < 0 && throw(ArgumentError("N must be nonnegative"))
    N == 0 && return [1.0]
    return cos.(pi .* collect(0:N) ./ N)
end

"""
    chebT_table(x, N)

Return a table whose `(m,n+1)` entry is `T_n(x[m])`.
"""
function chebT_table(x::AbstractVector, N::Int)
    N < 0 && throw(ArgumentError("N must be nonnegative"))
    M = length(x)
    T = zeros(Float64, M, N + 1)
    T[:, 1] .= 1.0

    if N >= 1
        T[:, 2] .= x
        for n in 2:N
            @views T[:, n + 1] .= 2.0 .* x .* T[:, n] .- T[:, n - 1]
        end
    end

    return T
end

"""
    cheb_eval_1d(c, x)

Evaluate `sum_n c[n+1] T_n(x)` by Clenshaw's recurrence.
"""
function cheb_eval_1d(c::AbstractVector, x::Real)
    N = length(c) - 1
    N < 0 && throw(ArgumentError("coefficient vector must be nonempty"))
    N == 0 && return Float64(c[1])

    xx = Float64(x)
    bkp1 = 0.0
    bkp2 = 0.0

    @inbounds for k in N:-1:1
        bk = 2.0 * xx * bkp1 - bkp2 + c[k + 1]
        bkp2 = bkp1
        bkp1 = bk
    end

    return xx * bkp1 - bkp2 + c[1]
end

"""
    cheb_eval_2d(C, z, q)

Evaluate `sum_i sum_j C[i+1,j+1] T_i(z) T_j(q)`.
"""
function cheb_eval_2d(C::AbstractMatrix, z::Real, q::Real)
    I = size(C, 1) - 1
    tmp = zeros(Float64, I + 1)

    @inbounds for i in 0:I
        tmp[i + 1] = cheb_eval_1d(@view(C[i + 1, :]), q)
    end

    return cheb_eval_1d(tmp, z)
end

"""
    jacobiP_table(x, N, a, b)

Return a table of Jacobi polynomials `P_n^(a,b)(x[m])`, `n=0,...,N`.
"""
function jacobiP_table(x::AbstractVector, N::Int, a::Real, b::Real)
    N < 0 && throw(ArgumentError("N must be nonnegative"))
    M = length(x)
    P = zeros(Float64, M, N + 1)
    P[:, 1] .= 1.0

    if N >= 1
        aa = Float64(a)
        bb = Float64(b)
        P[:, 2] .= 0.5 .* ((aa - bb) .+ (aa + bb + 2.0) .* x)

        for n in 1:(N - 1)
            nf = Float64(n)
            den = 2.0 * (nf + 1.0) * (nf + aa + bb + 1.0) *
                  (2.0 * nf + aa + bb)

            Bx = (2.0 * nf + aa + bb + 1.0) .*
                 ((2.0 * nf + aa + bb) * (2.0 * nf + aa + bb + 2.0) .* x .+
                  (aa^2 - bb^2))

            Cc = 2.0 * (nf + aa) * (nf + bb) *
                 (2.0 * nf + aa + bb + 2.0)

            @views P[:, n + 2] .= (Bx .* P[:, n + 1] .- Cc .* P[:, n]) ./ den
        end
    end

    return P
end

"""
    A0_table(q, Kmax)

Build `A_{0,k}(q) = P_k^(-1,0)(q)` for `k=0,...,Kmax`.

The implementation uses
`P_k^(-1,0)(q) = (q-1)/2 * P_{k-1}^{(1,0)}(q)` for `k>=1`.
"""
function A0_table(q::AbstractVector, Kmax::Int)
    Kmax < 0 && throw(ArgumentError("Kmax must be nonnegative"))
    M = length(q)
    A0 = zeros(Float64, M, Kmax + 1)
    A0[:, 1] .= 1.0

    if Kmax >= 1
        P10 = jacobiP_table(q, Kmax - 1, 1.0, 0.0)
        for k in 1:Kmax
            @views A0[:, k + 1] .= 0.5 .* (q .- 1.0) .* P10[:, k]
        end
    end

    return A0
end

function _residual_factor_general(q::AbstractVector, mu_sing::Float64, sigma::Float64, beta::Float64)
    gamma = (sigma + 1.0) / beta - 1.0
    mu_sing < 1.0 || throw(ArgumentError("mu must be less than 1"))
    sigma > -1.0 || throw(ArgumentError("sigma must be greater than -1"))
    beta > 0.0 || throw(ArgumentError("beta must be positive"))
    gamma > -1.0 || throw(ArgumentError("gamma=(sigma+1)/beta-1 must be greater than -1"))

    R = similar(q, Float64)
    a_left = -mu_sing
    # q is the [-1,1] image of s=((1+t)/2)^beta.  The transformed physical
    # weight contributes 2^(-mu)/beta; sigma is already carried by s^gamma.
    # Including 2^sigma here would make the direct initializer inconsistent
    # with both the DE initializer and the normalized JFP basis.
    scale = 2.0^(-mu_sing) / beta

    @inbounds for n in eachindex(q)
        qv = q[n]
        s = (1.0 + qv) / 2.0
        original = scale * (1.0 - s^(1.0 / beta))^(-mu_sing) * s^gamma
        extracted = (1.0 - qv)^a_left * (1.0 + qv)^gamma
        R[n] = original / extracted
    end

    return R
end

function _default_Qbase(J::Int, N::Int)
    return max(64, cld(N + J + 1, 2) + 80)
end

function _neumaier_sum(x)
    s = 0.0
    c = 0.0

    @inbounds for xi in x
        t = s + xi
        if abs(s) >= abs(xi)
            c += (s - t) + xi
        else
            c += (xi - t) + s
        end
        s = t
    end

    return s + c
end

function _neumaier_dot(x, y)
    length(x) == length(y) || throw(DimensionMismatch("dot inputs must have equal length"))
    s = 0.0
    c = 0.0

    @inbounds for i in eachindex(x, y)
        p = x[i] * y[i]
        t = s + p
        if abs(s) >= abs(p)
            c += (s - t) + p
        else
            c += (p - t) + s
        end
        s = t
    end

    return s + c
end

function _inf_rel_diff(a, b)
    return norm(a .- b, Inf) / max(norm(b, Inf), floatmin(Float64))
end

function _sech2_stable(y::Float64)
    ay = abs(y)
    if ay > 350.0
        return 0.0
    else
        e = exp(-2.0 * ay)
        return 4.0 * e / (1.0 + e)^2
    end
end

function _tanh_sinh_q_and_dq(s::Float64)
    y = (pi / 2.0) * sinh(s)
    q = 0.5 * (1.0 + tanh(y))
    dq = (pi / 4.0) * cosh(s) * _sech2_stable(y)
    return q, dq
end

function _tanh_sinh_S_for_qeps(epsq::Float64)
    0.0 < epsq < 0.5 || throw(ArgumentError("epsq must satisfy 0 < epsq < 0.5"))
    return asinh((1.0 / pi) * log(1.0 / epsq - 1.0))
end

function _S_beta_stable(q::Float64, beta_map::Float64, mu_sing::Float64)
    beta_map == 1.0 && return 1.0
    q <= 0.0 && return 1.0
    q >= 1.0 && return beta_map^mu_sing

    if q > 1.0 - 1e-12
        return beta_map^mu_sing
    end

    qp = q^(1.0 / beta_map)
    if qp >= 1.0
        return beta_map^mu_sing
    end

    log_num = log1p(-q)
    log_den = log1p(-qp)
    return exp(mu_sing * (log_num - log_den))
end

"""
    compute_H00k_DE(alpha, mu, beta_map, Kmax; qfactor=12, epsq=eps(Float64),
                    accurate_sum=false)

Compute the initial weak-singular moment row `H_{0,0,k}`,
`k=0,...,Kmax`, by DE/tanh-sinh quadrature after the `q=z^beta`
change of variables.  Here `mu` is the endpoint singularity strength in
`(1-z)^(-mu)`.  Under the public Volterra convention it is the same `mu`
as in `(x-y)^(-mu)`.
"""
function compute_H00k_DE(alpha::Real, mu::Real, beta_map::Real, Kmax::Int;
                         qfactor::Int = 12,
                         epsq::Float64 = eps(Float64),
                         accurate_sum::Bool = false)
    Kmax < 0 && throw(ArgumentError("Kmax must be nonnegative"))
    qfactor > 0 || throw(ArgumentError("qfactor must be positive"))

    alphaf = Float64(alpha)
    mu_sing = Float64(mu)
    betaf = Float64(beta_map)
    alphaf > -1.0 || throw(ArgumentError("alpha must be greater than -1"))
    mu_sing < 1.0 || throw(ArgumentError("mu must be less than 1"))
    betaf > 0.0 || throw(ArgumentError("beta_map must be positive"))

    H = zeros(Float64, Kmax + 1)
    H[1] = 2.0^(1.0 - mu_sing) *
           exp(loggamma(alphaf + 1.0) +
               loggamma(1.0 - mu_sing) -
               loggamma(alphaf + 2.0 - mu_sing))
    Kmax == 0 && return H

    A = (alphaf + 1.0) / betaf - 1.0
    B = 1.0 - mu_sing
    S = _tanh_sinh_S_for_qeps(epsq)
    Q = max(qfactor * Kmax, 80)
    h = 2.0 * S / (Q - 1)
    pref = -2.0^(1.0 - mu_sing) / betaf

    xnodes = Float64[]
    weights = Float64[]
    sizehint!(xnodes, Q)
    sizehint!(weights, Q)

    @inbounds for nu in 1:Q
        s = -S + (nu - 1) * h
        q, dq = _tanh_sinh_q_and_dq(s)

        if !(0.0 < q < 1.0) || dq == 0.0 || !isfinite(dq)
            continue
        end

        Sb = _S_beta_stable(q, betaf, mu_sing)
        w = h * dq * q^A * (1.0 - q)^B * Sb
        if isfinite(w)
            push!(xnodes, 2.0 * q - 1.0)
            push!(weights, w)
        end
    end

    H[2] = pref * (accurate_sum ? _neumaier_sum(weights) : sum(weights))
    Kmax == 1 && return H

    nnode = length(weights)
    R_nm1 = zeros(Float64, nnode)
    R_n = ones(Float64, nnode)
    R_np1 = similar(R_n)

    @inbounds for n in 0:(Kmax - 2)
        nf = Float64(n)
        a = (nf + 2.0) / (2.0 * nf + 3.0)
        b = -1.0 / ((2.0 * nf + 1.0) * (2.0 * nf + 3.0))
        c = nf / (2.0 * nf + 1.0)

        if accurate_sum
            @simd for i in 1:nnode
                R_np1[i] = ((xnodes[i] - b) * R_n[i] - c * R_nm1[i]) / a
            end
            H[n + 3] = pref * _neumaier_dot(weights, R_np1)
        else
            s = 0.0
            @simd for i in 1:nnode
                rnew = ((xnodes[i] - b) * R_n[i] - c * R_nm1[i]) / a
                R_np1[i] = rnew
                s += weights[i] * rnew
            end
            H[n + 3] = pref * s
        end
        R_nm1, R_n, R_np1 = R_n, R_np1, R_nm1
    end

    return H
end

"""
    compute_H00k_DE_adaptive(alpha, mu, beta_map, Kmax; ...)

Compute the initial moment row with successively doubled DE node density until
two consecutive rows agree in relative infinity norm.  The returned value is
`(H, info)`, where `info` records the density used and convergence status.
"""
function compute_H00k_DE_adaptive(alpha::Real, mu::Real, beta_map::Real, Kmax::Int;
                                  qfactor::Int = 12,
                                  qfactor_max::Int = 192,
                                  rtol::Float64 = 1e-13,
                                  epsq::Float64 = eps(Float64),
                                  accurate_sum::Bool = true,
                                  verbose::Bool = false)
    qfactor > 0 || throw(ArgumentError("qfactor must be positive"))
    qfactor_max >= qfactor ||
        throw(ArgumentError("qfactor_max must be at least qfactor"))
    rtol > 0.0 || throw(ArgumentError("rtol must be positive"))

    qf_old = qfactor
    Hold = compute_H00k_DE(alpha, mu, beta_map, Kmax;
                           qfactor = qf_old,
                           epsq = epsq,
                           accurate_sum = accurate_sum)

    while true
        qf_new = min(2 * qf_old, qfactor_max)
        Hnew = compute_H00k_DE(alpha, mu, beta_map, Kmax;
                               qfactor = qf_new,
                               epsq = epsq,
                               accurate_sum = accurate_sum)
        rel = _inf_rel_diff(Hnew, Hold)
        converged = rel <= rtol

        if verbose
            @info "DE H00 adaptive check" Kmax qf_old qf_new rel rtol converged
        end

        if converged || qf_new >= qfactor_max
            if !converged
                @warn "DE H00 adaptive check did not meet tolerance" Kmax qf_old qf_new rel rtol qfactor_max
            end
            return Hnew, (
                converged = converged,
                qfactor_used = qf_new,
                qfactor_prev = qf_old,
                rel_change = rel,
                rtol = rtol,
                accurate_sum = accurate_sum,
            )
        end

        qf_old = qf_new
        Hold = Hnew
    end
end

"""
    initial_moments_direct(mu, sigma, beta, J, N; Qbase=nothing)

Direct all-table Gauss-Jacobi computation of
`H[j+1,k+1] = H_{j,0,k}` for `0<=j<=J`, `0<=k<=N`.
Here `mu` is the public singularity strength in `(x-y)^(-mu)`, so the
Jacobi left endpoint exponent is `-mu`.
"""
function initial_moments_direct(mu::Real, sigma::Real, beta::Real, J::Int, N::Int; Qbase = nothing)
    J < 0 && throw(ArgumentError("J must be nonnegative"))
    N < 0 && throw(ArgumentError("N must be nonnegative"))

    mu_sing = Float64(mu)
    sigf = Float64(sigma)
    betaf = Float64(beta)
    mu_sing < 1.0 || throw(ArgumentError("mu must be less than 1"))
    Q = Qbase === nothing ? _default_Qbase(J, N) : Int(Qbase)
    gamma = (sigf + 1.0) / betaf - 1.0
    a_left = -mu_sing

    q, w = gaussjacobi(Q, a_left, gamma)
    R = _residual_factor_general(q, mu_sing, sigf, betaf)
    Tj = chebT_table(q, J)
    A0 = A0_table(q, N)
    wr = w .* R

    return Tj' * (A0 .* reshape(wr, :, 1))
end

function _q_times_Q_coeffs(k::Int)
    if k == 0
        return 2.0, 1.0, 0.0
    end

    kf = Float64(k)
    a = (kf + 1.0) / (2.0 * kf + 1.0)
    b = -1.0 / ((2.0 * kf - 1.0) * (2.0 * kf + 1.0))
    c = (kf - 1.0) / (2.0 * kf - 1.0)
    return a, b, c
end

function _base_row_gauss_jacobi(mu::Float64, sigma::Float64, beta::Float64,
                                Kmax::Int, Q::Int)
    gamma = (sigma + 1.0) / beta - 1.0
    mu_sing = mu
    mu_sing < 1.0 || throw(ArgumentError("mu must be less than 1"))
    a_left = -mu_sing
    q, w = gaussjacobi(Q, a_left, gamma)
    R = _residual_factor_general(q, mu_sing, sigma, beta)
    A0 = A0_table(q, Kmax)
    return A0' * (w .* R)
end

function _base_row_fast(mu::Float64, sigma::Float64, beta::Float64, Kmax::Int, Q::Int;
                        de_qfactor::Int = 12,
                        de_adaptive::Bool = false,
                        de_qfactor_max::Int = 192,
                        de_rtol::Float64 = 1e-13,
                        accurate_sum::Bool = false,
                        return_info::Bool = false)
    if de_adaptive
        H, info = compute_H00k_DE_adaptive(
            sigma,
            mu,
            beta,
            Kmax;
            qfactor = de_qfactor,
            qfactor_max = de_qfactor_max,
            rtol = de_rtol,
            accurate_sum = accurate_sum,
        )
        return return_info ? (H, info) : H
    end

    H = compute_H00k_DE(
        sigma,
        mu,
        beta,
        Kmax;
        qfactor = de_qfactor,
        accurate_sum = accurate_sum,
    )
    info = (
        converged = true,
        qfactor_used = de_qfactor,
        qfactor_prev = de_qfactor,
        rel_change = 0.0,
        rtol = de_rtol,
        accurate_sum = accurate_sum,
    )
    return return_info ? (H, info) : H
end

"""
    initial_moments_fast(mu, sigma, beta, J, N; Qbase=nothing)

Fast initialization of `M[j+1,k+1] = H_{j,0,k}`.

Only the base row `M[1,0:N+J]` is integrated. All other rows are generated
by the closed `(j,k)` recurrence for `P_k^(-1,0)`.
Here `mu` is the public singularity strength in `(x-y)^(-mu)`.
"""
function initial_moments_fast(mu::Real, sigma::Real, beta::Real, J::Int, N::Int;
                              Qbase = nothing,
                              de_qfactor::Int = 12,
                              de_adaptive::Bool = false,
                              de_qfactor_max::Int = 192,
                              de_rtol::Float64 = 1e-13,
                              accurate_sum::Bool = false,
                              return_info::Bool = false)
    J < 0 && throw(ArgumentError("J must be nonnegative"))
    N < 0 && throw(ArgumentError("N must be nonnegative"))

    mu_sing = Float64(mu)
    sigf = Float64(sigma)
    betaf = Float64(beta)
    mu_sing < 1.0 || throw(ArgumentError("mu must be less than 1"))
    Kmax = N + J
    Q = Qbase === nothing ? _default_Qbase(J, N) : Int(Qbase)

    M0, base_row_info = _base_row_fast(
        mu_sing,
        sigf,
        betaf,
        Kmax,
        Q;
        de_qfactor = de_qfactor,
        de_adaptive = de_adaptive,
        de_qfactor_max = de_qfactor_max,
        de_rtol = de_rtol,
        accurate_sum = accurate_sum,
        return_info = true,
    )
    Mfull = fill(NaN, J + 1, Kmax + 1)
    Mfull[1, :] .= M0

    if J >= 1
        @inbounds for k in 0:(Kmax - 1)
            a, b, c = _q_times_Q_coeffs(k)
            val = a * Mfull[1, k + 2] + b * Mfull[1, k + 1]
            if k >= 1
                val += c * Mfull[1, k]
            end
            Mfull[2, k + 1] = val
        end
    end

    if J >= 2
        @inbounds for j in 1:(J - 1)
            kmax_valid = Kmax - j - 1
            for k in 0:kmax_valid
                a, b, c = _q_times_Q_coeffs(k)
                val = 2.0 * a * Mfull[j + 1, k + 2] +
                      2.0 * b * Mfull[j + 1, k + 1] -
                      Mfull[j, k + 1]
                if k >= 1
                    val += 2.0 * c * Mfull[j + 1, k]
                end
                Mfull[j + 2, k + 1] = val
            end
        end
    end

    Minit = Mfull[1:(J + 1), 1:(N + 1)]
    info = (
        base_row_info = base_row_info,
        Qbase = Q,
        Kmax = Kmax,
        de_qfactor = de_qfactor,
        de_adaptive = de_adaptive,
        de_qfactor_max = de_qfactor_max,
        de_rtol = de_rtol,
        accurate_sum = accurate_sum,
    )
    return return_info ? (Minit, info) : Minit
end

"""
    build_H_for_one_j_from_initial_row(Mj0, N)

Given `H_{j,0,k}`, `k=0,...,N`, build the triangular table
`H[r+1,k+1] = H_{j,r,k}` by the legacy six-term recurrence.

This is retained only as a small-N/reference compatibility implementation;
the production `:block_i` path uses the contracted four-point recurrence.
"""
function build_H_for_one_j_from_initial_row(Mj0::AbstractVector, N::Int)
    length(Mj0) >= N + 1 || throw(ArgumentError("initial row is too short"))
    H = zeros(Float64, N + 1, N + 1)

    @inbounds for k in 0:N
        H[1, k + 1] = Mj0[k + 1]
    end

    if N >= 1
        H[2, 2] = H[1, 2] + H[1, 1]
    end

    @inbounds for r in 0:(N - 1)
        Wr = r == 0 ? 1.0 : 2.0
        kstart = r == 0 ? 1 : r

        for k in kstart:(N - 1)
            kf = Float64(k)
            rf = Float64(r)

            Lkp1 = Wr * ((kf + rf + 1.0) * (kf + rf + 2.0)) /
                   (2.0 * (kf + 1.0) * (2.0 * kf + 1.0))

            Gkp1 = ((kf - rf + 1.0) * (kf + rf + 1.0)) /
                   ((kf + 1.0) * (2.0 * kf + 1.0))

            Gk = 2.0 * (2.0 * kf^2 + 2.0 * rf^2 - 1.0) /
                 ((2.0 * kf - 1.0) * (2.0 * kf + 1.0))

            num = Gkp1 * H[r + 1, k + 2] + Gk * H[r + 1, k + 1]

            if k >= 2 && r <= k - 1
                Gkm1 = ((kf - rf - 1.0) * (kf + rf - 1.0)) /
                       ((kf - 1.0) * (2.0 * kf - 1.0))
                num += Gkm1 * H[r + 1, k]
            end

            if r + 1 <= k
                Lk = Wr * (2.0 * (kf + rf + 1.0) * (kf - rf - 1.0)) /
                     ((2.0 * kf - 1.0) * (2.0 * kf + 1.0))
                num -= Lk * H[r + 2, k + 1]
            end

            if k >= 2 && r + 1 <= k - 1
                Lkm1 = Wr * ((kf - rf - 1.0) * (kf - rf - 2.0)) /
                       (2.0 * (kf - 1.0) * (2.0 * kf - 1.0))
                num -= Lkm1 * H[r + 2, k]
            end

            H[r + 2, k + 2] = num / Lkp1
        end
    end

    return H
end

"""
    build_H_for_one_j_from_initial_row!(H, Mj0, N)

In-place version of the legacy six-term recurrence. It overwrites the triangular
entries `0 <= r <= k <= N` of `H` and intentionally leaves invalid entries
untouched; all downstream kernels only read the triangular region.
"""
function build_H_for_one_j_from_initial_row!(H::AbstractMatrix, Mj0::AbstractVector, N::Int)
    length(Mj0) >= N + 1 || throw(ArgumentError("initial row is too short"))
    size(H, 1) >= N + 1 || throw(ArgumentError("H has too few rows"))
    size(H, 2) >= N + 1 || throw(ArgumentError("H has too few columns"))

    @inbounds for k in 0:N
        H[1, k + 1] = Mj0[k + 1]
    end

    if N >= 1
        H[2, 2] = H[1, 2] + H[1, 1]
    end

    @inbounds for r in 0:(N - 1)
        Wr = r == 0 ? 1.0 : 2.0
        kstart = r == 0 ? 1 : r

        for k in kstart:(N - 1)
            kf = Float64(k)
            rf = Float64(r)

            Lkp1 = Wr * ((kf + rf + 1.0) * (kf + rf + 2.0)) /
                   (2.0 * (kf + 1.0) * (2.0 * kf + 1.0))

            Gkp1 = ((kf - rf + 1.0) * (kf + rf + 1.0)) /
                   ((kf + 1.0) * (2.0 * kf + 1.0))

            Gk = 2.0 * (2.0 * kf^2 + 2.0 * rf^2 - 1.0) /
                 ((2.0 * kf - 1.0) * (2.0 * kf + 1.0))

            num = Gkp1 * H[r + 1, k + 2] + Gk * H[r + 1, k + 1]

            if k >= 2 && r <= k - 1
                Gkm1 = ((kf - rf - 1.0) * (kf + rf - 1.0)) /
                       ((kf - 1.0) * (2.0 * kf - 1.0))
                num += Gkm1 * H[r + 1, k]
            end

            if r + 1 <= k
                Lk = Wr * (2.0 * (kf + rf + 1.0) * (kf - rf - 1.0)) /
                     ((2.0 * kf - 1.0) * (2.0 * kf + 1.0))
                num -= Lk * H[r + 2, k + 1]
            end

            if k >= 2 && r + 1 <= k - 1
                Lkm1 = Wr * ((kf - rf - 1.0) * (kf - rf - 2.0)) /
                       (2.0 * (kf - 1.0) * (2.0 * kf - 1.0))
                num -= Lkm1 * H[r + 2, k]
            end

            H[r + 2, k + 2] = num / Lkp1
        end
    end

    return H
end

"""
    duffy_kernel_value(K, z, q, beta)

Evaluate the Duffy-pulled kernel `K_D(z,q)`.
"""
function duffy_kernel_value(K, z::Real, q::Real, beta::Real)
    x = invphi_beta(z, beta)
    t = invphi_beta(q, beta)
    y = 0.5 * (x + 1.0) * t + 0.5 * (x - 1.0)
    return K(x, y)
end

"""
    sample_duffy_kernel(K, Nz, Nq, beta)

Sample `K_D(z,q)` on the tensor Chebyshev-Lobatto grid.
"""
function sample_duffy_kernel(K, Nz::Int, Nq::Int, beta::Real)
    z_nodes = cheb_lobatto_nodes(Nz)
    q_nodes = cheb_lobatto_nodes(Nq)
    F = zeros(Float64, Nz + 1, Nq + 1)

    @inbounds for a in 0:Nz
        z = z_nodes[a + 1]
        x = invphi_beta(z, beta)

        for b in 0:Nq
            q = q_nodes[b + 1]
            t = invphi_beta(q, beta)
            y = 0.5 * (x + 1.0) * t + 0.5 * (x - 1.0)
            F[a + 1, b + 1] = K(x, y)
        end
    end

    return F
end

"""
    cheb_coeffs_lobatto_1d(f)

Chebyshev interpolation coefficients from Lobatto values via DCT-I.
"""
function cheb_coeffs_lobatto_1d(f::AbstractVector)
    N = length(f) - 1
    N < 0 && throw(ArgumentError("value vector must be nonempty"))
    N == 0 && return Float64.(f)

    c = FFTW.r2r(Float64.(f), FFTW.REDFT00)
    c ./= N
    c[1] /= 2.0
    c[end] /= 2.0
    return c
end

"""
    cheb_coeffs_lobatto_2d(F)

Tensor Chebyshev interpolation coefficients from Lobatto grid values.
"""
function cheb_coeffs_lobatto_2d(F::AbstractMatrix)
    Nz = size(F, 1) - 1
    Nq = size(F, 2) - 1
    C = Array{Float64}(F)

    if Nz > 0
        C = FFTW.r2r(C, FFTW.REDFT00, 1)
        C ./= Nz
        C[1, :] ./= 2.0
        C[end, :] ./= 2.0
    end

    if Nq > 0
        C = FFTW.r2r(C, FFTW.REDFT00, 2)
        C ./= Nq
        C[:, 1] ./= 2.0
        C[:, end] ./= 2.0
    end

    return C
end

function coefficient_tail_info(C::AbstractMatrix; row_block::Int = 8, col_block::Int = 8,
                               abs_floor::Float64 = 1e-300)
    nz, nq = size(C)
    rb = min(row_block, nz)
    cb = min(col_block, nq)
    max_coeff = maximum(abs, C)
    row_tail = maximum(abs, @view(C[end - rb + 1:end, :]))
    col_tail = maximum(abs, @view(C[:, end - cb + 1:end]))
    denom = max(max_coeff, abs_floor)

    return (
        max_coeff = max_coeff,
        row_tail = row_tail,
        col_tail = col_tail,
        row_rel = row_tail / denom,
        col_rel = col_tail / denom,
    )
end

function trim_coefficients(C::AbstractMatrix; rtol::Float64 = 1e-13,
                           atol::Float64 = 1e-15, safety::Int = 0)
    maxc = maximum(abs, C)
    cutoff = max(atol, rtol * maxc)
    nz, nq = size(C)
    last_i = 1
    last_j = 1

    for i in 1:nz
        if maximum(abs, @view(C[i, :])) > cutoff
            last_i = i
        end
    end

    for j in 1:nq
        if maximum(abs, @view(C[:, j])) > cutoff
            last_j = j
        end
    end

    last_i = min(nz, last_i + safety)
    last_j = min(nq, last_j + safety)

    return C[1:last_i, 1:last_j], (
        cutoff = cutoff,
        I_effective = last_i - 1,
        J_effective = last_j - 1,
        max_coeff = maxc,
        safety = safety,
    )
end

"""
    duffy_cheb_coefficients(K; beta, ...)

Adaptively construct the tensor Chebyshev coefficients of the Duffy-pulled
kernel `K_D(z,q)`.
"""
function duffy_cheb_coefficients(K; beta::Real, N0::Int = 16, Nmax::Int = 512,
                                 tol::Float64 = _DEFAULT_INTERP_TOL,
                                 row_block::Int = 8, col_block::Int = 8,
                                 trim_rtol::Float64 = 1e-13,
                                 trim_atol::Float64 = 1e-15,
                                 safety::Int = 0,
                                 verbose::Bool = false)
    beta > 0 || throw(ArgumentError("beta must be positive"))
    N0 > 0 || throw(ArgumentError("N0 must be positive"))
    Nmax >= N0 || throw(ArgumentError("Nmax must be at least N0"))

    N = N0
    iterations = NamedTuple[]
    last_C = nothing
    last_tail = nothing

    while true
        F = sample_duffy_kernel(K, N, N, beta)
        C = cheb_coeffs_lobatto_2d(F)
        tail = coefficient_tail_info(C; row_block = row_block, col_block = col_block)
        push!(iterations, (N = N, tail_info = tail))

        if verbose
            @info "Duffy-Chebyshev interpolation" N tail.row_rel tail.col_rel tail.max_coeff
        end

        last_C = C
        last_tail = tail

        if tail.row_rel < tol && tail.col_rel < tol
            Ctrim, triminfo = trim_coefficients(C; rtol = trim_rtol,
                                                atol = trim_atol,
                                                safety = safety)
            return Ctrim, (
                N = N,
                raw_size = size(C),
                trimmed_size = size(Ctrim),
                tail_info = tail,
                trim_info = triminfo,
                iterations = iterations,
                beta = Float64(beta),
                tol = tol,
                reached_Nmax = false,
            )
        end

        if N >= Nmax
            @warn "Reached Nmax without satisfying Duffy-Chebyshev tail tolerance" Nmax tol
            Ctrim, triminfo = trim_coefficients(last_C; rtol = trim_rtol,
                                                atol = trim_atol,
                                                safety = safety)
            return Ctrim, (
                N = N,
                raw_size = size(last_C),
                trimmed_size = size(Ctrim),
                tail_info = last_tail,
                trim_info = triminfo,
                iterations = iterations,
                beta = Float64(beta),
                tol = tol,
                reached_Nmax = true,
            )
        end

        N *= 2
    end
end

"""
    validate_duffy_cheb_interpolation(K, C; beta, ntest=10000, seed=1234)

Validate the Duffy-Chebyshev expansion on random physical triangle points.
"""
function validate_duffy_cheb_interpolation(K, C::AbstractMatrix; beta::Real,
                                           ntest::Int = 10_000, seed::Int = 1234)
    rng = MersenneTwister(seed)
    max_abs = 0.0
    max_rel = 0.0
    rms_acc = 0.0
    worst = (x = 0.0, y = 0.0, exact = 0.0, approx = 0.0)

    for _ in 1:ntest
        x = 2.0 * rand(rng) - 1.0
        y = -1.0 + (x + 1.0) * rand(rng)

        t = if abs(x + 1.0) < 1e-14
            -1.0
        else
            clamp((2.0 * y - (x - 1.0)) / (x + 1.0), -1.0, 1.0)
        end

        z = phi_beta(x, beta)
        q = phi_beta(t, beta)
        exact = K(x, y)
        approx = cheb_eval_2d(C, z, q)
        err = abs(approx - exact)
        rel = err / max(abs(exact), 1e-14)
        rms_acc += err^2

        if err > max_abs
            max_abs = err
            worst = (x = x, y = y, exact = exact, approx = approx)
        end
        max_rel = max(max_rel, rel)
    end

    return (
        max_abs = max_abs,
        rms_abs = sqrt(rms_acc / ntest),
        max_rel = max_rel,
        worst = worst,
    )
end

"""
    short_hankel_toeplitz_apply_direct(c, g, N)

Reference direct computation of
`p[ell+1] = sum_i c[i+1]g[i+ell+1]` and
`q[ell+1] = sum_i c[i+1]g[abs(i-ell)+1]`.
"""
function short_hankel_toeplitz_apply_direct(c::AbstractVector, g::AbstractVector, N::Int)
    I = length(c) - 1
    Gmax = length(g) - 1
    p = zeros(Float64, N + 1)
    q = zeros(Float64, N + 1)

    @inbounds for ell in 0:N
        ps = 0.0
        qs = 0.0
        for i in 0:I
            r1 = i + ell
            if r1 <= Gmax
                ps += c[i + 1] * g[r1 + 1]
            end

            r2 = abs(i - ell)
            if r2 <= Gmax
                qs += c[i + 1] * g[r2 + 1]
            end
        end
        p[ell + 1] = ps
        q[ell + 1] = qs
    end

    return p, q
end

function _linear_convolve_blockfft(kernel::AbstractVector, x::AbstractVector; block_size = nothing)
    m = length(kernel)
    n = length(x)
    if m == 0 || n == 0
        return Float64[]
    end

    B = block_size === nothing ? max(16, 4m) : Int(block_size)
    B = max(B, 1)
    L = nextpow(2, B + m - 1)

    kpad = zeros(ComplexF64, L)
    @inbounds for i in 1:m
        kpad[i] = kernel[i]
    end
    Khat = FFTW.fft(kpad)

    out = zeros(Float64, m + n - 1)

    start = 1
    while start <= n
        stop = min(start + B - 1, n)
        len = stop - start + 1

        xpad = zeros(ComplexF64, L)
        @inbounds for i in 1:len
            xpad[i] = x[start + i - 1]
        end

        y = FFTW.ifft(FFTW.fft(xpad) .* Khat)
        valid = len + m - 1

        @inbounds for r in 1:valid
            out[start + r - 1] += real(y[r])
        end

        start += B
    end

    return out
end

"""
    short_hankel_toeplitz_apply_blockfft(c, g, N; block_size=nothing)

Block-FFT overlap-add helper for the short-kernel Hankel and symmetric
Toeplitz sums used in matrix assembly.
"""
function short_hankel_toeplitz_apply_blockfft(c::AbstractVector, g::AbstractVector, N::Int;
                                             block_size = nothing,
                                             direct_threshold::Int = 32)
    I = length(c) - 1
    I < 0 && throw(ArgumentError("c must be nonempty"))
    N < 0 && throw(ArgumentError("N must be nonnegative"))
    isempty(g) && throw(ArgumentError("g must be nonempty"))

    if I <= direct_threshold
        return short_hankel_toeplitz_apply_direct(c, g, N)
    end

    cf = Float64.(c)
    gf = Float64.(g)
    revconv = _linear_convolve_blockfft(reverse(cf), gf; block_size = block_size)
    leftconv = _linear_convolve_blockfft(cf, gf; block_size = block_size)

    p = zeros(Float64, N + 1)
    q = zeros(Float64, N + 1)
    g0 = gf[1]

    @inbounds for ell in 0:N
        idxp = I + ell + 1
        p[ell + 1] = idxp <= length(revconv) ? revconv[idxp] : 0.0

        left = ell + 1 <= length(leftconv) ? leftconv[ell + 1] : 0.0
        right = ell <= I ? revconv[I - ell + 1] : 0.0
        overlap = ell <= I ? cf[ell + 1] * g0 : 0.0
        q[ell + 1] = left + right - overlap
    end

    return p, q
end

function _copy_or_pad_coeffs(C::AbstractMatrix, I, J)
    Itarget = I === nothing ? size(C, 1) - 1 : Int(I)
    Jtarget = J === nothing ? size(C, 2) - 1 : Int(J)
    Itarget < 0 && throw(ArgumentError("I must be nonnegative"))
    Jtarget < 0 && throw(ArgumentError("J must be nonnegative"))

    out = zeros(Float64, Itarget + 1, Jtarget + 1)
    ni = min(size(C, 1), Itarget + 1)
    nj = min(size(C, 2), Jtarget + 1)
    out[1:ni, 1:nj] .= @view(C[1:ni, 1:nj])
    return out
end

"""
    assemble_inner_matrix_blockfft(C, M0, N, sigma; fft_block_size=nothing)

Assemble the inner matrix `S` using streamed fixed-`j` recurrences and the
short-kernel block-FFT helper.
"""
function assemble_inner_matrix_blockfft(C::AbstractMatrix, M0::AbstractMatrix, N::Int,
                                        sigma::Real; fft_block_size = nothing)
    I = size(C, 1) - 1
    J = size(C, 2) - 1
    size(M0, 1) >= J + 1 || throw(ArgumentError("initial moment table has too few rows"))
    size(M0, 2) >= N + 1 || throw(ArgumentError("initial moment table has too few columns"))

    S = zeros(Float64, N + 1, N + 1)
    factor = 2.0^(-Float64(sigma))
    gpad = zeros(Float64, N + I + 1)

    for j in 0:J
        Hj = build_H_for_one_j_from_initial_row(@view(M0[j + 1, :]), N)
        c = @view(C[:, j + 1])

        for k in 0:N
            fill!(gpad, 0.0)
            @inbounds for r in 0:k
                gpad[r + 1] = Hj[r + 1, k + 1]
            end

            p, q = short_hankel_toeplitz_apply_blockfft(
                c, gpad, N;
                block_size = fft_block_size,
                direct_threshold = -1,
            )

            @inbounds for ell in 0:N
                hell = cheb_h(ell)
                weighted_hankel = (pi / 2.0) * p[ell + 1]
                weighted_toeplitz = (pi / 2.0) * q[ell + 1]

                if ell == 0
                    weighted_hankel += (pi / 2.0) * c[1] * gpad[1]
                end

                if ell <= I
                    weighted_toeplitz += (pi / 2.0) * c[ell + 1] * gpad[1]
                end

                S[ell + 1, k + 1] += factor *
                                      (weighted_hankel + weighted_toeplitz) /
                                      (2.0 * hell)
            end
        end
    end

    return S
end


"""
    assemble_inner_matrix_direct_banded(C, M0, N, sigma)

Direct lower-band-aware assembly for the usual short-kernel case.  For a
kernel coefficient matrix with first-index degree `I`, the inner matrix `S`
has no entries below the lower bandwidth `I`; i.e. `S[ell+1,k+1] = 0` when
`ell > k + I`.  This routine therefore loops only over

    ell = 0:min(N, k + I)

for every column `k`.

Compared with the earlier scatter-style direct kernel, this version computes
one output entry at a time, accumulates into a local scalar, and writes each
`S[ell+1,k+1]` only once per fixed `j`.  It also makes the lower-band cut
explicit, which is useful when `beta = 1 - mu` and the final multiplier only
increases the lower bandwidth by one under the public weak-singular
convention.
"""
function assemble_inner_matrix_direct_banded(C::AbstractMatrix, M0::AbstractMatrix,
                                             N::Int, sigma::Real)
    I = size(C, 1) - 1
    J = size(C, 2) - 1
    size(M0, 1) >= J + 1 || throw(ArgumentError("initial moment table has too few rows"))
    size(M0, 2) >= N + 1 || throw(ArgumentError("initial moment table has too few columns"))

    S = zeros(Float64, N + 1, N + 1)
    factor = 2.0^(-Float64(sigma))

    # factor*(pi/2)/(2*h_ell), with h_0=pi and h_ell=pi/2 for ell>0.
    row_scale = fill(0.5 * factor, N + 1)
    row_scale[1] = 0.25 * factor

    @inbounds for j in 0:J
        Hj = build_H_for_one_j_from_initial_row(@view(M0[j + 1, :]), N)
        c = @view(C[:, j + 1])

        for k in 0:N
            col = k + 1
            ellmax = min(N, k + I)

            for ell in 0:ellmax
                acc = 0.0

                # Hankel part: p_ell = sum_i c_i g_{i+ell}, where
                # g_r = H_{j,r,k} is zero for r > k.
                imax_h = min(I, k - ell)
                if imax_h >= 0
                    @inbounds for i in 0:imax_h
                        acc += c[i + 1] * Hj[i + ell + 1, col]
                    end
                end

                # Symmetric Toeplitz part: q_ell = sum_i c_i g_{|i-ell|}.
                # Only indices with |i-ell| <= k can contribute.
                ilo = max(0, ell - k)
                ihi = min(I, ell + k)
                if ilo <= ihi
                    @inbounds for i in ilo:ihi
                        acc += c[i + 1] * Hj[abs(i - ell) + 1, col]
                    end
                end

                # Endpoint corrections used in the original block-FFT formula.
                g0 = Hj[1, col]
                if ell == 0
                    acc += c[1] * g0
                end
                if ell <= I
                    acc += c[ell + 1] * g0
                end

                S[ell + 1, col] += row_scale[ell + 1] * acc
            end
        end
    end

    return S
end

# Backward-compatible alias used by some old benchmark scripts.
assemble_inner_matrix_direct_shortkernel(C::AbstractMatrix, M0::AbstractMatrix,
                                         N::Int, sigma::Real) =
    assemble_inner_matrix_direct_banded(C, M0, N, sigma)


function _precombined_row_scale(N::Int, sigma::Real)
    factor = 2.0^(-Float64(sigma))
    row_scale = fill(0.5 * factor, N + 1)
    row_scale[1] = 0.25 * factor
    return row_scale
end

# Row normalization for the current moment convention. For l>=1 the paper's
# A-entry formula has the Chebyshev factor (2-delta_{r0})/2. The additional
# 2^(-sigma) preserves the existing weighted-basis normalization.
function _four_point_row_scale(N::Int, sigma::Real)
    factor = 2.0^(-Float64(sigma))
    row_scale = fill(factor, N + 1)
    row_scale[1] = 0.5 * factor
    return row_scale
end

function _precombine_kernel_moments(C::AbstractMatrix, M0::AbstractMatrix,
                                    N::Int, J::Int; mode::Symbol = :blas)
    mode in (:blas, :kahan) ||
        throw(ArgumentError("precombine_mode must be :blas or :kahan"))
    size(M0, 1) >= J + 1 || throw(ArgumentError("initial moment table has too few rows"))
    size(M0, 2) >= N + 1 || throw(ArgumentError("initial moment table has too few columns"))

    if mode === :blas
        return Array{Float64}(C) * Array(@view(M0[1:(J + 1), 1:(N + 1)]))
    end

    I1 = size(C, 1)
    Mcomb = zeros(Float64, I1, N + 1)
    @inbounds for i in 1:I1
        for k in 1:(N + 1)
            s = 0.0
            comp = 0.0
            for j in 1:(J + 1)
                p = Float64(C[i, j]) * Float64(M0[j, k])
                t = s + p
                if abs(s) >= abs(p)
                    comp += (s - t) + p
                else
                    comp += (p - t) + s
                end
                s = t
            end
            Mcomb[i, k] = s + comp
        end
    end

    return Mcomb
end

"""
    PackedLowerHessenberg(values, colptr, n, lower_bw)

Flat column-packed storage for an `n x n` matrix with lower bandwidth
`lower_bw`, i.e. entries with `row - col > lower_bw` are structurally zero.
Column `col` stores rows `1:min(n, col+lower_bw)`.
"""
struct PackedLowerHessenberg{T}
    values::Vector{T}
    colptr::Vector{Int}
    n::Int
    lower_bw::Int
end

function packed_lower_hessenberg_zeros(::Type{T}, n::Int, lower_bw::Int) where {T}
    n >= 1 || throw(ArgumentError("n must be positive"))
    lower_bw >= 0 || throw(ArgumentError("lower_bw must be nonnegative"))
    colptr = Vector{Int}(undef, n + 1)
    colptr[1] = 1
    for col in 1:n
        colptr[col + 1] = colptr[col] + min(n, col + lower_bw)
    end
    return PackedLowerHessenberg{T}(zeros(T, colptr[end] - 1), colptr, n, lower_bw)
end

function _packed_lower_hessenberg_uninitialized(::Type{T}, n::Int,
                                                lower_bw::Int) where {T}
    n >= 1 || throw(ArgumentError("n must be positive"))
    lower_bw >= 0 || throw(ArgumentError("lower_bw must be nonnegative"))
    colptr = Vector{Int}(undef, n + 1)
    colptr[1] = 1
    @inbounds for col in 1:n
        colptr[col + 1] = colptr[col] + min(n, col + lower_bw)
    end
    return PackedLowerHessenberg{T}(Vector{T}(undef, colptr[end] - 1),
                                    colptr, n, lower_bw)
end

Base.size(A::PackedLowerHessenberg) = (A.n, A.n)
Base.size(A::PackedLowerHessenberg, d::Int) = d == 1 || d == 2 ? A.n : 1
Base.length(A::PackedLowerHessenberg) = A.n * A.n
Base.eltype(::Type{PackedLowerHessenberg{T}}) where {T} = T
Base.eltype(A::PackedLowerHessenberg{T}) where {T} = T

@inline function Base.getindex(A::PackedLowerHessenberg{T}, row::Int, col::Int) where {T}
    @boundscheck begin
        1 <= row <= A.n || throw(BoundsError(A, (row, col)))
        1 <= col <= A.n || throw(BoundsError(A, (row, col)))
    end
    len = min(A.n, col + A.lower_bw)
    if row <= len
        @inbounds return A.values[A.colptr[col] + row - 1]
    else
        return zero(T)
    end
end

packed_getindex(A::PackedLowerHessenberg, row::Int, col::Int) = A[row, col]

@inline function packed_add!(A::PackedLowerHessenberg, row::Int, col::Int, val)
    @boundscheck begin
        1 <= row <= A.n || throw(BoundsError(A, (row, col)))
        1 <= col <= A.n || throw(BoundsError(A, (row, col)))
    end
    if row <= min(A.n, col + A.lower_bw)
        @inbounds A.values[A.colptr[col] + row - 1] += val
    end
    return nothing
end

@inline function packed_add0!(A::PackedLowerHessenberg, row0::Int, col::Int, val::Float64)
    row = row0 + 1
    @boundscheck begin
        1 <= row <= A.n || throw(BoundsError(A, (row, col)))
        1 <= col <= A.n || throw(BoundsError(A, (row, col)))
    end
    if row <= min(A.n, col + A.lower_bw)
        @inbounds A.values[A.colptr[col] + row - 1] += val
    end
    return nothing
end

"""
    packed_to_dense(A)

Materialize a `PackedLowerHessenberg` matrix as a dense `Matrix`.
"""
function packed_to_dense(A::PackedLowerHessenberg{T}) where {T}
    M = zeros(T, A.n, A.n)
    @inbounds for col in 1:A.n
        len = min(A.n, col + A.lower_bw)
        src = A.colptr[col]
        for row in 1:len
            M[row, col] = A.values[src + row - 1]
        end
    end
    return M
end

"""
    mul!(y, A::PackedLowerHessenberg, x)

Column-wise packed matrix-vector product `y = A*x` without materializing
the dense matrix.
"""
function LinearAlgebra.mul!(y::AbstractVector, A::PackedLowerHessenberg, x::AbstractVector)
    length(x) == A.n || throw(DimensionMismatch("x has length $(length(x)), expected $(A.n)"))
    length(y) == A.n || throw(DimensionMismatch("y has length $(length(y)), expected $(A.n)"))
    fill!(y, zero(eltype(y)))

    vals = A.values
    ptr = A.colptr
    n = A.n
    @inbounds for col in 1:n
        xc = x[col]
        xc == 0 && continue
        first = ptr[col]
        last = ptr[col + 1] - 1
        row = 1
        for p in first:last
            y[row] += vals[p] * xc
            row += 1
        end
    end

    return y
end

Base.:*(A::PackedLowerHessenberg, x::AbstractVector) =
    LinearAlgebra.mul!(zeros(promote_type(eltype(A), eltype(x)), A.n), A, x)

"""
    mul_transpose!(y, A, x)

Compute `y = A' * x` for packed lower-Hessenberg storage.
"""
function mul_transpose!(y::AbstractVector, A::PackedLowerHessenberg, x::AbstractVector)
    length(x) == A.n || throw(DimensionMismatch("x has length $(length(x)), expected $(A.n)"))
    length(y) == A.n || throw(DimensionMismatch("y has length $(length(y)), expected $(A.n)"))
    fill!(y, zero(eltype(y)))

    vals = A.values
    ptr = A.colptr
    n = A.n
    @inbounds for col in 1:n
        acc = zero(promote_type(eltype(vals), eltype(x), eltype(y)))
        first = ptr[col]
        last = ptr[col + 1] - 1
        row = 1
        for p in first:last
            acc += vals[p] * x[row]
            row += 1
        end
        y[col] = acc
    end

    return y
end

@inline function _scatter_S_contribution!(S::AbstractMatrix, row_scale::AbstractVector,
                                          ell::Int, col::Int, val::Float64)
    @inbounds S[ell + 1, col] += row_scale[ell + 1] * val
    return nothing
end

@inline function _scatter_S_contribution!(S::PackedLowerHessenberg, row_scale::AbstractVector,
                                          ell::Int, col::Int, val::Float64)
    packed_add0!(S, ell, col, row_scale[ell + 1] * val)
    return nothing
end

@inline function _scatter_eta1_contribution!(V::AbstractMatrix, row_scale::AbstractVector,
                                             ell::Int, col::Int, val::Float64, N::Int)
    s = row_scale[ell + 1] * val
    @inbounds V[ell + 1, col] += 0.5 * s
    if ell == 0
        if N >= 1
            @inbounds V[2, col] += 0.5 * s
        end
    else
        @inbounds V[ell, col] += 0.25 * s
        if ell < N
            @inbounds V[ell + 2, col] += 0.25 * s
        end
    end
    return nothing
end

@inline function _scatter_eta1_contribution!(V::PackedLowerHessenberg, row_scale::AbstractVector,
                                             ell::Int, col::Int, val::Float64, N::Int)
    s = row_scale[ell + 1] * val
    packed_add0!(V, ell, col, 0.5 * s)
    if ell == 0
        if N >= 1
            packed_add0!(V, 1, col, 0.5 * s)
        end
    else
        packed_add0!(V, ell - 1, col, 0.25 * s)
        if ell < N
            packed_add0!(V, ell + 1, col, 0.25 * s)
        end
    end
    return nothing
end

@inline function _scatter_integer_eta_contribution!(V::AbstractMatrix,
                                                    row_scale::AbstractVector,
                                                    ell::Int, col::Int,
                                                    val::Float64, N::Int,
                                                    g::AbstractVector)
    s = row_scale[ell + 1] * val
    p = length(g) - 1

    @inbounds V[ell + 1, col] += g[1] * s
    @inbounds for q in 1:p
        gq = g[q + 1]
        if ell == 0
            if q <= N
                V[q + 1, col] += gq * s
            end
        else
            rowp = ell + q
            if rowp <= N
                V[rowp + 1, col] += 0.5 * gq * s
            end
            rowm = ell >= q ? ell - q : q - ell
            if rowm <= N
                V[rowm + 1, col] += 0.5 * gq * s
            end
        end
    end

    return nothing
end

@inline function _scatter_integer_eta_contribution!(V::PackedLowerHessenberg,
                                                    row_scale::AbstractVector,
                                                    ell::Int, col::Int,
                                                    val::Float64, N::Int,
                                                    g::AbstractVector)
    s = row_scale[ell + 1] * val
    p = length(g) - 1

    packed_add0!(V, ell, col, g[1] * s)
    @inbounds for q in 1:p
        gq = g[q + 1]
        if ell == 0
            if q <= N
                packed_add0!(V, q, col, gq * s)
            end
        else
            rowp = ell + q
            if rowp <= N
                packed_add0!(V, rowp, col, 0.5 * gq * s)
            end
            rowm = ell >= q ? ell - q : q - ell
            if rowm <= N
                packed_add0!(V, rowm, col, 0.5 * gq * s)
            end
        end
    end

    return nothing
end

function _scatter_precombined_S_columns!(S::AbstractMatrix, Hi::AbstractMatrix,
                                         row_scale::AbstractVector, i::Int,
                                         N::Int, kfirst::Int, klast::Int)
    @inbounds for k in kfirst:klast
        col = k + 1

        # Hankel contribution: r = i + ell, so r starts at i.
        if i <= k
            @simd for r in i:k
                _scatter_S_contribution!(S, row_scale, r - i, col, Hi[r + 1, col])
            end
        end

        # Toeplitz contribution with ell = i + r.
        rmax_p = min(k, N - i)
        @simd for r in 0:rmax_p
            _scatter_S_contribution!(S, row_scale, i + r, col, Hi[r + 1, col])
        end

        # Toeplitz contribution with ell = i - r, excluding r=0.
        rmin_m = max(1, i - N)
        rmax_m = min(k, i)
        @simd for r in rmin_m:rmax_m
            _scatter_S_contribution!(S, row_scale, i - r, col, Hi[r + 1, col])
        end

        # Endpoint corrections from h_0 in the Chebyshev product weights.
        g0 = Hi[1, col]
        if i == 0
            _scatter_S_contribution!(S, row_scale, 0, col, g0)
        end
        if i <= N
            _scatter_S_contribution!(S, row_scale, i, col, g0)
        end
    end

    return S
end

function _scatter_precombined_eta1_columns!(V::AbstractMatrix, Hi::AbstractMatrix,
                                            row_scale::AbstractVector, i::Int,
                                            N::Int, kfirst::Int, klast::Int)
    @inbounds for k in kfirst:klast
        col = k + 1

        if i <= k
            for r in i:k
                _scatter_eta1_contribution!(V, row_scale, r - i, col, Hi[r + 1, col], N)
            end
        end

        rmax_p = min(k, N - i)
        for r in 0:rmax_p
            _scatter_eta1_contribution!(V, row_scale, i + r, col, Hi[r + 1, col], N)
        end

        rmin_m = max(1, i - N)
        rmax_m = min(k, i)
        for r in rmin_m:rmax_m
            _scatter_eta1_contribution!(V, row_scale, i - r, col, Hi[r + 1, col], N)
        end

        g0 = Hi[1, col]
        if i == 0
            _scatter_eta1_contribution!(V, row_scale, 0, col, g0, N)
        end
        if i <= N
            _scatter_eta1_contribution!(V, row_scale, i, col, g0, N)
        end
    end

    return V
end

function _scatter_precombined_integer_eta_columns!(V, Hi::AbstractMatrix,
                                                   row_scale::AbstractVector,
                                                   i::Int, N::Int,
                                                   kfirst::Int, klast::Int,
                                                   g::AbstractVector)
    @inbounds for k in kfirst:klast
        col = k + 1

        if i <= k
            for r in i:k
                _scatter_integer_eta_contribution!(V, row_scale, r - i, col,
                                                   Hi[r + 1, col], N, g)
            end
        end

        rmax_p = min(k, N - i)
        for r in 0:rmax_p
            _scatter_integer_eta_contribution!(V, row_scale, i + r, col,
                                               Hi[r + 1, col], N, g)
        end

        rmin_m = max(1, i - N)
        rmax_m = min(k, i)
        for r in rmin_m:rmax_m
            _scatter_integer_eta_contribution!(V, row_scale, i - r, col,
                                               Hi[r + 1, col], N, g)
        end

        g0 = Hi[1, col]
        if i == 0
            _scatter_integer_eta_contribution!(V, row_scale, 0, col, g0, N, g)
        end
        if i <= N
            _scatter_integer_eta_contribution!(V, row_scale, i, col, g0, N, g)
        end
    end

    return V
end

@inline function _packed_eta1_add!(values::Vector{Float64}, base::Int,
                                   ell::Int, s::Float64, N::Int)
    @inbounds values[base + ell] += 0.5 * s
    if ell == 0
        if N >= 1
            @inbounds values[base + 1] += 0.5 * s
        end
    else
        @inbounds values[base + ell - 1] += 0.25 * s
        if ell < N
            @inbounds values[base + ell + 1] += 0.25 * s
        end
    end
    return nothing
end

function _scatter_precombined_eta1_columns!(V::PackedLowerHessenberg{Float64},
                                            Hi::AbstractMatrix,
                                            row_scale::AbstractVector, i::Int,
                                            N::Int, kfirst::Int, klast::Int)
    vals = V.values
    ptr = V.colptr

    @inbounds for k in kfirst:klast
        col = k + 1
        base = ptr[col]

        if i <= k
            for r in i:k
                _packed_eta1_add!(vals, base, r - i,
                                  row_scale[r - i + 1] * Hi[r + 1, col], N)
            end
        end

        rmax_p = min(k, N - i)
        for r in 0:rmax_p
            _packed_eta1_add!(vals, base, i + r,
                              row_scale[i + r + 1] * Hi[r + 1, col], N)
        end

        rmin_m = max(1, i - N)
        rmax_m = min(k, i)
        for r in rmin_m:rmax_m
            _packed_eta1_add!(vals, base, i - r,
                              row_scale[i - r + 1] * Hi[r + 1, col], N)
        end

        g0 = Hi[1, col]
        if i == 0
            _packed_eta1_add!(vals, base, 0, row_scale[1] * g0, N)
        end
        if i <= N
            _packed_eta1_add!(vals, base, i, row_scale[i + 1] * g0, N)
        end
    end

    return V
end

struct PrecombinedThreadWorkspace
    scratch::Vector{Float64}
end

_make_precombined_workspaces() =
    [PrecombinedThreadWorkspace(Float64[]) for _ in 1:Threads.nthreads()]

_effective_thread_chunk_size(N::Int, requested::Int) =
    requested > 0 ? requested : max(8, cld(N + 1, 8 * max(Threads.nthreads(), 1)))

function _check_thread_schedule(schedule::Symbol)
    schedule in (:static, :reverse_static, :chunked, :reverse_chunked) ||
        throw(ArgumentError("thread_schedule must be one of :static, :reverse_static, :chunked, :reverse_chunked"))
    return schedule
end

function _scatter_precombined_S!(S::AbstractMatrix, Hi::AbstractMatrix,
                                 row_scale::AbstractVector, i::Int, N::Int,
                                 threaded::Bool, thread_schedule::Symbol,
                                 thread_chunk_size::Int, workspaces)
    if threaded
        schedule = _check_thread_schedule(thread_schedule)
        if schedule === :static || schedule === :reverse_static
            Threads.@threads for kk in 0:N
                ws = workspaces[Threads.threadid()]
                ws === nothing && nothing
                k = schedule === :reverse_static ? N - kk : kk
                _scatter_precombined_S_columns!(S, Hi, row_scale, i, N, k, k)
            end
        else
            chunk = _effective_thread_chunk_size(N, thread_chunk_size)
            nchunks = cld(N + 1, chunk)
            counter = Threads.Atomic{Int}(1)
            Threads.@threads for _ in 1:Threads.nthreads()
                ws = workspaces[Threads.threadid()]
                ws === nothing && nothing
                while true
                    cc = Threads.atomic_add!(counter, 1)
                    cc > nchunks && break
                    cidx = schedule === :reverse_chunked ? nchunks - cc + 1 : cc
                    kfirst = (cidx - 1) * chunk
                    klast = min(N, kfirst + chunk - 1)
                    _scatter_precombined_S_columns!(S, Hi, row_scale, i, N, kfirst, klast)
                end
            end
        end
    else
        _scatter_precombined_S_columns!(S, Hi, row_scale, i, N, 0, N)
    end
    return S
end

function _scatter_precombined_eta1!(V, Hi::AbstractMatrix,
                                    row_scale::AbstractVector, i::Int, N::Int,
                                    threaded::Bool, thread_schedule::Symbol,
                                    thread_chunk_size::Int, workspaces)
    if threaded
        schedule = _check_thread_schedule(thread_schedule)
        if schedule === :static || schedule === :reverse_static
            Threads.@threads for kk in 0:N
                ws = workspaces[Threads.threadid()]
                ws === nothing && nothing
                k = schedule === :reverse_static ? N - kk : kk
                _scatter_precombined_eta1_columns!(V, Hi, row_scale, i, N, k, k)
            end
        else
            chunk = _effective_thread_chunk_size(N, thread_chunk_size)
            nchunks = cld(N + 1, chunk)
            counter = Threads.Atomic{Int}(1)
            Threads.@threads for _ in 1:Threads.nthreads()
                ws = workspaces[Threads.threadid()]
                ws === nothing && nothing
                while true
                    cc = Threads.atomic_add!(counter, 1)
                    cc > nchunks && break
                    cidx = schedule === :reverse_chunked ? nchunks - cc + 1 : cc
                    kfirst = (cidx - 1) * chunk
                    klast = min(N, kfirst + chunk - 1)
                    _scatter_precombined_eta1_columns!(V, Hi, row_scale, i, N, kfirst, klast)
                end
            end
        end
    else
        _scatter_precombined_eta1_columns!(V, Hi, row_scale, i, N, 0, N)
    end
    return V
end

function _scatter_precombined_integer_eta!(V, Hi::AbstractMatrix,
                                           row_scale::AbstractVector, i::Int,
                                           N::Int, g::AbstractVector,
                                           threaded::Bool,
                                           thread_schedule::Symbol,
                                           thread_chunk_size::Int, workspaces)
    if threaded
        schedule = _check_thread_schedule(thread_schedule)
        if schedule === :static || schedule === :reverse_static
            Threads.@threads for kk in 0:N
                ws = workspaces[Threads.threadid()]
                ws === nothing && nothing
                k = schedule === :reverse_static ? N - kk : kk
                _scatter_precombined_integer_eta_columns!(V, Hi, row_scale, i,
                                                          N, k, k, g)
            end
        else
            chunk = _effective_thread_chunk_size(N, thread_chunk_size)
            nchunks = cld(N + 1, chunk)
            counter = Threads.Atomic{Int}(1)
            Threads.@threads for _ in 1:Threads.nthreads()
                ws = workspaces[Threads.threadid()]
                ws === nothing && nothing
                while true
                    cc = Threads.atomic_add!(counter, 1)
                    cc > nchunks && break
                    cidx = schedule === :reverse_chunked ? nchunks - cc + 1 : cc
                    kfirst = (cidx - 1) * chunk
                    klast = min(N, kfirst + chunk - 1)
                    _scatter_precombined_integer_eta_columns!(V, Hi, row_scale,
                                                              i, N, kfirst,
                                                              klast, g)
                end
            end
        end
    else
        _scatter_precombined_integer_eta_columns!(V, Hi, row_scale, i, N, 0, N, g)
    end
    return V
end

"""
    assemble_inner_matrix_precombined(C, M0, N, sigma; threaded=false)

Legacy scalar-i precombined-j inner assembly. It forms
`Mcomb = C*M0`, reuses one triangular `H` buffer for all kernel-z modes, and
uses split Hankel/Toeplitz loops to avoid `abs` and most per-entry branches.
"""
function assemble_inner_matrix_precombined(C::AbstractMatrix, M0::AbstractMatrix,
                                           N::Int, sigma::Real; threaded::Bool = false,
                                           thread_schedule::Symbol = :reverse_static,
                                           thread_chunk_size::Int = 0,
                                           precombine_mode::Symbol = :blas)
    I = size(C, 1) - 1
    J = size(C, 2) - 1
    size(M0, 1) >= J + 1 || throw(ArgumentError("initial moment table has too few rows"))
    size(M0, 2) >= N + 1 || throw(ArgumentError("initial moment table has too few columns"))

    S = zeros(Float64, N + 1, N + 1)
    row_scale = _precombined_row_scale(N, sigma)
    Mcomb = _precombine_kernel_moments(C, M0, N, J; mode = precombine_mode)
    Hi = Matrix{Float64}(undef, N + 1, N + 1)
    workspaces = threaded ? _make_precombined_workspaces() : nothing

    @inbounds for i in 0:I
        build_H_for_one_j_from_initial_row!(Hi, @view(Mcomb[i + 1, :]), N)
        _scatter_precombined_S!(S, Hi, row_scale, i, N, threaded,
                                thread_schedule, thread_chunk_size, workspaces)
    end

    return S
end

"""
    assemble_matrix_precombined_eta1(C, M0, N, sigma; threaded=false)

Assemble the final matrix directly for `eta=1`, fusing the sparse multiplier
by `(1+z)/2` into the precombined-j scatter.  This avoids materializing `S`
and skips the separate tridiagonal multiplier pass.
"""
function assemble_matrix_precombined_eta1(C::AbstractMatrix, M0::AbstractMatrix,
                                          N::Int, sigma::Real; threaded::Bool = false,
                                          thread_schedule::Symbol = :reverse_static,
                                          thread_chunk_size::Int = 0,
                                          output_format::Symbol = :dense,
                                          lower_bw::Int = size(C, 1),
                                          precombine_mode::Symbol = :blas)
    I = size(C, 1) - 1
    J = size(C, 2) - 1
    size(M0, 1) >= J + 1 || throw(ArgumentError("initial moment table has too few rows"))
    size(M0, 2) >= N + 1 || throw(ArgumentError("initial moment table has too few columns"))

    n = N + 1
    V = if output_format === :dense
        zeros(Float64, n, n)
    elseif output_format === :packed_lower_hessenberg
        packed_lower_hessenberg_zeros(Float64, n, lower_bw)
    else
        throw(ArgumentError("output_format must be :dense or :packed_lower_hessenberg"))
    end
    row_scale = _precombined_row_scale(N, sigma)
    Mcomb = _precombine_kernel_moments(C, M0, N, J; mode = precombine_mode)
    Hi = Matrix{Float64}(undef, N + 1, N + 1)
    workspaces = threaded ? _make_precombined_workspaces() : nothing

    @inbounds for i in 0:I
        build_H_for_one_j_from_initial_row!(Hi, @view(Mcomb[i + 1, :]), N)
        _scatter_precombined_eta1!(V, Hi, row_scale, i, N, threaded,
                                   thread_schedule, thread_chunk_size, workspaces)
    end

    return V
end

"""
    assemble_matrix_precombined_integer_eta(C, M0, N, sigma, g; ...)

Legacy scalar-i fused assembly for an integer Chebyshev multiplier
`g(z)=sum_q g[q+1]T_q(z)`.  This is the compatibility path used when the
caller explicitly requests `H_recurrence_mode=:legacy_six_term` (or its
backward-compatible `:scalar_i` alias).
"""
function assemble_matrix_precombined_integer_eta(C::AbstractMatrix, M0::AbstractMatrix,
                                                 N::Int, sigma::Real,
                                                 g::AbstractVector;
                                                 threaded::Bool = false,
                                                 thread_schedule::Symbol = :reverse_static,
                                                 thread_chunk_size::Int = 0,
                                                 output_format::Symbol = :dense,
                                                 lower_bw::Int = (size(C, 1) - 1) + (length(g) - 1),
                                                 precombine_mode::Symbol = :blas)
    I = size(C, 1) - 1
    J = size(C, 2) - 1
    size(M0, 1) >= J + 1 || throw(ArgumentError("initial moment table has too few rows"))
    size(M0, 2) >= N + 1 || throw(ArgumentError("initial moment table has too few columns"))

    n = N + 1
    V = if output_format === :dense
        zeros(Float64, n, n)
    elseif output_format === :packed_lower_hessenberg
        packed_lower_hessenberg_zeros(Float64, n, lower_bw)
    else
        throw(ArgumentError("output_format must be :dense or :packed_lower_hessenberg"))
    end
    row_scale = _precombined_row_scale(N, sigma)
    Mcomb = _precombine_kernel_moments(C, M0, N, J; mode = precombine_mode)
    Hi = Matrix{Float64}(undef, N + 1, N + 1)
    workspaces = threaded ? _make_precombined_workspaces() : nothing

    @inbounds for i in 0:I
        build_H_for_one_j_from_initial_row!(Hi, @view(Mcomb[i + 1, :]), N)
        _scatter_precombined_integer_eta!(V, Hi, row_scale, i, N, g,
                                          threaded, thread_schedule,
                                          thread_chunk_size, workspaces)
    end

    return V
end

function _blocki_scatter_eta1_layer!(V::AbstractMatrix, Hrow::AbstractMatrix,
                                     row_scale::AbstractVector, k::Int,
                                     I::Int, N::Int)
    @inbounds for ell in k:N
        col = ell + 1

        # The l=0 column is A[r,0] = Lambda[r,0,0], without the l>=1
        # Chebyshev row factor. Compensate the r=0 half normalization here.
        if ell == 0
            for i in 0:min(I, N)
                val = (i == 0 ? 2.0 : 1.0) * Hrow[i + 1, 1]
                _scatter_eta1_contribution!(V, row_scale, i, 1, val, N)
            end
            continue
        end

        moment_scale = Float64(ell) / Float64(ell + k)
        for i in 0:I
            val = moment_scale * Hrow[i + 1, col]

            # First A-entry term: k = abs(i-r).
            rowp = i + k
            if rowp <= N
                _scatter_eta1_contribution!(V, row_scale, rowp, col, val, N)
            end
            if k > 0 && i >= k && i - k <= N
                _scatter_eta1_contribution!(V, row_scale, i - k, col, val, N)
            end

            # Second A-entry term: k = i+r.
            if k >= i
                _scatter_eta1_contribution!(V, row_scale, k - i, col, val, N)
            end
        end
    end

    return V
end

@inline function _packed_blocki_eta1_add!(values::Vector{Float64}, base::Int,
                                          row_scale::AbstractVector, ell::Int,
                                          val::Float64, N::Int)
    _packed_eta1_add!(values, base, ell, row_scale[ell + 1] * val, N)
    return nothing
end

function _blocki_scatter_eta1_layer!(V::PackedLowerHessenberg{Float64},
                                     Hrow::AbstractMatrix, row_scale::AbstractVector,
                                     k::Int, I::Int, N::Int)
    vals = V.values
    ptr = V.colptr

    @inbounds for ell in k:N
        col = ell + 1
        base = ptr[col]

        if ell == 0
            for i in 0:min(I, N)
                val = (i == 0 ? 2.0 : 1.0) * Hrow[i + 1, 1]
                _packed_blocki_eta1_add!(vals, base, row_scale, i, val, N)
            end
            continue
        end

        moment_scale = Float64(ell) / Float64(ell + k)
        for i in 0:I
            val = moment_scale * Hrow[i + 1, col]

            rowp = i + k
            if rowp <= N
                _packed_blocki_eta1_add!(vals, base, row_scale, rowp, val, N)
            end
            if k > 0 && i >= k && i - k <= N
                _packed_blocki_eta1_add!(vals, base, row_scale, i - k, val, N)
            end
            if k >= i
                _packed_blocki_eta1_add!(vals, base, row_scale, k - i, val, N)
            end
        end
    end

    return V
end

function _blocki_scatter_integer_eta_layer!(V, Hrow::AbstractMatrix,
                                            row_scale::AbstractVector, k::Int,
                                            I::Int, N::Int, g::AbstractVector)
    @inbounds for ell in k:N
        col = ell + 1

        if ell == 0
            for i in 0:min(I, N)
                val = (i == 0 ? 2.0 : 1.0) * Hrow[i + 1, 1]
                _scatter_integer_eta_contribution!(V, row_scale, i, 1, val, N, g)
            end
            continue
        end

        moment_scale = Float64(ell) / Float64(ell + k)
        for i in 0:I
            val = moment_scale * Hrow[i + 1, col]

            rowp = i + k
            if rowp <= N
                _scatter_integer_eta_contribution!(V, row_scale, rowp, col, val, N, g)
            end
            if k > 0 && i >= k && i - k <= N
                _scatter_integer_eta_contribution!(V, row_scale, i - k, col, val, N, g)
            end
            if k >= i
                _scatter_integer_eta_contribution!(V, row_scale, k - i, col, val, N, g)
            end
        end
    end

    return V
end

function _blocki_scatter_S_layer!(S, Hrow::AbstractMatrix,
                                  row_scale::AbstractVector, k::Int,
                                  I::Int, N::Int)
    @inbounds for ell in k:N
        col = ell + 1

        if ell == 0
            for i in 0:min(I, N)
                val = (i == 0 ? 2.0 : 1.0) * Hrow[i + 1, 1]
                _scatter_S_contribution!(S, row_scale, i, 1, val)
            end
            continue
        end

        moment_scale = Float64(ell) / Float64(ell + k)
        for i in 0:I
            val = moment_scale * Hrow[i + 1, col]

            rowp = i + k
            if rowp <= N
                _scatter_S_contribution!(S, row_scale, rowp, col, val)
            end
            if k > 0 && i >= k && i - k <= N
                _scatter_S_contribution!(S, row_scale, i - k, col, val)
            end
            if k >= i
                _scatter_S_contribution!(S, row_scale, k - i, col, val)
            end
        end
    end

    return S
end

function _blocki_next_layer!(Hnext::AbstractMatrix, Hcur::AbstractMatrix,
                             k::Int, N::Int, I1::Int)
    k == N && return Hnext

    # Boundary Lambda[:,k+1,k] = 0. The ell loop is forward because the new
    # layer at ell depends on the new layer at ell-1.
    @inbounds @simd for ii in 1:I1
        Hnext[ii, k + 1] = 0.0
    end

    @inbounds for ell in (k + 1):N
        ellf = Float64(ell)
        kf = Float64(k)
        invden = 1.0 / (ellf + kf)
        a = (ellf - kf) * invden
        c = -(ellf - kf - 2.0) * invden

        @simd for ii in 1:I1
            Hnext[ii, ell + 1] =
                a * Hcur[ii, ell + 1] +
                Hcur[ii, ell] +
                c * Hnext[ii, ell]
        end
    end

    return Hnext
end

@inline function _eta1_column_entry(Acol::Vector{Float64}, row::Int,
                                    max_A_row::Int, N::Int)
    center = row <= max_A_row ? 0.5 * Acol[row + 1] : 0.0
    if row == 0
        return center + (max_A_row >= 1 ? 0.25 * Acol[2] : 0.0)
    elseif row == 1
        return center + 0.5 * Acol[1] +
               (max_A_row >= 2 ? 0.25 * Acol[3] : 0.0)
    else
        return center + 0.25 * Acol[row] +
               (row < N && row < max_A_row ? 0.25 * Acol[row + 2] : 0.0)
    end
end

function _write_eta1_column!(V::PackedLowerHessenberg{Float64},
                             Acol::Vector{Float64}, ell::Int,
                             max_A_row::Int, N::Int)
    col = ell + 1
    first = V.colptr[col]
    len = V.colptr[col + 1] - first
    vals = V.values
    @inbounds @simd for row in 0:(len - 1)
        vals[first + row] = _eta1_column_entry(Acol, row, max_A_row, N)
    end
    return V
end

function _write_eta1_column!(V::Matrix{Float64}, Acol::Vector{Float64},
                             ell::Int, max_A_row::Int, N::Int)
    col = ell + 1
    max_G_row = min(N, max_A_row + 1)
    @inbounds @simd for row in 0:max_G_row
        V[row + 1, col] = _eta1_column_entry(Acol, row, max_A_row, N)
    end
    return V
end

function _ellmajor_eta1_output(output_format::Symbol, n::Int, lower_bw::Int)
    if output_format === :dense
        return zeros(Float64, n, n)
    elseif output_format === :packed_lower_hessenberg
        # Every stored element is assigned exactly once, column by column, by
        # `_write_eta1_column!`; avoid zeroing the potentially hundreds of MiB
        # packed value buffer before overwriting it.
        return _packed_lower_hessenberg_uninitialized(Float64, n, lower_bw)
    else
        throw(ArgumentError("output_format must be :dense or :packed_lower_hessenberg"))
    end
end

function _ellmajor_inverse_integers(N::Int)
    invint = Vector{Float64}(undef, 2N)
    @inbounds @simd for d in 1:(2N)
        invint[d] = 1.0 / Float64(d)
    end
    return invint
end

"""
    _assemble_eta1_ellmajor_scatter_from_Mcomb(Mcomb, N, sigma; ...)

Column-major four-point assembly for the production eta=1 path. Each completed
moment column is scattered only into a cache-resident local A column,
normalized once, multiplied by `(1+z)/2` once, and written sequentially to the
final output.
"""
function _assemble_eta1_ellmajor_scatter_from_Mcomb(
    Mcomb::AbstractMatrix,
    N::Int,
    sigma::Real;
    output_format::Symbol = :packed_lower_hessenberg,
    lower_bw::Int = size(Mcomb, 1),
    use_reciprocal_table::Bool = true,
)
    return _assemble_eta1_ellmajor_scatter_core(
        Mcomb, N, sigma, Val(use_reciprocal_table);
        output_format = output_format,
        lower_bw = lower_bw,
    )
end

function _assemble_eta1_ellmajor_scatter_core(
    Mcomb::AbstractMatrix,
    N::Int,
    sigma::Real,
    ::Val{use_reciprocal_table};
    output_format::Symbol,
    lower_bw::Int,
) where {use_reciprocal_table}
    I = size(Mcomb, 1) - 1
    I1 = I + 1
    size(Mcomb, 2) >= N + 1 || throw(ArgumentError("Mcomb has too few columns"))

    V = _ellmajor_eta1_output(output_format, N + 1, lower_bw)
    base = Mcomb isa Matrix{Float64} ? Mcomb : Array{Float64}(Mcomb)
    prev = base
    curr = Matrix{Float64}(undef, I1, N + 1)
    Acol = Vector{Float64}(undef, N + 1)
    invint = use_reciprocal_table ? _ellmajor_inverse_integers(N) : Float64[]
    factor = 2.0^(-Float64(sigma))

    max_A_row = min(I, N)
    @inbounds @simd for i in 0:max_A_row
        Acol[i + 1] = factor * Float64(Mcomb[i + 1, 1])
    end
    _write_eta1_column!(V, Acol, 0, max_A_row, N)
    N == 0 && return V

    @inbounds @simd for ii in 1:I1
        prev[ii, 1] = Float64(Mcomb[ii, 1])
    end

    @inbounds for ell in 1:N
        max_A_row = min(N, ell + I)
        @simd for row in 0:max_A_row
            Acol[row + 1] = 0.0
        end
        @simd for ii in 1:I1
            curr[ii, 1] = base[ii, ell + 1]
            prev[ii, ell + 1] = 0.0
        end

        ellf = Float64(ell)
        for k in 0:ell
            invden = use_reciprocal_table ? invint[ell + k] :
                     1.0 / Float64(ell + k)
            moment_scale = ellf * invden

            if I <= N && k > I
                # In the dominant large-k region the negative-difference
                # branch is impossible and the sum/difference rows are known
                # directly.  Handle i=0 separately because both terms target
                # the same A entry, then SIMD the disjoint row pairs.
                val0 = moment_scale * curr[1, k + 1]
                Acol[k + 1] += 2.0 * val0
                if k <= N - I
                    @simd for i in 1:I
                        val = moment_scale * curr[i + 1, k + 1]
                        Acol[k + i + 1] += val
                        Acol[k - i + 1] += val
                    end
                else
                    imax = N - k
                    @simd for i in 1:imax
                        val = moment_scale * curr[i + 1, k + 1]
                        Acol[k + i + 1] += val
                        Acol[k - i + 1] += val
                    end
                    @simd for i in (imax + 1):I
                        Acol[k - i + 1] +=
                            moment_scale * curr[i + 1, k + 1]
                    end
                end
            else
                for i in 0:I
                    val = moment_scale * curr[i + 1, k + 1]
                    rowp = i + k
                    if rowp <= N
                        Acol[rowp + 1] += val
                    end
                    if k > 0 && i >= k && i - k <= N
                        Acol[i - k + 1] += val
                    end
                    if k >= i
                        Acol[k - i + 1] += val
                    end
                end
            end

            if k < ell
                kf = Float64(k)
                a = (ellf - kf) * invden
                c = -(ellf - kf - 2.0) * invden
                @simd for ii in 1:I1
                    curr[ii, k + 2] =
                        a * curr[ii, k + 1] +
                        prev[ii, k + 1] +
                        c * prev[ii, k + 2]
                end
            end
        end

        Acol[1] *= 0.5 * factor
        @simd for row in 1:max_A_row
            Acol[row + 1] *= factor
        end
        _write_eta1_column!(V, Acol, ell, max_A_row, N)
        prev, curr = curr, prev
    end

    return V
end

"""
    _assemble_eta1_ellmajor_gather_from_Mcomb(Mcomb, N, sigma; ...)

Experimental column-major four-point assembly using one direct A-entry gather
per output row after the complete contracted-moment column has been generated.
"""
function _assemble_eta1_ellmajor_gather_from_Mcomb(
    Mcomb::AbstractMatrix,
    N::Int,
    sigma::Real;
    output_format::Symbol = :packed_lower_hessenberg,
    lower_bw::Int = size(Mcomb, 1),
)
    I = size(Mcomb, 1) - 1
    I1 = I + 1
    size(Mcomb, 2) >= N + 1 || throw(ArgumentError("Mcomb has too few columns"))

    V = _ellmajor_eta1_output(output_format, N + 1, lower_bw)
    prev = Matrix{Float64}(undef, I1, N + 1)
    curr = Matrix{Float64}(undef, I1, N + 1)
    Acol = Vector{Float64}(undef, N + 1)
    invint = _ellmajor_inverse_integers(N)
    factor = 2.0^(-Float64(sigma))

    max_A_row = min(I, N)
    @inbounds @simd for i in 0:max_A_row
        Acol[i + 1] = factor * Float64(Mcomb[i + 1, 1])
    end
    _write_eta1_column!(V, Acol, 0, max_A_row, N)
    N == 0 && return V

    @inbounds @simd for ii in 1:I1
        prev[ii, 1] = Float64(Mcomb[ii, 1])
    end

    @inbounds for ell in 1:N
        @simd for ii in 1:I1
            curr[ii, 1] = Float64(Mcomb[ii, ell + 1])
            prev[ii, ell + 1] = 0.0
        end

        ellf = Float64(ell)
        for k in 0:(ell - 1)
            invden = invint[ell + k]
            kf = Float64(k)
            a = (ellf - kf) * invden
            c = -(ellf - kf - 2.0) * invden
            @simd for ii in 1:I1
                curr[ii, k + 2] =
                    a * curr[ii, k + 1] +
                    prev[ii, k + 1] +
                    c * prev[ii, k + 2]
            end
        end

        max_A_row = min(N, ell + I)
        for row in 0:max_A_row
            acc = 0.0
            for i in 0:I
                kabs = abs(i - row)
                if kabs <= ell
                    acc += ellf * invint[ell + kabs] * curr[i + 1, kabs + 1]
                end
                ksum = i + row
                if ksum <= ell
                    acc += ellf * invint[ell + ksum] * curr[i + 1, ksum + 1]
                end
            end
            Acol[row + 1] = factor * (row == 0 ? 0.5 : 1.0) * acc
        end

        _write_eta1_column!(V, Acol, ell, max_A_row, N)
        prev, curr = curr, prev
    end

    return V
end

"""
    assemble_matrix_precombined_eta1_blocki(C, M0, N, sigma; output_format=:dense)

Current fused-eta1 assembly. It precombines the kernel's j index once, then
uses the column-major four-point recurrence so that each completed A column is
assembled locally, multiplied by `(1+z)/2` once, and written sequentially.
"""
function assemble_matrix_precombined_eta1_blocki(C::AbstractMatrix, M0::AbstractMatrix,
                                                 N::Int, sigma::Real;
                                                 output_format::Symbol = :dense,
                                                 lower_bw::Int = size(C, 1),
                                                 precombine_mode::Symbol = :blas)
    J = size(C, 2) - 1
    size(M0, 1) >= J + 1 || throw(ArgumentError("initial moment table has too few rows"))
    size(M0, 2) >= N + 1 || throw(ArgumentError("initial moment table has too few columns"))
    Mcomb = _precombine_kernel_moments(C, M0, N, J; mode = precombine_mode)
    return _assemble_eta1_ellmajor_scatter_from_Mcomb(
        Mcomb,
        N,
        sigma;
        output_format = output_format,
        lower_bw = lower_bw,
    )
end

function _assemble_matrix_precombined_eta1_kmajor_from_Mcomb(Mcomb::AbstractMatrix,
                                                             N::Int, sigma::Real;
                                                             output_format::Symbol = :dense,
                                                             lower_bw::Int = size(Mcomb, 1))
    I = size(Mcomb, 1) - 1
    I1 = I + 1
    size(Mcomb, 2) >= N + 1 || throw(ArgumentError("Mcomb has too few columns"))

    n = N + 1
    V = if output_format === :dense
        zeros(Float64, n, n)
    elseif output_format === :packed_lower_hessenberg
        packed_lower_hessenberg_zeros(Float64, n, lower_bw)
    else
        throw(ArgumentError("output_format must be :dense or :packed_lower_hessenberg"))
    end

    row_scale = _four_point_row_scale(N, sigma)
    Hcur = Mcomb isa Matrix{Float64} ? Mcomb : Array{Float64}(Mcomb)
    Hnext = Matrix{Float64}(undef, I1, N + 1)

    for r in 0:N
        _blocki_scatter_eta1_layer!(V, Hcur, row_scale, r, I, N)
        r == N && break
        _blocki_next_layer!(Hnext, Hcur, r, N, I1)
        Hcur, Hnext = Hnext, Hcur
    end

    return V
end

function assemble_matrix_precombined_eta1_blocki_from_Mcomb(
    Mcomb::AbstractMatrix,
    N::Int,
    sigma::Real;
    output_format::Symbol = :dense,
    lower_bw::Int = size(Mcomb, 1),
)
    return _assemble_eta1_ellmajor_scatter_from_Mcomb(
        Mcomb,
        N,
        sigma;
        output_format = output_format,
        lower_bw = lower_bw,
    )
end

"""
    assemble_matrix_precombined_integer_eta_blocki(C, M0, N, sigma, g; ...)

Block-i fused assembly for a small integer multiplier.  The recurrence advances
all precombined kernel-z modes together and scatters each layer directly into
the final dense or packed output.
"""
function assemble_matrix_precombined_integer_eta_blocki(C::AbstractMatrix, M0::AbstractMatrix,
                                                        N::Int, sigma::Real,
                                                        g::AbstractVector;
                                                        output_format::Symbol = :dense,
                                                        lower_bw::Int = (size(C, 1) - 1) + (length(g) - 1),
                                                        precombine_mode::Symbol = :blas)
    I = size(C, 1) - 1
    J = size(C, 2) - 1
    I1 = I + 1
    size(M0, 1) >= J + 1 || throw(ArgumentError("initial moment table has too few rows"))
    size(M0, 2) >= N + 1 || throw(ArgumentError("initial moment table has too few columns"))

    n = N + 1
    V = if output_format === :dense
        zeros(Float64, n, n)
    elseif output_format === :packed_lower_hessenberg
        packed_lower_hessenberg_zeros(Float64, n, lower_bw)
    else
        throw(ArgumentError("output_format must be :dense or :packed_lower_hessenberg"))
    end

    row_scale = _four_point_row_scale(N, sigma)
    Hcur = _precombine_kernel_moments(C, M0, N, J; mode = precombine_mode)
    Hnext = Matrix{Float64}(undef, I1, N + 1)

    for r in 0:N
        _blocki_scatter_integer_eta_layer!(V, Hcur, row_scale, r, I, N, g)
        r == N && break
        _blocki_next_layer!(Hnext, Hcur, r, N, I1)
        Hcur, Hnext = Hnext, Hcur
    end

    return V
end

function assemble_matrix_precombined_integer_eta_blocki_from_Mcomb(Mcomb::AbstractMatrix,
                                                                   N::Int, sigma::Real,
                                                                   g::AbstractVector;
                                                                   output_format::Symbol = :dense,
                                                                   lower_bw::Int = (size(Mcomb, 1) - 1) + (length(g) - 1))
    I = size(Mcomb, 1) - 1
    I1 = I + 1
    size(Mcomb, 2) >= N + 1 || throw(ArgumentError("Mcomb has too few columns"))

    n = N + 1
    V = if output_format === :dense
        zeros(Float64, n, n)
    elseif output_format === :packed_lower_hessenberg
        packed_lower_hessenberg_zeros(Float64, n, lower_bw)
    else
        throw(ArgumentError("output_format must be :dense or :packed_lower_hessenberg"))
    end

    row_scale = _four_point_row_scale(N, sigma)
    Hcur = Mcomb isa Matrix{Float64} ? Mcomb : Array{Float64}(Mcomb)
    Hnext = Matrix{Float64}(undef, I1, N + 1)

    for r in 0:N
        _blocki_scatter_integer_eta_layer!(V, Hcur, row_scale, r, I, N, g)
        r == N && break
        _blocki_next_layer!(Hnext, Hcur, r, N, I1)
        Hcur, Hnext = Hnext, Hcur
    end

    return V
end

"""
    assemble_inner_matrix_precombined_blocki(C, M0, N, sigma; output_format=:packed_lower_hessenberg)

Block-i assembly of the inner matrix `S` without applying the outer
Chebyshev multiplier.  The packed form stores only the lower-Hessenberg band
`ell-k <= I`.
"""
function assemble_inner_matrix_precombined_blocki(C::AbstractMatrix, M0::AbstractMatrix,
                                                  N::Int, sigma::Real;
                                                  output_format::Symbol = :packed_lower_hessenberg,
                                                  lower_bw::Int = size(C, 1) - 1,
                                                  precombine_mode::Symbol = :blas)
    I = size(C, 1) - 1
    J = size(C, 2) - 1
    I1 = I + 1
    size(M0, 1) >= J + 1 || throw(ArgumentError("initial moment table has too few rows"))
    size(M0, 2) >= N + 1 || throw(ArgumentError("initial moment table has too few columns"))

    n = N + 1
    S = if output_format === :dense
        zeros(Float64, n, n)
    elseif output_format === :packed_lower_hessenberg
        packed_lower_hessenberg_zeros(Float64, n, lower_bw)
    else
        throw(ArgumentError("output_format must be :dense or :packed_lower_hessenberg"))
    end

    row_scale = _four_point_row_scale(N, sigma)
    Hcur = _precombine_kernel_moments(C, M0, N, J; mode = precombine_mode)
    Hnext = Matrix{Float64}(undef, I1, N + 1)

    for r in 0:N
        _blocki_scatter_S_layer!(S, Hcur, row_scale, r, I, N)
        r == N && break
        _blocki_next_layer!(Hnext, Hcur, r, N, I1)
        Hcur, Hnext = Hnext, Hcur
    end

    return S
end

function assemble_inner_matrix_precombined_blocki_from_Mcomb(Mcomb::AbstractMatrix,
                                                             N::Int, sigma::Real;
                                                             output_format::Symbol = :packed_lower_hessenberg,
                                                             lower_bw::Int = size(Mcomb, 1) - 1)
    I = size(Mcomb, 1) - 1
    I1 = I + 1
    size(Mcomb, 2) >= N + 1 || throw(ArgumentError("Mcomb has too few columns"))

    n = N + 1
    S = if output_format === :dense
        zeros(Float64, n, n)
    elseif output_format === :packed_lower_hessenberg
        packed_lower_hessenberg_zeros(Float64, n, lower_bw)
    else
        throw(ArgumentError("output_format must be :dense or :packed_lower_hessenberg"))
    end

    row_scale = _four_point_row_scale(N, sigma)
    Hcur = Mcomb isa Matrix{Float64} ? Mcomb : Array{Float64}(Mcomb)
    Hnext = Matrix{Float64}(undef, I1, N + 1)

    for r in 0:N
        _blocki_scatter_S_layer!(S, Hcur, row_scale, r, I, N)
        r == N && break
        _blocki_next_layer!(Hnext, Hcur, r, N, I1)
        Hcur, Hnext = Hnext, Hcur
    end

    return S
end

"""
    assemble_inner_matrix_fast(C, M0, N, sigma; direct_I_threshold=96,
                               precombine_J_threshold=1, fft_block_size=nothing)

Hybrid wrapper.  For short first-index degree `I`, the default path uses
`assemble_inner_matrix_precombined`, which eliminates the explicit `j` loop by
forming `C*M0` before the triangular recurrences.  If `J` is below
`precombine_J_threshold`, the older lower-band-aware direct kernel is used.
For wide kernels in the first index, the old block-FFT implementation remains
available as a fallback.
"""
function assemble_inner_matrix_fast(C::AbstractMatrix, M0::AbstractMatrix, N::Int,
                                    sigma::Real; direct_I_threshold::Int = 96,
                                    precombine_J_threshold::Int = 1,
                                    fft_block_size = nothing,
                                    threaded::Bool = false,
                                    thread_schedule::Symbol = :reverse_static,
                                    thread_chunk_size::Int = 0,
                                    precombine_mode::Symbol = :blas)
    I = size(C, 1) - 1
    J = size(C, 2) - 1
    if I <= direct_I_threshold
        if J >= precombine_J_threshold
            return assemble_inner_matrix_precombined(
                C, M0, N, sigma;
                threaded = threaded,
                thread_schedule = thread_schedule,
                thread_chunk_size = thread_chunk_size,
                precombine_mode = precombine_mode,
            )
        else
            return assemble_inner_matrix_direct_banded(C, M0, N, sigma)
        end
    else
        return assemble_inner_matrix_blockfft(C, M0, N, sigma; fft_block_size = fft_block_size)
    end
end

"""
    oneplus_over2_power_coeffs(p)

Chebyshev coefficients of `((1+z)/2)^p` for integer `p>=0`.
"""
function oneplus_over2_power_coeffs(p::Int)
    p < 0 && throw(ArgumentError("p must be nonnegative"))
    a = [1.0]

    for _ in 1:p
        old = a
        deg = length(old) - 1
        new = zeros(Float64, deg + 2)

        @inbounds for ell in 0:deg
            c = old[ell + 1]
            new[ell + 1] += 0.5 * c

            if ell == 0
                new[2] += 0.5 * c
            else
                new[ell] += 0.25 * c
                new[ell + 2] += 0.25 * c
            end
        end

        a = new
    end

    return a
end

function _oneplus_over2_eta_coeffs(eta::Float64, N::Int;
                                   integer_eta_tol::Float64 = 128eps(Float64),
                                   force_integer_eta::Bool = false)
    if force_integer_eta
        eta >= 0.0 || throw(ArgumentError("force_integer_eta requires eta >= 0"))
        return oneplus_over2_power_coeffs(Int(round(eta))), true
    end

    if _is_nonnegative_integer(eta; tol = integer_eta_tol)
        return oneplus_over2_power_coeffs(Int(round(eta))), true
    end

    nearest = round(eta)
    if eta >= 0.0 && abs(eta - nearest) < 1e-10
        @warn "eta is close to a nonnegative integer but outside integer_eta_tol; set force_integer_eta=true if this is intentional" eta nearest integer_eta_tol
    end

    nodes = cheb_lobatto_nodes(max(N, 1))
    vals = ((1.0 .+ nodes) ./ 2.0) .^ eta
    return cheb_coeffs_lobatto_1d(vals), false
end

@inline function _integer_cheb_multiplier_add!(V::AbstractMatrix, col::Int,
                                               ell::Int, sval::Float64,
                                               g::AbstractVector, N::Int)
    p = length(g) - 1
    @inbounds V[ell + 1, col] += g[1] * sval
    @inbounds for q in 1:p
        gq = g[q + 1]
        if ell == 0
            if q <= N
                V[q + 1, col] += gq * sval
            end
        else
            rowp = ell + q
            if rowp <= N
                V[rowp + 1, col] += 0.5 * gq * sval
            end
            rowm = ell >= q ? ell - q : q - ell
            V[rowm + 1, col] += 0.5 * gq * sval
        end
    end
    return nothing
end

@inline function _integer_cheb_multiplier_add!(V::PackedLowerHessenberg, col::Int,
                                               ell::Int, sval::Float64,
                                               g::AbstractVector, N::Int)
    p = length(g) - 1
    packed_add0!(V, ell, col, g[1] * sval)
    @inbounds for q in 1:p
        gq = g[q + 1]
        if ell == 0
            if q <= N
                packed_add0!(V, q, col, gq * sval)
            end
        else
            rowp = ell + q
            if rowp <= N
                packed_add0!(V, rowp, col, 0.5 * gq * sval)
            end
            rowm = ell >= q ? ell - q : q - ell
            packed_add0!(V, rowm, col, 0.5 * gq * sval)
        end
    end
    return nothing
end

function _integer_multiplier_output(output_format::Symbol, n::Int, lower_bw::Int)
    if output_format === :dense
        return zeros(Float64, n, n)
    elseif output_format === :packed_lower_hessenberg
        return packed_lower_hessenberg_zeros(Float64, n, lower_bw)
    else
        throw(ArgumentError("output_format must be :dense or :packed_lower_hessenberg"))
    end
end

"""
    apply_integer_cheb_multiplier_columns!(V, S, g, N, lower_bw_S)

Apply `sum_q g[q+1]T_q` to each column of the lower-Hessenberg inner matrix
`S` by the Chebyshev product stencil.  No dense multiplier matrix is formed.
"""
function apply_integer_cheb_multiplier_columns!(V, S::PackedLowerHessenberg,
                                                g::AbstractVector, N::Int,
                                                lower_bw_S::Int)
    size(V, 1) == N + 1 && size(V, 2) == N + 1 ||
        throw(DimensionMismatch("V must be $(N + 1) x $(N + 1)"))
    S.n == N + 1 || throw(DimensionMismatch("S has incompatible size"))

    vals = S.values
    ptr = S.colptr
    @inbounds for col in 1:(N + 1)
        first = ptr[col]
        len = min(N + 1, col + lower_bw_S)
        for row in 1:len
            sval = vals[first + row - 1]
            sval == 0.0 && continue
            _integer_cheb_multiplier_add!(V, col, row - 1, sval, g, N)
        end
    end

    return V
end

function apply_integer_cheb_multiplier_columns!(V, S::AbstractMatrix,
                                                g::AbstractVector, N::Int,
                                                lower_bw_S::Int)
    size(S, 1) == N + 1 && size(S, 2) == N + 1 ||
        throw(DimensionMismatch("S must be $(N + 1) x $(N + 1)"))
    size(V, 1) == N + 1 && size(V, 2) == N + 1 ||
        throw(DimensionMismatch("V must be $(N + 1) x $(N + 1)"))

    @inbounds for col in 1:(N + 1)
        len = min(N + 1, col + lower_bw_S)
        for row in 1:len
            sval = S[row, col]
            sval == 0.0 && continue
            _integer_cheb_multiplier_add!(V, col, row - 1, sval, g, N)
        end
    end

    return V
end

function apply_integer_cheb_multiplier_columns(S, g::AbstractVector, N::Int,
                                               lower_bw_S::Int;
                                               output_format::Symbol = :dense,
                                               lower_bw_V::Int = lower_bw_S + length(g) - 1)
    V = _integer_multiplier_output(output_format, N + 1, lower_bw_V)
    return apply_integer_cheb_multiplier_columns!(V, S, g, N, lower_bw_S)
end

"""
    apply_cheb_multiplier_to_matrix(S, g)

Apply multiplication by `sum_m g[m+1]T_m(z)` to each column of `S`,
projecting back to the existing row dimension.
"""
function apply_cheb_multiplier_to_matrix(S::AbstractMatrix, g::AbstractVector)
    N = size(S, 1) - 1
    size(S, 2) == N + 1 || throw(ArgumentError("S must be square"))
    p = length(g) - 1
    V = zeros(Float64, N + 1, N + 1)

    @inbounds for col in 1:(N + 1)
        for n in 0:N
            a = S[n + 1, col]
            a == 0.0 && continue

            for m in 0:p
                gm = g[m + 1]
                gm == 0.0 && continue

                if m == 0
                    V[n + 1, col] += gm * a
                elseif n == 0
                    if m <= N
                        V[m + 1, col] += gm * a
                    end
                else
                    n1 = m + n
                    n2 = abs(m - n)
                    if n1 <= N
                        V[n1 + 1, col] += 0.5 * gm * a
                    end
                    V[n2 + 1, col] += 0.5 * gm * a
                end
            end
        end
    end

    return V
end


function _is_oneplus_over2_coeffs(g::AbstractVector; tol::Float64 = 128eps(Float64))
    return length(g) == 2 &&
           abs(Float64(g[1]) - 0.5) <= tol &&
           abs(Float64(g[2]) - 0.5) <= tol
end

"""
    apply_oneplus_over2_multiplier_to_matrix(S)

Specialized sparse Chebyshev multiplier for `((1+z)/2)`.  If
`S[:,col]` stores coefficients `a_0,...,a_N`, this applies

    (1+z)/2 * sum_n a_n T_n(z)

and truncates back to degrees `0:N`.  The operation is tridiagonal in
Chebyshev coefficient space and costs O(N^2) for an `(N+1) x (N+1)` matrix.
"""
function apply_oneplus_over2_multiplier_to_matrix(S::AbstractMatrix)
    N = size(S, 1) - 1
    size(S, 2) == N + 1 || throw(ArgumentError("S must be square"))
    V = zeros(Float64, N + 1, N + 1)

    @inbounds for col in 1:(N + 1)
        if N == 0
            V[1, col] = 0.5 * S[1, col]
            continue
        end

        # row 0: 0.5*a_0 + 0.25*a_1, because z*T_1 contributes T_0/2.
        V[1, col] = 0.5 * S[1, col] + 0.25 * S[2, col]

        # row 1: z*T_0 = T_1 contributes 0.5*a_0; diagonal gives 0.5*a_1;
        # z*T_2 contributes 0.25*a_2 if it exists.
        V[2, col] = 0.5 * S[1, col] + 0.5 * S[2, col]
        if N >= 2
            V[2, col] += 0.25 * S[3, col]
        end

        # rows 2,...,N-1: 0.25*a_{r-1} + 0.5*a_r + 0.25*a_{r+1}.
        for r in 2:(N - 1)
            V[r + 1, col] = 0.25 * S[r, col] +
                            0.5  * S[r + 1, col] +
                            0.25 * S[r + 2, col]
        end

        # final retained row N; the T_{N+1} coefficient is truncated.
        if N >= 2
            V[N + 1, col] = 0.25 * S[N, col] + 0.5 * S[N + 1, col]
        end
    end

    return V
end


function _cheb_values_lobatto_from_coeffs_padded(c::AbstractVector, M::Int)
    M < 0 && throw(ArgumentError("M must be nonnegative"))
    isempty(c) && throw(ArgumentError("coefficient vector must be nonempty"))

    if M == 0
        return [Float64(c[1])]
    end

    tmp = zeros(Float64, M + 1)
    d = min(length(c) - 1, M)
    tmp[1] = Float64(c[1])

    # FFTW REDFT00 evaluates a Chebyshev series if interior coefficients are halved.
    for n in 1:min(d, M - 1)
        tmp[n + 1] = 0.5 * Float64(c[n + 1])
    end

    if d == M
        tmp[M + 1] = Float64(c[M + 1])
    end

    return FFTW.r2r(tmp, FFTW.REDFT00)
end

"""
    apply_cheb_multiplier_to_matrix_fft(S, g)

Exact padded-DCT multiplication by a Chebyshev series. If `S` has degree `N`
per column and `g` has degree `p`, the product has degree at most `N+p`; we
recover those coefficients on a Lobatto grid of size `N+p` and truncate back to
rows `0:N`. This avoids the O(N^2 p) triple loop when `p` is large.
"""
function _cheb_values_lobatto_from_coeffs_padded_matrix(C::AbstractMatrix, M::Int)
    nrows, ncols = size(C)
    nrows >= 1 || throw(ArgumentError("coefficient matrix must have at least one row"))
    M >= 0 || throw(ArgumentError("M must be nonnegative"))

    A = zeros(Float64, M + 1, ncols)
    d = min(nrows - 1, M)
    A[1, :] .= @view(C[1, :])

    if M > 0
        @inbounds for n in 1:min(d, M - 1)
            @views A[n + 1, :] .= 0.5 .* C[n + 1, :]
        end
        if d == M
            @views A[M + 1, :] .= C[M + 1, :]
        end
    end

    return FFTW.r2r(A, FFTW.REDFT00, 1)
end

function _cheb_coeffs_lobatto_1d_columns(Vals::AbstractMatrix)
    M = size(Vals, 1) - 1
    C = FFTW.r2r(Array{Float64}(Vals), FFTW.REDFT00, 1)
    if M > 0
        C ./= M
        C[1, :] ./= 2.0
        C[end, :] ./= 2.0
    end
    return C
end

function _cheb_values_lobatto_from_coeffs_padded_vector(c::AbstractVector, M::Int)
    C = reshape(Float64.(c), :, 1)
    return vec(_cheb_values_lobatto_from_coeffs_padded_matrix(C, M))
end

"""
    apply_cheb_multiplier_to_matrix_fft(S, g)

Batched padded-DCT multiplication by a Chebyshev series. Compared with the
column-by-column version, this executes the DCTs for all columns in one call
and avoids allocating/planning inside the column loop.
"""
function apply_cheb_multiplier_to_matrix_fft(S::AbstractMatrix, g::AbstractVector)
    N = size(S, 1) - 1
    size(S, 2) == N + 1 || throw(ArgumentError("S must be square"))
    p = length(g) - 1
    p < 0 && throw(ArgumentError("g must be nonempty"))

    M = N + p
    Svals = _cheb_values_lobatto_from_coeffs_padded_matrix(S, M)
    gvals = _cheb_values_lobatto_from_coeffs_padded_vector(g, M)

    @inbounds for col in 1:(N + 1)
        @views Svals[:, col] .*= gvals
    end

    coeffs = _cheb_coeffs_lobatto_1d_columns(Svals)
    return Array(@view(coeffs[1:(N + 1), :]))
end

function apply_cheb_multiplier_to_matrix_fast(S::AbstractMatrix, g::AbstractVector;
                                              fft_degree_threshold::Int = typemax(Int))
    # Stable default: do not use the batched REDFT00 path.  When beta=1-mu,
    # eta=1 and g=[1/2,1/2], so the multiplier is tridiagonal in Chebyshev
    # coefficient space and should be applied sparsely.
    if _is_oneplus_over2_coeffs(g)
        return apply_oneplus_over2_multiplier_to_matrix(S)
    end
    return apply_cheb_multiplier_to_matrix(S, g)
end

"""
    observed_lower_bandwidth(A; tol=1e-12)

Return the largest `row-col` among entries larger than `tol`.
"""
function observed_lower_bandwidth(A::AbstractMatrix; tol::Float64 = 1e-12)
    bw = 0
    n, m = size(A)

    @inbounds for col in 1:m
        for row in 1:n
            if abs(A[row, col]) > tol
                bw = max(bw, row - col)
            end
        end
    end

    return bw
end

"""
    max_below_lower_band(A, bw)

Maximum absolute entry with `row-col > bw`.
"""
function max_below_lower_band(A::AbstractMatrix, bw::Int)
    mx = 0.0
    n, m = size(A)

    @inbounds for col in 1:m
        for row in 1:n
            if row - col > bw
                mx = max(mx, abs(A[row, col]))
            end
        end
    end

    return mx
end

"""
    jfp_basis_value(n, x; sigma, beta)

Evaluate the weighted JFP-Chebyshev basis function `B_n(x)`.
"""
function jfp_basis_value(n::Int, x::Real; sigma::Real = 0.0, beta::Real)
    n < 0 && throw(ArgumentError("n must be nonnegative"))
    z = phi_beta(x, beta)
    return ((1.0 + x) / 2.0)^sigma * cheb_eval_1d([zeros(n); 1.0], z)
end

"""
    eval_jfp_basis_expansion(c, x; sigma, beta)

Evaluate `sum_n c[n+1]B_n(x)`.
"""
function eval_jfp_basis_expansion(c::AbstractVector, x::Real; sigma::Real = 0.0, beta::Real)
    z = phi_beta(x, beta)
    return ((1.0 + x) / 2.0)^sigma * cheb_eval_1d(c, z)
end

"""
    direct_volterra_basis_column_value(K, k, x; mu, sigma, beta, Q=300)

High-order Gauss-Jacobi reference value for `(V_K^mu B_k)(x)`, where
`V_K^mu` uses the public convention
`int_{-1}^x (x-y)^(-mu) K(x,y) B_k(y) dy`.
"""
function direct_volterra_basis_column_value(K, k::Int, x::Real; mu::Real, sigma::Real = 0.0,
                                            beta::Real, Q::Int = 300)
    k < 0 && throw(ArgumentError("k must be nonnegative"))
    x <= -1.0 && return 0.0

    mu_sing = Float64(mu)
    sigf = Float64(sigma)
    betaf = Float64(beta)
    mu_sing < 1.0 || throw(ArgumentError("mu must be less than 1"))
    gamma = (sigf + 1.0) / betaf - 1.0
    a_left = -mu_sing
    q, w = gaussjacobi(Q, a_left, gamma)
    outer = ((1.0 + x) / 2.0)^(1.0 - mu_sing + sigf)
    scale = 2.0^(-mu_sing) / betaf
    acc = 0.0

    @inbounds for n in eachindex(q)
        qv = q[n]
        s = (1.0 + qv) / 2.0
        tv = 2.0 * s^(1.0 / betaf) - 1.0
        y = 0.5 * (x + 1.0) * tv + 0.5 * (x - 1.0)
        zy = phi_beta(y, betaf)
        Tk = cheb_eval_1d([zeros(k); 1.0], zy)
        residual = scale * (1.0 - s^(1.0 / betaf))^(-mu_sing) * s^gamma /
                   ((1.0 - qv)^a_left * (1.0 + qv)^gamma)
        acc += w[n] * residual * K(x, y) * Tk
    end

    return outer * acc
end

"""
    validate_weak_voltmat_columns(K, V; mu, sigma=0, beta, cols, xs, Qref)

Compare selected final matrix columns against direct high-order quadrature in
value space.  The public matrix returned by `weak_voltmat_cheb` is already
postmultiplied back into the same weighted basis, so each selected column is
evaluated with weight exponent `sigma`.  The `mu` keyword follows the public
`(x-y)^(-mu)` convention.
"""
function validate_weak_voltmat_columns(K, V;
                                       mu::Real,
                                       sigma::Real = 0.0,
                                       beta::Real,
                                       cols = (0, 1, 2, 5, 10),
                                       xs = range(-0.95, 0.95; length = 80),
                                       Qref::Int = 800)
    Vdense = V isa PackedLowerHessenberg ? packed_to_dense(V) : Matrix(V)
    ncols = size(Vdense, 2)
    max_abs_all = 0.0
    max_rel_all = 0.0
    col_results = NamedTuple[]

    for k_raw in cols
        k = Int(k_raw)
        if k < 0 || k >= ncols
            continue
        end

        max_abs = 0.0
        max_rel = 0.0
        coeffs = @view Vdense[:, k + 1]

        for x in xs
            approx = eval_jfp_basis_expansion(
                coeffs,
                x;
                sigma = sigma,
                beta = beta,
            )
            ref = direct_volterra_basis_column_value(
                K,
                k,
                x;
                mu = mu,
                sigma = sigma,
                beta = beta,
                Q = Qref,
            )
            err = abs(approx - ref)
            rel = err / max(abs(ref), floatmin(Float64))
            max_abs = max(max_abs, err)
            max_rel = max(max_rel, rel)
        end

        max_abs_all = max(max_abs_all, max_abs)
        max_rel_all = max(max_rel_all, max_rel)
        push!(col_results, (k = k, max_abs = max_abs, max_rel = max_rel))
    end

    return (
        columns = col_results,
        max_abs = max_abs_all,
        max_rel = max_rel_all,
        Qref = Qref,
        npoints = length(xs),
    )
end

"""
    weak_voltmat_cheb(K_or_C; mu, sigma=0, beta, N, ...)

Construct the same-basis coefficient matrix for the weakly singular Volterra
operator

    int_{-1}^x (x-y)^(-mu) K(x,y) u(y) dy

in the weighted JFP-Chebyshev basis
`((x+1)/2)^sigma T_n(2*((x+1)/2)^beta - 1)`.  The usual weakly singular
range is `0 <= mu < 1`; the implementation requires `mu < 1` so that less
singular powers can also be represented.  Choosing `beta = 1 - mu` makes the
final multiplier exponent `eta = (1 - mu)/beta` equal to one.
"""
function weak_voltmat_cheb(
    K_or_C;
    mu,
    sigma = 0.0,
    beta,
    N::Int,
    C_is_coeffs::Bool = false,
    I = nothing,
    J = nothing,
    Qbase = nothing,
    interp_tol::Float64 = _DEFAULT_INTERP_TOL,
    threaded_N_threshold::Int = 400,
    thread_chunk_size::Int = 0,
    thread_schedule::Symbol = :reverse_static,
    fuse_eta1_multiplier::Bool = true,
    integer_eta_max_degree::Int = 8,
    integer_eta_strategy::Symbol = :auto,
    diagnostics::Bool = true,
    output_format::Symbol = :dense,
    H_recurrence_mode::Symbol = :auto,
    block_i_I_threshold::Int = 4,
    block_i_N_threshold::Int = 200,
    init_method::Symbol = :fast,
    de_qfactor::Int = 12,
    de_adaptive::Bool = false,
    de_qfactor_max::Int = 192,
    de_rtol::Float64 = 1e-13,
    accurate_sum::Bool = false,
    init_compare::Bool = false,
    init_compare_Qfactor::Int = 4,
    interp_trim_rtol::Float64 = 1e-13,
    interp_trim_atol::Float64 = 1e-15,
    interp_safety::Int = 0,
    integer_eta_tol::Float64 = 128eps(Float64),
    force_integer_eta::Bool = false,
    precombine_mode::Symbol = :blas,
    return_info::Bool = true,
)
    N < 0 && throw(ArgumentError("N must be nonnegative"))
    mu_sing = Float64(mu)
    sigf = Float64(sigma)
    betaf = Float64(beta)
    mu_sing < 1.0 || throw(ArgumentError("mu must be less than 1"))
    sigf > -1.0 || throw(ArgumentError("sigma must be greater than -1"))
    betaf > 0.0 || throw(ArgumentError("beta must be positive"))
    gamma = (sigf + 1.0) / betaf - 1.0
    gamma > -1.0 || throw(ArgumentError("gamma=(sigma+1)/beta-1 must be greater than -1"))
    integer_eta_max_degree >= 0 || throw(ArgumentError("integer_eta_max_degree must be nonnegative"))
    integer_eta_strategy in (:auto, :fused_scatter, :separate_banded_multiplier) ||
        throw(ArgumentError("integer_eta_strategy must be :auto, :fused_scatter, or :separate_banded_multiplier"))
    block_i_I_threshold >= 0 || throw(ArgumentError("block_i_I_threshold must be nonnegative"))
    block_i_N_threshold >= 0 || throw(ArgumentError("block_i_N_threshold must be nonnegative"))
    init_method in (:fast, :direct) ||
        throw(ArgumentError("init_method must be :fast or :direct"))
    de_qfactor > 0 || throw(ArgumentError("de_qfactor must be positive"))
    de_qfactor_max >= de_qfactor ||
        throw(ArgumentError("de_qfactor_max must be at least de_qfactor"))
    de_rtol > 0.0 || throw(ArgumentError("de_rtol must be positive"))
    init_compare_Qfactor >= 1 ||
        throw(ArgumentError("init_compare_Qfactor must be at least 1"))
    interp_safety >= 0 || throw(ArgumentError("interp_safety must be nonnegative"))
    integer_eta_tol >= 0.0 || throw(ArgumentError("integer_eta_tol must be nonnegative"))
    precombine_mode in (:blas, :kahan) ||
        throw(ArgumentError("precombine_mode must be :blas or :kahan"))
    assembly_direct_I_threshold = 96
    assembly_precombine_J_threshold = 0
    fft_block_size = nothing
    fuse_integer_eta_multiplier = true
    multiplier_fft_degree_threshold = typemax(Int)

    t_total = time_ns()

    t_interp = time_ns()
    if C_is_coeffs
        Craw = Array{Float64}(K_or_C)
        interp_info = (provided_coefficients = true,)
    else
        Craw, interp_info = duffy_cheb_coefficients(
            K_or_C;
            beta = betaf,
            tol = interp_tol,
            trim_rtol = interp_trim_rtol,
            trim_atol = interp_trim_atol,
            safety = interp_safety,
            verbose = false,
        )
        interp_info = merge((provided_coefficients = false,), interp_info)
    end
    interp_seconds = (time_ns() - t_interp) / 1e9

    C = _copy_or_pad_coeffs(Craw, I, J)
    Ieff = size(C, 1) - 1
    Jeff = size(C, 2) - 1
    inner_direct_threshold_eff = max(assembly_direct_I_threshold, Ieff)
    qbase_eff = Qbase === nothing ? _default_Qbase(Jeff, N) : Int(Qbase)
    eta = (1.0 - mu_sing) / betaf
    use_threaded = Threads.nthreads() > 1 && N >= threaded_N_threshold

    t_init = time_ns()
    Minit, init_info = if init_method === :fast
        initial_moments_fast(
            mu_sing,
            sigf,
            betaf,
            Jeff,
            N;
            Qbase = qbase_eff,
            de_qfactor = de_qfactor,
            de_adaptive = de_adaptive,
            de_qfactor_max = de_qfactor_max,
            de_rtol = de_rtol,
            accurate_sum = accurate_sum,
            return_info = true,
        )
    elseif init_method === :direct
        Mdirect_init = initial_moments_direct(mu_sing, sigf, betaf, Jeff, N; Qbase = qbase_eff)
        Mdirect_init, (
            method = :direct,
            Qbase = qbase_eff,
            base_row_info = nothing,
        )
    else
        throw(ArgumentError("init_method must be :fast or :direct"))
    end
    init_seconds = (time_ns() - t_init) / 1e9

    init_compare_info = if init_compare
        direct_Qbase = init_compare_Qfactor * qbase_eff
        Mdirect_cmp = initial_moments_direct(mu_sing, sigf, betaf, Jeff, N; Qbase = direct_Qbase)
        diff_inf = maximum(abs.(Minit .- Mdirect_cmp))
        rel_inf = diff_inf / max(maximum(abs, Mdirect_cmp), floatmin(Float64))
        (
            enabled = true,
            direct_Qbase = direct_Qbase,
            diff_inf = diff_inf,
            rel_inf = rel_inf,
        )
    else
        (
            enabled = false,
            direct_Qbase = 0,
            diff_inf = NaN,
            rel_inf = NaN,
        )
    end

    g, finite_multiplier = _oneplus_over2_eta_coeffs(
        eta,
        N;
        integer_eta_tol = integer_eta_tol,
        force_integer_eta = force_integer_eta,
    )
    multiplier_degree = length(g) - 1
    can_precombine = Jeff >= assembly_precombine_J_threshold
    integer_eta_optimized_available = fuse_integer_eta_multiplier &&
                                      finite_multiplier &&
                                      multiplier_degree <= integer_eta_max_degree &&
                                      can_precombine
    strategy_effective = if integer_eta_optimized_available
        integer_eta_strategy === :auto ?
            (multiplier_degree == 1 ? :fused_scatter : :separate_banded_multiplier) :
            integer_eta_strategy
    else
        :generic_separate_multiplier
    end
    use_fused_eta1 = fuse_eta1_multiplier &&
                     finite_multiplier &&
                     multiplier_degree == 1 &&
                     can_precombine &&
                     strategy_effective === :fused_scatter
    use_fused_integer_eta = integer_eta_optimized_available &&
                            strategy_effective === :fused_scatter
    use_hybrid_integer_eta = integer_eta_optimized_available &&
                             strategy_effective === :separate_banded_multiplier
    use_optimized_integer_eta = use_fused_eta1 || use_fused_integer_eta || use_hybrid_integer_eta
    use_fused_direct = use_fused_eta1 || use_fused_integer_eta
    H_recurrence_mode in (:scalar_i, :legacy_six_term, :block_i, :auto) ||
        throw(ArgumentError("H_recurrence_mode must be :block_i, :auto, :legacy_six_term, or the backward-compatible :scalar_i alias"))
    H_mode_effective = H_recurrence_mode in (:auto, :block_i) ?
                       :block_i : :legacy_six_term
    output_format in (:dense, :packed_lower_hessenberg) ||
        throw(ArgumentError("output_format must be :dense or :packed_lower_hessenberg"))
    if output_format === :packed_lower_hessenberg && !use_optimized_integer_eta
        throw(ArgumentError("packed_lower_hessenberg output requires optimized integer-eta precombined assembly"))
    end
    effective_chunk_size = _effective_thread_chunk_size(N, thread_chunk_size)
    precombine_seconds = 0.0
    Mcomb_pre = nothing
    if H_mode_effective === :block_i
        t_pre = time_ns()
        Mcomb_pre = _precombine_kernel_moments(
            C,
            Minit,
            N,
            Jeff;
            mode = precombine_mode,
        )
        precombine_seconds = (time_ns() - t_pre) / 1e9
    end

    assembly_seconds = 0.0
    multiplier_seconds = 0.0
    V = if use_fused_direct
        t_asm = time_ns()
        lower_bw_fused = Ieff + multiplier_degree
        Vtmp = if H_mode_effective === :block_i
            if multiplier_degree == 1 && use_fused_eta1
                assemble_matrix_precombined_eta1_blocki_from_Mcomb(
                    Mcomb_pre, N, sigf;
                    output_format = output_format,
                    lower_bw = lower_bw_fused,
                )
            else
                assemble_matrix_precombined_integer_eta_blocki_from_Mcomb(
                    Mcomb_pre, N, sigf, g;
                    output_format = output_format,
                    lower_bw = lower_bw_fused,
                )
            end
        else
            if multiplier_degree == 1 && use_fused_eta1
                assemble_matrix_precombined_eta1(
                    C, Minit, N, sigf;
                    threaded = use_threaded,
                    thread_schedule = thread_schedule,
                    thread_chunk_size = effective_chunk_size,
                    output_format = output_format,
                    lower_bw = lower_bw_fused,
                    precombine_mode = precombine_mode,
                )
            else
                assemble_matrix_precombined_integer_eta(
                    C, Minit, N, sigf, g;
                    threaded = use_threaded,
                    thread_schedule = thread_schedule,
                    thread_chunk_size = effective_chunk_size,
                    output_format = output_format,
                    lower_bw = lower_bw_fused,
                    precombine_mode = precombine_mode,
                )
            end
        end
        assembly_seconds = (time_ns() - t_asm) / 1e9
        Vtmp
    elseif use_hybrid_integer_eta
        t_asm = time_ns()
        S = if H_mode_effective === :block_i
            assemble_inner_matrix_precombined_blocki_from_Mcomb(
                Mcomb_pre, N, sigf;
                output_format = :packed_lower_hessenberg,
                lower_bw = Ieff,
            )
        else
            assemble_inner_matrix_precombined(
                C, Minit, N, sigf;
                threaded = use_threaded,
                thread_schedule = thread_schedule,
                thread_chunk_size = effective_chunk_size,
                precombine_mode = precombine_mode,
            )
        end
        assembly_seconds = (time_ns() - t_asm) / 1e9

        t_mul = time_ns()
        Vtmp = apply_integer_cheb_multiplier_columns(
            S, g, N, Ieff;
            output_format = output_format,
            lower_bw_V = Ieff + multiplier_degree,
        )
        multiplier_seconds = (time_ns() - t_mul) / 1e9
        Vtmp
    else
        t_asm = time_ns()
        S = if H_mode_effective === :block_i
            assemble_inner_matrix_precombined_blocki_from_Mcomb(
                Mcomb_pre, N, sigf;
                output_format = :dense,
                lower_bw = Ieff,
            )
        else
            assemble_inner_matrix_fast(
                C, Minit, N, sigf;
                fft_block_size = fft_block_size,
                direct_I_threshold = inner_direct_threshold_eff,
                precombine_J_threshold = assembly_precombine_J_threshold,
                threaded = use_threaded,
                thread_schedule = thread_schedule,
                thread_chunk_size = effective_chunk_size,
                precombine_mode = precombine_mode,
            )
        end
        assembly_seconds = (time_ns() - t_asm) / 1e9
        t_mul = time_ns()
        Vtmp = apply_cheb_multiplier_to_matrix_fast(
            S, g;
            fft_degree_threshold = multiplier_fft_degree_threshold,
        )
        multiplier_seconds = (time_ns() - t_mul) / 1e9
        Vtmp
    end

    lower_bw_theory = finite_multiplier ? Ieff + multiplier_degree : N
    t_diag = time_ns()
    obs_bw = if output_format === :packed_lower_hessenberg
        lower_bw_theory
    else
        diagnostics ? observed_lower_bandwidth(V) : lower_bw_theory
    end
    max_below = if output_format === :packed_lower_hessenberg
        0.0
    else
        diagnostics ? max_below_lower_band(V, lower_bw_theory) : NaN
    end
    diagnostics_seconds = (time_ns() - t_diag) / 1e9
    assembly_mode = if use_fused_direct
        H_mode_effective === :block_i ?
        (multiplier_degree == 1 && use_fused_eta1 ?
         :precombined_ellmajor_serial : :precombined_blocki_serial) :
        (use_threaded ? :precombined_threaded : :precombined_serial)
    elseif use_hybrid_integer_eta
        H_mode_effective === :block_i ? :precombined_blocki_inner_serial :
        (use_threaded ? :precombined_inner_threaded : :precombined_inner_serial)
    elseif H_mode_effective === :block_i
        :precombined_blocki_inner_serial
    elseif Ieff <= inner_direct_threshold_eff
        if Jeff >= assembly_precombine_J_threshold
            H_mode_effective === :block_i ? :precombined_blocki_serial :
            (use_threaded ? :precombined_threaded : :precombined_serial)
        else
            :direct_banded
        end
    else
        :blockfft
    end
    multiplier_mode = use_fused_direct ?
                      (multiplier_degree == 1 ? :deferred_eta1_column : :fused_integer_eta) :
                      (use_hybrid_integer_eta ? :separate_banded_multiplier :
                       (_is_oneplus_over2_coeffs(g) ? :sparse_eta1 : :separate_multiplier))
    dense_entries = (N + 1) * (N + 1)
    stored_entries = output_format === :packed_lower_hessenberg ?
                     length(V.values) : dense_entries
    storage_ratio = stored_entries / dense_entries

    info = (
        I = Ieff,
        J = Jeff,
        N = N,
        mu = mu_sing,
        sigma = sigf,
        beta = betaf,
        eta = eta,
        eta_is_integer = finite_multiplier,
        integer_eta_tol = integer_eta_tol,
        force_integer_eta = force_integer_eta,
        multiplier_degree = multiplier_degree,
        lower_bandwidth_theory = lower_bw_theory,
        lower_bandwidth_observed = obs_bw,
        max_below_lower_band = max_below,
        assembly_direct_I_threshold = assembly_direct_I_threshold,
        assembly_direct_I_threshold_effective = inner_direct_threshold_eff,
        assembly_precombine_J_threshold = assembly_precombine_J_threshold,
        threaded_N_threshold = threaded_N_threshold,
        threaded_assembly = H_mode_effective === :block_i ? false : use_threaded,
        thread_schedule = thread_schedule,
        nthreads_used = (use_threaded && H_mode_effective !== :block_i) ? Threads.nthreads() : 1,
        thread_chunk_size = (use_threaded && H_mode_effective !== :block_i) ? effective_chunk_size : 0,
        diagnostics = diagnostics,
        fuse_eta1_multiplier = fuse_eta1_multiplier,
        fuse_integer_eta_multiplier = fuse_integer_eta_multiplier,
        integer_eta_max_degree = integer_eta_max_degree,
        integer_eta_strategy = integer_eta_strategy,
        integer_eta_strategy_effective = strategy_effective,
        output_format = output_format,
        H_recurrence_mode = H_mode_effective,
        moment_recurrence = H_mode_effective === :block_i ? :four_point : :six_term,
        contraction_mode = :precombined_j,
        block_i_I_threshold = block_i_I_threshold,
        block_i_N_threshold = block_i_N_threshold,
        stored_entries = stored_entries,
        dense_entries = dense_entries,
        storage_ratio = storage_ratio,
        assembly_mode = assembly_mode,
        multiplier_mode = multiplier_mode,
        multiplier_fft_degree_threshold = multiplier_fft_degree_threshold,
        interp_info = interp_info,
        init_method = init_method,
        init_info = init_info,
        init_compare = init_compare_info,
        de_qfactor = de_qfactor,
        de_adaptive = de_adaptive,
        de_qfactor_max = de_qfactor_max,
        de_rtol = de_rtol,
        accurate_sum = accurate_sum,
        precombine_mode = precombine_mode,
        Qbase = qbase_eff,
        C = C,
        timings = (
            interpolation = interp_seconds,
            initial_moments = init_seconds,
            precombine = precombine_seconds,
            assembly = assembly_seconds,
            multiplier = multiplier_seconds,
            diagnostics = diagnostics_seconds,
            total = (time_ns() - t_total) / 1e9,
        ),
    )

    return return_info ? (V, info) : V
end

"""
    validate_packed_against_dense(C; kwargs...)

Build both dense and packed fused-eta1 matrices and compare after
materializing the packed result.  Intended for small benchmark sanity checks.
"""
function validate_packed_against_dense(C::AbstractMatrix; kwargs...)
    Vdense, info_dense = weak_voltmat_cheb(
        C;
        C_is_coeffs = true,
        output_format = :dense,
        diagnostics = true,
        kwargs...,
    )
    Vpacked, info_packed = weak_voltmat_cheb(
        C;
        C_is_coeffs = true,
        output_format = :packed_lower_hessenberg,
        diagnostics = false,
        kwargs...,
    )
    Dpacked = packed_to_dense(Vpacked)
    diff = maximum(abs.(Dpacked .- Vdense))
    return (
        diff_inf = diff,
        dense_info = info_dense,
        packed_info = info_packed,
    )
end

end # module Weak_VoltMat_Cheb
