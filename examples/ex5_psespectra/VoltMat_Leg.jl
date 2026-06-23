# VoltMat_Leg.jl
#
# Purpose:
#   Local auxiliary Legendre Volterra matrix implementation used by the Example
#   5 pseudospectrum workflow.
#
# Problem:
#   Complex-valued Legendre Volterra operator support for
#   ex5_volterra_schur_besselj1.jl.
#
# Method:
#   Provides Legendre interpolation, coefficient, quadrature, and matrix
#   assembly helpers. This file is intentionally kept local to ex5 as
#   reproducibility support and is not a standalone run script.
#
# Inputs:
#   Called by ex5_volterra_schur_besselj1.jl.
#
# Output files:
#   None directly.

module VoltMat_Leg

using LinearAlgebra
using SparseArrays

export voltmat_leg,
       adaptive_legendre_interp_coeffs_Khat,
       legendre_interp_coeffs_Khat,
       legendre_volterra_symband_from_C,
       gausslegendre_rule,
       legendre_values,
       legendre_values!

# =============================================================================
# VoltMat_Leg -- balanced implementation
#
# Main routine:
#
#     B = voltmat_leg(K, N)
#
# It adaptively interpolates
#
#     Khat(x,t) = K(x, (x+1)/2*t + (x-1)/2)
#
# in a tensor-product Legendre basis, then returns a square sparse matrix
# B ∈ R^{(N+1)×(N+1)} containing only the symmetric band
#
#     -bw <= ell-k <= bw.
#
# Default bw = size(C,1), where C is the adaptive Legendre coefficient matrix.
#
# Balanced strategy:
#   * keeps the reliable branch checks near matrix boundaries;
#   * uses compressed CH tables CH_s(i,r)=sum_j C_ij H_{j,s+r}^s;
#   * supports :auto / :dense / :ragged storage for CH;
#   * supports :auto / :dense / :ragged storage for Adams coefficients;
#   * uses SoA triplets and precomputed inverse denominators;
#   * directly constructs SparseMatrixCSC column-by-column.
#
# For small C, :auto tends to choose dense CH/alpha.
# For larger C or large memory estimates, :auto falls back to ragged storage.
# =============================================================================


# -----------------------------------------------------------------------------
# Gauss--Legendre rule by Golub--Welsch.
# -----------------------------------------------------------------------------

function gausslegendre_rule(Q::Integer)
    Q <= 0 && throw(ArgumentError("Q must be positive."))

    if Q == 1
        return [0.0], [2.0]
    end

    beta = [q / sqrt(4q^2 - 1) for q in 1:Q-1]
    J = SymTridiagonal(zeros(Float64, Q), beta)
    F = eigen(J)

    x = F.values
    w = 2.0 .* (F.vectors[1, :] .^ 2)

    return x, w
end


# -----------------------------------------------------------------------------
# Legendre values.
# -----------------------------------------------------------------------------

function legendre_values!(p::AbstractVector{Float64}, x::Real, N::Integer)
    length(p) >= N + 1 || throw(ArgumentError("length(p) must be at least N+1."))
    N < 0 && throw(ArgumentError("N must be nonnegative."))

    xf = Float64(x)

    p[1] = 1.0

    if N >= 1
        p[2] = xf
    end

    @inbounds for n in 1:N-1
        p[n+2] = ((2n + 1) * xf * p[n+1] - n * p[n]) / (n + 1)
    end

    return p
end


function legendre_values(x::Real, N::Integer)
    N < 0 && throw(ArgumentError("N must be nonnegative."))
    p = zeros(Float64, N + 1)
    return legendre_values!(p, x, N)
end


function _legendre_vandermonde(x::AbstractVector{<:Real}, deg::Integer)
    deg < 0 && throw(ArgumentError("deg must be nonnegative."))

    V = zeros(Float64, length(x), deg + 1)
    p = zeros(Float64, deg + 1)

    @inbounds for a in eachindex(x)
        legendre_values!(p, x[a], deg)

        for j in 1:deg+1
            V[a, j] = p[j]
        end
    end

    return V
end


# -----------------------------------------------------------------------------
# Legendre interpolation coefficients of Khat.
# -----------------------------------------------------------------------------

function legendre_interp_coeffs_Khat(
    K::Function,
    nrow::Integer,
    mcol::Integer;
    return_nodes::Bool=false,
)
    nrow <= 0 && throw(ArgumentError("nrow must be positive."))
    mcol <= 0 && throw(ArgumentError("mcol must be positive."))

    xs, _ = gausslegendre_rule(nrow)
    ts, _ = gausslegendre_rule(mcol)

    Vx = _legendre_vandermonde(xs, nrow - 1)
    Vt = _legendre_vandermonde(ts, mcol - 1)

    sample_x = xs[1]
    sample_t = ts[1]
    sample_y = 0.5 * (sample_x + 1.0) * sample_t + 0.5 * (sample_x - 1.0)
    sample = K(sample_x, sample_y)
    Tval = promote_type(Float64, typeof(sample))
    F = zeros(Tval, nrow, mcol)

    # Column-major friendly fill of F[:,c].
    @inbounds for c in 1:mcol
        t = ts[c]

        for a in 1:nrow
            x = xs[a]
            z = 0.5 * (x + 1.0)
            b = 0.5 * (x - 1.0)
            F[a, c] = (a == 1 && c == 1) ? sample : K(x, z * t + b)
        end
    end

    # Interpolation coefficients: F = Vx*C*Vt'.
    C = Vx \ F / transpose(Vt)

    return return_nodes ? (C, xs, ts) : C
end


function _tail_metrics(C::AbstractMatrix, tail_rows::Integer, tail_cols::Integer)
    nrow, mcol = size(C)

    tr = min(max(Int(tail_rows), 1), nrow)
    tc = min(max(Int(tail_cols), 1), mcol)

    scale = max(maximum(abs, C), eps(Float64))
    row_tail = maximum(abs, @view C[nrow-tr+1:nrow, :]) / scale
    col_tail = maximum(abs, @view C[:, mcol-tc+1:mcol]) / scale

    return row_tail, col_tail, scale
end


# -----------------------------------------------------------------------------
# Chebfun-like chopping utilities for adaptive Khat interpolation.
#
# This is not a literal copy of Chebfun's standardChop, but follows the same
# principle:
#   1. compress the 2D coefficient matrix into 1D row/column coefficient profiles;
#   2. form a nonincreasing envelope;
#   3. look for either a tolerance crossing or a sufficiently low plateau;
#   4. keep a few guard coefficients beyond the proposed chop before accepting.
# -----------------------------------------------------------------------------

function _coeff_profiles(C::AbstractMatrix)
    nrow, mcol = size(C)

    row_profile = zeros(Float64, nrow)
    col_profile = zeros(Float64, mcol)

    @inbounds for i in 1:nrow
        mx = 0.0
        for j in 1:mcol
            mx = max(mx, abs(C[i, j]))
        end
        row_profile[i] = mx
    end

    @inbounds for j in 1:mcol
        mx = 0.0
        for i in 1:nrow
            mx = max(mx, abs(C[i, j]))
        end
        col_profile[j] = mx
    end

    return row_profile, col_profile
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


function _standard_chop_like(
    coeff_profile::AbstractVector{<:Real};
    tol::Float64=1.0e-12,
    plateau_tol::Union{Nothing,Float64}=nothing,
    plateau_window::Integer=8,
    plateau_drop::Float64=0.1,
)
    tol <= 0 && throw(ArgumentError("tol must be positive."))
    plateau_window <= 0 && throw(ArgumentError("plateau_window must be positive."))
    0 < plateau_drop <= 1 || throw(ArgumentError("plateau_drop must lie in (0,1]."))

    env, scale = _monotone_envelope(coeff_profile)
    n = length(env)

    if n == 0 || scale <= eps(Float64)
        return (cutoff = 1, happy = true, tail = 0.0, plateau = true, scale = scale)
    end

    # Chebfun's chopping is deliberately less strict than "last coefficient < tol".
    # The 2/3 exponent is a standard practical compromise: it detects a low
    # plateau before insisting that the absolute envelope is already below tol.
    ptol = plateau_tol === nothing ? tol^(2/3) : Float64(plateau_tol)

    # First, accept a direct tolerance crossing.
    direct = findfirst(x -> x <= tol, env)
    if direct !== nothing
        return (
            cutoff = max(1, direct),
            happy = true,
            tail = env[end],
            plateau = false,
            scale = scale,
        )
    end

    # Then look for a low plateau.  A plateau is detected when the envelope is
    # already below ptol and does not decrease by more than roughly one decade
    # over a short window.  This is intentionally conservative near boundaries.
    w = min(Int(plateau_window), max(n - 1, 1))

    if n >= w + 2
        @inbounds for j in 2:(n - w)
            if env[j] <= ptol
                ratio = env[j+w] / max(env[j], eps(Float64))

                if ratio >= plateau_drop
                    return (
                        cutoff = j,
                        happy = true,
                        tail = env[end],
                        plateau = true,
                        scale = scale,
                    )
                end
            end
        end
    end

    return (
        cutoff = n,
        happy = false,
        tail = env[end],
        plateau = false,
        scale = scale,
    )
end


function _chop_coefficient_matrix(
    C::AbstractMatrix;
    tol::Float64,
    plateau_tol::Union{Nothing,Float64},
    plateau_window::Integer,
    plateau_drop::Float64,
    chop_guard::Integer,
)
    row_profile, col_profile = _coeff_profiles(C)

    row_chop = _standard_chop_like(
        row_profile;
        tol=tol,
        plateau_tol=plateau_tol,
        plateau_window=plateau_window,
        plateau_drop=plateau_drop,
    )

    col_chop = _standard_chop_like(
        col_profile;
        tol=tol,
        plateau_tol=plateau_tol,
        plateau_window=plateau_window,
        plateau_drop=plateau_drop,
    )

    nrow, mcol = size(C)
    guard = max(Int(chop_guard), 0)

    row_safe = row_chop.cutoff <= max(nrow - guard, 1)
    col_safe = col_chop.cutoff <= max(mcol - guard, 1)

    happy = row_chop.happy && col_chop.happy && row_safe && col_safe

    ncut = clamp(row_chop.cutoff, 1, nrow)
    mcut = clamp(col_chop.cutoff, 1, mcol)

    return (
        ncut = ncut,
        mcut = mcut,
        happy = happy,
        row_happy = row_chop.happy,
        col_happy = col_chop.happy,
        row_safe = row_safe,
        col_safe = col_safe,
        row_tail = row_chop.tail,
        col_tail = col_chop.tail,
        row_plateau = row_chop.plateau,
        col_plateau = col_chop.plateau,
        row_scale = row_chop.scale,
        col_scale = col_chop.scale,
    )
end


function adaptive_legendre_interp_coeffs_Khat(
    K::Function;
    tol::Float64=1.0e-12,
    nmin::Integer=8,
    mmin::Integer=8,
    nmax::Integer=128,
    mmax::Integer=128,
    growth::Real=1.5,
    tail::Integer=3,
    verbose::Bool=false,
    maxiters::Integer=50,
    warn_exceed_max::Bool=true,
    adaptive_mode::Symbol=:chebfun,
    plateau_tol::Union{Nothing,Float64}=nothing,
    plateau_window::Integer=8,
    plateau_drop::Float64=0.1,
    chop_guard::Integer=4,
)
    tol <= 0 && throw(ArgumentError("tol must be positive."))
    nmin <= 0 && throw(ArgumentError("nmin must be positive."))
    mmin <= 0 && throw(ArgumentError("mmin must be positive."))
    nmax < nmin && throw(ArgumentError("nmax must be >= nmin."))
    mmax < mmin && throw(ArgumentError("mmax must be >= mmin."))
    growth <= 1 && throw(ArgumentError("growth must be > 1."))
    maxiters <= 0 && throw(ArgumentError("maxiters must be positive."))
    adaptive_mode in (:chebfun, :tail) || throw(ArgumentError("adaptive_mode must be :chebfun or :tail."))

    # nmax and mmax are warning thresholds, not hard caps.
    nrow = Int(nmin)
    mcol = Int(mmin)

    last_C = zeros(Float64, 0, 0)
    last_row_tail = Inf
    last_col_tail = Inf
    last_scale = 0.0
    converged = false
    niters = 0

    warned_n = false
    warned_m = false

    # Information specific to Chebfun-like chopping.
    last_ncut = nrow
    last_mcut = mcol
    last_row_plateau = false
    last_col_plateau = false
    last_row_safe = false
    last_col_safe = false

    while true
        niters += 1

        C = legendre_interp_coeffs_Khat(K, nrow, mcol)

        if adaptive_mode == :tail
            row_tail, col_tail, scale = _tail_metrics(C, tail, tail)
            row_ok = row_tail <= tol
            col_ok = col_tail <= tol

            ncut = nrow
            mcut = mcol
            row_safe = true
            col_safe = true
            row_plateau = false
            col_plateau = false
            happy = row_ok && col_ok
        else
            chop = _chop_coefficient_matrix(
                C;
                tol=tol,
                plateau_tol=plateau_tol,
                plateau_window=plateau_window,
                plateau_drop=plateau_drop,
                chop_guard=chop_guard,
            )

            row_tail = chop.row_tail
            col_tail = chop.col_tail
            scale = max(chop.row_scale, chop.col_scale)

            row_ok = chop.row_happy && chop.row_safe
            col_ok = chop.col_happy && chop.col_safe

            ncut = chop.ncut
            mcut = chop.mcut
            row_safe = chop.row_safe
            col_safe = chop.col_safe
            row_plateau = chop.row_plateau
            col_plateau = chop.col_plateau
            happy = chop.happy
        end

        if verbose
            @info "adaptive Khat interpolation" adaptive_mode nrow mcol ncut mcut row_tail col_tail scale row_ok col_ok row_safe col_safe row_plateau col_plateau
        end

        last_C = C
        last_row_tail = row_tail
        last_col_tail = col_tail
        last_scale = scale
        last_ncut = ncut
        last_mcut = mcut
        last_row_plateau = row_plateau
        last_col_plateau = col_plateau
        last_row_safe = row_safe
        last_col_safe = col_safe

        if happy
            converged = true

            if adaptive_mode == :chebfun
                last_C = Matrix(@view C[1:ncut, 1:mcut])
            end

            break
        end

        if niters >= maxiters
            @warn "adaptive Legendre interpolation reached maxiters before meeting tolerance" niters nrow mcol ncut mcut row_tail col_tail tol adaptive_mode

            if adaptive_mode == :chebfun
                # If not fully happy, still return the current best chopped block
                # when a meaningful cutoff exists; otherwise return the full C.
                nbest = clamp(ncut, 1, nrow)
                mbest = clamp(mcut, 1, mcol)
                last_C = Matrix(@view C[1:nbest, 1:mbest])
            end

            break
        end

        new_nrow = row_ok ? nrow : max(nrow + 1, ceil(Int, growth * nrow))
        new_mcol = col_ok ? mcol : max(mcol + 1, ceil(Int, growth * mcol))

        if warn_exceed_max
            if !warned_n && new_nrow > nmax
                @warn "adaptive Legendre interpolation exceeded $nmax; continuing because nmax is only a warning threshold" nmax new_nrow row_tail tol adaptive_mode
                warned_n = true
            end

            if !warned_m && new_mcol > mmax
                @warn "adaptive Legendre interpolation exceeded $mmax; continuing because mmax is only a warning threshold" mmax new_mcol col_tail tol adaptive_mode
                warned_m = true
            end
        end

        nrow, mcol = new_nrow, new_mcol
    end

    info = (
        nrow = size(last_C, 1),
        mcol = size(last_C, 2),
        row_tail = last_row_tail,
        col_tail = last_col_tail,
        scale = last_scale,
        tol = tol,
        converged = converged,
        iterations = niters,
        adaptive_mode = adaptive_mode,
        ncut = last_ncut,
        mcut = last_mcut,
        row_plateau = last_row_plateau,
        col_plateau = last_col_plateau,
        row_safe = last_row_safe,
        col_safe = last_col_safe,
        nmax_warning_threshold = Int(nmax),
        mmax_warning_threshold = Int(mmax),
        exceeded_nmax = size(last_C, 1) > nmax,
        exceeded_mmax = size(last_C, 2) > mmax,
        maxiters = Int(maxiters),
    )

    return last_C, info
end


# -----------------------------------------------------------------------------
# H and Jacobi coefficients.
# -----------------------------------------------------------------------------

@inline function _H0_entry(s::Int, r::Int)
    r < 0 && return 0.0

    if r == 0
        return 2.0 / (s + 1)
    elseif iseven(r)
        p = r >>> 1
        return 2.0 * (2s + 2p + 1) / ((s + 2p) * (s + 2p + 1))
    else
        p = (r - 1) >>> 1
        return -4.0 * (s + p + 1) / ((s + 2p + 1) * (s + 2p + 2))
    end
end


@inline function _jacobi_x_coeffs(s::Int, r::Int)
    if r == 0
        return 1.0 / (s + 1), 1.0, 0.0
    end

    A = 2.0 * (r + 1) * (r + 2s + 1) /
        ((2r + 2s + 1) * (2r + 2s + 2))

    B = s == 0 ? 0.0 :
        s * (s + 1) / ((r + s) * (r + s + 1))

    C = 2.0 * (r - 1) * (r + 2s + 1) /
        ((2r + 2s) * (2r + 2s + 1))

    return A, B, C
end


# Fill H[j+1,r+1] = H_{j,s+r}^s for fixed s.
# Working vectors are reused by the caller.
function _fill_H_for_s!(
    H::Matrix{Float64},
    hm1::Vector{Float64},
    hcur::Vector{Float64},
    hnext::Vector{Float64},
    s::Int,
    jmax::Int,
    Rneed::Int,
)
    R0 = Rneed + jmax

    @inbounds for r in 0:R0
        hm1[r+1] = _H0_entry(s, r)
    end

    @inbounds for r in 0:Rneed
        H[1, r+1] = hm1[r+1]
    end

    if jmax == 0
        return nothing
    end

    @inbounds for r in 0:R0-1
        A, B, C = _jacobi_x_coeffs(s, r)

        val = A * hm1[r+2] + B * hm1[r+1]

        if r >= 1
            val += C * hm1[r]
        end

        hcur[r+1] = val
    end

    @inbounds for r in 0:Rneed
        H[2, r+1] = hcur[r+1]
    end

    curmax = R0 - 1

    @inbounds for j in 1:jmax-1
        nextmax = curmax - 1

        for r in 0:nextmax
            A, B, C = _jacobi_x_coeffs(s, r)

            rhs = A * hcur[r+2] + B * hcur[r+1]

            if r >= 1
                rhs += C * hcur[r]
            end

            hnext[r+1] =
                ((2j + 1) / (j + 1)) * rhs -
                (j / (j + 1)) * hm1[r+1]
        end

        for r in 0:Rneed
            H[j+2, r+1] = hnext[r+1]
        end

        hm1, hcur, hnext = hcur, hnext, hm1
        curmax = nextmax
    end

    return nothing
end


# -----------------------------------------------------------------------------
# Adams coefficients.
# -----------------------------------------------------------------------------

function _adams_coeffs(ell::Int, i::Int)
    amax = min(ell, i)
    alpha = zeros(Float64, amax + 1)

    if i == 0
        alpha[1] = 1.0
        return alpha
    end

    alpha0 = 1.0

    @inbounds for k in 1:i
        alpha0 *= ((2k - 1) * (ell + k)) /
                  (k * (2ell + 2k - 1))
    end

    alpha[1] = alpha0

    @inbounds for a in 1:amax
        R1_num =
            (ell - a + 1) *
            (2a - 1) *
            (i - a + 1) *
            (2 * (ell + i - a) + 1)

        R1_den =
            (2ell - 2a + 1) *
            a *
            (2i - 2a + 1) *
            (ell + i - a + 1)

        R2_num =
            (2ell + 2i - 4a + 1) *
            (2ell + 2i - 2a + 3)

        R2_den =
            (2ell + 2i - 4a + 5) *
            (2ell + 2i - 2a + 1)

        alpha[a+1] = alpha[a] * (R1_num / R1_den) * (R2_num / R2_den)
    end

    return alpha
end


abstract type AbstractAlphaCache end

struct DenseAlphaCache <: AbstractAlphaCache
    data::Array{Float64, 3}
end

struct RaggedAlphaCache <: AbstractAlphaCache
    data::Vector{Vector{Vector{Float64}}}
end


function _adams_cache_dense(maxell::Int, ndeg::Int)
    cache = zeros(Float64, ndeg + 1, ndeg + 1, maxell + 1)

    @inbounds for ell in 0:maxell
        for i in 0:ndeg
            alpha_vec = _adams_coeffs(ell, i)
            for a in 0:min(ell, i)
                cache[a+1, i+1, ell+1] = alpha_vec[a+1]
            end
        end
    end

    return DenseAlphaCache(cache)
end


function _adams_cache_ragged(maxell::Int, ndeg::Int)
    cache = Vector{Vector{Vector{Float64}}}(undef, maxell + 1)

    @inbounds for ell in 0:maxell
        cell = Vector{Vector{Float64}}(undef, ndeg + 1)

        for i in 0:ndeg
            cell[i+1] = _adams_coeffs(ell, i)
        end

        cache[ell+1] = cell
    end

    return RaggedAlphaCache(cache)
end


@inline function _alpha_get(cache::DenseAlphaCache, a::Int, i::Int, ell::Int)
    return cache.data[a+1, i+1, ell+1]
end


@inline function _alpha_get(cache::RaggedAlphaCache, a::Int, i::Int, ell::Int)
    vec = cache.data[ell+1][i+1]
    return a + 1 <= length(vec) ? vec[a+1] : 0.0
end


# -----------------------------------------------------------------------------
# CH caches.
# -----------------------------------------------------------------------------

abstract type AbstractCHCache end

struct DenseCHCache <: AbstractCHCache
    data::Array{Float64, 3}   # data[i+1,r+1,s+1]
    N::Int
    Rneed::Int
end

struct RaggedCHCache <: AbstractCHCache
    tables::Vector{Matrix{Float64}}  # tables[s+1][i+1,r+1]
    N::Int
end


@inline function _ch_get(cache::DenseCHCache, i::Int, r::Int, s::Int)
    if 0 <= s <= cache.N && 0 <= r <= cache.Rneed
        return cache.data[i+1, r+1, s+1]
    else
        return 0.0
    end
end


@inline function _ch_get(cache::RaggedCHCache, i::Int, r::Int, s::Int)
    if 0 <= s <= cache.N
        CH = cache.tables[s+1]

        if 0 <= r < size(CH, 2)
            return CH[i+1, r+1]
        end
    end

    return 0.0
end


function _zero_impossible_H_entries!(H::Matrix{Float64}, s::Int, mdeg::Int, Rs::Int)
    @inbounds for r in 0:Rs
        k = s + r

        if k < mdeg
            for j in k+1:mdeg
                H[j+1, r+1] = 0.0
            end
        end
    end

    return nothing
end


function _precompute_CH_dense(C::AbstractMatrix{<:Real}, N::Int, mdeg::Int, Rneed::Int)
    Cmat = Matrix{Float64}(C)
    nrow = size(Cmat, 1)

    data = zeros(Float64, nrow, Rneed + 1, N + 1)

    R0max = Rneed + mdeg
    hm1 = zeros(Float64, R0max + 2)
    hcur = zeros(Float64, R0max + 2)
    hnext = zeros(Float64, R0max + 2)
    Hbuf = zeros(Float64, mdeg + 1, Rneed + 1)

    @inbounds for s in 0:N
        Rs = min(Rneed, N - s)

        _fill_H_for_s!(Hbuf, hm1, hcur, hnext, s, mdeg, Rs)
        _zero_impossible_H_entries!(Hbuf, s, mdeg, Rs)

        Hview = @view Hbuf[:, 1:Rs+1]
        CHview = @view data[:, 1:Rs+1, s+1]

        mul!(CHview, Cmat, Hview)
    end

    return DenseCHCache(data, N, Rneed)
end


function _precompute_CH_ragged(C::AbstractMatrix{<:Real}, N::Int, mdeg::Int, Rneed::Int)
    Cmat = Matrix{Float64}(C)
    nrow = size(Cmat, 1)

    tables = Vector{Matrix{Float64}}(undef, N + 1)

    R0max = Rneed + mdeg
    hm1 = zeros(Float64, R0max + 2)
    hcur = zeros(Float64, R0max + 2)
    hnext = zeros(Float64, R0max + 2)
    Hbuf = zeros(Float64, mdeg + 1, Rneed + 1)

    @inbounds for s in 0:N
        Rs = min(Rneed, N - s)

        _fill_H_for_s!(Hbuf, hm1, hcur, hnext, s, mdeg, Rs)
        _zero_impossible_H_entries!(Hbuf, s, mdeg, Rs)

        Hview = @view Hbuf[:, 1:Rs+1]
        CH = Matrix{Float64}(undef, nrow, Rs + 1)

        mul!(CH, Cmat, Hview)

        tables[s+1] = CH
    end

    return RaggedCHCache(tables, N)
end


# -----------------------------------------------------------------------------
# Triplets in SoA layout.
# -----------------------------------------------------------------------------

struct TripletsSoA
    i::Vector{Int}
    a::Vector{Int}
    r::Vector{Int}
end


function _build_triplets_soa(dphi_min::Int, dphi_max::Int, ndeg::Int, Rneed::Int)
    ndphi = dphi_max - dphi_min + 1
    triplets = Vector{TripletsSoA}(undef, ndphi)

    for dphi in dphi_min:dphi_max
        ivec = Int[]
        avec = Int[]
        rvec = Int[]

        # Small vectors; sizehint helps when p is not tiny.
        sizehint!(ivec, (ndeg + 1) * (ndeg + 2) ÷ 2)
        sizehint!(avec, (ndeg + 1) * (ndeg + 2) ÷ 2)
        sizehint!(rvec, (ndeg + 1) * (ndeg + 2) ÷ 2)

        for i in 0:ndeg
            for a in 0:i
                r = 2a - i - dphi

                if 0 <= r <= Rneed
                    push!(ivec, i)
                    push!(avec, a)
                    push!(rvec, r)
                end
            end
        end

        triplets[dphi - dphi_min + 1] = TripletsSoA(ivec, avec, rvec)
    end

    return triplets
end


# -----------------------------------------------------------------------------
# Backend choice.
# -----------------------------------------------------------------------------

function _estimate_ch_bytes(nrow::Int, Rneed::Int, N::Int)
    return 8.0 * nrow * (Rneed + 1) * (N + 1)
end


function _estimate_alpha_bytes(ndeg::Int, maxell::Int)
    return 8.0 * (ndeg + 1) * (ndeg + 1) * (maxell + 1)
end


function _resolve_ch_backend(ch_backend::Symbol, nrow::Int, Rneed::Int, N::Int,
                             dense_ch_max_bytes::Real, dense_ch_pmax::Int)
    if ch_backend == :dense || ch_backend == :ragged
        return ch_backend
    elseif ch_backend == :auto
        bytes = _estimate_ch_bytes(nrow, Rneed, N)
        return (nrow <= dense_ch_pmax && bytes <= dense_ch_max_bytes) ? :dense : :ragged
    else
        throw(ArgumentError("ch_backend must be :auto, :dense, or :ragged."))
    end
end


function _resolve_alpha_backend(alpha_backend::Symbol, ndeg::Int, maxell::Int,
                                dense_alpha_max_bytes::Real, dense_alpha_pmax::Int)
    if alpha_backend == :dense || alpha_backend == :ragged
        return alpha_backend
    elseif alpha_backend == :auto
        p = ndeg + 1
        bytes = _estimate_alpha_bytes(ndeg, maxell)
        return (p <= dense_alpha_pmax && bytes <= dense_alpha_max_bytes) ? :dense : :ragged
    else
        throw(ArgumentError("alpha_backend must be :auto, :dense, or :ragged."))
    end
end


# -----------------------------------------------------------------------------
# Phi entry.
# -----------------------------------------------------------------------------

@inline function _phi_entry_balanced(
    CHcache::AbstractCHCache,
    Alphacache::AbstractAlphaCache,
    trip::TripletsSoA,
    inv_den::Vector{Float64},
    k::Int,
    dphi::Int,
    maxell::Int,
)
    ell = k + dphi

    if ell < 0 || ell > maxell
        return 0.0
    end

    ivec = trip.i
    avec = trip.a
    rvec = trip.r
    ntrip = length(ivec)

    twoell1 = 2ell + 1
    val = 0.0

    @inbounds for idx in 1:ntrip
        i = ivec[idx]
        a = avec[idx]
        r = rvec[idx]

        # This branch is essential near the upper-left corner and avoids
        # using invalid Adams entries for small ell.
        if a > ell
            continue
        end

        s = k - r

        if s < 0
            continue
        end

        ch = _ch_get(CHcache, i, r, s)

        if ch == 0.0
            continue
        end

        alpha = _alpha_get(Alphacache, a, i, ell)

        if alpha == 0.0
            continue
        end

        denom = 2k - r + 1
        val += twoell1 * inv_den[denom] * alpha * ch
    end

    return val
end


@inline function _get_phival(phivals::Vector{Float64}, d::Int, dphi_min::Int, dphi_max::Int)
    return (d >= dphi_min && d <= dphi_max) ? phivals[d - dphi_min + 1] : 0.0
end


# Function barrier to let Julia specialize on concrete CH/alpha cache types.
function _assemble_symband(
    CHcache::CHT,
    Alphacache::AT,
    triplets_all::Vector{TripletsSoA},
    inv_den::Vector{Float64},
    Nint::Int,
    bw::Int,
    dphi_min::Int,
    dphi_max::Int,
    maxell_phi::Int,
    dropzeros_tol::Float64,
) where {CHT<:AbstractCHCache, AT<:AbstractAlphaCache}

    ndphi = dphi_max - dphi_min + 1

    rowval = Int[]
    nzval = Float64[]
    colptr = Vector{Int}(undef, Nint + 2)

    colptr[1] = 1

    estimated_nnz = (Nint + 1) * (2bw + 1)
    sizehint!(rowval, estimated_nnz)
    sizehint!(nzval, estimated_nnz)

    phivals = zeros(Float64, ndphi)

    @inbounds for k in 0:Nint
        fill!(phivals, 0.0)

        for dphi in dphi_min:dphi_max
            ellphi = k + dphi

            if 0 <= ellphi <= maxell_phi
                idx = dphi - dphi_min + 1
                phivals[idx] = _phi_entry_balanced(
                    CHcache,
                    Alphacache,
                    triplets_all[idx],
                    inv_den,
                    k,
                    dphi,
                    maxell_phi,
                )
            end
        end

        d_start = max(-bw, -k)
        d_stop = min(bw, Nint - k)

        for d in d_start:d_stop
            ell = k + d

            val = 0.5 * _get_phival(phivals, d, dphi_min, dphi_max)

            if ell >= 1
                val += 0.5 * ell / (2ell - 1) *
                       _get_phival(phivals, d - 1, dphi_min, dphi_max)
            end

            val += 0.5 * (ell + 1) / (2ell + 3) *
                   _get_phival(phivals, d + 1, dphi_min, dphi_max)

            if abs(val) > dropzeros_tol
                push!(rowval, ell + 1)
                push!(nzval, val)
            end
        end

        colptr[k+2] = length(rowval) + 1
    end

    return SparseMatrixCSC(Nint + 1, Nint + 1, colptr, rowval, nzval)
end


# -----------------------------------------------------------------------------
# Matrix constructor from C.
# -----------------------------------------------------------------------------

function legendre_volterra_symband_from_C(
    C::AbstractMatrix{<:Real},
    N::Integer;
    bandwidth::Union{Nothing,Integer}=nothing,
    dropzeros_tol::Float64=0.0,
    ch_backend::Symbol=:auto,
    alpha_backend::Symbol=:auto,
    dense_ch_max_bytes::Real=512.0 * 1024.0^2,
    dense_alpha_max_bytes::Real=512.0 * 1024.0^2,
    dense_ch_pmax::Integer=24,
    dense_alpha_pmax::Integer=48,
)
    N < 0 && throw(ArgumentError("N must be nonnegative."))

    Nint = Int(N)
    nrow = size(C, 1)
    mcol = size(C, 2)
    ndeg = nrow - 1
    mdeg = mcol - 1

    bw = bandwidth === nothing ? nrow : Int(bandwidth)
    bw < 0 && throw(ArgumentError("bandwidth must be nonnegative."))
    bw = min(bw, Nint)

    dphi_min = -bw - 1
    dphi_max = min(bw + 1, ndeg)

    # Max r = 2a-i-dphi, attained at a=i=ndeg and dphi=dphi_min.
    Rneed = ndeg - dphi_min

    # Square output has rows ell=0,...,N. B_ell uses Phi_{ell+1}.
    maxell_phi = min(Nint + ndeg, Nint + 1)

    chosen_ch = _resolve_ch_backend(
        ch_backend,
        nrow,
        Rneed,
        Nint,
        dense_ch_max_bytes,
        Int(dense_ch_pmax),
    )

    chosen_alpha = _resolve_alpha_backend(
        alpha_backend,
        ndeg,
        maxell_phi,
        dense_alpha_max_bytes,
        Int(dense_alpha_pmax),
    )

    CHcache =
        chosen_ch == :dense ?
        _precompute_CH_dense(C, Nint, mdeg, Rneed) :
        _precompute_CH_ragged(C, Nint, mdeg, Rneed)

    Alphacache =
        chosen_alpha == :dense ?
        _adams_cache_dense(maxell_phi, ndeg) :
        _adams_cache_ragged(maxell_phi, ndeg)

    triplets_all = _build_triplets_soa(dphi_min, dphi_max, ndeg, Rneed)

    inv_den = zeros(Float64, 2Nint + 1)

    @inbounds for d in 1:2Nint+1
        inv_den[d] = 1.0 / d
    end

    return _assemble_symband(
        CHcache,
        Alphacache,
        triplets_all,
        inv_den,
        Nint,
        bw,
        dphi_min,
        dphi_max,
        maxell_phi,
        dropzeros_tol,
    )
end


function legendre_volterra_symband_from_C(
    C::AbstractMatrix{<:Complex},
    N::Integer;
    bandwidth::Union{Nothing,Integer}=nothing,
    dropzeros_tol::Float64=0.0,
    ch_backend::Symbol=:auto,
    alpha_backend::Symbol=:auto,
    dense_ch_max_bytes::Real=512.0 * 1024.0^2,
    dense_alpha_max_bytes::Real=512.0 * 1024.0^2,
    dense_ch_pmax::Integer=24,
    dense_alpha_pmax::Integer=48,
)
    Br = legendre_volterra_symband_from_C(
        real.(C),
        N;
        bandwidth=bandwidth,
        dropzeros_tol=dropzeros_tol,
        ch_backend=ch_backend,
        alpha_backend=alpha_backend,
        dense_ch_max_bytes=dense_ch_max_bytes,
        dense_alpha_max_bytes=dense_alpha_max_bytes,
        dense_ch_pmax=dense_ch_pmax,
        dense_alpha_pmax=dense_alpha_pmax,
    )

    Bi = legendre_volterra_symband_from_C(
        imag.(C),
        N;
        bandwidth=bandwidth,
        dropzeros_tol=dropzeros_tol,
        ch_backend=ch_backend,
        alpha_backend=alpha_backend,
        dense_ch_max_bytes=dense_ch_max_bytes,
        dense_alpha_max_bytes=dense_alpha_max_bytes,
        dense_ch_pmax=dense_ch_pmax,
        dense_alpha_pmax=dense_alpha_pmax,
    )

    return Br + im * Bi
end


function _assembly_info(
    C::AbstractMatrix,
    N::Integer,
    bandwidth,
    ch_backend::Symbol,
    alpha_backend::Symbol,
    dense_ch_max_bytes::Real,
    dense_alpha_max_bytes::Real,
    dense_ch_pmax::Integer,
    dense_alpha_pmax::Integer,
)
    Nint = Int(N)
    nrow = size(C, 1)
    ndeg = nrow - 1

    bw = bandwidth === nothing ? nrow : Int(bandwidth)
    bw = min(bw, Nint)

    dphi_min = -bw - 1
    Rneed = ndeg - dphi_min
    maxell_phi = min(Nint + ndeg, Nint + 1)

    chosen_ch = _resolve_ch_backend(
        ch_backend,
        nrow,
        Rneed,
        Nint,
        dense_ch_max_bytes,
        Int(dense_ch_pmax),
    )

    chosen_alpha = _resolve_alpha_backend(
        alpha_backend,
        ndeg,
        maxell_phi,
        dense_alpha_max_bytes,
        Int(dense_alpha_pmax),
    )

    return (
        bandwidth = bw,
        ch_backend = chosen_ch,
        alpha_backend = chosen_alpha,
        Rneed = Rneed,
        estimated_ch_bytes = _estimate_ch_bytes(nrow, Rneed, Nint),
        estimated_dense_alpha_bytes = _estimate_alpha_bytes(ndeg, maxell_phi),
    )
end


# -----------------------------------------------------------------------------
# Main function.
# -----------------------------------------------------------------------------

function voltmat_leg(
    K::Function,
    N::Integer;
    tol::Float64=1.0e-13,
    nmin::Integer=8,
    mmin::Integer=8,
    nmax::Integer=128,
    mmax::Integer=128,
    growth::Real=1.5,
    tail::Integer=3,
    maxiters::Integer=20,
    warn_exceed_max::Bool=true,
    adaptive_mode::Symbol=:chebfun,
    plateau_tol::Union{Nothing,Float64}=nothing,
    plateau_window::Integer=8,
    plateau_drop::Float64=0.1,
    chop_guard::Integer=4,
    bandwidth::Union{Nothing,Integer}=nothing,
    return_C::Bool=false,
    return_info::Bool=false,
    dropzeros_tol::Float64=0.0,
    verbose::Bool=false,
    ch_backend::Symbol=:auto,
    alpha_backend::Symbol=:auto,
    dense_ch_max_bytes::Real=512.0 * 1024.0^2,
    dense_alpha_max_bytes::Real=512.0 * 1024.0^2,
    dense_ch_pmax::Integer=24,
    dense_alpha_pmax::Integer=48,
)
    N < 0 && throw(ArgumentError("N must be nonnegative."))

    C, interp_info = adaptive_legendre_interp_coeffs_Khat(
        K;
        tol=tol,
        nmin=nmin,
        mmin=mmin,
        nmax=nmax,
        mmax=mmax,
        growth=growth,
        tail=tail,
        verbose=verbose,
        maxiters=maxiters,
        warn_exceed_max=warn_exceed_max,
        adaptive_mode=adaptive_mode,
        plateau_tol=plateau_tol,
        plateau_window=plateau_window,
        plateau_drop=plateau_drop,
        chop_guard=chop_guard,
    )

    assembly_info = _assembly_info(
        C,
        N,
        bandwidth,
        ch_backend,
        alpha_backend,
        dense_ch_max_bytes,
        dense_alpha_max_bytes,
        dense_ch_pmax,
        dense_alpha_pmax,
    )

    B = legendre_volterra_symband_from_C(
        C,
        N;
        bandwidth=bandwidth,
        dropzeros_tol=dropzeros_tol,
        ch_backend=ch_backend,
        alpha_backend=alpha_backend,
        dense_ch_max_bytes=dense_ch_max_bytes,
        dense_alpha_max_bytes=dense_alpha_max_bytes,
        dense_ch_pmax=dense_ch_pmax,
        dense_alpha_pmax=dense_alpha_pmax,
    )

    info = (
        interp_info...,
        assembly_info...,
        matrix_size = size(B),
        nnz = nnz(B),
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


# -----------------------------------------------------------------------------
# Main interface from a known coefficient matrix C.
#
#     B = voltmat_leg(C, N)
#
# Here C[i+1,j+1] is the Legendre coefficient of P_i(x)P_j(t) in
#
#     Khat(x,t) = K(x, (x+1)/2*t + (x-1)/2).
#
# This method bypasses adaptive interpolation and directly constructs the
# requested symmetric band of the square matrix B.
# -----------------------------------------------------------------------------

function voltmat_leg(
    C::AbstractMatrix{<:Number},
    N::Integer;
    bandwidth::Union{Nothing,Integer}=nothing,
    return_info::Bool=false,
    dropzeros_tol::Float64=0.0,
    ch_backend::Symbol=:auto,
    alpha_backend::Symbol=:auto,
    dense_ch_max_bytes::Real=512.0 * 1024.0^2,
    dense_alpha_max_bytes::Real=512.0 * 1024.0^2,
    dense_ch_pmax::Integer=24,
    dense_alpha_pmax::Integer=48,
)
    N < 0 && throw(ArgumentError("N must be nonnegative."))

    assembly_info = _assembly_info(
        C,
        N,
        bandwidth,
        ch_backend,
        alpha_backend,
        dense_ch_max_bytes,
        dense_alpha_max_bytes,
        dense_ch_pmax,
        dense_alpha_pmax,
    )

    B = legendre_volterra_symband_from_C(
        C,
        N;
        bandwidth=bandwidth,
        dropzeros_tol=dropzeros_tol,
        ch_backend=ch_backend,
        alpha_backend=alpha_backend,
        dense_ch_max_bytes=dense_ch_max_bytes,
        dense_alpha_max_bytes=dense_alpha_max_bytes,
        dense_ch_pmax=dense_ch_pmax,
        dense_alpha_pmax=dense_alpha_pmax,
    )

    info = (
        nrow = size(C, 1),
        mcol = size(C, 2),
        assembly_info...,
        matrix_size = size(B),
        nnz = nnz(B),
    )

    return return_info ? (B, info) : B
end

end # module VoltMat_Leg
