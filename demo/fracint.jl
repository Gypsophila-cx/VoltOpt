#!/usr/bin/env julia

# Fast demo-level construction for the K(x,t)=1 DE weakly singular Volterra
# fractional-integration matrix.
#
# Default path:
#   assembly=:narrow_direct_banded, output=:banded.
#   H_{0,r,r+s} is initialized by analytic h0,h1 values followed by the stable
#   three-term recurrence in s.  The Float64 terminating 3F2 formula is kept
#   only as an unstable diagnostic path.  FFT convolution remains available as
#   a verification/candidate path, but with fixed C=100x100 it is not the
#   default.
#
# Complexity, with fixed Csize_y=Csize_eta=100 and fixed upper_bw:
#   I = 99, J = 99, S = O(1), Lfft = O(1).
#   Initial H0 construction costs O(N * Sguard^2), hence O(N).
#   The Chebyshev-in-eta recurrence costs O(N * J * Sguard), hence O(N).
#   The intended FFT convolution assembly costs O(N * J * Lfft log Lfft),
#   hence O(N) for fixed J and Lfft.  The FFT path is verified against direct
#   convolution and direct scatter, but with fixed C=100x100 the
#   direct_convolution path is currently faster and is therefore the default.
#   The final banded multiplication by (1+y)/2 costs O(N * (I + upper_bw)),
#   hence O(N).
#   Total effective work is O(N) for fixed bandwidths and fixed C sizes.
#
# Storage is O(N^2) for the returned dense matrix, solely to make comparison
# with the older demo implementation straightforward.  A packed banded return
# would reduce effective storage to O(N).

include(joinpath(@__DIR__, "weak_voltmat_decheb.jl"))

module FastFracIntDE

export fast_fracint_K1_decheb, compare_with_old, compare_assemblies, benchmark_fast_only,
    band_to_dense, maxabs_banddiff, compare_fft_branches, test_convolution_index_random,
    compare_narrow_H_generator, compare_narrow_direct_banded, benchmark_narrow_only,
    benchmark_mu_sweep, compare_narrow_fft_batched, compare_narrow_fft_packed_absorbM,
    benchmark_fft_packed_blockcols, generate_weak_narrow_tensor,
    compare_narrow_fft_tensor_absorbM, benchmark_fft_tensor_absorbM,
    generate_weak_column_tensor, compare_narrow_fft_tensor_accum_absorbM,
    benchmark_fft_tensor_accum, compare_narrow_fft_tensor_accum_blocked_absorbM,
    benchmark_fft_tensor_accum_blocked, compare_narrow_rfft_tensor_accum_absorbM,
    compare_narrow_direct_absorbM, compare_narrow_rfft_stream_accum_absorbM,
    compare_narrow_rfft_tensor_accum_blocked_absorbM, benchmark_fft_stream_accum,
    benchmark_fft_rfft_blocked_absorbM, benchmark_fft_plan_vs_assembly,
    plan_fracint_K1_fft, assemble_fracint_K1_fft!

using LinearAlgebra
using Printf
using Random
using ..Weak_VoltMat_DE

const HAS_FFTW = Ref(false)
try
    @eval import FFTW
    HAS_FFTW[] = true
catch err
    @warn "FFTW.jl is not available; assembly=:fft_convolution will error until FFTW is available in this Julia environment." exception=err maxlog=1
end

const Csize_y = 100
const Csize_eta = 100
const I_DEG = Csize_y - 1
const J_DEG = Csize_eta - 1

struct PBlockFrac
    ii::Vector{Int}
    vv::Vector{Float64}
    dense::Vector{Float64}
end

mutable struct HLineRingCache
    Hcache::Array{Float64, 3}
    valid_r::Vector{Int}
    Bvals::Vector{Float64}
    S::Int
    Sguard::Int
    Rcache::Int
end

mutable struct FracIntK1FFTPlan
    C::Matrix{Float64}
    infoC::Any
    P::Matrix{Float64}
    mu::Float64
    omega::Float64
    N::Int
    upper_bw::Int
    lower_bw_A::Int
    S_A::Int
    Sguard_A::Int
    Lfft::Int
    Lfreq::Int
    block_cols::Int
    assembly::Symbol
    h0_init::Symbol
    prec_bits::Int
    fftw_flags::Any
    Fp::Matrix{ComplexF64}
    Fr::Matrix{ComplexF64}
    Qbuf::Array{Float64, 3}
    Qhat::Array{ComplexF64, 3}
    Uhat::Matrix{ComplexF64}
    Lhat::Matrix{ComplexF64}
    Ubuf::Matrix{Float64}
    Lbuf::Matrix{Float64}
    q_highwater::Matrix{Int}
    Hring::HLineRingCache
    HcompP::Matrix{Float64}
    scratch_ridx::Matrix{Int}
    scratch_weight::Matrix{Float64}
    left_p::Vector{Int}
    left_col::Vector{Int}
    left_value::Vector{Float64}
    plan_q_fwd::Any
    plan_u_inv::Any
    plan_l_inv::Any
    memory_estimate::Int
end

_timer_start(return_stats::Bool) = return_stats ? time_ns() : 0
_timer_elapsed(return_stats::Bool, t0::Integer) = return_stats ? (time_ns() - t0) / 1.0e9 : 0.0

macro stat_elapsed(return_stats, acc, ex)
    return quote
        if $(esc(return_stats))
            $(esc(acc)) += @elapsed $(esc(ex))
        else
            $(esc(ex))
        end
    end
end

function resolve_fftw_flags(fftw_flags)
    HAS_FFTW[] || error("FFTW.jl is required for FFT assembly")
    return fftw_flags === nothing ? FFTW.MEASURE : fftw_flags
end

maxabs_array(A) = isempty(A) ? 0.0 : maximum(abs, A)

@inline function band_get(B::AbstractMatrix{<:Real}, ell::Int, k::Int, lower_bw::Int, upper_bw::Int)
    d = k - ell
    -lower_bw <= d <= upper_bw || return 0.0
    0 <= k <= size(B, 2) - 1 || return 0.0
    p = d + lower_bw + 1
    1 <= p <= size(B, 1) || return 0.0
    return B[p, k + 1]
end

function band_to_dense(Aband::AbstractMatrix{<:Real}, N::Int, lower_bw::Int, upper_bw::Int)
    A = zeros(Float64, N, N)
    for k in 0:(N - 1)
        for d in -lower_bw:upper_bw
            ell = k - d
            0 <= ell <= N - 1 || continue
            p = d + lower_bw + 1
            A[ell + 1, k + 1] = Aband[p, k + 1]
        end
    end
    return A
end

maxabs_banddiff(Aband1::AbstractMatrix{<:Real}, Aband2::AbstractMatrix{<:Real}) =
    maxabs_array(Aband1 .- Aband2)

function neumaier_sum_terms(s::Int, r::Int, mu::Float64)
    total = 1.0
    comp = 0.0
    term = 1.0
    for q in 0:(s - 1)
        term *= ((-s + q) * (s + 2.0r + q) * (r + 1.0 + q)) /
                ((2.0r + 1.0 + q) * (r + mu + 1.0 + q) * (q + 1.0))
        t = total + term
        if abs(total) >= abs(term)
            comp += (total - t) + term
        else
            comp += (term - t) + total
        end
        total = t
    end
    return total + comp
end

# Float64 terminating 3F2 formula is numerically unstable for large s and is
# only kept as a diagnostic reference.
function h0_line_hypergeom(r::Int, mu::Float64, smax::Int, Br::Float64)
    H0 = Vector{Float64}(undef, smax + 1)
    pref = 2.0^mu * Br
    Aterm = 1.0
    for s in 0:smax
        if s == 0
            H0[s + 1] = pref
        else
            Aterm *= -(2.0r + s) / s
            H0[s + 1] = pref * Aterm * neumaier_sum_terms(s, r, mu)
        end
    end
    return H0
end

function h0_line_s_recurrence(r::Int, mu::Float64, smax::Int, Br::Float64)
    smax >= 0 || throw(ArgumentError("smax must be nonnegative"))
    H0 = Vector{Float64}(undef, smax + 1)
    h0 = 2.0^mu * Br
    H0[1] = h0
    smax == 0 && return H0

    H0[2] = -mu * (2.0r + 1.0) / (r + mu + 1.0) * h0
    for s in 0:(smax - 2)
        A = -0.5 * s * (s + 2.0r + 1.0) * (2.0s + 2.0r + 3.0) * (s + r - mu)
        B = (s + r + 1.0) *
            (
                mu * ((2.0r + 1.0)^2 + 2.0s * (s + 2.0r + 2.0)) -
                (s * (s + 2.0r + 2.0) + 2.0r + 1.0)
            )
        Cc = 0.5 * (s + 2.0) * (s + 2.0r + 1.0) * (2.0s + 2.0r + 1.0) *
             (s + r + mu + 2.0)
        isfinite(Cc) && Cc != 0.0 || error("invalid s-recurrence C coefficient at r=$r, s=$s: $Cc")
        H0[s + 3] = -(A * H0[s + 1] + B * H0[s + 2]) / Cc
        isfinite(H0[s + 3]) || error("non-finite H0 value from s-recurrence at r=$r, s=$(s + 2)")
    end
    return H0
end

function h0_line_bigfloat_hypergeom(
    r::Int,
    mu::Float64,
    smax::Int,
    Br::Float64;
    prec_bits::Int=1536,
)
    smax >= 0 || throw(ArgumentError("smax must be nonnegative"))
    return setprecision(BigFloat, prec_bits) do
        H0 = Vector{Float64}(undef, smax + 1)
        rB = BigFloat(r)
        muB = BigFloat(mu)
        pref = big(2)^muB * BigFloat(Br)
        Aterm = big(1)
        for s in 0:smax
            if s > 0
                Aterm *= -(2rB + BigFloat(s)) / BigFloat(s)
            end
            term = big(1)
            total = big(1)
            for q in 0:(s - 1)
                qB = BigFloat(q)
                sB = BigFloat(s)
                term *= ((-sB + qB) * (sB + 2rB + qB) * (rB + 1 + qB)) /
                        ((2rB + 1 + qB) * (rB + muB + 1 + qB) * (qB + 1))
                total += term
            end
            H0[s + 1] = Float64(pref * Aterm * total)
        end
        H0
    end
end

@inline function eta_coeffs(r::Int, s::Int)
    if s == 0
        return 2.0 / (2.0r + 1.0), 1.0, 0.0
    end
    n = Float64(s)
    beta = 2.0r
    a = 2.0 * (n + 1.0) * (n + beta) / ((2.0n + beta) * (2.0n + beta + 1.0))
    b = (beta^2 - 1.0) / ((2.0n + beta - 1.0) * (2.0n + beta + 1.0))
    c = 2.0 * (n - 1.0) * (n + beta) / ((2.0n + beta - 1.0) * (2.0n + beta))
    return a, b, c
end

@inline function eta_times_line(vals::AbstractVector{Float64}, r::Int, s::Int)
    a, b, c = eta_coeffs(r, s)
    out = b * vals[s + 1]
    if s + 2 <= length(vals)
        out += a * vals[s + 2]
    end
    if s >= 1
        out += c * vals[s]
    end
    return out
end

function build_Hjs_line!(
    dest::AbstractMatrix{Float64},
    r::Int,
    mu::Float64,
    S::Int,
    Sguard::Int,
    Br::Float64;
    h0_init::Symbol=:s_recurrence,
    prec_bits::Int=1536,
    layout::Symbol=:js,
)
    if layout === :js
        size(dest, 1) >= J_DEG + 1 || throw(ArgumentError("dest must have at least $(J_DEG + 1) rows"))
        size(dest, 2) >= S + 1 || throw(ArgumentError("dest must have at least $(S + 1) columns"))
    elseif layout === :sj
        size(dest, 1) >= S + 1 || throw(ArgumentError("dest must have at least $(S + 1) rows"))
        size(dest, 2) >= J_DEG + 1 || throw(ArgumentError("dest must have at least $(J_DEG + 1) columns"))
    else
        throw(ArgumentError("layout must be :js or :sj"))
    end
    H0_ext = if h0_init === :s_recurrence
        h0_line_s_recurrence(r, mu, Sguard + 1, Br)
    elseif h0_init === :hypergeom_float64
        h0_line_hypergeom(r, mu, Sguard + 1, Br)
    elseif h0_init === :bigfloat_hypergeom
        h0_line_bigfloat_hypergeom(r, mu, Sguard + 1, Br; prec_bits)
    else
        throw(ArgumentError("h0_init must be :s_recurrence, :hypergeom_float64, or :bigfloat_hypergeom"))
    end
    Hwork = zeros(Float64, J_DEG + 1, Sguard + 1)
    for s in 0:Sguard
        Hwork[1, s + 1] = H0_ext[s + 1]
    end
    if J_DEG >= 1
        for s in 0:Sguard
            Hwork[2, s + 1] = eta_times_line(H0_ext, r, s)
        end
    end
    for j in 1:(J_DEG - 1)
        prev = @view Hwork[j, :]
        curr = @view Hwork[j + 1, :]
        next = @view Hwork[j + 2, :]
        for s in 0:Sguard
            next[s + 1] = 2.0 * eta_times_line(curr, r, s) - prev[s + 1]
        end
    end
    if layout === :js
        @views dest[1:(J_DEG + 1), 1:(S + 1)] .= Hwork[:, 1:(S + 1)]
    else
        @inbounds for j in 0:J_DEG
            for s in 0:S
                dest[s + 1, j + 1] = Hwork[j + 1, s + 1]
            end
        end
    end
    return dest
end

function build_Hjs_line(
    r::Int,
    mu::Float64,
    S::Int,
    Sguard::Int,
    Br::Float64;
    h0_init::Symbol=:s_recurrence,
    prec_bits::Int=1536,
)
    Hjs = Matrix{Float64}(undef, J_DEG + 1, S + 1)
    build_Hjs_line!(Hjs, r, mu, S, Sguard, Br; h0_init, prec_bits)
    return Hjs
end

@inline function add_G!(Gband::Matrix{Float64}, ell::Int, k::Int, value::Float64, lower_bw::Int, upper_bw_G::Int)
    0 <= k <= size(Gband, 2) - 1 || return nothing
    d = k - ell
    -lower_bw <= d <= upper_bw_G || return nothing
    p = d + lower_bw + 1
    scale = (2.0 - (ell == 0 ? 1.0 : 0.0)) / 2.0
    Gband[p, k + 1] += scale * value
    return nothing
end

function assemble_G_direct_banded(
    C::Matrix{Float64},
    mu::Float64,
    N::Int,
    upper_bw::Int,
    S::Int,
    Sguard::Int;
    h0_init::Symbol=:s_recurrence,
    prec_bits::Int=1536,
)
    lower_bw = I_DEG
    upper_bw_G = upper_bw + 1
    G = zeros(Float64, lower_bw + upper_bw_G + 1, N)
    Br = 1.0 / mu

    for r in 0:(N - 1)
        Hjs = build_Hjs_line(r, mu, S, Sguard, Br; h0_init, prec_bits)
        smax = min(S, N - 1 - r)
        Hcomp = C * @view(Hjs[:, 1:(smax + 1)])

        for s in 0:smax
            k = r + s
            for i in 0:I_DEG
                h = Hcomp[i + 1, s + 1]
                h == 0.0 && continue

                if k == 0
                    # Special restriction formula column:
                    #   G[ell,0] = Hcomp[ell,0,0].
                    # There is no Chebyshev-product scale and ell=i.
                    if r == 0
                        d = -i
                        if -lower_bw <= d <= upper_bw_G
                            G[d + lower_bw + 1, 1] += h
                        end
                    end
                    continue
                end

                weighted = (k / (k + r)) * h

                # First branch: r = i + ell.
                ell1 = r - i
                if ell1 >= 0
                    add_G!(G, ell1, k, weighted, lower_bw, upper_bw_G)
                end

                # Second branch: r = |i - ell|.  For r=0 there is only one
                # ell=i solution; otherwise there are the two symmetric
                # solutions ell=i-r and ell=i+r.
                ell2 = i - r
                if ell2 >= 0
                    add_G!(G, ell2, k, weighted, lower_bw, upper_bw_G)
                end
                if r != 0
                    add_G!(G, i + r, k, weighted, lower_bw, upper_bw_G)
                end
            end
        end

        if r < N - 1
            Br *= (r + 1.0) / (r + mu + 1.0)
        end
    end
    return G
end

function scatter_H_value!(G, h::Float64, i::Int, r::Int, k::Int, lower_bw::Int, upper_bw_G::Int)
    h == 0.0 && return nothing

    if k == 0
        # Special restriction formula column:
        #   G[ell,0] = Hcomp[ell,0,0]
        # There is no (2-delta)/2 scale and ell is the y-Chebyshev index i.
        if r == 0
            d = -i
            if -lower_bw <= d <= upper_bw_G
                G[d + lower_bw + 1, 1] += h
            end
        end
        return nothing
    end

    weighted = (k / (k + r)) * h

    ell1 = r - i
    if ell1 >= 0
        add_G!(G, ell1, k, weighted, lower_bw, upper_bw_G)
    end

    ell2 = i - r
    if ell2 >= 0
        add_G!(G, ell2, k, weighted, lower_bw, upper_bw_G)
    end
    if r != 0
        add_G!(G, i + r, k, weighted, lower_bw, upper_bw_G)
    end
    return nothing
end

function build_verified_H_layers(mu::Float64, N::Int)
    Kext = N - 1
    H00 = Weak_VoltMat_DE.stable_de_H000(mu, Kext + J_DEG)
    H_r0 = Weak_VoltMat_DE.build_r0_layer_from_H00(H00, J_DEG, Kext)
    Hlayers = Vector{Matrix{Float64}}(undef, J_DEG + 1)
    for j in 0:J_DEG
        Hlayers[j + 1] = Weak_VoltMat_DE.build_H_fixed_j_from_r0(j, @view(H_r0[j + 1, :]), Kext; mu)
    end
    return Hlayers
end

function build_verified_H_seed(mu::Float64, N::Int)
    Kext = N - 1
    H00 = Weak_VoltMat_DE.stable_de_H000(mu, Kext + J_DEG)
    return Weak_VoltMat_DE.build_r0_layer_from_H00(H00, J_DEG, Kext)
end

function beta_r_values(mu::Float64, N::Int)
    Bvals = Vector{Float64}(undef, N)
    Bvals[1] = 1.0 / mu
    for r in 0:(N - 2)
        Bvals[r + 2] = (r + 1.0) / (r + mu + 1.0) * Bvals[r + 1]
    end
    return Bvals
end

function assemble_G_direct_verified(
    C::Matrix{Float64},
    mu::Float64,
    N::Int,
    upper_bw::Int;
    Hlayers=nothing,
)
    lower_bw = I_DEG
    upper_bw_G = upper_bw + 1
    Kext = N - 1
    G = zeros(Float64, lower_bw + upper_bw_G + 1, N)
    Hlocal = Hlayers === nothing ? build_verified_H_layers(mu, N) : Hlayers

    # Verified direct scatter: old stable fixed-j recurrence, streamed one
    # j-layer at a time and scattered only into the kept G band.
    for j in 0:J_DEG
        Hjr = Hlocal[j + 1]
        for k in 0:Kext
            for r in 0:k
                H = Hjr[r + 1, k + 1]
                H == 0.0 && continue
                for i in 0:I_DEG
                    cij = C[i + 1, j + 1]
                    cij == 0.0 && continue
                    scatter_H_value!(G, cij * H, i, r, k, lower_bw, upper_bw_G)
                end
            end
        end
    end
    return G
end

@inline function add_G_by_d!(Gband::Matrix{Float64}, k::Int, d::Int, value::Float64, lower_bw_G::Int, upper_bw_G::Int)
    value == 0.0 && return nothing
    ell = k - d
    ell >= 0 || return nothing
    -lower_bw_G <= d <= upper_bw_G || return nothing
    scale = (2.0 - (ell == 0 ? 1.0 : 0.0)) / 2.0
    Gband[d + lower_bw_G + 1, k + 1] += scale * value
    return nothing
end

function add_k0_column!(G::Matrix{Float64}, C::Matrix{Float64}, Hlayers)
    lower_bw_G = I_DEG
    for i in 0:I_DEG
        h = 0.0
        for j in 0:J_DEG
            h += C[i + 1, j + 1] * Hlayers[j + 1][1, 1]
        end
        G[-i + lower_bw_G + 1, 1] += h
    end
    return nothing
end

function add_left_corner_correction!(
    G::Matrix{Float64},
    C::Matrix{Float64},
    Hlayers,
    k::Int,
    smax::Int,
    lower_bw_G::Int,
    upper_bw_G::Int,
)
    # The FFT/direct convolution L_d term represents ell=i+r.  The symmetric
    # branch ell=i-r contributes only to a finite left-corner block of the kept
    # band; scatter it directly as an exact correction.
    for s in 0:smax
        r = k - s
        r == 0 && continue
        for i in r:I_DEG
            d = 2k - s - i
            -lower_bw_G <= d <= upper_bw_G || continue
            h = 0.0
            for j in 0:J_DEG
                h += C[i + 1, j + 1] * (k / (k + r)) * Hlayers[j + 1][r + 1, k + 1]
            end
            add_G_by_d!(G, k, d, h, lower_bw_G, upper_bw_G)
        end
    end
    return nothing
end

function add_k0_column_narrow!(
    G::Matrix{Float64},
    C::Matrix{Float64},
    mu::Float64,
    S::Int,
    Sguard::Int,
    Bvals::AbstractVector{Float64};
    h0_init::Symbol=:s_recurrence,
    prec_bits::Int=1536,
)
    lower_bw_G = I_DEG
    Hjs0 = build_Hjs_line(0, mu, S, Sguard, Bvals[1]; h0_init, prec_bits)
    Hcomp0 = C * @view(Hjs0[:, 1:1])
    for i in 0:I_DEG
        G[-i + lower_bw_G + 1, 1] += Hcomp0[i + 1, 1]
    end
    return nothing
end

function add_left_corner_correction_narrow!(
    G::Matrix{Float64},
    C::Matrix{Float64},
    mu::Float64,
    N::Int,
    upper_bw::Int,
    S::Int,
    Sguard::Int,
    Bvals::AbstractVector{Float64};
    h0_init::Symbol=:s_recurrence,
    prec_bits::Int=1536,
)
    lower_bw_G = I_DEG
    upper_bw_G = upper_bw + 1
    for r in 1:min(I_DEG, N - 1)
        Hjs = build_Hjs_line(r, mu, S, Sguard, Bvals[r + 1]; h0_init, prec_bits)
        smax = min(S, N - 1 - r)
        Hcomp = C * @view(Hjs[:, 1:(smax + 1)])
        for s in 0:smax
            k = r + s
            w = k / (k + r)
            for i in r:I_DEG
                d = 2k - s - i
                -lower_bw_G <= d <= upper_bw_G || continue
                add_G_by_d!(G, k, d, w * Hcomp[i + 1, s + 1], lower_bw_G, upper_bw_G)
            end
        end
    end
    return nothing
end

function assemble_G_direct_convolution(
    C::Matrix{Float64},
    mu::Float64,
    N::Int,
    upper_bw::Int;
    Hlayers=nothing,
    Hseed=nothing,
)
    lower_bw_G = I_DEG
    upper_bw_G = upper_bw + 1
    S = upper_bw_G + I_DEG + 1
    G = zeros(Float64, lower_bw_G + upper_bw_G + 1, N)

    if Hlayers === nothing
        H_r0 = Hseed === nothing ? build_verified_H_seed(mu, N) : Hseed
        for j in 0:J_DEG
            Hjr = Weak_VoltMat_DE.build_H_fixed_j_from_r0(j, @view(H_r0[j + 1, :]), N - 1; mu)
            for i in 0:I_DEG
                G[-i + lower_bw_G + 1, 1] += C[i + 1, j + 1] * Hjr[1, 1]
            end
            for k in 1:(N - 1)
                acc = zeros(Float64, lower_bw_G + upper_bw_G + 1)
                smax = min(k, S)
                for s in 0:smax
                    r = k - s
                    q = (k / (k + r)) * Hjr[r + 1, k + 1]
                    q == 0.0 && continue
                    for i in 0:I_DEG
                        cijq = C[i + 1, j + 1] * q
                        cijq == 0.0 && continue
                        dU = i + s
                        if 0 <= dU <= upper_bw_G
                            acc[dU + lower_bw_G + 1] += cijq
                        end
                        dL = s - i
                        if -lower_bw_G <= dL <= upper_bw_G
                            acc[dL + lower_bw_G + 1] += cijq
                        end
                    end
                    r == 0 && continue
                    for i in r:I_DEG
                        d = 2k - s - i
                        -lower_bw_G <= d <= upper_bw_G || continue
                        add_G_by_d!(G, k, d, C[i + 1, j + 1] * q, lower_bw_G, upper_bw_G)
                    end
                end
                for d in -lower_bw_G:upper_bw_G
                    add_G_by_d!(G, k, d, acc[d + lower_bw_G + 1], lower_bw_G, upper_bw_G)
                end
            end
        end
        return G
    end

    Hlocal = Hlayers
    add_k0_column!(G, C, Hlocal)

    for k in 1:(N - 1)
        acc = zeros(Float64, lower_bw_G + upper_bw_G + 1)
        smax = min(k, S)
        for j in 0:J_DEG
            Hjr = Hlocal[j + 1]
            for s in 0:smax
                r = k - s
                q = (k / (k + r)) * Hjr[r + 1, k + 1]
                q == 0.0 && continue
                for i in 0:I_DEG
                    cijq = C[i + 1, j + 1] * q
                    cijq == 0.0 && continue

                    dU = i + s
                    if 0 <= dU <= upper_bw_G
                        acc[dU + lower_bw_G + 1] += cijq
                    end

                    dL = s - i
                    if -lower_bw_G <= dL <= upper_bw_G
                        acc[dL + lower_bw_G + 1] += cijq
                    end
                end
            end
        end

        for d in -lower_bw_G:upper_bw_G
            add_G_by_d!(G, k, d, acc[d + lower_bw_G + 1], lower_bw_G, upper_bw_G)
        end
        add_left_corner_correction!(G, C, Hlocal, k, smax, lower_bw_G, upper_bw_G)
    end
    return G
end

function padded_fft(v::AbstractVector{<:Real}, Lfft::Int)
    x = zeros(Float64, Lfft)
    x[1:length(v)] .= v
    return FFTW.fft(x)
end

function assemble_G_fft_convolution(
    C::Matrix{Float64},
    mu::Float64,
    N::Int,
    upper_bw::Int;
    Hlayers=nothing,
    Hseed=nothing,
)
    HAS_FFTW[] || error("FFTW.jl is required for assembly=:fft_convolution. Add FFTW to the project or use assembly=:direct_convolution/:direct_scatter.")
    lower_bw_G = I_DEG
    upper_bw_G = upper_bw + 1
    S = upper_bw_G + I_DEG + 1
    Lfft = nextpow(2, I_DEG + S + 2)
    G = zeros(Float64, lower_bw_G + upper_bw_G + 1, N)

    Chat_plus = Matrix{ComplexF64}(undef, Lfft, J_DEG + 1)
    Chat_minus = Matrix{ComplexF64}(undef, Lfft, J_DEG + 1)
    for j in 0:J_DEG
        Chat_plus[:, j + 1] .= padded_fft(@view(C[:, j + 1]), Lfft)
        Chat_minus[:, j + 1] .= padded_fft(reverse(C[:, j + 1]), Lfft)
    end

    qbuf = zeros(Float64, Lfft)
    if Hlayers === nothing
        H_r0 = Hseed === nothing ? build_verified_H_seed(mu, N) : Hseed
        for j in 0:J_DEG
            Hjr = Weak_VoltMat_DE.build_H_fixed_j_from_r0(j, @view(H_r0[j + 1, :]), N - 1; mu)
            for i in 0:I_DEG
                G[-i + lower_bw_G + 1, 1] += C[i + 1, j + 1] * Hjr[1, 1]
            end
            for k in 1:(N - 1)
                fill!(qbuf, 0.0)
                smax = min(k, S)
                for s in 0:smax
                    r = k - s
                    qbuf[s + 1] = (k / (k + r)) * Hjr[r + 1, k + 1]
                end
                Qhat = FFTW.fft(qbuf)
                U = real.(FFTW.ifft(Chat_plus[:, j + 1] .* Qhat))
                L = real.(FFTW.ifft(Chat_minus[:, j + 1] .* Qhat))
                for d in -lower_bw_G:upper_bw_G
                    value = 0.0
                    d >= 0 && (value += U[d + 1])
                    rev_idx0 = I_DEG + d
                    0 <= rev_idx0 < Lfft && (value += L[rev_idx0 + 1])
                    add_G_by_d!(G, k, d, value, lower_bw_G, upper_bw_G)
                end
                for s in 0:smax
                    r = k - s
                    r == 0 && continue
                    q = qbuf[s + 1]
                    for i in r:I_DEG
                        d = 2k - s - i
                        -lower_bw_G <= d <= upper_bw_G || continue
                        add_G_by_d!(G, k, d, C[i + 1, j + 1] * q, lower_bw_G, upper_bw_G)
                    end
                end
            end
        end
        return G
    end

    Hlocal = Hlayers
    add_k0_column!(G, C, Hlocal)

    for k in 1:(N - 1)
        Uhat = zeros(ComplexF64, Lfft)
        Lhat = zeros(ComplexF64, Lfft)
        smax = min(k, S)
        for j in 0:J_DEG
            fill!(qbuf, 0.0)
            Hjr = Hlocal[j + 1]
            for s in 0:smax
                r = k - s
                qbuf[s + 1] = (k / (k + r)) * Hjr[r + 1, k + 1]
            end
            Qhat = FFTW.fft(qbuf)
            @. Uhat += Chat_plus[:, j + 1] * Qhat
            @. Lhat += Chat_minus[:, j + 1] * Qhat
        end

        U = real.(FFTW.ifft(Uhat))
        L = real.(FFTW.ifft(Lhat))
        for d in -lower_bw_G:upper_bw_G
            value = 0.0
            if d >= 0
                value += U[d + 1]
            end
            rev_idx0 = I_DEG + d
            if 0 <= rev_idx0 < Lfft
                value += L[rev_idx0 + 1]
            end
            add_G_by_d!(G, k, d, value, lower_bw_G, upper_bw_G)
        end
        add_left_corner_correction!(G, C, Hlocal, k, smax, lower_bw_G, upper_bw_G)
    end
    return G
end

function assemble_G_narrow_fft_batched(
    C::Matrix{Float64},
    mu::Float64,
    N::Int,
    upper_bw::Int,
    S::Int,
    Sguard::Int;
    h0_init::Symbol=:s_recurrence,
    prec_bits::Int=1536,
    block_cols::Int=64,
    fftw_flags=nothing,
)
    HAS_FFTW[] || error("FFTW.jl is required for assembly=:narrow_fft_batched")
    block_cols >= 1 || throw(ArgumentError("block_cols must be positive"))
    lower_bw_G = I_DEG
    upper_bw_G = upper_bw + 1
    Lfft = nextpow(2, I_DEG + S + 2)
    G = zeros(Float64, lower_bw_G + upper_bw_G + 1, N)
    Bvals = beta_r_values(mu, N)

    Chat_plus = Matrix{ComplexF64}(undef, Lfft, J_DEG + 1)
    Chat_minus = Matrix{ComplexF64}(undef, Lfft, J_DEG + 1)
    for j in 0:J_DEG
        Chat_plus[:, j + 1] .= padded_fft(@view(C[:, j + 1]), Lfft)
        Chat_minus[:, j + 1] .= padded_fft(reverse(C[:, j + 1]), Lfft)
    end

    add_k0_column_narrow!(G, C, mu, S, Sguard, Bvals; h0_init, prec_bits)

    for k0 in 1:block_cols:(N - 1)
        k1 = min(N - 1, k0 + block_cols - 1)
        nb = k1 - k0 + 1
        Qbuf = zeros(ComplexF64, Lfft, J_DEG + 1, nb)
        Uhat = zeros(ComplexF64, Lfft, nb)
        Lhat = zeros(ComplexF64, Lfft, nb)

        rlo = max(0, k0 - S)
        rhi = k1
        for r in rlo:rhi
            Hjs = build_Hjs_line(r, mu, S, Sguard, Bvals[r + 1]; h0_init, prec_bits)
            smax = min(S, N - 1 - r)
            for s in 0:smax
                k = r + s
                k0 <= k <= k1 || continue
                k > 0 || continue
                b = k - k0 + 1
                w = k / (k + r)
                @inbounds for j in 0:J_DEG
                    Qbuf[s + 1, j + 1, b] = w * Hjs[j + 1, s + 1]
                end
            end
        end

        FFTW.fft!(Qbuf, 1)
        @inbounds for p in 1:Lfft
            for b in 1:nb
                up = 0.0 + 0.0im
                lp = 0.0 + 0.0im
                for j in 1:(J_DEG + 1)
                    q = Qbuf[p, j, b]
                    up += Chat_plus[p, j] * q
                    lp += Chat_minus[p, j] * q
                end
                Uhat[p, b] = up
                Lhat[p, b] = lp
            end
        end
        FFTW.ifft!(Uhat, 1)
        FFTW.ifft!(Lhat, 1)

        @inbounds for b in 1:nb
            k = k0 + b - 1
            for d in -lower_bw_G:upper_bw_G
                value = 0.0
                d >= 0 && (value += real(Uhat[d + 1, b]))
                rev_idx0 = I_DEG + d
                0 <= rev_idx0 < Lfft && (value += real(Lhat[rev_idx0 + 1, b]))
                add_G_by_d!(G, k, d, value, lower_bw_G, upper_bw_G)
            end
        end
    end

    add_left_corner_correction_narrow!(G, C, mu, N, upper_bw, S, Sguard, Bvals; h0_init, prec_bits)
    return G
end

@inline function add_A_by_d!(Aband::Matrix{Float64}, k::Int, d::Int, value::Float64, lower_bw_A::Int, upper_bw::Int)
    value == 0.0 && return nothing
    ell = k - d
    0 <= ell <= size(Aband, 2) - 1 || return nothing
    -lower_bw_A <= d <= upper_bw || return nothing
    scale = (2.0 - (ell == 0 ? 1.0 : 0.0)) / 2.0
    Aband[d + lower_bw_A + 1, k + 1] += scale * value
    return nothing
end

function add_k0_column_absorbM!(
    Aband::Matrix{Float64},
    P::Matrix{Float64},
    mu::Float64,
    S_A::Int,
    Sguard_A::Int,
    Bvals::AbstractVector{Float64};
    h0_init::Symbol=:s_recurrence,
    prec_bits::Int=1536,
)
    IP_DEG = size(P, 1) - 1
    Hjs0 = build_Hjs_line(0, mu, S_A, Sguard_A, Bvals[1]; h0_init, prec_bits)
    Hcomp0 = P * @view(Hjs0[:, 1:1])
    for i in 0:IP_DEG
        i <= size(Aband, 2) - 1 || continue
        d = -i
        Aband[d + IP_DEG + 1, 1] += Hcomp0[i + 1, 1]
    end
    return nothing
end

function add_left_corner_correction_absorbM!(
    Aband::Matrix{Float64},
    P::Matrix{Float64},
    mu::Float64,
    N::Int,
    upper_bw::Int,
    S_A::Int,
    Sguard_A::Int,
    Bvals::AbstractVector{Float64};
    h0_init::Symbol=:s_recurrence,
    prec_bits::Int=1536,
)
    IP_DEG = size(P, 1) - 1
    lower_bw_A = IP_DEG
    for r in 1:min(IP_DEG, N - 1)
        Hjs = build_Hjs_line(r, mu, S_A, Sguard_A, Bvals[r + 1]; h0_init, prec_bits)
        smax = min(S_A, N - 1 - r)
        HcompP = P * @view(Hjs[:, 1:(smax + 1)])
        for s in 0:smax
            k = r + s
            weight = k / (k + r)
            for i in r:IP_DEG
                d = 2k - s - i
                -lower_bw_A <= d <= upper_bw || continue
                add_A_by_d!(Aband, k, d, weight * HcompP[i + 1, s + 1], lower_bw_A, upper_bw)
            end
        end
    end
    return nothing
end

function add_left_corner_correction_absorbM_fast!(
    Aband::Matrix{Float64},
    P::Matrix{Float64},
    mu::Float64,
    N::Int,
    upper_bw::Int,
    S_A::Int,
    Sguard_A::Int,
    Bvals::AbstractVector{Float64};
    h0_init::Symbol=:s_recurrence,
    prec_bits::Int=1536,
)
    IP_DEG = size(P, 1) - 1
    lower_bw_A = IP_DEG
    for r in 1:min(IP_DEG, N - 1)
        Hjs = build_Hjs_line(r, mu, S_A, Sguard_A, Bvals[r + 1]; h0_init, prec_bits)
        smax = min(S_A, N - 1 - r)
        HcompP = P * @view(Hjs[:, 1:(smax + 1)])
        for s in 0:smax
            k = r + s
            weight = k / (k + r)
            for i in r:IP_DEG
                d = 2k - s - i
                -lower_bw_A <= d <= upper_bw || continue
                add_A_by_d!(Aband, k, d, weight * HcompP[i + 1, s + 1], lower_bw_A, upper_bw)
            end
        end
    end
    return nothing
end

function assemble_A_narrow_fft_packed_absorbM(
    C::Matrix{Float64},
    mu::Float64,
    N::Int,
    upper_bw::Int,
    S::Int,
    Sguard::Int;
    h0_init::Symbol=:s_recurrence,
    prec_bits::Int=1536,
    block_cols::Int=256,
)
    HAS_FFTW[] || error("FFTW.jl is required for assembly=:narrow_fft_packed_absorbM")
    block_cols >= 1 || throw(ArgumentError("block_cols must be positive"))
    P = build_postmult_coeffs_frac(C)
    IP_DEG = size(P, 1) - 1
    lower_bw_A = IP_DEG
    S_A = upper_bw + IP_DEG
    Sguard_A = S_A + J_DEG + 2
    Lfft = nextpow(2, IP_DEG + S_A + 2)
    Aband = zeros(Float64, lower_bw_A + upper_bw + 1, N)
    Bvals = beta_r_values(mu, N)

    pblocks = build_P_blocks_frac(P)
    Fpack = precompute_packed_spectra_frac(pblocks, IP_DEG, Lfft)
    add_k0_column_absorbM!(Aband, P, mu, S_A, Sguard_A, Bvals; h0_init, prec_bits)

    for k0 in 1:block_cols:(N - 1)
        k1 = min(N - 1, k0 + block_cols - 1)
        nb = k1 - k0 + 1
        Qbuf = zeros(ComplexF64, Lfft, J_DEG + 1, nb)
        PackedHat = zeros(ComplexF64, Lfft, nb)

        rlo = max(0, k0 - S_A)
        rhi = k1
        for r in rlo:rhi
            Hjs = build_Hjs_line(r, mu, S_A, Sguard_A, Bvals[r + 1]; h0_init, prec_bits)
            smax = min(S_A, N - 1 - r)
            for s in 0:smax
                k = r + s
                k0 <= k <= k1 || continue
                k > 0 || continue
                b = k - k0 + 1
                weight = k / (k + r)
                @inbounds for j in 0:J_DEG
                    Qbuf[s + 1, j + 1, b] = weight * Hjs[j + 1, s + 1]
                end
            end
        end

        FFTW.fft!(Qbuf, 1)
        @inbounds for p in 1:Lfft
            for b in 1:nb
                acc = 0.0 + 0.0im
                for j in 1:(J_DEG + 1)
                    acc += Fpack[p, j] * Qbuf[p, j, b]
                end
                PackedHat[p, b] = acc
            end
        end
        FFTW.ifft!(PackedHat, 1)

        @inbounds for b in 1:nb
            k = k0 + b - 1
            for d in -lower_bw_A:upper_bw
                value = 0.0
                d >= 0 && (value += real(PackedHat[d + 1, b]))
                idx = IP_DEG + d
                0 <= idx < Lfft && (value += imag(PackedHat[idx + 1, b]))
                add_A_by_d!(Aband, k, d, value, lower_bw_A, upper_bw)
            end
        end
    end

    add_left_corner_correction_absorbM!(Aband, P, mu, N, upper_bw, S_A, Sguard_A, Bvals; h0_init, prec_bits)
    return Aband
end

function generate_weak_narrow_tensor(
    mu,
    N::Int,
    S::Int,
    Sguard::Int;
    h0_init::Symbol=:s_recurrence,
    prec_bits::Int=1536,
    weighted::Bool=true,
)
    mu64 = Float64(mu)
    Htensor = [zeros(Float64, S + 1, N) for _ in 0:J_DEG]
    Br = 1.0 / mu64
    for r in 0:(N - 1)
        Hjs = build_Hjs_line(r, mu64, S, Sguard, Br; h0_init, prec_bits)
        smax = min(S, N - 1 - r)
        for s in 0:smax
            k = r + s
            w = weighted && k > 0 ? k / (k + r) : 1.0
            @inbounds for j in 0:J_DEG
                Htensor[j + 1][s + 1, r + 1] = w * Hjs[j + 1, s + 1]
            end
        end
        if r < N - 1
            Br *= (r + 1.0) / (r + mu64 + 1.0)
        end
    end
    return Htensor
end

function generate_weak_column_tensor(
    mu,
    N::Int,
    S::Int,
    Sguard::Int;
    h0_init::Symbol=:s_recurrence,
    prec_bits::Int=1536,
    weighted::Bool=true,
)
    mu64 = Float64(mu)
    Qcols = zeros(Float64, S + 1, J_DEG + 1, N)
    Br = 1.0 / mu64
    for r in 0:(N - 1)
        Hjs = build_Hjs_line(r, mu64, S, Sguard, Br; h0_init, prec_bits)
        smax = min(S, N - 1 - r)
        for s in 0:smax
            k = r + s
            weight = weighted && k > 0 ? k / (k + r) : 1.0
            @inbounds for j in 0:J_DEG
                Qcols[s + 1, j + 1, k + 1] = weight * Hjs[j + 1, s + 1]
            end
        end
        if r < N - 1
            Br *= (r + 1.0) / (r + mu64 + 1.0)
        end
    end
    return Qcols
end

function assemble_A_narrow_fft_tensor_absorbM(
    C::Matrix{Float64},
    mu::Float64,
    N::Int,
    upper_bw::Int,
    S::Int,
    Sguard::Int;
    h0_init::Symbol=:s_recurrence,
    prec_bits::Int=1536,
    return_stats::Bool=false,
)
    HAS_FFTW[] || error("FFTW.jl is required for assembly=:narrow_fft_tensor_absorbM")
    P = build_postmult_coeffs_frac(C)
    IP_DEG = size(P, 1) - 1
    lower_bw_A = IP_DEG
    S_A = upper_bw + IP_DEG
    Sguard_A = S_A + J_DEG + 2
    Lfft = nextpow(2, IP_DEG + S_A + 2)
    Aband = zeros(Float64, lower_bw_A + upper_bw + 1, N)

    local Htensor
    time_tensor_generation = @elapsed begin
        Htensor = generate_weak_narrow_tensor(
            mu,
            N,
            S_A,
            Sguard_A;
            h0_init,
            prec_bits,
            weighted=true,
        )
    end

    pblocks = build_P_blocks_frac(P)
    Fpack = precompute_packed_spectra_frac(pblocks, IP_DEG, Lfft)
    buf = zeros(ComplexF64, Lfft)
    PackedHat = zeros(ComplexF64, Lfft)

    time_fft_assembly = @elapsed begin
        # k=0 has no restriction scaling or k/(k+r) weight.
        Bvals = beta_r_values(mu, max(N, 1))
        add_k0_column_absorbM!(Aband, P, mu, S_A, Sguard_A, Bvals; h0_init, prec_bits)

        for k in 1:(N - 1)
            fill!(PackedHat, 0.0 + 0.0im)
            for j in 0:J_DEG
                fill!(buf, 0.0 + 0.0im)
                rlo = max(0, k - S_A)
                for r in rlo:k
                    s = k - r
                    # The convolution index is s.  The tensor column is r.
                    buf[s + 1] = Htensor[j + 1][s + 1, r + 1]
                end
                FFTW.fft!(buf)
                @inbounds for p in 1:Lfft
                    PackedHat[p] += Fpack[p, j + 1] * buf[p]
                end
            end
            FFTW.ifft!(PackedHat)
            @inbounds for d in -lower_bw_A:upper_bw
                value = 0.0
                d >= 0 && (value += real(PackedHat[d + 1]))
                idx = IP_DEG + d
                0 <= idx < Lfft && (value += imag(PackedHat[idx + 1]))
                add_A_by_d!(Aband, k, d, value, lower_bw_A, upper_bw)
            end
        end

        # Third branch ell=i-r.  Htensor is already weighted by k/(k+r).
        for r in 1:min(IP_DEG, N - 1)
            smax = min(S_A, N - 1 - r)
            for s in 0:smax
                k = r + s
                for i in r:IP_DEG
                    d = 2k - s - i
                    -lower_bw_A <= d <= upper_bw || continue
                    h = 0.0
                    @inbounds for j in 0:J_DEG
                        h += P[i + 1, j + 1] * Htensor[j + 1][s + 1, r + 1]
                    end
                    add_A_by_d!(Aband, k, d, h, lower_bw_A, upper_bw)
                end
            end
        end
    end

    memory_estimate = sum(sizeof, Htensor) + sizeof(Aband)
    stats = (;
        time_tensor_generation,
        time_fft_assembly,
        memory_estimate,
        tensor_size=(S_A + 1, N, J_DEG + 1),
    )
    return return_stats ? (Aband, stats) : Aband
end

function assemble_A_narrow_fft_tensor_accum_absorbM(
    C::Matrix{Float64},
    mu::Float64,
    N::Int,
    upper_bw::Int,
    S::Int,
    Sguard::Int;
    h0_init::Symbol=:s_recurrence,
    prec_bits::Int=1536,
    return_stats::Bool=false,
)
    HAS_FFTW[] || error("FFTW.jl is required for assembly=:narrow_fft_tensor_accum_absorbM")
    P = build_postmult_coeffs_frac(C)
    IP_DEG = size(P, 1) - 1
    lower_bw_A = IP_DEG
    S_A = upper_bw + IP_DEG
    Sguard_A = S_A + J_DEG + 2
    Lfft = nextpow(2, IP_DEG + S_A + 2)
    Aband = zeros(Float64, lower_bw_A + upper_bw + 1, N)

    local Qcols
    time_tensor_generation = @elapsed begin
        Qcols = generate_weak_column_tensor(
            mu,
            N,
            S_A,
            Sguard_A;
            h0_init,
            prec_bits,
            weighted=true,
        )
    end

    local Fpack
    time_packed_spectra = @elapsed begin
        pblocks = build_P_blocks_frac(P)
        Fpack = precompute_packed_spectra_frac(pblocks, IP_DEG, Lfft)
    end

    Qmat = zeros(ComplexF64, Lfft, J_DEG + 1)
    Facc = zeros(ComplexF64, Lfft)
    plan_q_fwd = FFTW.plan_fft!(Qmat, 1; flags=FFTW.ESTIMATE)
    plan_acc_inv = FFTW.plan_ifft!(Facc; flags=FFTW.ESTIMATE)

    time_fill_Qmat = 0.0
    time_forward_fft = 0.0
    time_freq_accum = 0.0
    time_inverse_fft = 0.0
    time_write_A = 0.0
    time_left_corner = 0.0

    Bvals = beta_r_values(mu, max(N, 1))
    add_k0_column_absorbM!(Aband, P, mu, S_A, Sguard_A, Bvals; h0_init, prec_bits)

    for k in 1:(N - 1)
        @stat_elapsed return_stats time_fill_Qmat begin
            fill!(Qmat, 0.0 + 0.0im)
            smax = min(S_A, k)
            @inbounds for j in 0:J_DEG
                for s in 0:smax
                    Qmat[s + 1, j + 1] = Qcols[s + 1, j + 1, k + 1]
                end
            end
        end

        @stat_elapsed return_stats time_forward_fft begin
            plan_q_fwd * Qmat
        end

        @stat_elapsed return_stats time_freq_accum begin
            fill!(Facc, 0.0 + 0.0im)
            @inbounds for j in 1:(J_DEG + 1)
                for p in 1:Lfft
                    Facc[p] += Fpack[p, j] * Qmat[p, j]
                end
            end
        end

        @stat_elapsed return_stats time_inverse_fft begin
            plan_acc_inv * Facc
        end

        @stat_elapsed return_stats time_write_A begin
            @inbounds for d in -lower_bw_A:upper_bw
                value = 0.0
                d >= 0 && (value += real(Facc[d + 1]))
                idx = IP_DEG + d
                0 <= idx < Lfft && (value += imag(Facc[idx + 1]))
                add_A_by_d!(Aband, k, d, value, lower_bw_A, upper_bw)
            end
        end
    end

    time_left_corner = @elapsed begin
        add_left_corner_correction_absorbM_fast!(
            Aband,
            P,
            mu,
            N,
            upper_bw,
            S_A,
            Sguard_A,
            Bvals;
            h0_init,
            prec_bits,
        )
    end

    memory_estimate = sizeof(Qcols) + sizeof(Aband)
    stats = (;
        time_tensor_generation,
        time_packed_spectra,
        time_fill_Qmat,
        time_forward_fft,
        time_freq_accum,
        time_inverse_fft,
        time_write_A,
        time_left_corner,
        memory_estimate,
        tensor_layout=:column_ready,
        tensor_size=size(Qcols),
    )
    return return_stats ? (Aband, stats) : Aband
end

function assemble_A_narrow_fft_tensor_accum_blocked_absorbM(
    C::Matrix{Float64},
    mu::Float64,
    N::Int,
    upper_bw::Int,
    S::Int,
    Sguard::Int;
    h0_init::Symbol=:s_recurrence,
    prec_bits::Int=1536,
    block_cols::Int=128,
    return_stats::Bool=false,
)
    HAS_FFTW[] || error("FFTW.jl is required for assembly=:narrow_fft_tensor_accum_blocked_absorbM")
    block_cols >= 1 || throw(ArgumentError("block_cols must be positive"))
    P = build_postmult_coeffs_frac(C)
    IP_DEG = size(P, 1) - 1
    lower_bw_A = IP_DEG
    S_A = upper_bw + IP_DEG
    Sguard_A = S_A + J_DEG + 2
    Lfft = nextpow(2, IP_DEG + S_A + 2)
    Aband = zeros(Float64, lower_bw_A + upper_bw + 1, N)

    local Qcols
    time_tensor_generation = @elapsed begin
        Qcols = generate_weak_column_tensor(mu, N, S_A, Sguard_A; h0_init, prec_bits, weighted=true)
    end

    local Fpack
    time_packed_spectra = @elapsed begin
        pblocks = build_P_blocks_frac(P)
        Fpack = precompute_packed_spectra_frac(pblocks, IP_DEG, Lfft)
    end

    max_nb = min(block_cols, max(1, N - 1))
    Qbuf = zeros(ComplexF64, Lfft, J_DEG + 1, max_nb)
    Facc = zeros(ComplexF64, Lfft, max_nb)
    plan_q_fwd = FFTW.plan_fft!(Qbuf, 1; flags=FFTW.ESTIMATE)
    plan_acc_inv = FFTW.plan_ifft!(Facc, 1; flags=FFTW.ESTIMATE)

    time_fill_Qmat = 0.0
    time_forward_fft = 0.0
    time_freq_accum = 0.0
    time_inverse_fft = 0.0
    time_write_A = 0.0
    time_left_corner = 0.0

    Bvals = beta_r_values(mu, max(N, 1))
    add_k0_column_absorbM!(Aband, P, mu, S_A, Sguard_A, Bvals; h0_init, prec_bits)

    for k0 in 1:block_cols:(N - 1)
        k1 = min(N - 1, k0 + block_cols - 1)
        nb = k1 - k0 + 1
        Qview = @view Qbuf[:, :, 1:nb]
        Fview = @view Facc[:, 1:nb]

        @stat_elapsed return_stats time_fill_Qmat begin
            fill!(Qview, 0.0 + 0.0im)
            @inbounds for b in 1:nb
                k = k0 + b - 1
                smax = min(S_A, k)
                for j in 0:J_DEG
                    for s in 0:smax
                        Qbuf[s + 1, j + 1, b] = Qcols[s + 1, j + 1, k + 1]
                    end
                end
            end
        end

        @stat_elapsed return_stats time_forward_fft begin
            plan_q_fwd * Qbuf
        end

        @stat_elapsed return_stats time_freq_accum begin
            fill!(Fview, 0.0 + 0.0im)
            @inbounds for b in 1:nb
                for j in 1:(J_DEG + 1)
                    for p in 1:Lfft
                        Facc[p, b] += Fpack[p, j] * Qbuf[p, j, b]
                    end
                end
            end
        end

        @stat_elapsed return_stats time_inverse_fft begin
            plan_acc_inv * Facc
        end

        @stat_elapsed return_stats time_write_A begin
            @inbounds for b in 1:nb
                k = k0 + b - 1
                for d in -lower_bw_A:upper_bw
                    value = 0.0
                    d >= 0 && (value += real(Facc[d + 1, b]))
                    idx = IP_DEG + d
                    0 <= idx < Lfft && (value += imag(Facc[idx + 1, b]))
                    add_A_by_d!(Aband, k, d, value, lower_bw_A, upper_bw)
                end
            end
        end
    end

    time_left_corner = @elapsed begin
        add_left_corner_correction_absorbM_fast!(
            Aband,
            P,
            mu,
            N,
            upper_bw,
            S_A,
            Sguard_A,
            Bvals;
            h0_init,
            prec_bits,
        )
    end

    memory_estimate = sizeof(Qcols) + sizeof(Aband) + sizeof(Qbuf) + sizeof(Facc)
    stats = (;
        time_tensor_generation,
        time_packed_spectra,
        time_fill_Qmat,
        time_forward_fft,
        time_freq_accum,
        time_inverse_fft,
        time_write_A,
        time_left_corner,
        memory_estimate,
        tensor_layout=:column_ready,
        tensor_size=size(Qcols),
    )
    return return_stats ? (Aband, stats) : Aband
end

function assemble_A_narrow_direct_absorbM(
    C::Matrix{Float64},
    mu::Float64,
    N::Int,
    upper_bw::Int,
    S::Int,
    Sguard::Int;
    h0_init::Symbol=:s_recurrence,
    prec_bits::Int=1536,
)
    P = build_postmult_coeffs_frac(C)
    IP_DEG = size(P, 1) - 1
    lower_bw_A = IP_DEG
    S_A = upper_bw + IP_DEG
    Sguard_A = S_A + J_DEG + 2
    Aband = zeros(Float64, lower_bw_A + upper_bw + 1, N)

    Br = 1.0 / mu
    for r in 0:(N - 1)
        Hjs = build_Hjs_line(r, mu, S_A, Sguard_A, Br; h0_init, prec_bits)
        smax = min(S_A, N - 1 - r)
        HcompP = P * @view(Hjs[:, 1:(smax + 1)])
        for s in 0:smax
            k = r + s
            if k == 0
                for i in 0:min(IP_DEG, N - 1)
                    d = -i
                    Aband[d + lower_bw_A + 1, 1] += HcompP[i + 1, 1]
                end
                continue
            end

            weighted_col_scale = k / (k + r)
            for i in 0:IP_DEG
                h = weighted_col_scale * HcompP[i + 1, s + 1]
                h == 0.0 && continue

                d1 = s + i
                if d1 <= upper_bw
                    add_A_by_d!(Aband, k, d1, h, lower_bw_A, upper_bw)
                end

                d2 = s - i
                if -lower_bw_A <= d2 <= upper_bw
                    add_A_by_d!(Aband, k, d2, h, lower_bw_A, upper_bw)
                end

                if i >= r && r != 0
                    d3 = 2r + s - i
                    if -lower_bw_A <= d3 <= upper_bw
                        add_A_by_d!(Aband, k, d3, h, lower_bw_A, upper_bw)
                    end
                end
            end
        end

        if r < N - 1
            Br *= (r + 1.0) / (r + mu + 1.0)
        end
    end
    return Aband
end

function assemble_A_narrow_rfft_tensor_accum_absorbM(
    C::Matrix{Float64},
    mu::Float64,
    N::Int,
    upper_bw::Int,
    S::Int,
    Sguard::Int;
    h0_init::Symbol=:s_recurrence,
    prec_bits::Int=1536,
    return_stats::Bool=false,
)
    P = build_postmult_coeffs_frac(C)
    IP_DEG = size(P, 1) - 1
    lower_bw_A = IP_DEG
    S_A = upper_bw + IP_DEG
    Sguard_A = S_A + J_DEG + 2
    Lfft = nextpow(2, IP_DEG + S_A + 2)
    Lfreq = div(Lfft, 2) + 1
    Aband = zeros(Float64, lower_bw_A + upper_bw + 1, N)

    local Qcols
    time_tensor_generation = @elapsed begin
        Qcols = generate_weak_column_tensor(mu, N, S_A, Sguard_A; h0_init, prec_bits, weighted=true)
    end

    local Fp, Fr
    time_packed_spectra = @elapsed begin
        Fp = Matrix{ComplexF64}(undef, Lfreq, J_DEG + 1)
        Fr = Matrix{ComplexF64}(undef, Lfreq, J_DEG + 1)
        tmp = zeros(Float64, Lfft)
        for j in 0:J_DEG
            fill!(tmp, 0.0)
            tmp[1:(IP_DEG + 1)] .= @view P[:, j + 1]
            Fp[:, j + 1] .= FFTW.rfft(tmp)
            fill!(tmp, 0.0)
            tmp[1:(IP_DEG + 1)] .= reverse(@view P[:, j + 1])
            Fr[:, j + 1] .= FFTW.rfft(tmp)
        end
    end

    Qmat = zeros(Float64, Lfft, J_DEG + 1)
    Uhat = zeros(ComplexF64, Lfreq)
    Lhat = zeros(ComplexF64, Lfreq)
    time_fill_Qmat = 0.0
    time_forward_fft = 0.0
    time_freq_accum = 0.0
    time_inverse_fft = 0.0
    time_write_A = 0.0
    time_left_corner = 0.0

    Bvals = beta_r_values(mu, max(N, 1))
    add_k0_column_absorbM!(Aband, P, mu, S_A, Sguard_A, Bvals; h0_init, prec_bits)

    for k in 1:(N - 1)
        @stat_elapsed return_stats time_fill_Qmat begin
            fill!(Qmat, 0.0)
            smax = min(S_A, k)
            @inbounds for j in 0:J_DEG
                for s in 0:smax
                    Qmat[s + 1, j + 1] = Qcols[s + 1, j + 1, k + 1]
                end
            end
        end

        local Qhat
        @stat_elapsed return_stats time_forward_fft begin
            Qhat = FFTW.rfft(Qmat, 1)
        end

        @stat_elapsed return_stats time_freq_accum begin
            fill!(Uhat, 0.0 + 0.0im)
            fill!(Lhat, 0.0 + 0.0im)
            @inbounds for j in 1:(J_DEG + 1)
                for p in 1:Lfreq
                    q = Qhat[p, j]
                    Uhat[p] += Fp[p, j] * q
                    Lhat[p] += Fr[p, j] * q
                end
            end
        end

        local U, L
        @stat_elapsed return_stats time_inverse_fft begin
            U = FFTW.irfft(Uhat, Lfft)
            L = FFTW.irfft(Lhat, Lfft)
        end

        @stat_elapsed return_stats time_write_A begin
            @inbounds for d in -lower_bw_A:upper_bw
                value = 0.0
                d >= 0 && (value += U[d + 1])
                idx = IP_DEG + d
                0 <= idx < Lfft && (value += L[idx + 1])
                add_A_by_d!(Aband, k, d, value, lower_bw_A, upper_bw)
            end
        end
    end

    time_left_corner = @elapsed begin
        add_left_corner_correction_absorbM_fast!(
            Aband,
            P,
            mu,
            N,
            upper_bw,
            S_A,
            Sguard_A,
            Bvals;
            h0_init,
            prec_bits,
        )
    end

    memory_estimate = sizeof(Qcols) + sizeof(Aband) + sizeof(Qmat)
    stats = (;
        time_tensor_generation,
        time_packed_spectra,
        time_fill_Qmat,
        time_forward_fft,
        time_freq_accum,
        time_inverse_fft,
        time_write_A,
        time_left_corner,
        memory_estimate,
        tensor_layout=:column_ready,
        tensor_size=size(Qcols),
    )
    return return_stats ? (Aband, stats) : Aband
end

function assemble_A_narrow_rfft_stream_accum_absorbM(
    C::Matrix{Float64},
    mu::Float64,
    N::Int,
    upper_bw::Int,
    S::Int,
    Sguard::Int;
    h0_init::Symbol=:s_recurrence,
    prec_bits::Int=1536,
    return_stats::Bool=false,
)
    P = build_postmult_coeffs_frac(C)
    IP_DEG = size(P, 1) - 1
    lower_bw_A = IP_DEG
    S_A = upper_bw + IP_DEG
    Sguard_A = S_A + J_DEG + 2
    Lfft = nextpow(2, IP_DEG + S_A + 2)
    Lfreq = div(Lfft, 2) + 1
    Aband = zeros(Float64, lower_bw_A + upper_bw + 1, N)

    local Fp, Fr
    time_packed_spectra = @elapsed begin
        Fp = Matrix{ComplexF64}(undef, Lfreq, J_DEG + 1)
        Fr = Matrix{ComplexF64}(undef, Lfreq, J_DEG + 1)
        tmp = zeros(Float64, Lfft)
        for j in 0:J_DEG
            fill!(tmp, 0.0)
            tmp[1:(IP_DEG + 1)] .= @view P[:, j + 1]
            Fp[:, j + 1] .= FFTW.rfft(tmp)
            fill!(tmp, 0.0)
            tmp[1:(IP_DEG + 1)] .= reverse(@view P[:, j + 1])
            Fr[:, j + 1] .= FFTW.rfft(tmp)
        end
    end

    Rcache = S_A + 1
    Hcache = zeros(Float64, J_DEG + 1, S_A + 1, Rcache)
    Hvalid_r = fill(-1, Rcache)
    Bvals = beta_r_values(mu, N)
    time_H_generation_stream = 0.0

    function ensure_Hjs_for_r!(r::Int)
        idx = mod(r, Rcache) + 1
        if Hvalid_r[idx] != r
            @stat_elapsed return_stats time_H_generation_stream begin
                Hjs = build_Hjs_line(r, mu, S_A, Sguard_A, Bvals[r + 1]; h0_init, prec_bits)
                @views Hcache[:, :, idx] .= Hjs
                Hvalid_r[idx] = r
            end
        end
        return idx
    end

    Qmat = zeros(Float64, Lfft, J_DEG + 1)
    Uhat = zeros(ComplexF64, Lfreq)
    Lhat = zeros(ComplexF64, Lfreq)
    U = zeros(Float64, Lfft)
    L = zeros(Float64, Lfft)
    plan_q_fwd = FFTW.plan_rfft(Qmat, 1; flags=FFTW.ESTIMATE)

    time_fill_Qmat = 0.0
    time_forward_fft = 0.0
    time_freq_accum = 0.0
    time_inverse_fft = 0.0
    time_write_A = 0.0
    time_left_corner = 0.0

    idx0 = ensure_Hjs_for_r!(0)
    Hcomp0 = P * @view(Hcache[:, 1:1, idx0])
    for i in 0:min(IP_DEG, N - 1)
        d = -i
        Aband[d + lower_bw_A + 1, 1] += Hcomp0[i + 1, 1]
    end

    for k in 1:(N - 1)
        @stat_elapsed return_stats time_fill_Qmat begin
            fill!(Qmat, 0.0)
            smax = min(S_A, k)
            for s in 0:smax
                r = k - s
                idx = ensure_Hjs_for_r!(r)
                weight = k / (k + r)
                @inbounds for j in 0:J_DEG
                    Qmat[s + 1, j + 1] = weight * Hcache[j + 1, s + 1, idx]
                end
            end
        end

        local Qhat
        @stat_elapsed return_stats time_forward_fft begin
            Qhat = plan_q_fwd * Qmat
        end

        @stat_elapsed return_stats time_freq_accum begin
            fill!(Uhat, 0.0 + 0.0im)
            fill!(Lhat, 0.0 + 0.0im)
            @inbounds for j in 1:(J_DEG + 1)
                for p in 1:Lfreq
                    q = Qhat[p, j]
                    Uhat[p] += Fp[p, j] * q
                    Lhat[p] += Fr[p, j] * q
                end
            end
        end

        @stat_elapsed return_stats time_inverse_fft begin
            U .= FFTW.irfft(Uhat, Lfft)
            L .= FFTW.irfft(Lhat, Lfft)
        end

        @stat_elapsed return_stats time_write_A begin
            @inbounds for d in -lower_bw_A:upper_bw
                value = 0.0
                d >= 0 && (value += U[d + 1])
                conv_idx = IP_DEG + d
                0 <= conv_idx < Lfft && (value += L[conv_idx + 1])
                add_A_by_d!(Aband, k, d, value, lower_bw_A, upper_bw)
            end
        end
    end

    time_left_corner = @elapsed begin
        for r in 1:min(IP_DEG, N - 1)
            idx = ensure_Hjs_for_r!(r)
            smax = min(S_A, N - 1 - r)
            HcompP = P * @view(Hcache[:, 1:(smax + 1), idx])
            for s in 0:smax
                k = r + s
                weight = k / (k + r)
                for i in r:IP_DEG
                    d = 2r + s - i
                    -lower_bw_A <= d <= upper_bw || continue
                    add_A_by_d!(Aband, k, d, weight * HcompP[i + 1, s + 1], lower_bw_A, upper_bw)
                end
            end
        end
    end

    memory_estimate =
        sizeof(Hcache) + sizeof(Qmat) + sizeof(Uhat) + sizeof(Lhat) +
        sizeof(U) + sizeof(L) + sizeof(Aband)
    stats = (;
        time_H_generation_stream,
        time_packed_spectra,
        time_fill_Qmat,
        time_forward_fft,
        time_freq_accum,
        time_inverse_fft,
        time_write_A,
        time_left_corner,
        memory_estimate,
        tensor_layout=:streaming_ring_buffer,
        cache_size=size(Hcache),
    )
    return return_stats ? (Aband, stats) : Aband
end

function assemble_G(
    C::Matrix{Float64},
    mu::Float64,
    N::Int,
    upper_bw::Int,
    assembly::Symbol;
    Hlayers=nothing,
    Hseed=nothing,
    h0_init::Symbol=:s_recurrence,
    prec_bits::Int=1536,
    block_cols::Int=64,
    fftw_flags=nothing,
)
    if assembly === :direct_scatter
        return assemble_G_direct_verified(C, mu, N, upper_bw; Hlayers)
    elseif assembly === :narrow_direct_banded
        upper_bw_G = upper_bw + 1
        S = upper_bw_G + I_DEG + 1
        Sguard = S + J_DEG + 2
        return assemble_G_direct_banded(C, mu, N, upper_bw, S, Sguard; h0_init, prec_bits)
    elseif assembly === :narrow_fft_batched
        upper_bw_G = upper_bw + 1
        S = upper_bw_G + I_DEG + 1
        Sguard = S + J_DEG + 2
        return assemble_G_narrow_fft_batched(C, mu, N, upper_bw, S, Sguard; h0_init, prec_bits, block_cols)
    elseif assembly === :direct_convolution
        return assemble_G_direct_convolution(C, mu, N, upper_bw; Hlayers, Hseed)
    elseif assembly === :fft_convolution
        return assemble_G_fft_convolution(C, mu, N, upper_bw; Hlayers, Hseed)
    else
        throw(ArgumentError("assembly must be :fft_convolution, :direct_convolution, :direct_scatter, :narrow_direct_banded, or :narrow_fft_batched"))
    end
end

function multiply_by_one_plus_y_over_two_banded(Gband::AbstractMatrix{<:Real}, N::Int, upper_bw::Int)
    lower_bw_G = I_DEG
    upper_bw_G = upper_bw + 1
    lower_bw_A = I_DEG + 1
    Aband = zeros(Float64, lower_bw_A + upper_bw + 1, N)
    for k0 in 0:(N - 1)
        for d in -lower_bw_A:upper_bw
            ell0 = k0 - d
            0 <= ell0 <= N - 1 || continue
            value = 0.5 * band_get(Gband, ell0, k0, lower_bw_G, upper_bw_G)
            if ell0 == 0
                value += 0.25 * band_get(Gband, 1, k0, lower_bw_G, upper_bw_G)
            elseif ell0 == 1
                value += 0.5 * band_get(Gband, 0, k0, lower_bw_G, upper_bw_G)
                value += 0.25 * band_get(Gband, 2, k0, lower_bw_G, upper_bw_G)
            else
                value += 0.25 * band_get(Gband, ell0 - 1, k0, lower_bw_G, upper_bw_G)
                value += 0.25 * band_get(Gband, ell0 + 1, k0, lower_bw_G, upper_bw_G)
            end
            Aband[d + lower_bw_A + 1, k0 + 1] = value
        end
    end
    return Aband
end

function kept_band_mask(N::Int, upper_bw::Int)
    lower_bw = I_DEG + 1
    mask = falses(N, N)
    for k in 0:(N - 1), ell in 0:(N - 1)
        d = k - ell
        mask[ell + 1, k + 1] = -lower_bw <= d <= upper_bw
    end
    return mask
end

function build_C(mu::Float64, omega::Float64)
    C, infoC, _ = Weak_VoltMat_DE.build_Kphi_cheb_coeffs(
        (x, t) -> 1.0,
        x -> Weak_VoltMat_DE.paper_de_phi(x; omega),
        x -> Weak_VoltMat_DE.paper_de_dphi(x; omega);
        mu,
        stable_de_ktilde=true,
        de_omega=omega,
        Csize_y,
        Csize_eta,
    )
    return C, infoC
end

function build_postmult_coeffs_frac(C::Matrix{Float64})
    size(C, 1) == Csize_y || throw(ArgumentError("C must have $Csize_y rows"))
    size(C, 2) == Csize_eta || throw(ArgumentError("C must have $Csize_eta columns"))
    P = zeros(Float64, I_DEG + 2, J_DEG + 1)
    for j in 1:(J_DEG + 1)
        for i0 in 0:I_DEG
            cij = C[i0 + 1, j]
            P[i0 + 1, j] += 0.5 * cij
            if i0 >= 1
                P[i0, j] += 0.25 * cij
            else
                P[2, j] += 0.25 * cij
            end
            P[i0 + 2, j] += 0.25 * cij
        end
    end
    return P
end

function build_P_blocks_frac(P::Matrix{Float64})
    blocks = Vector{PBlockFrac}(undef, size(P, 2))
    for j in axes(P, 2)
        dense = copy(@view P[:, j])
        ii = Int[]
        vv = Float64[]
        for i0 in 0:(length(dense) - 1)
            v = dense[i0 + 1]
            if v != 0.0
                push!(ii, i0)
                push!(vv, v)
            end
        end
        blocks[j] = PBlockFrac(ii, vv, dense)
    end
    return blocks
end

function precompute_packed_spectra_frac(pblocks, IP_DEG::Int, Lfft::Int, plan_fwd=nothing)
    Fpack = Matrix{ComplexF64}(undef, Lfft, length(pblocks))
    tmp = zeros(ComplexF64, Lfft)
    for j in eachindex(pblocks)
        fill!(tmp, 0)
        dense = pblocks[j].dense
        tmp[1:length(dense)] .= dense
        Fp = plan_fwd === nothing ? FFTW.fft(tmp) : plan_fwd * tmp
        fill!(tmp, 0)
        tmp[1:length(dense)] .= reverse(dense)
        Fr = plan_fwd === nothing ? FFTW.fft(tmp) : plan_fwd * tmp
        Fpack[:, j] .= Fp .+ im .* Fr
    end
    return Fpack
end

function ensure_Hline!(ring::HLineRingCache, r::Int, mu::Float64, h0_init::Symbol, prec_bits::Int)
    idx = mod(r, ring.Rcache) + 1
    if ring.valid_r[idx] != r
        Hview = @view ring.Hcache[:, :, idx]
        build_Hjs_line!(Hview, r, mu, ring.S, ring.Sguard, ring.Bvals[r + 1]; h0_init, prec_bits, layout=:sj)
        ring.valid_r[idx] = r
    end
    return idx
end

function mul_P_Hsj!(
    Hcomp::AbstractMatrix{Float64},
    P::Matrix{Float64},
    Hsj::AbstractMatrix{Float64},
    smax::Int,
)
    @inbounds for s in 0:smax
        for i in 1:size(P, 1)
            acc = 0.0
            @simd for j in 1:size(P, 2)
                acc += P[i, j] * Hsj[s + 1, j]
            end
            Hcomp[i, s + 1] = acc
        end
    end
    return nothing
end

function build_left_corner_cache_frac!(
    left_p::Vector{Int},
    left_col::Vector{Int},
    left_value::Vector{Float64},
    plan,
)
    empty!(left_p)
    empty!(left_col)
    empty!(left_value)
    IP_DEG = plan.lower_bw_A
    for r in 1:min(IP_DEG, plan.N - 1)
        idx = ensure_Hline!(plan.Hring, r, plan.mu, plan.h0_init, plan.prec_bits)
        smax = min(plan.S_A, plan.N - 1 - r)
        Hview = @view plan.Hring.Hcache[:, :, idx]
        Hcomp = @view plan.HcompP[:, 1:(smax + 1)]
        mul_P_Hsj!(Hcomp, plan.P, Hview, smax)
        @inbounds for s in 0:smax
            k = r + s
            weight = k / (k + r)
            for i in r:IP_DEG
                d = 2r + s - i
                -IP_DEG <= d <= plan.upper_bw || continue
                ell = k - d
                0 <= ell <= plan.N - 1 || continue
                scale = (2.0 - (ell == 0 ? 1.0 : 0.0)) / 2.0
                push!(left_p, d + IP_DEG + 1)
                push!(left_col, k + 1)
                push!(left_value, scale * weight * Hcomp[i + 1, s + 1])
            end
        end
    end
    return nothing
end

function precompute_rfft_spectra_frac!(
    Fp::Matrix{ComplexF64},
    Fr::Matrix{ComplexF64},
    P::Matrix{Float64},
    tmp::Vector{Float64},
    plan_tmp_rfft,
)
    IP_DEG = size(P, 1) - 1
    Ftmp = Vector{ComplexF64}(undef, size(Fp, 1))
    @inbounds for j in 1:size(P, 2)
        fill!(tmp, 0.0)
        for i in 1:(IP_DEG + 1)
            tmp[i] = P[i, j]
        end
        mul!(Ftmp, plan_tmp_rfft, tmp)
        Fp[:, j] .= Ftmp

        fill!(tmp, 0.0)
        for i in 0:IP_DEG
            tmp[i + 1] = P[IP_DEG - i + 1, j]
        end
        mul!(Ftmp, plan_tmp_rfft, tmp)
        Fr[:, j] .= Ftmp
    end
    return nothing
end

function plan_fracint_K1_fft(;
    mu,
    N::Int,
    omega::Real=4.0,
    upper_bw::Int=120,
    assembly::Symbol=:narrow_rfft_tensor_accum_blocked_absorbM,
    block_cols::Int=64,
    h0_init::Symbol=:s_recurrence,
    prec_bits::Int=1536,
    fftw_flags=nothing,
    C=nothing,
    infoC=nothing,
)
    assembly === :narrow_rfft_tensor_accum_blocked_absorbM ||
        throw(ArgumentError("plan_fracint_K1_fft currently supports :narrow_rfft_tensor_accum_blocked_absorbM"))
    block_cols >= 1 || throw(ArgumentError("block_cols must be positive"))
    flags = resolve_fftw_flags(fftw_flags)
    mu64 = Float64(mu)
    omega64 = Float64(omega)
    Cmat, infoC_local = C === nothing ? build_C(mu64, omega64) : (C, infoC)
    P = build_postmult_coeffs_frac(Cmat)
    IP_DEG = size(P, 1) - 1
    lower_bw_A = IP_DEG
    S_A = upper_bw + IP_DEG
    Sguard_A = S_A + J_DEG + 2
    Lfft = nextpow(2, IP_DEG + S_A + 2)
    Lfreq = div(Lfft, 2) + 1
    max_nb = min(block_cols, max(1, N - 1))

    Fp = Matrix{ComplexF64}(undef, Lfreq, J_DEG + 1)
    Fr = Matrix{ComplexF64}(undef, Lfreq, J_DEG + 1)
    tmp = zeros(Float64, Lfft)
    plan_tmp_rfft = FFTW.plan_rfft(tmp; flags)
    precompute_rfft_spectra_frac!(Fp, Fr, P, tmp, plan_tmp_rfft)

    Qbuf = zeros(Float64, Lfft, J_DEG + 1, max_nb)
    Qhat = zeros(ComplexF64, Lfreq, J_DEG + 1, max_nb)
    Uhat = zeros(ComplexF64, Lfreq, max_nb)
    Lhat = zeros(ComplexF64, Lfreq, max_nb)
    Ubuf = zeros(Float64, Lfft, max_nb)
    Lbuf = zeros(Float64, Lfft, max_nb)
    q_highwater = fill(-1, J_DEG + 1, max_nb)
    Rcache = S_A + max_nb + 1
    Hring = HLineRingCache(
        zeros(Float64, S_A + 1, J_DEG + 1, Rcache),
        fill(-1, Rcache),
        beta_r_values(mu64, N),
        S_A,
        Sguard_A,
        Rcache,
    )
    HcompP = zeros(Float64, IP_DEG + 1, S_A + 1)
    scratch_ridx = zeros(Int, S_A + 1, max_nb)
    scratch_weight = zeros(Float64, S_A + 1, max_nb)
    left_p = Int[]
    left_col = Int[]
    left_value = Float64[]

    plan_q_fwd = FFTW.plan_rfft(Qbuf, 1; flags)
    plan_u_inv = FFTW.plan_irfft(Uhat, Lfft, 1; flags)
    plan_l_inv = FFTW.plan_irfft(Lhat, Lfft, 1; flags)
    memory_estimate =
        sizeof(Fp) + sizeof(Fr) + sizeof(Qbuf) + sizeof(Qhat) + sizeof(Uhat) +
        sizeof(Lhat) + sizeof(Ubuf) + sizeof(Lbuf) + sizeof(Hring.Hcache) +
        sizeof(HcompP) + sizeof(scratch_ridx) + sizeof(scratch_weight)

    plan_obj = FracIntK1FFTPlan(
        Cmat, infoC_local, P, mu64, omega64, N, upper_bw, lower_bw_A, S_A, Sguard_A,
        Lfft, Lfreq, block_cols, assembly, h0_init, prec_bits, flags, Fp, Fr,
        Qbuf, Qhat, Uhat, Lhat, Ubuf, Lbuf, q_highwater, Hring, HcompP,
        scratch_ridx, scratch_weight, left_p, left_col, left_value, plan_q_fwd, plan_u_inv, plan_l_inv,
        memory_estimate,
    )
    build_left_corner_cache_frac!(plan_obj.left_p, plan_obj.left_col, plan_obj.left_value, plan_obj)
    plan_obj.memory_estimate +=
        sizeof(plan_obj.left_p) + sizeof(plan_obj.left_col) + sizeof(plan_obj.left_value)
    return plan_obj
end

function add_k0_column_absorbM_cached!(Aband::Matrix{Float64}, plan::FracIntK1FFTPlan)
    idx0 = ensure_Hline!(plan.Hring, 0, plan.mu, plan.h0_init, plan.prec_bits)
    H0 = @view plan.Hring.Hcache[:, :, idx0]
    Hcomp0 = @view plan.HcompP[:, 1:1]
    mul_P_Hsj!(Hcomp0, plan.P, H0, 0)
    @inbounds for i in 0:min(plan.lower_bw_A, plan.N - 1)
        Aband[-i + plan.lower_bw_A + 1, 1] += plan.HcompP[i + 1, 1]
    end
    return nothing
end

function add_left_corner_correction_absorbM_cached!(Aband::Matrix{Float64}, plan::FracIntK1FFTPlan)
    @inbounds for n in eachindex(plan.left_value)
        Aband[plan.left_p[n], plan.left_col[n]] += plan.left_value[n]
    end
    return nothing
end

function assemble_fracint_K1_fft!(Aband::Matrix{Float64}, plan::FracIntK1FFTPlan; return_stats::Bool=false)
    plan.assembly === :narrow_rfft_tensor_accum_blocked_absorbM ||
        throw(ArgumentError("unsupported FFT plan assembly $(plan.assembly)"))
    fill!(Aband, 0.0)
    IP_DEG = plan.lower_bw_A
    max_nb = size(plan.Qbuf, 3)
    time_fill_Qmat = 0.0
    time_forward_fft = 0.0
    time_freq_accum = 0.0
    time_inverse_fft = 0.0
    time_write_A = 0.0
    time_left_corner = 0.0

    add_k0_column_absorbM_cached!(Aband, plan)
    for k0 in 1:plan.block_cols:(plan.N - 1)
        k1 = min(plan.N - 1, k0 + plan.block_cols - 1)
        nb = k1 - k0 + 1

        t0 = _timer_start(return_stats)
        @inbounds for b in 1:nb
            k = k0 + b - 1
            smax = min(plan.S_A, k)
            for j in 1:(J_DEG + 1)
                old = plan.q_highwater[j, b]
                if old > smax
                    for s in (smax + 1):old
                        plan.Qbuf[s + 1, j, b] = 0.0
                    end
                end
                plan.q_highwater[j, b] = smax
            end
        end
        rlo = max(0, k0 - plan.S_A)
        rhi = k1
        @inbounds for r in rlo:rhi
            idx = ensure_Hline!(plan.Hring, r, plan.mu, plan.h0_init, plan.prec_bits)
            s_lo = max(0, k0 - r)
            s_hi = min(plan.S_A, k1 - r, plan.N - 1 - r)
            s_lo <= s_hi || continue
            for s in s_lo:s_hi
                k = r + s
                b = k - k0 + 1
                plan.scratch_ridx[s + 1, b] = idx
                plan.scratch_weight[s + 1, b] = k / (k + r)
            end
        end
        @inbounds for b in 1:nb
            k = k0 + b - 1
            smax = min(plan.S_A, k)
            for j in 1:(J_DEG + 1)
                for s in 0:smax
                    idx = plan.scratch_ridx[s + 1, b]
                    plan.Qbuf[s + 1, j, b] =
                        plan.scratch_weight[s + 1, b] * plan.Hring.Hcache[s + 1, j, idx]
                end
            end
        end
        time_fill_Qmat += _timer_elapsed(return_stats, t0)

        t0 = _timer_start(return_stats)
        mul!(plan.Qhat, plan.plan_q_fwd, plan.Qbuf)
        time_forward_fft += _timer_elapsed(return_stats, t0)

        t0 = _timer_start(return_stats)
        @inbounds for p in 1:plan.Lfreq
            for b in 1:nb
                u = 0.0 + 0.0im
                l = 0.0 + 0.0im
                @simd for j in 1:(J_DEG + 1)
                    q = plan.Qhat[p, j, b]
                    u += plan.Fp[p, j] * q
                    l += plan.Fr[p, j] * q
                end
                plan.Uhat[p, b] = u
                plan.Lhat[p, b] = l
            end
        end
        time_freq_accum += _timer_elapsed(return_stats, t0)

        t0 = _timer_start(return_stats)
        mul!(plan.Ubuf, plan.plan_u_inv, plan.Uhat)
        mul!(plan.Lbuf, plan.plan_l_inv, plan.Lhat)
        time_inverse_fft += _timer_elapsed(return_stats, t0)

        t0 = _timer_start(return_stats)
        @inbounds for b in 1:nb
            k = k0 + b - 1
            for d in -IP_DEG:plan.upper_bw
                value = 0.0
                d >= 0 && (value += plan.Ubuf[d + 1, b])
                conv_idx = IP_DEG + d
                0 <= conv_idx < plan.Lfft && (value += plan.Lbuf[conv_idx + 1, b])
                add_A_by_d!(Aband, k, d, value, IP_DEG, plan.upper_bw)
            end
        end
        time_write_A += _timer_elapsed(return_stats, t0)
    end

    t0 = _timer_start(return_stats)
    add_left_corner_correction_absorbM_cached!(Aband, plan)
    time_left_corner += _timer_elapsed(return_stats, t0)

    stats = (;
        time_fill_Qmat,
        time_forward_fft,
        time_freq_accum,
        time_inverse_fft,
        time_write_A,
        time_left_corner,
        memory_estimate=plan.memory_estimate + sizeof(Aband),
        tensor_layout=:streaming_blocked_rfft,
        cache_size=size(plan.Hring.Hcache),
        block_cols=plan.block_cols,
        fftw_flags=plan.fftw_flags,
    )
    return return_stats ? (Aband, stats) : Aband
end

function assemble_A_narrow_rfft_tensor_accum_blocked_absorbM(
    C::Matrix{Float64},
    mu::Float64,
    N::Int,
    upper_bw::Int,
    S::Int,
    Sguard::Int;
    h0_init::Symbol=:s_recurrence,
    prec_bits::Int=1536,
    block_cols::Int=64,
    return_stats::Bool=false,
    fftw_flags=nothing,
)
    HAS_FFTW[] || error("FFTW.jl is required for assembly=:narrow_rfft_tensor_accum_blocked_absorbM")
    plan = plan_fracint_K1_fft(;
        mu,
        N,
        omega=4.0,
        upper_bw,
        assembly=:narrow_rfft_tensor_accum_blocked_absorbM,
        block_cols,
        h0_init,
        prec_bits,
        fftw_flags,
        C,
    )
    Aband = zeros(Float64, plan.lower_bw_A + upper_bw + 1, N)
    return assemble_fracint_K1_fft!(Aband, plan; return_stats)
end

auto_fracint_K1_assembly(::Int, ::Int) = :narrow_direct_absorbM

function fast_fracint_K1_decheb(;
    mu,
    N::Int,
    omega::Real=4.0,
    upper_bw::Int=120,
    return_info::Bool=false,
    compare_old::Bool=false,
    assembly::Symbol=:narrow_direct_banded,
    check_fft::Bool=false,
    output::Symbol=:banded,
    h0_init::Symbol=:s_recurrence,
    prec_bits::Int=1536,
    block_cols::Int=64,
    fftw_flags=nothing,
)
    0.0 < Float64(mu) < 1.0 || throw(ArgumentError("mu must satisfy 0 < mu < 1"))
    N >= 2 || throw(ArgumentError("N must be at least 2"))
    upper_bw >= 0 || throw(ArgumentError("upper_bw must be nonnegative"))

    mu64 = Float64(mu)
    omega64 = Float64(omega)
    upper_bw_G = upper_bw + 1
    S = upper_bw_G + I_DEG + 1
    Sguard = S + J_DEG + 2
    lower_bw = I_DEG + 1
    assembly_requested = assembly
    assembly_used = assembly === :auto ? auto_fracint_K1_assembly(N, upper_bw) : assembly

    assembly_used === :direct_scatter ||
        assembly_used === :direct_convolution ||
        assembly_used === :fft_convolution ||
        assembly_used === :narrow_direct_banded ||
        assembly_used === :narrow_fft_batched ||
        assembly_used === :narrow_fft_packed_absorbM ||
        assembly_used === :narrow_fft_tensor_absorbM ||
        assembly_used === :narrow_fft_tensor_accum_absorbM ||
        assembly_used === :narrow_fft_tensor_accum_blocked_absorbM ||
        assembly_used === :narrow_rfft_tensor_accum_absorbM ||
        assembly_used === :narrow_rfft_tensor_accum_blocked_absorbM ||
        assembly_used === :narrow_direct_absorbM ||
        assembly_used === :narrow_rfft_stream_accum_absorbM ||
        throw(ArgumentError("assembly must be :auto, :fft_convolution, :direct_convolution, :direct_scatter, :narrow_direct_banded, :narrow_fft_batched, :narrow_fft_packed_absorbM, :narrow_fft_tensor_absorbM, :narrow_fft_tensor_accum_absorbM, :narrow_fft_tensor_accum_blocked_absorbM, :narrow_rfft_tensor_accum_absorbM, :narrow_rfft_tensor_accum_blocked_absorbM, :narrow_rfft_stream_accum_absorbM, or :narrow_direct_absorbM"))
    output === :dense || output === :banded ||
        throw(ArgumentError("output must be :dense or :banded"))
    h0_init === :s_recurrence ||
        h0_init === :hypergeom_float64 ||
        h0_init === :bigfloat_hypergeom ||
        throw(ArgumentError("h0_init must be :s_recurrence, :hypergeom_float64, or :bigfloat_hypergeom"))

    local Aout, Aband, G, C, infoC
    local Hlayers, Hseed
    local fft_branch = nothing
    local tensor_stats = nothing
    time_H_generation = 0.0
    time_assembly = 0.0
    time_final_multiply = 0.0
    time_dense_fill = 0.0
    local fft_vs_direct_err = nothing
    local fft_vs_direct_convolution_err = nothing
    local direct_convolution_vs_scatter_err = nothing
    local G_direct_check = nothing
    build_time_fast = @elapsed begin
        C, infoC = build_C(mu64, omega64)
        time_H_generation = @elapsed begin
            if assembly_used === :narrow_direct_banded || assembly_used === :narrow_fft_batched ||
               assembly_used === :narrow_fft_packed_absorbM || assembly_used === :narrow_fft_tensor_absorbM ||
               assembly_used === :narrow_fft_tensor_accum_absorbM ||
               assembly_used === :narrow_fft_tensor_accum_blocked_absorbM ||
               assembly_used === :narrow_rfft_tensor_accum_absorbM ||
               assembly_used === :narrow_rfft_tensor_accum_blocked_absorbM ||
               assembly_used === :narrow_direct_absorbM ||
               assembly_used === :narrow_rfft_stream_accum_absorbM
                Hlayers = nothing
                Hseed = nothing
            elseif check_fft
                Hlayers = build_verified_H_layers(mu64, N)
                Hseed = nothing
                fft_branch = assembly_used === :fft_convolution ? :prebuilt_Hlayers : nothing
            else
                Hlayers = nothing
                Hseed = build_verified_H_seed(mu64, N)
                fft_branch = assembly_used === :fft_convolution ? :seed_full_H_per_j : nothing
            end
        end

        # FFT/direct convolution assemble the main kept diagonals of G.  The
        # left-corner finite block, including k=0 and the ell=i-r branch, is
        # handled by direct correction with range bounded by I,S,upper_bw.
        absorbs_final_multiplier = assembly_used === :narrow_fft_packed_absorbM ||
            assembly_used === :narrow_fft_tensor_absorbM ||
            assembly_used === :narrow_fft_tensor_accum_absorbM ||
            assembly_used === :narrow_fft_tensor_accum_blocked_absorbM ||
            assembly_used === :narrow_rfft_tensor_accum_absorbM ||
            assembly_used === :narrow_rfft_tensor_accum_blocked_absorbM ||
            assembly_used === :narrow_direct_absorbM ||
            assembly_used === :narrow_rfft_stream_accum_absorbM
        if absorbs_final_multiplier
            time_assembly = @elapsed begin
                if assembly_used === :narrow_direct_absorbM
                    Aband = assemble_A_narrow_direct_absorbM(
                        C,
                        mu64,
                        N,
                        upper_bw,
                        S,
                        Sguard;
                        h0_init,
                        prec_bits,
                    )
                elseif assembly_used === :narrow_fft_tensor_absorbM
                    result = assemble_A_narrow_fft_tensor_absorbM(
                        C,
                        mu64,
                        N,
                        upper_bw,
                        S,
                        Sguard;
                        h0_init,
                        prec_bits,
                        return_stats=return_info,
                    )
                    return_info ? ((Aband, tensor_stats) = result) : (Aband = result)
                elseif assembly_used === :narrow_fft_tensor_accum_absorbM
                    result = assemble_A_narrow_fft_tensor_accum_absorbM(
                        C,
                        mu64,
                        N,
                        upper_bw,
                        S,
                        Sguard;
                        h0_init,
                        prec_bits,
                        return_stats=return_info,
                    )
                    return_info ? ((Aband, tensor_stats) = result) : (Aband = result)
                elseif assembly_used === :narrow_fft_tensor_accum_blocked_absorbM
                    result = assemble_A_narrow_fft_tensor_accum_blocked_absorbM(
                        C,
                        mu64,
                        N,
                        upper_bw,
                        S,
                        Sguard;
                        h0_init,
                        prec_bits,
                        block_cols,
                        return_stats=return_info,
                    )
                    return_info ? ((Aband, tensor_stats) = result) : (Aband = result)
                elseif assembly_used === :narrow_rfft_tensor_accum_absorbM
                    result = assemble_A_narrow_rfft_tensor_accum_absorbM(
                        C,
                        mu64,
                        N,
                        upper_bw,
                        S,
                        Sguard;
                        h0_init,
                        prec_bits,
                        return_stats=return_info,
                    )
                    return_info ? ((Aband, tensor_stats) = result) : (Aband = result)
                elseif assembly_used === :narrow_rfft_tensor_accum_blocked_absorbM
                    plan_fft = plan_fracint_K1_fft(;
                        mu=mu64,
                        N,
                        omega=omega64,
                        upper_bw,
                        assembly=assembly_used,
                        block_cols,
                        h0_init,
                        prec_bits,
                        fftw_flags,
                        C,
                        infoC,
                    )
                    Aband = zeros(Float64, plan_fft.lower_bw_A + upper_bw + 1, N)
                    result = assemble_fracint_K1_fft!(Aband, plan_fft; return_stats=return_info)
                    return_info ? ((Aband, tensor_stats) = result) : (Aband = result)
                elseif assembly_used === :narrow_rfft_stream_accum_absorbM
                    result = assemble_A_narrow_rfft_stream_accum_absorbM(
                        C,
                        mu64,
                        N,
                        upper_bw,
                        S,
                        Sguard;
                        h0_init,
                        prec_bits,
                        return_stats=return_info,
                    )
                    return_info ? ((Aband, tensor_stats) = result) : (Aband = result)
                else
                    Aband = assemble_A_narrow_fft_packed_absorbM(
                        C,
                        mu64,
                        N,
                        upper_bw,
                        S,
                        Sguard;
                        h0_init,
                        prec_bits,
                        block_cols,
                    )
                end
            end
            time_final_multiply = 0.0
        else
            time_assembly = @elapsed begin
                G = assemble_G(C, mu64, N, upper_bw, assembly_used; Hlayers, Hseed, h0_init, prec_bits, block_cols)
            end
            time_final_multiply = @elapsed begin
                Aband = multiply_by_one_plus_y_over_two_banded(G, N, upper_bw)
            end
        end

        if check_fft
            G_direct_conv = assembly_used === :direct_convolution ?
                G :
                assemble_G_direct_convolution(C, mu64, N, upper_bw; Hlayers)
            G_scatter = assembly_used === :direct_scatter ?
                G :
                assemble_G_direct_verified(C, mu64, N, upper_bw; Hlayers)
            if assembly_used === :fft_convolution
                G_fft = G
            else
                G_fft = assemble_G_fft_convolution(C, mu64, N, upper_bw; Hlayers)
            end
            fft_vs_direct_convolution_err = maxabs_array(G_fft .- G_direct_conv)
            direct_convolution_vs_scatter_err = maxabs_array(G_direct_conv .- G_scatter)
            fft_vs_direct_err = max(fft_vs_direct_convolution_err, maxabs_array(G_fft .- G_scatter))
            G_direct_check = G_direct_conv
        end

        time_dense_fill = @elapsed begin
            Aout = output === :dense ? band_to_dense(Aband, N, lower_bw, upper_bw) : Aband
        end
    end

    finite_check = all(isfinite, Aout)
    last_row_max = maximum(abs, @view C[end, :])
    last_col_max = maximum(abs, @view C[:, end])
    mask = kept_band_mask(N, upper_bw)
    nnz_band = count(mask)
    info = Dict{Symbol, Any}(
        :N => N,
        :mu => mu64,
        :omega => omega64,
        :C_size => (Csize_y, Csize_eta),
        :upper_bw => upper_bw,
        :lower_bw => lower_bw,
        :S => S,
        :Sguard => Sguard,
        :build_time_fast => build_time_fast,
        :max_abs_A => maxabs_array(Aout),
        :nnz_band => nnz_band,
        :C_tail_row_col_diagnostics => (;
            last_row_max,
            last_col_max,
            tail_last_row_col_max=max(last_row_max, last_col_max),
            old_tail=getproperty(infoC, :tail),
            sample_error=getproperty(infoC, :sample_error),
        ),
        :finite_check => finite_check,
        :assembly => assembly_used,
        :output => output,
        :h0_init => h0_init,
        :prec_bits => prec_bits,
        :block_cols => block_cols,
        :fftw_flags => fftw_flags,
        :assembly_requested => assembly_requested,
        :assembly_used => assembly_used,
        :check_fft => check_fft,
        :time_H_generation => time_H_generation,
        :time_assembly => time_assembly,
        :time_final_multiply => time_final_multiply,
        :time_dense_fill => time_dense_fill,
        :fft_vs_direct_err => fft_vs_direct_err,
        :fft_vs_direct_convolution_err => fft_vs_direct_convolution_err,
        :direct_convolution_vs_scatter_err => direct_convolution_vs_scatter_err,
        :absorbs_final_multiplier => absorbs_final_multiplier,
        :fft_branch => assembly_used === :narrow_rfft_stream_accum_absorbM ? :rfft_stream_accum_absorbM :
            (assembly_used === :narrow_rfft_tensor_accum_absorbM ? :rfft_tensor_accum_absorbM :
            (assembly_used === :narrow_rfft_tensor_accum_blocked_absorbM ? :rfft_tensor_accum_blocked_absorbM :
            (assembly_used === :narrow_fft_tensor_accum_blocked_absorbM ? :tensor_accum_blocked_absorbM :
            (assembly_used === :narrow_fft_tensor_accum_absorbM ? :tensor_accum_absorbM :
            (assembly_used === :narrow_fft_tensor_absorbM ? :tensor_absorbM :
            (assembly_used === :narrow_fft_packed_absorbM ? :packed_absorbM :
            (assembly_used === :narrow_fft_batched ? :narrow_batched : fft_branch))))))),
        :H_branch => (assembly_used === :narrow_direct_banded || assembly_used === :narrow_fft_batched ||
            assembly_used === :narrow_fft_packed_absorbM || assembly_used === :narrow_fft_tensor_absorbM ||
            assembly_used === :narrow_fft_tensor_accum_absorbM ||
            assembly_used === :narrow_fft_tensor_accum_blocked_absorbM ||
            assembly_used === :narrow_rfft_tensor_accum_absorbM ||
            assembly_used === :narrow_rfft_tensor_accum_blocked_absorbM ||
            assembly_used === :narrow_direct_absorbM ||
            assembly_used === :narrow_rfft_stream_accum_absorbM) ?
            :narrow_fixed_r_jk : :seed_full_H_per_j,
        :IP_DEG => absorbs_final_multiplier ? I_DEG + 1 : missing,
        :P_size => absorbs_final_multiplier ? (I_DEG + 2, J_DEG + 1) : missing,
        :time_tensor_generation => tensor_stats === nothing || !hasproperty(tensor_stats, :time_tensor_generation) ? 0.0 : tensor_stats.time_tensor_generation,
        :time_fft_assembly => tensor_stats === nothing || !hasproperty(tensor_stats, :time_fft_assembly) ? 0.0 : tensor_stats.time_fft_assembly,
        :memory_estimate => tensor_stats === nothing ? 0 : tensor_stats.memory_estimate,
        :tensor_layout => tensor_stats === nothing || !hasproperty(tensor_stats, :tensor_layout) ? missing : tensor_stats.tensor_layout,
        :time_packed_spectra => tensor_stats === nothing || !hasproperty(tensor_stats, :time_packed_spectra) ? 0.0 : tensor_stats.time_packed_spectra,
        :time_fill_Qmat => tensor_stats === nothing || !hasproperty(tensor_stats, :time_fill_Qmat) ? 0.0 : tensor_stats.time_fill_Qmat,
        :time_forward_fft => tensor_stats === nothing || !hasproperty(tensor_stats, :time_forward_fft) ? 0.0 : tensor_stats.time_forward_fft,
        :time_freq_accum => tensor_stats === nothing || !hasproperty(tensor_stats, :time_freq_accum) ? 0.0 : tensor_stats.time_freq_accum,
        :time_inverse_fft => tensor_stats === nothing || !hasproperty(tensor_stats, :time_inverse_fft) ? 0.0 : tensor_stats.time_inverse_fft,
        :time_write_A => tensor_stats === nothing || !hasproperty(tensor_stats, :time_write_A) ? 0.0 : tensor_stats.time_write_A,
        :time_left_corner => tensor_stats === nothing || !hasproperty(tensor_stats, :time_left_corner) ? 0.0 : tensor_stats.time_left_corner,
        :time_H_generation_stream => tensor_stats === nothing || !hasproperty(tensor_stats, :time_H_generation_stream) ? 0.0 : tensor_stats.time_H_generation_stream,
    )

    if compare_old
        Aold, _ = Weak_VoltMat_DE.weak_voltmat_decheb(
            (x, t) -> 1.0;
            mu=mu64,
            N,
            omega=omega64,
            Csize_y,
            Csize_eta,
            return_info=true,
        )
        Adense = output === :dense ? Aout : band_to_dense(Aout, N, lower_bw, upper_bw)
        info[:compare_old_full_max_abs_error] = maxabs_array(Adense .- Aold)
    end

    return return_info ? (Aout, info) : Aout
end

function compare_assemblies(; mu=0.5, omega=4.0, N=256, upper_bw=120)
    mu64 = Float64(mu)
    omega64 = Float64(omega)
    C, _ = build_C(mu64, omega64)
    Hlayers = build_verified_H_layers(mu64, N)

    local G_scatter, G_direct_conv, G_fft
    t_scatter = @elapsed begin
        G_scatter = assemble_G_direct_verified(C, mu64, N, upper_bw; Hlayers)
    end
    t_direct_conv = @elapsed begin
        G_direct_conv = assemble_G_direct_convolution(C, mu64, N, upper_bw; Hlayers)
    end
    t_fft = @elapsed begin
        G_fft = assemble_G_fft_convolution(C, mu64, N, upper_bw; Hlayers)
    end

    A_scatter = multiply_by_one_plus_y_over_two_banded(G_scatter, N, upper_bw)
    A_fft = multiply_by_one_plus_y_over_two_banded(G_fft, N, upper_bw)
    g_scatter_direct = maxabs_array(G_scatter .- G_direct_conv)
    g_scatter_fft = maxabs_array(G_scatter .- G_fft)
    a_scatter_fft = maxabs_array(A_scatter .- A_fft)

    println("===== Assembly mode comparison =====")
    println("N = ", N, ", mu = ", mu64, ", omega = ", omega64, ", upper_bw = ", upper_bw)
    @printf("max |G_direct_scatter - G_direct_convolution| = %.6e\n", g_scatter_direct)
    @printf("max |G_direct_scatter - G_fft_convolution|    = %.6e\n", g_scatter_fft)
    @printf("max |A_direct_scatter - A_fft_convolution|    = %.6e\n", a_scatter_fft)
    @printf("timing direct_scatter     %.6g s\n", t_scatter)
    @printf("timing direct_convolution %.6g s\n", t_direct_conv)
    @printf("timing fft_convolution    %.6g s\n", t_fft)

    return (;
        g_scatter_direct,
        g_scatter_fft,
        a_scatter_fft,
        t_scatter,
        t_direct_conv,
        t_fft,
    )
end

function compare_fft_branches(; mu=0.5, omega=4.0, N=256, upper_bw=120)
    mu64 = Float64(mu)
    omega64 = Float64(omega)
    C, _ = build_C(mu64, omega64)
    Hseed = build_verified_H_seed(mu64, N)
    Hlayers = build_verified_H_layers(mu64, N)

    G_fft_seed = assemble_G_fft_convolution(
        C,
        mu64,
        N,
        upper_bw;
        Hlayers=nothing,
        Hseed,
    )
    G_fft_layers = assemble_G_fft_convolution(
        C,
        mu64,
        N,
        upper_bw;
        Hlayers,
    )
    err = maxabs_banddiff(G_fft_seed, G_fft_layers)
    println("===== FFT branch comparison =====")
    println("N = ", N, ", mu = ", mu64, ", omega = ", omega64, ", upper_bw = ", upper_bw)
    @printf("max |G_fft_seed - G_fft_layers| = %.6e\n", err)
    if err > 1e-10
        @warn "FFT production branch and validation branch disagree." err
    end
    return err
end

function compare_narrow_H_generator(; mu=0.5, N=256, upper_bw=120, h0_init::Symbol=:s_recurrence, prec_bits::Int=1536)
    mu64 = Float64(mu)
    upper_bw_G = upper_bw + 1
    S = upper_bw_G + I_DEG + 1
    Sguard = S + J_DEG + 2
    Hlayers = build_verified_H_layers(mu64, N)

    maxerr = 0.0
    max_h0_oracle_err = 0.0
    worst = (r=0, j=0, s=0, k=0, narrow=0.0, ref=0.0)
    worst_h0 = (err=0.0, r=0, s=0, k=0)
    worst_h1 = (err=0.0, r=0, s=0, k=0)
    worst_h0_oracle = (err=0.0, r=0, s=0, k=0, candidate=0.0, oracle=0.0)
    worst_r = 0
    worst_r_err = 0.0
    Br = 1.0 / mu64
    for r in 0:(N - 1)
        H0_candidate = if h0_init === :s_recurrence
            h0_line_s_recurrence(r, mu64, min(S, N - 1 - r), Br)
        elseif h0_init === :hypergeom_float64
            h0_line_hypergeom(r, mu64, min(S, N - 1 - r), Br)
        elseif h0_init === :bigfloat_hypergeom
            h0_line_bigfloat_hypergeom(r, mu64, min(S, N - 1 - r), Br; prec_bits)
        else
            throw(ArgumentError("h0_init must be :s_recurrence, :hypergeom_float64, or :bigfloat_hypergeom"))
        end
        H0_oracle = h0_init === :bigfloat_hypergeom ?
            H0_candidate :
            h0_line_bigfloat_hypergeom(r, mu64, min(S, N - 1 - r), Br; prec_bits)
        for s in 0:(length(H0_candidate) - 1)
            k = r + s
            err0 = abs(H0_candidate[s + 1] - H0_oracle[s + 1])
            if err0 > max_h0_oracle_err
                max_h0_oracle_err = err0
                worst_h0_oracle = (;
                    err=err0,
                    r,
                    s,
                    k,
                    candidate=H0_candidate[s + 1],
                    oracle=H0_oracle[s + 1],
                )
            end
        end

        Hjs = build_Hjs_line(r, mu64, S, Sguard, Br; h0_init, prec_bits)
        rerr = 0.0
        smax = min(S, N - 1 - r)
        for j in 0:J_DEG
            Href = Hlayers[j + 1]
            for s in 0:smax
                k = r + s
                err = abs(Hjs[j + 1, s + 1] - Href[r + 1, k + 1])
                if err > maxerr
                    maxerr = err
                    worst = (;
                        r,
                        j,
                        s,
                        k,
                        narrow=Hjs[j + 1, s + 1],
                        ref=Href[r + 1, k + 1],
                    )
                end
                if j == 0 && err > worst_h0.err
                    worst_h0 = (; err, r, s, k)
                elseif j == 1 && err > worst_h1.err
                    worst_h1 = (; err, r, s, k)
                end
                rerr = max(rerr, err)
            end
        end
        if rerr > worst_r_err
            worst_r_err = rerr
            worst_r = r
        end
        if r < N - 1
            Br *= (r + 1.0) / (r + mu64 + 1.0)
        end
    end

    println("===== Narrow fixed-r H generator comparison =====")
    println("N = ", N, ", mu = ", mu64, ", upper_bw = ", upper_bw, ", h0_init = ", h0_init)
    println("S = ", S, ", Sguard = ", Sguard)
    @printf("max |H0_candidate - H0_bigfloat| = %.6e\n", max_h0_oracle_err)
    @printf("max |Hjs - Hlayers| = %.6e\n", maxerr)
    println("worst H0 oracle tuple: ", worst_h0_oracle)
    println("worst r-wise error at r = ", worst_r, ": ", @sprintf("%.6e", worst_r_err))
    println("worst tuple: ", worst)
    println("worst H0 tuple: ", worst_h0)
    println("worst H1 tuple: ", worst_h1)
    if max_h0_oracle_err > 1e-11
        @warn "H0 candidate disagrees with BigFloat hypergeom oracle." max_h0_oracle_err worst_h0_oracle
    end
    if maxerr > 1e-10
        @warn "Narrow fixed-r H generator disagrees with verified Hlayers." maxerr worst
    end
    return (; maxerr, max_h0_oracle_err, worst_h0_oracle, worst_r, worst_r_err, worst, worst_h0, worst_h1)
end

function compare_narrow_direct_banded(; mu=0.5, omega=4.0, N=256, upper_bw=120, h0_init::Symbol=:s_recurrence, prec_bits::Int=1536)
    mu64 = Float64(mu)
    omega64 = Float64(omega)
    C, _ = build_C(mu64, omega64)
    upper_bw_G = upper_bw + 1
    S = upper_bw_G + I_DEG + 1
    Sguard = S + J_DEG + 2

    local G_ref, G_narrow
    t_ref = @elapsed begin
        G_ref = assemble_G_direct_convolution(C, mu64, N, upper_bw)
    end
    t_narrow = @elapsed begin
        G_narrow = assemble_G_direct_banded(C, mu64, N, upper_bw, S, Sguard; h0_init, prec_bits)
    end
    A_ref = multiply_by_one_plus_y_over_two_banded(G_ref, N, upper_bw)
    A_narrow = multiply_by_one_plus_y_over_two_banded(G_narrow, N, upper_bw)
    Gerr = maxabs_banddiff(G_narrow, G_ref)
    Aerr = maxabs_banddiff(A_narrow, A_ref)

    println("===== Narrow direct-banded assembly comparison =====")
    println("N = ", N, ", mu = ", mu64, ", omega = ", omega64, ", upper_bw = ", upper_bw, ", h0_init = ", h0_init)
    println("S = ", S, ", Sguard = ", Sguard)
    @printf("max |G_narrow - G_ref| = %.6e\n", Gerr)
    @printf("max |A_narrow - A_ref| = %.6e\n", Aerr)
    @printf("time direct_convolution %.6g s\n", t_ref)
    @printf("time narrow_direct_banded %.6g s\n", t_narrow)
    if max(Gerr, Aerr) > 1e-10
        @warn "Narrow direct-banded assembly disagrees with reference assembly." Gerr Aerr
    end
    return (; Gerr, Aerr, t_ref, t_narrow)
end

function compare_narrow_fft_batched(;
    N=400,
    mu=0.5,
    omega=4.0,
    upper_bw=120,
    block_cols=64,
    h0_init::Symbol=:s_recurrence,
    prec_bits::Int=1536,
)
    mu64 = Float64(mu)
    omega64 = Float64(omega)
    C, _ = build_C(mu64, omega64)
    upper_bw_G = upper_bw + 1
    S = upper_bw_G + I_DEG + 1
    Sguard = S + J_DEG + 2

    local G_ref, G_fft
    t_ref = @elapsed begin
        G_ref = assemble_G_direct_banded(C, mu64, N, upper_bw, S, Sguard; h0_init, prec_bits)
    end
    t_fft = @elapsed begin
        G_fft = assemble_G_narrow_fft_batched(
            C,
            mu64,
            N,
            upper_bw,
            S,
            Sguard;
            h0_init,
            prec_bits,
            block_cols,
        )
    end
    A_ref = multiply_by_one_plus_y_over_two_banded(G_ref, N, upper_bw)
    A_fft = multiply_by_one_plus_y_over_two_banded(G_fft, N, upper_bw)
    Gerr = maxabs_banddiff(G_fft, G_ref)
    Aerr = maxabs_banddiff(A_fft, A_ref)

    println("===== Narrow FFT batched assembly comparison =====")
    println("N = ", N, ", mu = ", mu64, ", omega = ", omega64, ", upper_bw = ", upper_bw, ", block_cols = ", block_cols)
    @printf("max |G_fft-G_ref| = %.6e\n", Gerr)
    @printf("max |A_fft-A_ref| = %.6e\n", Aerr)
    @printf("time narrow_direct_banded %.6g s\n", t_ref)
    @printf("time narrow_fft_batched %.6g s\n", t_fft)
    if max(Gerr, Aerr) > 1e-12
        @warn "narrow_fft_batched disagrees with narrow_direct_banded." Gerr Aerr
    end
    return (; Gerr, Aerr, t_ref, t_fft)
end

function compare_narrow_fft_packed_absorbM(;
    N=400,
    mu=0.5,
    omega=4.0,
    upper_bw=120,
    block_cols=256,
    h0_init::Symbol=:s_recurrence,
    prec_bits::Int=1536,
)
    local A_ref, A_fft, info_ref, info_fft
    t_ref = @elapsed begin
        A_ref, info_ref = fast_fracint_K1_decheb(;
            mu,
            N,
            omega,
            upper_bw,
            assembly=:narrow_direct_banded,
            output=:banded,
            return_info=true,
            h0_init,
            prec_bits,
        )
    end
    t_fft = @elapsed begin
        A_fft, info_fft = fast_fracint_K1_decheb(;
            mu,
            N,
            omega,
            upper_bw,
            assembly=:narrow_fft_packed_absorbM,
            output=:banded,
            return_info=true,
            h0_init,
            prec_bits,
            block_cols,
        )
    end
    err = maxabs_banddiff(A_fft, A_ref)
    println("===== Narrow FFT packed absorbM comparison =====")
    println("N = ", N, ", mu = ", mu, ", omega = ", omega, ", upper_bw = ", upper_bw, ", block_cols = ", block_cols)
    @printf("max |A_fft-A_ref| = %.6e\n", err)
    @printf("time narrow_direct_banded %.6g s\n", t_ref)
    @printf("time narrow_fft_packed_absorbM %.6g s\n", t_fft)
    if err > 1e-12
        @warn "narrow_fft_packed_absorbM disagrees with narrow_direct_banded." err
    end
    return (; err, t_ref, t_fft, info_ref, info_fft)
end

function compare_narrow_fft_tensor_absorbM(;
    N=400,
    mu=0.5,
    omega=4.0,
    upper_bw=120,
    h0_init::Symbol=:s_recurrence,
    prec_bits::Int=1536,
)
    local A_ref, A_tensor, info_ref, info_tensor
    t_ref = @elapsed begin
        A_ref, info_ref = fast_fracint_K1_decheb(;
            mu,
            N,
            omega,
            upper_bw,
            assembly=:narrow_direct_banded,
            output=:banded,
            return_info=true,
            h0_init,
            prec_bits,
        )
    end
    t_tensor = @elapsed begin
        A_tensor, info_tensor = fast_fracint_K1_decheb(;
            mu,
            N,
            omega,
            upper_bw,
            assembly=:narrow_fft_tensor_absorbM,
            output=:banded,
            return_info=true,
            h0_init,
            prec_bits,
        )
    end
    err = maxabs_banddiff(A_tensor, A_ref)
    println("===== Narrow FFT tensor absorbM comparison =====")
    println("N = ", N, ", mu = ", mu, ", omega = ", omega, ", upper_bw = ", upper_bw)
    @printf("max |A_tensor-A_ref| = %.6e\n", err)
    @printf("time narrow_direct_banded %.6g s\n", t_ref)
    @printf("time narrow_fft_tensor_absorbM %.6g s\n", t_tensor)
    @printf("time tensor generation %.6g s\n", info_tensor[:time_tensor_generation])
    @printf("time fft assembly %.6g s\n", info_tensor[:time_fft_assembly])
    @printf("memory estimate %.3f MiB\n", info_tensor[:memory_estimate] / 2.0^20)
    if err > 1e-12
        @warn "narrow_fft_tensor_absorbM disagrees with narrow_direct_banded." err
    end
    return (; err, t_ref, t_tensor, info_ref, info_tensor)
end

function compare_narrow_fft_tensor_accum_absorbM(;
    N=400,
    mu=0.5,
    omega=4.0,
    upper_bw=120,
    h0_init::Symbol=:s_recurrence,
    prec_bits::Int=1536,
)
    local A_ref, A_acc, info_ref, info_acc
    t_ref = @elapsed begin
        A_ref, info_ref = fast_fracint_K1_decheb(;
            mu,
            N,
            omega,
            upper_bw,
            assembly=:narrow_direct_banded,
            output=:banded,
            return_info=true,
            h0_init,
            prec_bits,
        )
    end
    t_acc = @elapsed begin
        A_acc, info_acc = fast_fracint_K1_decheb(;
            mu,
            N,
            omega,
            upper_bw,
            assembly=:narrow_fft_tensor_accum_absorbM,
            output=:banded,
            return_info=true,
            h0_init,
            prec_bits,
        )
    end
    err = maxabs_banddiff(A_acc, A_ref)
    println("===== Narrow FFT tensor accum absorbM comparison =====")
    println("N = ", N, ", mu = ", mu, ", omega = ", omega, ", upper_bw = ", upper_bw)
    @printf("max |A_acc-A_ref| = %.6e\n", err)
    @printf("time narrow_direct_banded %.6g s\n", t_ref)
    @printf("time narrow_fft_tensor_accum_absorbM %.6g s\n", t_acc)
    @printf("time tensor generation %.6g s\n", info_acc[:time_tensor_generation])
    @printf("time forward fft %.6g s\n", info_acc[:time_forward_fft])
    @printf("time freq accum %.6g s\n", info_acc[:time_freq_accum])
    @printf("time inverse fft %.6g s\n", info_acc[:time_inverse_fft])
    @printf("time write A %.6g s\n", info_acc[:time_write_A])
    @printf("time left corner %.6g s\n", info_acc[:time_left_corner])
    @printf("memory estimate %.3f MiB\n", info_acc[:memory_estimate] / 2.0^20)
    if err > 1e-12
        @warn "narrow_fft_tensor_accum_absorbM disagrees with narrow_direct_banded." err
    end
    return (; err, t_ref, t_acc, info_ref, info_acc)
end

function compare_narrow_fft_tensor_accum_blocked_absorbM(;
    N=400,
    mu=0.5,
    omega=4.0,
    upper_bw=120,
    block_cols=128,
    h0_init::Symbol=:s_recurrence,
    prec_bits::Int=1536,
)
    A_ref, info_ref = fast_fracint_K1_decheb(;
        mu, N, omega, upper_bw,
        assembly=:narrow_direct_banded,
        output=:banded,
        return_info=true,
        h0_init,
        prec_bits,
    )
    A_blk, info_blk = fast_fracint_K1_decheb(;
        mu, N, omega, upper_bw,
        assembly=:narrow_fft_tensor_accum_blocked_absorbM,
        output=:banded,
        return_info=true,
        h0_init,
        prec_bits,
        block_cols,
    )
    err = maxabs_banddiff(A_blk, A_ref)
    println("===== Narrow FFT tensor accum blocked absorbM comparison =====")
    println("N = ", N, ", mu = ", mu, ", omega = ", omega, ", upper_bw = ", upper_bw, ", block_cols = ", block_cols)
    @printf("max |A_blocked-A_ref| = %.6e\n", err)
    @printf("time direct %.6g s\n", info_ref[:build_time_fast])
    @printf("time blocked %.6g s\n", info_blk[:build_time_fast])
    @printf("time tensor generation %.6g s\n", info_blk[:time_tensor_generation])
    @printf("time forward fft %.6g s\n", info_blk[:time_forward_fft])
    @printf("time freq accum %.6g s\n", info_blk[:time_freq_accum])
    @printf("time inverse fft %.6g s\n", info_blk[:time_inverse_fft])
    @printf("time write A %.6g s\n", info_blk[:time_write_A])
    if err > 1e-12
        @warn "narrow_fft_tensor_accum_blocked_absorbM disagrees with narrow_direct_banded" err
    end
    return (; err, info_ref, info_blk)
end

function compare_narrow_direct_absorbM(;
    N=400,
    mu=0.5,
    omega=4.0,
    upper_bw=120,
    h0_init::Symbol=:s_recurrence,
    prec_bits::Int=1536,
)
    A_ref, info_ref = fast_fracint_K1_decheb(;
        mu,
        N,
        omega,
        upper_bw,
        assembly=:narrow_direct_banded,
        output=:banded,
        return_info=true,
        h0_init,
        prec_bits,
    )
    A_abs, info_abs = fast_fracint_K1_decheb(;
        mu,
        N,
        omega,
        upper_bw,
        assembly=:narrow_direct_absorbM,
        output=:banded,
        return_info=true,
        h0_init,
        prec_bits,
    )
    err = maxabs_banddiff(A_abs, A_ref)
    println("===== Narrow direct absorbM comparison =====")
    println("N = ", N, ", mu = ", mu, ", omega = ", omega, ", upper_bw = ", upper_bw)
    @printf("max |A_abs-A_ref| = %.6e\n", err)
    @printf("time narrow_direct_banded %.6g s\n", info_ref[:build_time_fast])
    @printf("time narrow_direct_absorbM %.6g s\n", info_abs[:build_time_fast])
    if err > 1e-12
        @warn "narrow_direct_absorbM disagrees with narrow_direct_banded" err
    end
    return (; err, info_ref, info_abs)
end

function compare_narrow_rfft_tensor_accum_absorbM(;
    N=400,
    mu=0.5,
    omega=4.0,
    upper_bw=120,
    h0_init::Symbol=:s_recurrence,
    prec_bits::Int=1536,
)
    A_ref, info_ref = fast_fracint_K1_decheb(;
        mu, N, omega, upper_bw,
        assembly=:narrow_direct_banded,
        output=:banded,
        return_info=true,
        h0_init,
        prec_bits,
    )
    A_rfft, info_rfft = fast_fracint_K1_decheb(;
        mu, N, omega, upper_bw,
        assembly=:narrow_rfft_tensor_accum_absorbM,
        output=:banded,
        return_info=true,
        h0_init,
        prec_bits,
    )
    err = maxabs_banddiff(A_rfft, A_ref)
    println("===== Narrow RFFT tensor accum absorbM comparison =====")
    println("N = ", N, ", mu = ", mu, ", omega = ", omega, ", upper_bw = ", upper_bw)
    @printf("max |A_rfft-A_ref| = %.6e\n", err)
    @printf("time direct %.6g s\n", info_ref[:build_time_fast])
    @printf("time rfft %.6g s\n", info_rfft[:build_time_fast])
    @printf("time tensor generation %.6g s\n", info_rfft[:time_tensor_generation])
    @printf("time forward fft %.6g s\n", info_rfft[:time_forward_fft])
    @printf("time freq accum %.6g s\n", info_rfft[:time_freq_accum])
    @printf("time inverse fft %.6g s\n", info_rfft[:time_inverse_fft])
    if err > 1e-12
        @warn "narrow_rfft_tensor_accum_absorbM disagrees with narrow_direct_banded" err
    end
    return (; err, info_ref, info_rfft)
end

function compare_narrow_rfft_stream_accum_absorbM(;
    N=400,
    mu=0.5,
    omega=4.0,
    upper_bw=120,
    h0_init::Symbol=:s_recurrence,
    prec_bits::Int=1536,
)
    A_ref, info_ref = fast_fracint_K1_decheb(;
        mu, N, omega, upper_bw,
        assembly=:narrow_direct_absorbM,
        output=:banded,
        return_info=true,
        h0_init,
        prec_bits,
    )
    A_stream, info_stream = fast_fracint_K1_decheb(;
        mu, N, omega, upper_bw,
        assembly=:narrow_rfft_stream_accum_absorbM,
        output=:banded,
        return_info=true,
        h0_init,
        prec_bits,
    )
    err = maxabs_banddiff(A_stream, A_ref)
    println("===== Narrow RFFT stream accum absorbM comparison =====")
    println("N = ", N, ", mu = ", mu, ", omega = ", omega, ", upper_bw = ", upper_bw)
    @printf("max |A_stream-A_ref| = %.6e\n", err)
    @printf("time direct_absorb %.6g s\n", info_ref[:build_time_fast])
    @printf("time rfft_stream %.6g s\n", info_stream[:build_time_fast])
    @printf("time H generation stream %.6g s\n", info_stream[:time_H_generation_stream])
    @printf("time forward fft %.6g s\n", info_stream[:time_forward_fft])
    @printf("time freq accum %.6g s\n", info_stream[:time_freq_accum])
    @printf("time inverse fft %.6g s\n", info_stream[:time_inverse_fft])
    @printf("time left corner %.6g s\n", info_stream[:time_left_corner])
    @printf("memory estimate %.3f MiB\n", info_stream[:memory_estimate] / 2.0^20)
    if err > 1e-12
        @warn "narrow_rfft_stream_accum_absorbM disagrees with direct_absorb" err
    end
    return (; err, info_ref, info_stream)
end

function compare_narrow_rfft_tensor_accum_blocked_absorbM(;
    N=400,
    mu=0.5,
    omega=4.0,
    upper_bw=120,
    block_cols=64,
    h0_init::Symbol=:s_recurrence,
    prec_bits::Int=1536,
    fftw_flags=nothing,
)
    A_ref, info_ref = fast_fracint_K1_decheb(;
        mu, N, omega, upper_bw,
        assembly=:narrow_direct_absorbM,
        output=:banded,
        return_info=true,
        h0_init,
        prec_bits,
    )
    A_blk, info_blk = fast_fracint_K1_decheb(;
        mu, N, omega, upper_bw,
        assembly=:narrow_rfft_tensor_accum_blocked_absorbM,
        output=:banded,
        return_info=true,
        h0_init,
        prec_bits,
        block_cols,
        fftw_flags,
    )
    err = maxabs_banddiff(A_blk, A_ref)
    println("===== Narrow blocked batched RFFT absorbM comparison =====")
    println("N = ", N, ", mu = ", mu, ", omega = ", omega, ", upper_bw = ", upper_bw, ", block_cols = ", block_cols)
    @printf("max |A_rfft_blocked-A_direct_absorbM| = %.6e\n", err)
    @printf("time direct_absorb %.6g s\n", info_ref[:build_time_fast])
    @printf("time rfft_blocked %.6g s\n", info_blk[:build_time_fast])
    @printf("time fill Q %.6g s\n", info_blk[:time_fill_Qmat])
    @printf("time forward rfft %.6g s\n", info_blk[:time_forward_fft])
    @printf("time freq accum %.6g s\n", info_blk[:time_freq_accum])
    @printf("time inverse irfft %.6g s\n", info_blk[:time_inverse_fft])
    @printf("time write A %.6g s\n", info_blk[:time_write_A])
    @printf("time left corner %.6g s\n", info_blk[:time_left_corner])
    @printf("memory estimate %.3f MiB\n", info_blk[:memory_estimate] / 2.0^20)
    if err > 1e-12
        @warn "narrow_rfft_tensor_accum_blocked_absorbM disagrees with narrow_direct_absorbM" err
    end
    return (; err, info_ref, info_blk)
end

function test_convolution_index_random(; N=64, upper_bw=20)
    rng = Random.MersenneTwister(20260607)
    C = randn(rng, Csize_y, Csize_eta)
    Hlayers = Vector{Matrix{Float64}}(undef, J_DEG + 1)
    for j in 0:J_DEG
        Hj = zeros(Float64, N, N)
        for k in 0:(N - 1), r in 0:k
            Hj[r + 1, k + 1] = randn(rng)
        end
        Hlayers[j + 1] = Hj
    end
    G_direct = assemble_G_direct_convolution(C, 0.5, N, upper_bw; Hlayers)
    G_fft = assemble_G_fft_convolution(C, 0.5, N, upper_bw; Hlayers)
    err = maxabs_banddiff(G_direct, G_fft)
    println("===== Random convolution-index test =====")
    println("N = ", N, ", upper_bw = ", upper_bw)
    @printf("max |G_direct_convolution - G_fft_convolution| = %.6e\n", err)
    if err > 1e-10
        @warn "FFT convolution indexing disagrees with direct convolution on random data." err
    end
    return err
end

function loglog_tail_slope(Ns, times; tail_count::Int=4)
    n = min(length(Ns), length(times), tail_count)
    n >= 2 || return NaN
    xs = log.(Float64.(Ns[(end - n + 1):end]))
    ys = log.(Float64.(times[(end - n + 1):end]))
    xbar = sum(xs) / n
    ybar = sum(ys) / n
    denom = sum((x - xbar)^2 for x in xs)
    denom == 0.0 && return NaN
    return sum((xs[i] - xbar) * (ys[i] - ybar) for i in 1:n) / denom
end

function benchmark_fast_only(;
    Ns=[256, 400, 600, 800, 1200, 1600, 2000],
    mu=0.5,
    omega=4.0,
    upper_bw=120,
    output::Symbol=:banded,
    h0_init::Symbol=:s_recurrence,
    prec_bits::Int=1536,
)
    println("===== Fast-only benchmark: K=1 DE weak Volterra banded construction =====")
    println("C size fixed: 100 x 100")
    println("mu = ", mu, ", omega = ", omega, ", upper_bw = ", upper_bw)
    println("Old implementation is not called.")
    println()
    println("output = ", output)
    println()
    @printf("%-7s %-14s %-14s %-14s %-10s %-10s %-10s %-10s %-10s %-12s %-12s\n",
        "N", "direct_time", "fft_time", "narrow_time", "H_time", "dir_asm", "fft_asm", "narr_asm", "nnz_band", "fft_err", "narrow_err")

    direct_times = Float64[]
    fft_times = Float64[]
    narrow_times = Float64[]
    rows = NamedTuple[]

    for N in Ns
        A_direct, info_direct = fast_fracint_K1_decheb(;
            mu,
            N,
            omega,
            upper_bw,
            assembly=:direct_convolution,
            return_info=true,
            check_fft=false,
            output,
        )
        A_fft, info_fft = fast_fracint_K1_decheb(;
            mu,
            N,
            omega,
            upper_bw,
            assembly=:fft_convolution,
            return_info=true,
            check_fft=false,
            output,
        )
        A_narrow, info_narrow = fast_fracint_K1_decheb(;
            mu,
            N,
            omega,
            upper_bw,
            assembly=:narrow_direct_banded,
            return_info=true,
            check_fft=false,
            output,
            h0_init,
            prec_bits,
        )
        err = output === :banded ? maxabs_banddiff(A_direct, A_fft) : maxabs_array(A_direct .- A_fft)
        narrow_err = output === :banded ? maxabs_banddiff(A_direct, A_narrow) : maxabs_array(A_direct .- A_narrow)
        t_direct = info_direct[:build_time_fast]
        t_fft = info_fft[:build_time_fast]
        t_narrow = info_narrow[:build_time_fast]
        push!(direct_times, t_direct)
        push!(fft_times, t_fft)
        push!(narrow_times, t_narrow)
        ratio = t_direct == 0.0 ? Inf : t_fft / t_direct

        @printf("%-7d %-14.6g %-14.6g %-14.6g %-10.3g %-10.3g %-10.3g %-10.3g %-10d %-12.3e %-12.3e\n",
            N,
            t_direct,
            t_fft,
            t_narrow,
            info_direct[:time_H_generation],
            info_direct[:time_assembly],
            info_fft[:time_assembly],
            info_narrow[:time_assembly],
            info_direct[:nnz_band],
            err,
            narrow_err,
        )
        if err > 1e-10
            @warn "FFT benchmark result disagrees with direct_convolution." N fft_err=err
        end
        if narrow_err > 1e-10
            @warn "Narrow direct-banded result disagrees with direct_convolution." N narrow_err
        end

        push!(rows, (;
            N,
            direct_total=t_direct,
            fft_total=t_fft,
            narrow_total=t_narrow,
            fft_over_direct=ratio,
            H_time=info_direct[:time_H_generation],
            direct_assembly=info_direct[:time_assembly],
            fft_assembly=info_fft[:time_assembly],
            narrow_assembly=info_narrow[:time_assembly],
            final_multiply=info_direct[:time_final_multiply],
            nnz_band=info_direct[:nnz_band],
            fft_direct_err=err,
            narrow_direct_err=narrow_err,
        ))
    end

    slope_direct = loglog_tail_slope(Ns, direct_times)
    slope_fft = loglog_tail_slope(Ns, fft_times)
    slope_narrow = loglog_tail_slope(Ns, narrow_times)
    println()
    @printf("log-log tail slope direct_convolution: %.3f\n", slope_direct)
    @printf("log-log tail slope fft_convolution:    %.3f\n", slope_fft)
    @printf("log-log tail slope narrow_direct_banded: %.3f\n", slope_narrow)
    return (; rows, slope_direct, slope_fft, slope_narrow)
end

function benchmark_narrow_only(;
    Ns=[400, 800, 1200, 2000, 4000, 8000, 10000],
    mu=0.5,
    omega=4.0,
    upper_bw=120,
    block_cols::Int=64,
)
    println("===== Narrow-only benchmark: K=1 DE weak Volterra banded construction =====")
    println("assembly = narrow_direct_banded, output = banded")
    println("C size fixed: 100 x 100")
    println("mu = ", mu, ", omega = ", omega, ", upper_bw = ", upper_bw)
    println()
    @printf("%-8s %-14s %-14s %-14s %-14s %-12s %-12s %-12s %-14s\n",
        "N", "narrow_time", "direct_absorb", "fft_batch", "fft_packed", "abs_err", "batch_err", "packed_err", "final_multiply")

    times = Float64[]
    fft_times = Float64[]
    packed_times = Float64[]
    rows = NamedTuple[]
    for N in Ns
        A_direct, info = fast_fracint_K1_decheb(;
            mu,
            N,
            omega,
            upper_bw,
            assembly=:narrow_direct_banded,
            output=:banded,
            return_info=true,
        )
        A_fft, info_fft = fast_fracint_K1_decheb(;
            mu,
            N,
            omega,
            upper_bw,
            assembly=:narrow_fft_batched,
            output=:banded,
            return_info=true,
            block_cols,
        )
        A_abs, info_abs = fast_fracint_K1_decheb(;
            mu,
            N,
            omega,
            upper_bw,
            assembly=:narrow_direct_absorbM,
            output=:banded,
            return_info=true,
        )
        A_packed, info_packed = fast_fracint_K1_decheb(;
            mu,
            N,
            omega,
            upper_bw,
            assembly=:narrow_fft_packed_absorbM,
            output=:banded,
            return_info=true,
            block_cols,
        )
        abs_err = maxabs_banddiff(A_abs, A_direct)
        fft_err = maxabs_banddiff(A_fft, A_direct)
        packed_err = maxabs_banddiff(A_packed, A_direct)
        t = info[:build_time_fast]
        t_abs = info_abs[:build_time_fast]
        t_fft = info_fft[:build_time_fast]
        t_packed = info_packed[:build_time_fast]
        push!(times, t)
        push!(fft_times, t_fft)
        push!(packed_times, t_packed)
        @printf("%-8d %-14.6g %-14.6g %-14.6g %-14.6g %-12.3e %-12.3e %-12.3e %-14.3g\n",
            N,
            t,
            t_abs,
            t_fft,
            t_packed,
            abs_err,
            fft_err,
            packed_err,
            info[:time_final_multiply],
        )
        if abs_err > 1e-12
            @warn "narrow_direct_absorbM benchmark result disagrees with narrow_direct_banded" N abs_err
        end
        if fft_err > 1e-12
            @warn "narrow_fft_batched benchmark result disagrees with narrow_direct_banded" N fft_err
        end
        if packed_err > 1e-12
            @warn "narrow_fft_packed_absorbM benchmark result disagrees with narrow_direct_banded" N packed_err
        end
        push!(rows, (;
            N,
            narrow_time=t,
            direct_absorb_time=t_abs,
            fft_batch_time=t_fft,
            fft_packed_time=t_packed,
            H_time=info[:time_H_generation],
            assembly_time=info[:time_assembly],
            fft_batch_assembly_time=info_fft[:time_assembly],
            fft_packed_assembly_time=info_packed[:time_assembly],
            final_multiply_time=info[:time_final_multiply],
            nnz_band=info[:nnz_band],
            direct_absorb_err=abs_err,
            fft_batch_err=fft_err,
            fft_packed_err=packed_err,
            finite=info[:finite_check],
        ))
    end
    slope = loglog_tail_slope(Ns, times)
    slope_fft = loglog_tail_slope(Ns, fft_times)
    slope_packed = loglog_tail_slope(Ns, packed_times)
    println()
    @printf("log-log tail slope narrow_direct_banded: %.3f\n", slope)
    @printf("log-log tail slope narrow_fft_batched: %.3f\n", slope_fft)
    @printf("log-log tail slope narrow_fft_packed_absorbM: %.3f\n", slope_packed)
    return (; rows, slope, slope_fft, slope_packed)
end

function benchmark_fft_packed_blockcols(;
    Ns=[2000, 4000, 8000],
    block_cols_list=[128, 256, 384],
    mu=0.5,
    omega=4.0,
    upper_bw=120,
)
    println("===== FFT packed absorbM block-cols benchmark =====")
    println("assembly = narrow_fft_packed_absorbM, output = banded")
    println("mu = ", mu, ", omega = ", omega, ", upper_bw = ", upper_bw)
    println()
    @printf("%-8s %-10s %-14s %-14s %-12s\n", "N", "block", "direct_time", "packed_time", "err")
    rows = NamedTuple[]
    for N in Ns
        Aref, info_ref = fast_fracint_K1_decheb(;
            mu,
            N,
            omega,
            upper_bw,
            assembly=:narrow_direct_banded,
            output=:banded,
            return_info=true,
        )
        for block_cols in block_cols_list
            Afast, info = fast_fracint_K1_decheb(;
                mu,
                N,
                omega,
                upper_bw,
                assembly=:narrow_fft_packed_absorbM,
                output=:banded,
                return_info=true,
                block_cols,
            )
            err = maxabs_banddiff(Afast, Aref)
            @printf("%-8d %-10d %-14.6g %-14.6g %-12.3e\n",
                N,
                block_cols,
                info_ref[:build_time_fast],
                info[:build_time_fast],
                err,
            )
            if err > 1e-12
                @warn "narrow_fft_packed_absorbM block-cols benchmark disagrees with direct reference" N block_cols err
            end
            push!(rows, (;
                N,
                block_cols,
                direct_time=info_ref[:build_time_fast],
                packed_time=info[:build_time_fast],
                err,
            ))
        end
    end
    return rows
end

function benchmark_fft_tensor_absorbM(;
    Ns=[1000, 2000, 4000, 8000, 10000],
    mu=0.5,
    omega=4.0,
    upper_bw=120,
    max_memory_mib::Float64=2048.0,
)
    println("===== FFT tensor absorbM benchmark =====")
    println("Compare narrow_direct_banded, narrow_fft_packed_absorbM, narrow_fft_tensor_absorbM")
    println("mu = ", mu, ", omega = ", omega, ", upper_bw = ", upper_bw)
    println()
    @printf("%-8s %-12s %-12s %-12s %-12s %-12s %-12s %-12s\n",
        "N", "direct", "packed", "tensor", "tensor_gen", "fft_asm", "mem_MiB", "err")

    rows = NamedTuple[]
    for N in Ns
        S_A = upper_bw + I_DEG + 1
        mem_est = ((J_DEG + 1) * (S_A + 1) * N + (I_DEG + 1 + upper_bw + 1) * N) * sizeof(Float64)
        mem_mib = mem_est / 2.0^20
        if mem_mib > max_memory_mib
            @warn "Skipping N due to tensor memory estimate" N mem_mib max_memory_mib
            continue
        end
        Aref, info_ref = fast_fracint_K1_decheb(;
            mu,
            N,
            omega,
            upper_bw,
            assembly=:narrow_direct_banded,
            output=:banded,
            return_info=true,
        )
        Apacked, info_packed = fast_fracint_K1_decheb(;
            mu,
            N,
            omega,
            upper_bw,
            assembly=:narrow_fft_packed_absorbM,
            output=:banded,
            return_info=true,
        )
        Atensor, info_tensor = fast_fracint_K1_decheb(;
            mu,
            N,
            omega,
            upper_bw,
            assembly=:narrow_fft_tensor_absorbM,
            output=:banded,
            return_info=true,
        )
        err = maxabs_banddiff(Atensor, Aref)
        packed_err = maxabs_banddiff(Apacked, Aref)
        @printf("%-8d %-12.6g %-12.6g %-12.6g %-12.3g %-12.3g %-12.1f %-12.3e\n",
            N,
            info_ref[:build_time_fast],
            info_packed[:build_time_fast],
            info_tensor[:build_time_fast],
            info_tensor[:time_tensor_generation],
            info_tensor[:time_fft_assembly],
            info_tensor[:memory_estimate] / 2.0^20,
            err,
        )
        if err > 1e-12
            @warn "narrow_fft_tensor_absorbM benchmark disagrees with direct reference" N err
        end
        if packed_err > 1e-12
            @warn "narrow_fft_packed_absorbM benchmark disagrees with direct reference" N packed_err
        end
        push!(rows, (;
            N,
            direct_time=info_ref[:build_time_fast],
            packed_time=info_packed[:build_time_fast],
            tensor_time=info_tensor[:build_time_fast],
            time_tensor_generation=info_tensor[:time_tensor_generation],
            time_fft_assembly=info_tensor[:time_fft_assembly],
            memory_estimate=info_tensor[:memory_estimate],
            err,
            packed_err,
        ))
    end
    return rows
end

function benchmark_fft_tensor_accum(;
    Ns=[1000, 2000, 4000, 8000],
    mu=0.5,
    omega=4.0,
    upper_bw=120,
    max_memory_mib=1200,
)
    println("===== FFT tensor accum absorbM benchmark =====")
    println("Compare narrow_direct_banded, tensor_absorbM, tensor_accum_absorbM, rfft_tensor_accum_absorbM")
    println("mu = ", mu, ", omega = ", omega, ", upper_bw = ", upper_bw)
    println()
    @printf("%-8s %-12s %-12s %-12s %-12s %-12s %-12s %-12s %-12s\n",
        "N", "direct", "tensor_old", "complex_acc", "rfft_acc", "err", "rfft_err", "fwd_fft", "freq_acc")

    rows = NamedTuple[]
    for N in Ns
        S_A = upper_bw + I_DEG + 1
        mem_est = ((S_A + 1) * (J_DEG + 1) * N + (I_DEG + 1 + upper_bw + 1) * N) * sizeof(Float64)
        mem_mib = mem_est / 2.0^20
        if mem_mib > max_memory_mib
            @warn "Skipping N due to tensor memory estimate" N mem_mib max_memory_mib
            continue
        end
        Aref, info_ref = fast_fracint_K1_decheb(;
            mu,
            N,
            omega,
            upper_bw,
            assembly=:narrow_direct_banded,
            output=:banded,
            return_info=true,
        )
        Aold, info_old = fast_fracint_K1_decheb(;
            mu,
            N,
            omega,
            upper_bw,
            assembly=:narrow_fft_tensor_absorbM,
            output=:banded,
            return_info=true,
        )
        Aacc, info_acc = fast_fracint_K1_decheb(;
            mu,
            N,
            omega,
            upper_bw,
            assembly=:narrow_fft_tensor_accum_absorbM,
            output=:banded,
            return_info=true,
        )
        Arfft, info_rfft = fast_fracint_K1_decheb(;
            mu,
            N,
            omega,
            upper_bw,
            assembly=:narrow_rfft_tensor_accum_absorbM,
            output=:banded,
            return_info=true,
        )
        err = maxabs_banddiff(Aacc, Aref)
        rfft_err = maxabs_banddiff(Arfft, Aref)
        old_err = maxabs_banddiff(Aold, Aref)
        @printf("%-8d %-12.6g %-12.6g %-12.6g %-12.6g %-12.3e %-12.3e %-12.3g %-12.3g\n",
            N,
            info_ref[:build_time_fast],
            info_old[:build_time_fast],
            info_acc[:build_time_fast],
            info_rfft[:build_time_fast],
            err,
            rfft_err,
            info_acc[:time_forward_fft],
            info_acc[:time_freq_accum],
        )
        if err > 1e-12
            @warn "narrow_fft_tensor_accum_absorbM benchmark disagrees with direct reference" N err
        end
        if old_err > 1e-12
            @warn "narrow_fft_tensor_absorbM benchmark disagrees with direct reference" N old_err
        end
        if rfft_err > 1e-12
            @warn "narrow_rfft_tensor_accum_absorbM benchmark disagrees with direct reference" N rfft_err
        end
        push!(rows, (;
            N,
            direct_time=info_ref[:build_time_fast],
            tensor_old_time=info_old[:build_time_fast],
            tensor_accum_time=info_acc[:build_time_fast],
            rfft_accum_time=info_rfft[:build_time_fast],
            err,
            rfft_err,
            old_err,
            memory_estimate=info_acc[:memory_estimate],
            time_tensor_generation=info_acc[:time_tensor_generation],
            time_forward_fft=info_acc[:time_forward_fft],
            time_freq_accum=info_acc[:time_freq_accum],
            time_inverse_fft=info_acc[:time_inverse_fft],
            time_write_A=info_acc[:time_write_A],
            time_left_corner=info_acc[:time_left_corner],
        ))
    end
    return rows
end

function benchmark_fft_tensor_accum_blocked(;
    Ns=[1000, 2000, 4000, 8000],
    block_cols_list=[64, 128, 256],
    mu=0.5,
    omega=4.0,
    upper_bw=120,
    max_memory_mib=1600,
)
    println("===== FFT tensor accum blocked benchmark =====")
    println("Compare direct, tensor_accum, tensor_accum_blocked")
    println("mu = ", mu, ", omega = ", omega, ", upper_bw = ", upper_bw)
    println()
    @printf("%-8s %-8s %-12s %-12s %-12s %-12s %-12s %-10s %-10s %-10s %-10s\n",
        "N", "block", "direct", "accum", "blocked", "err", "mem_MiB", "tensor", "fwd", "accumF", "ifft")

    rows = NamedTuple[]
    for N in Ns
        S_A = upper_bw + I_DEG + 1
        mem_est = ((S_A + 1) * (J_DEG + 1) * N + (I_DEG + 1 + upper_bw + 1) * N) * sizeof(Float64)
        mem_mib = mem_est / 2.0^20
        if mem_mib > max_memory_mib
            @warn "Skipping N due to tensor memory estimate" N mem_mib max_memory_mib
            continue
        end
        Aref, info_ref = fast_fracint_K1_decheb(;
            mu, N, omega, upper_bw,
            assembly=:narrow_direct_banded,
            output=:banded,
            return_info=true,
        )
        Aacc, info_acc = fast_fracint_K1_decheb(;
            mu, N, omega, upper_bw,
            assembly=:narrow_fft_tensor_accum_absorbM,
            output=:banded,
            return_info=true,
        )
        for block_cols in block_cols_list
            Ablk, info_blk = fast_fracint_K1_decheb(;
                mu, N, omega, upper_bw,
                assembly=:narrow_fft_tensor_accum_blocked_absorbM,
                output=:banded,
                return_info=true,
                block_cols,
            )
            err = maxabs_banddiff(Ablk, Aref)
            @printf("%-8d %-8d %-12.6g %-12.6g %-12.6g %-12.3e %-12.1f %-10.3g %-10.3g %-10.3g %-10.3g\n",
                N,
                block_cols,
                info_ref[:build_time_fast],
                info_acc[:build_time_fast],
                info_blk[:build_time_fast],
                err,
                info_blk[:memory_estimate] / 2.0^20,
                info_blk[:time_tensor_generation],
                info_blk[:time_forward_fft],
                info_blk[:time_freq_accum],
                info_blk[:time_inverse_fft],
            )
            if err > 1e-12
                @warn "blocked tensor accum disagrees with direct reference" N block_cols err
            end
            push!(rows, (;
                N,
                block_cols,
                direct_time=info_ref[:build_time_fast],
                tensor_accum_time=info_acc[:build_time_fast],
                blocked_time=info_blk[:build_time_fast],
                err,
                memory_estimate=info_blk[:memory_estimate],
                time_tensor_generation=info_blk[:time_tensor_generation],
                time_forward_fft=info_blk[:time_forward_fft],
                time_freq_accum=info_blk[:time_freq_accum],
                time_inverse_fft=info_blk[:time_inverse_fft],
                time_write_A=info_blk[:time_write_A],
            ))
        end
    end
    return rows
end

function benchmark_fft_stream_accum(;
    Ns=[400, 1000, 2000, 4000, 8000],
    mu=0.5,
    omega=4.0,
    upper_bw=120,
)
    println("===== RFFT stream accum absorbM benchmark =====")
    println("Compare direct_absorbM, rfft_tensor_accum, rfft_stream_accum")
    println("mu = ", mu, ", omega = ", omega, ", upper_bw = ", upper_bw)
    println()
    @printf("%-8s %-14s %-14s %-14s %-12s %-10s %-10s %-10s %-10s %-10s\n",
        "N", "direct_absorb", "tensor_rfft", "stream_rfft", "err", "mem_MiB", "Hgen", "fwd", "accum", "ifft")

    rows = NamedTuple[]
    for N in Ns
        Aref, info_ref = fast_fracint_K1_decheb(;
            mu, N, omega, upper_bw,
            assembly=:narrow_direct_absorbM,
            output=:banded,
            return_info=true,
        )
        Atensor, info_tensor = fast_fracint_K1_decheb(;
            mu, N, omega, upper_bw,
            assembly=:narrow_rfft_tensor_accum_absorbM,
            output=:banded,
            return_info=true,
        )
        Astream, info_stream = fast_fracint_K1_decheb(;
            mu, N, omega, upper_bw,
            assembly=:narrow_rfft_stream_accum_absorbM,
            output=:banded,
            return_info=true,
        )
        err = maxabs_banddiff(Astream, Aref)
        tensor_err = maxabs_banddiff(Atensor, Aref)
        @printf("%-8d %-14.6g %-14.6g %-14.6g %-12.3e %-10.1f %-10.3g %-10.3g %-10.3g %-10.3g\n",
            N,
            info_ref[:build_time_fast],
            info_tensor[:build_time_fast],
            info_stream[:build_time_fast],
            err,
            info_stream[:memory_estimate] / 2.0^20,
            info_stream[:time_H_generation_stream],
            info_stream[:time_forward_fft],
            info_stream[:time_freq_accum],
            info_stream[:time_inverse_fft],
        )
        if err > 1e-12
            @warn "stream rFFT disagrees with direct_absorbM" N err
        end
        if tensor_err > 1e-12
            @warn "tensor rFFT disagrees with direct_absorbM" N tensor_err
        end
        push!(rows, (;
            N,
            direct_absorb_time=info_ref[:build_time_fast],
            tensor_rfft_time=info_tensor[:build_time_fast],
            stream_rfft_time=info_stream[:build_time_fast],
            err,
            memory_estimate=info_stream[:memory_estimate],
            time_H_generation_stream=info_stream[:time_H_generation_stream],
            time_forward_fft=info_stream[:time_forward_fft],
            time_freq_accum=info_stream[:time_freq_accum],
            time_inverse_fft=info_stream[:time_inverse_fft],
            time_left_corner=info_stream[:time_left_corner],
        ))
    end
    return rows
end

function benchmark_fft_rfft_blocked_absorbM(;
    Ns=[400, 1000, 2000, 4000],
    block_cols_list=[64, 128],
    mu=0.5,
    omega=4.0,
    upper_bw=120,
    fftw_flags=nothing,
)
    println("===== Blocked batched RFFT absorbM benchmark =====")
    println("Compare direct_absorbM, old rfft_stream stats path, new rfft_blocked")
    println("mu = ", mu, ", omega = ", omega, ", upper_bw = ", upper_bw)
    println()
    @printf("%-8s %-8s %-14s %-14s %-14s %-12s %-10s %-10s %-10s %-10s\n",
        "N", "block", "direct_absorb", "old_stream", "rfft_block", "err", "fillQ", "fwd", "accum", "ifft")

    rows = NamedTuple[]
    for N in Ns
        Aref, info_ref = fast_fracint_K1_decheb(;
            mu, N, omega, upper_bw,
            assembly=:narrow_direct_absorbM,
            output=:banded,
            return_info=true,
        )
        Aold, info_old = fast_fracint_K1_decheb(;
            mu, N, omega, upper_bw,
            assembly=:narrow_rfft_stream_accum_absorbM,
            output=:banded,
            return_info=true,
        )
        old_err = maxabs_banddiff(Aold, Aref)
        for block_cols in block_cols_list
            Ablk, info_blk = fast_fracint_K1_decheb(;
                mu, N, omega, upper_bw,
                assembly=:narrow_rfft_tensor_accum_blocked_absorbM,
                output=:banded,
                return_info=true,
                block_cols,
                fftw_flags,
            )
            err = maxabs_banddiff(Ablk, Aref)
            @printf("%-8d %-8d %-14.6g %-14.6g %-14.6g %-12.3e %-10.3g %-10.3g %-10.3g %-10.3g\n",
                N,
                block_cols,
                info_ref[:build_time_fast],
                info_old[:build_time_fast],
                info_blk[:build_time_fast],
                err,
                info_blk[:time_fill_Qmat],
                info_blk[:time_forward_fft],
                info_blk[:time_freq_accum],
                info_blk[:time_inverse_fft],
            )
            if err > 1e-12
                @warn "blocked batched RFFT disagrees with direct_absorbM" N block_cols err
            end
            if old_err > 1e-12
                @warn "old stream RFFT disagrees with direct_absorbM" N old_err
            end
            push!(rows, (;
                N,
                block_cols,
                direct_absorb_time=info_ref[:build_time_fast],
                old_stream_time=info_old[:build_time_fast],
                rfft_blocked_time=info_blk[:build_time_fast],
                err,
                old_err,
                memory_estimate=info_blk[:memory_estimate],
                time_fill_Qmat=info_blk[:time_fill_Qmat],
                time_forward_fft=info_blk[:time_forward_fft],
                time_freq_accum=info_blk[:time_freq_accum],
                time_inverse_fft=info_blk[:time_inverse_fft],
                time_write_A=info_blk[:time_write_A],
                time_left_corner=info_blk[:time_left_corner],
            ))
        end
    end
    return rows
end

function _find_benchmark_row(rows, N::Int, upper_bw::Int)
    for row in rows
        row.N == N && row.upper_bw == upper_bw && return row
    end
    return nothing
end

function _first_stable_fft_region(rows, Ns, upper_bws; speedup_threshold=1.10, err_tol=1e-12)
    good(row) = row !== nothing && row.err <= err_tol && row.speedup >= speedup_threshold
    for N in Ns
        for idx in 1:(length(upper_bws) - 1)
            ub1 = upper_bws[idx]
            ub2 = upper_bws[idx + 1]
            row1 = _find_benchmark_row(rows, N, ub1)
            row2 = _find_benchmark_row(rows, N, ub2)
            if good(row1) && good(row2)
                return (; kind=:adjacent_upper_bw, N, upper_bws=(ub1, ub2),
                    speedups=(row1.speedup, row2.speedup))
            end
        end
    end
    for upper_bw in upper_bws
        for idx in 1:(length(Ns) - 1)
            N1 = Ns[idx]
            N2 = Ns[idx + 1]
            row1 = _find_benchmark_row(rows, N1, upper_bw)
            row2 = _find_benchmark_row(rows, N2, upper_bw)
            if good(row1) && good(row2)
                return (; kind=:adjacent_N, Ns=(N1, N2), upper_bw,
                    speedups=(row1.speedup, row2.speedup))
            end
        end
    end
    return nothing
end

function benchmark_fft_plan_vs_assembly(;
    Ns=[400, 1000, 2000, 4000],
    upper_bws=[40, 80, 120, 200, 300],
    mu=0.5,
    omega=4.0,
    block_cols::Int=64,
    fftw_flags=nothing,
    speedup_threshold::Float64=1.10,
    err_tol::Float64=1e-12,
)
    println("===== FFT plan vs assembly benchmark =====")
    println("Compare narrow_direct_absorbM and narrow_rfft_tensor_accum_blocked_absorbM")
    println("mu = ", mu, ", omega = ", omega, ", block_cols = ", block_cols)
    println("Ns = ", Ns, ", upper_bws = ", upper_bws)
    println()
    @printf("%-7s %-8s %-11s %-11s %-11s %-11s %-11s %-11s %-8s %-10s %-9s %-9s %-9s %-10s %-10s %-10s\n",
        "N", "ubw", "direct", "plan", "asm_noinfo", "asm_info", "fast_noinf",
        "fast_info", "speedup", "err", "fillQ", "freq", "left", "asm_ovh", "fast_ovh", "memMiB")

    rows = NamedTuple[]
    max_asm_info_ratio = 0.0
    max_fast_info_ratio = 0.0
    for N in Ns
        for upper_bw in upper_bws
            A_direct, info_direct = fast_fracint_K1_decheb(;
                mu, N, omega, upper_bw,
                assembly=:narrow_direct_absorbM,
                output=:banded,
                return_info=true,
            )

            local plan_noinfo
            plan_time = @elapsed begin
                plan_noinfo = plan_fracint_K1_fft(;
                    mu, N, omega, upper_bw,
                    assembly=:narrow_rfft_tensor_accum_blocked_absorbM,
                    block_cols,
                    fftw_flags,
                )
            end
            Aband_noinfo = zeros(Float64, plan_noinfo.lower_bw_A + upper_bw + 1, N)
            assemble_time_noinfo = @elapsed begin
                assemble_fracint_K1_fft!(Aband_noinfo, plan_noinfo; return_stats=false)
            end

            plan_info = plan_fracint_K1_fft(;
                mu, N, omega, upper_bw,
                assembly=:narrow_rfft_tensor_accum_blocked_absorbM,
                block_cols,
                fftw_flags,
            )
            Aband_info = zeros(Float64, plan_info.lower_bw_A + upper_bw + 1, N)
            local stats_info
            assemble_time_info = @elapsed begin
                _, stats_info = assemble_fracint_K1_fft!(Aband_info, plan_info; return_stats=true)
            end

            local Afast_noinfo
            fast_time_noinfo = @elapsed begin
                Afast_noinfo = fast_fracint_K1_decheb(;
                    mu, N, omega, upper_bw,
                    assembly=:narrow_rfft_tensor_accum_blocked_absorbM,
                    output=:banded,
                    return_info=false,
                    block_cols,
                    fftw_flags,
                )
            end

            local Afast_info, info_fast
            fast_time_info = @elapsed begin
                Afast_info, info_fast = fast_fracint_K1_decheb(;
                    mu, N, omega, upper_bw,
                    assembly=:narrow_rfft_tensor_accum_blocked_absorbM,
                    output=:banded,
                    return_info=true,
                    block_cols,
                    fftw_flags,
                )
            end

            err = max(
                maxabs_banddiff(Aband_noinfo, A_direct),
                maxabs_banddiff(Aband_info, A_direct),
                maxabs_banddiff(Afast_noinfo, A_direct),
                maxabs_banddiff(Afast_info, A_direct),
            )
            time_direct = info_direct[:build_time_fast]
            time_fft = fast_time_noinfo
            speedup = time_fft == 0.0 ? Inf : time_direct / time_fft
            asm_info_ratio = assemble_time_noinfo == 0.0 ? Inf : assemble_time_info / assemble_time_noinfo
            fast_info_ratio = fast_time_noinfo == 0.0 ? Inf : fast_time_info / fast_time_noinfo
            max_asm_info_ratio = max(max_asm_info_ratio, asm_info_ratio)
            max_fast_info_ratio = max(max_fast_info_ratio, fast_info_ratio)
            memory_mib = info_fast[:memory_estimate] / 2.0^20

            @printf("%-7d %-8d %-11.4g %-11.4g %-11.4g %-11.4g %-11.4g %-11.4g %-8.3f %-10.3e %-9.3g %-9.3g %-9.3g %-10.3f %-10.3f %-10.1f\n",
                N, upper_bw, time_direct, plan_time, assemble_time_noinfo,
                assemble_time_info, fast_time_noinfo, fast_time_info, speedup, err,
                stats_info.time_fill_Qmat, stats_info.time_freq_accum, stats_info.time_left_corner,
                asm_info_ratio, fast_info_ratio, memory_mib)
            if err > err_tol
                @warn "FFT blocked path disagrees with direct_absorbM" N upper_bw err err_tol
            end

            push!(rows, (;
                N,
                upper_bw,
                time_direct,
                plan_time,
                assemble_time_noinfo,
                assemble_time_info,
                fast_time_noinfo,
                fast_time_info,
                time_fft,
                speedup,
                err,
                asm_info_ratio,
                fast_info_ratio,
                memory_estimate=info_fast[:memory_estimate],
                time_fill_Qmat=stats_info.time_fill_Qmat,
                time_freq_accum=stats_info.time_freq_accum,
                time_left_corner=stats_info.time_left_corner,
                stats_info,
            ))
        end
    end

    stable_region = _first_stable_fft_region(
        rows,
        Ns,
        upper_bws;
        speedup_threshold,
        err_tol,
    )
    println()
    println("===== Summary =====")
    @printf("max assemble return_info overhead ratio: %.3f\n", max_asm_info_ratio)
    @printf("max fast return_info overhead ratio: %.3f\n", max_fast_info_ratio)
    if max_asm_info_ratio <= 1.05
        println("return_info=true adds no material assemble-only timing overhead in this run.")
    else
        println("return_info=true has measurable assemble-only overhead in this run.")
    end
    if stable_region === nothing
        println("No stable FFT-winning region found under speedup >= ", speedup_threshold,
            " and err <= ", err_tol, ".")
        println("FFT is not currently worthwhile as the default; assembly=:auto should remain narrow_direct_absorbM.")
    elseif stable_region.kind === :adjacent_upper_bw
        println("First stable FFT-winning region: N = ", stable_region.N,
            ", adjacent upper_bw = ", stable_region.upper_bws,
            ", speedups = ", stable_region.speedups, ".")
        println("Use this benchmark result before changing assembly=:auto away from narrow_direct_absorbM.")
    else
        println("First stable FFT-winning region: upper_bw = ", stable_region.upper_bw,
            ", adjacent Ns = ", stable_region.Ns,
            ", speedups = ", stable_region.speedups, ".")
        println("Use this benchmark result before changing assembly=:auto away from narrow_direct_absorbM.")
    end
    println("Default path recommendation: keep narrow_direct_absorbM unless the stable FFT region above is confirmed on the target machine.")
    return (; rows, stable_region, max_asm_info_ratio, max_fast_info_ratio)
end

function benchmark_mu_sweep(;
    mus=[0.25, 0.5, 0.75],
    Ns=[400, 800, 1200, 2000],
    omega=4.0,
    upper_bw=120,
    compare_N::Int=400,
)
    println("===== Mu sweep: narrow_direct_banded =====")
    println("output = banded, omega = ", omega, ", upper_bw = ", upper_bw)
    println()
    @printf("%-8s %-8s %-14s %-10s %-10s %-12s\n",
        "mu", "N", "time", "finite", "max_abs", "direct_err")

    rows = NamedTuple[]
    for mu in mus
        for N in Ns
            Aband, info = fast_fracint_K1_decheb(;
                mu,
                N,
                omega,
                upper_bw,
                assembly=:narrow_direct_banded,
                output=:banded,
                return_info=true,
            )
            direct_err = NaN
            if N <= compare_N
                Aref, _ = fast_fracint_K1_decheb(;
                    mu,
                    N,
                    omega,
                    upper_bw,
                    assembly=:direct_convolution,
                    output=:banded,
                    return_info=true,
                )
                direct_err = maxabs_banddiff(Aband, Aref)
            end
            @printf("%-8.3g %-8d %-14.6g %-10s %-10.3e %-12.3e\n",
                Float64(mu),
                N,
                info[:build_time_fast],
                string(info[:finite_check]),
                info[:max_abs_A],
                direct_err,
            )
            if !info[:finite_check]
                @warn "narrow_direct_banded produced non-finite output" mu N
            end
            if isfinite(direct_err) && direct_err > 1e-10
                @warn "narrow_direct_banded disagrees with direct_convolution in mu sweep" mu N direct_err
            end
            push!(rows, (;
                mu=Float64(mu),
                N,
                time=info[:build_time_fast],
                finite=info[:finite_check],
                max_abs=info[:max_abs_A],
                direct_err,
            ))
        end
    end
    return rows
end

function compare_with_old(; mu=0.5, omega=4.0, Ns=[96, 160, 256, 400], upper_bw=120)
    println("===== Fast K=1 DE weak Volterra banded construction =====")
    println("C size fixed: 100 x 100")
    println("mu = ", mu, ", omega = ", omega)
    println("upper_bw = ", upper_bw)
    println("assembly = narrow_direct_banded, output = banded")
    println()
    @printf("%-6s %-10s %-10s %-8s %-10s %-12s %-12s %-14s %-12s %-12s\n",
        "N", "old_time", "fast_time", "speedup", "kept_frac", "band_err", "full_err", "outside_old", "fft_dir_err", "relF_band")

    rows = NamedTuple[]
    for N in Ns
        local Aold, info_old
        old_time = @elapsed begin
            Aold, info_old = Weak_VoltMat_DE.weak_voltmat_decheb(
                (x, t) -> 1.0;
                mu,
                N,
                omega,
                Csize_y,
                Csize_eta,
                return_info=true,
            )
        end

        Afast_raw, info_fast = fast_fracint_K1_decheb(;
            mu,
            N,
            omega,
            upper_bw,
            return_info=true,
            assembly=:narrow_direct_banded,
            output=:banded,
            check_fft=false,
        )
        Afast = band_to_dense(Afast_raw, N, info_fast[:lower_bw], info_fast[:upper_bw])

        mask = kept_band_mask(N, upper_bw)
        kept_fraction = count(mask) / length(mask)
        diff = Afast .- Aold
        band_err = maximum(abs, diff[mask])
        full_err = maximum(abs, diff)
        outside = .!mask
        old_outside = any(outside) ? maximum(abs, Aold[outside]) : 0.0
        denom = norm(Aold[mask])
        relF = denom == 0.0 ? 0.0 : norm(diff[mask]) / denom
        fast_time = info_fast[:build_time_fast]
        speedup = fast_time == 0.0 ? Inf : old_time / fast_time
        fft_vs_direct_err = info_fast[:fft_vs_direct_err] === nothing ? NaN : info_fast[:fft_vs_direct_err]

        @printf("%-6d %-10.3g %-10.3g %-8.2f %-10.3f %-12.3e %-12.3e %-14.3e %-12.3e %-12.3e\n",
            N, old_time, fast_time, speedup, kept_fraction, band_err, full_err, old_outside, fft_vs_direct_err, relF)

        if !info_fast[:finite_check]
            @warn "A_fast contains non-finite entries" N
        end
        if kept_fraction > 0.95
            @warn "band test is trivial for this N; use larger N." N kept_fraction
        end
        if band_err > 1e-8
            @warn "Kept-band error is larger than expected; inspect H recurrence or assembly indexing." N band_err
        end
        if fft_vs_direct_err > 1e-8
            @warn "FFT convolution disagrees with direct scatter." N fft_vs_direct_err
        end
        if old_outside > 1e-10
            @warn "Old matrix has visible mass outside the kept band; upper_bw may need to increase." N old_outside
        end

        push!(rows, (;
            N,
            old_time,
            fast_time,
            speedup,
            kept_fraction,
            band_err,
            full_err,
            old_outside,
            fft_vs_direct_err,
            relF,
            finite=info_fast[:finite_check],
        ))
        _ = info_old
    end
    return rows
end

end # module FastFracIntDE

if abspath(PROGRAM_FILE) == @__FILE__
    using .FastFracIntDE
    FastFracIntDE.compare_with_old()
end
