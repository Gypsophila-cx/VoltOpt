# voltmat_cheb.jl
#
# Experimental single-thread FFT-convolution assembly for the Chebyshev Volterra
# matrix.
#
# The default Chebyshev assembly path is the FFT-based Toeplitz--Hankel
# implementation. The compressed no-FFT paths are retained for validation and
# experimentation and must be requested explicitly with the `assembly` keyword.
#
# Goal:
#   Reduce the assembly dependence on the absorbed kernel width from roughly
#
#       O((3n + 2m) * P * N),    P ≈ m(n+1),
#
#   to approximately
#
#       O(m * N * (n+m) log(n+m))
#
#   for the bottom banded block, and optionally also reduce the top dense block
#   from O(m n^2 N) to O(m n log(n) N).
#
# Notes:
#   1. This is a theoretically faster version for large n,m. For small n,m the
#      previous direct single-thread version may still be faster because FFT
#      constants are not small.
#   2. FFTW is explicitly forced to one thread.
#   3. The dense tensor is stored in transposed layout:
#
#          h_dense[j][k+1, s+1]
#
#      exactly as in the previous optimized same-name version.
#   4. This square version outputs an (N+1) x (N+1) matrix only.
#      It sets
#
#          L_max = K_max = N,
#
#      so columns k > N are never assembled.  The bottom banded block is also
#      assembled with a forced symmetric half-bandwidth.  By default the
#      half-bandwidth is n, i.e. r = k-l is restricted to
#
#          -n <= k-l <= n.
#
#      This intentionally truncates the larger positive-r tail of the full
#      rectangular matrix and omits every entry outside the prescribed square.
#   5. The FFT assembly introduces roundoff at roughly FFT precision level, so
#      exact bitwise equality with the direct assembly should not be expected.
#
# Main interfaces:
#
#     B = voltmat_cheb(K, N; ...)
#     B = voltmat_cheb(C, N; ...)
#
# The output is a square SparseMatrixCSC of size (N+1) x (N+1).  The bottom
# block stores only the symmetric band
#
#     -tail_half_bandwidth <= k-l <= tail_half_bandwidth,
#
# with default half-bandwidth size(C,1), or the adaptively determined first
# dimension of C when K is supplied.
#
# K usage:
#   K is a callable original Volterra kernel K(x,t) on the triangular domain
#   -1 <= t <= x <= 1.  The constructor first forms the absorbed tensor kernel
#
#       Khat(x,y) = K(x, (1+x)*(1+y)/2 - 1),
#
#   computes its Chebyshev coefficient matrix by adaptive_cheb2_coeffmat, and
#   then assembles the Volterra matrix.
#
# C usage:
#   C is a precomputed Chebyshev coefficient matrix for the same absorbed
#   kernel Khat, not for the original K(x,t) directly:
#
#       Khat(x,y) ~= sum C[i+1,j+1] * T_i(x) * T_j(y).
#
#   Use voltmat_cheb(C, N; ...) when this coefficient matrix has already been
#   computed or when the interpolation/chopping of Khat must be controlled
#   outside this constructor.  In this path no kernel function is sampled.

module VoltMat_Cheb

using SparseArrays
using SpecialFunctions
using FFTW
using LinearAlgebra

FFTW.set_num_threads(1)

include("../chebtechs.jl")


# =============================================================================
# Utility
# =============================================================================

@inline function _nextpow2_len(n::Int)
    p = 1
    while p < n
        p <<= 1
    end
    return p
end


@inline function safe_gamma_ratio(a::Float64, b::Float64, c::Float64, d::Float64)
    lga, sga = logabsgamma(a)
    lgb, sgb = logabsgamma(b)
    lgc, sgc = logabsgamma(c)
    lgd, sgd = logabsgamma(d)
    return (sga * sgb * sgc * sgd) * exp(lga + lgb - lgc - lgd)
end


@inline function H_zero_exact_dense(j::Int, k::Int)
    if k > j + 1 || (k == 0 && isodd(j))
        return 0.0
    end

    if iseven(j) && (k == 0 || k == 1)
        sgn = isodd(k + 1) ? -1.0 : 1.0
        return sgn * 2.0^(1 - k) / (j^2 - 1.0)
    end

    if k >= 1 && j >= k && iseven(j + k)
        return -(j / 8.0) * safe_gamma_ratio(
            (j + k) / 2.0,
            (j - k - 1.0) / 2.0,
            (j - k + 2.0) / 2.0,
            (j + k + 3.0) / 2.0,
        )
    end

    if k >= 2 && j >= k - 1 && isodd(j + k)
        return (j / 8.0) * safe_gamma_ratio(
            (j + k - 1.0) / 2.0,
            (j - k) / 2.0,
            (j - k + 3.0) / 2.0,
            (j + k + 2.0) / 2.0,
        )
    end

    return 0.0
end


# =============================================================================
# Module 1: dense part generator
#
# Return layout:
#     h_dense[k+1, s+1]
# =============================================================================

function generate_dense_top_rows(j::Int, n::Int, K_max::Int)
    Kp = K_max + 1
    ns = 2 * n

    h_dense = zeros(Float64, Kp, ns)

    if ns == 0
        return h_dense
    end

    H_prev = zeros(Float64, Kp)
    H_curr = zeros(Float64, Kp)

    k_stop = min(K_max, j + 1)

    @inbounds for k in 0:k_stop
        H_prev[k + 1] = H_zero_exact_dense(j, k)
    end

    @inbounds @simd for k1 in 1:Kp
        h_dense[k1, 1] = H_prev[k1]
    end

    @inbounds for s in 0:(2 * n - 2)
        ss = s + 1

        # The recurrence starts at k = ss.  In the square version K_max can be
        # smaller than 2n, so guard the seed entry.  Columns beyond the available
        # k-range stay zero, which is exactly what is needed because they are
        # outside the requested square.
        if ss <= K_max
            H_curr[ss + 1] = H_prev[ss + 1] / (2.0 * s + 1.0) + H_prev[ss]
        end

        for k in (ss + 1):K_max
            denom = Float64(k + s)
            H_curr[k + 1] =
                ((k - s) / denom) * H_prev[k + 1] +
                H_prev[k] -
                ((k - s - 2) / denom) * H_curr[k]
        end

        col = ss + 1

        for k in ss:K_max
            h_dense[k + 1, col] = (k / (ss + k)) * H_curr[k + 1]
        end

        H_prev, H_curr = H_curr, H_prev
    end

    return h_dense
end


function generate_dense_blocks(n::Int, m_total::Int, K_max::Int)
    tensor_dense = Vector{Matrix{Float64}}(undef, m_total)

    @inbounds for j in 0:(m_total - 1)
        tensor_dense[j + 1] = generate_dense_top_rows(j, n, K_max)
    end

    return tensor_dense
end


# =============================================================================
# Module 2: banded tail generator
#
# Return layout:
#     h_banded[s - 2n + 1, r + 1]
# =============================================================================

function build_base_H_banded!(
    H_mat::Matrix{Float64},
    j::Int,
    n::Int,
    S_max::Int,
    max_r::Int,
)
    s_start = 2 * n
    num_rows = size(H_mat, 1)

    if num_rows <= 0
        return nothing
    end

    if j == 0
        vals = Vector{Float64}(undef, num_rows)

        @inbounds @simd for s_idx in 1:num_rows
            s = Float64(s_start + s_idx - 1)
            val = 1.0 / (s + 1.0)

            vals[s_idx] = val
            H_mat[s_idx, 1] = 2.0 * val
        end

        @inbounds for r in 1:max_r
            rf = Float64(r)
            col = r + 1

            @simd for s_idx in 1:num_rows
                s = Float64(s_start + s_idx - 1)
                val = vals[s_idx]

                if r == 1
                    val = val / (s + 2.0)
                else
                    val = val * (2.0 - s - rf) / (s + 1.0 + rf)
                end

                vals[s_idx] = val
                H_mat[s_idx, col] = -2.0 * (rf + 2.0 * s) * val
            end
        end

    else
        vals0 = Vector{Float64}(undef, num_rows)
        vals1 = Vector{Float64}(undef, num_rows)

        @inbounds @simd for s_idx in 1:num_rows
            s = Float64(s_start + s_idx - 1)

            val0 = 1.0 / (s + 1.0)
            val1 = 1.0 / (s + 2.0)

            vals0[s_idx] = val0
            vals1[s_idx] = val1

            H_mat[s_idx, 1] = 2.0 * (-val0 + 2.0 * val1)
        end

        @inbounds for r in 1:max_r
            rf = Float64(r)
            col = r + 1

            @simd for s_idx in 1:num_rows
                s = Float64(s_start + s_idx - 1)

                val0 = vals0[s_idx]
                val1 = vals1[s_idx]

                if r == 1
                    val0 = val0 / (s + 2.0)
                    val1 = val1 / (s + 3.0)
                else
                    val0 = val0 * (2.0 - s - rf) / (s + 1.0 + rf)
                    val1 = val1 * (3.0 - s - rf) / (s + 2.0 + rf)
                end

                vals0[s_idx] = val0
                vals1[s_idx] = val1

                H_mat[s_idx, col] = -2.0 * (rf + 2.0 * s) * (-val0 + 2.0 * val1)
            end
        end
    end

    return nothing
end


@inline function _copy_scaled_h!(
    dest::Matrix{Float64},
    src::Matrix{Float64},
    n::Int,
    r_target::Int,
)
    s_start = 2 * n
    num_rows = size(dest, 1)

    @inbounds for r in 0:r_target
        rf = Float64(r)
        col = r + 1

        @simd for s_idx in 1:num_rows
            s = Float64(s_start + s_idx - 1)
            denom = 2.0 * s + rf
            scale = denom == 0.0 ? 1.0 : (s + rf) / denom

            dest[s_idx, col] = scale * src[s_idx, col]
        end
    end

    return nothing
end


function generate_banded_blocks(
    n::Int,
    m_total::Int,
    S_max::Int;
    r_target_override::Union{Nothing,Int} = nothing,
)
    # Thin wrapper only.  Do not put the recurrence here.
    # The old keyword-version put the whole hot recurrence in this method and
    # could trigger massive scalar allocations.  All hot loops are now in the
    # concrete-typed core below.
    r_full_target::Int = 2 * n + m_total - 1

    r_target::Int = if r_target_override === nothing
        r_full_target
    else
        if r_target_override < 0
            throw(ArgumentError("r_target_override must be nonnegative"))
        end
        min(Int(r_target_override), r_full_target)
    end

    return _generate_banded_blocks_core(n, m_total, S_max, r_target)
end


function _generate_banded_blocks_core(
    n::Int,
    m_total::Int,
    S_max::Int,
    r_target::Int,
)
    s_start::Int = 2 * n
    num_rows::Int = S_max - s_start + 1

    if r_target < 0
        throw(ArgumentError("r_target must be nonnegative"))
    end

    if num_rows <= 0
        return [zeros(Float64, 0, r_target + 1) for _ in 1:m_total]
    end

    h_tensor = [zeros(Float64, num_rows, r_target + 1) for _ in 1:m_total]

    # The Chebyshev recurrence in j requires a look-ahead in the offset index.
    max_r_needed::Int = r_target + m_total - 1

    H_prev = zeros(Float64, num_rows, max_r_needed + 1)
    H_curr = zeros(Float64, num_rows, max_r_needed + 1)
    H_next = zeros(Float64, num_rows, max_r_needed + 1)

    build_base_H_banded!(H_prev, 0, n, S_max, max_r_needed)
    _copy_scaled_h!(h_tensor[1], H_prev, n, r_target)

    if m_total >= 2
        build_base_H_banded!(H_curr, 1, n, S_max, max_r_needed)
        _copy_scaled_h!(h_tensor[2], H_curr, n, r_target)
    end

    if m_total <= 2
        return h_tensor
    end

    @inbounds for step::Int in 1:(m_total - 2)
        curr_j::Int = step + 1
        curr_r_max::Int = r_target + m_total - 1 - curr_j

        # Important: write by columns for Julia's column-major layout.
        for r::Int in 0:curr_r_max
            rf::Float64 = Float64(r)
            col::Int = r + 1

            if r == 0
                @simd for s_idx in 1:num_rows
                    s::Float64 = Float64(s_start + s_idx - 1)

                    A_r::Float64 =
                        (rf + 1.0) * (rf + 2.0 * s) /
                        ((rf + s) * (2.0 * rf + 2.0 * s + 1.0))

                    B_r::Float64 =
                        (4.0 * s * s - 1.0) /
                        ((2.0 * rf + 2.0 * s - 1.0) *
                         (2.0 * rf + 2.0 * s + 1.0))

                    H_next[s_idx, col] =
                        2.0 * A_r * H_curr[s_idx, 2] +
                        2.0 * B_r * H_curr[s_idx, 1] -
                        H_prev[s_idx, 1]
                end
            else
                @simd for s_idx in 1:num_rows
                    s::Float64 = Float64(s_start + s_idx - 1)

                    A_r::Float64 =
                        (rf + 1.0) * (rf + 2.0 * s) /
                        ((rf + s) * (2.0 * rf + 2.0 * s + 1.0))

                    B_r::Float64 =
                        (4.0 * s * s - 1.0) /
                        ((2.0 * rf + 2.0 * s - 1.0) *
                         (2.0 * rf + 2.0 * s + 1.0))

                    C_r::Float64 =
                        (rf - 1.0) * (rf + 2.0 * s) /
                        ((2.0 * rf + 2.0 * s - 1.0) * (rf + s))

                    H_next[s_idx, col] =
                        2.0 * A_r * H_curr[s_idx, col + 1] +
                        2.0 * B_r * H_curr[s_idx, col] +
                        2.0 * C_r * H_curr[s_idx, col - 1] -
                        H_prev[s_idx, col]
                end
            end
        end

        j_idx::Int = curr_j + 1
        _copy_scaled_h!(h_tensor[j_idx], H_next, n, r_target)

        H_prev, H_curr, H_next = H_curr, H_next, H_prev
    end

    return h_tensor
end


# =============================================================================
# P-coefficients after absorbing M
# =============================================================================

struct PBlock
    ii::Vector{Int}          # nonzero 0-based i
    vv::Vector{Float64}      # corresponding values
    dense::Vector{Float64}   # dense length n+1 vector
end


function _build_postmult_coeffs(C::Matrix{Float64})
    n, m = size(C)

    P_coeffs = zeros(Float64, n + 1, m)

    @inbounds for j in 1:m
        for i0 in 0:(n - 1)
            c_i = C[i0 + 1, j]

            if c_i == 0.0
                continue
            end

            P_coeffs[i0 + 1, j] += 0.5 * c_i

            if i0 >= 1
                P_coeffs[i0, j] += 0.25 * c_i
            else
                P_coeffs[2, j] += 0.25 * c_i
            end

            P_coeffs[i0 + 2, j] += 0.25 * c_i
        end
    end

    return P_coeffs
end


function _build_P_blocks(C::Matrix{Float64})
    n, m = size(C)
    P_coeffs = _build_postmult_coeffs(C)

    pblocks = Vector{PBlock}(undef, m)

    @inbounds for j in 1:m
        p_is = Int[]
        p_vs = Float64[]
        p_dense = zeros(Float64, n + 1)

        sizehint!(p_is, n + 1)
        sizehint!(p_vs, n + 1)

        for i1 in 1:(n + 1)
            P_val = P_coeffs[i1, j]
            p_dense[i1] = P_val

            if P_val != 0.0
                push!(p_is, i1 - 1)
                push!(p_vs, P_val)
            end
        end

        pblocks[j] = PBlock(p_is, p_vs, p_dense)
    end

    return pblocks
end


# =============================================================================
# FFT preparation
# =============================================================================

function _precompute_packed_spectra(
    pblocks::Vector{PBlock},
    n::Int,
    Lfft::Int,
    plan_fwd,
)
    m = length(pblocks)

    Fpack = Vector{Vector{ComplexF64}}(undef, m)

    buf = zeros(ComplexF64, Lfft)
    Fp = zeros(ComplexF64, Lfft)
    Fr = zeros(ComplexF64, Lfft)

    @inbounds for j in 1:m
        pb = pblocks[j]

        fill!(buf, 0.0 + 0.0im)

        for q in eachindex(pb.ii)
            i = pb.ii[q]
            buf[i + 1] = pb.vv[q]
        end

        mul!(Fp, plan_fwd, buf)

        fill!(buf, 0.0 + 0.0im)

        # reversed p over length n+1:
        #   p_rev[t] = p[n-t], t=0,...,n
        for q in eachindex(pb.ii)
            i = pb.ii[q]
            buf[n - i + 1] = pb.vv[q]
        end

        mul!(Fr, plan_fwd, buf)

        F = Vector{ComplexF64}(undef, Lfft)

        for t in 1:Lfft
            # Packing trick:
            #   ifft((Fp + im*Fr) .* Fg)
            # gives
            #   real part = conv(p,g),
            #   imag part = conv(reverse(p),g).
            F[t] = Fp[t] + im * Fr[t]
        end

        Fpack[j] = F
    end

    return Fpack
end


@inline function _fft_packed_convolution!(
    seq::Vector{ComplexF64},
    work::Vector{ComplexF64},
    g_fft::Vector{ComplexF64},
    g_buf::Vector{ComplexF64},
    Fpack_j::Vector{ComplexF64},
    plan_fwd,
    plan_inv,
)
    mul!(g_fft, plan_fwd, g_buf)

    @inbounds @simd for t in eachindex(work)
        work[t] = Fpack_j[t] * g_fft[t]
    end

    mul!(seq, plan_inv, work)

    return nothing
end


# =============================================================================
# FFT assembly
# =============================================================================

function assemble_VK_sparse_block_fft_convolution(
    C::Matrix{Float64},
    h_dense::Vector{Matrix{Float64}},
    h_banded::Vector{Matrix{Float64}},
    L_max::Int,
    K_max::Int;
    tol::Float64 = eps(Float64),
    use_fft_top::Bool = true,
    tail_half_bandwidth::Union{Nothing,Int} = nothing,
)
    n, m = size(C)

    if L_max < 0 || K_max < 0
        throw(ArgumentError("L_max and K_max must be nonnegative"))
    end

    Kp = K_max + 1
    r_target = 2 * n + m - 1

    # Only the first n rows form the dense top block, but for a small requested
    # square we must not assemble rows ell > L_max.
    n_top = min(n, L_max + 1)

    # Full formula has an asymmetric tail: roughly -n <= k-l <= n+m-1.
    # To force equal upper/lower bandwidth in the bottom block, we use the same
    # half-bandwidth on both sides.  The default requested half-bandwidth is n.
    # The effective bandwidth is clipped to the square dimensions and r_target,
    # so loops never spend time on entries that cannot lie inside the output.
    r_full_upper = n + m - 1
    r_full_lower = n
    r_sym_requested = tail_half_bandwidth === nothing ? r_full_lower : tail_half_bandwidth

    if r_sym_requested < 0
        throw(ArgumentError("tail_half_bandwidth must be nonnegative"))
    end

    r_sym = min(r_sym_requested, L_max, K_max, r_target)

    if r_sym_requested > r_full_upper
        @warn "tail_half_bandwidth exceeds the full positive bandwidth; extra stored entries are structurally zero" r_sym_requested r_full_upper
    end

    pblocks = _build_P_blocks(C)

    # -------------------------------------------------------------------------
    # Sparse triplet storage
    # -------------------------------------------------------------------------
    estimated_nnz =
        n_top * (K_max + 1) +
        max(0, L_max - n + 1) * (2 * r_sym + 1)

    I = Int[]
    J = Int[]
    V = Float64[]

    sizehint!(I, estimated_nnz)
    sizehint!(J, estimated_nnz)
    sizehint!(V, estimated_nnz)

    # -------------------------------------------------------------------------
    # 1. Top dense block
    #
    # For fixed (j,k), define
    #     g[s] = h_j(k-s, s) = h_dense[j][k+1, s+1], s=0,...,2n-1.
    #
    # For l=0,...,n-1:
    #
    #   sum_i p[i] g[|l-i|] =
    #       conv(p,g)[l] + conv(reverse(p),g)[n-l] - p[l] g[0],
    #
    #   sum_i p[i] g[l+i] =
    #       conv(reverse(p),g)[n+l].
    #
    # Hence top value is obtained from two convolutions, packed into one inverse
    # FFT.
    # -------------------------------------------------------------------------
    if n_top > 0
        V_dense_mat = zeros(Float64, Kp, n_top)

        if use_fft_top
            conv_len_top = 3 * n
            Lfft_top = _nextpow2_len(conv_len_top)

            top_buf = zeros(ComplexF64, Lfft_top)
            top_fft = zeros(ComplexF64, Lfft_top)
            top_work = zeros(ComplexF64, Lfft_top)
            top_seq = zeros(ComplexF64, Lfft_top)

            plan_top_fwd = plan_fft(top_buf; flags=FFTW.ESTIMATE)
            plan_top_inv = plan_ifft(top_buf; flags=FFTW.ESTIMATE)

            Fpack_top = _precompute_packed_spectra(pblocks, n, Lfft_top, plan_top_fwd)

            @inbounds for j in 1:m
                Hd = h_dense[j]
                pb = pblocks[j]
                Fpj = Fpack_top[j]

                for k in 0:K_max
                    fill!(top_buf, 0.0 + 0.0im)

                    for s in 0:(2 * n - 1)
                        top_buf[s + 1] = Hd[k + 1, s + 1]
                    end

                    _fft_packed_convolution!(
                        top_seq,
                        top_work,
                        top_fft,
                        top_buf,
                        Fpj,
                        plan_top_fwd,
                        plan_top_inv,
                    )

                    g0 = Hd[k + 1, 1]

                    for l in 0:(n_top - 1)
                        # real(top_seq[l+1])        = conv(p,g)[l]
                        # imag(top_seq[n-l+1])      = conv(reverse(p),g)[n-l]
                        # imag(top_seq[n+l+1])      = conv(reverse(p),g)[n+l]
                        val =
                            real(top_seq[l + 1]) +
                            imag(top_seq[n - l + 1]) -
                            pb.dense[l + 1] * g0 +
                            imag(top_seq[n + l + 1])

                        V_dense_mat[k + 1, l + 1] += val
                    end
                end
            end
        else
            # Direct top fallback. This is usually faster when n is tiny.
            @inbounds for j in 1:m
                Hd = h_dense[j]
                pb = pblocks[j]

                for q in eachindex(pb.vv)
                    i = pb.ii[q]
                    P_val = pb.vv[q]

                    for l in 0:(n_top - 1)
                        s1 = abs(l - i)
                        s2 = l + i

                        col_out = l + 1
                        col1 = s1 + 1
                        col2 = s2 + 1

                        @simd for k1 in 1:Kp
                            V_dense_mat[k1, col_out] +=
                                P_val * (Hd[k1, col1] + Hd[k1, col2])
                        end
                    end
                end
            end
        end

        # Chebyshev constant mode weight for ell = 0.
        @inbounds @simd for k1 in 1:Kp
            V_dense_mat[k1, 1] *= 0.5
        end

        @inbounds for k1 in 1:Kp
            for l1 in 1:n_top
                val = V_dense_mat[k1, l1]

                if abs(val) > tol
                    push!(I, l1)
                    push!(J, k1)
                    push!(V, val)
                end
            end
        end
    end

    # -------------------------------------------------------------------------
    # 2. Bottom banded block
    #
    # For fixed output column k and fixed j, define the anti-diagonal vector
    #
    #     g[q] = h_j(s = k-q, q),     q = 0,...,r_target.
    #
    # Then for r = k-l:
    #
    #     term s = l+i gives  sum_i p[i] g[r-i] = conv(p,g)[r],
    #     term s = l-i gives  sum_i p[i] g[r+i] = conv(reverse(p),g)[r+n].
    #
    # The full formula is not symmetric in r.  Here we force the stored tail band
    # to be symmetric by only emitting r = -r_sym,...,r_sym.
    # We again pack these two convolutions into one inverse FFT.
    # -------------------------------------------------------------------------
    # Only q-values that can contribute to r in [-r_sym,r_sym] are needed.
    # The first convolution needs q <= r_sym, while the reversed convolution
    # needs q <= r_sym+n.  We also clip by K_max because q <= k and k <= K_max.
    q_needed = min(r_target, r_sym + n, K_max)

    conv_len_bottom = q_needed + n + 1
    Lfft_bottom = _nextpow2_len(conv_len_bottom)

    bot_buf = zeros(ComplexF64, Lfft_bottom)
    bot_fft = zeros(ComplexF64, Lfft_bottom)
    bot_work = zeros(ComplexF64, Lfft_bottom)
    bot_seq = zeros(ComplexF64, Lfft_bottom)

    plan_bot_fwd = plan_fft(bot_buf; flags=FFTW.ESTIMATE)
    plan_bot_inv = plan_ifft(bot_buf; flags=FFTW.ESTIMATE)

    Fpack_bottom = _precompute_packed_spectra(pblocks, n, Lfft_bottom, plan_bot_fwd)

    col_vals = Vector{Float64}(undef, 2 * r_sym + 1)

    @inbounds for k in 0:K_max
        fill!(col_vals, 0.0)

        for j in 1:m
            Hd = h_dense[j]
            Hb = h_banded[j]
            Fpj = Fpack_bottom[j]

            fill!(bot_buf, 0.0 + 0.0im)

            # Build g[q] = h_j(k-q, q).
            # Only q <= min(r_target,k) can give s >= 0.
            q_hi = min(q_needed, k)
            s_last_band = 2 * n + size(Hb, 1) - 1

            for q in 0:q_hi
                s = k - q

                if s < 2 * n
                    bot_buf[q + 1] = Hd[k + 1, s + 1]
                elseif s <= s_last_band
                    bot_buf[q + 1] = Hb[s - 2 * n + 1, q + 1]
                end
            end

            _fft_packed_convolution!(
                bot_seq,
                bot_work,
                bot_fft,
                bot_buf,
                Fpj,
                plan_bot_fwd,
                plan_bot_inv,
            )

            # Accumulate into current output column.
            for r in -r_sym:r_sym
                l = k - r

                if l < n || l > L_max
                    continue
                end

                val = 0.0

                # conv(p,g)[r]
                if r >= 0
                    val += real(bot_seq[r + 1])
                end

                # conv(reverse(p),g)[r+n]
                idx = r + n
                if idx >= 0
                    val += imag(bot_seq[idx + 1])
                end

                col_vals[r + r_sym + 1] += val
            end
        end

        # Push current column.
        for r in -r_sym:r_sym
            l = k - r

            if l < n || l > L_max
                continue
            end

            val = col_vals[r + r_sym + 1]

            if abs(val) > tol
                push!(I, l + 1)
                push!(J, k + 1)
                push!(V, val)
            end
        end
    end

    return sparse(I, J, V, L_max + 1, K_max + 1)
end


# =============================================================================
# Compressed no-FFT assembly
# =============================================================================

function _precompute_H_zero_matrix(m_total::Int, K_max::Int)
    H0 = zeros(Float64, K_max + 1, m_total)

    @inbounds for j in 0:(m_total - 1)
        k_stop = min(K_max, j + 1)

        for k in 0:k_stop
            H0[k + 1, j + 1] = H_zero_exact_dense(j, k)
        end
    end

    return H0
end


function _generate_compressed_dense_top_rows(
    coeffs::AbstractVector{Float64},
    H_zero::Matrix{Float64},
    n::Int,
    K_max::Int,
)
    Kp = K_max + 1
    ns = 2 * n

    h_dense = zeros(Float64, Kp, ns)

    if ns == 0
        return h_dense
    end

    H_prev = zeros(Float64, Kp)
    H_curr = zeros(Float64, Kp)

    # Offset r = 0 is the only place where the test function enters.  Because
    # the following recurrence is linear in this initial vector, a Chebyshev
    # linear combination F(t) = sum_j coeffs[j] T_j(t) can be propagated
    # directly without forming every H_{j,r,k}.
    mul!(H_prev, H_zero, coeffs)

    @inbounds @simd for k1 in 1:Kp
        h_dense[k1, 1] = H_prev[k1]
    end

    # Only the top offsets 0,...,2n-1 are needed in dense layout.  If the
    # requested square is smaller than this, the remaining columns are
    # structurally zero for all available k.
    top_last = min(ns - 1, K_max)

    @inbounds for s in 0:(top_last - 1)
        ss = s + 1

        H_curr[ss + 1] = H_prev[ss + 1] / (2.0 * s + 1.0) + H_prev[ss]

        for k in (ss + 1):K_max
            denom = Float64(k + s)
            H_curr[k + 1] =
                ((k - s) / denom) * H_prev[k + 1] +
                H_prev[k] -
                ((k - s - 2) / denom) * H_curr[k]
        end

        col = ss + 1

        for k in ss:K_max
            h_dense[k + 1, col] = (k / (ss + k)) * H_curr[k + 1]
        end

        H_prev, H_curr = H_curr, H_prev
    end

    return h_dense
end


@inline function _has_nonzero(v::AbstractVector{Float64})
    @inbounds for x in v
        if x != 0.0
            return true
        end
    end

    return false
end


function _copy_scaled_h_flat!(
    dest::Vector{Float64},
    src::Matrix{Float64},
    n::Int,
    r_target::Int,
)
    s_start = 2 * n
    num_rows = size(src, 1)

    @inbounds for r in 0:r_target
        rf = Float64(r)
        base = r * num_rows

        @simd for s_idx in 1:num_rows
            s = Float64(s_start + s_idx - 1)
            denom = 2.0 * s + rf
            scale = denom == 0.0 ? 1.0 : (s + rf) / denom

            dest[base + s_idx] = scale * src[s_idx, r + 1]
        end
    end

    return nothing
end


function _rank_update_compressed_banded!(
    compressed::Matrix{Float64},
    P_coeffs::Matrix{Float64},
    j_idx::Int,
    scaled_h::Vector{Float64},
)
    coeff_col = @view P_coeffs[:, j_idx]

    if _has_nonzero(coeff_col)
        LinearAlgebra.BLAS.ger!(1.0, coeff_col, scaled_h, compressed)
    end

    return nothing
end


function _generate_compressed_banded_matrix_rankupdate(
    P_coeffs::Matrix{Float64},
    n::Int,
    m_total::Int,
    S_max::Int,
    r_target::Int,
)
    if r_target < 0
        throw(ArgumentError("r_target must be nonnegative"))
    end

    p_count = size(P_coeffs, 1)
    s_start = 2 * n
    num_rows = S_max - s_start + 1

    if num_rows <= 0
        return zeros(Float64, p_count, 0)
    end

    grid_len = num_rows * (r_target + 1)
    compressed = zeros(Float64, p_count, grid_len)
    scaled_h = zeros(Float64, grid_len)

    # The available stable tail recurrence is still the Chebyshev-j recurrence.
    # We do not store the full h[j][s,q] tensor; each generated T_j block is
    # immediately added to the compressed moments for all post-multiplication
    # x-modes.  This avoids the large 3D tensor, but the rank updates still
    # carry an extra factor m_total.  A fully compressed tail recurrence would
    # propagate each F_i(t) directly instead.
    max_r_needed = r_target + m_total - 1

    H_prev = zeros(Float64, num_rows, max_r_needed + 1)
    H_curr = zeros(Float64, num_rows, max_r_needed + 1)
    H_next = zeros(Float64, num_rows, max_r_needed + 1)

    build_base_H_banded!(H_prev, 0, n, S_max, max_r_needed)
    _copy_scaled_h_flat!(scaled_h, H_prev, n, r_target)
    _rank_update_compressed_banded!(compressed, P_coeffs, 1, scaled_h)

    if m_total >= 2
        build_base_H_banded!(H_curr, 1, n, S_max, max_r_needed)
        _copy_scaled_h_flat!(scaled_h, H_curr, n, r_target)
        _rank_update_compressed_banded!(compressed, P_coeffs, 2, scaled_h)
    end

    if m_total <= 2
        return compressed
    end

    @inbounds for step::Int in 1:(m_total - 2)
        curr_j::Int = step + 1
        curr_r_max::Int = r_target + m_total - 1 - curr_j

        for r::Int in 0:curr_r_max
            rf::Float64 = Float64(r)
            col::Int = r + 1

            if r == 0
                @simd for s_idx in 1:num_rows
                    s::Float64 = Float64(s_start + s_idx - 1)

                    A_r::Float64 =
                        (rf + 1.0) * (rf + 2.0 * s) /
                        ((rf + s) * (2.0 * rf + 2.0 * s + 1.0))

                    B_r::Float64 =
                        (4.0 * s * s - 1.0) /
                        ((2.0 * rf + 2.0 * s - 1.0) *
                         (2.0 * rf + 2.0 * s + 1.0))

                    H_next[s_idx, col] =
                        2.0 * A_r * H_curr[s_idx, 2] +
                        2.0 * B_r * H_curr[s_idx, 1] -
                        H_prev[s_idx, 1]
                end
            else
                @simd for s_idx in 1:num_rows
                    s::Float64 = Float64(s_start + s_idx - 1)

                    A_r::Float64 =
                        (rf + 1.0) * (rf + 2.0 * s) /
                        ((rf + s) * (2.0 * rf + 2.0 * s + 1.0))

                    B_r::Float64 =
                        (4.0 * s * s - 1.0) /
                        ((2.0 * rf + 2.0 * s - 1.0) *
                         (2.0 * rf + 2.0 * s + 1.0))

                    C_r::Float64 =
                        (rf - 1.0) * (rf + 2.0 * s) /
                        ((2.0 * rf + 2.0 * s - 1.0) * (rf + s))

                    H_next[s_idx, col] =
                        2.0 * A_r * H_curr[s_idx, col + 1] +
                        2.0 * B_r * H_curr[s_idx, col] +
                        2.0 * C_r * H_curr[s_idx, col - 1] -
                        H_prev[s_idx, col]
                end
            end
        end

        _copy_scaled_h_flat!(scaled_h, H_next, n, r_target)
        _rank_update_compressed_banded!(compressed, P_coeffs, curr_j + 1, scaled_h)

        H_prev, H_curr, H_next = H_curr, H_next, H_prev
    end

    return compressed
end


function _apply_tail_cheb_operator!(
    dest::Matrix{Float64},
    src::Matrix{Float64},
    n::Int,
    q_stop::Int,
)
    s_start = 2 * n
    num_rows = size(src, 1)

    @inbounds for q in 0:q_stop
        qf = Float64(q)
        col = q + 1

        if q == 0
            @simd for s_idx in 1:num_rows
                s = Float64(s_start + s_idx - 1)

                A_q =
                    (qf + 1.0) * (qf + 2.0 * s) /
                    ((qf + s) * (2.0 * qf + 2.0 * s + 1.0))

                B_q =
                    (4.0 * s * s - 1.0) /
                    ((2.0 * qf + 2.0 * s - 1.0) *
                     (2.0 * qf + 2.0 * s + 1.0))

                dest[s_idx, col] =
                    2.0 * A_q * src[s_idx, col + 1] +
                    2.0 * B_q * src[s_idx, col]
            end
        else
            @simd for s_idx in 1:num_rows
                s = Float64(s_start + s_idx - 1)

                A_q =
                    (qf + 1.0) * (qf + 2.0 * s) /
                    ((qf + s) * (2.0 * qf + 2.0 * s + 1.0))

                B_q =
                    (4.0 * s * s - 1.0) /
                    ((2.0 * qf + 2.0 * s - 1.0) *
                     (2.0 * qf + 2.0 * s + 1.0))

                C_q =
                    (qf - 1.0) * (qf + 2.0 * s) /
                    ((2.0 * qf + 2.0 * s - 1.0) * (qf + s))

                dest[s_idx, col] =
                    2.0 * A_q * src[s_idx, col + 1] +
                    2.0 * B_q * src[s_idx, col] +
                    2.0 * C_q * src[s_idx, col - 1]
            end
        end
    end

    return nothing
end


function _generate_compressed_banded_matrix_clenshaw(
    P_coeffs::Matrix{Float64},
    n::Int,
    m_total::Int,
    S_max::Int,
    r_target::Int,
)
    if r_target < 0
        throw(ArgumentError("r_target must be nonnegative"))
    end

    p_count = size(P_coeffs, 1)
    s_start = 2 * n
    num_rows = S_max - s_start + 1

    if num_rows <= 0
        return zeros(Float64, p_count, 0)
    end

    grid_len = num_rows * (r_target + 1)
    compressed = zeros(Float64, p_count, grid_len)

    # The tail recurrence for individual Chebyshev modes can be written as
    #
    #     H_{j+1} = L H_j - H_{j-1},
    #
    # where L is the q-banded operator implemented by
    # _apply_tail_cheb_operator!.  This routine evaluates the Chebyshev
    # coefficient vector for each compressed test function by Clenshaw's
    # algorithm, so it does not generate a T_j block and BLAS-rank-update it
    # into every i.  It is exact for the same finite q-lookahead as the
    # rank-update recurrence.  The price is that applying L to a dense
    # coefficient row is still O(m_total * r_target * N), so the tail is not a
    # strict O(n^2 N) closure for arbitrary dense t-coefficients.
    q_max = r_target + m_total

    H0 = zeros(Float64, num_rows, q_max + 2)
    B_next = zeros(Float64, num_rows, q_max + 2)
    B_next2 = zeros(Float64, num_rows, q_max + 2)
    B_curr = zeros(Float64, num_rows, q_max + 2)
    L_work = zeros(Float64, num_rows, q_max + 2)

    build_base_H_banded!(H0, 0, n, S_max, q_max + 1)

    @inbounds for i in 1:p_count
        coeffs = @view P_coeffs[i, :]

        if !_has_nonzero(coeffs)
            continue
        end

        fill!(B_next, 0.0)
        fill!(B_next2, 0.0)

        for j in (m_total - 1):-1:1
            _apply_tail_cheb_operator!(L_work, B_next, n, q_max)

            cj = coeffs[j + 1]

            for q in 0:q_max
                col = q + 1

                @simd for s_idx in 1:num_rows
                    B_curr[s_idx, col] =
                        cj * H0[s_idx, col] +
                        L_work[s_idx, col] -
                        B_next2[s_idx, col]
                end
            end

            # Keep the one-column look-ahead boundary homogeneous.  This is the
            # same finite-degree closure used implicitly by the j-forward tail
            # recurrence.
            @simd for s_idx in 1:num_rows
                B_curr[s_idx, q_max + 2] = 0.0
            end

            B_next2, B_next, B_curr = B_next, B_curr, B_next2
        end

        _apply_tail_cheb_operator!(L_work, B_next, n, r_target)

        c0 = coeffs[1]

        for q in 0:r_target
            qf = Float64(q)
            base = q * num_rows

            @simd for s_idx in 1:num_rows
                s = Float64(s_start + s_idx - 1)
                denom = 2.0 * s + qf
                scale = denom == 0.0 ? 1.0 : (s + qf) / denom

                compressed[i, base + s_idx] =
                    scale * (
                        c0 * H0[s_idx, q + 1] +
                        0.5 * L_work[s_idx, q + 1] -
                        B_next2[s_idx, q + 1]
                    )
            end
        end
    end

    return compressed
end


function _generate_compressed_banded_matrix(
    P_coeffs::Matrix{Float64},
    n::Int,
    m_total::Int,
    S_max::Int,
    r_target::Int;
    method::Symbol = :clenshaw,
)
    if method === :clenshaw
        return _generate_compressed_banded_matrix_clenshaw(
            P_coeffs,
            n,
            m_total,
            S_max,
            r_target,
        )
    elseif method === :rankupdate
        return _generate_compressed_banded_matrix_rankupdate(
            P_coeffs,
            n,
            m_total,
            S_max,
            r_target,
        )
    end

    throw(ArgumentError("compressed tail method must be :clenshaw or :rankupdate"))
end


"""
    assemble_VK_sparse_block_compressed_nofft(C, L_max, K_max; ...)

Assemble the Chebyshev Volterra matrix with a compressed, non-FFT path.  The
top dense rows propagate the compressed test functions directly.  By default the
banded tail evaluates the Chebyshev coefficient vector with a no-FFT Clenshaw
recurrence; `tail_method = :rankupdate` keeps the older compressed rank-update
tail for diagnostics and benchmarking.
"""
function assemble_VK_sparse_block_compressed_nofft(
    C::Matrix{Float64},
    L_max::Int,
    K_max::Int;
    tol::Float64 = eps(Float64),
    tail_half_bandwidth::Union{Nothing,Int} = nothing,
    tail_method::Symbol = :clenshaw,
)
    n, m = size(C)

    if L_max < 0 || K_max < 0
        throw(ArgumentError("L_max and K_max must be nonnegative"))
    end

    Kp = K_max + 1
    r_target = 2 * n + m - 1
    n_top = min(n, L_max + 1)

    r_full_upper = n + m - 1
    r_full_lower = n
    r_sym_requested = tail_half_bandwidth === nothing ? r_full_lower : tail_half_bandwidth

    if r_sym_requested < 0
        throw(ArgumentError("tail_half_bandwidth must be nonnegative"))
    end

    r_sym = min(r_sym_requested, L_max, K_max, r_target)

    if r_sym_requested > r_full_upper
        @warn "tail_half_bandwidth exceeds the full positive bandwidth; extra stored entries are structurally zero" r_sym_requested r_full_upper
    end

    q_needed = min(r_target, r_sym + n, K_max)

    estimated_nnz =
        n_top * (K_max + 1) +
        max(0, L_max - n + 1) * (2 * r_sym + 1)

    I = Int[]
    J = Int[]
    V = Float64[]

    sizehint!(I, estimated_nnz)
    sizehint!(J, estimated_nnz)
    sizehint!(V, estimated_nnz)

    P_coeffs = _build_postmult_coeffs(C)
    H_zero = _precompute_H_zero_matrix(m, K_max)

    top_vals = zeros(Float64, Kp, n_top)
    band_vals = zeros(Float64, 2 * r_sym + 1, Kp)

    # -------------------------------------------------------------------------
    # 1. Top dense rows and the dense-offset contribution to the band tail.
    #
    # For each post-multiplication x-mode i, the t-direction coefficients define
    # F_i(t).  The dense offset recurrence is applied once to F_i, not once for
    # every individual T_j(t).
    # The k = 0 column is handled here as well: all offsets s > 0 are zero, so
    # the exact Chebyshev integral formula survives after the M[(1+x)/2]
    # coefficient absorption.
    # -------------------------------------------------------------------------
    @inbounds for i in 0:n
        coeffs = @view P_coeffs[i + 1, :]

        if !_has_nonzero(coeffs)
            continue
        end

        Hd = _generate_compressed_dense_top_rows(coeffs, H_zero, n, K_max)

        for l in 0:(n_top - 1)
            s1 = abs(l - i) + 1
            s2 = l + i + 1
            col_out = l + 1

            @simd for k1 in 1:Kp
                top_vals[k1, col_out] += Hd[k1, s1] + Hd[k1, s2]
            end
        end

        for k in 0:K_max
            for r in -r_sym:r_sym
                l = k - r

                if l < n || l > L_max
                    continue
                end

                val = 0.0

                s_plus = l + i
                if s_plus < 2 * n
                    val += Hd[k + 1, s_plus + 1]
                end

                s_minus = l - i
                if s_minus < 2 * n
                    val += Hd[k + 1, s_minus + 1]
                end

                band_vals[r + r_sym + 1, k + 1] += val
            end
        end
    end

    # -------------------------------------------------------------------------
    # 2. Banded tail offsets s >= 2n.
    #
    # The compressed banded matrix is indexed as
    #     compressed[i+1, q*num_rows + (s-2n+1)] = Hhat_i[s,q],
    # where q = k-s.  Only q-values that can appear inside the emitted symmetric
    # tail band are generated.
    # -------------------------------------------------------------------------
    compressed_banded = _generate_compressed_banded_matrix(
        P_coeffs,
        n,
        m,
        L_max,
        q_needed;
        method = tail_method,
    )

    num_band_rows = max(0, L_max - 2 * n + 1)

    if num_band_rows > 0
        @inbounds for i in 0:n
            for k in 0:K_max
                for r in -r_sym:r_sym
                    l = k - r

                    if l < n || l > L_max
                        continue
                    end

                    val = 0.0

                    s_plus = l + i
                    if s_plus >= 2 * n && s_plus <= L_max
                        q = k - s_plus

                        if 0 <= q <= q_needed
                            idx = q * num_band_rows + (s_plus - 2 * n + 1)
                            val += compressed_banded[i + 1, idx]
                        end
                    end

                    s_minus = l - i
                    if s_minus >= 2 * n && s_minus <= L_max
                        q = k - s_minus

                        if 0 <= q <= q_needed
                            idx = q * num_band_rows + (s_minus - 2 * n + 1)
                            val += compressed_banded[i + 1, idx]
                        end
                    end

                    band_vals[r + r_sym + 1, k + 1] += val
                end
            end
        end
    end

    # Chebyshev constant mode weight for ell = 0.
    if n_top > 0
        @inbounds @simd for k1 in 1:Kp
            top_vals[k1, 1] *= 0.5
        end
    end

    @inbounds for k1 in 1:Kp
        for l1 in 1:n_top
            val = top_vals[k1, l1]

            if abs(val) > tol
                push!(I, l1)
                push!(J, k1)
                push!(V, val)
            end
        end
    end

    @inbounds for k in 0:K_max
        for r in -r_sym:r_sym
            l = k - r

            if l < n || l > L_max
                continue
            end

            val = band_vals[r + r_sym + 1, k + 1]

            if abs(val) > tol
                push!(I, l + 1)
                push!(J, k + 1)
                push!(V, val)
            end
        end
    end

    return sparse(I, J, V, L_max + 1, K_max + 1)
end


# =============================================================================
# Top-level interfaces
# =============================================================================

# -----------------------------------------------------------------------------
# Main constructor from analytic K.
# -----------------------------------------------------------------------------

@inline function _check_cheb_assembly(assembly::Symbol)
    if assembly === :fft ||
       assembly === :compressed ||
       assembly === :compressed_nofft ||
       assembly === :compressed_full ||
       assembly === :compressed_rankupdate
        return nothing
    end

    throw(ArgumentError("assembly must be :fft, :compressed, :compressed_nofft, :compressed_full, or :compressed_rankupdate"))
end


@inline function _compressed_tail_method(assembly::Symbol)
    return assembly === :compressed_rankupdate ? :rankupdate : :clenshaw
end


function voltmat_cheb(
    K,
    N::Int;
    assembly::Symbol = :fft,
    use_fft_top::Bool = true,
    tol::Float64 = eps(Float64),
    tail_half_bandwidth::Union{Nothing,Int} = nothing,
)
    _check_cheb_assembly(assembly)

    KK(x, y) = K(x, (1 + x) * (1 + y) / 2 - 1)

    C = adaptive_cheb2_coeffmat(KK)

    N_PARAM, M_TOTAL = size(C)

    L_MAX_TEST = N
    # Square output: only modes k = 0:N are assembled.
    K_MAX_TEST = N

    VK = if assembly === :fft
        h_dense_tensors = generate_dense_blocks(N_PARAM, M_TOTAL, K_MAX_TEST)

        # For square output, the largest possible s = k-q is at most N.
        # The largest needed q is also smaller than the full rectangular value:
        # q <= b_eff + n, because only r in [-b_eff,b_eff] is emitted.
        r_target_full = 2 * N_PARAM + M_TOTAL - 1
        b_req = tail_half_bandwidth === nothing ? N_PARAM : tail_half_bandwidth
        if b_req < 0
            throw(ArgumentError("tail_half_bandwidth must be nonnegative"))
        end
        b_eff = min(b_req, L_MAX_TEST, K_MAX_TEST, r_target_full)
        q_needed_tail = min(r_target_full, b_eff + N_PARAM, K_MAX_TEST)

        h_banded_tensors = _generate_banded_blocks_core(
            N_PARAM,
            M_TOTAL,
            L_MAX_TEST,
            q_needed_tail,
        )

        assemble_VK_sparse_block_fft_convolution(
            C,
            h_dense_tensors,
            h_banded_tensors,
            L_MAX_TEST,
            K_MAX_TEST;
            tol = tol,
            use_fft_top = use_fft_top,
            tail_half_bandwidth = tail_half_bandwidth,
        )
    else
        assemble_VK_sparse_block_compressed_nofft(
            C,
            L_MAX_TEST,
            K_MAX_TEST;
            tol = tol,
            tail_half_bandwidth = tail_half_bandwidth,
            tail_method = _compressed_tail_method(assembly),
        )
    end

    return VK
end


# -----------------------------------------------------------------------------
# Main constructor from known C.
# -----------------------------------------------------------------------------

function voltmat_cheb(
    C::Matrix{Float64},
    N::Int;
    assembly::Symbol = :fft,
    use_fft_top::Bool = true,
    tol::Float64 = eps(Float64),
    tail_half_bandwidth::Union{Nothing,Int} = nothing,
)
    _check_cheb_assembly(assembly)

    N_PARAM, M_TOTAL = size(C)

    L_MAX_TEST = N
    # Square output: only modes k = 0:N are assembled.
    K_MAX_TEST = N

    VK = if assembly === :fft
        h_dense_tensors = generate_dense_blocks(N_PARAM, M_TOTAL, K_MAX_TEST)

        # For square output, the largest possible s = k-q is at most N.
        # The largest needed q is also smaller than the full rectangular value:
        # q <= b_eff + n, because only r in [-b_eff,b_eff] is emitted.
        r_target_full = 2 * N_PARAM + M_TOTAL - 1
        b_req = tail_half_bandwidth === nothing ? N_PARAM : tail_half_bandwidth
        if b_req < 0
            throw(ArgumentError("tail_half_bandwidth must be nonnegative"))
        end
        b_eff = min(b_req, L_MAX_TEST, K_MAX_TEST, r_target_full)
        q_needed_tail = min(r_target_full, b_eff + N_PARAM, K_MAX_TEST)

        h_banded_tensors = _generate_banded_blocks_core(
            N_PARAM,
            M_TOTAL,
            L_MAX_TEST,
            q_needed_tail,
        )

        assemble_VK_sparse_block_fft_convolution(
            C,
            h_dense_tensors,
            h_banded_tensors,
            L_MAX_TEST,
            K_MAX_TEST;
            tol = tol,
            use_fft_top = use_fft_top,
            tail_half_bandwidth = tail_half_bandwidth,
        )
    else
        assemble_VK_sparse_block_compressed_nofft(
            C,
            L_MAX_TEST,
            K_MAX_TEST;
            tol = tol,
            tail_half_bandwidth = tail_half_bandwidth,
            tail_method = _compressed_tail_method(assembly),
        )
    end

    return VK
end

# Explicit aliases with the requested misspelling.
const voltmat_cheb_spuare = voltmat_cheb
const voltmat_spuare_from_coeffmat_fft_convolution = voltmat_cheb

# Correctly spelled aliases too.
const voltmat_cheb_square = voltmat_cheb
const voltmat_square_from_coeffmat_fft_convolution = voltmat_cheb

end
