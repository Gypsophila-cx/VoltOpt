using FastGaussQuadrature
using LinearAlgebra
using Random
using Test
using VoltOpt

const WeakCurrent = VoltOpt.Weak_VoltMat_Cheb

# At fixed q, expand
#   T_l(((1+q)z + q - 1)/2) = sum_k a[k,l](q) T_k(z)
# by the defining Chebyshev recurrence.  This is independent of either moment
# recurrence used by the implementation.
function direct_duffy_cheb_coefficients(q::Float64, N::Int)
    A = zeros(Float64, N + 1, N + 1)
    A[1, 1] = 1.0
    N == 0 && return A

    aq = 0.5 * (1.0 + q)
    bq = 0.5 * (q - 1.0)
    A[1, 2] = bq
    A[2, 2] = aq
    ztimes = zeros(Float64, N + 1)

    @inbounds for ell in 1:(N - 1)
        fill!(ztimes, 0.0)
        for k in 0:ell
            ck = A[k + 1, ell + 1]
            if k == 0
                ztimes[2] += ck
            else
                ztimes[k] += 0.5 * ck
                ztimes[k + 2] += 0.5 * ck
            end
        end
        for k in 0:(ell + 1)
            A[k + 1, ell + 2] =
                2.0 * aq * ztimes[k + 1] +
                2.0 * bq * A[k + 1, ell + 1] -
                A[k + 1, ell]
        end
    end
    return A
end

function direct_moment_tensor(mu::Float64, sigma::Float64, beta::Float64,
                              J::Int, N::Int; Q::Int = 400)
    gamma = (sigma + 1.0) / beta - 1.0
    q, w = gaussjacobi(Q, -mu, gamma)
    residual = WeakCurrent._residual_factor_general(q, mu, sigma, beta)
    Tj = WeakCurrent.chebT_table(q, J)
    H = zeros(Float64, J + 1, N + 1, N + 1)

    @inbounds for iq in eachindex(q)
        Aq = direct_duffy_cheb_coefficients(q[iq], N)
        wi = w[iq] * residual[iq]
        for ell in 0:N, k in 0:ell, j in 0:J
            # The paper's tilde-H convention equals the raw Chebyshev
            # coefficient for k=0 and (l+k)/(2l) times it for k>0.
            moment_normalization = k == 0 ? 1.0 : (ell + k) / (2.0 * ell)
            H[j + 1, k + 1, ell + 1] +=
                wi * Tj[iq, j + 1] * moment_normalization * Aq[k + 1, ell + 1]
        end
    end
    return H
end

function contract_moments(C::AbstractMatrix, H::Array{Float64,3})
    I1, J1 = size(C)
    _, K1, L1 = size(H)
    Lambda = zeros(Float64, I1, K1, L1)
    @inbounds for i in 1:I1, k in 1:K1, ell in 1:L1
        for j in 1:J1
            Lambda[i, k, ell] += C[i, j] * H[j, k, ell]
        end
    end
    return Lambda
end

function four_point_from_k0(Lambda0::AbstractMatrix, N::Int)
    I1 = size(Lambda0, 1)
    Lambda = zeros(Float64, I1, N + 1, N + 1)
    Lambda[:, 1, :] .= Lambda0
    current = Matrix{Float64}(Lambda0)
    next = zeros(Float64, I1, N + 1)
    for k in 0:(N - 1)
        WeakCurrent._blocki_next_layer!(next, current, k, N, I1)
        @views Lambda[:, k + 2, k + 1] .= next[:, k + 1]
        @views Lambda[:, k + 2, (k + 2):(N + 1)] .= next[:, (k + 2):(N + 1)]
        current, next = next, current
    end
    return Lambda
end

function direct_inner_matrix(C::AbstractMatrix, H::Array{Float64,3},
                             N::Int, sigma::Float64)
    Lambda = contract_moments(C, H)
    I = size(C, 1) - 1
    A = zeros(Float64, N + 1, N + 1)
    factor = 2.0^(-sigma)

    @inbounds for r in 0:min(I, N)
        A[r + 1, 1] = factor * Lambda[r + 1, 1, 1]
    end
    @inbounds for ell in 1:N, r in 0:N
        acc = 0.0
        for i in 0:I
            kabs = abs(i - r)
            if kabs <= ell
                acc += ell / (ell + kabs) * Lambda[i + 1, kabs + 1, ell + 1]
            end
            ksum = i + r
            if ksum <= ell
                acc += ell / (ell + ksum) * Lambda[i + 1, ksum + 1, ell + 1]
            end
        end
        A[r + 1, ell + 1] = factor * (r == 0 ? 0.5 : 1.0) * acc
    end
    return A
end

function direct_cheb_product_matrix(A::AbstractMatrix, g::AbstractVector)
    N = size(A, 1) - 1
    G = zeros(Float64, N + 1, N + 1)
    @inbounds for col in 1:(N + 1), r in 0:N
        ar = A[r + 1, col]
        for q in 0:(length(g) - 1)
            term = ar * g[q + 1]
            if q == 0
                G[r + 1, col] += term
            elseif r == 0
                q <= N && (G[q + 1, col] += term)
            else
                r + q <= N && (G[r + q + 1, col] += 0.5 * term)
                G[abs(r - q) + 1, col] += 0.5 * term
            end
        end
    end
    return G
end

@testset "Current weak-singular four-point algorithm" begin
    @testset "k=0 initialization" begin
        mu, sigma, beta = 0.5, 0.2, 0.5
        J, N = 4, 9
        fast = WeakCurrent.initial_moments_fast(
            mu, sigma, beta, J, N;
            de_qfactor = 48,
            accurate_sum = true,
        )
        direct = WeakCurrent.initial_moments_direct(mu, sigma, beta, J, N; Qbase = 800)
        err = maximum(abs, fast .- direct)
        rel = err / maximum(abs, direct)
        @info "weak k=0 initialization" err rel
        @test err <= 2.0e-11
        @test rel <= 2.0e-11
    end

    @testset "direct moments and four-point recurrence" begin
        mu, sigma, beta = 0.5, 0.0, 0.5
        J, N = 3, 9
        H = direct_moment_tensor(mu, sigma, beta, J, N; Q = 500)
        max_residual = 0.0
        for j in 0:J, k in 0:(N - 1), ell in (k + 1):N
            rhs = ((ell - k) * H[j + 1, k + 1, ell + 1] +
                   (ell + k) * H[j + 1, k + 1, ell] -
                   (ell - k - 2) * H[j + 1, k + 2, ell]) / (ell + k)
            max_residual = max(max_residual, abs(H[j + 1, k + 2, ell + 1] - rhs))
        end
        @info "weak direct four-point recurrence" max_residual
        @test max_residual <= 3.0e-12
        boundary = maximum(
            abs(H[j + 1, k + 2, k + 1])
            for j in 0:J, k in 0:(N - 1)
        )
        @test boundary <= 3.0e-14

        # Explicitly cover l=k+1, l=k+2, and a well-separated interior point.
        for (k, ell) in ((0, 1), (2, 3), (2, 4), (1, 8))
            j = min(k, J)
            rhs = ((ell - k) * H[j + 1, k + 1, ell + 1] +
                   (ell + k) * H[j + 1, k + 1, ell] -
                   (ell - k - 2) * H[j + 1, k + 2, ell]) / (ell + k)
            @test H[j + 1, k + 2, ell + 1] ≈ rhs atol = 3.0e-12 rtol = 3.0e-12
        end
    end

    @testset "contraction before recurrence" begin
        rng = MersenneTwister(20260908)
        mu, sigma, beta = 0.35, 0.1, 0.65
        I, J, N = 4, 3, 8
        C = randn(rng, I + 1, J + 1)
        H = direct_moment_tensor(mu, sigma, beta, J, N; Q = 500)
        explicit = contract_moments(C, H)
        streamed = four_point_from_k0(@view(explicit[:, 1, :]), N)
        err = 0.0
        for k in 0:N, ell in k:N, i in 1:(I + 1)
            err = max(
                err,
                abs(streamed[i, k + 1, ell + 1] - explicit[i, k + 1, ell + 1]),
            )
        end
        rel = err / maximum(abs, explicit)
        @info "weak contraction-before-recurrence" err rel
        @test err <= 2.0e-11
        @test rel <= 2.0e-11
    end

    @testset "assembled dense and packed matrices" begin
        rng = MersenneTwister(17)
        I, J, N = 3, 3, 9
        C = 0.1 .* randn(rng, I + 1, J + 1)
        C[1, 1] += 1.0

        for (mu, beta, eta) in ((0.5, 0.5, 1), (0.0, 0.5, 2))
            sigma = 0.0
            H = direct_moment_tensor(mu, sigma, beta, J, N; Q = 500)
            Aref = direct_inner_matrix(C, H, N, sigma)
            g, finite = WeakCurrent._oneplus_over2_eta_coeffs((1.0 - mu) / beta, N)
            @test finite
            @test length(g) - 1 == eta
            Gref = direct_cheb_product_matrix(Aref, g)

            dense, dinfo = VoltOpt.weak_voltmat_cheb(
                C;
                C_is_coeffs = true,
                mu = mu,
                sigma = sigma,
                beta = beta,
                N = N,
                init_method = :direct,
                Qbase = 500,
                output_format = :dense,
                H_recurrence_mode = :block_i,
                return_info = true,
            )
            packed, pinfo = VoltOpt.weak_voltmat_cheb(
                C;
                C_is_coeffs = true,
                mu = mu,
                sigma = sigma,
                beta = beta,
                N = N,
                init_method = :direct,
                Qbase = 500,
                output_format = :packed_lower_hessenberg,
                H_recurrence_mode = :block_i,
                return_info = true,
            )
            packed_dense = VoltOpt.packed_to_dense(packed)
            reference_err = maximum(abs, dense .- Gref)
            packed_err = maximum(abs, dense .- packed_dense)
            below = WeakCurrent.max_below_lower_band(dense, I + eta)
            @info "weak assembled matrix" eta reference_err packed_err below
            @test reference_err <= 2.0e-11
            @test packed_err <= 2.0e-14
            @test dinfo.lower_bandwidth_theory == I + eta
            @test pinfo.lower_bandwidth_theory == I + eta
            @test below <= 2.0e-14
            @test dinfo.moment_recurrence === :four_point
            @test dinfo.contraction_mode === :precombined_j
        end

        # Kernel degree may exceed the solution truncation in adaptive/error
        # studies; all three A-entry branches must truncate rows safely.
        Iwide, Jwide, Nsmall = 8, 2, 4
        Cwide = 0.1 .* randn(rng, Iwide + 1, Jwide + 1)
        Cwide[1, 1] += 1.0
        Hwide = direct_moment_tensor(0.5, 0.0, 0.5, Jwide, Nsmall; Q = 400)
        Awide = direct_inner_matrix(Cwide, Hwide, Nsmall, 0.0)
        gwide, _ = WeakCurrent._oneplus_over2_eta_coeffs(1.0, Nsmall)
        Gwide = direct_cheb_product_matrix(Awide, gwide)
        densewide, _ = VoltOpt.weak_voltmat_cheb(
            Cwide;
            C_is_coeffs = true,
            mu = 0.5,
            sigma = 0.0,
            beta = 0.5,
            N = Nsmall,
            init_method = :direct,
            Qbase = 400,
            output_format = :dense,
            H_recurrence_mode = :block_i,
        )
        packedwide, _ = VoltOpt.weak_voltmat_cheb(
            Cwide;
            C_is_coeffs = true,
            mu = 0.5,
            sigma = 0.0,
            beta = 0.5,
            N = Nsmall,
            init_method = :direct,
            Qbase = 400,
            output_format = :packed_lower_hessenberg,
            H_recurrence_mode = :block_i,
        )
        @test maximum(abs, densewide .- Gwide) <= 2.0e-11
        @test VoltOpt.packed_to_dense(packedwide) == densewide
    end

    @testset "direct operator columns" begin
        K(x, y) = 1.0
        N = 20
        V, info = VoltOpt.weak_voltmat_cheb(
            reshape([1.0], 1, 1);
            C_is_coeffs = true,
            mu = 0.5,
            sigma = 0.0,
            beta = 0.5,
            N = N,
            output_format = :packed_lower_hessenberg,
            H_recurrence_mode = :block_i,
            return_info = true,
        )
        validation = VoltOpt.validate_weak_voltmat_columns(
            K, V;
            mu = 0.5,
            sigma = 0.0,
            beta = 0.5,
            cols = (0, 1, 2, 5, 10, 18),
            xs = range(-0.95, 0.95; length = 31),
            Qref = 800,
        )
        @info "weak direct-column validation" validation.max_abs validation.max_rel
        @test info.H_recurrence_mode === :block_i
        @test validation.max_abs <= 2.0e-11
        @test validation.max_rel <= 2.0e-10

        _, auto_info = VoltOpt.weak_voltmat_cheb(
            reshape([1.0], 1, 1);
            C_is_coeffs = true,
            mu = 0.5,
            sigma = 0.0,
            beta = 0.5,
            N = 4,
            H_recurrence_mode = :auto,
            return_info = true,
        )
        @test auto_info.H_recurrence_mode === :block_i
        @test auto_info.moment_recurrence === :four_point

        _, generic_auto_info = VoltOpt.weak_voltmat_cheb(
            reshape([1.0], 1, 1);
            C_is_coeffs = true,
            mu = 0.25,
            sigma = 0.0,
            beta = 0.5,
            N = 8,
            output_format = :dense,
            H_recurrence_mode = :auto,
            return_info = true,
        )
        @test generic_auto_info.H_recurrence_mode === :block_i
        @test generic_auto_info.moment_recurrence === :four_point
        @test generic_auto_info.contraction_mode === :precombined_j

        Kexp(x, y) = exp(-x * y)
        Cexp = WeakCurrent.cheb_coeffs_lobatto_2d(
            WeakCurrent.sample_duffy_kernel(Kexp, 31, 31, 0.5),
        )
        Vexp, _ = VoltOpt.weak_voltmat_cheb(
            Cexp;
            C_is_coeffs = true,
            mu = 0.5,
            sigma = 0.0,
            beta = 0.5,
            N = 72,
            output_format = :packed_lower_hessenberg,
            H_recurrence_mode = :block_i,
        )
        validation_exp = VoltOpt.validate_weak_voltmat_columns(
            Kexp, Vexp;
            mu = 0.5,
            sigma = 0.0,
            beta = 0.5,
            cols = (0, 1, 3, 8, 16, 28),
            xs = range(-0.95, 0.95; length = 31),
            Qref = 800,
        )
        @info "weak nonconstant direct-column validation" validation_exp.max_abs validation_exp.max_rel
        @test validation_exp.max_abs <= 5.0e-11
        @test validation_exp.max_rel <= 5.0e-10
    end

    @testset "large-N smoke and recurrence stability" begin
        rng = MersenneTwister(123)
        I, J, N = 5, 4, 512
        C = 0.05 .* randn(rng, I + 1, J + 1)
        C[1, 1] += 1.0
        V, info = VoltOpt.weak_voltmat_cheb(
            C;
            C_is_coeffs = true,
            mu = 0.5,
            sigma = 0.0,
            beta = 0.5,
            N = N,
            output_format = :packed_lower_hessenberg,
            diagnostics = false,
            H_recurrence_mode = :block_i,
            return_info = true,
        )
        @test all(isfinite, V.values)
        @test info.stored_entries == length(V.values)
        @test info.lower_bandwidth_theory == I + 1
        @test V.lower_bw == I + 1

        M0 = WeakCurrent.initial_moments_fast(0.5, 0.0, 0.5, J, N)
        current = WeakCurrent._precombine_kernel_moments(C, M0, N, J)
        next = zeros(Float64, I + 1, N + 1)
        max_scaled_residual = 0.0
        for k in 0:(N - 1)
            WeakCurrent._blocki_next_layer!(next, current, k, N, I + 1)
            @inbounds for ell in (k + 1):N, ii in 1:(I + 1)
                lhs = (ell + k) * next[ii, ell + 1]
                t1 = (ell - k) * current[ii, ell + 1]
                t2 = (ell + k) * current[ii, ell]
                t3 = -(ell - k - 2) * next[ii, ell]
                scale = max(abs(t1) + abs(t2) + abs(t3), floatmin(Float64))
                max_scaled_residual = max(max_scaled_residual, abs(lhs - t1 - t2 - t3) / scale)
            end
            current, next = next, current
        end
        @info "weak large-N smoke" N max_scaled_residual maxabs=maximum(abs, V.values)
        @test max_scaled_residual <= 8eps(Float64)
    end
end
