module Weak_VoltMat_DE

export weak_voltmat_decheb
export build_C_adaptive_de
export build_postmult_coeffs_de, add_A_by_d!, band_to_dense,
    assemble_A_narrow_direct_absorbM_de, assemble_A_narrow_rfft_absorbM_de,
    plan_weak_de_fft
export logcosh, sinhc, log_sinhc, phi_de, log_phi_prime_de
export paper_de_phi, paper_de_dphi, stable_paper_de_transformed_kernel_value


# Self-contained demo for constructing the Chebyshev coefficient matrix of a
# weakly singular Volterra operator.  This file intentionally stays inside
# demo/ and does not include or modify the core src/ VoltOpt implementation.

using LinearAlgebra
using Random

const HAS_FFTW_DE = Ref(false)
try
    @eval import FFTW
    HAS_FFTW_DE[] = true
catch
end

const USES_DE_INITIAL_MOMENTS = true
const USES_STREAMING_J_COMPRESSION = true
const STORES_FULL_H_JRK_TENSOR = false
const PROJECT_FILES_MODIFIED = false

struct Cheb2Expansion
    C::Matrix{Float64}
    full_size::Tuple{Int, Int}
    tail::Float64
    max_coeff::Float64
    sample_error::Float64
    converged::Bool
end

struct TanhSinhNode
    eta::Float64
    log_one_minus_eta::Float64
    log_one_plus_eta::Float64
    rounded_to_right_endpoint::Bool
    rounded_to_left_endpoint::Bool
end

struct DEConvergenceRow
    h::Float64
    node_count::Int
    left_endpoint_distance::Float64
    right_endpoint_distance::Float64
    change_from_previous_h::Float64
    closed_form_error::Float64
    nonfinite_seen::Bool
    rounded_endpoint_nodes::Int
    right_virtual_nodes::Int
    right_final_log1m::Float64
    right_final_tail_estimate::Float64
    k0_tail_correction_used::Bool
end

struct DEDiagnostics
    h::Float64
    node_count::Int
    left_endpoint_distance::Float64
    right_endpoint_distance::Float64
    nonfinite_seen::Bool
    rounded_endpoint_nodes::Int
    right_virtual_nodes::Int
    right_final_log1m::Float64
    right_final_tail_estimate::Float64
    k0_tail_correction_used::Bool
    convergence_rows::Vector{DEConvergenceRow}
end

function maxabs_array(A)
    m = 0.0
    for x in A
        ax = abs(x)
        if ax > m
            m = ax
        end
    end
    return m
end

allfinite_array(A) = all(isfinite, A)

function min_abs_nonzero(A::AbstractMatrix{<:Real})
    best = Inf
    for x in A
        ax = abs(x)
        if ax > 0.0 && ax < best
            best = ax
        end
    end
    return isfinite(best) ? best : 0.0
end

duffy_tau(y, eta) = 0.5 * (y + 1.0) * eta + 0.5 * (y - 1.0)

identity_phi(x) = x
identity_dphi(x) = 1.0

function stable_phi_ratio(y::Float64, eta::Float64, phi, dphi)
    tau = duffy_tau(y, eta)
    denom = 1.0 - eta
    if abs(denom) < sqrt(eps(Float64))
        return 0.5 * (1.0 + y) * dphi(y)
    end

    ratio = (phi(y) - phi(tau)) / denom
    if ratio < 0.0 && abs(ratio) <= 100eps(Float64)
        return 0.0
    elseif ratio < 0.0
        error("phi appears non-monotone at y=$y, eta=$eta; stable ratio is negative")
    end
    return ratio
end

function transformed_kernel_value(K, phi, dphi, mu::Float64, y::Float64, eta::Float64)
    tau = duffy_tau(y, eta)
    if phi === identity_phi && dphi === identity_dphi
        ratio = max(0.5 * (1.0 + y), floatmin(Float64))
        return ratio^(mu - 1.0) * K(y, tau)
    end

    ratio = stable_phi_ratio(y, eta, phi, dphi)
    ratio = max(ratio, floatmin(Float64))
    return dphi(tau) * ratio^(mu - 1.0) * K(phi(y), phi(tau))
end

@inline function stable_log_sinhc(x::Float64)
    ax = abs(x)
    if ax < 1e-5
        x2 = x * x
        return log(1.0 + x2 / 6.0 + x2^2 / 120.0 + x2^3 / 5040.0)
    elseif ax > 700.0
        return ax - log(2.0) - log(ax)
    else
        return log(abs(sinh(x) / x))
    end
end

paper_de_phi(y::Real; omega::Float64=4.0) = Float64(y) == -1.0 ? -1.0 :
    (Float64(y) == 1.0 ? 1.0 : tanh(0.5 * pi * sinh(omega * Float64(y))))

function paper_de_dphi(y::Real; omega::Float64=4.0)
    y64 = Float64(y)
    Ay = 0.5 * pi * sinh(omega * y64)
    value = 0.5 * pi * omega * cosh(omega * y64) * exp(-2.0 * logcosh_stable(Ay))
    return isfinite(value) ? value : 0.0
end

paper_de_log1p_phi(y::Real; omega::Float64=4.0) =
    log(2.0) - log1pexp_stable(-pi * sinh(omega * Float64(y)))

paper_de_log1m_phi(y::Real; omega::Float64=4.0) =
    log(2.0) - log1pexp_stable(pi * sinh(omega * Float64(y)))

function stable_paper_de_log_ratio(y::Float64, eta::Float64; omega::Float64=4.0)
    t = 0.5 * (1.0 - eta)
    tau = y - (1.0 + y) * t
    if t <= sqrt(eps(Float64))
        return log(0.5 * (1.0 + y)) + log(paper_de_dphi(y; omega))
    end

    Ay = 0.5 * pi * sinh(omega * y)
    Atau = 0.5 * pi * sinh(omega * tau)
    xi2 = 0.5 * omega * (1.0 + y) * t
    midpoint = 0.5 * omega * (y + tau)
    xi3 = pi * cosh(midpoint) * sinh(xi2)

    log_delta_over_t =
        log(0.5 * pi * omega * (1.0 + y)) +
        logcosh_stable(midpoint) +
        stable_log_sinhc(xi3) +
        stable_log_sinhc(xi2) -
        logcosh_stable(Ay) -
        logcosh_stable(Atau)
    return log_delta_over_t - log(2.0)
end

function stable_paper_de_transformed_kernel_value(
    K,
    mu::Float64,
    y::Float64,
    eta::Float64;
    omega::Float64=4.0,
)
    t = 0.5 * (1.0 - eta)
    tau = y - (1.0 + y) * t
    Atau = 0.5 * pi * sinh(omega * tau)
    log_dphi_tau =
        log(0.5 * pi * omega) +
        logcosh_stable(omega * tau) -
        2.0 * logcosh_stable(Atau)
    log_ratio = stable_paper_de_log_ratio(y, eta; omega)
    value = exp(log_dphi_tau + (mu - 1.0) * log_ratio) *
            K(paper_de_phi(y; omega), paper_de_phi(tau; omega))
    return isfinite(value) ? value : NaN
end

function cheb_firstkind_nodes(n::Int)
    n >= 1 || error("need at least one Chebyshev node")
    return [cospi((j + 0.5) / n) for j in 0:(n - 1)]
end

function cheb_root_coeff_matrix(n::Int)
    B = Matrix{Float64}(undef, n, n)
    invn = inv(float(n))
    for k in 0:(n - 1)
        scale = k == 0 ? invn : 2.0 * invn
        for j in 0:(n - 1)
            B[k + 1, j + 1] = scale * cospi(k * (j + 0.5) / n)
        end
    end
    return B
end

function cheb2_coeffs_from_root_values(F::AbstractMatrix{<:Real})
    By = cheb_root_coeff_matrix(size(F, 1))
    Be = cheb_root_coeff_matrix(size(F, 2))
    return By * Matrix{Float64}(F) * transpose(Be)
end

function cheb_values_at(x::Float64, n::Int)
    T = Vector{Float64}(undef, n)
    if n == 0
        return T
    end
    T[1] = 1.0
    if n >= 2
        T[2] = x
        for k in 3:n
            T[k] = 2.0 * x * T[k - 1] - T[k - 2]
        end
    end
    return T
end

function eval_cheb2(C::AbstractMatrix{<:Real}, y::Float64, eta::Float64)
    Ty = cheb_values_at(y, size(C, 1))
    Te = cheb_values_at(eta, size(C, 2))
    return dot(Ty, Matrix(C) * Te)
end

function cheb_tail_estimate(C::AbstractMatrix{<:Real})
    n1, n2 = size(C)
    w1 = max(1, min(16, max(1, div(n1, 8))))
    w2 = max(1, min(16, max(1, div(n2, 8))))
    tail = 0.0

    for i in max(1, n1 - w1 + 1):n1, j in 1:n2
        tail = max(tail, abs(C[i, j]))
    end
    for i in 1:n1, j in max(1, n2 - w2 + 1):n2
        tail = max(tail, abs(C[i, j]))
    end
    return tail
end

function truncate_cheb2_coeffs(Cfull::Matrix{Float64}; tol::Float64, max_m=nothing, max_n=nothing)
    scale = max(1.0, maxabs_array(Cfull))
    threshold = tol * scale
    row_norms = [maximum(abs, @view Cfull[i, :]) for i in 1:size(Cfull, 1)]
    col_norms = [maximum(abs, @view Cfull[:, j]) for j in 1:size(Cfull, 2)]

    last_row = findlast(x -> x > threshold, row_norms)
    last_col = findlast(x -> x > threshold, col_norms)
    m = last_row === nothing ? 0 : last_row - 1
    n = last_col === nothing ? 0 : last_col - 1

    if max_m !== nothing
        m = min(m, max_m)
    end
    if max_n !== nothing
        n = min(n, max_n)
    end

    return copy(@view Cfull[1:(m + 1), 1:(n + 1)])
end

function random_cheb2_error(Ktilde, C; samples=32)
    rng = MersenneTwister(20260606)
    err = 0.0
    for _ in 1:samples
        # Keep random validation away from the algebraic endpoint y=-1.
        y = -0.98 + 1.98 * rand(rng)
        eta = -0.98 + 1.96 * rand(rng)
        err = max(err, abs(eval_cheb2(C, y, eta) - Ktilde(y, eta)))
    end
    return err
end

function adaptive_cheb2_coeffs(Ktilde; tol=1e-14, n0=17, nmax=129, max_m=nothing, max_n=nothing)
    n = n0
    last_full = Matrix{Float64}(undef, 0, 0)
    last_tail = Inf
    converged = false

    while true
        y = cheb_firstkind_nodes(n)
        eta = cheb_firstkind_nodes(n)
        F = Matrix{Float64}(undef, n, n)
        for b in 1:n, a in 1:n
            F[a, b] = Ktilde(y[a], eta[b])
        end

        Cfull = cheb2_coeffs_from_root_values(F)
        tail = cheb_tail_estimate(Cfull)
        scale = max(1.0, maxabs_array(Cfull))
        converged = tail <= tol * scale
        last_full = Cfull
        last_tail = tail

        if converged || n >= nmax
            break
        end
        n = min(2n - 1, nmax)
    end

    C = truncate_cheb2_coeffs(last_full; tol, max_m, max_n)
    return Cheb2Expansion(
        C,
        size(last_full),
        last_tail,
        maxabs_array(C),
        random_cheb2_error(Ktilde, C),
        converged,
    )
end

function fixed_cheb2_coeffs(Ktilde, ny::Int, neta::Int)
    ny >= 1 || error("ny must be positive")
    neta >= 1 || error("neta must be positive")

    y = cheb_firstkind_nodes(ny)
    eta = cheb_firstkind_nodes(neta)
    F = Matrix{Float64}(undef, ny, neta)
    for b in 1:neta, a in 1:ny
        F[a, b] = Ktilde(y[a], eta[b])
    end

    C = cheb2_coeffs_from_root_values(F)
    return Cheb2Expansion(
        C,
        size(C),
        cheb_tail_estimate(C),
        maxabs_array(C),
        random_cheb2_error(Ktilde, C),
        true,
    )
end

function build_Kphi_cheb_coeffs(
    K,
    phi=identity_phi,
    dphi=identity_dphi;
    mu::Real,
    tol::Float64=1e-14,
    n0::Int=17,
    nmax::Int=129,
    max_m=64,
    max_n=64,
    stable_de_ktilde::Bool=false,
    de_omega::Float64=4.0,
    Csize_y=nothing,
    Csize_eta=nothing,
)
    mu64 = Float64(mu)
    Ktilde = if stable_de_ktilde
        (y, eta) -> stable_paper_de_transformed_kernel_value(
            K,
            mu64,
            Float64(y),
            Float64(eta);
            omega=de_omega,
        )
    else
        (y, eta) -> transformed_kernel_value(K, phi, dphi, mu64, Float64(y), Float64(eta))
    end
    cheb = if Csize_y !== nothing || Csize_eta !== nothing
        ny = Csize_y === nothing ? Int(Csize_eta) : Int(Csize_y)
        neta = Csize_eta === nothing ? Int(Csize_y) : Int(Csize_eta)
        fixed_cheb2_coeffs(Ktilde, ny, neta)
    else
        adaptive_cheb2_coeffs(Ktilde; tol, n0, nmax, max_m, max_n)
    end
    info = (;
        C_size=size(cheb.C),
        full_grid_size=cheb.full_size,
        max_coeff=cheb.max_coeff,
        tail=cheb.tail,
        sample_error=cheb.sample_error,
        converged=cheb.converged,
    )
    return cheb.C, info, Ktilde
end

@inline function log1pexp_stable(x::Float64)
    if x > 0.0
        return x + log1p(exp(-x))
    else
        return log1p(exp(x))
    end
end

@inline function logcosh_stable(x::Float64)
    ax = abs(x)
    return ax + log1p(exp(-2.0 * ax)) - log(2.0)
end

@inline function de_log_distances(t::Float64)
    z = pi * sinh(t)
    log1m = log(2.0) - log1pexp_stable(z)
    log1p_eta = log(2.0) - log1pexp_stable(-z)
    return log1m, log1p_eta
end

function tanh_sinh_node(t::Float64)
    log_one_minus_eta, log_one_plus_eta = de_log_distances(t)
    eta_raw = tanh(0.5 * pi * sinh(t))
    rounded_to_right = eta_raw == 1.0
    rounded_to_left = eta_raw == -1.0

    eta = if rounded_to_right
        prevfloat(1.0)
    elseif rounded_to_left
        nextfloat(-1.0)
    else
        eta_raw
    end

    return TanhSinhNode(
        eta,
        log_one_minus_eta,
        log_one_plus_eta,
        rounded_to_right,
        rounded_to_left,
    )
end

function de_log_weight(mu::Float64, h::Float64, t::Float64)
    log1m, log1p_eta = de_log_distances(t)
    return log(h) + mu * log1m + log(pi / 2.0) + logcosh_stable(t) + log1p_eta
end

right_tail_estimate(mu::Float64, log1m::Float64) = exp(mu * log1m) / mu

function choose_de_extent(mu::Float64, sign::Int, h::Float64; tail_tol=1e-15, log_weight_cutoff=-60.0)
    sign == -1 || sign == 1 || error("sign must be -1 or 1")
    M = 0
    while true
        t = sign * M * h
        log1m, log1p_eta = de_log_distances(t)
        if sign > 0
            tail = right_tail_estimate(mu, log1m)
            if tail <= tail_tol
                return M, exp(log1m), log1m, tail
            end
        else
            reached_machine_endpoint = log1p_eta <= log(eps(Float64))
            reached_tiny_weight = de_log_weight(mu, h, t) <= log_weight_cutoff
            if reached_machine_endpoint && reached_tiny_weight
                return M, exp(log1p_eta), log1p_eta, 0.0
            end
        end
        M += 1
        M < 100_000 || error("DE truncation search did not converge")
    end
end

function neumaier_add!(sums::Vector{Float64}, comps::Vector{Float64}, idx::Int, value::Float64)
    t = sums[idx] + value
    if abs(sums[idx]) >= abs(value)
        comps[idx] += (sums[idx] - t) + value
    else
        comps[idx] += (value - t) + sums[idx]
    end
    sums[idx] = t
    return nothing
end

function jacobi_m1_0_values_at_eta(eta::Float64, kmax::Int)
    p = zeros(Float64, kmax + 1)
    p[1] = 1.0
    if kmax == 0
        return p
    end

    p[2] = 0.5 * (eta - 1.0)
    for k in 1:(kmax - 1)
        a = (k + 1.0) / (2.0 * k + 1.0)
        b = -1.0 / ((2.0 * k - 1.0) * (2.0 * k + 1.0))
        c = (k - 1.0) / (2.0 * k - 1.0)
        p[k + 2] = ((eta - b) * p[k + 1] - c * p[k]) / a
    end
    return p
end

function jacobi_m1_0_values_from_right_delta(delta::Float64, kmax::Int)
    p = zeros(Float64, kmax + 1)
    p[1] = 1.0
    if kmax == 0
        return p
    end

    if delta > sqrt(eps(Float64))
        return jacobi_m1_0_values_at_eta(1.0 - delta, kmax)
    end

    for k in 1:kmax
        p[k + 1] = -0.5 * k * delta
    end
    return p
end

function add_weighted_jacobi_values!(moments, comps, weight, pvals)
    for idx in eachindex(pvals)
        neumaier_add!(moments, comps, idx, weight * pvals[idx])
    end
    return nothing
end

function finish_compensated_blocks(blocks, kmax::Int)
    sums = zeros(Float64, kmax + 1)
    comps = zeros(Float64, kmax + 1)
    for (block_sums, block_comps) in blocks
        for idx in 1:(kmax + 1)
            neumaier_add!(sums, comps, idx, block_sums[idx])
            neumaier_add!(sums, comps, idx, block_comps[idx])
        end
    end
    return sums .+ comps
end

function de_initial_moments_once(
    mu::Float64,
    kmax::Int;
    h=0.025,
    tail_tol=1e-15,
    log_weight_cutoff=-60.0,
)
    kmax >= 0 || error("kmax must be nonnegative")
    Mleft, left_dist, _, _ = choose_de_extent(mu, -1, h; tail_tol, log_weight_cutoff)
    Mright, right_dist, right_final_log1m, right_final_tail =
        choose_de_extent(mu, 1, h; tail_tol, log_weight_cutoff)

    center = zeros(Float64, kmax + 1)
    center_comp = zeros(Float64, kmax + 1)
    left = zeros(Float64, kmax + 1)
    left_comp = zeros(Float64, kmax + 1)
    right = zeros(Float64, kmax + 1)
    right_comp = zeros(Float64, kmax + 1)
    nonfinite_seen = false
    rounded_endpoint_nodes = 0
    right_virtual_nodes = 0

    function add_node!(sums, comps, m)
        t = m * h
        node = tanh_sinh_node(t)
        weight = exp(de_log_weight(mu, h, t))
        if !isfinite(weight)
            nonfinite_seen = true
            return nothing
        end

        pvals = if isfinite(node.eta) &&
                   node.eta < 1.0 &&
                   node.eta > -1.0 &&
                   !node.rounded_to_right_endpoint &&
                   !node.rounded_to_left_endpoint
            jacobi_m1_0_values_at_eta(node.eta, kmax)
        elseif node.rounded_to_right_endpoint
            rounded_endpoint_nodes += 1
            right_virtual_nodes += 1
            delta = exp(node.log_one_minus_eta)
            jacobi_m1_0_values_from_right_delta(delta, kmax)
        elseif node.rounded_to_left_endpoint
            rounded_endpoint_nodes += 1
            jacobi_m1_0_values_at_eta(nextfloat(-1.0), kmax)
        else
            nonfinite_seen = true
            return nothing
        end

        add_weighted_jacobi_values!(sums, comps, weight, pvals)
        return nothing
    end

    add_node!(center, center_comp, 0)
    for m in 1:Mright
        add_node!(right, right_comp, m)
    end
    for m in 1:Mleft
        add_node!(left, left_comp, -m)
    end

    moments = finish_compensated_blocks(
        ((center, center_comp), (right, right_comp), (left, left_comp)),
        kmax,
    )

    k0_tail_correction_used = right_final_tail > 0.0
    if k0_tail_correction_used
        moments[1] += right_final_tail
    end

    diag = DEDiagnostics(
        h,
        Mleft + Mright + 1,
        left_dist,
        right_dist,
        nonfinite_seen,
        rounded_endpoint_nodes,
        right_virtual_nodes,
        right_final_log1m,
        right_final_tail,
        k0_tail_correction_used,
        DEConvergenceRow[],
    )
    return moments, diag
end

function closed_initial_moments(mu::Float64, kmax::Int)
    h = zeros(Float64, kmax + 1)
    h[1] = 2.0^mu / mu
    if kmax >= 1
        h[2] = -2.0^mu / (mu + 1.0)
        for k in 1:(kmax - 1)
            h[k + 2] = ((k - mu) / (k + mu + 1.0)) * h[k + 1]
        end
    end
    return h
end

function de_initial_moments(
    mu::Float64,
    kmax::Int;
    h=0.025,
    tail_tol=1e-15,
    log_weight_cutoff=-60.0,
)
    selected_h = min(h, 3.0 / max(kmax, 1))
    diagnostic_hs = Float64[0.08, 0.05, 0.035, 0.025]
    if all(abs(step - selected_h) > 10eps(Float64) for step in diagnostic_hs)
        push!(diagnostic_hs, selected_h)
        sort!(diagnostic_hs; rev=true)
    end

    closed = closed_initial_moments(mu, kmax)
    rows = DEConvergenceRow[]
    previous = nothing
    final_moments = nothing
    final_diag = nothing

    for step in diagnostic_hs
        moments, diag = de_initial_moments_once(
            mu,
            kmax;
            h=step,
            tail_tol,
            log_weight_cutoff,
        )
        change = previous === nothing ? NaN : maxabs_array(moments .- previous)
        closed_error = maxabs_array(moments .- closed)
        push!(
            rows,
            DEConvergenceRow(
                step,
                diag.node_count,
                diag.left_endpoint_distance,
                diag.right_endpoint_distance,
                change,
                closed_error,
                diag.nonfinite_seen,
                diag.rounded_endpoint_nodes,
                diag.right_virtual_nodes,
                diag.right_final_log1m,
                diag.right_final_tail_estimate,
                diag.k0_tail_correction_used,
            ),
        )

        if abs(step - selected_h) <= 10eps(Float64)
            final_moments = moments
            final_diag = diag
        end
        previous = moments
    end

    if final_moments === nothing || final_diag === nothing
        final_moments, final_diag = de_initial_moments_once(
            mu,
            kmax;
            h=selected_h,
            tail_tol,
            log_weight_cutoff,
        )
    end

    diag = DEDiagnostics(
        final_diag.h,
        final_diag.node_count,
        final_diag.left_endpoint_distance,
        final_diag.right_endpoint_distance,
        final_diag.nonfinite_seen,
        final_diag.rounded_endpoint_nodes,
        final_diag.right_virtual_nodes,
        final_diag.right_final_log1m,
        final_diag.right_final_tail_estimate,
        final_diag.k0_tail_correction_used,
        rows,
    )
    return final_moments, diag
end

function stable_de_H000(
    mu::Real,
    Kmax::Int;
    tol::Float64=1e-15,
    h::Float64=0.025,
    maxlevel::Int=6,
    log_weight_cutoff::Float64=-60.0,
    return_info::Bool=false,
)
    Kmax >= 0 || error("Kmax must be nonnegative")
    maxlevel >= 1 || error("maxlevel must be positive")
    moments, diag = de_initial_moments(
        Float64(mu),
        Kmax;
        h,
        tail_tol=tol,
        log_weight_cutoff,
    )
    return return_info ? (moments, diag) : moments
end

function build_r0_layer_from_H00(H00::AbstractVector{<:Real}, nmax::Int, Kext::Int)
    nmax >= 0 || error("nmax must be nonnegative")
    Kext >= 0 || error("Kext must be nonnegative")

    available_kmax = length(H00) - 1
    available_kmax >= Kext + nmax ||
        error("H00 must be available through k=$(Kext + nmax), got $available_kmax")

    hwork = zeros(Float64, nmax + 1, available_kmax + 1)
    hwork[1, :] .= H00

    if nmax >= 1
        validmax = available_kmax - 1
        hwork[2, 1] = hwork[1, 1] + 2.0 * hwork[1, 2]
        for k in 1:validmax
            a = (k + 1.0) / (2.0 * k + 1.0)
            b = -1.0 / ((2.0 * k - 1.0) * (2.0 * k + 1.0))
            c = (k - 1.0) / (2.0 * k - 1.0)
            hwork[2, k + 1] =
                a * hwork[1, k + 2] + b * hwork[1, k + 1] + c * hwork[1, k]
        end
    end

    if nmax >= 2
        for j in 1:(nmax - 1)
            validmax = available_kmax - (j + 1)
            hwork[j + 2, 1] =
                2.0 * (hwork[j + 1, 1] + 2.0 * hwork[j + 1, 2]) - hwork[j, 1]
            for k in 1:validmax
                a = (k + 1.0) / (2.0 * k + 1.0)
                b = -1.0 / ((2.0 * k - 1.0) * (2.0 * k + 1.0))
                c = (k - 1.0) / (2.0 * k - 1.0)
                hwork[j + 2, k + 1] =
                    2.0 * (
                        a * hwork[j + 1, k + 2] +
                        b * hwork[j + 1, k + 1] +
                        c * hwork[j + 1, k]
                    ) - hwork[j, k + 1]
            end
        end
    end

    return copy(@view hwork[:, 1:(Kext + 1)])
end

function build_H_fixed_j_from_r0(j::Int, H_j_r0::AbstractVector{<:Real}, Kext::Int; mu::Real)
    j >= 0 || error("j must be nonnegative")
    Kext >= 0 || error("Kext must be nonnegative")
    length(H_j_r0) >= Kext + 1 || error("H_j_r0 is too short for Kext=$Kext")

    Hjr = zeros(Float64, Kext + 1, Kext + 1)
    for k in 0:Kext
        Hjr[1, k + 1] = H_j_r0[k + 1]
    end

    # Fixed-j four-point (r,k) recurrence from the current derivation.
    for r in 0:(Kext - 1)
        Hjr[r + 2, r + 2] =
            Hjr[r + 1, r + 1] + Hjr[r + 1, r + 2] / (2.0 * r + 1.0)

        for k in (r + 1):(Kext - 1)
            Hjr[r + 2, k + 2] =
                Hjr[r + 1, k + 1] +
                ((k - r + 1.0) / (k + r + 1.0)) * Hjr[r + 1, k + 2] -
                ((k - r - 1.0) / (k + r + 1.0)) * Hjr[r + 2, k + 1]
        end
    end

    return Hjr
end

function accumulate_Hcomp_streaming(C::Matrix{Float64}, H_r0::Matrix{Float64}, Kext::Int; mu::Real)
    m = size(C, 1) - 1
    n = size(C, 2) - 1
    size(H_r0, 1) >= n + 1 || error("H_r0 does not contain all j layers")
    size(H_r0, 2) >= Kext + 1 || error("H_r0 does not contain all k columns")

    Hcomp = zeros(Float64, m + 1, Kext + 1, Kext + 1)
    for j in 0:n
        Hjr = build_H_fixed_j_from_r0(j, @view(H_r0[j + 1, :]), Kext; mu)
        for i in 0:m
            cij = C[i + 1, j + 1]
            if cij != 0.0
                @inbounds for k in 0:Kext
                    for r in 0:k
                        Hcomp[i + 1, r + 1, k + 1] += cij * Hjr[r + 1, k + 1]
                    end
                end
            end
        end
    end
    return Hcomp
end

function assemble_G_from_Hcomp(Hcomp::Array{Float64, 3}, N::Int)
    m = size(Hcomp, 1) - 1
    Kavailable = size(Hcomp, 3) - 1
    Kavailable >= N - 1 || error("Hcomp has insufficient k range")

    G = zeros(Float64, N + 1, N)

    for ell in 0:N
        if ell <= m
            G[ell + 1, 1] = Hcomp[ell + 1, 1, 1]
        end
    end

    for k in 1:(N - 1)
        for ell in 0:N
            scale = (2.0 - (ell == 0 ? 1.0 : 0.0)) / 2.0
            s = 0.0
            for i in 0:m
                r1 = i + ell
                if r1 <= k
                    s += (k / (k + r1)) * Hcomp[i + 1, r1 + 1, k + 1]
                end

                r2 = abs(i - ell)
                if r2 <= k
                    s += (k / (k + r2)) * Hcomp[i + 1, r2 + 1, k + 1]
                end
            end
            G[ell + 1, k + 1] = scale * s
        end
    end

    return G
end

function multiply_by_one_plus_y_over_two(G::AbstractMatrix{<:Real}, N::Int)
    size(G, 1) >= N + 1 || error("G must have at least N+1 rows")
    size(G, 2) >= N || error("G must have at least N columns")

    A = zeros(Float64, N, N)
    for k in 1:N
        A[1, k] = 0.5 * G[1, k] + 0.25 * G[2, k]
        if N >= 2
            A[2, k] = 0.5 * G[2, k] + 0.5 * G[1, k] + 0.25 * G[3, k]
        end
        for ell in 2:(N - 1)
            A[ell + 1, k] =
                0.5 * G[ell + 1, k] + 0.25 * G[ell, k] + 0.25 * G[ell + 2, k]
        end
    end
    return A
end

const DEFAULT_C0_DE = 17

const logcosh = logcosh_stable
sinhc(x::Real) = abs(Float64(x)) < 1e-5 ? 1.0 + Float64(x)^2 / 6.0 + Float64(x)^4 / 120.0 : sinh(Float64(x)) / Float64(x)
const log_sinhc = stable_log_sinhc
phi_de(y::Real; omega::Real=4.0) = paper_de_phi(y; omega=Float64(omega))

function log_phi_prime_de(y::Real; omega::Real=4.0)
    y64 = Float64(y)
    if y64 == -1.0 || y64 == 1.0
        return -Inf
    end
    A = 0.5 * pi * sinh(Float64(omega) * y64)
    return log(0.5 * pi * Float64(omega)) +
           logcosh_stable(Float64(omega) * y64) -
           2.0 * logcosh_stable(A)
end

function de_ktilde_value(K, mu::Float64, omega::Float64, y::Float64, eta::Float64)
    eta < 1.0 || throw(ArgumentError("eta samples must avoid eta=1"))
    tau = duffy_tau(y, eta)
    x = phi_de(y; omega)
    t = phi_de(tau; omega)
    kval = K(x, t)
    isfinite(kval) || error("K returned non-finite value at y=$y, eta=$eta, x=$x, t=$t")
    logp_tau = log_phi_prime_de(tau; omega)
    log_ratio = if 1.0 - eta <= sqrt(eps(Float64))
        log(0.5 * (1.0 + y)) + log_phi_prime_de(y; omega)
    else
        stable_paper_de_log_ratio(y, eta; omega)
    end
    val = exp(logp_tau + (mu - 1.0) * log_ratio) * Float64(kval)
    isfinite(val) || error("Ktilde_phi became non-finite at y=$y, eta=$eta, tau=$tau")
    return val
end

function cheb2_tail_metrics(C::AbstractMatrix{<:Real})
    ny, ne = size(C)
    wy = max(1, min(8, div(ny + 7, 8)))
    we = max(1, min(8, div(ne + 7, 8)))
    row_tail = maximum(abs, @view C[max(1, ny - wy + 1):ny, :])
    col_tail = maximum(abs, @view C[:, max(1, ne - we + 1):ne])
    corner_tail = maximum(abs, @view C[max(1, ny - wy + 1):ny, max(1, ne - we + 1):ne])
    return row_tail, col_tail, corner_tail
end

function _sample_de_ktilde_grid(K, mu::Float64, omega::Float64, ny::Int, ne::Int)
    ys = cheb_firstkind_nodes(ny)
    etas = cheb_firstkind_nodes(ne)
    F = Matrix{Float64}(undef, ny, ne)
    for b in 1:ne, a in 1:ny
        F[a, b] = de_ktilde_value(K, mu, omega, ys[a], etas[b])
    end
    return F
end

_next_adaptive_size(n::Int, nmax::Int) = n >= nmax ? nmax : min(2n - 1, nmax)

function build_C_adaptive_de(
    K;
    mu,
    omega::Real=4.0,
    tol::Float64=1e-14,
    Csize_y=nothing,
    Csize_eta=nothing,
    Cmax_y::Int=513,
    Cmax_eta::Int=513,
)
    mu64 = Float64(mu)
    omega64 = Float64(omega)
    0.0 < mu64 < 1.0 || throw(ArgumentError("mu must satisfy 0 < mu < 1"))
    Cmax_y >= 3 || throw(ArgumentError("Cmax_y must be at least 3"))
    Cmax_eta >= 3 || throw(ArgumentError("Cmax_eta must be at least 3"))

    fixed = Csize_y !== nothing || Csize_eta !== nothing
    ny = fixed ? Int(Csize_y === nothing ? Csize_eta : Csize_y) : min(DEFAULT_C0_DE, Cmax_y)
    ne = fixed ? Int(Csize_eta === nothing ? Csize_y : Csize_eta) : min(DEFAULT_C0_DE, Cmax_eta)
    ny >= 1 && ne >= 1 || throw(ArgumentError("Csize_y and Csize_eta must be positive"))

    C = Matrix{Float64}(undef, 0, 0)
    row_tail = col_tail = corner_tail = max_abs_C = Inf
    tail_abs = tail_rel = Inf
    converged = false

    while true
        F = _sample_de_ktilde_grid(K, mu64, omega64, ny, ne)
        C = cheb2_coeffs_from_root_values(F)
        max_abs_C = maxabs_array(C)
        row_tail, col_tail, corner_tail = cheb2_tail_metrics(C)
        tail_abs = max(row_tail, col_tail, corner_tail)
        coeff_scale = max(max_abs_C, floatmin(Float64))
        tail_rel = tail_abs / coeff_scale
        converged = tail_rel <= tol || tail_abs <= 100.0 * eps(Float64) * coeff_scale
        if fixed || converged || (ny >= Cmax_y && ne >= Cmax_eta)
            break
        end
        grow_y = row_tail / coeff_scale > tol && ny < Cmax_y
        grow_e = col_tail / coeff_scale > tol && ne < Cmax_eta
        ny = grow_y || (!grow_e && ny < Cmax_y) ? _next_adaptive_size(ny, Cmax_y) : ny
        ne = grow_e || (!grow_y && ne < Cmax_eta) ? _next_adaptive_size(ne, Cmax_eta) : ne
    end

    if fixed && tail_rel > tol
        @warn "Using user-forced fixed Csize; Chebyshev tail exceeds tol, so banded truncation is not guaranteed near machine precision" C_size=size(C) tail_abs tail_rel tol
    elseif !fixed && !converged
        @warn "DE transformed-kernel Chebyshev expansion did not converge before Cmax" C_size=size(C) Cmax_y Cmax_eta tail_abs tail_rel tol
    end
    info = (;
        C_size=size(C),
        Cmax_y,
        Cmax_eta,
        max_abs_C,
        row_tail,
        col_tail,
        corner_tail,
        tail_abs,
        tail_rel,
        adaptive_converged=!fixed && converged,
        fixed_C=fixed,
    )
    return C, info
end

function build_postmult_coeffs_de(C::AbstractMatrix{<:Real})
    My, Meta = size(C)
    P = zeros(Float64, My + 1, Meta)
    for j in 1:Meta
        for i in 0:My
            val = 0.0
            if i <= My - 1
                val += 0.5 * C[i + 1, j]
            end
            if i >= 1 && i - 1 <= My - 1
                val += 0.25 * C[i, j]
            end
            if i + 1 <= My - 1
                val += 0.25 * C[i + 2, j]
            end
            if i == 1
                val += 0.25 * C[1, j]
            end
            P[i + 1, j] = val
        end
    end
    return P
end

@inline function add_A_by_d!(Aband::Matrix{Float64}, k::Int, d::Int, value::Float64, lower_bw::Int, upper_bw::Int)
    value == 0.0 && return nothing
    0 <= k <= size(Aband, 2) - 1 || return nothing
    -lower_bw <= d <= upper_bw || return nothing
    ell = k - d
    0 <= ell <= size(Aband, 2) - 1 || return nothing
    scale = (2.0 - (ell == 0 ? 1.0 : 0.0)) / 2.0
    Aband[d + lower_bw + 1, k + 1] += scale * value
    return nothing
end

function band_to_dense(Aband::AbstractMatrix{<:Real}, N::Int, lower_bw::Int, upper_bw::Int)
    A = zeros(Float64, N, N)
    for k in 0:(N - 1)
        for d in -lower_bw:upper_bw
            ell = k - d
            0 <= ell <= N - 1 || continue
            A[ell + 1, k + 1] = Aband[d + lower_bw + 1, k + 1]
        end
    end
    return A
end

function h0_line_s_recurrence_de(r::Int, mu::Float64, smax::Int, Br::Float64)
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

@inline function eta_coeffs_de(r::Int, s::Int)
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

@inline function eta_times_line_de(vals::AbstractVector{Float64}, r::Int, s::Int)
    a, b, c = eta_coeffs_de(r, s)
    out = b * vals[s + 1]
    if s + 2 <= length(vals)
        out += a * vals[s + 2]
    end
    if s >= 1
        out += c * vals[s]
    end
    return out
end

function build_Hjs_line(
    r::Int,
    mu::Float64,
    S::Int,
    Sguard::Int,
    Br::Float64;
    Meta::Int,
    h0_init::Symbol=:s_recurrence,
    prec_bits::Int=1536,
)
    Meta >= 1 || throw(ArgumentError("Meta must be positive"))
    Sguard >= S || throw(ArgumentError("Sguard must be at least S"))
    h0_init === :s_recurrence ||
        throw(ArgumentError("only h0_init=:s_recurrence is supported in this demo path"))
    _ = prec_bits

    H0_ext = h0_line_s_recurrence_de(r, mu, Sguard + 1, Br)
    Hwork = zeros(Float64, Meta, Sguard + 1)
    for s in 0:Sguard
        Hwork[1, s + 1] = H0_ext[s + 1]
    end
    if Meta >= 2
        for s in 0:Sguard
            Hwork[2, s + 1] = eta_times_line_de(H0_ext, r, s)
        end
    end
    for j in 1:(Meta - 2)
        prev = @view Hwork[j, :]
        curr = @view Hwork[j + 1, :]
        next = @view Hwork[j + 2, :]
        for s in 0:Sguard
            next[s + 1] = 2.0 * eta_times_line_de(curr, r, s) - prev[s + 1]
        end
    end
    return copy(@view Hwork[:, 1:(S + 1)])
end

function build_Hjs_line!(
    dest::AbstractMatrix{Float64},
    r::Int,
    mu::Float64,
    S::Int,
    Sguard::Int,
    Br::Float64;
    Meta::Int,
    h0_init::Symbol=:s_recurrence,
    prec_bits::Int=1536,
    layout::Symbol=:js,
)
    Hjs = build_Hjs_line(r, mu, S, Sguard, Br; Meta, h0_init, prec_bits)
    if layout === :js
        size(dest, 1) >= Meta || throw(ArgumentError("dest has too few rows"))
        size(dest, 2) >= S + 1 || throw(ArgumentError("dest has too few columns"))
        @views dest[1:Meta, 1:(S + 1)] .= Hjs
    elseif layout === :sj
        size(dest, 1) >= S + 1 || throw(ArgumentError("dest has too few rows"))
        size(dest, 2) >= Meta || throw(ArgumentError("dest has too few columns"))
        @inbounds for j in 1:Meta, s in 0:S
            dest[s + 1, j] = Hjs[j, s + 1]
        end
    else
        throw(ArgumentError("layout must be :js or :sj"))
    end
    return dest
end

mutable struct WeakDEHLineRingCache
    Hcache::Array{Float64, 3}
    valid_r::Vector{Int}
    Bvals::Vector{Float64}
    S::Int
    Sguard::Int
    Rcache::Int
    Meta::Int
end

mutable struct WeakDEFFTPlan
    P::Matrix{Float64}
    mu::Float64
    N::Int
    IP_DEG::Int
    J_DEG::Int
    lower_bw::Int
    upper_bw::Int
    S_A::Int
    Sguard_A::Int
    Lfft::Int
    Lfreq::Int
    block_cols::Int
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
    Hring::WeakDEHLineRingCache
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

_timer_start_de(collect_stats::Bool) = collect_stats ? time_ns() : 0
_timer_elapsed_de(collect_stats::Bool, t0::Integer) =
    collect_stats ? (time_ns() - t0) / 1.0e9 : 0.0

function _resolve_fftw_flags_de(fftw_flags)
    HAS_FFTW_DE[] ||
        error("FFTW.jl is required for assembly=:narrow_rfft_absorbM. Add FFTW to this Julia environment or use assembly=:narrow_direct_absorbM.")
    return fftw_flags === nothing ? FFTW.MEASURE : fftw_flags
end

function beta_r_values_de(mu::Float64, N::Int)
    Bvals = Vector{Float64}(undef, N)
    Bvals[1] = 1.0 / mu
    for r in 0:(N - 2)
        Bvals[r + 2] = (r + 1.0) / (r + mu + 1.0) * Bvals[r + 1]
    end
    return Bvals
end

function ensure_Hline_de!(ring::WeakDEHLineRingCache, r::Int, mu::Float64, h0_init::Symbol, prec_bits::Int)
    idx = mod(r, ring.Rcache) + 1
    if ring.valid_r[idx] != r
        Hview = @view ring.Hcache[:, :, idx]
        build_Hjs_line!(
            Hview,
            r,
            mu,
            ring.S,
            ring.Sguard,
            ring.Bvals[r + 1];
            Meta=ring.Meta,
            h0_init,
            prec_bits,
            layout=:sj,
        )
        ring.valid_r[idx] = r
    end
    return idx
end

function mul_P_Hsj_de!(
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

function precompute_rfft_spectra_de!(
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

function build_left_corner_cache_de!(left_p::Vector{Int}, left_col::Vector{Int}, left_value::Vector{Float64}, plan::WeakDEFFTPlan)
    empty!(left_p)
    empty!(left_col)
    empty!(left_value)
    for r in 1:min(plan.IP_DEG, plan.N - 1)
        idx = ensure_Hline_de!(plan.Hring, r, plan.mu, plan.h0_init, plan.prec_bits)
        smax = min(plan.S_A, plan.N - 1 - r)
        Hview = @view plan.Hring.Hcache[:, :, idx]
        Hcomp = @view plan.HcompP[:, 1:(smax + 1)]
        mul_P_Hsj_de!(Hcomp, plan.P, Hview, smax)
        @inbounds for s in 0:smax
            k = r + s
            weight = k / (k + r)
            for i in r:plan.IP_DEG
                d = 2r + s - i
                -plan.lower_bw <= d <= plan.upper_bw || continue
                ell = k - d
                0 <= ell <= plan.N - 1 || continue
                scale = (2.0 - (ell == 0 ? 1.0 : 0.0)) / 2.0
                push!(left_p, d + plan.lower_bw + 1)
                push!(left_col, k + 1)
                push!(left_value, scale * weight * Hcomp[i + 1, s + 1])
            end
        end
    end
    return nothing
end

function plan_weak_de_fft(
    P::Matrix{Float64},
    mu,
    N::Int;
    upper_bw,
    fftw_flags=nothing,
    block_cols::Int=64,
    h0_init::Symbol=:s_recurrence,
    prec_bits::Int=1536,
)
    N >= 2 || throw(ArgumentError("N must be at least 2"))
    block_cols >= 1 || throw(ArgumentError("block_cols must be positive"))
    ubw = Int(upper_bw)
    ubw >= 0 || throw(ArgumentError("upper_bw must be nonnegative"))
    flags = _resolve_fftw_flags_de(fftw_flags)
    mu64 = Float64(mu)
    IP_DEG = size(P, 1) - 1
    J_DEG = size(P, 2) - 1
    lower_bw = IP_DEG
    S_A = ubw + IP_DEG
    Sguard_A = S_A + J_DEG + 2
    Lfft = nextpow(2, IP_DEG + S_A + 2)
    Lfreq = div(Lfft, 2) + 1
    max_nb = min(block_cols, max(1, N - 1))

    Fp = Matrix{ComplexF64}(undef, Lfreq, J_DEG + 1)
    Fr = Matrix{ComplexF64}(undef, Lfreq, J_DEG + 1)
    tmp = zeros(Float64, Lfft)
    plan_tmp_rfft = FFTW.plan_rfft(tmp; flags)
    precompute_rfft_spectra_de!(Fp, Fr, P, tmp, plan_tmp_rfft)

    Qbuf = zeros(Float64, Lfft, J_DEG + 1, max_nb)
    Qhat = zeros(ComplexF64, Lfreq, J_DEG + 1, max_nb)
    Uhat = zeros(ComplexF64, Lfreq, max_nb)
    Lhat = zeros(ComplexF64, Lfreq, max_nb)
    Ubuf = zeros(Float64, Lfft, max_nb)
    Lbuf = zeros(Float64, Lfft, max_nb)
    q_highwater = fill(-1, J_DEG + 1, max_nb)
    Rcache = S_A + max_nb + 1
    Hring = WeakDEHLineRingCache(
        zeros(Float64, S_A + 1, J_DEG + 1, Rcache),
        fill(-1, Rcache),
        beta_r_values_de(mu64, N),
        S_A,
        Sguard_A,
        Rcache,
        J_DEG + 1,
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

    plan = WeakDEFFTPlan(
        P, mu64, N, IP_DEG, J_DEG, lower_bw, ubw, S_A, Sguard_A, Lfft, Lfreq,
        block_cols, h0_init, prec_bits, flags, Fp, Fr, Qbuf, Qhat, Uhat, Lhat,
        Ubuf, Lbuf, q_highwater, Hring, HcompP, scratch_ridx, scratch_weight,
        left_p, left_col, left_value, plan_q_fwd, plan_u_inv, plan_l_inv,
        memory_estimate,
    )
    build_left_corner_cache_de!(plan.left_p, plan.left_col, plan.left_value, plan)
    plan.memory_estimate += sizeof(plan.left_p) + sizeof(plan.left_col) + sizeof(plan.left_value)
    return plan
end

function add_k0_column_absorbM_de!(Aband::Matrix{Float64}, plan::WeakDEFFTPlan)
    idx0 = ensure_Hline_de!(plan.Hring, 0, plan.mu, plan.h0_init, plan.prec_bits)
    H0 = @view plan.Hring.Hcache[:, :, idx0]
    Hcomp0 = @view plan.HcompP[:, 1:1]
    mul_P_Hsj_de!(Hcomp0, plan.P, H0, 0)
    @inbounds for i in 0:min(plan.lower_bw, plan.N - 1)
        Aband[-i + plan.lower_bw + 1, 1] += plan.HcompP[i + 1, 1]
    end
    return nothing
end

function apply_left_corner_correction!(Aband::Matrix{Float64}, plan::WeakDEFFTPlan)
    @inbounds for n in eachindex(plan.left_value)
        Aband[plan.left_p[n], plan.left_col[n]] += plan.left_value[n]
    end
    return nothing
end

function assemble_weak_de_fft!(Aband::Matrix{Float64}, plan::WeakDEFFTPlan; return_stats::Bool=false)
    fill!(Aband, 0.0)
    time_fill_Qmat = 0.0
    time_forward_fft = 0.0
    time_freq_accum = 0.0
    time_inverse_fft = 0.0
    time_write_A = 0.0
    time_left_corner = 0.0

    add_k0_column_absorbM_de!(Aband, plan)
    for k0 in 1:plan.block_cols:(plan.N - 1)
        k1 = min(plan.N - 1, k0 + plan.block_cols - 1)
        nb = k1 - k0 + 1

        t0 = _timer_start_de(return_stats)
        @inbounds for b in 1:nb
            k = k0 + b - 1
            smax = min(plan.S_A, k)
            for j in 1:(plan.J_DEG + 1)
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
            idx = ensure_Hline_de!(plan.Hring, r, plan.mu, plan.h0_init, plan.prec_bits)
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
            for j in 1:(plan.J_DEG + 1)
                for s in 0:smax
                    idx = plan.scratch_ridx[s + 1, b]
                    plan.Qbuf[s + 1, j, b] =
                        plan.scratch_weight[s + 1, b] * plan.Hring.Hcache[s + 1, j, idx]
                end
            end
        end
        time_fill_Qmat += _timer_elapsed_de(return_stats, t0)

        t0 = _timer_start_de(return_stats)
        mul!(plan.Qhat, plan.plan_q_fwd, plan.Qbuf)
        time_forward_fft += _timer_elapsed_de(return_stats, t0)

        t0 = _timer_start_de(return_stats)
        @inbounds for p in 1:plan.Lfreq
            for b in 1:nb
                u = 0.0 + 0.0im
                l = 0.0 + 0.0im
                @simd for j in 1:(plan.J_DEG + 1)
                    q = plan.Qhat[p, j, b]
                    u += plan.Fp[p, j] * q
                    l += plan.Fr[p, j] * q
                end
                plan.Uhat[p, b] = u
                plan.Lhat[p, b] = l
            end
        end
        time_freq_accum += _timer_elapsed_de(return_stats, t0)

        t0 = _timer_start_de(return_stats)
        mul!(plan.Ubuf, plan.plan_u_inv, plan.Uhat)
        mul!(plan.Lbuf, plan.plan_l_inv, plan.Lhat)
        time_inverse_fft += _timer_elapsed_de(return_stats, t0)

        t0 = _timer_start_de(return_stats)
        @inbounds for b in 1:nb
            k = k0 + b - 1
            for d in -plan.lower_bw:plan.upper_bw
                value = 0.0
                d >= 0 && (value += plan.Ubuf[d + 1, b])
                idx = plan.IP_DEG + d
                0 <= idx < plan.Lfft && (value += plan.Lbuf[idx + 1, b])
                add_A_by_d!(Aband, k, d, value, plan.lower_bw, plan.upper_bw)
            end
        end
        time_write_A += _timer_elapsed_de(return_stats, t0)
    end

    t0 = _timer_start_de(return_stats)
    apply_left_corner_correction!(Aband, plan)
    time_left_corner += _timer_elapsed_de(return_stats, t0)

    stats = (;
        P_size=size(plan.P),
        lower_bw=plan.lower_bw,
        upper_bw=plan.upper_bw,
        S_A=plan.S_A,
        Sguard_A=plan.Sguard_A,
        band_storage_size=size(Aband),
        bandwidth_rule=:explicit_upper_bw,
        absorbs_final_multiplier=true,
        fft_used=true,
        block_cols=plan.block_cols,
        fftw_flags=plan.fftw_flags,
        Lfft=plan.Lfft,
        Lfreq=plan.Lfreq,
        memory_estimate=plan.memory_estimate + sizeof(Aband),
        time_fill_Qmat,
        time_forward_fft,
        time_freq_accum,
        time_inverse_fft,
        time_write_A,
        time_left_corner,
    )
    return return_stats ? (Aband, stats) : Aband
end

function assemble_A_narrow_rfft_absorbM_de(
    P::Matrix{Float64},
    mu::Float64,
    N::Int;
    upper_bw,
    block_cols::Int=64,
    fftw_flags=nothing,
    return_info::Bool=false,
    h0_init::Symbol=:s_recurrence,
    prec_bits::Int=1536,
)
    plan = plan_weak_de_fft(
        P,
        mu,
        N;
        upper_bw,
        fftw_flags,
        block_cols,
        h0_init,
        prec_bits,
    )
    Aband = zeros(Float64, plan.lower_bw + plan.upper_bw + 1, N)
    return assemble_weak_de_fft!(Aband, plan; return_stats=return_info)
end

function assemble_A_narrow_direct_absorbM_de(
    C::Matrix{Float64},
    mu::Float64,
    N::Int;
    upper_bw=nothing,
    h0_init::Symbol=:s_recurrence,
    prec_bits::Int=1536,
)
    N >= 2 || throw(ArgumentError("N must be at least 2"))
    P = build_postmult_coeffs_de(C)
    IP_DEG = size(P, 1) - 1
    lower_bw = IP_DEG
    bandwidth_rule = upper_bw === nothing ? :C_rows : :explicit_upper_bw
    ubw = upper_bw === nothing ? size(C, 1) : Int(upper_bw)
    ubw >= 0 || throw(ArgumentError("upper_bw must be nonnegative"))
    S_A = ubw + IP_DEG
    J_DEG = size(P, 2) - 1
    Sguard_A = S_A + J_DEG + 2
    Aband = zeros(Float64, lower_bw + ubw + 1, N)

    Br = 1.0 / mu
    for r in 0:(N - 1)
        Hjs = build_Hjs_line(r, mu, S_A, Sguard_A, Br; Meta=size(P, 2), h0_init, prec_bits)
        smax = min(S_A, N - 1 - r)
        HcompP = P * @view(Hjs[:, 1:(smax + 1)])
        for s in 0:smax
            k = r + s
            if k == 0
                for i in 0:min(IP_DEG, N - 1)
                    d = -i
                    Aband[d + lower_bw + 1, 1] += HcompP[i + 1, 1]
                end
                continue
            end

            weight = k / (k + r)
            for i in 0:IP_DEG
                h = weight * HcompP[i + 1, s + 1]
                h == 0.0 && continue

                ell1 = r - i
                if ell1 >= 0
                    d1 = s + i
                    d1 <= ubw && add_A_by_d!(Aband, k, d1, h, lower_bw, ubw)
                end

                d2 = s - i
                -lower_bw <= d2 <= ubw && add_A_by_d!(Aband, k, d2, h, lower_bw, ubw)

                if r != 0 && i >= r
                    d3 = 2r + s - i
                    -lower_bw <= d3 <= ubw && add_A_by_d!(Aband, k, d3, h, lower_bw, ubw)
                end
            end
        end
        if r < N - 1
            Br *= (r + 1.0) / (r + mu + 1.0)
        end
    end

    info = (;
        P_size=size(P),
        lower_bw,
        upper_bw=ubw,
        S_A,
        Sguard_A,
        band_storage_size=size(Aband),
        bandwidth_rule,
        absorbs_final_multiplier=true,
        fft_used=false,
        block_cols=missing,
        fftw_flags=missing,
    )
    return Aband, info
end

function _assemble_dense_full_from_C(C::Matrix{Float64}, mu::Float64, N::Int)
    m = size(C, 1) - 1
    n = size(C, 2) - 1
    Kext_guard = 4
    Kext = N + n + m + Kext_guard
    H00_Kmax = Kext + n

    H00 = stable_de_H000(mu, H00_Kmax; tol=1e-15, h=0.025)
    H_r0 = build_r0_layer_from_H00(H00, n, Kext)
    Hcomp = accumulate_Hcomp_streaming(C, H_r0, Kext; mu)
    G = assemble_G_from_Hcomp(Hcomp, N)
    A = multiply_by_one_plus_y_over_two(G, N)
    info = (;
        H00_finite=allfinite_array(H00),
        Hcomp_finite=allfinite_array(Hcomp),
        G_finite=allfinite_array(G),
        A_finite=allfinite_array(A),
        max_abs_A=maxabs_array(A),
        min_positive_abs_A=min_abs_nonzero(A),
        Kext,
        H00_Kmax,
    )
    return A, info
end

function _info_for_decheb(; mu, omega, N, C, Cinfo, Aout, output, assembly_requested, assembly_used,
        time_build_C, time_assembly, time_total, assembly_info)
    return (;
        mu=Float64(mu),
        omega=Float64(omega),
        N,
        assembly=assembly_used,
        assembly_requested,
        assembly_used,
        assembly_region=assembly_used === :dense_full ? :dense_full : :banded_effective,
        C_size=Cinfo.C_size,
        Cmax_y=Cinfo.Cmax_y,
        Cmax_eta=Cinfo.Cmax_eta,
        fixed_C=Cinfo.fixed_C,
        C,
        max_abs_C=Cinfo.max_abs_C,
        row_tail=Cinfo.row_tail,
        col_tail=Cinfo.col_tail,
        corner_tail=Cinfo.corner_tail,
        tail_abs=Cinfo.tail_abs,
        tail_rel=Cinfo.tail_rel,
        adaptive_converged=Cinfo.adaptive_converged,
        P_size=get(assembly_info, :P_size, missing),
        lower_bw=get(assembly_info, :lower_bw, missing),
        upper_bw=get(assembly_info, :upper_bw, missing),
        bandwidth_rule=get(assembly_info, :bandwidth_rule, missing),
        band_storage_size=get(assembly_info, :band_storage_size, missing),
        S_A=get(assembly_info, :S_A, missing),
        Sguard_A=get(assembly_info, :Sguard_A, missing),
        absorbs_final_multiplier=get(assembly_info, :absorbs_final_multiplier, true),
        fft_used=get(assembly_info, :fft_used, false),
        block_cols=get(assembly_info, :block_cols, missing),
        fftw_flags=get(assembly_info, :fftw_flags, missing),
        H00_finite=get(assembly_info, :H00_finite, missing),
        Hcomp_finite=get(assembly_info, :Hcomp_finite, missing),
        G_finite=get(assembly_info, :G_finite, missing),
        A_finite=allfinite_array(Aout),
        max_abs_A=maxabs_array(Aout),
        min_positive_abs_A=isa(Aout, AbstractMatrix) ? min_abs_nonzero(Aout) : missing,
        time_build_C,
        time_assembly,
        time_total,
        output,
        final_matrix_includes_external_factor=true,
        external_factor="(1+y)/2",
    )
end

"""
    weak_voltmat_decheb(K; mu, N, omega=4.0, tol=1e-14,
                        Csize_y=nothing, Csize_eta=nothing,
                        Cmax_y=513, Cmax_eta=513,
                        assembly=:narrow_direct_absorbM,
                        upper_bw=nothing, block_cols=64,
                        return_info=false,
                        output=:dense)

Construct the final `N x N` Chebyshev coefficient matrix for the weakly
singular Volterra operator pulled back by the Liu--Xu DE map. The returned
matrix already includes the external multiplication by `(1+y)/2`.
"""
function weak_voltmat_decheb(
    K;
    mu,
    N::Int,
    omega::Real=4.0,
    tol::Float64=1e-14,
    Csize_y=nothing,
    Csize_eta=nothing,
    Cmax_y::Int=513,
    Cmax_eta::Int=513,
    assembly::Symbol=:narrow_direct_absorbM,
    upper_bw=nothing,
    block_cols::Int=64,
    fftw_flags=nothing,
    h0_init::Symbol=:s_recurrence,
    prec_bits::Int=1536,
    return_info::Bool=false,
    output::Symbol=:dense,
)
    output === :dense || output === :banded || throw(ArgumentError("output must be :dense or :banded"))
    assembly_requested = assembly
    assembly_used = if assembly === :auto
        :narrow_direct_absorbM
    elseif assembly === :narrow_fft_absorbM
        :narrow_rfft_absorbM
    else
        assembly
    end
    assembly_used === :narrow_direct_absorbM ||
        assembly_used === :narrow_rfft_absorbM ||
        assembly_used === :dense_full ||
        throw(ArgumentError("assembly must be :narrow_direct_absorbM, :narrow_rfft_absorbM, :narrow_fft_absorbM, :auto, or :dense_full"))
    assembly === :dense_full && output === :banded &&
        throw(ArgumentError("output=:banded is only available with banded assemblies"))
    mu64 = Float64(mu)
    omega64 = Float64(omega)
    0.0 < mu64 < 1.0 || throw(ArgumentError("mu must satisfy 0 < mu < 1"))
    N >= 2 || throw(ArgumentError("N must be at least 2"))

    t_total0 = time()
    local C, Cinfo
    time_build_C = missing
    if return_info
        time_build_C = @elapsed begin
            C, Cinfo = build_C_adaptive_de(
                K;
                mu=mu64,
                omega=omega64,
                tol,
                Csize_y,
                Csize_eta,
                Cmax_y,
                Cmax_eta,
            )
        end
    else
        C, Cinfo = build_C_adaptive_de(
            K;
            mu=mu64,
            omega=omega64,
            tol,
            Csize_y,
            Csize_eta,
            Cmax_y,
            Cmax_eta,
        )
    end

    local rawA, Aout, assembly_info
    time_assembly = missing
    function assemble_selected!()
        if assembly_used === :narrow_direct_absorbM
            Aband, band_info = assemble_A_narrow_direct_absorbM_de(
                C,
                mu64,
                N;
                upper_bw,
                h0_init,
                prec_bits,
            )
            rawA = Aband
            assembly_info = band_info
            Aout = output === :dense ?
                band_to_dense(Aband, N, band_info.lower_bw, band_info.upper_bw) :
                Aband
        elseif assembly_used === :narrow_rfft_absorbM
            P = build_postmult_coeffs_de(C)
            ubw = upper_bw === nothing ? size(C, 1) : Int(upper_bw)
            bandwidth_rule = upper_bw === nothing ? :C_rows : :explicit_upper_bw
            if return_info
                Aband, fft_info0 = assemble_A_narrow_rfft_absorbM_de(
                    P,
                    mu64,
                    N;
                    upper_bw=ubw,
                    block_cols,
                    fftw_flags,
                    return_info=true,
                    h0_init,
                    prec_bits,
                )
                assembly_info = merge(fft_info0, (; bandwidth_rule))
            else
                Aband = assemble_A_narrow_rfft_absorbM_de(
                    P,
                    mu64,
                    N;
                    upper_bw=ubw,
                    block_cols,
                    fftw_flags,
                    return_info=false,
                    h0_init,
                    prec_bits,
                )
                assembly_info = (;
                    P_size=size(P),
                    lower_bw=size(P, 1) - 1,
                    upper_bw=ubw,
                    S_A=ubw + size(P, 1) - 1,
                    Sguard_A=ubw + size(P, 1) - 1 + size(P, 2) - 1 + 2,
                    band_storage_size=size(Aband),
                    bandwidth_rule,
                    absorbs_final_multiplier=true,
                    fft_used=true,
                    block_cols,
                    fftw_flags,
                )
            end
            rawA = Aband
            Aout = output === :dense ?
                band_to_dense(Aband, N, assembly_info.lower_bw, assembly_info.upper_bw) :
                Aband
        else
            A, dense_info = _assemble_dense_full_from_C(C, mu64, N)
            rawA = A
            Aout = A
            assembly_info = dense_info
        end
        return nothing
    end
    if return_info
        time_assembly = @elapsed assemble_selected!()
    else
        assemble_selected!()
    end
    time_total = return_info ? time() - t_total0 : missing

    return_info || return Aout

    info = _info_for_decheb(
        mu=mu64,
        omega=omega64,
        N=N,
        C=C,
        Cinfo=Cinfo,
        Aout=Aout,
        output=output,
        assembly_requested=assembly_requested,
        assembly_used=assembly_used,
        time_build_C=time_build_C,
        time_assembly=time_assembly,
        time_total=time_total,
        assembly_info=assembly_info,
    )
    return return_info ? (Aout, info) : Aout
end

end # module Weak_VoltMat_DE
