#!/usr/bin/env julia

# Self-contained demo for constructing the Chebyshev coefficient matrix of a
# weakly singular Volterra operator.  This file intentionally stays inside
# demo/ and does not include or modify the core src/ VoltOpt implementation.

using LinearAlgebra
using Printf
using Random

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

function usage()
    println("""
    Usage:
      julia --project=. demo/weak_singular_volterra_demo.jl [options]

    Options:
      --mu=0.6             weak singularity parameter, 0 < mu < 1
      --N=64               final matrix dimension, matrix is NxN
      --tol=1e-14          adaptive 2D Chebyshev coefficient tolerance
      --kernel=exp         exp, rational, or constant
      --n0=17              initial first-kind Chebyshev grid size
      --nmax=129           maximum first-kind Chebyshev grid size
      --max-m=64           maximum retained y-degree in transformed kernel
      --max-n=64           maximum retained eta-degree in transformed kernel
      --show-matrices      print leading 5x5 blocks of G and A
      --skip-sanity        skip extra N=32,64 sanity builds
      --help               show this message
    """)
end

function parse_cli(args)
    mu = 0.6
    N = 64
    tol = 1e-14
    kernel = "exp"
    n0 = 17
    nmax = 129
    max_m = 64
    max_n = 64
    show_matrices = false
    run_sanity = true

    for arg in args
        if arg == "--help" || arg == "-h"
            usage()
            exit(0)
        elseif startswith(arg, "--mu=")
            mu = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--N=")
            N = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--tol=")
            tol = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--kernel=")
            kernel = lowercase(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--n0=")
            n0 = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--nmax=")
            nmax = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--max-m=")
            max_m = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--max-n=")
            max_n = parse(Int, split(arg, "=", limit=2)[2])
        elseif arg == "--rational"
            kernel = "rational"
        elseif arg == "--show-matrices"
            show_matrices = true
        elseif arg == "--skip-sanity"
            run_sanity = false
        else
            error("Unknown option: $arg")
        end
    end

    0.0 < mu < 1.0 || error("--mu must satisfy 0 < mu < 1")
    N >= 2 || error("--N must be at least 2")
    n0 >= 3 || error("--n0 must be at least 3")
    nmax >= n0 || error("--nmax must be at least --n0")
    max_m >= 0 || error("--max-m must be nonnegative")
    max_n >= 0 || error("--max-n must be nonnegative")
    kernel in ("exp", "rational", "constant") || error("--kernel must be exp, rational, or constant")

    return (; mu, N, tol, kernel, n0, nmax, max_m, max_n, show_matrices, run_sanity)
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

function kernel_function(name::AbstractString)
    if name == "exp"
        return (x, t) -> exp(-x * t)
    elseif name == "rational"
        return (x, t) -> inv((x + 1.0)^2 + (t - 1.0)^2)
    elseif name == "constant"
        return (x, t) -> 1.0
    else
        error("unknown kernel: $name")
    end
end

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

function build_checks(; mu, H00, Hcomp, G, A, C, Ktilde, Kmax, N)
    closed = closed_initial_moments(Float64(mu), Kmax)
    return (;
        C_size=size(C),
        C_max=maxabs_array(C),
        C_sample_error=random_cheb2_error(Ktilde, C),
        H00_all_finite=allfinite_array(H00),
        H00_closed_form_max_error=maxabs_array(H00 .- closed),
        Hcomp_all_finite=allfinite_array(Hcomp),
        G_all_finite=allfinite_array(G),
        A_all_finite=allfinite_array(A),
        matrix_size=size(A),
        output_is_N_by_N=size(A) == (N, N),
        final_matrix_includes_external_factor=true,
        external_factor="(1+y)/2",
    )
end

function construct_volterra_matrix(
    K;
    mu::Real,
    N::Int,
    tol::Float64,
    phi=identity_phi,
    dphi=identity_dphi,
    n0::Int=17,
    nmax::Int=129,
    max_m=64,
    max_n=64,
    de_h::Float64=0.025,
    de_tol::Float64=1e-15,
    Kext_guard::Int=4,
    stable_de_ktilde::Bool=false,
    de_omega::Float64=4.0,
    Csize_y=nothing,
    Csize_eta=nothing,
)
    0.0 < Float64(mu) < 1.0 || error("mu must satisfy 0 < mu < 1")
    N >= 2 || error("N must be at least 2")
    n0 >= 3 || error("n0 must be at least 3")
    nmax >= n0 || error("nmax must be at least n0")
    Kext_guard >= 0 || error("Kext_guard must be nonnegative")

    local C, infoC, Ktilde
    time_cheb = @elapsed begin
        C, infoC, Ktilde = build_Kphi_cheb_coeffs(
            K,
            phi,
            dphi;
            mu,
            tol,
            n0,
            nmax,
            max_m,
            max_n,
            stable_de_ktilde,
            de_omega,
            Csize_y,
            Csize_eta,
        )
    end

    m = size(C, 1) - 1
    n = size(C, 2) - 1
    Kext = N + n + m + Kext_guard
    H00_Kmax = Kext + n

    local H00, de_diag
    time_de = @elapsed begin
        H00, de_diag = stable_de_H000(
            mu,
            H00_Kmax;
            tol=de_tol,
            h=de_h,
            return_info=true,
        )
    end

    local H_r0
    time_r0 = @elapsed begin
        H_r0 = build_r0_layer_from_H00(H00, n, Kext)
    end

    local Hcomp
    time_Hcomp = @elapsed begin
        Hcomp = accumulate_Hcomp_streaming(C, H_r0, Kext; mu)
    end

    local G, A
    time_assembly = @elapsed begin
        G = assemble_G_from_Hcomp(Hcomp, N)
        A = multiply_by_one_plus_y_over_two(G, N)
    end

    checks = build_checks(; mu, H00, Hcomp, G, A, C, Ktilde, Kmax=H00_Kmax, N)

    info = (;
        C_size=size(C),
        C_full_grid_size=infoC.full_grid_size,
        C_max=infoC.max_coeff,
        C_tail=infoC.tail,
        C_sample_error=infoC.sample_error,
        C_converged=infoC.converged,
        m,
        n,
        Kext,
        H00_Kmax,
        de_node_count=de_diag.node_count,
        de_selected_h=de_diag.h,
        de_left_endpoint_distance=de_diag.left_endpoint_distance,
        de_right_endpoint_distance=de_diag.right_endpoint_distance,
        de_nonfinite_seen=de_diag.nonfinite_seen,
        de_rounded_endpoint_nodes=de_diag.rounded_endpoint_nodes,
        right_virtual_de_nodes=de_diag.right_virtual_nodes,
        right_final_log1m=de_diag.right_final_log1m,
        right_final_tail_estimate=de_diag.right_final_tail_estimate,
        k0_tail_correction_used=de_diag.k0_tail_correction_used,
        stable_de_ktilde,
        de_omega,
        Csize_y,
        Csize_eta,
        de_diagnostics=de_diag,
        uses_de_initial_moments=USES_DE_INITIAL_MOMENTS,
        uses_streaming_j_compression=USES_STREAMING_J_COMPRESSION,
        uses_j_contracted_assembly=USES_STREAMING_J_COMPRESSION,
        stores_full_H_jrk_tensor=STORES_FULL_H_JRK_TENSOR,
        project_files_modified=PROJECT_FILES_MODIFIED,
        matrix_size=size(A),
        final_matrix_includes_external_factor=true,
        external_factor="(1+y)/2",
        checks,
        time_cheb,
        time_de,
        time_r0,
        time_Hcomp,
        time_assembly,
    )

    return (; Ktilde, C, H00, H_r0, Hcomp, G, A, info)
end

function build_weak_singular_volterra_matrix(
    K;
    mu::Real=0.5,
    N::Int=64,
    tol::Float64=1e-14,
    return_info::Bool=false,
    phi=identity_phi,
    dphi=identity_dphi,
    n0::Int=17,
    nmax::Int=129,
    max_m=64,
    max_n=64,
    de_h::Float64=0.025,
    de_tol::Float64=1e-15,
    Kext_guard::Int=4,
    stable_de_ktilde::Bool=false,
    de_omega::Float64=4.0,
    Csize_y=nothing,
    Csize_eta=nothing,
)
    result = construct_volterra_matrix(
        K;
        mu,
        N,
        tol,
        phi,
        dphi,
        n0,
        nmax,
        max_m,
        max_n,
        de_h,
        de_tol,
        Kext_guard,
        stable_de_ktilde,
        de_omega,
        Csize_y,
        Csize_eta,
    )
    return return_info ? (result.A, result.info) : result.A
end

function print_diag(label, value)
    println(rpad(label * ":", 60), value)
end

function print_matrix_block(name, A; rows=5, cols=5)
    r = min(rows, size(A, 1))
    c = min(cols, size(A, 2))
    println(name, " leading ", r, "x", c, " block:")
    for i in 1:r
        for j in 1:c
            @printf(" % .4e", A[i, j])
        end
        println()
    end
end

function print_de_convergence(rows::Vector{DEConvergenceRow})
    println("DE convergence diagnostics:")
    println("  h        nodes   right tail    change vs prev  closed-form err  virtual  nonfinite")
    for row in rows
        change = isnan(row.change_from_previous_h) ? "n/a" : @sprintf("%.3e", row.change_from_previous_h)
        @printf(
            "  %-7.3g %-7d %-13.3e %-15s %-16.3e %-8d %s\n",
            row.h,
            row.node_count,
            row.right_final_tail_estimate,
            change,
            row.closed_form_error,
            row.right_virtual_nodes,
            string(row.nonfinite_seen),
        )
    end
end

function print_sanity_checks(K; mu, tol, n0, nmax, max_m, max_n)
    println()
    println("Small-N sanity checks:")
    for Nsane in (32, 64)
        A, info = build_weak_singular_volterra_matrix(
            K;
            mu,
            N=Nsane,
            tol,
            return_info=true,
            n0,
            nmax,
            max_m,
            max_n,
        )
        @printf(
            "  N=%-3d size=%s finite(A)=%s finite(H00/Hcomp/G)=%s/%s/%s external=%s C=%dx%d\n",
            Nsane,
            string(size(A)),
            string(info.checks.A_all_finite),
            string(info.checks.H00_all_finite),
            string(info.checks.Hcomp_all_finite),
            string(info.checks.G_all_finite),
            string(info.final_matrix_includes_external_factor),
            info.C_size[1],
            info.C_size[2],
        )
    end
end

function main()
    opts = parse_cli(ARGS)
    K = kernel_function(opts.kernel)

    result = construct_volterra_matrix(
        K;
        mu=opts.mu,
        N=opts.N,
        tol=opts.tol,
        n0=opts.n0,
        nmax=opts.nmax,
        max_m=opts.max_m,
        max_n=opts.max_n,
    )
    info = result.info
    checks = info.checks

    println("Weak singular Volterra coefficient matrix demo")
    print_diag("kernel", opts.kernel)
    print_diag("mu", opts.mu)
    print_diag("N (final matrix dimension)", opts.N)
    print_diag("C size", "$(info.C_size[1]) x $(info.C_size[2])")
    print_diag("C full sampled grid", "$(info.C_full_grid_size[1]) x $(info.C_full_grid_size[2])")
    print_diag("C max coefficient", @sprintf("%.6e", info.C_max))
    print_diag("C tail estimate", @sprintf("%.6e", info.C_tail))
    print_diag("C random sample error", @sprintf("%.6e", info.C_sample_error))
    print_diag("C expansion converged", info.C_converged)
    print_diag("m, n", "$(info.m), $(info.n)")
    print_diag("max retained m, n", "$(opts.max_m), $(opts.max_n)")
    print_diag("Kext", info.Kext)
    print_diag("H00 Kmax", info.H00_Kmax)
    print_diag("H00 all finite", checks.H00_all_finite)
    print_diag("H00 closed-form max error", @sprintf("%.6e", checks.H00_closed_form_max_error))
    print_diag("Hcomp all finite", checks.Hcomp_all_finite)
    print_diag("G all finite", checks.G_all_finite)
    print_diag("A all finite", checks.A_all_finite)
    print_diag("output matrix size", "$(size(result.A, 1)) x $(size(result.A, 2))")
    print_diag("output is N x N", checks.output_is_N_by_N)
    print_diag("external factor included in A", "$(info.final_matrix_includes_external_factor), $(info.external_factor)")
    print_diag("DE node count", info.de_node_count)
    print_diag("DE selected h", @sprintf("%.6g", info.de_selected_h))
    print_diag("DE left endpoint distance", @sprintf("%.6e", info.de_left_endpoint_distance))
    print_diag("DE right endpoint distance", @sprintf("%.6e", info.de_right_endpoint_distance))
    print_diag("DE NaN/Inf encountered", info.de_nonfinite_seen)
    print_diag("DE rounded endpoint nodes handled", info.de_rounded_endpoint_nodes)
    print_diag("right virtual DE nodes", info.right_virtual_de_nodes)
    print_diag("right final log(1-eta)", @sprintf("%.6e", info.right_final_log1m))
    print_diag("right final tail estimate", @sprintf("%.6e", info.right_final_tail_estimate))
    print_diag("k=0 tail correction used", info.k0_tail_correction_used)
    print_de_convergence(info.de_diagnostics.convergence_rows)
    print_diag("time for transformed-kernel Chebyshev expansion", @sprintf("%.6f s", info.time_cheb))
    print_diag("time for stable DE H00", @sprintf("%.6f s", info.time_de))
    print_diag("time for r=0 (j,k) recurrence", @sprintf("%.6f s", info.time_r0))
    print_diag("time for fixed-j streaming compression", @sprintf("%.6f s", info.time_Hcomp))
    print_diag("time for G assembly and M[(1+y)/2] multiply", @sprintf("%.6f s", info.time_assembly))
    print_diag("max abs matrix entry", @sprintf("%.6e", maxabs_array(result.A)))
    print_diag("min abs nonzero matrix entry", @sprintf("%.6e", min_abs_nonzero(result.A)))
    print_diag("uses DE initial moments", info.uses_de_initial_moments)
    print_diag("uses streaming compression over j", info.uses_streaming_j_compression)
    print_diag("stores full H[j,r,k] tensor", info.stores_full_H_jrk_tensor)
    print_diag("project files modified", info.project_files_modified)

    if opts.show_matrices
        println()
        print_matrix_block("G before external factor", result.G)
        print_matrix_block("A = M[(1+y)/2] * G", result.A)
    end

    if opts.run_sanity
        print_sanity_checks(
            K;
            mu=opts.mu,
            tol=opts.tol,
            n0=opts.n0,
            nmax=opts.nmax,
            max_m=opts.max_m,
            max_n=opts.max_n,
        )
    end

    println()
    println("Plot command:")
    println("  julia --project=. demo/plot_logabs_spy_K1_mu0p5.jl")
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
