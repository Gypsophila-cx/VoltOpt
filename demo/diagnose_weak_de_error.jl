#!/usr/bin/env julia

# Diagnose the residual action-validation plateau for the weak singular
# Volterra demo with the Liu-Xu paper double-exponential map.  Float64 only.

include(joinpath(@__DIR__, "weak_singular_volterra_demo.jl"))

using LinearAlgebra
using Printf

const OMEGA = 4.0
const MU = 0.5
const TOL = 1e-14

phi(x::Real) = paper_de_phi(x; omega=OMEGA)
dphi(x::Real) = paper_de_dphi(x; omega=OMEGA)

function cheb_eval(coeff::AbstractVector{<:Real}, x::Real)
    n = length(coeff)
    n == 0 && return 0.0
    x64 = Float64(x)
    bkp1 = 0.0
    bkp2 = 0.0
    for k in n:-1:2
        bk = 2.0 * x64 * bkp1 - bkp2 + Float64(coeff[k])
        bkp2 = bkp1
        bkp1 = bk
    end
    return Float64(coeff[1]) + x64 * bkp1 - bkp2
end

function gauss_legendre_01(n::Int)
    diag = zeros(Float64, n)
    offdiag = [k / sqrt(4.0 * k^2 - 1.0) for k in 1:(n - 1)]
    eig = eigen(SymTridiagonal(diag, offdiag))
    q = 0.5 .* (eig.values .+ 1.0)
    w = vec(eig.vectors[1, :]).^2
    return q, w
end

function chebT(j::Int, x::Float64)
    j == 0 && return 1.0
    j == 1 && return x
    tm2 = 1.0
    tm1 = x
    t = x
    for _ in 2:j
        t = 2.0 * x * tm1 - tm2
        tm2 = tm1
        tm1 = t
    end
    return t
end

function jacobiP(n::Int, alpha::Float64, beta::Float64, x::Float64)
    n == 0 && return 1.0
    p0 = 1.0
    p1 = 0.5 * ((alpha - beta) + (alpha + beta + 2.0) * x)
    n == 1 && return p1
    pm1 = p0
    p = p1
    for m in 1:(n - 1)
        a = 2.0 * (m + 1.0) * (m + alpha + beta + 1.0) * (2.0 * m + alpha + beta)
        b = (2.0 * m + alpha + beta + 1.0) *
            ((2.0 * m + alpha + beta) * (2.0 * m + alpha + beta + 2.0) * x +
             alpha^2 - beta^2)
        c = 2.0 * (m + alpha) * (m + beta) * (2.0 * m + alpha + beta + 2.0)
        pp1 = (b * p - c * pm1) / a
        pm1 = p
        p = pp1
    end
    return p
end

function weighted_integral(qnodes, qweights, f)
    s = 0.0
    for (q, w) in zip(qnodes, qweights)
        qq = Float64(q)
        sigma = qq^4
        eta = 1.0 - 2.0 * sigma
        jac = 8.0 * qq^3
        s += w * (2.0 * sigma)^(MU - 1.0) * f(eta) * jac
    end
    return s
end

weighted_integral(f::Function, qnodes, qweights) = weighted_integral(qnodes, qweights, f)

function h00_ref_values(kmax::Int, qnodes, qweights)
    refs = zeros(Float64, kmax + 1)
    for k in 0:kmax
        refs[k + 1] = weighted_integral(qnodes, qweights) do eta
            jacobiP(k, -1.0, 0.0, eta)
        end
    end
    return refs
end

function h_ref(j::Int, r::Int, k::Int, qnodes, qweights)
    weighted_integral(qnodes, qweights) do eta
        chebT(j, eta) * ((1.0 + eta) / 2.0)^r *
        jacobiP(k - r, -1.0, 2.0 * r, eta)
    end
end

function build_Hjr_from_demo(H00, jmax::Int, kext::Int)
    H_r0 = build_r0_layer_from_H00(H00, jmax, kext)
    return H_r0
end

function stable_Ktilde(y::Float64, eta::Float64)
    stable_paper_de_transformed_kernel_value((x, t) -> 1.0, MU, y, eta; omega=OMEGA)
end

function ktilde_tail(C)
    last_row = maximum(abs, @view C[end, :])
    last_col = maximum(abs, @view C[:, end])
    return max(last_row, last_col), last_row, last_col
end

function test_u(t::Real)
    t64 = Float64(t)
    T3 = 4.0 * t64^3 - 3.0 * t64
    T5 = 16.0 * t64^5 - 20.0 * t64^3 + 5.0 * t64
    return 1.0 + t64 + 0.5 * T3 - 0.2 * T5
end

function direct_duffy_value(y::Float64, qnodes, qweights)
    scale = 0.5 * (1.0 + y)
    weighted_integral(qnodes, qweights) do eta
        stable_Ktilde(y, eta) * test_u(duffy_tau(y, eta))
    end * scale
end

function direct_duffy_value_with_Ktilde(y::Float64, qnodes, qweights, Ktilde_func)
    scale = 0.5 * (1.0 + y)
    weighted_integral(qnodes, qweights) do eta
        Ktilde_func(y, eta) * test_u(duffy_tau(y, eta))
    end * scale
end

function action_error_for_q(nq::Int, A)
    qnodes, qweights = gauss_legendre_01(nq)
    N = size(A, 1)
    ucoeff = zeros(Float64, N)
    ucoeff[1] = 1.0
    ucoeff[2] = 1.0
    ucoeff[4] = 0.5
    ucoeff[6] = -0.2
    vcoeff = A * ucoeff
    err = 0.0
    for y in range(-0.75, 0.75; length=40)
        y64 = Float64(y)
        err = max(err, abs(cheb_eval(vcoeff, y64) - direct_duffy_value(y64, qnodes, qweights)))
    end
    return err
end

function action_errors_for_q(nq::Int, A)
    qnodes, qweights = gauss_legendre_01(nq)
    N = size(A, 1)
    ucoeff = zeros(Float64, N)
    ucoeff[1] = 1.0
    ucoeff[2] = 1.0
    ucoeff[4] = 0.5
    ucoeff[6] = -0.2
    vcoeff = A * ucoeff

    max_abs = 0.0
    max_rel = 0.0
    for y in range(-0.75, 0.75; length=40)
        y64 = Float64(y)
        v_mat = cheb_eval(vcoeff, y64)
        v_ref = direct_duffy_value(y64, qnodes, qweights)
        err = abs(v_mat - v_ref)
        max_abs = max(max_abs, err)
        max_rel = max(max_rel, err / max(abs(v_ref), 1.0))
    end
    return max_abs, max_rel
end

function ktilde_reconstruction_errors(C, Ktilde)
    max_abs = 0.0
    max_rel = 0.0
    for y in range(-0.75, 0.75; length=41), eta in range(-0.95, 0.95; length=41)
        y64 = Float64(y)
        eta64 = Float64(eta)
        exact = Ktilde(y64, eta64)
        recon = eval_cheb2(C, y64, eta64)
        err = abs(recon - exact)
        max_abs = max(max_abs, err)
        max_rel = max(max_rel, err / max(abs(exact), 1.0))
    end
    return max_abs, max_rel
end

function ktilde_reconstruction_diagnostic(C)
    max_abs = 0.0
    max_rel = 0.0
    for y in range(-0.75, 0.75; length=41), eta in range(-0.95, 0.95; length=41)
        y64 = Float64(y)
        eta64 = Float64(eta)
        exact = stable_Ktilde(y64, eta64)
        recon = eval_cheb2(C, y64, eta64)
        err = abs(exact - recon)
        max_abs = max(max_abs, err)
        max_rel = max(max_rel, err / max(abs(exact), 1.0))
    end
    tail, last_row, last_col = ktilde_tail(C)
    return (; max_abs, max_rel, last_row, last_col, tail)
end

function error_decomposition(A, C, qnodes, qweights)
    N = size(A, 1)
    ucoeff = zeros(Float64, N)
    ucoeff[1] = 1.0
    ucoeff[2] = 1.0
    ucoeff[4] = 0.5
    ucoeff[6] = -0.2
    vcoeff = A * ucoeff
    Ktilde_C = (y, eta) -> eval_cheb2(C, y, eta)

    E_total = 0.0
    E_C = 0.0
    E_alg = 0.0
    R_total = 0.0
    R_C = 0.0
    R_alg = 0.0
    for y in range(-0.75, 0.75; length=40)
        y64 = Float64(y)
        v_A = cheb_eval(vcoeff, y64)
        v_exact = direct_duffy_value_with_Ktilde(y64, qnodes, qweights, stable_Ktilde)
        v_Cref = direct_duffy_value_with_Ktilde(y64, qnodes, qweights, Ktilde_C)

        e_total = abs(v_A - v_exact)
        e_C = abs(v_Cref - v_exact)
        e_alg = abs(v_A - v_Cref)
        E_total = max(E_total, e_total)
        E_C = max(E_C, e_C)
        E_alg = max(E_alg, e_alg)
        R_total = max(R_total, e_total / max(abs(v_exact), 1.0))
        R_C = max(R_C, e_C / max(abs(v_exact), 1.0))
        R_alg = max(R_alg, e_alg / max(abs(v_Cref), 1.0))
    end
    return (; E_total, E_C, E_alg, R_total, R_C, R_alg)
end

function build_result_for_N(K, N::Int)
    construct_volterra_matrix(
        K;
        mu=MU,
        N=N,
        tol=TOL,
        phi=phi,
        dphi=dphi,
        stable_de_ktilde=true,
        de_omega=OMEGA,
        Csize_y=100,
        Csize_eta=100,
    )
end

function main()
    K = (x, t) -> 1.0
    Nlist = [48, 96, 160]

    println("diagnostic: fixed C=100x100 paper DE error decomposition")
    println("omega: ", OMEGA)
    println("mu: ", MU)
    println("stable_de_ktilde: true")
    println("Csize_y: 100")
    println("Csize_eta: 100")
    println("reference_qnodes: 1600")
    println()
    q1600, w1600 = gauss_legendre_01(1600)

    result96 = nothing
    println("N, actual_C_size, E_total, E_C, E_alg, R_total, R_C, R_alg")
    for N in Nlist
        result = build_result_for_N(K, N)
        if N == 96
            result96 = result
        end
        decomp = error_decomposition(result.A, result.C, q1600, w1600)
        @printf(
            "%d, %s, %.6e, %.6e, %.6e, %.6e, %.6e, %.6e\n",
            N,
            string(size(result.C)),
            decomp.E_total,
            decomp.E_C,
            decomp.E_alg,
            decomp.R_total,
            decomp.R_C,
            decomp.R_alg,
        )
    end

    println()
    println("Ktilde reconstruction diagnostic for N=96 C")
    kd = ktilde_reconstruction_diagnostic(result96.C)
    println("actual size(C): ", size(result96.C))
    @printf("max_abs_Ktilde_error: %.6e\n", kd.max_abs)
    @printf("max_scaled_rel_Ktilde_error: %.6e\n", kd.max_rel)
    @printf("last_row_max: %.6e\n", kd.last_row)
    @printf("last_col_max: %.6e\n", kd.last_col)
    @printf("tail_last_row_col_max: %.6e\n", kd.tail)

    println()
    println("Quadrature sensitivity for N=96")
    println("qnodes, E_total, E_C, E_alg, R_total, R_C, R_alg")
    for nq in [800, 1600, 3200]
        q, w = gauss_legendre_01(nq)
        decomp = error_decomposition(result96.A, result96.C, q, w)
        @printf(
            "%d, %.6e, %.6e, %.6e, %.6e, %.6e, %.6e\n",
            nq,
            decomp.E_total,
            decomp.E_C,
            decomp.E_alg,
            decomp.R_total,
            decomp.R_C,
            decomp.R_alg,
        )
    end

    println()
    decomp96 = error_decomposition(result96.A, result96.C, q1600, w1600)
    if decomp96.E_C < 0.25 * decomp96.E_total && decomp96.E_alg > 0.75 * decomp96.E_total
        println("Conclusion: E_alg dominates for C=100x100.")
    elseif decomp96.E_alg < 0.25 * decomp96.E_total && decomp96.E_C > 0.75 * decomp96.E_total
        println("Conclusion: E_C dominates for C=100x100.")
    else
        println("Conclusion: E_total is a mixed contribution for C=100x100.")
    end
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
