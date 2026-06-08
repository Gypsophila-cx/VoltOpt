#!/usr/bin/env julia

# Lightweight action validation for the weakly singular Volterra demo.
# This script deliberately avoids BigFloat, matrix-entry double integrals,
# and figure generation.

include(joinpath(@__DIR__, "weak_singular_volterra_demo.jl"))

using LinearAlgebra
using Printf

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
    n > 0 || error("number of quadrature nodes must be positive")
    diag = zeros(Float64, n)
    offdiag = [k / sqrt(4.0 * k^2 - 1.0) for k in 1:(n - 1)]
    eig = eigen(SymTridiagonal(diag, offdiag))
    q = 0.5 .* (eig.values .+ 1.0)
    w = vec(eig.vectors[1, :]).^2
    return q, w
end

function test_u(t::Real)
    t64 = Float64(t)
    T3 = 4.0 * t64^3 - 3.0 * t64
    T5 = 16.0 * t64^5 - 20.0 * t64^3 + 5.0 * t64
    return 1.0 + t64 + 0.5 * T3 - 0.2 * T5
end

function direct_duffy_value(y::Float64, mu::Float64, qnodes, qweights)
    scale = 0.5 * (y + 1.0)
    integral = 0.0
    for (q, w) in zip(qnodes, qweights)
        s = q^4
        eta = 1.0 - 2.0 * s
        t = scale * eta + 0.5 * (y - 1.0)
        jac = 8.0 * q^3
        integral += w * (2.0 * s)^(mu - 1.0) * test_u(t) * jac
    end
    return scale^mu * integral
end

function main()
    K = (x, t) -> 1.0
    mu = 0.5
    N = 24
    tol = 1e-14

    A, info = build_weak_singular_volterra_matrix(K; mu=mu, N=N, tol=tol, return_info=true)

    size_ok = size(A) == (N, N)
    finite_ok = all(isfinite, A)

    ucoeff = zeros(Float64, N)
    ucoeff[1] = 1.0
    ucoeff[2] = 1.0
    ucoeff[4] = 0.5
    ucoeff[6] = -0.2

    vcoeff = A * ucoeff
    qnodes, qweights = gauss_legendre_01(400)

    max_abs_error = 0.0
    max_rel_error = 0.0
    for y in range(-0.95, 0.95; length=20)
        y64 = Float64(y)
        v_matrix = cheb_eval(vcoeff, y64)
        v_ref = direct_duffy_value(y64, mu, qnodes, qweights)
        err = abs(v_matrix - v_ref)
        rel = err / max(abs(v_ref), eps(Float64))
        max_abs_error = max(max_abs_error, err)
        max_rel_error = max(max_rel_error, rel)
    end

    action_ok = max_abs_error <= 2e-3 && max_rel_error <= 1.5e-2
    passed = size_ok && finite_ok && action_ok

    @printf("max abs error: %.6e\n", max_abs_error)
    @printf("max rel error: %.6e\n", max_rel_error)
    println("validation passed: ", passed)

    return info
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
