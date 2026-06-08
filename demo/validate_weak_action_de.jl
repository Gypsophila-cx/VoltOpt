#!/usr/bin/env julia

# Lightweight action validation for the weakly singular Volterra demo with a
# double-exponential endpoint-clustering map.  This script uses Float64 only,
# validates operator action rather than matrix entries, and generates no plots.

include(joinpath(@__DIR__, "weak_voltmat_decheb.jl"))
using .Weak_VoltMat_DE

using LinearAlgebra
using Printf

const omega = 4.0
const mu = 0.5
const TOL = 1e-14
const NLIST = [24, 48, 96, 160]
const YGRID = collect(range(-0.75, 0.75; length=40))

phi(x::Real) = Weak_VoltMat_DE.paper_de_phi(x; omega)
dphi(x::Real) = Weak_VoltMat_DE.paper_de_dphi(x; omega)

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

function reference_Ktilde_phi(y::Float64, eta::Float64, mu::Float64)
    return Weak_VoltMat_DE.stable_paper_de_transformed_kernel_value(
        (x, t) -> 1.0,
        mu,
        y,
        eta;
        omega,
    )
end

function direct_duffy_value(y::Float64, mu::Float64, qnodes, qweights)
    scale = 0.5 * (y + 1.0)
    integral = 0.0
    for (q, w) in zip(qnodes, qweights)
        s = q^4
        eta = 1.0 - 2.0 * s
        tau = Weak_VoltMat_DE.duffy_tau(y, eta)
        jac = 8.0 * q^3
        integral += w * (2.0 * s)^(mu - 1.0) *
                    reference_Ktilde_phi(y, eta, mu) * test_u(tau) * jac
    end
    return scale * integral
end

function max_action_errors(vcoeff, reference)
    max_abs_error = 0.0
    max_rel_error = 0.0
    for y in YGRID
        y64 = Float64(y)
        v_mat = cheb_eval(vcoeff, y64)
        v_ref = reference(y64)
        err = abs(v_mat - v_ref)
        rel = err / max(abs(v_ref), eps(Float64))
        max_abs_error = max(max_abs_error, err)
        max_rel_error = max(max_rel_error, rel)
    end
    return max_abs_error, max_rel_error
end

decreases(errors) = all(errors[i + 1] < errors[i] for i in 1:(length(errors) - 1))

function ktilde_diagnostics(qnodes)
    min_finite = Inf
    max_finite = -Inf
    nan_count = 0
    inf_count = 0

    for y in YGRID, q in qnodes
        s = q^4
        eta = 1.0 - 2.0 * s
        value = reference_Ktilde_phi(Float64(y), eta, mu)
        if isnan(value)
            nan_count += 1
        elseif isinf(value)
            inf_count += 1
        elseif isfinite(value)
            min_finite = min(min_finite, value)
            max_finite = max(max_finite, value)
        end
    end

    return (;
        min_finite=min_finite,
        max_finite=max_finite,
        nan_count,
        inf_count,
    )
end

function main()
    K = (x, t) -> 1.0
    qnodes, qweights = gauss_legendre_01(400)

    case1_abs = Float64[]
    case2_abs = Float64[]

    println("phi name: paper DE transform")
    println("omega: ", omega)
    println("mu: ", mu)
    println("phi(-1), phi(0), phi(1): ", phi(-1.0), ", ", phi(0.0), ", ", phi(1.0))
    println("dphi(-1), dphi(0), dphi(1): ", dphi(-1.0), ", ", dphi(0.0), ", ", dphi(1.0))
    diag = ktilde_diagnostics(qnodes)
    println("min finite Ktilde: ", diag.min_finite)
    println("max finite Ktilde: ", diag.max_finite)
    println("number of NaN Ktilde: ", diag.nan_count)
    println("number of Inf Ktilde: ", diag.inf_count)
    println("case, N, max_abs_error, max_rel_error")
    println("case, N, size(A), all_finite_A")
    for N in NLIST
        A, info = Weak_VoltMat_DE.weak_voltmat_decheb(
            K;
            mu=mu,
            N=N,
            omega=omega,
            tol=TOL,
            return_info=true,
            Csize_y=100,
            Csize_eta=100,
        )
        println("matrix, $N, ", size(A), ", ", all(isfinite, A))

        ucoeff = zeros(Float64, N)
        ucoeff[1] = 1.0
        vcoeff = A * ucoeff
        exact = y -> exp(mu * Weak_VoltMat_DE.paper_de_log1p_phi(y; omega)) / mu
        abs1, rel1 = max_action_errors(vcoeff, exact)
        push!(case1_abs, abs1)
        @printf("case1, %d, %.6e, %.6e\n", N, abs1, rel1)

        fill!(ucoeff, 0.0)
        ucoeff[1] = 1.0
        ucoeff[2] = 1.0
        ucoeff[4] = 0.5
        ucoeff[6] = -0.2
        vcoeff = A * ucoeff
        ref = y -> direct_duffy_value(y, mu, qnodes, qweights)
        abs2, rel2 = max_action_errors(vcoeff, ref)
        push!(case2_abs, abs2)
        @printf("case2, %d, %.6e, %.6e\n", N, abs2, rel2)

        _ = info
    end

    println("case1 errors decrease as N increases: ", decreases(case1_abs))
    println("case2 errors decrease as N increases: ", decreases(case2_abs))
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
