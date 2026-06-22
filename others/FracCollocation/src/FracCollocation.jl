module FracCollocation

using LinearAlgebra
using FastGaussQuadrature: gaussjacobi

export FracVIEProblem,
       FracCollocationSystem,
       FracCollocationSolution,
       shifted_jacobi_gauss,
       frac_collocation_nodes,
       build_system,
       solve,
       evaluate,
       collocation_residual

"""
    FracVIEProblem(; mu, lambda, alpha, beta, K, g, sigma=1.0)

Weakly singular second-kind Volterra integral equation

    u(x) = g(x) + sigma * int_0^x (x-s)^(-mu) K(x,s) u(s) ds.

The implementation convention is that `N` is the number of solution nodes,
collocation points, and quadrature points. Thus the approximation space is
`span(1, x^lambda, ..., x^((N-1)*lambda))`, not the literature's `N+1`
node convention.
"""
struct FracVIEProblem{T<:AbstractFloat,TK,TG}
    mu::T
    lambda::T
    alpha::T
    beta::T
    K::TK
    g::TG
    sigma::T
end

function FracVIEProblem(; mu, lambda, alpha, beta, K, g, sigma=1.0)
    mu_f, lambda_f, alpha_f, beta_f, sigma_f =
        promote(float(mu), float(lambda), float(alpha), float(beta), float(sigma))

    zero(mu_f) < mu_f < one(mu_f) ||
        throw(ArgumentError("mu must satisfy 0 < mu < 1"))
    lambda_f > zero(lambda_f) ||
        throw(ArgumentError("lambda must be positive"))
    _check_jacobi_params(alpha_f, beta_f)

    return FracVIEProblem(mu_f, lambda_f, alpha_f, beta_f, K, g, sigma_f)
end

struct FracCollocationSystem{T<:AbstractFloat,TP}
    problem::TP
    x::Vector{T}
    z::Vector{T}
    theta::Vector{T}
    wtheta::Vector{T}
    V::Matrix{T}
    A::Matrix{T}
    rhs::Vector{T}
    bary::Vector{T}
end

struct FracCollocationSolution{T<:AbstractFloat,TP,TS}
    problem::TP
    system::TS
    x::Vector{T}
    z::Vector{T}
    values::Vector{T}
    bary::Vector{T}
end

function _check_N(N::Integer)
    N > 0 || throw(ArgumentError("N must be positive"))
    return Int(N)
end

function _check_jacobi_params(alpha::Real, beta::Real)
    alpha > -one(alpha) ||
        throw(ArgumentError("Jacobi alpha must be greater than -1"))
    beta > -one(beta) ||
        throw(ArgumentError("Jacobi beta must be greater than -1"))
    return nothing
end

"""
    shifted_jacobi_gauss(N, alpha, beta)

Return the `N` shifted Jacobi-Gauss nodes and weights for

    int_0^1 (1-z)^alpha z^beta f(z) dz.

Nodes are sorted increasingly. The returned vectors have length exactly `N`.
"""
function shifted_jacobi_gauss(N::Integer, alpha::Real, beta::Real)
    n = _check_N(N)
    alpha_f, beta_f = promote(float(alpha), float(beta))
    _check_jacobi_params(alpha_f, beta_f)

    # FastGaussQuadrature uses (1-t)^alpha * (1+t)^beta on [-1, 1].
    # With t = 2z - 1 this becomes 2^(alpha+beta+1)
    # times (1-z)^alpha * z^beta on [0, 1].
    t, w_ref = gaussjacobi(n, alpha_f, beta_f)
    two = one(alpha_f) + one(alpha_f)
    z = (t .+ one(alpha_f)) ./ two
    w = w_ref ./ (two ^ (alpha_f + beta_f + one(alpha_f)))

    perm = sortperm(z)
    return collect(z[perm]), collect(w[perm])
end

"""
    frac_collocation_nodes(N; lambda, alpha, beta)
    frac_collocation_nodes(N, lambda, alpha, beta)

Return a named tuple `(x, z, w)` where `z` are shifted Jacobi-Gauss nodes and
`x = z^(1/lambda)`. All returned vectors have length exactly `N`.
"""
function frac_collocation_nodes(N::Integer; lambda, alpha, beta)
    lambda_f, alpha_f, beta_f = promote(float(lambda), float(alpha), float(beta))
    lambda_f > zero(lambda_f) ||
        throw(ArgumentError("lambda must be positive"))
    z, w = shifted_jacobi_gauss(N, alpha_f, beta_f)
    x = z .^ inv(lambda_f)
    return (x=x, z=z, w=w)
end

frac_collocation_nodes(N::Integer, lambda::Real, alpha::Real, beta::Real) =
    frac_collocation_nodes(N; lambda=lambda, alpha=alpha, beta=beta)

"""
    barycentric_weights(z)

Compute relative barycentric weights for distinct nodes using log-scaled
magnitudes. Only relative scaling matters for Lagrange interpolation.
"""
function barycentric_weights(z::AbstractVector{<:Real})
    n = length(z)
    n > 0 || throw(ArgumentError("at least one node is required"))

    T = typeof(float(z[begin]))
    logabsw = Vector{T}(undef, n)
    signw = Vector{T}(undef, n)

    @inbounds for j in 1:n
        zj = z[j]
        acc = zero(T)
        sgn = one(T)
        for m in 1:n
            m == j && continue
            diff = zj - z[m]
            diff == zero(diff) && throw(ArgumentError("barycentric nodes must be distinct"))
            acc -= log(abs(diff))
            sgn *= ifelse(diff > zero(diff), one(T), -one(T))
        end
        logabsw[j] = acc
        signw[j] = sgn
    end

    maxlog = maximum(logabsw)
    bary = Vector{T}(undef, n)
    @inbounds for j in 1:n
        bary[j] = signw[j] * exp(logabsw[j] - maxlog)
    end
    return bary
end

"""
    lagrange_values!(ell, znodes, bary, z_eval)

Overwrite `ell` with the Lagrange cardinal basis values at `z_eval`.
"""
function lagrange_values!(
    ell::AbstractVector,
    znodes::AbstractVector,
    bary::AbstractVector,
    z_eval::Real,
)
    n = length(znodes)
    length(ell) == n || throw(DimensionMismatch("ell must have length $(n)"))
    length(bary) == n || throw(DimensionMismatch("bary must have length $(n)"))

    T = promote_type(typeof(float(z_eval)), eltype(znodes), eltype(bary), eltype(ell))
    tol = 100 * eps(T)

    @inbounds for j in 1:n
        zj = znodes[j]
        if abs(z_eval - zj) <= tol * max(one(T), abs(z_eval), abs(zj))
            fill!(ell, zero(eltype(ell)))
            ell[j] = one(eltype(ell))
            return ell
        end
    end

    denom = zero(T)
    @inbounds for j in 1:n
        tmp = bary[j] / (z_eval - znodes[j])
        ell[j] = tmp
        denom += tmp
    end

    invdenom = inv(denom)
    @inbounds for j in 1:n
        ell[j] *= invdenom
    end
    return ell
end

"""
    ratio_one_minus_power(theta, p)

Stable evaluation of `(1 - theta^p) / (1 - theta)`.
"""
function ratio_one_minus_power(theta::Real, p::Real)
    theta_f, p_f = promote(float(theta), float(p))
    theta_f >= zero(theta_f) ||
        throw(DomainError(theta, "theta must be nonnegative"))

    theta_f == zero(theta_f) && return one(theta_f)
    theta_f == one(theta_f) && return p_f

    if abs(one(theta_f) - theta_f) <= sqrt(eps(theta_f)) * max(one(theta_f), abs(theta_f))
        return p_f
    end

    return (-expm1(p_f * log(theta_f))) / (one(theta_f) - theta_f)
end

"""
    build_system(prob, N)

Build the dense collocation matrix for `N` nodes. In this codebase `N` is the
solution vector length, collocation node count, and quadrature node count.
The integral matrix is assembled with an O(N^3) triple loop and one length-N
scratch vector for current Lagrange basis values.
"""
function build_system(prob::FracVIEProblem, N::Integer)
    n = _check_N(N)
    p = inv(prob.lambda)

    node_data = frac_collocation_nodes(n; lambda=prob.lambda, alpha=prob.alpha, beta=prob.beta)
    x = collect(node_data.x)
    z = collect(node_data.z)

    alpha_q = -prob.mu
    beta_q = p - one(p)
    theta, wtheta = shifted_jacobi_gauss(n, alpha_q, beta_q)
    theta_pow = theta .^ p

    T = promote_type(eltype(x), eltype(z), eltype(theta), eltype(wtheta), typeof(prob.mu))
    x = Vector{T}(x)
    z = Vector{T}(z)
    theta = Vector{T}(theta)
    wtheta = Vector{T}(wtheta)
    theta_pow = Vector{T}(theta_pow)

    bary = barycentric_weights(z)
    V = zeros(T, n, n)
    ell = Vector{T}(undef, n)

    K = prob.K
    mu = prob.mu
    lambda = prob.lambda

    @views @inbounds for i in 1:n
        xi = x[i]
        zi = z[i]
        xfactor = xi^(one(T) - mu) / lambda

        for k in 1:n
            th = theta[k]
            spoint = xi * theta_pow[k]
            z_eval = zi * th

            lagrange_values!(ell, z, bary, z_eval)

            ratio = ratio_one_minus_power(th, p)
            coeff = wtheta[k] * xfactor * ratio^(-mu) * K(xi, spoint)

            for j in 1:n
                V[i, j] += coeff * ell[j]
            end
        end
    end

    A = Matrix{T}(I, n, n)
    sigma = prob.sigma
    @inbounds for j in 1:n, i in 1:n
        A[i, j] -= sigma * V[i, j]
    end

    rhs = Vector{T}(undef, n)
    g = prob.g
    @inbounds for i in 1:n
        rhs[i] = g(x[i])
    end

    return FracCollocationSystem(prob, x, z, theta, wtheta, V, A, rhs, bary)
end

"""
    solve(prob, N)
    solve(sys)

Solve `(I - sigma*V) u = g` and return a `FracCollocationSolution`.
"""
solve(prob::FracVIEProblem, N::Integer) = solve(build_system(prob, N))

function solve(sys::FracCollocationSystem)
    values = sys.A \ sys.rhs
    return FracCollocationSolution(
        sys.problem,
        sys,
        sys.x,
        sys.z,
        Vector{eltype(sys.x)}(values),
        sys.bary,
    )
end

"""
    collocation_residual(sys, values)

Return the collocation residual

    values - sys.rhs - sys.problem.sigma * sys.V * values.

This is algebraically identical to `sys.A * values - sys.rhs` because
`sys.A = I - sigma*V`.
"""
function collocation_residual(sys::FracCollocationSystem, values::AbstractVector)
    n = length(sys.rhs)
    length(values) == n ||
        throw(DimensionMismatch("values must have length $(n)"))

    T = promote_type(eltype(sys.rhs), eltype(sys.V), eltype(values), typeof(sys.problem.sigma))
    tmp = Vector{T}(undef, n)
    r = Vector{T}(undef, n)

    mul!(tmp, sys.V, values)
    sigma = sys.problem.sigma
    @inbounds for i in 1:n
        r[i] = values[i] - sys.rhs[i] - sigma * tmp[i]
    end
    return r
end

function _evaluate_with_scratch(sol::FracCollocationSolution, xval::Real, ell)
    z_eval = xval^sol.problem.lambda
    lagrange_values!(ell, sol.z, sol.bary, z_eval)

    acc = zero(eltype(sol.values))
    @inbounds for j in eachindex(sol.values)
        acc += sol.values[j] * ell[j]
    end
    return acc
end

"""
    evaluate(sol, x)
    evaluate(sol, xs)

Evaluate the fractional polynomial interpolant
`sum_j sol.values[j] * ell_j(x^lambda)` at a scalar or array of points.
"""
function evaluate(sol::FracCollocationSolution, xval::Real)
    ell = similar(sol.values)
    return _evaluate_with_scratch(sol, xval, ell)
end

function evaluate(sol::FracCollocationSolution, xs::AbstractArray{<:Real})
    vals = similar(xs, eltype(sol.values))
    ell = similar(sol.values)
    @inbounds for idx in eachindex(xs)
        vals[idx] = _evaluate_with_scratch(sol, xs[idx], ell)
    end
    return vals
end

end # module
