"""
    restmat_jacobi(a, b, N, alpha, beta; T=Float64, check=false)

Construct the restriction matrix in the Jacobi basis on `[-1, 1]`.

For `z = c*x + d`, with `c = (b-a)/2` and `d = (b+a)/2`, the returned
matrix `R` has size `(N+1, N+1)` and satisfies

    P_j^(alpha,beta)(c*x+d) = sum(R[i+1,j+1] * P_i^(alpha,beta)(x), i=0:j)

for `j = 0, ..., N`.  Thus `R * u` gives the restricted Jacobi
coefficients for a coefficient vector `u` truncated at degree `N`.

The implementation uses the standard Jacobi normalization
`P_n^(alpha,beta)(1) = binomial(n+alpha,n)` and the three-term recurrence
for multiplication by `x`.  `alpha` and `beta` must be greater than `-1`.

Set `check=true` to run a small value-space consistency check.  A numeric
value for `check` is interpreted as the relative tolerance.
"""
function restmat_jacobi(a, b, N, alpha, beta; T=Float64, check=false)
    TT = _restmat_jacobi_float_type(T)
    NN = _restmat_jacobi_check_N(N)

    aa = TT(a)
    bb = TT(b)
    alphaT = TT(alpha)
    betaT = TT(beta)
    _restmat_jacobi_check_ab(alphaT, betaT)

    c = (bb - aa) / TT(2)
    d = (bb + aa) / TT(2)

    R = zeros(TT, NN + 1, NN + 1)
    qprev = zeros(TT, NN + 1)
    qcur = zeros(TT, NN + 1)
    qnext = zeros(TT, NN + 1)
    xq = zeros(TT, NN + 1)

    qprev[1] = one(TT)
    R[1, 1] = one(TT)

    if NN >= 1
        _restmat_jacobi_mulx!(xq, qprev, 0, alphaT, betaT)

        p1_const = (alphaT - betaT) / TT(2)
        p1_slope = (alphaT + betaT + TT(2)) / TT(2)

        qcur[1] = p1_slope * c * xq[1] + p1_const + p1_slope * d
        qcur[2] = p1_slope * c * xq[2]
        R[1, 2] = qcur[1]
        R[2, 2] = qcur[2]
    end

    @inbounds for n in 1:NN-1
        _restmat_jacobi_mulx!(xq, qcur, n, alphaT, betaT)
        A, B, C = _restmat_jacobi_x_coeffs(n, alphaT, betaT)
        shift = d - B
        invA = one(TT) / A

        for i in 1:n+2
            qnext[i] = c * xq[i]
        end
        for i in 1:n+1
            qnext[i] += shift * qcur[i]
        end
        for i in 1:n
            qnext[i] -= C * qprev[i]
        end
        for i in 1:n+2
            qnext[i] *= invA
            R[i, n+2] = qnext[i]
        end

        qprev, qcur, qnext = qcur, qnext, qprev
    end

    if check !== false
        _restmat_jacobi_selfcheck(R, c, d, NN, alphaT, betaT, check)
    end

    return R
end

function _restmat_jacobi_float_type(T)
    (T isa Type && T <: AbstractFloat) ||
        throw(ArgumentError("T must be an AbstractFloat type."))
    return T
end

function _restmat_jacobi_check_N(N)
    N isa Integer || throw(ArgumentError("N must be an integer."))
    N >= 0 || throw(ArgumentError("N must be nonnegative."))
    return Int(N)
end

function _restmat_jacobi_check_ab(alpha, beta)
    alpha > -one(alpha) || throw(ArgumentError("alpha must be greater than -1."))
    beta > -one(beta) || throw(ArgumentError("beta must be greater than -1."))
    return nothing
end

# Coefficients in
#   x P_n = A_n P_{n+1} + B_n P_n + C_n P_{n-1}.
@inline function _restmat_jacobi_x_coeffs(n::Int, alpha::T, beta::T) where {T<:AbstractFloat}
    s = alpha + beta

    if n == 0
        A = T(2) / (s + T(2))
        B = (beta - alpha) / (s + T(2))
        return A, B, zero(T)
    end

    nn = T(n)
    A = T(2) * (nn + one(T)) * (nn + s + one(T)) /
        ((T(2)*nn + s + one(T)) * (T(2)*nn + s + T(2)))
    B = (beta^2 - alpha^2) /
        ((T(2)*nn + s) * (T(2)*nn + s + T(2)))
    C = T(2) * (nn + alpha) * (nn + beta) /
        ((T(2)*nn + s) * (T(2)*nn + s + one(T)))

    return A, B, C
end

function _restmat_jacobi_mulx!(
    out::AbstractVector{T},
    v::AbstractVector{T},
    deg::Int,
    alpha::T,
    beta::T,
) where {T<:AbstractFloat}
    z = zero(T)
    @inbounds for i in 1:deg+2
        out[i] = z
    end

    @inbounds for n in 0:deg
        vn = v[n+1]
        vn == z && continue
        A, B, C = _restmat_jacobi_x_coeffs(n, alpha, beta)
        out[n+2] += A * vn
        out[n+1] += B * vn
        if n > 0
            out[n] += C * vn
        end
    end

    return out
end

function _restmat_jacobi_values!(
    p::AbstractVector{T},
    x::T,
    N::Int,
    alpha::T,
    beta::T,
) where {T<:AbstractFloat}
    p[1] = one(T)

    if N >= 1
        p[2] = ((alpha - beta) + (alpha + beta + T(2)) * x) / T(2)
    end

    @inbounds for n in 1:N-1
        A, B, C = _restmat_jacobi_x_coeffs(n, alpha, beta)
        p[n+2] = ((x - B) * p[n+1] - C * p[n]) / A
    end

    return p
end

function _restmat_jacobi_check_tol(check, ::Type{T}) where {T<:AbstractFloat}
    if check === true
        return sqrt(eps(T))
    elseif check isa Real
        tol = T(check)
        tol > zero(T) || throw(ArgumentError("check tolerance must be positive."))
        return tol
    else
        throw(ArgumentError("check must be false, true, or a positive tolerance."))
    end
end

function _restmat_jacobi_selfcheck(
    R::AbstractMatrix{T},
    c::T,
    d::T,
    N::Int,
    alpha::T,
    beta::T,
    check,
) where {T<:AbstractFloat}
    tol = _restmat_jacobi_check_tol(check, T)
    m = max(N + 1, 5)
    px = zeros(T, N + 1)
    pz = zeros(T, N + 1)
    maxerr = zero(T)
    maxscale = one(T)

    @inbounds for k in 0:m-1
        x = -one(T) + T(2) * T(k) / T(m - 1)
        z = c * x + d
        _restmat_jacobi_values!(px, x, N, alpha, beta)
        _restmat_jacobi_values!(pz, z, N, alpha, beta)

        for j in 0:N
            rhs = zero(T)
            for i in 0:j
                rhs += R[i+1, j+1] * px[i+1]
            end
            err = abs(rhs - pz[j+1])
            scale = max(abs(rhs), abs(pz[j+1]), one(T))
            maxerr = max(maxerr, err)
            maxscale = max(maxscale, scale)
        end
    end

    maxerr <= tol * maxscale ||
        throw(ErrorException("restmat_jacobi self-check failed: relative error $(maxerr / maxscale) exceeds $(tol)."))

    return nothing
end
