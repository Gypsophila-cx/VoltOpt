"""
    restmat_cheb(a, b, N; T=Float64, check=false)

Construct the restriction matrix in the Chebyshev basis `T_n(x)` on
`[-1, 1]`.

For `z = c*x + d`, with `c = (b-a)/2` and `d = (b+a)/2`, the returned
matrix `R` has size `(N+1, N+1)` and satisfies

    T_j(c*x+d) = sum(R[i+1,j+1] * T_i(x), i=0:j)

for `j = 0, ..., N`.  Thus `R * u` gives the restricted Chebyshev
coefficients for a coefficient vector `u` truncated at degree `N`.

Set `check=true` to run a small value-space consistency check.  A numeric
value for `check` is interpreted as the relative tolerance.
"""
function restmat_cheb(a, b, N; T=Float64, check=false)
    TT = _restmat_cheb_float_type(T)
    NN = _restmat_cheb_check_N(N)

    aa = TT(a)
    bb = TT(b)
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
        qcur[1] = d
        qcur[2] = c
        R[1, 2] = d
        R[2, 2] = c
    end

    @inbounds for n in 1:NN-1
        _restmat_cheb_mulx!(xq, qcur, n)

        for i in 1:n+2
            qnext[i] = TT(2) * c * xq[i]
        end
        for i in 1:n+1
            qnext[i] += TT(2) * d * qcur[i]
        end
        for i in 1:n
            qnext[i] -= qprev[i]
        end
        for i in 1:n+2
            R[i, n+2] = qnext[i]
        end

        qprev, qcur, qnext = qcur, qnext, qprev
    end

    if check !== false
        _restmat_cheb_selfcheck(R, c, d, NN, check)
    end

    return R
end

function _restmat_cheb_float_type(T)
    (T isa Type && T <: AbstractFloat) ||
        throw(ArgumentError("T must be an AbstractFloat type."))
    return T
end

function _restmat_cheb_check_N(N)
    N isa Integer || throw(ArgumentError("N must be an integer."))
    N >= 0 || throw(ArgumentError("N must be nonnegative."))
    return Int(N)
end

function _restmat_cheb_mulx!(
    out::AbstractVector{T},
    v::AbstractVector{T},
    deg::Int,
) where {T<:AbstractFloat}
    z = zero(T)
    @inbounds for i in 1:deg+2
        out[i] = z
    end

    @inbounds for n in 0:deg
        vn = v[n+1]
        vn == z && continue

        if n == 0
            out[2] += vn
        else
            half_vn = vn / T(2)
            out[n] += half_vn
            out[n+2] += half_vn
        end
    end

    return out
end

function _restmat_cheb_values!(
    p::AbstractVector{T},
    x::T,
    N::Int,
) where {T<:AbstractFloat}
    p[1] = one(T)

    if N >= 1
        p[2] = x
    end

    @inbounds for n in 1:N-1
        p[n+2] = T(2) * x * p[n+1] - p[n]
    end

    return p
end

function _restmat_cheb_check_tol(check, ::Type{T}) where {T<:AbstractFloat}
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

function _restmat_cheb_selfcheck(
    R::AbstractMatrix{T},
    c::T,
    d::T,
    N::Int,
    check,
) where {T<:AbstractFloat}
    tol = _restmat_cheb_check_tol(check, T)
    m = max(N + 1, 5)
    tx = zeros(T, N + 1)
    tz = zeros(T, N + 1)
    maxerr = zero(T)
    maxscale = one(T)

    @inbounds for k in 0:m-1
        x = -one(T) + T(2) * T(k) / T(m - 1)
        z = c * x + d
        _restmat_cheb_values!(tx, x, N)
        _restmat_cheb_values!(tz, z, N)

        for j in 0:N
            rhs = zero(T)
            for i in 0:j
                rhs += R[i+1, j+1] * tx[i+1]
            end
            err = abs(rhs - tz[j+1])
            scale = max(abs(rhs), abs(tz[j+1]), one(T))
            maxerr = max(maxerr, err)
            maxscale = max(maxscale, scale)
        end
    end

    maxerr <= tol * maxscale ||
        throw(ErrorException("restmat_cheb self-check failed: relative error $(maxerr / maxscale) exceeds $(tol)."))

    return nothing
end
