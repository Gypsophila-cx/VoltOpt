"""
    triQgen(N)

Return the triangle integration operator `Q_y` on `JacobiTriangle()`.
It maps triangle coefficients to the one-dimensional shifted Jacobi trace
used by the Gutleb--Olver Volterra construction.
"""
function triQgen(N::Integer)
    # Generates Q operator as a block banded matrix.
    # Q*f(x,y) can be used to integrate f from 0 to 1-u via Jacobi polynomials on triangle.
    Q = BlockBandedMatrix(Zeros(N, sum(1:N)), Ones{Int}(N), 1:N, (0,0))
        for n=1:N
            view(Q, Block(n,n))[1,1] = 1.
        end
    return Q
end
"""
    triEygenP01(N)

Return the extension operator `E_y` from the one-dimensional
`Jacobi(0,1,0..1)` basis into the canonical triangle Jacobi basis.
"""
function triEygenP01(N::Integer)
    Ey = BlockBandedMatrix(Zeros(sum(1:N),N), 1:N,Ones{Int}(N), (0,0))
        for n=1:N
            for j=1:n
                view(Ey, Block(n,n))[j,1] = (-1)^n*(-1)^j*(2*j-1)/n
            end
        end
    return Ey
end
"""
    triQEygenP01(N)

Return the diagonal `Q_y * E_y` operator for the shifted `P^(0,1)` basis
without forming the block operators separately.
"""
function triQEygenP01(N::Integer)
    QEy=BandedMatrix{Float64}(undef, (N, N), (0,0))
    for n=1:N
        QEy[n,n]=(-1)^(n+1)/n
    end
    return QEy
end
"""
    WLoweringP01P00(N)

Return the weighted lowering operator that absorbs the `(1-x)` factor and
maps coefficients from `Jacobi(0,1,0..1)` to `Jacobi(0,0,0..1)`.
"""
function WLoweringP01P00(N::Integer)
    WLow=I+BandedMatrix{Float64}(-0.5*Ones(N,N), (1,0))
    return WLow
end
"""
    reflectPabtoPba(N)

Return the diagonal reflection operator that maps coefficients in the
reflected `P^(a,b)(1-x)` basis to the unreflected `P^(b,a)(x)` basis.  The
Jacobi parameters are interchanged by reflection and the diagonal signs are
`(-1)^(n+1)` in the package's one-based coefficient indexing.
"""
function reflectPabtoPba(N::Integer)
    Refl=BandedMatrix{Float64}(undef, (N, N), (0,0))
    for n=1:N
        Refl[n,n]=(-1)^(n+1)
    end
    return Refl
end
"""
    WLoweringP01Pm11(N)

Return the legacy weighted lowering operator that absorbs `(1-x)` and maps
from `Jacobi(0,1,0..1)` to `Jacobi(1,-1,0..1)`.  This is kept for regression
checks of the original workflow.
"""
function WLoweringP01Pm11(N::Integer)
    WLow=BandedMatrix{Float64}(undef, (N, N), (1,0))
    for c=1:N-1
        WLow[c,c]=(1-c)/(2*c)
        WLow[c+1,c]=-0.5
    end
    WLow[N,N]=(1-N)/(2*N)
    return WLow
end
"""
    ReflectorP01(N)

Return the legacy upper-triangular reflector from `P^(1,0)(x)` coefficients
to `P^(1,0)(1-t)` coefficients.  The strict-band workflows use
`reflectPabtoPba` instead.
"""
function ReflectorP01(N::Integer)
    Reflector=BandedMatrix{Float64}(undef, (N, N), (0,N))
    for r=1:N
        for col=r+1:N
            Reflector[r,col]=(-1)^(r+1)*2*r/col
        end
            Reflector[r,r]=(-1)^(r+1)
    end
    return Reflector
end
"""
    triEygenP00(N)

Return the extension operator from the one-dimensional Legendre
`Jacobi(0,0,0..1)` basis to the canonical triangle Jacobi basis.
"""
function triEygenP00(N::Integer)
    Ey = BlockBandedMatrix(Zeros(sum(1:N),N), 1:N,Ones{Int}(N), (0,1))
        for n=1:N
            for j=1:n-1
                view(Ey, Block(n-1,n))[j,1] = (-1)^n*(-1)^j*(2*j-1)/(2*n-1)
                view(Ey, Block(n,n))[j,1] = (-1)^n*(-1)^j*(2*j-1)/(2*n-1)
            end
            view(Ey, Block(n,n))[n,1] = 1.0
        end
    return Ey
end
"""
    triEygenP11(N)

Return the extension operator from the one-dimensional `Jacobi(1,1,0..1)`
basis into the triangle basis used by the legacy tests.
"""
function triEygenP11(N::Integer)
    Ey = BlockBandedMatrix(Zeros(sum(1:N),N), 1:N,Ones{Int}(N), (0,0))
        for n=1:N
                view(Ey, Block(n,n))[1,1] = (-1)^(n+1)
        end
    return Ey
end
"""
    triExgenP01(N)

Return the extension operator in the `x` direction from `Jacobi(0,1,0..1)`
coefficients into the canonical triangle Jacobi basis.
"""
function triExgenP01(N::Integer)
    Ex = BlockBandedMatrix(Zeros(sum(1:N), N), 1:N, Ones{Int}(N), (0,0))
        for n=1:N
            view(Ex, Block(n,n))[1,1] = 1
        end
    return Ex
end

"""
    build_derivative_P01_to_P12(N)

Return the first derivative operator used by the Section 5.2.2 VIDE body:
`Jacobi(0,1,0..1) -> Jacobi(1,2,0..1)`.
"""
function build_derivative_P01_to_P12(N::Integer)
    N < 1 && throw(ArgumentError("N must be positive"))
    return Derivative(Jacobi(0, 1, 0..1), 1)[1:N, 1:N]
end

initial_row_value(j::Integer) = isodd(j) ? 1.0 : -1.0

"""
    build_initial_evaluation_row(N)

Return the dense coefficient row for evaluating a `Jacobi(0,1,0..1)` expansion
at `x = 0`.  Assembly code inserts this row directly into sparse triplets so
the final semi-banded system is not materialized as a dense matrix.
"""
function build_initial_evaluation_row(N::Integer)
    N < 1 && throw(ArgumentError("N must be positive"))
    row = zeros(Float64, Int(N))
    @inbounds for j = 1:Int(N)
        row[j] = initial_row_value(j)
    end
    return row
end
