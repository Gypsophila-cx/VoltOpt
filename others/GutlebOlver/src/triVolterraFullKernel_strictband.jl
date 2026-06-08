#############################
## Strict-banded triangle-Jacobi full-kernel Volterra construction.
##
## This file keeps the Clenshaw recurrence from triVolterraFullKernelOpP01,
## but stores each recurrence state explicitly as a vector of narrow
## BandedMatrix blocks.  In particular, the left and right multiplication by
## the one-dimensional Jacobi multiplication matrix are hand-written
## tridiagonal/banded kernels, so no full-band BandedMatrix temporaries are
## created by generic matrix multiplication.
#############################

function theoretical_bandwidth_P01(kerneldepth::Integer; flip::Bool=true)
    kerneldepth < 1 && throw(ArgumentError("kerneldepth must be positive"))
    b = max(Int(kerneldepth) - 1, 0)
    return (lower_band=b, upper_band=b)
end

function _strictband_resolve_band(v, auto::Integer, name::AbstractString)
    v === :auto && return Int(auto)
    if v isa Integer && v >= 0
        return Int(v)
    end
    throw(ArgumentError("$name must be :auto or a non-negative integer"))
end

_strictband_clamp(n::Integer, b::Integer) = min(max(Int(b), 0), max(Int(n) - 1, 0))

function _strictband_zero(n::Integer, lower_band::Integer, upper_band::Integer)
    l = _strictband_clamp(n, lower_band)
    u = _strictband_clamp(n, upper_band)
    A = BandedMatrix{Float64}(undef, (Int(n), Int(n)), (l, u))
    fill!(getfield(A, :data), 0.0)
    return A
end

function _strictband_bandwidths(A)
    l, u = BandedMatrices.bandwidths(A)
    return max(Int(l), 0), max(Int(u), 0)
end

@inline function _strictband_inband(i::Integer, j::Integer, l::Integer, u::Integer)
    return i - j <= l && j - i <= u
end

function _strictband_structural_nnz(n::Integer, l::Integer, u::Integer)
    nn = 0
    for j = 1:n
        nn += max(0, min(n, j + l) - max(1, j - u) + 1)
    end
    return nn
end

function _strictband_state_nnz(state)
    isempty(state) && return 0
    total = 0
    for A in state
        l, u = _strictband_bandwidths(A)
        total += _strictband_structural_nnz(size(A, 1), l, u)
    end
    return total
end

function _strictband_debug_state(label, state)
    if isempty(state)
        println(label, ": empty state")
        return
    end
    widths = unique(BandedMatrices.bandwidths.(state))
    println(label, ": blocks=", length(state), ", bandwidths=", widths,
            ", structural_nnz=", _strictband_state_nnz(state))
end

function axpy_banded!(Y, 伪, X; check_band::Bool=true)
    伪 == 0 && return Y
    size(Y) == size(X) || throw(DimensionMismatch("banded axpy sizes do not match"))
    lY, uY = _strictband_bandwidths(Y)
    lX, uX = _strictband_bandwidths(X)
    if check_band && (lX > lY || uX > uY)
        throw(ArgumentError("target bandwidth ($lY,$uY) does not contain source bandwidth ($lX,$uX)"))
    end

    n = size(Y, 1)
    l = min(lX, lY)
    u = min(uX, uY)
    @inbounds for j = 1:n
        ilo = max(1, j - u)
        ihi = min(n, j + l)
        for i = ilo:ihi
            Y[i, j] += 伪 * X[i, j]
        end
    end
    return Y
end

function lincomb_banded(coeffs, states, target_l::Integer, target_u::Integer; check_band::Bool=true)
    length(coeffs) == length(states) ||
        throw(DimensionMismatch("number of coefficients and states must match"))
    isempty(states) && throw(ArgumentError("at least one state is required"))
    n = size(first(states), 1)
    Y = _strictband_zero(n, target_l, target_u)
    for (伪, X) in zip(coeffs, states)
        axpy_banded!(Y, 伪, X; check_band=check_band)
    end
    return Y
end

function _strictband_scale(伪, X, target_l::Integer, target_u::Integer; check_band::Bool=true)
    Y = _strictband_zero(size(X, 1), target_l, target_u)
    axpy_banded!(Y, 伪, X; check_band=check_band)
    return Y
end

function _strictband_state_from_coeffs(cfs_block, D, target_l::Integer, target_u::Integer;
                                       check_band::Bool=true)
    return [_strictband_scale(c, D, target_l, target_u; check_band=check_band)
            for c in cfs_block]
end

function _strictband_mat_state_mul(M, states, target_l::Integer, target_u::Integer;
                                   check_band::Bool=true)
    rows, cols = size(M)
    length(states) == cols ||
        throw(DimensionMismatch("small recurrence matrix and state length do not match"))
    isempty(states) && return BandedMatrix{Float64}[]
    n = size(first(states), 1)
    out = [_strictband_zero(n, target_l, target_u) for _ = 1:rows]
    lM, uM = _strictband_bandwidths(M)

    @inbounds for i = 1:rows
        jlo = max(1, i - lM)
        jhi = min(cols, i + uM)
        for j = jlo:jhi
            coeff = M[i, j]
            coeff == 0 && continue
            axpy_banded!(out[i], coeff, states[j]; check_band=check_band)
        end
    end
    return out
end

function jacobi_xmat_P01_tridiag(depth::Integer)
    depth < 1 && throw(ArgumentError("depth must be positive"))
    S = Jacobi(0, 1, 0..1)
    x = Fun(identity, S)
    J = Multiplication(x, S)[1:depth, 1:depth]
    subdiag = zeros(Float64, max(Int(depth) - 1, 0))
    diag = zeros(Float64, Int(depth))
    supdiag = zeros(Float64, max(Int(depth) - 1, 0))
    @inbounds for i = 1:depth
        diag[i] = J[i, i]
        if i < depth
            subdiag[i] = J[i + 1, i]
            supdiag[i] = J[i, i + 1]
        end
    end
    return subdiag, diag, supdiag
end

function leftmul_tridiag_banded(sub, diag, sup, A, target_l::Integer, target_u::Integer;
                                check_band::Bool=true)
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("A must be square"))
    length(diag) == n || throw(DimensionMismatch("tridiagonal diagonal has wrong length"))
    length(sub) == max(n - 1, 0) || throw(DimensionMismatch("tridiagonal subdiag has wrong length"))
    length(sup) == max(n - 1, 0) || throw(DimensionMismatch("tridiagonal supdiag has wrong length"))

    lA, uA = _strictband_bandwidths(A)
    required_l = _strictband_clamp(n, lA + 1)
    required_u = _strictband_clamp(n, uA + 1)
    if check_band && (target_l < required_l || target_u < required_u)
        throw(ArgumentError("target bandwidth ($target_l,$target_u) is too small for left tridiagonal multiply; need ($required_l,$required_u)"))
    end

    Y = _strictband_zero(n, target_l, target_u)
    lY, uY = _strictband_bandwidths(Y)
    @inbounds for j = 1:n
        ilo = max(1, j - uY)
        ihi = min(n, j + lY)
        for i = ilo:ihi
            val = 0.0
            if i > 1 && _strictband_inband(i - 1, j, lA, uA)
                val += sub[i - 1] * A[i - 1, j]
            end
            if _strictband_inband(i, j, lA, uA)
                val += diag[i] * A[i, j]
            end
            if i < n && _strictband_inband(i + 1, j, lA, uA)
                val += sup[i] * A[i + 1, j]
            end
            Y[i, j] = val
        end
    end
    return Y
end

function rightmul_tridiag_banded(A, sub, diag, sup, target_l::Integer, target_u::Integer;
                                 check_band::Bool=true)
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("A must be square"))
    length(diag) == n || throw(DimensionMismatch("tridiagonal diagonal has wrong length"))
    length(sub) == max(n - 1, 0) || throw(DimensionMismatch("tridiagonal subdiag has wrong length"))
    length(sup) == max(n - 1, 0) || throw(DimensionMismatch("tridiagonal supdiag has wrong length"))

    lA, uA = _strictband_bandwidths(A)
    required_l = _strictband_clamp(n, lA + 1)
    required_u = _strictband_clamp(n, uA + 1)
    if check_band && (target_l < required_l || target_u < required_u)
        throw(ArgumentError("target bandwidth ($target_l,$target_u) is too small for right tridiagonal multiply; need ($required_l,$required_u)"))
    end

    Y = _strictband_zero(n, target_l, target_u)
    lY, uY = _strictband_bandwidths(Y)
    @inbounds for j = 1:n
        ilo = max(1, j - uY)
        ihi = min(n, j + lY)
        for i = ilo:ihi
            val = 0.0
            if j > 1 && _strictband_inband(i, j - 1, lA, uA)
                val += sup[j - 1] * A[i, j - 1]
            end
            if _strictband_inband(i, j, lA, uA)
                val += diag[j] * A[i, j]
            end
            if j < n && _strictband_inband(i, j + 1, lA, uA)
                val += sub[j] * A[i, j + 1]
            end
            Y[i, j] = val
        end
    end
    return Y
end

function _strictband_leftmul_banded(Aop, V, target_l::Integer, target_u::Integer;
                                    check_band::Bool=true)
    n = size(V, 1)
    size(V, 2) == n || throw(DimensionMismatch("V must be square"))
    size(Aop, 1) == n && size(Aop, 2) == n ||
        throw(DimensionMismatch("left operator and V sizes do not match"))

    lA, uA = _strictband_bandwidths(Aop)
    lV, uV = _strictband_bandwidths(V)
    required_l = _strictband_clamp(n, lA + lV)
    required_u = _strictband_clamp(n, uA + uV)
    if check_band && (target_l < required_l || target_u < required_u)
        throw(ArgumentError("target bandwidth ($target_l,$target_u) is too small for left banded multiply; need ($required_l,$required_u)"))
    end

    Y = _strictband_zero(n, target_l, target_u)
    lY, uY = _strictband_bandwidths(Y)
    @inbounds for j = 1:n
        ilo = max(1, j - uY)
        ihi = min(n, j + lY)
        for i = ilo:ihi
            klo = max(1, i - lA, j - uV)
            khi = min(n, i + uA, j + lV)
            val = 0.0
            for k = klo:khi
                val += Aop[i, k] * V[k, j]
            end
            Y[i, j] = val
        end
    end
    return Y
end

function _strictband_qey_diag(depth::Integer)
    D = _strictband_zero(depth, 0, 0)
    @inbounds for n = 1:depth
        D[n, n] = (-1)^(n + 1) / n
    end
    return D
end

function _strictband_add_state!(Y, states, 伪=1.0; check_band::Bool=true)
    length(Y) == length(states) || throw(DimensionMismatch("state lengths do not match"))
    for i = 1:length(Y)
        axpy_banded!(Y[i], 伪, states[i]; check_band=check_band)
    end
    return Y
end

function _strictband_initial_state(cfs, D, K::Integer, target_l::Integer, target_u::Integer;
                                   check_band::Bool=true)
    return _strictband_state_from_coeffs(view(cfs, Block(K)), D, target_l, target_u;
                                         check_band=check_band)
end

function _strictband_clenshaw_step(cfs, C, K::Integer, B1, B2, D,
                                   sub, diag, sup, target_l::Integer, target_u::Integer;
                                   check_band::Bool=true)
    state = _strictband_initial_state(cfs, D, K, target_l, target_u;
                                      check_band=check_band)

    Bx = getfield(C, 2)
    By = getfield(C, 3)
    B = getfield(C, 4)
    Ctail = getfield(C, 5)

    xstate = _strictband_mat_state_mul(Bx[K], B1, max(target_l - 1, 0), max(target_u - 1, 0);
                                       check_band=check_band)
    for i = 1:K
        axpy_banded!(state[i],
                     1.0,
                     leftmul_tridiag_banded(sub, diag, sup, xstate[i], target_l, target_u;
                                            check_band=check_band);
                     check_band=check_band)
    end

    ystate = _strictband_mat_state_mul(By[K], B1, max(target_l - 1, 0), max(target_u - 1, 0);
                                       check_band=check_band)
    for i = 1:K
        axpy_banded!(state[i],
                     1.0,
                     rightmul_tridiag_banded(ystate[i], sub, diag, sup, target_l, target_u;
                                             check_band=check_band);
                     check_band=check_band)
    end

    _strictband_add_state!(
        state,
        _strictband_mat_state_mul(B[K], B1, target_l, target_u; check_band=check_band),
        -1.0;
        check_band=check_band,
    )

    if K <= length(Ctail)
        _strictband_add_state!(
            state,
            _strictband_mat_state_mul(Ctail[K], B2, target_l, target_u; check_band=check_band),
            -1.0;
            check_band=check_band,
        )
    end

    return state
end

struct TriangularKernelCoefficients{CFS,REC}
    cfs::CFS
    recurrence::REC
    kerneldepth::Int
    requested_kerneldepth::Int
    flip::Bool
end

struct TriangularKernelPlan{SP,BASIS,GRID,TRANSFORM,REC}
    space::SP
    basis::BASIS
    grid::GRID
    transform::TRANSFORM
    recurrence::REC
    kerneldepth::Int
    requested_kerneldepth::Int
    flip::Bool
end

function _triangular_kernel_blocked_array_type()
    return isdefined(BlockArrays, :BlockedArray) ?
        getfield(BlockArrays, :BlockedArray) :
        getfield(BlockArrays, :PseudoBlockArray)
end

function _triangular_kernel_blocked_coeffs(Kfun, kerneldepth::Integer)
    BlockedArrayCompat = _triangular_kernel_blocked_array_type()
    return BlockedArrayCompat(pad(Kfun, sum(1:Int(kerneldepth))), 1:Int(kerneldepth))
end

function precompute_triangular_kernel_plan(kerneldepth::Integer, flip::Bool)
    kerneldepth < 1 && throw(ArgumentError("kerneldepth must be positive"))

    sp = JacobiTriangle()
    basis = sp[:, Block.(1:kerneldepth)]
    grid, transform = MultivariateOrthogonalPolynomials.plan_grid_transform(
        sp,
        (Block(kerneldepth),),
    )
    recurrence = ClenshawRecurrenceData(sp, Int(kerneldepth) + 2)

    return TriangularKernelPlan(
        sp,
        basis,
        grid,
        transform,
        recurrence,
        Int(kerneldepth),
        Int(kerneldepth),
        flip,
    )
end

function sample_triangular_kernel_values(plan::TriangularKernelPlan, K::Function)
    if plan.flip
        return map(xy -> K(1 - xy[1], xy[2]), plan.grid)
    else
        return map(xy -> K(xy[1], xy[2]), plan.grid)
    end
end

function project_triangular_kernel_coeffs_from_values(plan::TriangularKernelPlan, values)
    size(values) == size(plan.grid) ||
        throw(DimensionMismatch("sample values must have the same size as the plan grid"))

    Kfun = plan.transform * values
    realized_kerneldepth = blocksize(Kfun)[1]
    cfs = _triangular_kernel_blocked_coeffs(Kfun, realized_kerneldepth)

    return TriangularKernelCoefficients(
        cfs,
        plan.recurrence,
        Int(realized_kerneldepth),
        plan.requested_kerneldepth,
        plan.flip,
    )
end

function triangular_kernel_coeffs_from_plan(plan::TriangularKernelPlan, K::Function)
    values = sample_triangular_kernel_values(plan, K)
    return project_triangular_kernel_coeffs_from_values(plan, values)
end

function precompute_triangular_kernel_coeffs(K::Function, kerneldepth::Integer, flip::Bool)
    kerneldepth < 1 && throw(ArgumentError("kerneldepth must be positive"))

    sp = JacobiTriangle()
    xy = axes(sp, 1)
    x, y = first.(xy), last.(xy)
    if flip
        Kfun = sp[:, Block.(1:kerneldepth)] \ (@. K(1 - x, y))
    else
        Kfun = sp[:, Block.(1:kerneldepth)] \ (@. K(x, y))
    end

    realized_kerneldepth = blocksize(Kfun)[1]
    cfs = _triangular_kernel_blocked_coeffs(Kfun, realized_kerneldepth)
    recurrence = ClenshawRecurrenceData(sp, realized_kerneldepth + 2)

    return TriangularKernelCoefficients(cfs, recurrence, Int(realized_kerneldepth),
                                        Int(kerneldepth), flip)
end

function triVolterraFullKernelOpP01_strictband_from_coeffs(
    coeffs::TriangularKernelCoefficients,
    depth::Integer,
    flip::Bool=coeffs.flip;
    lower_band::Union{Int,Symbol}=:auto,
    upper_band::Union{Int,Symbol}=:auto,
    check_band::Bool=true,
    debug::Bool=false,
)
    depth < 1 && throw(ArgumentError("depth must be positive"))
    flip == coeffs.flip ||
        throw(ArgumentError("flip=$flip does not match precomputed coefficients with flip=$(coeffs.flip)"))

    theory = theoretical_bandwidth_P01(coeffs.kerneldepth; flip=flip)
    max_l = _strictband_resolve_band(lower_band, theory.lower_band, "lower_band")
    max_u = _strictband_resolve_band(upper_band, theory.upper_band, "upper_band")

    cfs = coeffs.cfs
    C = coeffs.recurrence
    N = coeffs.kerneldepth
    required_final = max(N - 1, 0)
    if check_band && (max_l < required_final || max_u < required_final)
        throw(ArgumentError("requested bandwidth ($max_l,$max_u) is too small for kernel block depth $N; need at least ($required_final,$required_final)"))
    end

    D = _strictband_qey_diag(depth)
    sub, diag, sup = jacobi_xmat_P01_tridiag(depth)

    B2 = _strictband_initial_state(cfs, D, N, 0, 0; check_band=check_band)
    debug && _strictband_debug_state("strictband level $N", B2)
    N == 1 && return first(B2)

    target_l = 1
    target_u = 1
    B1 = _strictband_initial_state(cfs, D, N - 1, target_l, target_u;
                                   check_band=check_band)

    Bx = getfield(C, 2)
    By = getfield(C, 3)
    B = getfield(C, 4)
    xstate = _strictband_mat_state_mul(Bx[N - 1], B2, 0, 0; check_band=check_band)
    ystate = _strictband_mat_state_mul(By[N - 1], B2, 0, 0; check_band=check_band)
    for i = 1:(N - 1)
        axpy_banded!(B1[i],
                     1.0,
                     leftmul_tridiag_banded(sub, diag, sup, xstate[i], target_l, target_u;
                                            check_band=check_band);
                     check_band=check_band)
        axpy_banded!(B1[i],
                     1.0,
                     rightmul_tridiag_banded(ystate[i], sub, diag, sup, target_l, target_u;
                                             check_band=check_band);
                     check_band=check_band)
    end
    _strictband_add_state!(
        B1,
        _strictband_mat_state_mul(B[N - 1], B2, target_l, target_u; check_band=check_band),
        -1.0;
        check_band=check_band,
    )
    debug && _strictband_debug_state("strictband level $(N - 1)", B1)

    for level = (N - 2):-1:1
        target_l = N - level
        target_u = N - level
        if check_band && (target_l > max_l || target_u > max_u)
            throw(ArgumentError("internal target bandwidth ($target_l,$target_u) exceeds requested maximum ($max_l,$max_u)"))
        end
        newB1 = _strictband_clenshaw_step(cfs, C, level, B1, B2, D,
                                          sub, diag, sup, target_l, target_u;
                                          check_band=check_band)
        B2 = B1
        B1 = newB1
        debug && _strictband_debug_state("strictband level $level", B1)
    end

    return first(B1)
end

function triVolterraFullKernelOpP01_strictband(
    K::Function,
    depth::Integer,
    flip::Bool,
    kerneldepth::Integer;
    lower_band::Union{Int,Symbol}=:auto,
    upper_band::Union{Int,Symbol}=:auto,
    check_band::Bool=true,
    debug::Bool=false,
)
    depth < 1 && throw(ArgumentError("depth must be positive"))
    kerneldepth < 1 && throw(ArgumentError("kerneldepth must be positive"))

    coeffs = precompute_triangular_kernel_coeffs(K, kerneldepth, flip)
    return triVolterraFullKernelOpP01_strictband_from_coeffs(coeffs, depth, flip;
                                                             lower_band=lower_band,
                                                             upper_band=upper_band,
                                                             check_band=check_band,
                                                             debug=debug)

end

function apply_reflect_lowering_strictband(V, n::Integer; check_band::Bool=true)
    n < 1 && throw(ArgumentError("n must be positive"))
    size(V, 1) == n && size(V, 2) == n ||
        throw(DimensionMismatch("V must be n by n"))
    lV, uV = _strictband_bandwidths(V)
    W = WLoweringP01P00(n)
    tmp = _strictband_leftmul_banded(W, V, lV + 1, uV; check_band=check_band)
    R = reflectPabtoPba(n)
    return _strictband_leftmul_banded(R, tmp, lV + 1, uV; check_band=check_band)
end

function apply_conversion_00_to_12_strictband(V, n::Integer; check_band::Bool=true)
    n < 1 && throw(ArgumentError("n must be positive"))
    size(V, 1) == n && size(V, 2) == n ||
        throw(DimensionMismatch("V must be n by n"))
    lV, uV = _strictband_bandwidths(V)
    C00to12 = Conversion(Jacobi(0, 0, 0..1), Jacobi(1, 2, 0..1))[1:n, 1:n]
    lC, uC = _strictband_bandwidths(C00to12)
    return _strictband_leftmul_banded(C00to12, V, lV + lC, uV + uC;
                                      check_band=check_band)
end

function build_VIDE_LHS_sec522_strictband(V, n::Integer; check_band::Bool=true)
    Vrl = apply_reflect_lowering_strictband(V, n; check_band=check_band)
    Vconv = apply_conversion_00_to_12_strictband(Vrl, n; check_band=check_band)
    lV, uV = _strictband_bandwidths(Vconv)
    D = Derivative(Jacobi(0, 1, 0..1), 1)[1:n, 1:n]
    lD, uD = _strictband_bandwidths(D)
    out = _strictband_zero(n, max(lV, lD), max(uV, uD))
    axpy_banded!(out, 1.0, D; check_band=false)
    axpy_banded!(out, -1.0, Vconv; check_band=check_band)
    return out
end
