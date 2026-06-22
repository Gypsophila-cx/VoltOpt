using LinearAlgebra
using Printf

const SCRIPT_DIR = @__DIR__
const PROJECT_ROOT = normpath(joinpath(SCRIPT_DIR, "..", ".."))
const GUTLEB_OLVER_PROJECT = joinpath(PROJECT_ROOT, "others", "GutlebOlver")
const DATA_DIR = joinpath(SCRIPT_DIR, "data")
const OUTPUT_CSV = joinpath(DATA_DIR, "ex4_GO_depth_sweep_N200.csv")
const BIGFLOAT_CONSTRUCT_ONLY_CSV =
    joinpath(DATA_DIR, "ex4_GO_depth_sweep_bigfloat_construct_only.csv")

if !(GUTLEB_OLVER_PROJECT in LOAD_PATH)
    pushfirst!(LOAD_PATH, GUTLEB_OLVER_PROJECT)
end

using ApproxFun
using BandedMatrices
using BlockArrays
using FastGaussQuadrature
using GutlebOlver

const OMEGA = 20.0
const EXACT_FREQ = 50.0
const BASIS_LEN = 200
const DEGREE_USED = BASIS_LEN - 1
const DEPTHS = 2:60
const FLIP = true
const XGRID = range(0.0, 1.0; length=20001)
const SOLUTION_SPACE = Jacobi(0, 1, 0..1)
const BIGFLOAT_PRECISION_BITS = (128, 256)
const BIGFLOAT_KERNEL_QUAD_N = 96

@inline function sinc1(z::Real)
    return iszero(z) ? one(z) : sin(z) / z
end

@inline exact_solution(x::Real) = cos(EXACT_FREQ * Float64(x))

function rhs_exact(x, omega)
    T = promote_type(typeof(x), typeof(omega))
    xx = T(x)
    ww = T(omega)
    fifty = T(50)
    return cos(fifty * xx) +
           xx / T(2) * (
               sinc1((ww * xx - fifty) * xx) +
               sinc1((ww * xx + fifty) * xx)
           )
end

@inline analytic_rhs(x::Real) = rhs_exact(x, OMEGA)

@inline kernel(x::Real, y::Real) = cos(OMEGA * Float64(x) * Float64(y))

function csv_cell(value)
    if value === missing
        return ""
    elseif value isa AbstractFloat
        return @sprintf("%.17e", Float64(value))
    else
        s = string(value)
        if occursin(",", s) || occursin("\"", s) || occursin("\n", s) || occursin("\r", s)
            return "\"" * replace(s, "\"" => "\"\"") * "\""
        end
        return s
    end
end

const CSV_COLUMNS = (
    :depth,
    :basis_len,
    :degree_used,
    :omega,
    :max_error,
    :residual_relative,
    :operator_build_time,
    :solve_time,
    :maxabs_V,
    :maxabs_A,
    :nnz_V,
    :nnz_A,
    :status,
    :rhs_projection_time,
    :kernel_coeff_time,
    :volterra_operator_time,
    :system_assembly_time,
    :evaluation_time,
    :rhs_validation_error,
    :rhs_tail_norm,
    :realized_kerneldepth,
    :raw_operator_type,
    :system_type,
    :raw_operator_nnz,
    :system_nnz,
    :raw_lower_bandwidth,
    :raw_upper_bandwidth,
    :system_lower_bandwidth,
    :system_upper_bandwidth,
)

function write_header(path::AbstractString)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(string.(CSV_COLUMNS), ","))
        flush(io)
    end
    return path
end

function append_row(path::AbstractString, row::NamedTuple)
    open(path, "a") do io
        println(io, join((csv_cell(getproperty(row, c)) for c in CSV_COLUMNS), ","))
        flush(io)
    end
    return path
end

function bandwidth_component(A, idx::Integer)
    try
        return Int(safe_bandwidth(A)[idx])
    catch
        return missing
    end
end

function safe_nnz_value(A)
    try
        return Int(safe_nnz(A))
    catch
        return missing
    end
end

function error_message(err)
    return sprint(showerror, err)
end

function failure_row(depth::Integer, err; rhs_projection_time::Float64, rhs_info)
    return (
        depth=Int(depth),
        basis_len=BASIS_LEN,
        degree_used=DEGREE_USED,
        omega=OMEGA,
        max_error=NaN,
        residual_relative=NaN,
        operator_build_time=NaN,
        solve_time=NaN,
        maxabs_V=NaN,
        maxabs_A=NaN,
        nnz_V=missing,
        nnz_A=missing,
        status="failed: " * error_message(err),
        rhs_projection_time=rhs_projection_time,
        kernel_coeff_time=NaN,
        volterra_operator_time=NaN,
        system_assembly_time=NaN,
        evaluation_time=NaN,
        rhs_validation_error=getproperty(rhs_info, :validation_error),
        rhs_tail_norm=getproperty(rhs_info, :tail_norm),
        realized_kerneldepth=missing,
        raw_operator_type="",
        system_type="",
        raw_operator_nnz=missing,
        system_nnz=missing,
        raw_lower_bandwidth=missing,
        raw_upper_bandwidth=missing,
        system_lower_bandwidth=missing,
        system_upper_bandwidth=missing,
    )
end

function solve_for_depth(depth::Integer, rhs_coeff::AbstractVector{<:Real};
                         rhs_projection_time::Float64,
                         rhs_info)
    depth_int = Int(depth)
    local coeffs
    local Vraw
    local A
    local kernel_coeff_time
    local volterra_operator_time
    local system_assembly_time
    local solution_coeff
    local max_error

    operator_build_time = @elapsed begin
        kernel_coeff_time = @elapsed begin
            plan = precompute_triangular_kernel_plan(depth_int, FLIP)
            coeffs = triangular_kernel_coeffs_from_plan(plan, kernel)
        end
        volterra_operator_time = @elapsed begin
            Vraw = triVolterraFullKernelOpP01_strictband_from_coeffs(
                coeffs,
                BASIS_LEN,
                FLIP,
            )
        end
        system_assembly_time = @elapsed begin
            A = build_secondkind_system_strictband(Vraw, BASIS_LEN; sign=1.0)
        end
    end

    solve_time = @elapsed begin
        solution_coeff = A \ rhs_coeff
    end

    residual = A * solution_coeff - rhs_coeff
    residual_relative = norm(residual) / max(norm(rhs_coeff), eps(Float64))

    evaluation_time = @elapsed begin
        u_num = Fun(SOLUTION_SPACE, solution_coeff)
        max_error = maximum(abs(u_num(x) - exact_solution(x)) for x in XGRID)
    end

    return (
        depth=depth_int,
        basis_len=BASIS_LEN,
        degree_used=DEGREE_USED,
        omega=OMEGA,
        max_error=max_error,
        residual_relative=residual_relative,
        operator_build_time=operator_build_time,
        solve_time=solve_time,
        maxabs_V=Float64(matrix_maxabs(Vraw)),
        maxabs_A=Float64(matrix_maxabs(A)),
        nnz_V=safe_nnz_value(Vraw),
        nnz_A=safe_nnz_value(A),
        status="success",
        rhs_projection_time=rhs_projection_time,
        kernel_coeff_time=kernel_coeff_time,
        volterra_operator_time=volterra_operator_time,
        system_assembly_time=system_assembly_time,
        evaluation_time=evaluation_time,
        rhs_validation_error=getproperty(rhs_info, :validation_error),
        rhs_tail_norm=getproperty(rhs_info, :tail_norm),
        realized_kerneldepth=Int(coeffs.kerneldepth),
        raw_operator_type=string(nameof(typeof(Vraw))),
        system_type=string(nameof(typeof(A))),
        raw_operator_nnz=safe_nnz_value(Vraw),
        system_nnz=safe_nnz_value(A),
        raw_lower_bandwidth=bandwidth_component(Vraw, 1),
        raw_upper_bandwidth=bandwidth_component(Vraw, 2),
        system_lower_bandwidth=bandwidth_component(A, 1),
        system_upper_bandwidth=bandwidth_component(A, 2),
    )
end

const BIGFLOAT_CONSTRUCT_COLUMNS = (
    :precision_bits,
    :depth,
    :basis_len,
    :degree_used,
    :omega,
    :max_error,
    :residual_relative,
    :operator_build_time_bigfloat,
    :convert_time,
    :solve_time_float64,
    :maxabs_V_bigfloat,
    :maxabs_V_float64,
    :maxabs_A_float64,
    :status,
)

function write_bigfloat_construct_header(path::AbstractString)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(string.(BIGFLOAT_CONSTRUCT_COLUMNS), ","))
        flush(io)
    end
    return path
end

function append_bigfloat_construct_row(path::AbstractString, row::NamedTuple)
    open(path, "a") do io
        println(io, join((csv_cell(getproperty(row, c)) for c in BIGFLOAT_CONSTRUCT_COLUMNS), ","))
        flush(io)
    end
    return path
end

@inline triangle_coeff_index(m::Int, k::Int) = div(m * (m + 1), 2) + k + 1

function jacobi_values(alpha::T, beta::T, z::T, maxdegree::Int) where {T}
    values = Vector{T}(undef, maxdegree + 1)
    values[1] = one(T)
    maxdegree == 0 && return values

    values[2] = ((alpha - beta) + (alpha + beta + T(2)) * z) / T(2)
    for n = 1:(maxdegree - 1)
        nn = T(n)
        two = T(2)
        denom = two * (nn + one(T)) * (nn + alpha + beta + one(T)) *
                (two * nn + alpha + beta)
        term1 = (two * nn + alpha + beta + one(T)) *
                ((two * nn + alpha + beta) * (two * nn + alpha + beta + two) * z +
                 alpha^2 - beta^2)
        term2 = two * (nn + alpha) * (nn + beta) * (two * nn + alpha + beta + two)
        values[n + 2] = (term1 * values[n + 1] - term2 * values[n]) / denom
    end
    return values
end

function project_kernel_coefficients_bigfloat(maxdepth::Int, omega::BigFloat;
                                              quad_n::Int=BIGFLOAT_KERNEL_QUAD_N,
                                              flip::Bool=FLIP)
    xraw, wxraw = gausslegendre(BigFloat, quad_n)
    traw, wtraw = gausslegendre(BigFloat, quad_n)
    half = BigFloat(0.5)
    xs = (xraw .+ one(BigFloat)) .* half
    ts = (traw .+ one(BigFloat)) .* half
    wx = wxraw .* half
    wt = wtraw .* half

    tpoly = Matrix{BigFloat}(undef, maxdepth, quad_n)
    for it in 1:quad_n
        vals = jacobi_values(zero(BigFloat), zero(BigFloat), BigFloat(2) * ts[it] - one(BigFloat), maxdepth - 1)
        @inbounds for k = 1:maxdepth
            tpoly[k, it] = vals[k]
        end
    end

    xpoly = Vector{Matrix{BigFloat}}(undef, maxdepth)
    for k0 = 0:(maxdepth - 1)
        maxr = maxdepth - 1 - k0
        mat = Matrix{BigFloat}(undef, maxr + 1, quad_n)
        alpha = BigFloat(2 * k0 + 1)
        beta = zero(BigFloat)
        for ix in 1:quad_n
            vals = jacobi_values(alpha, beta, BigFloat(2) * xs[ix] - one(BigFloat), maxr)
            @inbounds mat[:, ix] .= vals
        end
        xpoly[k0 + 1] = mat
    end

    omx_powers = Matrix{BigFloat}(undef, maxdepth, quad_n)
    for ix in 1:quad_n
        omx = one(BigFloat) - xs[ix]
        p = omx
        for k0 = 0:(maxdepth - 1)
            omx_powers[k0 + 1, ix] = p
            p *= omx
        end
    end

    coeffs = zeros(BigFloat, div(maxdepth * (maxdepth + 1), 2))
    for ix in 1:quad_n
        xtri = xs[ix]
        omx = one(BigFloat) - xtri
        xphys = flip ? omx : xtri
        for it in 1:quad_n
            t = ts[it]
            y = omx * t
            kval = cos(omega * xphys * y)
            base = wx[ix] * wt[it] * kval
            for k0 = 0:(maxdepth - 1)
                common = base * omx_powers[k0 + 1, ix] * tpoly[k0 + 1, it]
                xpk = xpoly[k0 + 1]
                maxr = maxdepth - 1 - k0
                for r = 0:maxr
                    m = k0 + r
                    coeffs[triangle_coeff_index(m, k0)] += common * xpk[r + 1, ix]
                end
            end
        end
    end

    for m = 0:(maxdepth - 1)
        for k0 = 0:m
            coeffs[triangle_coeff_index(m, k0)] *= BigFloat(2 * (m + 1) * (2 * k0 + 1))
        end
    end
    return coeffs
end

function big_banded_copy(A)
    l, u = BandedMatrices.bandwidths(A)
    B = BandedMatrix{BigFloat}(undef, size(A), (Int(l), Int(u)))
    fill!(getfield(B, :data), zero(BigFloat))
    @inbounds for j in 1:size(A, 2)
        for i in max(1, j - Int(u)):min(size(A, 1), j + Int(l))
            B[i, j] = BigFloat(A[i, j])
        end
    end
    return B
end

function big_recurrence_from_float_plan(maxdepth::Int)
    plan = precompute_triangular_kernel_plan(maxdepth, FLIP)
    C = plan.recurrence
    return (
        getfield(C, 1),
        [big_banded_copy(A) for A in getfield(C, 2)],
        [big_banded_copy(A) for A in getfield(C, 3)],
        [big_banded_copy(A) for A in getfield(C, 4)],
        [big_banded_copy(A) for A in getfield(C, 5)],
    )
end

function big_kernel_coefficients_object(full_coeffs::Vector{BigFloat}, recurrence, depth::Int)
    ncoeff = div(depth * (depth + 1), 2)
    BlockedArrayCompat = GutlebOlver._triangular_kernel_blocked_array_type()
    cfs = BlockedArrayCompat(copy(view(full_coeffs, 1:ncoeff)), 1:depth)
    return GutlebOlver.TriangularKernelCoefficients(cfs, recurrence, depth, depth, FLIP)
end

_big_clamp(n::Integer, b::Integer) = min(max(Int(b), 0), max(Int(n) - 1, 0))

function big_zero_banded(::Type{T}, n::Integer, lower_band::Integer, upper_band::Integer) where {T}
    l = _big_clamp(n, lower_band)
    u = _big_clamp(n, upper_band)
    A = BandedMatrix{T}(undef, (Int(n), Int(n)), (l, u))
    fill!(getfield(A, :data), zero(T))
    return A
end

function big_bandwidths(A)
    l, u = BandedMatrices.bandwidths(A)
    return max(Int(l), 0), max(Int(u), 0)
end

@inline big_inband(i::Integer, j::Integer, l::Integer, u::Integer) =
    i - j <= l && j - i <= u

function big_axpy_banded!(Y, a, X; check_band::Bool=true)
    iszero(a) && return Y
    size(Y) == size(X) || throw(DimensionMismatch("banded axpy sizes do not match"))
    lY, uY = big_bandwidths(Y)
    lX, uX = big_bandwidths(X)
    if check_band && (lX > lY || uX > uY)
        throw(ArgumentError("target bandwidth ($lY,$uY) does not contain source bandwidth ($lX,$uX)"))
    end
    n = size(Y, 1)
    l = min(lX, lY)
    u = min(uX, uY)
    @inbounds for j = 1:n
        for i = max(1, j - u):min(n, j + l)
            Y[i, j] += a * X[i, j]
        end
    end
    return Y
end

function big_scale_banded(a, X, target_l::Integer, target_u::Integer; check_band::Bool=true)
    Y = big_zero_banded(eltype(X), size(X, 1), target_l, target_u)
    big_axpy_banded!(Y, a, X; check_band=check_band)
    return Y
end

function big_state_from_coeffs(cfs_block, D, K::Integer, target_l::Integer, target_u::Integer;
                               check_band::Bool=true)
    return [big_scale_banded(c, D, target_l, target_u; check_band=check_band) for c in cfs_block]
end

function big_mat_state_mul(M, states, target_l::Integer, target_u::Integer;
                           check_band::Bool=true)
    rows, cols = size(M)
    length(states) == cols || throw(DimensionMismatch("small recurrence matrix and state length do not match"))
    isempty(states) && return BandedMatrix{eltype(M)}[]
    T = promote_type(eltype(M), eltype(first(states)))
    n = size(first(states), 1)
    out = [big_zero_banded(T, n, target_l, target_u) for _ = 1:rows]
    lM, uM = big_bandwidths(M)
    @inbounds for i = 1:rows
        for j = max(1, i - lM):min(cols, i + uM)
            coeff = M[i, j]
            iszero(coeff) && continue
            big_axpy_banded!(out[i], coeff, states[j]; check_band=check_band)
        end
    end
    return out
end

function big_jacobi_xmat_P01_tridiag(depth::Integer)
    S = Jacobi(0, 1, 0..1)
    x = Fun(identity, S)
    J = Multiplication(x, S)[1:depth, 1:depth]
    subdiag = Vector{BigFloat}(undef, max(Int(depth) - 1, 0))
    diag = Vector{BigFloat}(undef, Int(depth))
    supdiag = Vector{BigFloat}(undef, max(Int(depth) - 1, 0))
    @inbounds for i = 1:Int(depth)
        diag[i] = BigFloat(J[i, i])
        if i < depth
            subdiag[i] = BigFloat(J[i + 1, i])
            supdiag[i] = BigFloat(J[i, i + 1])
        end
    end
    return subdiag, diag, supdiag
end

function big_leftmul_tridiag_banded(sub, diag, sup, A, target_l::Integer, target_u::Integer;
                                    check_band::Bool=true)
    n = size(A, 1)
    lA, uA = big_bandwidths(A)
    if check_band && (target_l < _big_clamp(n, lA + 1) || target_u < _big_clamp(n, uA + 1))
        throw(ArgumentError("target bandwidth is too small for left tridiagonal multiply"))
    end
    Y = big_zero_banded(eltype(A), n, target_l, target_u)
    lY, uY = big_bandwidths(Y)
    @inbounds for j = 1:n
        for i = max(1, j - uY):min(n, j + lY)
            val = zero(eltype(Y))
            if i > 1 && big_inband(i - 1, j, lA, uA)
                val += sub[i - 1] * A[i - 1, j]
            end
            if big_inband(i, j, lA, uA)
                val += diag[i] * A[i, j]
            end
            if i < n && big_inband(i + 1, j, lA, uA)
                val += sup[i] * A[i + 1, j]
            end
            Y[i, j] = val
        end
    end
    return Y
end

function big_rightmul_tridiag_banded(A, sub, diag, sup, target_l::Integer, target_u::Integer;
                                     check_band::Bool=true)
    n = size(A, 1)
    lA, uA = big_bandwidths(A)
    if check_band && (target_l < _big_clamp(n, lA + 1) || target_u < _big_clamp(n, uA + 1))
        throw(ArgumentError("target bandwidth is too small for right tridiagonal multiply"))
    end
    Y = big_zero_banded(eltype(A), n, target_l, target_u)
    lY, uY = big_bandwidths(Y)
    @inbounds for j = 1:n
        for i = max(1, j - uY):min(n, j + lY)
            val = zero(eltype(Y))
            if j > 1 && big_inband(i, j - 1, lA, uA)
                val += sup[j - 1] * A[i, j - 1]
            end
            if big_inband(i, j, lA, uA)
                val += diag[j] * A[i, j]
            end
            if j < n && big_inband(i, j + 1, lA, uA)
                val += sub[j] * A[i, j + 1]
            end
            Y[i, j] = val
        end
    end
    return Y
end

function big_leftmul_banded(Aop, V, target_l::Integer, target_u::Integer; check_band::Bool=true)
    n = size(V, 1)
    lA, uA = big_bandwidths(Aop)
    lV, uV = big_bandwidths(V)
    if check_band && (target_l < _big_clamp(n, lA + lV) || target_u < _big_clamp(n, uA + uV))
        throw(ArgumentError("target bandwidth is too small for left banded multiply"))
    end
    T = promote_type(eltype(Aop), eltype(V))
    Y = big_zero_banded(T, n, target_l, target_u)
    lY, uY = big_bandwidths(Y)
    @inbounds for j = 1:n
        for i = max(1, j - uY):min(n, j + lY)
            val = zero(T)
            for k = max(1, i - lA, j - uV):min(n, i + uA, j + lV)
                val += Aop[i, k] * V[k, j]
            end
            Y[i, j] = val
        end
    end
    return Y
end

function big_qey_diag(depth::Integer)
    D = big_zero_banded(BigFloat, depth, 0, 0)
    @inbounds for n = 1:Int(depth)
        D[n, n] = (isodd(n + 1) ? -one(BigFloat) : one(BigFloat)) / BigFloat(n)
    end
    return D
end

function big_add_state!(Y, states, a=one(BigFloat); check_band::Bool=true)
    length(Y) == length(states) || throw(DimensionMismatch("state lengths do not match"))
    for i = 1:length(Y)
        big_axpy_banded!(Y[i], a, states[i]; check_band=check_band)
    end
    return Y
end

function big_clenshaw_step(cfs, C, K::Integer, B1, B2, D, sub, diag, sup,
                           target_l::Integer, target_u::Integer; check_band::Bool=true)
    state = big_state_from_coeffs(view(cfs, Block(K)), D, K, target_l, target_u;
                                  check_band=check_band)
    Bx = getfield(C, 2)
    By = getfield(C, 3)
    B = getfield(C, 4)
    Ctail = getfield(C, 5)

    xstate = big_mat_state_mul(Bx[K], B1, max(target_l - 1, 0), max(target_u - 1, 0);
                               check_band=check_band)
    ystate = big_mat_state_mul(By[K], B1, max(target_l - 1, 0), max(target_u - 1, 0);
                               check_band=check_band)
    for i = 1:K
        big_axpy_banded!(state[i], one(BigFloat),
                         big_leftmul_tridiag_banded(sub, diag, sup, xstate[i], target_l, target_u;
                                                    check_band=check_band);
                         check_band=check_band)
        big_axpy_banded!(state[i], one(BigFloat),
                         big_rightmul_tridiag_banded(ystate[i], sub, diag, sup, target_l, target_u;
                                                     check_band=check_band);
                         check_band=check_band)
    end
    big_add_state!(state, big_mat_state_mul(B[K], B1, target_l, target_u; check_band=check_band),
                   -one(BigFloat); check_band=check_band)
    if K <= length(Ctail)
        big_add_state!(state, big_mat_state_mul(Ctail[K], B2, target_l, target_u; check_band=check_band),
                       -one(BigFloat); check_band=check_band)
    end
    return state
end

function big_triVolterraFullKernelOpP01_strictband_from_coeffs(coeffs, depth::Integer;
                                                               check_band::Bool=true)
    cfs = coeffs.cfs
    C = coeffs.recurrence
    N = coeffs.kerneldepth
    D = big_qey_diag(depth)
    sub, diag, sup = big_jacobi_xmat_P01_tridiag(depth)

    B2 = big_state_from_coeffs(cfs, D, N, 0, 0; check_band=check_band)
    N == 1 && return first(B2)

    B1 = big_state_from_coeffs(cfs, D, N - 1, 1, 1; check_band=check_band)
    Bx = getfield(C, 2)
    By = getfield(C, 3)
    B = getfield(C, 4)
    xstate = big_mat_state_mul(Bx[N - 1], B2, 0, 0; check_band=check_band)
    ystate = big_mat_state_mul(By[N - 1], B2, 0, 0; check_band=check_band)
    for i = 1:(N - 1)
        big_axpy_banded!(B1[i], one(BigFloat),
                         big_leftmul_tridiag_banded(sub, diag, sup, xstate[i], 1, 1;
                                                    check_band=check_band);
                         check_band=check_band)
        big_axpy_banded!(B1[i], one(BigFloat),
                         big_rightmul_tridiag_banded(ystate[i], sub, diag, sup, 1, 1;
                                                     check_band=check_band);
                         check_band=check_band)
    end
    big_add_state!(B1, big_mat_state_mul(B[N - 1], B2, 1, 1; check_band=check_band),
                   -one(BigFloat); check_band=check_band)

    for level = (N - 2):-1:1
        target_l = N - level
        target_u = N - level
        newB1 = big_clenshaw_step(cfs, C, level, B1, B2, D, sub, diag, sup,
                                  target_l, target_u; check_band=check_band)
        B2 = B1
        B1 = newB1
    end
    return first(B1)
end

function big_WLoweringP01P00(n::Integer)
    W = big_zero_banded(BigFloat, n, 1, 0)
    @inbounds for j = 1:Int(n)
        W[j, j] = one(BigFloat)
        j < n && (W[j + 1, j] = -BigFloat(0.5))
    end
    return W
end

function big_reflectPabtoPba(n::Integer)
    R = big_zero_banded(BigFloat, n, 0, 0)
    @inbounds for j = 1:Int(n)
        R[j, j] = isodd(j + 1) ? -one(BigFloat) : one(BigFloat)
    end
    return R
end

function big_conversion_00_to_01(n::Integer)
    C = big_zero_banded(BigFloat, n, 0, 1)
    @inbounds for j = 1:Int(n)
        C[j, j] = BigFloat(j) / BigFloat(2 * j - 1)
        j < n && (C[j, j + 1] = -BigFloat(j) / BigFloat(2 * j + 1))
    end
    return C
end

function big_build_secondkind_volterra_strictband(Vraw, n::Integer)
    lV, uV = big_bandwidths(Vraw)
    tmp = big_leftmul_banded(big_WLoweringP01P00(n), Vraw, lV + 1, uV)
    Vrl = big_leftmul_banded(big_reflectPabtoPba(n), tmp, lV + 1, uV)
    lR, uR = big_bandwidths(Vrl)
    C = big_conversion_00_to_01(n)
    lC, uC = big_bandwidths(C)
    return big_leftmul_banded(C, Vrl, lR + lC, uR + uC)
end

function matrix_maxabs(A)
    mx = zero(eltype(A))
    @inbounds for j = 1:size(A, 2)
        for i = 1:size(A, 1)
            v = abs(A[i, j])
            v > mx && (mx = v)
        end
    end
    return mx
end

function bigfloat_volterra_for_depth(full_coeffs::Vector{BigFloat}, recurrence, depth::Int)
    coeffs = big_kernel_coefficients_object(full_coeffs, recurrence, depth)
    Vraw = big_triVolterraFullKernelOpP01_strictband_from_coeffs(coeffs, BASIS_LEN)
    return big_build_secondkind_volterra_strictband(Vraw, BASIS_LEN)
end

function banded_float64_copy(A)
    l, u = BandedMatrices.bandwidths(A)
    return BandedMatrix{Float64}(Float64.(Matrix(A)), (Int(l), Int(u)))
end

function add_identity_banded!(A)
    @inbounds for i = 1:min(size(A)...)
        A[i, i] += 1.0
    end
    return A
end

function bigfloat_construct_failure_row(precision_bits::Integer, depth::Integer, err)
    return (
        precision_bits=Int(precision_bits),
        depth=Int(depth),
        basis_len=BASIS_LEN,
        degree_used=DEGREE_USED,
        omega=OMEGA,
        max_error=NaN,
        residual_relative=NaN,
        operator_build_time_bigfloat=NaN,
        convert_time=NaN,
        solve_time_float64=NaN,
        maxabs_V_bigfloat=NaN,
        maxabs_V_float64=NaN,
        maxabs_A_float64=NaN,
        status="failed: " * error_message(err),
    )
end

function solve_bigfloat_construct_only_depth(precision_bits::Int, depth::Int,
                                             full_coeffs::Vector{BigFloat}, recurrence,
                                             rhs_coeff::Vector{Float64})
    local V_big
    operator_build_time_bigfloat = @elapsed begin
        V_big = setprecision(BigFloat, precision_bits) do
            bigfloat_volterra_for_depth(full_coeffs, recurrence, depth)
        end
    end
    maxabs_V_bigfloat = Float64(matrix_maxabs(V_big))

    local V_float64
    convert_time = @elapsed begin
        V_float64 = banded_float64_copy(V_big)
    end
    maxabs_V_float64 = matrix_maxabs(V_float64)

    A = copy(V_float64)
    add_identity_banded!(A)
    maxabs_A_float64 = matrix_maxabs(A)

    local solution_coeff
    solve_time_float64 = @elapsed begin
        solution_coeff = A \ rhs_coeff
    end

    residual = A * solution_coeff - rhs_coeff
    residual_relative = norm(residual) / max(norm(rhs_coeff), eps(Float64))

    u_num = Fun(SOLUTION_SPACE, solution_coeff)
    max_error = maximum(abs(u_num(x) - exact_solution(x)) for x in XGRID)

    return (
        precision_bits=precision_bits,
        depth=depth,
        basis_len=BASIS_LEN,
        degree_used=DEGREE_USED,
        omega=OMEGA,
        max_error=max_error,
        residual_relative=residual_relative,
        operator_build_time_bigfloat=operator_build_time_bigfloat,
        convert_time=convert_time,
        solve_time_float64=solve_time_float64,
        maxabs_V_bigfloat=maxabs_V_bigfloat,
        maxabs_V_float64=maxabs_V_float64,
        maxabs_A_float64=maxabs_A_float64,
        status="ok",
    )
end

function print_bigfloat_construct_row(row)
    if row.status == "ok"
        @printf(
            "construct precision_bits=%3d depth=%2d max_error=%.6e residual_relative=%.6e build_big=%.6e convert=%.6e solve=%.6e\n",
            row.precision_bits,
            row.depth,
            row.max_error,
            row.residual_relative,
            row.operator_build_time_bigfloat,
            row.convert_time,
            row.solve_time_float64,
        )
    else
        println("construct precision_bits=", row.precision_bits,
                " depth=", row.depth, " status=", row.status)
    end
    flush(stdout)
end

function run_bigfloat_construct_only_depth_sweep()
    println()
    println("Gutleb--Olver BigFloat construct-only depth sweep")
    println("output_csv   = ", BIGFLOAT_CONSTRUCT_ONLY_CSV)
    println("basis_len    = ", BASIS_LEN)
    println("degree_used  = ", DEGREE_USED)
    println("omega        = ", OMEGA)
    println("depths       = ", first(DEPTHS), ":", last(DEPTHS))
    println("quad_n       = ", BIGFLOAT_KERNEL_QUAD_N)

    rhs_coeff, _ = build_go_rhs_coefficients(
        analytic_rhs,
        BASIS_LEN;
        alpha=0,
        beta=1,
        domain=0..1,
        method=:fixed_gauss_jacobi,
        validate=false,
    )
    rhs_coeff = Vector{Float64}(rhs_coeff)

    write_bigfloat_construct_header(BIGFLOAT_CONSTRUCT_ONLY_CSV)
    rows = NamedTuple[]
    for precision_bits in BIGFLOAT_PRECISION_BITS
        println()
        println("Preparing BigFloat GO construction cache, precision_bits = ", precision_bits)
        local full_coeffs
        local recurrence
        cache_time = @elapsed begin
            setprecision(BigFloat, precision_bits) do
                full_coeffs = project_kernel_coefficients_bigfloat(
                    last(DEPTHS),
                    BigFloat(OMEGA);
                    quad_n=BIGFLOAT_KERNEL_QUAD_N,
                    flip=FLIP,
                )
                recurrence = big_recurrence_from_float_plan(last(DEPTHS))
            end
        end
        @printf("construct cache precision_bits=%d time=%.6e\n", precision_bits, cache_time)

        precision_rows = NamedTuple[]
        for depth in DEPTHS
            row = try
                solve_bigfloat_construct_only_depth(
                    Int(precision_bits),
                    Int(depth),
                    full_coeffs,
                    recurrence,
                    rhs_coeff,
                )
            catch err
                bigfloat_construct_failure_row(precision_bits, depth, err)
            end
            append_bigfloat_construct_row(BIGFLOAT_CONSTRUCT_ONLY_CSV, row)
            push!(rows, row)
            push!(precision_rows, row)
            print_bigfloat_construct_row(row)
        end

        ok_rows = [r for r in precision_rows if r.status == "ok" && isfinite(r.max_error)]
        if isempty(ok_rows)
            println("construct precision_bits = ", precision_bits, ": no successful depth rows.")
        else
            best = ok_rows[argmin([r.max_error for r in ok_rows])]
            @printf(
                "construct precision_bits = %d: best depth = %d, min max_error = %.17e\n",
                precision_bits,
                best.depth,
                best.max_error,
            )
        end
    end
    println("Wrote CSV: ", BIGFLOAT_CONSTRUCT_ONLY_CSV)
    return rows
end

function print_row(row)
    if row.status == "success"
        @printf(
            "depth=%2d max_error=%.6e residual_relative=%.6e build=%.6e solve=%.6e\n",
            row.depth,
            row.max_error,
            row.residual_relative,
            row.operator_build_time,
            row.solve_time,
        )
    else
        println("depth=", row.depth, " status=", row.status)
    end
    flush(stdout)
end

function main()
    println("Gutleb--Olver depth sweep for oscillatory second-kind Volterra equation")
    println("project_root = ", PROJECT_ROOT)
    println("go_project   = ", GUTLEB_OLVER_PROJECT)
    println("output_csv   = ", OUTPUT_CSV)
    println("basis_len    = ", BASIS_LEN)
    println("degree_used  = ", DEGREE_USED)
    println("omega        = ", OMEGA)
    println("depths       = ", first(DEPTHS), ":", last(DEPTHS))

    write_header(OUTPUT_CSV)

    local rhs_coeff
    local rhs_info
    rhs_projection_time = @elapsed begin
        rhs_coeff, rhs_info = build_go_rhs_coefficients(
            analytic_rhs,
            BASIS_LEN;
            alpha=0,
            beta=1,
            domain=0..1,
            method=:fixed_gauss_jacobi,
            validate=true,
            validation_grid=XGRID,
        )
    end

    println("rhs_projection_time = ", rhs_projection_time)
    println("rhs_validation_error = ", rhs_info.validation_error)
    println("rhs_tail_norm = ", rhs_info.tail_norm)
    println()

    rows = NamedTuple[]
    for depth in DEPTHS
        row = try
            solve_for_depth(
                depth,
                rhs_coeff;
                rhs_projection_time=rhs_projection_time,
                rhs_info=rhs_info,
            )
        catch err
            failure_row(depth, err; rhs_projection_time=rhs_projection_time, rhs_info=rhs_info)
        end
        append_row(OUTPUT_CSV, row)
        push!(rows, row)
        print_row(row)
    end

    ok_rows = [r for r in rows if r.status == "success" && isfinite(r.max_error)]
    println()
    if isempty(ok_rows)
        println("Summary: no successful depth rows.")
    else
        best_index = argmin([r.max_error for r in ok_rows])
        best = ok_rows[best_index]
        @printf(
            "Summary: best depth = %d, max_error = %.17e, residual_relative = %.17e\n",
            best.depth,
            best.max_error,
            best.residual_relative,
        )
    end
    println("Wrote CSV: ", OUTPUT_CSV)
    return rows
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
