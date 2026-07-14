# Shared Gutleb-Olver instability validation utilities.
#
# Problem verified by this file:
#   T_{d,M} z = b is the finite strict-band operator-valued Clenshaw system
#   used by GutlebOlver.triVolterraFullKernelOpP01_strictband_from_coeffs.
#   The output map P_{Nout,M} extracts the level-1 first raw Volterra block,
#   applies weighted lowering, reflection, Jacobi(0,0)->Jacobi(0,1)
#   conversion, and crops the leading Nout-by-Nout block.
#
# Inputs:
#   A GutlebOlver TriangularKernelCoefficients object, matrix truncation N,
#   kernel label, and working scalar type.
#
# Outputs:
#   Matrix-free apply_T, solve_T, apply_P, explicit small-system T/P builders,
#   state packing helpers, and CSV utilities used by the final validation
#   scripts in this directory.
#
# Recommended use:
#   include("go_instability_core.jl") from scripts in this directory.
#
# Precision:
#   The native coefficients and recurrence matrices are produced by the
#   current Float64 Gutleb-Olver projection.  BigFloat experiments lift that
#   same finite system to higher precision before applying T, solving T, and
#   applying P.
#
# Explicit T:
#   Dense T and P are formed only by explicit_T_matrix/explicit_P_matrix for
#   small d and M.  Production-size experiments use apply_T/solve_T/apply_P.

script_dir = @__DIR__
project_root = normpath(joinpath(script_dir, "..", ".."))
go_project = joinpath(project_root, "others", "GutlebOlver")
if !(go_project in LOAD_PATH)
    pushfirst!(LOAD_PATH, go_project)
end

module GOInstabilityCore

using ApproxFun
using BandedMatrices
using BlockArrays
using GutlebOlver
using LinearAlgebra
using Printf
using Random

export KernelSpec,
       GOTransferSystem,
       GOClenshawState,
       default_kernel_specs,
       make_coeffs,
       make_exact_constant_coeffs,
       make_controlled_tail_constant_coeffs,
       build_transfer_system,
       build_transfer_system_explicit,
       build_transfer_system_independent,
       lift_transfer_system,
       analytic_jacobi_xmat_P01_tridiag,
       analytic_triangle_jacobi_block,
       analytic_clenshaw_recurrence_blocks,
       analytic_WLoweringP01P00,
       analytic_reflectPabtoPba,
       analytic_conversion_00_to_01,
       rhs_state,
       solve_T_rhs,
       solve_T,
       apply_T,
       recurrence_residual,
       streaming_residual_stats,
       apply_P_raw,
       apply_P_final,
       apply_P,
       apply_P_stage,
       native_raw_operator,
       native_final_operator,
       convert_banded,
       convert_state,
       crop_matrix,
       state_difference,
       state_add,
       state_axpy!,
       matrix_difference,
       matrix_add,
       matrix_fro_norm,
       state_fro_norm,
       flatten_state,
       unflatten_state,
       flatten_matrix_full,
       unflatten_matrix_full,
       explicit_T_matrix,
       explicit_P_matrix,
       state_dimension,
       estimate_state_entries,
       estimate_single_state_bytes,
       estimate_peak_bytes,
       default_memory_budget_bytes,
       state_layout_rows,
       random_state,
       coeff_maxabs,
       coeff_tail_maxabs,
       coeff_tail_norm,
       coeff_difference_from_exact_constant,
       max_state_block_norm,
       clenshaw_T_norm_upper_bound,
       normalized_backward_error,
       recommended_internal_M,
       padding_sequence,
       write_csv,
       csv_cell,
       sci,
       unit_roundoff_value,
       estimate_state_dimension

struct KernelSpec
    name::String
    definition::String
    fn::Function
end

struct GOTransferSystem{T}
    kernel_name::String
    kernel_definition::String
    Nout::Int
    M::Int
    n::Int
    depth::Int
    flip::Bool
    coefficient_source::String
    arithmetic_type::String
    coeffs_float64::Any
    cfs_blocks::Vector{Vector{T}}
    Bx::Vector{BandedMatrix{T,Matrix{T}}}
    By::Vector{BandedMatrix{T,Matrix{T}}}
    B::Vector{BandedMatrix{T,Matrix{T}}}
    Ctail::Vector{BandedMatrix{T,Matrix{T}}}
    D::BandedMatrix{T,Matrix{T}}
    subdiag::Vector{T}
    diag::Vector{T}
    supdiag::Vector{T}
    Wlower::BandedMatrix{T,Matrix{T}}
    Reflect::BandedMatrix{T,Matrix{T}}
    C00to01::BandedMatrix{T,Matrix{T}}
end

struct GOClenshawState{T}
    levels::Vector{Vector{BandedMatrix{T,Matrix{T}}}}
end

function unstable_mixed_kernel(x, y)
    return cos(20.0 * Float64(x) * Float64(y))
end

function stable_constant_kernel(x, y)
    return 1.0
end

function default_kernel_specs()
    return (
        KernelSpec("mixed_cos20xy",
                   "K(x,y) = cos(20*x*y), with GO flip=true sampling K(1-x,y)",
                   unstable_mixed_kernel),
        KernelSpec("constant_one",
                   "K(x,y) = 1, separable stable control, with GO flip=true",
                   stable_constant_kernel),
    )
end

function make_coeffs(spec::KernelSpec, depth::Integer; flip::Bool=true)
    plan = GutlebOlver.precompute_triangular_kernel_plan(Int(depth), flip)
    return GutlebOlver.triangular_kernel_coeffs_from_plan(plan, spec.fn)
end

function blocked_coeffs_from_vector(values::AbstractVector, depth::Integer)
    blocked_type = GutlebOlver._triangular_kernel_blocked_array_type()
    return blocked_type(values, 1:Int(depth))
end

function make_exact_constant_coeffs(depth::Integer; flip::Bool=true)
    d = Int(depth)
    plan = GutlebOlver.precompute_triangular_kernel_plan(d, flip)
    values = zeros(Float64, sum(1:d))
    values[1] = 1.0
    cfs = blocked_coeffs_from_vector(values, d)
    return GutlebOlver.TriangularKernelCoefficients(
        cfs,
        plan.recurrence,
        d,
        d,
        flip,
    )
end

function make_controlled_tail_constant_coeffs(depth::Integer, amplitude::Real;
                                              flip::Bool=true)
    d = Int(depth)
    plan = GutlebOlver.precompute_triangular_kernel_plan(d, flip)
    values = zeros(Float64, sum(1:d))
    values[1] = 1.0
    for idx = 2:length(values)
        sign = isodd(idx) ? -1.0 : 1.0
        values[idx] = Float64(amplitude) * sign / (1.0 + mod(idx, 7))
    end
    cfs = blocked_coeffs_from_vector(values, d)
    return GutlebOlver.TriangularKernelCoefficients(
        cfs,
        plan.recurrence,
        d,
        d,
        flip,
    )
end

function bandwidths_int(A)
    l, u = BandedMatrices.bandwidths(A)
    return max(Int(l), 0), max(Int(u), 0)
end

function banded_clamp(n::Integer, b::Integer)
    return min(max(Int(b), 0), max(Int(n) - 1, 0))
end

@inline function inband(i::Integer, j::Integer, l::Integer, u::Integer)
    return i - j <= l && j - i <= u
end

function banded_zero(::Type{T}, n::Integer, lower_band::Integer,
                     upper_band::Integer) where {T}
    l = banded_clamp(n, lower_band)
    u = banded_clamp(n, upper_band)
    A = BandedMatrix{T}(undef, (Int(n), Int(n)), (l, u))
    fill!(getfield(A, :data), zero(T))
    return A
end

function banded_zero_rect(::Type{T}, rows::Integer, cols::Integer,
                          lower_band::Integer, upper_band::Integer) where {T}
    r = Int(rows)
    c = Int(cols)
    l = min(max(Int(lower_band), 0), max(r - 1, 0))
    u = min(max(Int(upper_band), 0), max(c - 1, 0))
    A = BandedMatrix{T}(undef, (r, c), (l, u))
    fill!(getfield(A, :data), zero(T))
    return A
end

function convert_banded(::Type{T}, A) where {T}
    rows, cols = size(A)
    l, u = A isa BandedMatrix ? bandwidths_int(A) : (rows - 1, cols - 1)
    B = BandedMatrix{T}(undef, (rows, cols), (l, u))
    fill!(getfield(B, :data), zero(T))
    @inbounds for j = 1:cols
        ilo = max(1, j - u)
        ihi = min(rows, j + l)
        for i = ilo:ihi
            B[i, j] = T(A[i, j])
        end
    end
    return B
end

function dense_to_banded(::Type{T}, A::AbstractMatrix,
                         lower_band::Integer,
                         upper_band::Integer) where {T}
    rows, cols = size(A)
    B = banded_zero_rect(T, rows, cols, lower_band, upper_band)
    l, u = bandwidths_int(B)
    scale = zero(T)
    @inbounds for v in A
        scale = max(scale, abs(T(v)))
    end
    ztol = scale * eps(T) * T(4096)
    @inbounds for j = 1:cols
        ilo = max(1, j - u)
        ihi = min(rows, j + l)
        for i = ilo:ihi
            v = T(A[i, j])
            B[i, j] = abs(v) <= ztol ? zero(T) : v
        end
    end
    return B
end

function q_e_y_diag(::Type{T}, n::Integer) where {T}
    D = banded_zero(T, n, 0, 0)
    @inbounds for k = 1:Int(n)
        D[k, k] = isodd(k + 1) ? -inv(T(k)) : inv(T(k))
    end
    return D
end

function convert_vector(::Type{T}, v) where {T}
    return [T(x) for x in collect(v)]
end

function build_transfer_system_explicit(coeffs; depth::Integer=coeffs.kerneldepth,
                                        Nout::Integer,
                                        M::Integer,
                                        arithmetic_type::Type{T},
                                        kernel_name::AbstractString,
                                        kernel_definition::AbstractString,
                                        coefficient_source::AbstractString,
                                        flip::Bool=coeffs.flip) where {T}
    n = Int(M)
    n >= Int(Nout) || throw(ArgumentError("M must be at least Nout"))
    d = Int(coeffs.kerneldepth)
    d == Int(depth) ||
        throw(ArgumentError("depth=$depth does not match coefficient depth $d"))
    recurrence = coeffs.recurrence

    cfs_blocks = Vector{Vector{T}}(undef, d)
    for k = 1:d
        cfs_blocks[k] = convert_vector(T, view(coeffs.cfs, Block(k)))
    end

    Bx_src = getfield(recurrence, 2)
    By_src = getfield(recurrence, 3)
    B_src = getfield(recurrence, 4)
    C_src = getfield(recurrence, 5)

    Bx = [convert_banded(T, Bx_src[k]) for k = 1:max(d - 1, 0)]
    By = [convert_banded(T, By_src[k]) for k = 1:max(d - 1, 0)]
    B = [convert_banded(T, B_src[k]) for k = 1:max(d - 1, 0)]
    Ctail = [convert_banded(T, C_src[k]) for k = 1:max(d - 2, 0)]

    D = q_e_y_diag(T, n)
    sub_f64, diag_f64, sup_f64 = GutlebOlver.jacobi_xmat_P01_tridiag(n)
    subdiag = convert_vector(T, sub_f64)
    diag = convert_vector(T, diag_f64)
    supdiag = convert_vector(T, sup_f64)

    Wlower = convert_banded(T, GutlebOlver.WLoweringP01P00(n))
    Reflect = convert_banded(T, GutlebOlver.reflectPabtoPba(n))
    C00to01 = convert_banded(
        T,
        Conversion(Jacobi(0, 0, 0..1), Jacobi(0, 1, 0..1))[1:n, 1:n],
    )

    return GOTransferSystem{T}(
        String(kernel_name),
        String(kernel_definition),
        Int(Nout),
        n,
        n,
        d,
        Bool(flip),
        String(coefficient_source),
        string(T),
        coeffs,
        cfs_blocks,
        Bx,
        By,
        B,
        Ctail,
        D,
        subdiag,
        diag,
        supdiag,
        Wlower,
        Reflect,
        C00to01,
    )
end

function analytic_jacobi_xmat_P01_tridiag(::Type{T}, n::Integer) where {T}
    m = Int(n)
    m >= 1 || throw(ArgumentError("n must be positive"))
    subdiag = Vector{T}(undef, max(m - 1, 0))
    diag = Vector{T}(undef, m)
    supdiag = Vector{T}(undef, max(m - 1, 0))
    @inbounds for i = 1:m
        ii = T(i)
        diag[i] = inv(T(2)) - inv(T(2) * (T(2) * ii - one(T)) *
                                  (T(2) * ii + one(T)))
        if i < m
            subdiag[i] = (ii + one(T)) / (T(2) * (T(2) * ii + one(T)))
            supdiag[i] = ii / (T(2) * (T(2) * ii + one(T)))
        end
    end
    return subdiag, diag, supdiag
end

function _abc(::Type{T}, a, b, c) where {T}
    return T(a), T(b), T(c), T(a) + T(b) + T(c), T(b) + T(c)
end

function _tri_Lx_block(::Type{T}, a, b, c, row_block::Integer,
                       col_block::Integer) where {T}
    r = Int(row_block)
    n = Int(col_block)
    A = zeros(T, r, n)
    aa, bb, cc, abc, _ = _abc(T, a, b, c)
    den = T(2) * T(n) + abc
    if r == n
        @inbounds for k = 1:n
            A[k, k] = (T(n) - T(k) + aa) / den
        end
    elseif r == n + 1
        @inbounds for k = 1:n
            A[k, k] = (T(n) - T(k) + one(T)) / den
        end
    end
    return A
end

function _tri_Rx_block(::Type{T}, a, b, c, row_block::Integer,
                       col_block::Integer) where {T}
    r = Int(row_block)
    n = Int(col_block)
    A = zeros(T, r, n)
    aa, bb, cc, abc, bc = _abc(T, a, b, c)
    den = T(2) * T(n) + abc
    if r == n
        @inbounds for k = 1:n
            A[k, k] = (T(n) + T(k) + abc) / den
        end
    elseif r == n - 1
        @inbounds for k = 1:r
            A[k, k] = (T(n) + T(k) + bc - one(T)) / den
        end
    end
    return A
end

function _tri_Ly_block(::Type{T}, a, b, c, row_block::Integer,
                       col_block::Integer) where {T}
    r = Int(row_block)
    n = Int(col_block)
    A = zeros(T, r, n)
    aa, bb, cc, abc, bc = _abc(T, a, b, c)
    den_n = T(2) * T(n) + abc
    if r == n
        @inbounds for k = 1:n
            den_k = T(2) * T(k) + bc - one(T)
            A[k, k] = ((T(k) + bb - one(T)) *
                       (T(n) + T(k) + bc - one(T))) / (den_n * den_k)
            if k < n
                A[k + 1, k] = (T(k) * (T(k) - T(n) - aa)) /
                              (den_n * den_k)
            end
        end
    elseif r == n + 1
        @inbounds for k = 1:n
            den_k = T(2) * T(k) + bc - one(T)
            A[k, k] = ((T(k) + bb - one(T)) *
                       (T(k) - T(n) - one(T))) / (den_n * den_k)
            A[k + 1, k] = (T(k) * (T(n) + T(k) + abc)) /
                          (den_n * den_k)
        end
    end
    return A
end

function _tri_Ry_block(::Type{T}, a, b, c, row_block::Integer,
                       col_block::Integer) where {T}
    r = Int(row_block)
    n = Int(col_block)
    A = zeros(T, r, n)
    aa, bb, cc, abc, bc = _abc(T, a, b, c)
    den_n = T(2) * T(n) + abc
    if r == n
        @inbounds for k = 1:n
            den_k = T(2) * T(k) + bc - one(T)
            A[k, k] = ((T(n) + T(k) + abc) * (T(k) + bc)) /
                      (den_n * den_k)
            if k < n
                kp = T(k + 1)
                den_kp = T(2) * kp + bc - one(T)
                A[k, k + 1] = ((kp + cc - one(T)) *
                               (kp - T(n) - one(T))) /
                              (den_n * den_kp)
            end
        end
    elseif r == n - 1
        @inbounds for k = 1:r
            den_k = T(2) * T(k) + bc - one(T)
            A[k, k] = ((T(k) - T(n) - aa) * (T(k) + bc)) /
                      (den_n * den_k)
            kp = T(k + 1)
            den_kp = T(2) * kp + bc - one(T)
            A[k, k + 1] = ((kp + cc - one(T)) *
                           (T(n) + kp + bc - one(T))) /
                          (den_n * den_kp)
        end
    end
    return A
end

function analytic_triangle_jacobi_block(::Type{T}, which::Symbol,
                                        row_block::Integer,
                                        col_block::Integer) where {T}
    r = Int(row_block)
    c = Int(col_block)
    r >= 1 && c >= 1 || throw(ArgumentError("block indices are one-based"))
    out = zeros(T, r, c)
    if which == :x
        # x = Lx(1,0,0) * Rx(0,0,0)
        for m = max(1, min(r, c) - 1):(max(r, c) + 1)
            L = _tri_Lx_block(T, 1, 0, 0, r, m)
            any(!iszero, L) || continue
            R = _tri_Rx_block(T, 0, 0, 0, m, c)
            any(!iszero, R) || continue
            out .+= L * R
        end
    elseif which == :y
        # y = Ly(0,1,0) * Ry(0,0,0)
        for m = max(1, min(r, c) - 1):(max(r, c) + 1)
            L = _tri_Ly_block(T, 0, 1, 0, r, m)
            any(!iszero, L) || continue
            R = _tri_Ry_block(T, 0, 0, 0, m, c)
            any(!iszero, R) || continue
            out .+= L * R
        end
    else
        throw(ArgumentError("which must be :x or :y"))
    end
    return out
end

function analytic_clenshaw_recurrence_blocks(::Type{T},
                                             max_level::Integer) where {T}
    nmax = Int(max_level)
    nmax >= 1 || throw(ArgumentError("max_level must be positive"))
    Bplus_x = Vector{BandedMatrix{T,Matrix{T}}}(undef, nmax)
    Bplus_y = Vector{BandedMatrix{T,Matrix{T}}}(undef, nmax)
    B = Vector{BandedMatrix{T,Matrix{T}}}(undef, nmax)
    Ctail = Vector{BandedMatrix{T,Matrix{T}}}(undef, max(nmax - 1, 0))

    for k = 1:nmax
        Ax = analytic_triangle_jacobi_block(T, :x, k, k)
        Ay = analytic_triangle_jacobi_block(T, :y, k, k)
        Bx_raw = analytic_triangle_jacobi_block(T, :x, k + 1, k)
        By_raw = analytic_triangle_jacobi_block(T, :y, k + 1, k)
        b_last = By_raw[k + 1, k]

        Px = banded_zero_rect(T, k, k + 1, 0, 2)
        Py = banded_zero_rect(T, k, k + 1, 0, 2)
        @inbounds for i = 1:k
            Px[i, i] = inv(Bx_raw[i, i])
        end
        Px[k, k + 1] = -inv(b_last) * By_raw[k, k] / Bx_raw[k, k]
        Py[k, k + 1] = inv(b_last)
        if k > 1
            Px[k - 1, k + 1] =
                -inv(b_last) * By_raw[k - 1, k] / Bx_raw[k - 1, k - 1]
        end

        Bplus_x[k] = Px
        Bplus_y[k] = Py
        B[k] = dense_to_banded(T, Ax * Matrix(Px) + Ay * Matrix(Py), 0, 2)
        if k > 1
            Cx = analytic_triangle_jacobi_block(T, :x, k - 1, k)
            Cy = analytic_triangle_jacobi_block(T, :y, k - 1, k)
            Ctail[k - 1] =
                dense_to_banded(T, Cx * Matrix(Px) + Cy * Matrix(Py), 0, 2)
        end
    end
    return Bplus_x, Bplus_y, B, Ctail
end

function analytic_WLoweringP01P00(::Type{T}, n::Integer) where {T}
    W = banded_zero(T, n, 1, 0)
    half = inv(T(2))
    @inbounds for i = 1:Int(n)
        W[i, i] = half
        i < Int(n) && (W[i + 1, i] = -half)
    end
    return W
end

function analytic_reflectPabtoPba(::Type{T}, n::Integer) where {T}
    R = banded_zero(T, n, 0, 0)
    @inbounds for i = 1:Int(n)
        R[i, i] = isodd(i + 1) ? -one(T) : one(T)
    end
    return R
end

function analytic_conversion_00_to_01(::Type{T}, n::Integer) where {T}
    C = banded_zero(T, n, 0, 1)
    @inbounds for i = 1:Int(n)
        ii = T(i)
        C[i, i] = ii / (T(2) * ii - one(T))
        if i < Int(n)
            C[i, i + 1] = -ii / (T(2) * ii + one(T))
        end
    end
    return C
end

function build_transfer_system_independent(coeffs;
                                           depth::Integer=coeffs.kerneldepth,
                                           Nout::Integer,
                                           M::Integer,
                                           arithmetic_type::Type{T},
                                           kernel_name::AbstractString,
                                           kernel_definition::AbstractString,
                                           coefficient_source::AbstractString,
                                           flip::Bool=coeffs.flip) where {T}
    n = Int(M)
    n >= Int(Nout) || throw(ArgumentError("M must be at least Nout"))
    d = Int(depth)
    d >= 1 || throw(ArgumentError("depth must be positive"))

    cfs_blocks = Vector{Vector{T}}(undef, d)
    for k = 1:d
        cfs_blocks[k] = convert_vector(T, view(coeffs.cfs, Block(k)))
    end

    Bx_all, By_all, B_all, C_all = analytic_clenshaw_recurrence_blocks(T, d + 1)
    Bx = Bx_all[1:max(d - 1, 0)]
    By = By_all[1:max(d - 1, 0)]
    B = B_all[1:max(d - 1, 0)]
    Ctail = C_all[1:max(d - 2, 0)]
    D = q_e_y_diag(T, n)
    subdiag, diag, supdiag = analytic_jacobi_xmat_P01_tridiag(T, n)
    Wlower = analytic_WLoweringP01P00(T, n)
    Reflect = analytic_reflectPabtoPba(T, n)
    C00to01 = analytic_conversion_00_to_01(T, n)

    return GOTransferSystem{T}(
        String(kernel_name),
        String(kernel_definition),
        Int(Nout),
        n,
        n,
        d,
        Bool(flip),
        String(coefficient_source),
        string(T),
        coeffs,
        cfs_blocks,
        Bx,
        By,
        B,
        Ctail,
        D,
        subdiag,
        diag,
        supdiag,
        Wlower,
        Reflect,
        C00to01,
    )
end

function lift_transfer_system(::Type{T}, sys::GOTransferSystem) where {T}
    cfs_blocks = [[T(x) for x in block] for block in sys.cfs_blocks]
    return GOTransferSystem{T}(
        sys.kernel_name,
        sys.kernel_definition,
        sys.Nout,
        sys.M,
        sys.n,
        sys.depth,
        sys.flip,
        sys.coefficient_source * "_lifted",
        string(T),
        sys.coeffs_float64,
        cfs_blocks,
        [convert_banded(T, A) for A in sys.Bx],
        [convert_banded(T, A) for A in sys.By],
        [convert_banded(T, A) for A in sys.B],
        [convert_banded(T, A) for A in sys.Ctail],
        convert_banded(T, sys.D),
        [T(x) for x in sys.subdiag],
        [T(x) for x in sys.diag],
        [T(x) for x in sys.supdiag],
        convert_banded(T, sys.Wlower),
        convert_banded(T, sys.Reflect),
        convert_banded(T, sys.C00to01),
    )
end

function build_transfer_system(coeffs, matrix_n::Integer, ::Type{T};
                               kernel_name::AbstractString,
                               kernel_definition::AbstractString,
                               flip::Bool=coeffs.flip,
                               coefficient_source::AbstractString="native_float64_projection") where {T}
    return build_transfer_system_explicit(
        coeffs;
        depth=coeffs.kerneldepth,
        Nout=Int(matrix_n),
        M=Int(matrix_n),
        arithmetic_type=T,
        kernel_name=kernel_name,
        kernel_definition=kernel_definition,
        coefficient_source=coefficient_source,
        flip=flip,
    )
end

function level_bandwidth(sys::GOTransferSystem, level::Integer)
    return banded_clamp(sys.n, sys.depth - Int(level))
end

function copy_banded(A::BandedMatrix{T,Matrix{T}}) where {T}
    l, u = bandwidths_int(A)
    B = banded_zero(T, size(A, 1), l, u)
    @inbounds for j = 1:size(A, 2)
        ilo = max(1, j - u)
        ihi = min(size(A, 1), j + l)
        for i = ilo:ihi
            B[i, j] = A[i, j]
        end
    end
    return B
end

function copy_state(state::GOClenshawState{T}) where {T}
    return GOClenshawState{T}([[copy_banded(A) for A in level]
                               for level in state.levels])
end

function zero_level(sys::GOTransferSystem{T}, level::Integer) where {T}
    k = Int(level)
    bw = level_bandwidth(sys, k)
    return [banded_zero(T, sys.n, bw, bw) for _ = 1:k]
end

function zero_state(sys::GOTransferSystem{T}) where {T}
    return GOClenshawState{T}([zero_level(sys, k) for k = 1:sys.depth])
end

function axpy_banded!(Y::BandedMatrix{T,Matrix{T}}, alpha, X;
                      check_band::Bool=true) where {T}
    a = T(alpha)
    a == zero(T) && return Y
    size(Y) == size(X) || throw(DimensionMismatch("banded axpy sizes do not match"))
    lY, uY = bandwidths_int(Y)
    lX, uX = bandwidths_int(X)
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
            Y[i, j] += a * T(X[i, j])
        end
    end
    return Y
end

function scale_banded(alpha, X::BandedMatrix{T,Matrix{T}},
                      target_l::Integer, target_u::Integer) where {T}
    Y = banded_zero(T, size(X, 1), target_l, target_u)
    axpy_banded!(Y, T(alpha), X)
    return Y
end

function state_axpy!(Y::GOClenshawState{T}, alpha, X::GOClenshawState;
                     check_band::Bool=true) where {T}
    length(Y.levels) == length(X.levels) || throw(DimensionMismatch("state levels"))
    for k = 1:length(Y.levels)
        length(Y.levels[k]) == length(X.levels[k]) ||
            throw(DimensionMismatch("state level block count"))
        for j = 1:length(Y.levels[k])
            axpy_banded!(Y.levels[k][j], alpha, X.levels[k][j];
                         check_band=check_band)
        end
    end
    return Y
end

function state_add(X::GOClenshawState{T}, Y::GOClenshawState) where {T}
    Z = copy_state(X)
    state_axpy!(Z, one(T), Y)
    return Z
end

function state_difference(X::GOClenshawState{T}, Y::GOClenshawState) where {T}
    Z = copy_state(X)
    state_axpy!(Z, -one(T), Y)
    return Z
end

function state_fro_norm(state::GOClenshawState{T}) where {T}
    acc_type = T <: BigFloat ? BigFloat : Float64
    s = zero(acc_type)
    for level in state.levels
        for A in level
            l, u = bandwidths_int(A)
            @inbounds for j = 1:size(A, 2)
                ilo = max(1, j - u)
                ihi = min(size(A, 1), j + l)
                for i = ilo:ihi
                    v = A[i, j]
                    s += abs2(v)
                end
            end
        end
    end
    return sqrt(s)
end

function matrix_fro_norm(A)
    T = eltype(A)
    acc_type = T <: BigFloat ? BigFloat : Float64
    s = zero(acc_type)
    rows, cols = size(A)
    if A isa BandedMatrix
        l, u = bandwidths_int(A)
        @inbounds for j = 1:cols
            ilo = max(1, j - u)
            ihi = min(rows, j + l)
            for i = ilo:ihi
                s += abs2(A[i, j])
            end
        end
    else
        @inbounds for j = 1:cols
            for i = 1:rows
                s += abs2(A[i, j])
            end
        end
    end
    return sqrt(s)
end

function max_state_block_norm(state::GOClenshawState)
    best = 0.0
    for level in state.levels
        for A in level
            best = max(best, Float64(matrix_fro_norm(A)))
        end
    end
    return best
end

function initial_state(sys::GOTransferSystem{T}, level::Integer,
                       target_l::Integer, target_u::Integer) where {T}
    k = Int(level)
    coeffs = sys.cfs_blocks[k]
    length(coeffs) == k ||
        throw(DimensionMismatch("coefficient block length does not match level"))
    return [scale_banded(coeffs[j], sys.D, target_l, target_u) for j = 1:k]
end

function rhs_state(sys::GOTransferSystem{T}) where {T}
    levels = Vector{Vector{BandedMatrix{T,Matrix{T}}}}(undef, sys.depth)
    for k = 1:sys.depth
        bw = level_bandwidth(sys, k)
        levels[k] = initial_state(sys, k, bw, bw)
    end
    return GOClenshawState{T}(levels)
end

function solve_T_rhs(sys::GOTransferSystem{T}) where {T}
    d = sys.depth
    levels = Vector{Vector{BandedMatrix{T,Matrix{T}}}}(undef, d)
    levels[d] = initial_state(sys, d, level_bandwidth(sys, d),
                              level_bandwidth(sys, d))
    for k = (d - 1):-1:1
        bw = level_bandwidth(sys, k)
        current = initial_state(sys, k, bw, bw)
        next_state = levels[k + 1]
        next_next_state = k + 2 <= d ? levels[k + 2] : nothing
        add_forward_couplings!(sys, current, k, next_state, next_next_state)
        levels[k] = current
    end
    return GOClenshawState{T}(levels)
end

function mat_state_mul(M::BandedMatrix{T,Matrix{T}}, states,
                       target_l::Integer, target_u::Integer) where {T}
    rows, cols = size(M)
    length(states) == cols ||
        throw(DimensionMismatch("small recurrence matrix and state length do not match"))
    isempty(states) && return BandedMatrix{T,Matrix{T}}[]
    n = size(first(states), 1)
    out = [banded_zero(T, n, target_l, target_u) for _ = 1:rows]
    lM, uM = bandwidths_int(M)
    @inbounds for i = 1:rows
        jlo = max(1, i - lM)
        jhi = min(cols, i + uM)
        for j = jlo:jhi
            coeff = M[i, j]
            coeff == zero(T) && continue
            axpy_banded!(out[i], coeff, states[j])
        end
    end
    return out
end

function leftmul_tridiag_banded(sub, diag, sup, A::BandedMatrix{T,Matrix{T}},
                                target_l::Integer, target_u::Integer) where {T}
    n = size(A, 1)
    lA, uA = bandwidths_int(A)
    Y = banded_zero(T, n, target_l, target_u)
    lY, uY = bandwidths_int(Y)
    @inbounds for j = 1:n
        ilo = max(1, j - uY)
        ihi = min(n, j + lY)
        for i = ilo:ihi
            val = zero(T)
            if i > 1 && inband(i - 1, j, lA, uA)
                val += sub[i - 1] * A[i - 1, j]
            end
            if inband(i, j, lA, uA)
                val += diag[i] * A[i, j]
            end
            if i < n && inband(i + 1, j, lA, uA)
                val += sup[i] * A[i + 1, j]
            end
            Y[i, j] = val
        end
    end
    return Y
end

function rightmul_tridiag_banded(A::BandedMatrix{T,Matrix{T}}, sub, diag, sup,
                                 target_l::Integer, target_u::Integer) where {T}
    n = size(A, 1)
    lA, uA = bandwidths_int(A)
    Y = banded_zero(T, n, target_l, target_u)
    lY, uY = bandwidths_int(Y)
    @inbounds for j = 1:n
        ilo = max(1, j - uY)
        ihi = min(n, j + lY)
        for i = ilo:ihi
            val = zero(T)
            if j > 1 && inband(i, j - 1, lA, uA)
                val += sup[j - 1] * A[i, j - 1]
            end
            if inband(i, j, lA, uA)
                val += diag[j] * A[i, j]
            end
            if j < n && inband(i, j + 1, lA, uA)
                val += sub[j] * A[i, j + 1]
            end
            Y[i, j] = val
        end
    end
    return Y
end

function abs_banded(A::BandedMatrix{T,Matrix{T}}) where {T}
    l, u = bandwidths_int(A)
    B = banded_zero(T, size(A, 1), l, u)
    @inbounds for j = 1:size(A, 2)
        ilo = max(1, j - u)
        ihi = min(size(A, 1), j + l)
        for i = ilo:ihi
            B[i, j] = abs(A[i, j])
        end
    end
    return B
end

function abs_level(level)
    return [abs_banded(A) for A in level]
end

function constant_banded(::Type{T}, n::Integer, lower_band::Integer,
                         upper_band::Integer, value) where {T}
    A = banded_zero(T, n, lower_band, upper_band)
    l, u = bandwidths_int(A)
    v = T(value)
    @inbounds for j = 1:Int(n)
        ilo = max(1, j - u)
        ihi = min(Int(n), j + l)
        for i = ilo:ihi
            A[i, j] = v
        end
    end
    return A
end

function constant_level(sys::GOTransferSystem{T}, level::Integer, value) where {T}
    bw = level_bandwidth(sys, level)
    return [constant_banded(T, sys.n, bw, bw, value) for _ = 1:Int(level)]
end

function mat_state_mul_abs(M::BandedMatrix{T,Matrix{T}}, states,
                           target_l::Integer, target_u::Integer) where {T}
    rows, cols = size(M)
    isempty(states) && return BandedMatrix{T,Matrix{T}}[]
    n = size(first(states), 1)
    out = [banded_zero(T, n, target_l, target_u) for _ = 1:rows]
    lM, uM = bandwidths_int(M)
    @inbounds for i = 1:rows
        jlo = max(1, i - lM)
        jhi = min(cols, i + uM)
        for j = jlo:jhi
            coeff = abs(M[i, j])
            coeff == zero(T) && continue
            axpy_banded!(out[i], coeff, states[j])
        end
    end
    return out
end

function leftmul_tridiag_banded_abs(sub, diag, sup,
                                    A::BandedMatrix{T,Matrix{T}},
                                    target_l::Integer,
                                    target_u::Integer) where {T}
    n = size(A, 1)
    lA, uA = bandwidths_int(A)
    Y = banded_zero(T, n, target_l, target_u)
    lY, uY = bandwidths_int(Y)
    @inbounds for j = 1:n
        ilo = max(1, j - uY)
        ihi = min(n, j + lY)
        for i = ilo:ihi
            val = zero(T)
            if i > 1 && inband(i - 1, j, lA, uA)
                val += abs(sub[i - 1]) * abs(A[i - 1, j])
            end
            if inband(i, j, lA, uA)
                val += abs(diag[i]) * abs(A[i, j])
            end
            if i < n && inband(i + 1, j, lA, uA)
                val += abs(sup[i]) * abs(A[i + 1, j])
            end
            Y[i, j] = val
        end
    end
    return Y
end

function rightmul_tridiag_banded_abs(A::BandedMatrix{T,Matrix{T}}, sub,
                                     diag, sup, target_l::Integer,
                                     target_u::Integer) where {T}
    n = size(A, 1)
    lA, uA = bandwidths_int(A)
    Y = banded_zero(T, n, target_l, target_u)
    lY, uY = bandwidths_int(Y)
    @inbounds for j = 1:n
        ilo = max(1, j - uY)
        ihi = min(n, j + lY)
        for i = ilo:ihi
            val = zero(T)
            if j > 1 && inband(i, j - 1, lA, uA)
                val += abs(sup[j - 1]) * abs(A[i, j - 1])
            end
            if inband(i, j, lA, uA)
                val += abs(diag[j]) * abs(A[i, j])
            end
            if j < n && inband(i, j + 1, lA, uA)
                val += abs(sub[j]) * abs(A[i, j + 1])
            end
            Y[i, j] = val
        end
    end
    return Y
end

function add_level!(state_level, terms, alpha)
    length(state_level) == length(terms) ||
        throw(DimensionMismatch("state level lengths do not match"))
    for i = 1:length(state_level)
        axpy_banded!(state_level[i], alpha, terms[i])
    end
    return state_level
end

function add_forward_couplings!(sys::GOTransferSystem{T}, state_level,
                                level::Integer, next_state,
                                next_next_state) where {T}
    k = Int(level)
    raw_target = max(sys.depth - k, 0)
    target = banded_clamp(sys.n, raw_target)
    pretarget = banded_clamp(sys.n, max(raw_target - 1, 0))

    xstate = mat_state_mul(sys.Bx[k], next_state, pretarget, pretarget)
    xterms = [leftmul_tridiag_banded(sys.subdiag, sys.diag, sys.supdiag,
                                     xstate[i], target, target)
              for i = 1:k]
    add_level!(state_level, xterms, one(T))

    ystate = mat_state_mul(sys.By[k], next_state, pretarget, pretarget)
    yterms = [rightmul_tridiag_banded(ystate[i], sys.subdiag, sys.diag,
                                      sys.supdiag, target, target)
              for i = 1:k]
    add_level!(state_level, yterms, one(T))

    bterms = mat_state_mul(sys.B[k], next_state, target, target)
    add_level!(state_level, bterms, -one(T))

    if next_next_state !== nothing && k <= length(sys.Ctail)
        cterms = mat_state_mul(sys.Ctail[k], next_next_state, target, target)
        add_level!(state_level, cterms, -one(T))
    end
    return state_level
end

function add_abs_forward_couplings!(sys::GOTransferSystem{T}, state_level,
                                    level::Integer, next_abs_state,
                                    next_next_abs_state) where {T}
    k = Int(level)
    raw_target = max(sys.depth - k, 0)
    target = banded_clamp(sys.n, raw_target)
    pretarget = banded_clamp(sys.n, max(raw_target - 1, 0))

    xstate = mat_state_mul_abs(sys.Bx[k], next_abs_state, pretarget, pretarget)
    xterms = [leftmul_tridiag_banded_abs(sys.subdiag, sys.diag, sys.supdiag,
                                         xstate[i], target, target)
              for i = 1:k]
    add_level!(state_level, xterms, one(T))

    ystate = mat_state_mul_abs(sys.By[k], next_abs_state, pretarget, pretarget)
    yterms = [rightmul_tridiag_banded_abs(ystate[i], sys.subdiag, sys.diag,
                                          sys.supdiag, target, target)
              for i = 1:k]
    add_level!(state_level, yterms, one(T))

    bterms = mat_state_mul_abs(sys.B[k], next_abs_state, target, target)
    add_level!(state_level, bterms, one(T))

    if next_next_abs_state !== nothing && k <= length(sys.Ctail)
        cterms = mat_state_mul_abs(sys.Ctail[k], next_next_abs_state,
                                   target, target)
        add_level!(state_level, cterms, one(T))
    end
    return state_level
end

function level_max_abs(level)
    best = 0.0
    for A in level
        l, u = bandwidths_int(A)
        @inbounds for j = 1:size(A, 2)
            ilo = max(1, j - u)
            ihi = min(size(A, 1), j + l)
            for i = ilo:ihi
                best = max(best, Float64(abs(A[i, j])))
            end
        end
    end
    return best
end

function clenshaw_T_inf_norm_upper_bound(sys::GOTransferSystem{T}) where {T}
    bound = 1.0
    for k = 1:sys.depth
        row_sums = constant_level(sys, k, one(T))
        if k < sys.depth
            next_ones = constant_level(sys, k + 1, one(T))
            next_next_ones = k + 2 <= sys.depth ?
                             constant_level(sys, k + 2, one(T)) : nothing
            add_abs_forward_couplings!(sys, row_sums, k, next_ones,
                                       next_next_ones)
        end
        bound = max(bound, level_max_abs(row_sums))
    end
    return bound
end

function solve_T(sys::GOTransferSystem{T}, rhs::GOClenshawState{T}) where {T}
    d = sys.depth
    levels = Vector{Vector{BandedMatrix{T,Matrix{T}}}}(undef, d)
    levels[d] = [copy_banded(A) for A in rhs.levels[d]]
    for k = (d - 1):-1:1
        current = [copy_banded(A) for A in rhs.levels[k]]
        next_state = levels[k + 1]
        next_next_state = k + 2 <= d ? levels[k + 2] : nothing
        add_forward_couplings!(sys, current, k, next_state, next_next_state)
        levels[k] = current
    end
    return GOClenshawState{T}(levels)
end

function apply_T(sys::GOTransferSystem{T}, z::GOClenshawState{T}) where {T}
    d = sys.depth
    levels = Vector{Vector{BandedMatrix{T,Matrix{T}}}}(undef, d)
    for k = 1:d
        levels[k] = [copy_banded(A) for A in z.levels[k]]
    end
    for k = 1:(d - 1)
        raw_target = max(sys.depth - k, 0)
        target = banded_clamp(sys.n, raw_target)
        pretarget = banded_clamp(sys.n, max(raw_target - 1, 0))
        next_state = z.levels[k + 1]

        xstate = mat_state_mul(sys.Bx[k], next_state, pretarget, pretarget)
        xterms = [leftmul_tridiag_banded(sys.subdiag, sys.diag, sys.supdiag,
                                         xstate[i], target, target)
                  for i = 1:k]
        add_level!(levels[k], xterms, -one(T))

        ystate = mat_state_mul(sys.By[k], next_state, pretarget, pretarget)
        yterms = [rightmul_tridiag_banded(ystate[i], sys.subdiag, sys.diag,
                                          sys.supdiag, target, target)
                  for i = 1:k]
        add_level!(levels[k], yterms, -one(T))

        bterms = mat_state_mul(sys.B[k], next_state, target, target)
        add_level!(levels[k], bterms, one(T))

        if k + 2 <= d && k <= length(sys.Ctail)
            cterms = mat_state_mul(sys.Ctail[k], z.levels[k + 2],
                                   target, target)
            add_level!(levels[k], cterms, one(T))
        end
    end
    return GOClenshawState{T}(levels)
end

function recurrence_residual(sys::GOTransferSystem{T},
                             z::GOClenshawState{T}) where {T}
    return state_difference(apply_T(sys, z), rhs_state(sys))
end

function streaming_residual_stats(sys::GOTransferSystem{T},
                                  z::GOClenshawState{T};
                                  tiny::Real=floatmin(Float64)) where {T}
    length(z.levels) == sys.depth ||
        throw(DimensionMismatch("state depth does not match system depth"))
    norm_T_inf = clenshaw_T_inf_norm_upper_bound(sys)
    max_abs_rho = 0.0
    sum_abs2_rho = 0.0
    max_abs_z = 0.0
    max_abs_b = 0.0
    max_componentwise_ratio = 0.0
    tiny64 = Float64(tiny)

    for k = 1:sys.depth
        bw = level_bandwidth(sys, k)
        rho_level = [copy_banded(A) for A in z.levels[k]]
        denominator_level = abs_level(z.levels[k])
        rhs_level = initial_state(sys, k, bw, bw)
        add_level!(denominator_level, abs_level(rhs_level), one(T))

        if k < sys.depth
            signed_couplings = zero_level(sys, k)
            next_next_state = k + 2 <= sys.depth ? z.levels[k + 2] : nothing
            add_forward_couplings!(sys, signed_couplings, k, z.levels[k + 1],
                                   next_next_state)
            add_level!(rho_level, signed_couplings, -one(T))

            next_abs = abs_level(z.levels[k + 1])
            next_next_abs = k + 2 <= sys.depth ? abs_level(z.levels[k + 2]) :
                            nothing
            add_abs_forward_couplings!(sys, denominator_level, k, next_abs,
                                       next_next_abs)
        end

        add_level!(rho_level, rhs_level, -one(T))

        for block = 1:length(rho_level)
            R = rho_level[block]
            D = denominator_level[block]
            Z = z.levels[k][block]
            B = rhs_level[block]
            l, u = bandwidths_int(R)
            @inbounds for j = 1:size(R, 2)
                ilo = max(1, j - u)
                ihi = min(size(R, 1), j + l)
                for i = ilo:ihi
                    abs_r = Float64(abs(R[i, j]))
                    abs_z = Float64(abs(Z[i, j]))
                    abs_b = Float64(abs(B[i, j]))
                    denom = max(Float64(abs(D[i, j])), tiny64)
                    max_abs_rho = max(max_abs_rho, abs_r)
                    sum_abs2_rho += abs_r * abs_r
                    max_abs_z = max(max_abs_z, abs_z)
                    max_abs_b = max(max_abs_b, abs_b)
                    max_componentwise_ratio =
                        max(max_componentwise_ratio, abs_r / denom)
                end
            end
        end
    end

    beta_inf = max_abs_rho /
               max(norm_T_inf * max_abs_z + max_abs_b, tiny64)
    return (
        norm_rho_inf=max_abs_rho,
        norm_rho_fro=sqrt(sum_abs2_rho),
        norm_zhat_inf=max_abs_z,
        norm_b_inf=max_abs_b,
        norm_T_inf=norm_T_inf,
        norm_T_kind="rigorous_upper_bound",
        beta_inf=beta_inf,
        beta_componentwise=max_componentwise_ratio,
        denominator_tiny=tiny64,
    )
end

function leftmul_banded(Aop::BandedMatrix{T,Matrix{T}},
                        V::BandedMatrix{T,Matrix{T}},
                        target_l::Integer, target_u::Integer) where {T}
    n = size(V, 1)
    lA, uA = bandwidths_int(Aop)
    lV, uV = bandwidths_int(V)
    Y = banded_zero(T, n, target_l, target_u)
    lY, uY = bandwidths_int(Y)
    @inbounds for j = 1:n
        ilo = max(1, j - uY)
        ihi = min(n, j + lY)
        for i = ilo:ihi
            klo = max(1, i - lA, j - uV)
            khi = min(n, i + uA, j + lV)
            val = zero(T)
            for k = klo:khi
                val += Aop[i, k] * V[k, j]
            end
            Y[i, j] = val
        end
    end
    return Y
end

function crop_matrix(A, nout::Integer)
    n = Int(nout)
    rows, cols = size(A)
    n <= rows && n <= cols || throw(DimensionMismatch("crop size exceeds matrix"))
    T = eltype(A)
    B = Matrix{T}(undef, n, n)
    @inbounds for j = 1:n
        for i = 1:n
            B[i, j] = A[i, j]
        end
    end
    return B
end

function apply_P_raw(sys::GOTransferSystem, z::GOClenshawState)
    return crop_matrix(z.levels[1][1], sys.Nout)
end

function apply_P_final(sys::GOTransferSystem{T}, z::GOClenshawState{T}) where {T}
    Vraw = copy_banded(z.levels[1][1])
    lV, uV = bandwidths_int(Vraw)
    tmp = leftmul_banded(sys.Wlower, Vraw, lV + 1, uV)
    tmp2 = leftmul_banded(sys.Reflect, tmp, lV + 1, uV)
    lT, uT = bandwidths_int(tmp2)
    lC, uC = bandwidths_int(sys.C00to01)
    return crop_matrix(leftmul_banded(sys.C00to01, tmp2, lT + lC, uT + uC),
                       sys.Nout)
end

function apply_P_stage(sys::GOTransferSystem{T}, z::GOClenshawState{T},
                       stage::Symbol) where {T}
    Vraw = copy_banded(z.levels[1][1])
    stage == :extraction && return Vraw

    lV, uV = bandwidths_int(Vraw)
    lowered = leftmul_banded(sys.Wlower, Vraw, lV + 1, uV)
    stage == :lowering && return lowered

    reflected = leftmul_banded(sys.Reflect, lowered, lV + 1, uV)
    stage == :reflection && return reflected

    lR, uR = bandwidths_int(reflected)
    lC, uC = bandwidths_int(sys.C00to01)
    converted = leftmul_banded(sys.C00to01, reflected, lR + lC, uR + uC)
    stage == :conversion && return converted
    stage == :crop && return crop_matrix(converted, sys.Nout)
    stage == :full && return crop_matrix(converted, sys.Nout)

    throw(ArgumentError("unknown P stage: $stage"))
end

function apply_P(sys::GOTransferSystem, z::GOClenshawState; output::Symbol=:final)
    output == :raw && return apply_P_raw(sys, z)
    output == :final && return apply_P_final(sys, z)
    throw(ArgumentError("output must be :raw or :final"))
end

function native_raw_operator(sys::GOTransferSystem)
    V = GutlebOlver.triVolterraFullKernelOpP01_strictband_from_coeffs(
        sys.coeffs_float64,
        sys.M,
        sys.flip,
    )
    return crop_matrix(V, sys.Nout)
end

function native_final_operator(sys::GOTransferSystem)
    Vraw = GutlebOlver.triVolterraFullKernelOpP01_strictband_from_coeffs(
        sys.coeffs_float64,
        sys.M,
        sys.flip,
    )
    Vfinal = GutlebOlver.build_secondkind_volterra_strictband(Vraw, sys.M)
    return crop_matrix(Vfinal, sys.Nout)
end

function convert_state(::Type{T}, state::GOClenshawState) where {T}
    levels = Vector{Vector{BandedMatrix{T,Matrix{T}}}}(undef, length(state.levels))
    for k = 1:length(state.levels)
        levels[k] = [convert_banded(T, A) for A in state.levels[k]]
    end
    return GOClenshawState{T}(levels)
end

function matrix_difference(A::BandedMatrix{T,Matrix{T}}, B) where {T}
    B isa BandedMatrix || return matrix_difference(Matrix(A), B)
    lA, uA = bandwidths_int(A)
    lB, uB = bandwidths_int(B)
    C = banded_zero(T, size(A, 1), max(lA, lB), max(uA, uB))
    axpy_banded!(C, one(T), A; check_band=false)
    axpy_banded!(C, -one(T), B; check_band=false)
    return C
end

function matrix_difference(A::BandedMatrix{T,Matrix{T}},
                           B::AbstractMatrix) where {T}
    return matrix_difference(Matrix(A), B)
end

function matrix_difference(A::AbstractMatrix{T}, B::AbstractMatrix) where {T}
    size(A) == size(B) || throw(DimensionMismatch("matrix sizes do not match"))
    C = Matrix{T}(undef, size(A))
    @inbounds for j = 1:size(A, 2)
        for i = 1:size(A, 1)
            C[i, j] = A[i, j] - T(B[i, j])
        end
    end
    return C
end

function matrix_add(A::BandedMatrix{T,Matrix{T}}, B) where {T}
    B isa BandedMatrix || return matrix_add(Matrix(A), B)
    lA, uA = bandwidths_int(A)
    lB, uB = bandwidths_int(B)
    C = banded_zero(T, size(A, 1), max(lA, lB), max(uA, uB))
    axpy_banded!(C, one(T), A; check_band=false)
    axpy_banded!(C, one(T), B; check_band=false)
    return C
end

function matrix_add(A::BandedMatrix{T,Matrix{T}}, B::AbstractMatrix) where {T}
    return matrix_add(Matrix(A), B)
end

function matrix_add(A::AbstractMatrix{T}, B::AbstractMatrix) where {T}
    size(A) == size(B) || throw(DimensionMismatch("matrix sizes do not match"))
    C = Matrix{T}(undef, size(A))
    @inbounds for j = 1:size(A, 2)
        for i = 1:size(A, 1)
            C[i, j] = A[i, j] + T(B[i, j])
        end
    end
    return C
end

function flatten_state(state::GOClenshawState{T}) where {T}
    vals = T[]
    for k = 1:length(state.levels)
        for A in state.levels[k]
            l, u = bandwidths_int(A)
            @inbounds for j = 1:size(A, 2)
                ilo = max(1, j - u)
                ihi = min(size(A, 1), j + l)
                for i = ilo:ihi
                    push!(vals, A[i, j])
                end
            end
        end
    end
    return vals
end

function unflatten_state(sys::GOTransferSystem{T}, vals::AbstractVector) where {T}
    expected = state_dimension(sys)
    length(vals) == expected ||
        throw(DimensionMismatch("state vector length $(length(vals)) != $expected"))
    levels = Vector{Vector{BandedMatrix{T,Matrix{T}}}}(undef, sys.depth)
    cursor = 1
    for k = 1:sys.depth
        bw = level_bandwidth(sys, k)
        blocks = Vector{BandedMatrix{T,Matrix{T}}}(undef, k)
        for block = 1:k
            A = banded_zero(T, sys.n, bw, bw)
            l, u = bandwidths_int(A)
            @inbounds for j = 1:sys.n
                ilo = max(1, j - u)
                ihi = min(sys.n, j + l)
                for i = ilo:ihi
                    A[i, j] = T(vals[cursor])
                    cursor += 1
                end
            end
            blocks[block] = A
        end
        levels[k] = blocks
    end
    return GOClenshawState{T}(levels)
end

function structural_nnz_square(n::Integer, lower_band::Integer, upper_band::Integer)
    total = 0
    l = banded_clamp(n, lower_band)
    u = banded_clamp(n, upper_band)
    for j = 1:Int(n)
        total += max(0, min(Int(n), j + l) - max(1, j - u) + 1)
    end
    return total
end

function state_dimension(sys::GOTransferSystem)
    return estimate_state_dimension(sys.n, sys.depth)
end

function estimate_state_dimension(n::Integer, depth::Integer)
    total = 0
    for k = 1:Int(depth)
        bw = banded_clamp(n, Int(depth) - k)
        total += k * structural_nnz_square(n, bw, bw)
    end
    return total
end

function estimate_state_entries(depth::Integer, M::Integer)
    total = 0
    n = Int(M)
    for k = 1:Int(depth)
        bw = banded_clamp(n, Int(depth) - k)
        total += k * (2 * bw + 1) * n
    end
    return total
end

function scalar_storage_bytes(::Type{T}; bits::Integer=precision(BigFloat)) where {T}
    T === BigFloat && return max(64, 32 + cld(Int(bits), 8))
    return sizeof(T)
end

function estimate_single_state_bytes(depth::Integer, M::Integer, ::Type{T}=Float64;
                                     bits::Integer=precision(BigFloat)) where {T}
    return estimate_state_entries(depth, M) * scalar_storage_bytes(T; bits=bits)
end

function estimate_peak_bytes(depth::Integer, M::Integer, ::Type{T}=Float64;
                             bits::Integer=precision(BigFloat),
                             complete_states::Real=1,
                             work_state_equivalents::Real=0,
                             dense_output_matrices::Integer=0,
                             Nout::Integer=M,
                             extra_bytes::Integer=0) where {T}
    single = estimate_single_state_bytes(depth, M, T; bits=bits)
    output_bytes = Int(dense_output_matrices) * Int(Nout)^2 *
                   scalar_storage_bytes(T; bits=bits)
    return ceil(Int, single * (complete_states + work_state_equivalents)) +
           output_bytes + Int(extra_bytes)
end

function default_memory_budget_bytes()
    return 1024^3
end

function flatten_matrix_full(A)
    T = eltype(A)
    vals = Vector{T}(undef, prod(size(A)))
    cursor = 1
    rows, cols = size(A)
    @inbounds for j = 1:cols
        for i = 1:rows
            vals[cursor] = A[i, j]
            cursor += 1
        end
    end
    return vals
end

function unflatten_matrix_full(::Type{T}, vals::AbstractVector, n::Integer) where {T}
    length(vals) == Int(n) * Int(n) ||
        throw(DimensionMismatch("matrix vector length does not match n^2"))
    A = Matrix{T}(undef, Int(n), Int(n))
    cursor = 1
    @inbounds for j = 1:Int(n)
        for i = 1:Int(n)
            A[i, j] = T(vals[cursor])
            cursor += 1
        end
    end
    return A
end

function explicit_T_matrix(sys::GOTransferSystem{T}) where {T}
    dim = state_dimension(sys)
    Tmat = Matrix{T}(undef, dim, dim)
    for col = 1:dim
        e = zeros(T, dim)
        e[col] = one(T)
        z = unflatten_state(sys, e)
        Tz = apply_T(sys, z)
        Tmat[:, col] = flatten_state(Tz)
    end
    return Tmat
end

function explicit_P_matrix(sys::GOTransferSystem{T}; output::Symbol=:final) where {T}
    dim = state_dimension(sys)
    outdim = sys.Nout * sys.Nout
    Pmat = Matrix{T}(undef, outdim, dim)
    for col = 1:dim
        e = zeros(T, dim)
        e[col] = one(T)
        z = unflatten_state(sys, e)
        Pmat[:, col] = flatten_matrix_full(apply_P(sys, z; output=output))
    end
    return Pmat
end

function random_state(sys::GOTransferSystem{T}; seed::Integer=1234) where {T}
    rng = MersenneTwister(Int(seed))
    vals = [T(randn(rng)) for _ = 1:state_dimension(sys)]
    return unflatten_state(sys, vals)
end

function state_layout_rows(sys::GOTransferSystem)
    rows = NamedTuple[]
    first_index = 1
    for k = 1:sys.depth
        bw = level_bandwidth(sys, k)
        entries_per_block = structural_nnz_square(sys.n, bw, bw)
        level_dim = k * entries_per_block
        push!(rows, (
            level=k,
            block_count=k,
            matrix_rows=sys.n,
            matrix_cols=sys.n,
            lower_bandwidth=bw,
            upper_bandwidth=bw,
            entries_per_block=entries_per_block,
            first_state_index=first_index,
            last_state_index=first_index + level_dim - 1,
        ))
        first_index += level_dim
    end
    return rows
end

function coeff_maxabs(coeffs)
    vals = Float64[]
    for k = 1:Int(coeffs.kerneldepth)
        append!(vals, abs.(Float64.(collect(view(coeffs.cfs, Block(k))))))
    end
    isempty(vals) && return 0.0
    return maximum(vals)
end

function coeff_tail_maxabs(coeffs; tail_blocks::Integer=3)
    d = Int(coeffs.kerneldepth)
    start_block = max(1, d - Int(tail_blocks) + 1)
    mx = 0.0
    for k = start_block:d
        block_vals = abs.(Float64.(collect(view(coeffs.cfs, Block(k)))))
        isempty(block_vals) && continue
        mx = max(mx, maximum(block_vals))
    end
    return mx
end

function coeff_tail_norm(coeffs; start_block::Integer=2)
    d = Int(coeffs.kerneldepth)
    s = 0.0
    for k = max(1, Int(start_block)):d
        for v in collect(view(coeffs.cfs, Block(k)))
            s += abs2(Float64(v))
        end
    end
    return sqrt(s)
end

function coeff_difference_from_exact_constant(coeffs)
    d = Int(coeffs.kerneldepth)
    s = 0.0
    for k = 1:d
        block = collect(view(coeffs.cfs, Block(k)))
        for j = 1:length(block)
            exact = (k == 1 && j == 1) ? 1.0 : 0.0
            s += abs2(Float64(block[j]) - exact)
        end
    end
    return sqrt(s)
end

function jacobi_tridiag_opnorm(sys::GOTransferSystem)
    J = zeros(Float64, sys.M, sys.M)
    for i = 1:sys.M
        J[i, i] = Float64(sys.diag[i])
        if i < sys.M
            J[i + 1, i] = Float64(sys.subdiag[i])
            J[i, i + 1] = Float64(sys.supdiag[i])
        end
    end
    return opnorm(J, 2)
end

function clenshaw_T_norm_upper_bound(sys::GOTransferSystem)
    d = sys.depth
    A = zeros(Float64, d, d)
    for k = 1:d
        A[k, k] = 1.0
    end
    jnorm = jacobi_tridiag_opnorm(sys)
    for k = 1:(d - 1)
        bx = opnorm(Matrix{Float64}(sys.Bx[k]), 2)
        by = opnorm(Matrix{Float64}(sys.By[k]), 2)
        b = opnorm(Matrix{Float64}(sys.B[k]), 2)
        A[k, k + 1] = jnorm * (bx + by) + b
        if k <= length(sys.Ctail)
            A[k, k + 2] = opnorm(Matrix{Float64}(sys.Ctail[k]), 2)
        end
    end
    return opnorm(A, 2)
end

function normalized_backward_error(sys::GOTransferSystem, zhat::GOClenshawState,
                                   rho::GOClenshawState, b::GOClenshawState;
                                   norm_T=nothing,
                                   norm_T_kind::AbstractString="upper_bound")
    rho_norm = state_fro_norm(rho)
    z_norm = state_fro_norm(zhat)
    b_norm = state_fro_norm(b)
    tnorm = norm_T === nothing ? clenshaw_T_norm_upper_bound(sys) : norm_T
    beta = rho_norm / max(tnorm * z_norm + b_norm, eps(Float64))
    return (
        beta=beta,
        norm_T=tnorm,
        norm_T_kind=String(norm_T_kind),
    )
end

function recommended_internal_M(Nout::Integer, depth::Integer; safety::Integer=8)
    return Int(Nout) + Int(depth) + Int(safety)
end

function padding_sequence(Nout::Integer, depth::Integer; safety::Integer=8)
    n = Int(Nout)
    d = Int(depth)
    s = Int(safety)
    return unique((n, n + d, n + d + 4, n + d + s, n + 2d + s))
end

function csv_cell(value)
    if value === missing
        return ""
    elseif value isa BigFloat
        return sci(value; digits=17)
    elseif value isa AbstractFloat
        return @sprintf("%.17e", Float64(value))
    elseif value isa Integer || value isa Bool
        return string(value)
    else
        text = string(value)
        if occursin(",", text) || occursin("\"", text) ||
           occursin("\n", text) || occursin("\r", text)
            return "\"" * replace(text, "\"" => "\"\"") * "\""
        end
        return text
    end
end

function write_csv(path::AbstractString, columns, rows)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(string.(columns), ","))
        for row in rows
            println(io, join((csv_cell(getproperty(row, col)) for col in columns), ","))
        end
    end
    return path
end

function sci(value; digits::Integer=6)
    if value === missing
        return ""
    end
    if value isa BigFloat
        !isfinite(value) && return string(value)
        return string(Float64(value))
    elseif value isa Real
        v = Float64(value)
        !isfinite(v) && return string(v)
        return @sprintf("%.*e", Int(digits), v)
    else
        return string(value)
    end
end

function unit_roundoff_value(::Type{Float32})
    return Float64(eps(Float32) / 2)
end

function unit_roundoff_value(::Type{Float64})
    return eps(Float64) / 2
end

function unit_roundoff_value(::Type{BigFloat}; bits::Integer=precision(BigFloat))
    return BigFloat(2.0)^(-Int(bits))
end

end
