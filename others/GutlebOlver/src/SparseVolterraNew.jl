module SparseVolterraNew

using ApproxFun
using BandedMatrices
using BlockArrays
using BlockBandedMatrices
using LazyArrays
using LinearAlgebra
using MultivariateOrthogonalPolynomials
using SparseArrays

include("clenshaw.jl")
include("triQEgen.jl")
include("triVolterraFullKernel.jl")
include("triVolterraFullKernel_strictband.jl")

export GutlebOlverConfig,
    build_gutleb_operator,
    build_gutleb_operator_from_coefficients,
    build_gutleb_kernel_coefficients,
    operator_basic_check,
    matrix_sparse_stats,
    triQgen,
    triEygenP01,
    triQEygenP01,
    WLoweringP01P00,
    reflectPabtoPba,
    triVolterraFullKernelFunP01,
    triVolterraFullKernelOpP01,
    triVolterraFullKernelOpP01_strictband,
    TriangularKernelCoefficients,
    TriangularKernelPlan,
    precompute_triangular_kernel_coeffs,
    precompute_triangular_kernel_plan,
    sample_triangular_kernel_values,
    project_triangular_kernel_coeffs_from_values,
    triangular_kernel_coeffs_from_plan,
    triVolterraFullKernelOpP01_strictband_from_coeffs,
    theoretical_bandwidth_P01,
    jacobi_xmat_P01_tridiag,
    leftmul_tridiag_banded,
    rightmul_tridiag_banded,
    axpy_banded!,
    lincomb_banded,
    apply_reflect_lowering_strictband,
    apply_conversion_00_to_12_strictband,
    build_VIDE_LHS_sec522_strictband,
    triBandedSgen

Base.@kwdef struct GutlebOlverConfig
    N::Int
    kerneldepth::Int
    flip::Bool = true
end

function build_gutleb_kernel_coefficients(K::Function, kerneldepth::Integer; flip::Bool=true)
    return precompute_triangular_kernel_coeffs(K, kerneldepth, flip)
end

function build_gutleb_operator_from_coefficients(
    coeffs::TriangularKernelCoefficients,
    N::Integer;
    flip::Bool=coeffs.flip,
)
    return triVolterraFullKernelOpP01_strictband_from_coeffs(coeffs, N, flip)
end

function build_gutleb_operator(K::Function, config::GutlebOlverConfig)
    coeffs = build_gutleb_kernel_coefficients(K, config.kerneldepth; flip=config.flip)
    return build_gutleb_operator_from_coefficients(coeffs, config.N; flip=config.flip)
end

function matrix_sparse_stats(A; tol::Real=0.0)
    S = sparse(A)
    rows, cols, vals = findnz(S)
    actual_nnz = 0
    lower_bw = 0
    upper_bw = 0
    for (i, j, v) in zip(rows, cols, vals)
        if abs(v) > tol
            actual_nnz += 1
            lower_bw = max(lower_bw, i - j)
            upper_bw = max(upper_bw, j - i)
        end
    end
    return (nnz=actual_nnz, lower_band=lower_bw, upper_band=upper_bw)
end

function operator_basic_check(A, N::Integer)
    size(A) == (N, N) || return false
    return all(isfinite, A)
end

end
