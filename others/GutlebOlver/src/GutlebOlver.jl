module GutlebOlver

using ApproxFun
using BandedMatrices
using BlockArrays
using BlockBandedMatrices
using LazyArrays
using LinearAlgebra
using MultivariateOrthogonalPolynomials
using Printf
using SparseArrays
using SpecialFunctions

include("clenshaw.jl")
include("operators.jl")
include("triangular_kernel_projection.jl")
include("volterra_strictband.jl")
include("rhs_projection.jl")
include("equation_assembly.jl")
include("diagnostics.jl")
include("workflows.jl")

export
    build_go_rhs_coefficients,
    validate_rhs_projection,
    precompute_triangular_kernel_plan,
    sample_triangular_kernel_values,
    project_triangular_kernel_coeffs_from_values,
    triangular_kernel_coeffs_from_plan,
    triVolterraFullKernelOpP01_strictband_from_coeffs,
    build_secondkind_volterra_strictband,
    build_secondkind_system_strictband,
    build_VIDE_LHS_sec522_strictband,
    triQgen,
    triEygenP01,
    triQEygenP01,
    WLoweringP01P00,
    reflectPabtoPba,
    apply_reflect_lowering_strictband,
    apply_conversion_00_to_01_strictband,
    apply_conversion_00_to_12_strictband,
    build_derivative_P01_to_P12,
    build_initial_evaluation_row,
    assemble_semibanded_system,
    assemble_section522_semibanded_system,
    build_secondkind_equation_system,
    build_simple_secondkind_system,
    solve_simple_secondkind_sparse,
    build_section522_operator,
    build_section522_rhs,
    solve_section522_k100_sparse,
    safe_nnz,
    safe_bandwidth,
    safe_density,
    assert_no_dense_matrix,
    operator_summary,
    result_summary,
    write_results_csv,
    append_result_csv,
    print_section522_result,
    print_section522_summary,
    SECTION522_K,
    SECTION522_ACCEPTANCE_N,
    SECTION522_ACCEPTANCE_KERNELDEPTH

end
