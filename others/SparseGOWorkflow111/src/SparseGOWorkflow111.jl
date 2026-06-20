module SparseGOWorkflow111

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
include("triQEgen.jl")
include("triVolterraFullKernel_strictband.jl")
include("workflows.jl")

export
    build_simple_secondkind_system,
    solve_simple_secondkind_sparse,
    build_section522_operator,
    build_section522_rhs,
    solve_section522_k100_sparse,
    safe_bandwidth,
    safe_nnz,
    safe_density,
    write_results_csv,
    append_result_csv,
    print_section522_result,
    print_section522_summary,
    SECTION522_K,
    SECTION522_ACCEPTANCE_N,
    SECTION522_ACCEPTANCE_KERNELDEPTH

end
