# GutlebOlver

Standalone Julia 1.11 sparse/strict-band Gutleb--Olver module for reusable Volterra and VIDE workflows.

## Structure

```text
src/GutlebOlver.jl
src/rhs_projection.jl
src/triangular_kernel_projection.jl
src/clenshaw.jl
src/operators.jl
src/volterra_strictband.jl
src/equation_assembly.jl
src/diagnostics.jl
src/workflows.jl
examples/solve_simple_secondkind_sparse.jl
examples/solve_section522_k100_sparse.jl
test/runtests.jl
test/test_rhs_projection.jl
test/test_triangular_kernel_projection.jl
test/test_operators.jl
test/test_simple_secondkind_sparse.jl
test/test_section522_k100_sparse.jl
test/test_dense_path_scan.jl
test/golden_section522_k100.toml
```

`src/GutlebOlver.jl` is only the package entry point: it loads dependencies, includes submodules, and exports stable public API.

## Public API

```julia
using GutlebOlver

rhs, info = build_go_rhs_coefficients(x -> sin(x), 32; alpha=0, beta=1)
plan = precompute_triangular_kernel_plan(12, true)
coeffs = triangular_kernel_coeffs_from_plan(plan, (x, y) -> y * exp(x^2))
Vraw = triVolterraFullKernelOpP01_strictband_from_coeffs(coeffs, 128, true)

simple = solve_simple_secondkind_sparse(N=32)
result = solve_section522_k100_sparse(N=128, kerneldepth=12)
```

Stable exports include RHS projection, triangular kernel projection, strict-band Volterra construction, basic operators, derivative/evaluation-row builders, system assembly, diagnostics, workflows, and constants:

```text
build_go_rhs_coefficients, validate_rhs_projection
precompute_triangular_kernel_plan, sample_triangular_kernel_values
project_triangular_kernel_coeffs_from_values, triangular_kernel_coeffs_from_plan
triVolterraFullKernelOpP01_strictband_from_coeffs
build_secondkind_volterra_strictband, build_secondkind_system_strictband
build_VIDE_LHS_sec522_strictband
triQgen, triEygenP01, triQEygenP01, WLoweringP01P00, reflectPabtoPba
apply_reflect_lowering_strictband
apply_conversion_00_to_01_strictband, apply_conversion_00_to_12_strictband
build_derivative_P01_to_P12, build_initial_evaluation_row
assemble_semibanded_system, assemble_section522_semibanded_system
build_simple_secondkind_system, solve_simple_secondkind_sparse
build_section522_operator, build_section522_rhs, solve_section522_k100_sparse
safe_nnz, safe_bandwidth, safe_density, assert_no_dense_matrix
operator_summary, result_summary, write_results_csv, append_result_csv
SECTION522_K, SECTION522_ACCEPTANCE_N, SECTION522_ACCEPTANCE_KERNELDEPTH
```

## Workflow

```text
RHS:
f(x) -> shifted Jacobi projection -> rhs_coeff

Kernel:
K(x,y) -> triangular sampling -> triangle coefficients -> Vraw

Volterra:
Vraw -> WLowering -> reflection -> conversion -> Vphys

Second-kind:
A = I + Vphys

VIDE:
A = [u(0) row; D - converted Vphys body rows]

Solve:
A \ rhs -> coefficients -> physical evaluation -> error/residual
```

## Dense Path Policy

Formal sparse workflows do not use dense operator materialization. Diagnostics scan banded storage or `SparseMatrixCSC` internals. The test suite includes a dense-path scan over `src`, `examples`, and `test` for direct `Matrix(`, `Array(`, or `collect(` calls.

## Run

From this directory:

```powershell
julia --project=. -e "using Pkg; Pkg.test()"
julia --project=. examples/solve_simple_secondkind_sparse.jl
julia --project=. examples/solve_section522_k100_sparse.jl
```

From the VoltOpt root:

```powershell
D:\Julia\Julia-1.11.1\bin\julia.exe --project=others/GutlebOlver -e "using Pkg; Pkg.test()"
```

## Golden Result

`test/golden_section522_k100.toml` is the protected Section 5.2.2 regression target:

```text
N = 128
kerneldepth = 12
k = 100
max_abs_error = 9.404194067919036e-8
body_residual_relative = 1.1167327725282074e-7
final_system_nnz = 3388
final_system_bandwidth = (13, 127)
```

`kerneldepth = 12` remains the conservative standalone golden setting. The current varying-depth benchmark shows `kerneldepth = 10` is the smallest passing value in that sweep, while `4`, `6`, and `8` complete with `status = check`.
