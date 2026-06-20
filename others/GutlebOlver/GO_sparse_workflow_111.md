# Gutleb--Olver Sparse Workflow

This project packages the strict-band Gutleb--Olver sparse Volterra workflow as `GutlebOlver`. The refactor is modular only: the current strict-band / semi-banded numerical path and the Section 5.2.2 golden result are preserved.

## Module Structure

```text
Project.toml
Manifest.toml
README.md
GO_sparse_workflow_111.md
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

`src/GutlebOlver.jl` contains no implementation logic beyond dependency loading, includes, and exports.

## Public API

RHS projection:

```text
build_go_rhs_coefficients
validate_rhs_projection
```

Triangular kernel projection:

```text
precompute_triangular_kernel_plan
sample_triangular_kernel_values
project_triangular_kernel_coeffs_from_values
triangular_kernel_coeffs_from_plan
```

Strict-band Volterra construction:

```text
triVolterraFullKernelOpP01_strictband_from_coeffs
build_secondkind_volterra_strictband
build_secondkind_system_strictband
build_VIDE_LHS_sec522_strictband
```

Basic operators:

```text
triQgen
triEygenP01
triQEygenP01
WLoweringP01P00
reflectPabtoPba
apply_reflect_lowering_strictband
apply_conversion_00_to_01_strictband
apply_conversion_00_to_12_strictband
build_derivative_P01_to_P12
build_initial_evaluation_row
```

Equation assembly and workflows:

```text
assemble_semibanded_system
assemble_section522_semibanded_system
build_simple_secondkind_system
solve_simple_secondkind_sparse
build_section522_operator
build_section522_rhs
solve_section522_k100_sparse
```

Diagnostics and output:

```text
safe_nnz
safe_bandwidth
safe_density
assert_no_dense_matrix
operator_summary
result_summary
write_results_csv
append_result_csv
```

Constants:

```text
SECTION522_K
SECTION522_ACCEPTANCE_N
SECTION522_ACCEPTANCE_KERNELDEPTH
```

## RHS Projection

`rhs_projection.jl` provides fixed-length shifted Jacobi RHS coefficient construction:

```julia
rhs, info = build_go_rhs_coefficients(f, N; alpha, beta, method=:fixed_gauss_jacobi)
```

The returned coefficient vector has length `N`. `info` records:

```text
method
basis
coefficient_length
tail_norm
validation_error
status
notes
```

The default path is the fixed-`N` ApproxFun/Jacobi coefficient path already used by the golden workflow. `method=:exact_polynomial` accepts monomial coefficients for exact manufactured RHS tests. `method=:adaptive_approxfun` is kept as a verification mode, not as the formal sparse benchmark path.

## Triangular Kernel Projection

`triangular_kernel_projection.jl` owns:

```text
TriangularKernelPlan
TriangularKernelCoefficients
precompute_triangular_kernel_plan
sample_triangular_kernel_values
project_triangular_kernel_coeffs_from_values
triangular_kernel_coeffs_from_plan
```

The plan stores the `JacobiTriangle` basis block, transform grid, Clenshaw recurrence data, requested `kerneldepth`, realized coefficient depth, and `flip`. With `flip=true`, samples use `K(1-X,Y)`; otherwise samples use `K(X,Y)`. Projection returns triangular coefficients used directly by the strict-band Volterra builder and never forms a dense Volterra operator.

## Operators

`operators.jl` contains the low-level sparse/banded operators:

- `triQgen`: triangle integration `Q_y`;
- `triEygenP01`: extension `E_y` from `Jacobi(0,1,0..1)` to the triangle basis;
- `triQEygenP01`: diagonal `Q_y E_y`;
- `WLoweringP01P00`: weighted lowering that absorbs `(1-x)` and maps `Jacobi(0,1)` to `Jacobi(0,0)`;
- `reflectPabtoPba`: diagonal reflection from `P^(a,b)(1-x)` coefficients to `P^(b,a)(x)` coefficients;
- `apply_conversion_00_to_01_strictband`: second-kind conversion chain;
- `apply_conversion_00_to_12_strictband`: first-derivative VIDE conversion chain;
- `build_derivative_P01_to_P12`: derivative `Jacobi(0,1) -> Jacobi(1,2)`;
- `build_initial_evaluation_row`: coefficient row for `u(0)`.

Docstrings on these functions record the basis mapping.

## Strict-Band Volterra

`volterra_strictband.jl` owns the operator-valued Clenshaw recurrence and strict-band multiplication kernels. For a triangular coefficient object, it builds:

```text
Vraw = Q_y K(1-X,Y) E_y
```

The second-kind physical operator is:

```text
Vphys = S_00_to_01 * R * WLowering * Vraw
```

The Section 5.2.2 VIDE Volterra term is:

```text
Vvide = S_00_to_12 * R * WLowering * Vraw
```

Banded left multiplication, tridiagonal left/right multiplication, and banded axpy are implemented by band scans. The formal code path avoids dense operator construction.

## Equation Assembly

`equation_assembly.jl` separates equation semantics from Volterra construction:

```text
u + int_0^x K(x,y)u(y)dy = f       -> A = I + Vphys
u = g + int_0^x K(x,y)u(y)dy       -> A = I - Vphys
u' - int_0^x K(x,y)u(y)dy = g      -> A = [u(0) row; D - Vvide body rows]
```

`assemble_semibanded_system` inserts the evaluation row into sparse triplets and scans only the declared band of the body operator. The final system is `SparseMatrixCSC`; the body operator remains `BandedMatrix`. Full bandwidth and body bandwidth are reported separately because the top evaluation row makes the final upper bandwidth close to `N-1`.

## Diagnostics

`diagnostics.jl` provides `safe_nnz`, `safe_bandwidth`, and `safe_density`.

- For `BandedMatrix`, diagnostics scan only the structural band.
- For `SparseMatrixCSC`, diagnostics walk `colptr`, `rowval`, and `nzval`.
- Generic dense inputs are rejected by `assert_no_dense_matrix`.

CSV helpers write stable workflow result fields without introducing plotting or tabular dependencies.

## Workflows and Examples

`workflows.jl` is intentionally high-level. It contains only:

```text
build_simple_secondkind_system
solve_simple_secondkind_sparse
build_section522_operator
build_section522_rhs
solve_section522_k100_sparse
print_section522_result
print_section522_summary
```

Examples are thin wrappers. They set parameters, call public workflow functions, and print results.

## Text Workflow Diagram

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

## Tests

The test suite covers:

```text
test_rhs_projection.jl
test_triangular_kernel_projection.jl
test_operators.jl
test_simple_secondkind_sparse.jl
test_section522_k100_sparse.jl
test_dense_path_scan.jl
```

Run:

```powershell
D:\Julia\Julia-1.11.1\bin\julia.exe --project=others/GutlebOlver -e "using Pkg; Pkg.test()"
```

## Golden Regression

The protected golden file is:

```text
test/golden_section522_k100.toml
```

It records:

```text
N = 128
kerneldepth = 12
k = 100
rhs_validation_error = 1.901613757127052e-6
tau_residual = 4.70264109582792e-11
body_residual_relative = 1.1167327725282074e-7
max_abs_error = 9.404194067919036e-8
final_system_nnz = 3388
final_system_bandwidth = (13, 127)
```

The golden test uses `atol=1e-10, rtol=1e-8` for floating-point diagnostics and exact equality for storage type, nnz, and bandwidth.

## Migration Checklist

1. Copy the full `src/` layout, not only `workflows.jl`.
2. Keep `Project.toml` dependencies for ApproxFun, BandedMatrices, BlockArrays, BlockBandedMatrices, LazyArrays, MultivariateOrthogonalPolynomials, SparseArrays, SpecialFunctions, Printf, TOML, and Test.
3. Keep examples as thin public-API wrappers.
4. Keep `test/golden_section522_k100.toml` unchanged unless a deliberate algorithm change is reviewed.
5. Run `Pkg.test()` after migration.
6. Run the dense-path scan test before accepting sparse benchmark changes.
7. Preserve the distinction between body bandwidth and final-system bandwidth.

## Dense Path Policy

Formal sparse code must not call dense materialization helpers on operators or systems. The dense-path test scans Julia files in `src`, `examples`, and `test` for direct `Matrix(`, `Array(`, or `collect(` calls. Occurrences in prose documentation are allowed but should not be copied into formal code paths.

## Verified Result

Current `Pkg.test()` passes under Julia 1.11.1 after the modular refactor. The Section 5.2.2 golden result remains:

| N | kerneldepth | status | rhs error | body residual | max abs error | raw nnz/bw | body nnz/bw | final nnz/bw |
|---:|---:|---|---:|---:|---:|---|---|---|
| 128 | 12 | pass | 1.901613757127052e-6 | 1.1167327725282074e-7 | 9.404194067919036e-8 | 2812/(11,11) | 3273/(12,14) | 3388/(13,127) |
