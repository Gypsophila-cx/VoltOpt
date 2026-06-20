# Julia 1.11.1 Sparse Gutleb--Olver Workflow

This standalone project contains the minimal sparse/strict-band workflow needed for:

1. the manufactured second-kind Volterra equation with `K=1`;
2. the Section 5.2.2 `k=100` Volterra integro-differential equation.

## Core Files

```text
Project.toml
src/SparseGOWorkflow111.jl
src/clenshaw.jl
src/triQEgen.jl
src/triVolterraFullKernel_strictband.jl
src/workflows.jl
examples/solve_section522_k100_sparse.jl
test/runtests.jl
test/test_section522_k100_sparse.jl
test/golden_section522_k100.toml
```

The copied core sources are limited to the sparse path:

- triangle Jacobi Clenshaw recurrence;
- `Q_y`, `E_y`, `Q_yE_y`;
- weighted lowering and reflection;
- strict-band full-kernel construction;
- `00 -> 01` and `00 -> 12` conversions;
- semi-banded sparse system assembly and diagnostics.

## Public API and Migration Notes

The recommended public entry points are exported from `src/SparseGOWorkflow111.jl`:

```julia
build_simple_secondkind_system
solve_simple_secondkind_sparse
build_section522_rhs
build_section522_operator
solve_section522_k100_sparse
safe_bandwidth
safe_nnz
safe_density
write_results_csv
append_result_csv
print_section522_result
print_section522_summary
```

Minimal usage:

```julia
using SparseGOWorkflow111

result = solve_section522_k100_sparse(N=128, kerneldepth=12)
println(result.max_abs_error)
println(result.final_system_type)
```

The workflow result is a `NamedTuple` with stable top-level diagnostics:

```text
k
N
kerneldepth
rhs_validation_error
tau_residual
body_residual_relative
max_abs_error
raw_operator_type
raw_operator_nnz
raw_operator_bandwidth
body_operator_type
body_operator_nnz
body_operator_bandwidth
final_system_type
final_system_nnz
full_bandwidth
body_bandwidth
density
timings
status
notes
```

`build_section522_rhs` projects the right-hand side into `Jacobi(1,2,0..1)`. `build_section522_operator` constructs the strict-band raw Volterra operator, applies the reflected/lowered/raised physical chain, builds the derivative body, and assembles the semi-banded sparse system. `solve_section522_k100_sparse` combines RHS, operator, exact residual, sparse solve, and physical-grid error validation.

The files that must migrate together are:

```text
src/SparseGOWorkflow111.jl
src/clenshaw.jl
src/triQEgen.jl
src/triVolterraFullKernel_strictband.jl
src/workflows.jl
Project.toml dependency entries for the listed packages
```

The implementation helper functions in `clenshaw.jl`, `triQEgen.jl`, and `triVolterraFullKernel_strictband.jl` are intentionally not exported as stable API in this standalone package. They are implementation details for the public workflow functions above. If the workflow is merged into a larger project, keep those helper functions in an internal namespace or submodule, and expose only the workflow-level functions until additional low-level APIs are deliberately designed.

Dense fallback is avoided by these rules:

- do not call `Matrix(A)`, `Array(A)`, or `collect(A)` on operators or systems;
- keep raw Volterra and body operators as `BandedMatrix`;
- assemble tau/evaluation rows and body rows directly into `SparseMatrixCSC` from triplets;
- use `safe_nnz`, `safe_bandwidth`, and `safe_density` for diagnostics;
- solve the final semi-banded sparse system directly with `A \ b`.

CSV output is available without heavy dependencies:

```julia
result = solve_section522_k100_sparse(N=128, kerneldepth=12)
write_results_csv("section522_results.csv", (result,))
append_result_csv("section522_results.csv", result)
```

When merging into a larger project, keep example scripts thin: they should set parameters, call `solve_*` or `build_*` APIs, and print or write results. Mathematical assembly should remain in `src/workflows.jl` or the corresponding library module.

## Stable Public API

`build_simple_secondkind_system`: builds the manufactured `K=1` second-kind Volterra raw operator, physical Volterra operator, banded system, RHS coefficients, and exact coefficients without dense conversion.

`solve_simple_secondkind_sparse`: solves the manufactured `K=1` second-kind sparse problem and returns validation errors, sparse/banded diagnostics, timings, status, and notes.

`build_section522_operator`: constructs the Section 5.2.2 strict-band raw Volterra operator, reflected/lowered/raised Volterra term, derivative body, and final semi-banded sparse system.

`build_section522_rhs`: projects the Section 5.2.2 RHS into the final `Jacobi(1,2,0..1)` output basis and returns coefficients plus validation metadata.

`solve_section522_k100_sparse`: solves the Section 5.2.2 VIDE for `k=100` by combining the public RHS and operator builders, exact residual diagnostics, sparse solve, and physical-grid error check.

`safe_bandwidth`: reports actual lower/upper nonzero bandwidth for `BandedMatrix` or `SparseMatrixCSC` without dense materialization.

`safe_nnz`: counts nonzero entries in `BandedMatrix` or `SparseMatrixCSC` without converting to dense storage.

`safe_density`: computes `safe_nnz(A) / prod(size(A))` for sparse or banded matrices.

`write_results_csv`: writes one or more structured workflow results to CSV using the stable public result fields.

`append_result_csv`: appends a single structured workflow result to CSV and creates the header if needed.

`print_section522_result`: prints the detailed Section 5.2.2 result, including residuals, storage diagnostics, and timings.

`print_section522_summary`: prints a CSV-style one-line Section 5.2.2 summary.

## Golden Regression

The reference Section 5.2.2 `k=100` result is stored in:

```text
test/golden_section522_k100.toml
```

It records:

```text
k = 100
N = 128
kerneldepth = 12
rhs_validation_error
tau_residual
body_residual_relative
max_abs_error
raw/body/final storage type
raw/body/final nnz
raw/body/final bandwidth
density
```

The package test reads this TOML file. Floating-point diagnostics use
`atol=1e-10, rtol=1e-8`; storage type, nnz, and bandwidth must match exactly.

## Section 5.2.2 Equation

The standalone example solves

```math
u'(x) - \int_0^x y\exp(x^2)u(y)\,dy = g(x,100),
\qquad u(0)=0,
```

with exact solution

```math
u(x)=\arctan(100x).
```

The right-hand side is projected to `Jacobi(1,2,0..1)`, the final output basis for a first derivative of a `Jacobi(0,1,0..1)` solution.

## Operator Chain

For `int_0^x`, kernel coefficients are built with `flip=true`, so the raw strict-band Volterra operator is

```math
V_{\rm raw} = Q_y K(1-X,Y) E_y.
```

The physical VIDE body is

```math
D^{(1)} - S_{00\to12} R L V_{\rm raw}.
```

The final system has one dense initial-condition row and `N-1` banded body rows:

```math
A =
\begin{bmatrix}
E_0 \\
B_{1:N-1,1:N}
\end{bmatrix}.
```

It is assembled directly into `SparseMatrixCSC` from triplets.

## Run

```powershell
julia --project=. -e "using Pkg; Pkg.instantiate(); Pkg.test()"
julia --project=. examples/solve_section522_k100_sparse.jl
```

Acceptance parameters:

```text
N = 128
kerneldepth = 12
k = 100
```

Acceptance criteria:

```text
rhs_validation_error <= 1e-5
body_residual_relative <= 1e-6
physical-grid max_abs_error <= 1e-6
final system is sparse/semi-banded
```

## Julia 1.11.1 Compatibility

Verified command:

```powershell
& 'D:\Julia\Julia-1.11.1\bin\julia.exe' --project=SparseGOWorkflow111 -e "using InteractiveUtils; versioninfo()"
```

Key version:

```text
Julia Version 1.11.1
```

The standalone project uses these direct dependencies:

```text
ApproxFun
BandedMatrices
BlockArrays
BlockBandedMatrices
LazyArrays
MultivariateOrthogonalPolynomials
SpecialFunctions
LinearAlgebra
Printf
SparseArrays
Test
TOML
```

`FastTransforms` and `ContinuumArrays` are pulled as transitive dependencies by the orthogonal-polynomial stack. The Julia 1.11-compatible path avoids declaring them as direct dependencies in this local project. The triangular transform planner is accessed through the loaded `MultivariateOrthogonalPolynomials.ContinuumArrays` binding, and the kernel sampler accepts both the older two-grid form and the current one-grid `SVector{2}` form.

## Verified k=100 Result

Command:

```powershell
& 'D:\Julia\Julia-1.11.1\bin\julia.exe' --project=SparseGOWorkflow111 SparseGOWorkflow111\examples\solve_section522_k100_sparse.jl
```

Result:

| k | N | kerneldepth | rhs_validation_error | body_residual_relative | max_abs_error | raw type | body type | final system | nnz | lower bw | upper bw | density |
|---:|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|
| 100 | 128 | 12 | 1.901613757127052e-6 | 1.1167327725282074e-7 | 9.404194067919036e-8 | BandedMatrix | BandedMatrix | SparseMatrixCSC | 3388 | 13 | 127 | 0.206787109375 |

Detailed matrix data:

```text
raw V      : BandedMatrix nnz=2812 bw=(11,11) density=1.716309e-01
body       : BandedMatrix nnz=3273 bw=(12,14) density=1.997681e-01
system     : SparseMatrixCSC nnz=3388 bw=(13,127) density=2.067871e-01
```

Detailed timings from the verified run:

```text
rhs                     = 1.611964e-01 s
kernel coefficients     = 9.313016e+00 s
Volterra construction   = 1.153221e+00 s
system assembly         = 3.707160e-02 s
solve                   = 3.735000e-04 s
evaluation              = 5.195000e-04 s
total                   = 1.067928e+01 s
```

## Tests

Command:

```powershell
& 'D:\Julia\Julia-1.11.1\bin\julia.exe' --project=SparseGOWorkflow111 -e "using Pkg; Pkg.instantiate(); Pkg.test()"
```

Result:

```text
simple second-kind sparse solve      | 10 pass
Section 5.2.2 k=100 sparse solve     | 30 pass
SparseGOWorkflow111 tests passed
```
