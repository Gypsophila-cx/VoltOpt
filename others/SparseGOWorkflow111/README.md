# SparseGOWorkflow111

Standalone Julia 1.11.1 project for the strict-band / semi-banded sparse Gutleb--Olver Volterra workflow used in this repository.

## Run Tests

From this directory:

```powershell
julia --project=. -e "using Pkg; Pkg.instantiate(); Pkg.test()"
```

From the parent repository:

```powershell
julia --project=SparseGOWorkflow111 -e "using Pkg; Pkg.instantiate(); Pkg.test()"
```

## Run Section 5.2.2 Example

From this directory:

```powershell
julia --project=. examples/solve_section522_k100_sparse.jl
```

From the parent repository:

```powershell
julia --project=SparseGOWorkflow111 SparseGOWorkflow111/examples/solve_section522_k100_sparse.jl
```

The example solves

```math
u'(x) - \int_0^x y\exp(x^2)u(y)\,dy = g(x,100),
\qquad u(0)=0,
```

with exact solution `u(x)=atan(100*x)`.

## API Usage

```julia
using SparseGOWorkflow111

result = solve_section522_k100_sparse(N=128, kerneldepth=12)
println(result.max_abs_error)
println(result.final_system_type)
println(result.full_bandwidth)
```

Useful public entry points:

```julia
build_simple_secondkind_system(N=32)
solve_simple_secondkind_sparse(N=32)

rhs = build_section522_rhs(N=128, k=100.0)
ops = build_section522_operator(N=128, kerneldepth=12)
result = solve_section522_k100_sparse(N=128, kerneldepth=12)

safe_nnz(ops.system)
safe_bandwidth(ops.system)
safe_density(ops.system)

write_results_csv("section522_results.csv", (result,))
append_result_csv("section522_results.csv", result)
```

The result returned by `solve_section522_k100_sparse` is a `NamedTuple` with stable top-level diagnostics including:

```text
k, N, kerneldepth, rhs_validation_error, tau_residual,
body_residual_relative, max_abs_error,
raw_operator_type, raw_operator_nnz, raw_operator_bandwidth,
body_operator_type, body_operator_nnz, body_operator_bandwidth,
final_system_type, final_system_nnz, full_bandwidth,
body_bandwidth, density, timings, status, notes
```

## Migration Checklist

When moving this reference workflow into another Julia 1.11.1 project, keep the following pieces together:

- Copy the direct dependencies from `Project.toml`: `ApproxFun`, `BandedMatrices`, `BlockArrays`, `BlockBandedMatrices`, `LazyArrays`, `MultivariateOrthogonalPolynomials`, `SpecialFunctions`, `LinearAlgebra`, `Printf`, `SparseArrays`, `Test`, and `TOML`.
- Copy `src/clenshaw.jl`.
- Copy `src/triQEgen.jl`.
- Copy `src/triVolterraFullKernel_strictband.jl`.
- Copy `src/workflows.jl`.
- Keep the public API exported by `src/SparseGOWorkflow111.jl`; avoid exposing low-level helpers until the larger project has a deliberate API design.
- Keep `test/test_section522_k100_sparse.jl` and `test/golden_section522_k100.toml` as regression coverage.
- Keep examples thin: set parameters, call public workflow functions, print or write results.
- Keep diagnostics on `safe_nnz`, `safe_bandwidth`, and `safe_density`; do not use dense matrix conversion for statistics.
- Re-run `Pkg.test()` and the Section 5.2.2 example after migration.

## Sparse Path

The implementation uses:

- `Jacobi(0,1,0..1)` for solution coefficients;
- `Jacobi(1,2,0..1)` for the first-derivative output space;
- triangle kernel coefficients with `flip=true`, i.e. `K(1-X,Y)`;
- strict-band `Q_y K(1-X,Y) E_y`;
- `WLowering`, reflection, and `00 -> 12` conversion;
- a semi-banded final `SparseMatrixCSC` with one dense initial row and banded body rows.

The code does not use dense collocation and does not construct a dense matrix before converting to sparse.

## Verified Result

Verified with `D:\Julia\Julia-1.11.1\bin\julia.exe`:

```text
Julia Version 1.11.1
```

For `N=128`, `kerneldepth=12`, `k=100`:

```text
rhs_validation_error      = 1.901613757127052e-6
body_residual_relative    = 1.1167327725282074e-7
physical-grid max error   = 9.404194067919036e-8
final system              = SparseMatrixCSC
system nnz                = 3388
system bandwidth          = (13, 127)
density                   = 0.206787109375
```

`Pkg.test()` passes for both the `K=1` second-kind equation and the Section 5.2.2 `k=100` sparse VIDE case. The Section 5.2.2 test reads `test/golden_section522_k100.toml` and checks errors with `atol=1e-10, rtol=1e-8`; storage type, nnz, and bandwidth must match exactly.
