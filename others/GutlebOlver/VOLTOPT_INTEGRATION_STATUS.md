# VoltOpt Integration Status

Date: 2026-06-09

## Scope

This status records the standalone validation of the copied Gutleb-Olver sparse workflow inside VoltOpt. No mathematical algorithm files, strict-band paths, or semi-banded sparse numerical paths were changed during this validation.

## Source And Placement

- Copy source: modular Gutleb--Olver sparse workflow.
- Current placement: `others/GutlebOlver`.
- Current package name: `GutlebOlver`.
- Current module name: `GutlebOlver`.
- The package was renamed from the copied `SparseGOWorkflow111` source during the final modularization pass.

## Checked Structure

The copied subproject contains:

- `Project.toml`
- `Manifest.toml`
- `src/`
- `examples/`
- `test/`
- `README.md`
- `GO_sparse_workflow_111.md`
- `test/golden_section522_k100.toml`

## Julia 1.11.1 Environment

Command:

```powershell
D:\Julia\Julia-1.11.1\bin\julia.exe --project=others/GutlebOlver -e "using InteractiveUtils; versioninfo()"
```

Summary:

- Julia Version: 1.11.1
- Commit: `8f5b7ca12a`
- Platform: Windows x86_64-w64-mingw32
- CPU: 20 x 13th Gen Intel(R) Core(TM) i7-13650HX
- Threads: 1 default, 0 interactive, 1 GC
- `JULIA_PKG_SERVER`: `https://mirrors.bfsu.edu.cn/julia`

## Test Result

Command:

```powershell
D:\Julia\Julia-1.11.1\bin\julia.exe --project=others/GutlebOlver -e "using Pkg; Pkg.instantiate(); Pkg.test()"
```

Result: passed.

Current modular test summaries:

- `RHS projection`: 23 passed, 23 total.
- `triangular kernel projection`: 17 passed, 17 total.
- `basic sparse operators`: 12 passed, 12 total.
- `simple second-kind sparse solve`: 10 passed, 10 total.
- `Section 5.2.2 k=100 sparse solve`: 31 passed, 31 total.
- `dense path scan`: 1 passed, 1 total.

Note: the initial sandboxed run could not write Julia's user depot log file at `C:\Users\chenx\.julia\logs\manifest_usage.toml.pid`. The same command passed when run with normal Julia depot write access.

## Standalone Example Result

Command:

```powershell
D:\Julia\Julia-1.11.1\bin\julia.exe --project=others/GutlebOlver others/GutlebOlver/examples/solve_section522_k100_sparse.jl
```

Result: passed.

Observed diagnostics:

- `rhs_validation_error = 1.90161375712705194e-06`
- `tau_residual = 4.70264109582792014e-11`
- `body_residual_relative = 1.11673277252820736e-07`
- `max_abs_error = 9.40419406791903612e-08`
- raw `V`: `BandedMatrix`, `nnz = 2812`, bandwidth `(11, 11)`
- body: `BandedMatrix`, `nnz = 3273`, bandwidth `(12, 14)`
- final system: `SparseMatrixCSC`, `nnz = 3388`, bandwidth `(13, 127)`

## Golden Regression

`test/golden_section522_k100.toml` is preserved and the regression passed under Julia 1.11.1.

The test checks:

- storage types for raw operator, body operator, and final system
- `nnz` values
- bandwidth values
- density
- floating diagnostics using `atol=1e-10, rtol=1e-8`

Current golden structural values:

- raw operator: `BandedMatrix`, `nnz = 2812`, bandwidth `[11, 11]`
- body operator: `BandedMatrix`, `nnz = 3273`, bandwidth `[12, 14]`
- final system: `SparseMatrixCSC`, `nnz = 3388`, bandwidth `[13, 127]`
- density: `2.06787109375e-1`

## Dense Fallback Scan

Scan command:

```powershell
rg -n --pcre2 "(?<![A-Za-z0-9_])(?:Matrix|Array|collect)\(" others/GutlebOlver -g "*.jl"
```

Result: no bare `Matrix(`, `Array(`, or `collect(` tokens were found in `others/GutlebOlver/**/*.jl`.

An earlier broad substring scan matched sparse or banded constructor names such as `BandedMatrix(` and `BlockBandedMatrix(`. Those are not dense fallback paths.

## Public API

The following exported API names are present and were checked with `names(GutlebOlver)`:

- `build_go_rhs_coefficients`
- `validate_rhs_projection`
- `precompute_triangular_kernel_plan`
- `sample_triangular_kernel_values`
- `project_triangular_kernel_coeffs_from_values`
- `triangular_kernel_coeffs_from_plan`
- `triVolterraFullKernelOpP01_strictband_from_coeffs`
- `build_secondkind_volterra_strictband`
- `build_secondkind_system_strictband`
- `build_VIDE_LHS_sec522_strictband`
- `triQgen`
- `triEygenP01`
- `triQEygenP01`
- `WLoweringP01P00`
- `reflectPabtoPba`
- `apply_reflect_lowering_strictband`
- `apply_conversion_00_to_01_strictband`
- `apply_conversion_00_to_12_strictband`
- `build_derivative_P01_to_P12`
- `build_initial_evaluation_row`
- `assemble_semibanded_system`
- `assemble_section522_semibanded_system`
- `build_simple_secondkind_system`
- `solve_simple_secondkind_sparse`
- `build_section522_operator`
- `build_section522_rhs`
- `solve_section522_k100_sparse`
- `safe_bandwidth`
- `safe_nnz`
- `safe_density`
- `write_results_csv`
- `append_result_csv`
- `print_section522_result`
- `print_section522_summary`
- `SECTION522_K`
- `SECTION522_ACCEPTANCE_N`
- `SECTION522_ACCEPTANCE_KERNELDEPTH`

Direct API smoke test:

```powershell
D:\Julia\Julia-1.11.1\bin\julia.exe --project=others/GutlebOlver -e "using GutlebOlver; r = solve_section522_k100_sparse(N=128, kerneldepth=12); println(r.max_abs_error)"
```

Output:

```text
9.404194067919036e-8
```

## Next Step For VoltOpt Benchmark Integration

After this standalone subproject remains stable, the next integration step is to add a VoltOpt benchmark adapter that calls the existing public API from `others/GutlebOlver` without modifying the strict-band or semi-banded sparse implementation. Suggested order:

1. Add an explicit benchmark entry point in the VoltOpt benchmark layer.
2. Keep `others/GutlebOlver` as an independent Julia project using module `GutlebOlver`.
3. Compare VoltOpt ex1/ex4 benchmark output against the preserved Section 5.2.2 k=100 golden result.
4. Add only adapter-level tests first, leaving the copied sparse workflow tests as the authoritative standalone regression.

## Stable Files Not To Change In This Phase

The following files are treated as stable for the modularized sparse workflow and should not be changed when wiring benchmarks unless a separate reviewed task requires it:

- `others/GutlebOlver/Project.toml`
- `others/GutlebOlver/Manifest.toml`
- `others/GutlebOlver/src/GutlebOlver.jl`
- `others/GutlebOlver/src/rhs_projection.jl`
- `others/GutlebOlver/src/triangular_kernel_projection.jl`
- `others/GutlebOlver/src/operators.jl`
- `others/GutlebOlver/src/volterra_strictband.jl`
- `others/GutlebOlver/src/equation_assembly.jl`
- `others/GutlebOlver/src/diagnostics.jl`
- `others/GutlebOlver/src/workflows.jl`
- `others/GutlebOlver/src/clenshaw.jl`
- `others/GutlebOlver/test/runtests.jl`
- `others/GutlebOlver/test/test_rhs_projection.jl`
- `others/GutlebOlver/test/test_triangular_kernel_projection.jl`
- `others/GutlebOlver/test/test_operators.jl`
- `others/GutlebOlver/test/test_simple_secondkind_sparse.jl`
- `others/GutlebOlver/test/test_section522_k100_sparse.jl`
- `others/GutlebOlver/test/test_dense_path_scan.jl`
- `others/GutlebOlver/test/golden_section522_k100.toml`
- `others/GutlebOlver/examples/solve_section522_k100_sparse.jl`
- `others/GutlebOlver/examples/solve_simple_secondkind_sparse.jl`
