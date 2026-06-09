# Gutleb-Olver Section 5.2.2 Sparse Benchmark

This directory contains a minimal VoltOpt-side benchmark adapter for the standalone package in `others/GutlebOlver`.

The benchmark calls the independent Julia package `SparseGOWorkflow111` by activating `others/GutlebOlver` inside the script. It does not add `SparseGOWorkflow111` to the VoltOpt root `Project.toml`.

No VoltOpt main algorithm files are modified by this benchmark. No Gutleb-Olver mathematical algorithm path is modified; the script only calls the public API `solve_section522_k100_sparse(N=128, kerneldepth=12)`.

## Benchmarks

`bench_section522_k100_sparse.jl` is a smoke benchmark. It runs the Section 5.2.2 `k=100` sparse solve once and records a single wall-clock time. Use it to check that the standalone package can be called from the VoltOpt root.

`bench_section522_k100_sparse_repeated.jl` is the formal timing benchmark. It runs one warmup solve first and does not include that first run in the timing summary. It then runs at least five timed repeats and reports `min_time`, `median_time`, and `mean_time`.

For formal comparisons against VoltOpt timings, use `min_time` or `median_time`; do not use the first run.

## Run

From the VoltOpt root directory:

```powershell
julia --project=. test/compare/gutleb_olver/bench_section522_k100_sparse.jl
julia --project=. test/compare/gutleb_olver/bench_section522_k100_sparse_repeated.jl
```

Smoke command record:

```text
julia --project=. test/compare/gutleb_olver/bench_section522_k100_sparse.jl
julia --project=. test/compare/gutleb_olver/bench_section522_k100_sparse_repeated.jl
```

The script temporarily activates:

```julia
Pkg.activate(joinpath(root, "others", "GutlebOlver"))
```

and then imports:

```julia
using SparseGOWorkflow111
```

## Smoke Output

The smoke benchmark writes:

```text
test/compare/gutleb_olver/data/section522_k100_sparse_results.csv
```

CSV fields:

- `N`: Section 5.2.2 solve size.
- `kerneldepth`: triangular kernel coefficient depth.
- `elapsed_time`: wall-clock seconds for `solve_section522_k100_sparse(N=128, kerneldepth=12)` in this script run.
- `rhs_validation_error`: public result diagnostic for RHS construction.
- `tau_residual`: tau row residual diagnostic.
- `body_residual_relative`: relative residual for the body rows.
- `max_abs_error`: maximum absolute physical-grid error.
- `raw_operator_type`, `body_operator_type`, `final_system_type`: storage type labels.
- `raw_operator_nnz`, `body_operator_nnz`, `final_system_nnz`: sparse or banded nonzero counts.
- `raw_lower_bandwidth`, `raw_upper_bandwidth`: raw operator bandwidth.
- `body_lower_bandwidth`, `body_upper_bandwidth`: body operator bandwidth.
- `final_lower_bandwidth`, `final_upper_bandwidth`: final system bandwidth.
- `density`: final system density diagnostic.
- `status`: status returned by the standalone package workflow.

## Formal Timing Output

The repeated timing benchmark writes:

```text
test/compare/gutleb_olver/data/section522_k100_sparse_repeated_results.csv
```

CSV fields:

- `N`: Section 5.2.2 solve size.
- `kerneldepth`: triangular kernel coefficient depth.
- `repeats`: number of timed repeats after warmup. This is at least 5.
- `warmup_done`: whether the unrecorded warmup solve completed.
- `min_time`: minimum wall-clock seconds over timed repeats.
- `median_time`: median wall-clock seconds over timed repeats.
- `mean_time`: arithmetic mean wall-clock seconds over timed repeats.
- `rhs_validation_error`: public result diagnostic for RHS construction.
- `tau_residual`: tau row residual diagnostic.
- `body_residual_relative`: relative residual for the body rows.
- `max_abs_error`: maximum absolute physical-grid error.
- `raw_storage`, `body_storage`, `final_storage`: storage type labels.
- `raw_nnz`, `body_nnz`, `final_nnz`: sparse or banded nonzero counts.
- `raw_lower_bw`, `raw_upper_bw`: raw operator bandwidth.
- `body_lower_bw`, `body_upper_bw`: body operator bandwidth.
- `final_lower_bw`, `final_upper_bw`: final system bandwidth.
- `status`: aggregate status for the timed repeats.
- `elapsed_times`: semicolon-separated wall-clock seconds for each timed repeat.
