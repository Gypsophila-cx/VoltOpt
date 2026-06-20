# Gutleb-Olver Section 5.2.2 Sparse Benchmark

This directory contains a minimal VoltOpt-side benchmark adapter for the standalone package in `others/GutlebOlver`.

The benchmark calls the independent Julia package `GutlebOlver` by activating `others/GutlebOlver` inside the script. It does not add `GutlebOlver` to the VoltOpt root `Project.toml`.

No VoltOpt main algorithm files are modified by this benchmark. No Gutleb-Olver mathematical algorithm path is modified; the script only calls the public API `solve_section522_k100_sparse(N=128, kerneldepth=12)`.

## Benchmarks

`bench_section522_k100_sparse.jl` is a smoke benchmark. It runs the Section 5.2.2 `k=100` sparse solve once and records a single wall-clock time. Use it to check that the standalone package can be called from the VoltOpt root.

`bench_section522_k100_sparse_repeated.jl` is the fixed-`N` formal timing benchmark. It runs one warmup solve first and does not include that first run in the timing summary. It then runs at least five timed repeats and reports `min_time`, `median_time`, and `mean_time`.

`bench_section522_k100_sparse_varyN.jl` is the varying-`N` timing benchmark. It fixes `kerneldepth = 12`, runs one warmup per `N`, then records at least five timed repeats per `N`. The default `N` list is `[32, 48, 64, 96, 128, 192, 256, 384, 512]`. If a large `N` is too slow or errors, the script keeps already completed rows in the CSV and records the failed `N` with `status = error`.

`bench_section522_k100_sparse_vary_kerneldepth.jl` is the varying-`kerneldepth` timing benchmark. It fixes `N = 128`, runs one warmup per `kerneldepth`, then records at least five timed repeats per `kerneldepth`. The default `kerneldepth` list is `[4, 6, 8, 10, 12, 14, 16, 20, 24]`. If a large `kerneldepth` is too slow, unstable, or errors, the script keeps already completed rows in the CSV and records the failed `kerneldepth` with `status = error`.

`solve_warmup_with_gutleb_olver.jl` is a warmup problem adapter. It reads the low-order VIDE problem from `examples/warmup/solve_vide_low_order_cheb_leg.jl`, maps the original `[-1,1]` equation to `s in [0,1]`, and solves it through the public `GutlebOlver` sparse API. The adapter does not modify `others/GutlebOlver/src` or the golden regression file.

For formal comparisons against VoltOpt timings, use `min_time` or `median_time`; do not use the first run.

## Run

From the VoltOpt root directory:

```powershell
julia --project=. test/compare/gutleb_olver/bench_section522_k100_sparse.jl
julia --project=. test/compare/gutleb_olver/bench_section522_k100_sparse_repeated.jl
julia --project=. test/compare/gutleb_olver/bench_section522_k100_sparse_varyN.jl
julia --project=. test/compare/gutleb_olver/bench_section522_k100_sparse_vary_kerneldepth.jl
julia --project=. test/compare/gutleb_olver/solve_warmup_with_gutleb_olver.jl
```

Smoke command record:

```text
julia --project=. test/compare/gutleb_olver/bench_section522_k100_sparse.jl
julia --project=. test/compare/gutleb_olver/bench_section522_k100_sparse_repeated.jl
julia --project=. test/compare/gutleb_olver/bench_section522_k100_sparse_varyN.jl
julia --project=. test/compare/gutleb_olver/bench_section522_k100_sparse_vary_kerneldepth.jl
julia --project=. test/compare/gutleb_olver/solve_warmup_with_gutleb_olver.jl
```

The script temporarily activates:

```julia
Pkg.activate(joinpath(root, "others", "GutlebOlver"))
```

and then imports:

```julia
using GutlebOlver
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

## Fixed-N Formal Timing Output

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

## Varying-N Timing Output

The varying-`N` timing benchmark writes:

```text
test/compare/gutleb_olver/data/section522_k100_sparse_varyN_results.csv
```

CSV fields:

- `N`: Section 5.2.2 solve size.
- `kerneldepth`: fixed triangular kernel coefficient depth, currently `12`.
- `repeats`: number of timed repeats after warmup. This is at least 5.
- `warmup_done`: whether the unrecorded warmup solve completed for this `N`.
- `min_time`: minimum wall-clock seconds over timed repeats for this `N`.
- `median_time`: median wall-clock seconds over timed repeats for this `N`.
- `mean_time`: arithmetic mean wall-clock seconds over timed repeats for this `N`.
- `rhs_validation_error`: public result diagnostic for RHS construction.
- `tau_residual`: tau row residual diagnostic.
- `body_residual_relative`: relative residual for the body rows.
- `max_abs_error`: maximum absolute physical-grid error.
- `raw_storage`, `body_storage`, `final_storage`: storage type labels.
- `raw_nnz`, `body_nnz`, `final_nnz`: sparse or banded nonzero counts.
- `raw_lower_bw`, `raw_upper_bw`: raw operator bandwidth.
- `body_lower_bw`, `body_upper_bw`: body operator bandwidth.
- `final_lower_bw`, `final_upper_bw`: final system bandwidth.
- `status`: aggregate status for the timed repeats, or `error` if this `N` could not complete.
- `elapsed_times`: semicolon-separated wall-clock seconds for each timed repeat.
- `error_message`: exception text for failed rows, otherwise empty.

You can override the default `N` list and repeat count for exploratory runs:

```powershell
$env:GUTLEB_OLVER_NS = "32,64,128"
$env:GUTLEB_OLVER_REPEATS = "5"
julia --project=. test/compare/gutleb_olver/bench_section522_k100_sparse_varyN.jl
```

## Varying-Kerneldepth Timing Output

The varying-`kerneldepth` timing benchmark writes:

```text
test/compare/gutleb_olver/data/section522_k100_sparse_vary_kerneldepth_results.csv
```

CSV fields:

- `N`: fixed Section 5.2.2 solve size, currently `128`.
- `kerneldepth`: triangular kernel coefficient depth for this row.
- `repeats`: number of timed repeats after warmup. This is at least 5.
- `warmup_done`: whether the unrecorded warmup solve completed for this `kerneldepth`.
- `min_time`: minimum wall-clock seconds over timed repeats for this `kerneldepth`.
- `median_time`: median wall-clock seconds over timed repeats for this `kerneldepth`.
- `mean_time`: arithmetic mean wall-clock seconds over timed repeats for this `kerneldepth`.
- `rhs_validation_error`: public result diagnostic for RHS construction.
- `tau_residual`: tau row residual diagnostic.
- `body_residual_relative`: relative residual for the body rows.
- `max_abs_error`: maximum absolute physical-grid error.
- `raw_storage`, `body_storage`, `final_storage`: storage type labels.
- `raw_nnz`, `body_nnz`, `final_nnz`: sparse or banded nonzero counts.
- `raw_lower_bw`, `raw_upper_bw`: raw operator bandwidth.
- `body_lower_bw`, `body_upper_bw`: body operator bandwidth.
- `final_lower_bw`, `final_upper_bw`: final system bandwidth.
- `status`: aggregate status for the timed repeats, or `error` if this `kerneldepth` could not complete.
- `elapsed_times`: semicolon-separated wall-clock seconds for each timed repeat.
- `error_message`: exception text for failed rows, otherwise empty.

Current fixed-`N`, varying-`kerneldepth` results show that `kerneldepth = 4`, `6`, and `8` complete the computation but do not reach full acceptance, so their aggregate `status` is `check`. All tested values with `kerneldepth >= 10` reach `pass`, making `kerneldepth = 10` the smallest passing value in this sweep. The standalone golden result continues to use `kerneldepth = 12` as a conservative setting. The final system upper bandwidth is close to `N - 1` because the top evaluation row couples across the system; the body bandwidth is the diagnostic that reflects the strict-band operator body. The nonzero counts and timings increase with `kerneldepth`, matching the expected bandwidth growth.

Figures generated from this CSV are written to:

- `test/compare/gutleb_olver/figures/section522_k100_time_vs_kerneldepth.(png|pdf)`
- `test/compare/gutleb_olver/figures/section522_k100_error_vs_kerneldepth.(png|pdf)`

You can override the default `kerneldepth` list and repeat count for exploratory runs:

```powershell
$env:GUTLEB_OLVER_KERNELDEPTHS = "8,12,16"
$env:GUTLEB_OLVER_REPEATS = "5"
julia --project=. test/compare/gutleb_olver/bench_section522_k100_sparse_vary_kerneldepth.jl
```

## Warmup VIDE Output

The warmup adapter writes:

```text
test/compare/gutleb_olver/data/warmup_gutleb_olver_results.csv
```

The original warmup equation is

```text
2u_x(x) + u(x) - int_{-1}^x K(x,t)u(t)dt = x + 2,  u(-1)=1,
K(x,t) = ((x+1)(x+2)/4) exp(((t+1)(x-t))/4),
u(x) = exp((x+1)^2/4).
```

With `x = 2s - 1` and `t = 2r - 1`, the adapter solves

```text
u_s(s) + u(s) - int_0^s K_go(s,r)u(r)dr = 2s + 1,  u(0)=1,
K_go(s,r) = 2*K(2s-1, 2r-1).
```

The sparse chain is:

```text
K_go -> triangular coefficients with flip=true -> Vraw
Vraw -> weighted lowering -> reflection -> 00->12 conversion
body = D_01_to_12 + Conversion(01->12) - converted Volterra
final system = [u(0) row; first N-1 body rows]
```

For `N in [16, 32, 64, 128]` and `kerneldepth in [4, 8, 12]`, the current run completed every row. `kerneldepth=4` completed but stayed at `status=check`; `kerneldepth=8` and `12` reached `status=pass` for all tested `N`. The `N=128`, `kerneldepth=12` row has `body_residual_relative = 4.980386944903045e-13`, `max_abs_error = 4.1522341120980855e-13`, final system `SparseMatrixCSC`, `nnz = 3388`, and bandwidth `(13,127)`.
