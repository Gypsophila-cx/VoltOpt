# Issue List

Last updated: 2026-06-23

## Summary

- P0: 0 current blocking issues.
- P1: 0 current high-priority reproducibility issues.
- P2: 0 current medium-priority documentation/path/schema issues.
- P3: 0 current low-priority cleanup issues.

This audit is based on the current filesystem state in `D:\Projects\VoltOpt`
and the current working tree.

## P0 - Blocking issues

No current P0 issues are open.

## P1 - High-priority issues

No current P1 issues are open.

## P2 - Medium-priority issues

No current P2 issues are open.

## P3 - Low-priority cleanup / documentation issues

No current P3 issues are open after the generated-artifact documentation,
plot-reference-line documentation, static layout audit script, and lightweight
smoke test were added in the working tree.

## Release / Staging Notes

- Stage `README.md`, `issue_list.md`, `Project.toml`, `tools/audit_layout.jl`,
  and `test/runtests.jl` with the existing documentation/layout cleanup before
  the next commit.
- If deleted-plus-added artifact moves reappear in `git status`, stage both
  sides of each move together so the commit preserves the intended relocation.

## Resolved in working tree

### README listed missing core example directories and run commands

- Resolution: The top-level README uses current paths such as
  `examples/ex1_warmup/`, `examples/ex2_2ndVolterraEqn/`, and
  `test/nonlinear_bench/`.

### README did not document current supported examples and nonlinear benchmark

- Resolution: The top-level README documents the current supported example and
  benchmark directories, including `test/nonlinear_bench/`.

### Top-level README described ex5 as a data/figures exception

- Resolution: The README documents ex5 as using local
  `examples/ex5_psespectra/data/` and `examples/ex5_psespectra/figures/`, and
  the MATLAB command uses `run('examples/ex5_psespectra/ex5_plot.m')`.

### Ex5 scripts, README, and artifacts disagreed on paths

- Resolution:
  - `examples/ex5_psespectra/ex5_volterra_schur_besselj1.jl` writes CSV output
    to local `data/` and PDF/PNG output to local `figures/`.
  - `examples/ex5_psespectra/ex5_plot.m` reads from local `data/` and writes
    PDF/PNG output to local `figures/`.
  - `examples/ex5_psespectra/README.md` documents the local `data/`/`figures/`
    workflow.

### Ex3 had two current scripts with different artifact layouts

- Resolution: `examples/ex3_pantograph/ex3_pantograph_firstkind.jl` uses local
  `data/` for CSV output and local `figures/` for PDF/PNG output, matching the
  layout already used by `ex3_run.jl` and `ex3_plot.m`.

### Warmup `N=1` rows were reported as a P1 failure

- Resolution:
  - README states that the `N=1` rows in
    `examples/ex1_warmup/data/ex1_error_N1_100.csv` are expected boundary
    diagnostics.
  - The solvers require `N >= 2`.
  - `examples/ex1_warmup/ex1_plot.m` filters to `status=success` rows for
    formal error and figure output.

### Smooth benchmark root CSV artifacts were treated as current outputs

- Resolution:
  - The old root-level smooth CSV snapshots were moved to
    `test/smooth_bench/legacy/`.
  - README states that current smooth CSV outputs are under
    `test/smooth_bench/data/` and older root-level snapshots are kept under
    `test/smooth_bench/legacy/`.

### `test/` had no conventional package test entry

- Resolution:
  - `test/runtests.jl` now provides lightweight package smoke tests.
  - `Project.toml` declares the standard-library `Test` dependency for the
    package test target.
  - README states that `Pkg.test()` runs only smoke tests and does not run the
    benchmark scripts.

### Benchmark CSV schemas were not aligned across benchmark directories

- Resolution: README states that smooth, weakly singular, and nonlinear
  benchmark CSV schemas are directory-specific and are consumed by their
  corresponding plotting scripts rather than a single shared schema.

### Example runner coverage was inconsistent

- Resolution: README says that examples with multiple Julia scripts should run
  them in the listed order before running the MATLAB plotting script. The README
  lists each example's available Julia and MATLAB entry points.

### Ex4 plot comments mentioned old build-time file names

- Resolution: `examples/ex4_oscillatory/ex4_plot.m` header comments list the
  actual output names `ex4_depth_time_N200.pdf` and
  `ex4_depth_time_N200.png`.

### Ex6 comments referenced an obsolete lowercase path without naming the canonical path

- Resolution:
  - `examples/ex6_nonlinear/ex6_nonlinear_picard.jl` names
    `src/CompVolt/compVolt.jl` as the canonical path and describes
    `src/compVolt.jl` as legacy compatibility.
  - `examples/ex6_nonlinear/ex6_fixed10_varyN.jl` has the same canonical-path
    clarification.

### Ex5 local copied source file needed a clearer note

- Resolution: `examples/ex5_psespectra/README.md` states that
  `VoltMat_Leg.jl` is retained intentionally as an ex5-specific complex-valued
  Legendre support copy for reproduction history, while the current Julia entry
  point imports `src/VoltMat/voltmat_leg.jl`.

### Nonlinear benchmark uses an internal helper outside the exported package API

- Resolution: README documents that
  `test/nonlinear_bench/bench_nonlinear_volterra_clenshaw_vs_compvolt.jl`
  directly includes `src/CompVolt/clenshaw_valuemat.jl` as an internal
  benchmark helper and that the helper is not currently exported.

### Generated artifacts needed clearer positioning

- Resolution: README now states that CSV, MAT, PDF, and PNG files under local
  `data/` and `figures/` directories are reproducibility snapshots or
  regenerated outputs, not stable public APIs, and may be overwritten by reruns.

### Plot reference lines and complexity labels needed documentation

- Resolution: README now explains that timing-figure reference lines and labels
  such as `O(N)`, `O(N^2 log N)`, and `O(N^3)` are visual guides, not additional
  measured data, and are controlled by the MATLAB plotting scripts.

### Layout and path checks were manual

- Resolution: `tools/audit_layout.jl` now performs a static, non-benchmark
  layout audit for key directories, scripts, root-level generated artifacts, and
  stale README paths.

## Notes

- No `src/` files were modified.
- No full benchmark was run for this audit.
- No CSV, MAT, PDF, or PNG artifacts were regenerated.
- Static checks only read existing files and paths.
