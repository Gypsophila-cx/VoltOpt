# Issue List

Last updated: 2026-06-23

## Summary

- P0: 0 current blocking issues.
- P1: 0 current high-priority reproducibility issues.
- P2: 0 current medium-priority documentation/path/schema issues.
- P3: 4 low-priority cleanup and future-maintenance issues.

This audit is based on the current filesystem state in `D:\Projects\VoltOpt`
and the current working tree. Historical tracked deletions shown by
`git status` are not treated as runnable-script blockers when the files no
longer exist on disk.

## P0 - Blocking issues

No current P0 issues are open.

## P1 - High-priority issues

No current P1 issues are open after the warmup boundary behavior was documented
and the smooth benchmark root CSV snapshots were moved to `legacy/`.

## P2 - Medium-priority issues

No current P2 issues are open after the remaining low-risk documentation and
comment updates in the working tree.

## P3 - Low-priority cleanup / documentation issues

### P3-1: Generated artifacts are checked in and may be overwritten

- Location: `examples/*/data`, `examples/*/figures`, `test/*_bench/data`,
  `test/*_bench/figures`
- Evidence: Current tree contains many generated CSV, MAT, PDF, and PNG files.
- Impact: Useful for reproducibility snapshots, but they are not source
  interfaces and can change after reruns.
- Suggested fix: Document which artifacts are source-of-truth snapshots and
  which are regenerated outputs.
- Status: Open.

### P3-2: Some plotting scripts have richer reference-line annotations than README describes

- Location:
  - `test/smooth_bench/plot_end2end_smooth_bench.m`
  - `test/nonlinear_bench/plot_nonlinear_bench.m`
  - `examples/ex4_oscillatory/ex4_plot.m`
- Evidence: Current scripts draw tail reference lines and complexity labels,
  while README only describes basic plotting locations.
- Impact: Low; plots are still reproducible, but figure semantics are not
  documented in detail.
- Suggested fix: Add a short note in benchmark/example docs if these reference
  lines are part of the intended published figures.
- Status: Open.

### P3-3: Lightweight audit checks are manual

- Location: repository maintenance workflow
- Evidence: Recent updates used manual static checks for path existence,
  include references, README paths, ex5 root artifacts, CSV status summaries,
  and schema headers.
- Impact: Path drift can recur when examples are renamed or reorganized.
- Suggested fix: Add a lightweight non-benchmark path/schema audit script in a
  future task.
- Status: Open.

### P3-4: Optional package smoke-test entry is still absent

- Location: `test/runtests.jl`
- Evidence: Current filesystem does not contain `test/runtests.jl`.
- Impact: README now documents that `test/` is currently a benchmark
  collection rather than a conventional `Pkg.test()` suite, so this is not a
  current blocker. A future lightweight smoke test would still improve package
  maintenance.
- Suggested fix: Add a minimal `test/runtests.jl` in a separate task if the
  project should support `Pkg.test()`.
- Status: Open as optional future work.

## Release / Staging Notes

- Current directory migrations still need an intentional staging pass:
  - tracked deletions under `examples/ex1_2ndVolterraEqn/`,
    `examples/ex2_1stVolterraEqn/`, and `examples/warmup/`;
  - untracked replacement directories such as `examples/ex1_warmup/`,
    `examples/ex2_2ndVolterraEqn/`, and `test/nonlinear_bench/`.
- Ex5 artifact moves still appear as deleted root files plus untracked files
  under `examples/ex5_psespectra/data/` and
  `examples/ex5_psespectra/figures/`.
- Smooth benchmark legacy CSV moves appear as deleted root CSV files plus
  untracked files under `test/smooth_bench/legacy/`.
- These are release/staging concerns rather than open code or documentation
  defects. Stage both sides of each move together in a follow-up commit.

## Resolved in working tree

### README listed missing core example directories and run commands

- Resolution: The top-level README has been updated in the working tree to use
  current paths such as `examples/ex1_warmup/`,
  `examples/ex2_2ndVolterraEqn/`, and `test/nonlinear_bench/`.
- Remaining follow-up: Stage the README change with the rest of the intended
  documentation updates.

### README did not document current supported examples and nonlinear benchmark

- Resolution: The top-level README now documents the current supported example
  and benchmark directories, including `test/nonlinear_bench/`.
- Remaining follow-up: None beyond staging.

### Top-level README described ex5 as a data/figures exception

- Resolution: The README now documents ex5 as using local
  `examples/ex5_psespectra/data/` and `examples/ex5_psespectra/figures/`, and
  the MATLAB command uses `run('examples/ex5_psespectra/ex5_plot.m')`.
- Remaining follow-up: None beyond staging.

### Ex5 scripts, README, and artifacts disagreed on paths

- Resolution:
  - `examples/ex5_psespectra/ex5_volterra_schur_besselj1.jl` now writes CSV
    output to local `data/` and PDF/PNG output to local `figures/`.
  - `examples/ex5_psespectra/ex5_plot.m` now reads from local `data/` and
    writes PDF/PNG output to local `figures/`.
  - `examples/ex5_psespectra/README.md` now documents the local
    `data/`/`figures/` workflow.
  - Existing ex5 root CSV/PDF artifacts were moved into local `data/` or
    `figures/`.
- Remaining follow-up: Stage the old deleted artifact paths and new artifact
  paths together.

### Ex3 had two current scripts with different artifact layouts

- Resolution: `examples/ex3_pantograph/ex3_pantograph_firstkind.jl` now uses
  local `data/` for CSV output and local `figures/` for PDF/PNG output, matching
  the layout already used by `ex3_run.jl` and `ex3_plot.m`.
- Remaining follow-up: Full `ex3_pantograph_firstkind.jl` regeneration was not
  run during the issue-list update.

### Warmup `N=1` rows were reported as a P1 failure

- Resolution:
  - README now states that the `N=1` rows in
    `examples/ex1_warmup/data/ex1_error_N1_100.csv` are expected boundary
    diagnostics.
  - The solvers require `N >= 2`.
  - `examples/ex1_warmup/ex1_plot.m` filters to `status=success` rows for
    formal error and figure output.
  - A static CSV check found `297` success rows, `3` failed rows, and `0`
    failed rows with `N >= 2`.
- Remaining follow-up: None unless maintainers later decide to remove the
  `N=1` diagnostic rows from the dataset.

### Smooth benchmark root CSV artifacts were treated as current outputs

- Resolution:
  - `test/smooth_bench/bench_end2end_fixedC_varyN.csv` and
    `test/smooth_bench/bench_end2end_fixedN_varyC.csv` were moved to
    `test/smooth_bench/legacy/`.
  - README now states that current smooth CSV outputs are under
    `test/smooth_bench/data/` and older root-level snapshots are kept under
    `test/smooth_bench/legacy/`.
- Remaining follow-up: Stage the root deletions and legacy additions together.

### `test/` had no conventional package test entry

- Resolution: README now states that `test/` currently stores benchmark scripts
  and plotting helpers, not a conventional `Pkg.test()` suite, and that
  `test/runtests.jl` is not currently provided.
- Remaining follow-up: Optional future smoke test tracked as P3-4.

### Benchmark CSV schemas were not aligned across benchmark directories

- Resolution: README now states that smooth, weakly singular, and nonlinear
  benchmark CSV schemas are directory-specific and are consumed by their
  corresponding plotting scripts rather than a single shared schema.
- Remaining follow-up: None unless a unified schema is desired later.

### Example runner coverage was inconsistent

- Resolution: README now says that examples with multiple Julia scripts should
  run them in the listed order before running the MATLAB plotting script. The
  README already lists each example's available Julia and MATLAB entry points.
- Remaining follow-up: None unless maintainers want unified `ex*_run.jl`
  wrappers for every example.

### Ex4 plot comments mentioned old build-time file names

- Resolution: `examples/ex4_oscillatory/ex4_plot.m` header comments now list
  the actual output names `ex4_depth_time_N200.pdf` and
  `ex4_depth_time_N200.png`.
- Remaining follow-up: None.

### Ex6 comments referenced an obsolete lowercase path without naming the canonical path

- Resolution:
  - `examples/ex6_nonlinear/ex6_nonlinear_picard.jl` now names
    `src/CompVolt/compVolt.jl` as the canonical path and describes
    `src/compVolt.jl` as legacy compatibility.
  - `examples/ex6_nonlinear/ex6_fixed10_varyN.jl` has the same canonical-path
    clarification.
- Remaining follow-up: None unless maintainers later remove the legacy
  compatibility lookup.

### Ex5 local copied source file needed a clearer note

- Resolution: `examples/ex5_psespectra/README.md` now states that
  `VoltMat_Leg.jl` is retained intentionally as an ex5-specific complex-valued
  Legendre support copy for reproduction history, while the current Julia entry
  point imports `src/VoltMat/voltmat_leg.jl`.
- Remaining follow-up: None unless maintainers later decide to archive or
  remove the local reproduction file.

### Nonlinear benchmark uses an internal helper outside the exported package API

- Resolution: README now documents that
  `test/nonlinear_bench/bench_nonlinear_volterra_clenshaw_vs_compvolt.jl`
  directly includes `src/CompVolt/clenshaw_valuemat.jl` as an internal
  benchmark helper and that the helper is not currently exported.
- Remaining follow-up: None unless maintainers want to make the helper public.

## Notes

- No `src/` files were modified for these issue-list updates.
- No full benchmark was run for this audit.
- No CSV data were regenerated.
- The smooth root CSV files were moved, not deleted.
- Static checks read existing CSV data only.
