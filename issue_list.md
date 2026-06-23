# Issue List

Last updated: 2026-06-23

## Summary

- P0: 1 current blocking documentation/path issue.
- P1: 6 high-priority reproducibility and release-surface issues.
- P2: 8 medium-priority consistency, API-boundary, and schema issues.
- P3: 5 low-priority cleanup and maintenance issues.

This audit is based on the current filesystem state in `D:\Projects\VoltOpt`.
Historical tracked deletions shown by `git status` are not treated as current
runnable-script blockers when the files no longer exist on disk.

## P0 - Blocking issues

### P0-1: README lists missing core example directories and run commands

- Location: `README.md`
- Evidence:
  - README project structure lists `examples/ex1_2ndVolterraEqn/`,
    `examples/ex2_1stVolterraEqn/`, and `examples/warmup/`.
  - Current filesystem has none of those directories.
  - Current filesystem instead has `examples/ex1_warmup/` and
    `examples/ex2_2ndVolterraEqn/`.
  - README representative commands still include:
    - `julia --project=. examples/ex1_2ndVolterraEqn/ex1_run.jl`
    - `julia --project=. examples/ex1_2ndVolterraEqn/ex1_VoltMat.jl`
    - `julia --project=. examples/ex1_2ndVolterraEqn/ex1_GO.jl`
    - `julia --project=. examples/ex2_1stVolterraEqn/ex2_firstkind_direct_vs_differentiated.jl`
  - The README MATLAB example text also points to
    `examples/ex1_2ndVolterraEqn/ex1_plot.m`, which is missing.
- Impact: Users following the top-level README hit missing-path failures for
  major example commands.
- Suggested fix: Update README project structure and representative commands
  to the current supported directories, or restore the old directories if they
  are still intended to be public.
- Status: Open. Maintainer confirmation needed because the old directories are
  tracked deletions while the replacement directories are currently untracked.

## P1 - High-priority issues

### P1-1: Current directory migration is not staged as renames/additions

- Location: worktree status for `examples/` and `test/`
- Evidence:
  - `git status --short` shows many deleted tracked files under
    `examples/ex1_2ndVolterraEqn/`, `examples/ex2_1stVolterraEqn/`, and
    `examples/warmup/`.
  - It also shows untracked current directories:
    - `examples/ex1_warmup/`
    - `examples/ex2_2ndVolterraEqn/`
    - `test/nonlinear_bench/`
- Impact: The filesystem can run the new layout locally, but a commit/release
  can accidentally omit the new directories or present the move as unrelated
  deletion plus unrelated addition.
- Suggested fix: Decide whether the new layout is official, then stage it as
  intentional renames/additions with a clear cleanup commit.
- Status: Open. Maintainer confirmation required.

### P1-2: README does not document current supported examples and nonlinear benchmark

- Location: `README.md`
- Evidence:
  - README does not mention `examples/ex1_warmup/`.
  - README does not mention `examples/ex2_2ndVolterraEqn/`.
  - README benchmark section documents smooth and weakly singular benchmarks,
    but not `test/nonlinear_bench/`.
- Impact: The current runnable surface is not discoverable from README, while
  stale paths are still advertised.
- Suggested fix: Add a supported examples/benchmarks table based on the current
  filesystem, including `test/nonlinear_bench/bench_nonlinear_volterra_clenshaw_vs_compvolt.jl`
  and `test/nonlinear_bench/plot_nonlinear_bench.m`.
- Status: Open.

### P1-3: Ex5 Julia script, MATLAB plot script, README, and artifacts disagree on paths

- Location: `examples/ex5_psespectra/`
- Evidence:
  - `ex5_volterra_schur_besselj1.jl` writes CSV/PDF/PNG to
    `joinpath(PROJECT_ROOT, "examples", "figures")`.
  - `ex5_plot.m` reads `ex5_volterra_schur_besselj1_pseudospectrum.csv` from
    the current MATLAB working directory and writes `ex5_pseudospectrum.pdf`
    locally.
  - Current existing ex5 CSV/PDF artifacts are in the ex5 directory root.
  - `examples/ex5_psespectra/README.md` says to run from the ex5 folder and
    lists local outputs, while root README says to run the Julia script from
    the project root.
- Impact: Running the Julia script from root does not naturally feed the
  MATLAB plot script, and generated artifacts can be split across locations.
- Suggested fix: Normalize ex5 to local `data/` and `figures/`, or document a
  single official workflow and update both scripts and README together.
- Status: Open.

### P1-4: Ex3 has two current scripts with different artifact layouts

- Location: `examples/ex3_pantograph/`
- Evidence:
  - `ex3_run.jl` writes `ex3_error_time.csv` and `ex3_A_Nmax.mat` to local
    `examples/ex3_pantograph/data/`.
  - `ex3_plot.m` reads local `data/` and writes local `figures/`.
  - `ex3_pantograph_firstkind.jl` still sets
    `FIGURES_DIR = joinpath(VOLTERRA_ROOT, "examples", "figures")` and writes
    CSV/PDF/PNG outputs through that top-level folder.
- Impact: Two README-listed ex3 entry scripts produce incompatible artifact
  layouts.
- Suggested fix: Decide whether `ex3_pantograph_firstkind.jl` is official.
  Normalize it to local `data/`/`figures/` or document/archive it as secondary.
- Status: Open.

### P1-5: Current warmup error CSV contains failed rows

- Location: `examples/ex1_warmup/data/ex1_error_N1_100.csv`
- Evidence:
  - Static CSV status check found `297` rows with `status=success` and `3`
    rows with `status=failed`.
  - The failed rows are the three methods at `N=1`, with messages such as
    `VoltMat warmup solve requires N >= 2` and
    `Gutleb--Olver warmup solve requires N >= 2`.
- Impact: The example runner can complete and plots can filter success rows,
  but the refreshed "complete" dataset includes known failures.
- Suggested fix: Decide whether `N=1` is an intentional edge-case diagnostic.
  If not, start the sweep at `N=2`; if yes, document that the first three rows
  are expected failures.
- Status: Open. Maintainer decision needed because changing the N list changes
  the benchmark definition.

### P1-6: Root-level generated artifacts remain in official benchmark/example directories

- Location:
  - `test/smooth_bench/bench_end2end_fixedC_varyN.csv`
  - `test/smooth_bench/bench_end2end_fixedN_varyC.csv`
  - `examples/ex5_psespectra/ex5_volterra_schur_besselj1_pseudospectrum.csv`
  - `examples/ex5_psespectra/ex5_volterra_schur_besselj1.pdf`
  - `examples/ex5_psespectra/ex5_pseudospectrum.pdf`
- Evidence:
  - Current smooth scripts write current CSVs to `test/smooth_bench/data/`.
  - Current smooth root CSVs use an older aggregate schema with columns such as
    `cheb_min_ms` and `gutleb_min_ms`.
  - Ex5 has no local `data/` or `figures/`, so generated files are mixed with
    source scripts.
- Impact: It is easy to pick stale or non-source artifacts as current data.
- Suggested fix: Mark these as legacy artifacts or move them in a separate
  cleanup task after maintainer approval.
- Status: Open. Do not delete without confirmation.

## P2 - Medium-priority issues

### P2-1: `clenshaw_valuemat.jl` is used by a benchmark but not exposed by `VoltOpt`

- Location: `src/CompVolt/clenshaw_valuemat.jl` and `test/nonlinear_bench/`
- Evidence:
  - `test/nonlinear_bench/bench_nonlinear_volterra_clenshaw_vs_compvolt.jl`
    directly includes `src/CompVolt/clenshaw_valuemat.jl`.
  - `src/VoltOpt.jl` does not include or export `clenshaw_valuemat`.
- Impact: The benchmark depends on an internal helper that is not part of the
  package's public interface. That may be intentional, but the boundary is not
  documented.
- Suggested fix: Either export/document it as an internal benchmark helper, or
  keep the direct include and add a comment explaining why it is not public.
- Status: Open.

### P2-2: `test/` has no conventional package test entry

- Location: `test/runtests.jl`
- Evidence: Current filesystem does not contain `test/runtests.jl`.
- Impact: `Pkg.test()` is not a useful smoke-test entry; `test/` currently acts
  as a benchmark collection.
- Suggested fix: Add a lightweight `test/runtests.jl` smoke test in a future
  task, or state in README that `test/` contains benchmark scripts only.
- Status: Open.

### P2-3: Benchmark CSV schemas are not aligned across smooth, weak, and nonlinear benches

- Location:
  - `test/smooth_bench/data/bench_smooth_*.csv`
  - `test/weaksingular_bench/data/bench_weak_*.csv`
  - `test/nonlinear_bench/data/bench_nonlinear_volterra_clenshaw_vs_compvolt.csv`
- Evidence:
  - Smooth CSVs include `status`, `method_status`, `t_total_median`, bandwidth
    diagnostics, and error-message columns.
  - Weak CSVs have a smaller schema.
  - Nonlinear CSV has method, timing, allocation, memory, and check-norm
    columns, but no `status` column.
- Impact: Combined benchmark summaries need per-directory adapters.
- Suggested fix: Define a minimal common benchmark schema for future scripts
  while preserving existing data.
- Status: Open.

### P2-4: Example runner coverage is inconsistent

- Location: `examples/`
- Evidence:
  - `examples/ex1_warmup/` and `examples/ex2_2ndVolterraEqn/` have `ex*_run.jl`
    runners.
  - `examples/ex4_oscillatory/`, `examples/ex6_nonlinear/`, and
    `examples/ex7_weaksingular/` use separate method scripts and plot scripts
    but no unified runner.
- Impact: Users need to know which scripts to run and in what order for each
  example.
- Suggested fix: Either add runners consistently or document the intended order
  per example in README.
- Status: Open.

### P2-5: Ex4 plot comments mention old build-time file names

- Location: `examples/ex4_oscillatory/ex4_plot.m`
- Evidence:
  - Header comments list `figures/ex4_depth_build_time_N200.pdf` and `.png`.
  - The script actually saves `ex4_depth_time_N200.pdf` and `.png`.
- Impact: Minor documentation mismatch for figure outputs.
- Suggested fix: Update comments to match actual output names.
- Status: Open.

### P2-6: Ex6 scripts keep compatibility code for an obsolete lowercase path

- Location:
  - `examples/ex6_nonlinear/ex6_nonlinear_picard.jl`
  - `examples/ex6_nonlinear/ex6_fixed10_varyN.jl`
- Evidence:
  - Both scripts search for both `src/compVolt.jl` and
    `src/CompVolt/compVolt.jl`.
  - Current filesystem only has `src/CompVolt/compVolt.jl`.
- Impact: Not a runtime failure, but comments and errors reference an obsolete
  path that can confuse maintenance.
- Suggested fix: Keep compatibility if needed, but update comments to emphasize
  the current canonical path.
- Status: Open.

### P2-7: Ex5 carries a local copied source file

- Location: `examples/ex5_psespectra/VoltMat_Leg.jl`
- Evidence:
  - Local README says the file is copied and extended for complex-valued
    Legendre interpolation.
  - The note references `../op/VoltMat_Leg.jl`, which is not a current
    repository path.
- Impact: Local code can diverge from `src/VoltMat/voltmat_leg.jl`, and the
  provenance note is stale.
- Suggested fix: Document why the local copy remains necessary, or factor the
  complex-valued support into `src/` in a separate, explicit task.
- Status: Open.

### P2-8: README still describes local data/figures as a general convention with exceptions

- Location: `README.md`
- Evidence:
  - README says several directories contain local `data/` and `figures/`.
  - Most current examples now follow this, but `examples/ex5_psespectra/` still
    stores generated artifacts in its root and its Julia script writes to
    top-level `examples/figures/`.
- Impact: Documentation suggests stronger consistency than the tree currently
  provides.
- Suggested fix: Either normalize ex5 or document it as an explicit exception.
- Status: Open.

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

### P3-2: Historical deletions need archive/cleanup confirmation

- Location: git status entries under:
  - `examples/ex1_2ndVolterraEqn/`
  - `examples/ex2_1stVolterraEqn/`
  - `examples/warmup/`
- Evidence: These paths no longer exist in the current filesystem but appear as
  tracked deletions.
- Impact: Not current runnable-script blockers, but the cleanup needs an
  intentional commit or restoration.
- Suggested fix: Confirm with the maintainer whether the deletion is intended.
- Status: Open.

### P3-3: Some plotting scripts have richer reference-line annotations than README describes

- Location:
  - `test/smooth_bench/plot_end2end_smooth_bench.m`
  - `test/nonlinear_bench/plot_nonlinear_bench.m`
  - `examples/ex4_oscillatory/ex4_plot.m`
- Evidence: Current scripts draw tail reference lines and complexity labels,
  while README only describes basic plotting locations.
- Impact: Low; plots are still reproducible, but figure semantics are not
  documented.
- Suggested fix: Add a short note in benchmark/example docs if these reference
  lines are part of the intended figures.
- Status: Open.

### P3-4: `issue_list.md` itself is currently untracked

- Location: `issue_list.md`
- Evidence: `git status --short` reports `?? issue_list.md`.
- Impact: The updated issue list will not be included unless explicitly added.
- Suggested fix: Add `issue_list.md` to git if it is intended to be part of the
  repository.
- Status: Open.

### P3-5: Lightweight audit checks are manual

- Location: repository maintenance workflow
- Evidence: This update used manual static checks for path existence, include
  references, README paths, and CSV status summaries.
- Impact: Path drift can recur when examples are renamed or reorganized.
- Suggested fix: Add a lightweight non-benchmark path/schema audit script in a
  future task.
- Status: Open.

## Notes

- This update only modified `issue_list.md`.
- No `src/` files were modified.
- No `examples/` files were modified.
- No `test/` files were modified.
- No full benchmark was run for this audit.
- A lightweight `julia --project=. -e "using VoltOpt"` load check was run and
  succeeded after two failed attempts caused by PowerShell quoting, not by
  project code.
- Static CSV checks read existing data only:
  - `examples/ex1_warmup/data/ex1_error_N1_100.csv`: 297 success, 3 failed.
  - `examples/ex2_2ndVolterraEqn/data/ex2_*_error.csv`: 100 ok rows each.
  - `test/smooth_bench/data/bench_smooth_*.csv`: all rows status success.
  - `test/weaksingular_bench/data/bench_weak_*.csv`: all rows status success.
  - `test/nonlinear_bench/data/bench_nonlinear_volterra_clenshaw_vs_compvolt.csv`:
    7 `clenshaw_valuemat` rows and 10 `compVolt` rows.
