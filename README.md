# VoltOpt

VoltOpt is a Julia research codebase for fast spectral construction and
application of Volterra integral operators.  The current implementation focuses
on polynomial coefficient representations, fast matrix assembly, and
reproducible numerical experiments for smooth and weakly singular kernels.

The project includes:

- smooth Volterra operator construction in Chebyshev, Legendre, and Jacobi
  polynomial bases;
- weakly singular Volterra construction using mapped Chebyshev/JFP-type bases;
- fast composed and nonlinear Volterra--Urysohn actions;
- numerical examples for Volterra equations and related spectral experiments;
- benchmark scripts, including Gutleb--Olver-type comparisons where applicable.

## Project Structure

```text
VoltOpt/
|_ Project.toml                 Julia project environment
|_ Manifest.toml                pinned package manifest
|_ README.md                    project overview and usage notes
|_ src/                         package source code
|  |_ VoltOpt.jl                main module entry point and exports
|  |_ VoltMat/                  Volterra matrix construction routines
|  |_ CompVolt/                 composed and nonlinear Volterra actions
|  |_ RestMat/                  coefficient-space restriction matrices
|_ examples/                    reproducible numerical examples and comparisons
|  |_ ex1_warmup/               low-order warmup comparison scripts
|  |_ ex2_2ndVolterraEqn/       second-kind smooth Volterra example and comparisons
|  |_ ex3_pantograph/           pantograph first-kind Volterra example
|  |_ ex4_oscillatory/          oscillatory-kernel comparison scripts
|  |_ ex5_psespectra/           Legendre Volterra pseudospectrum experiment
|  |_ ex6_nonlinear/            nonlinear Volterra--Urysohn examples
|  |_ ex7_weaksingular/         weakly singular Volterra comparison scripts
|_ test/                        benchmark and diagnostic directories
|  |_ runtests.jl               lightweight package smoke tests
|  |_ smooth_bench/             smooth-kernel benchmark scripts and plots
|  |_ weaksingular_bench/       weakly singular benchmark scripts and plots
|  |_ compvolt_adaptive/        fixed/adaptive nonlinear compVolt tests, benchmarks, and plots
|  |_ sparse_spy/               Volterra matrix sparsity/magnitude plot workflow
|_ others/                      external/reference comparison code
|  |_ GutlebOlver/              Gutleb--Olver-type comparison implementation
|  |_ FracCollocation/          fractional-collocation comparison implementation
|_ tools/                       static layout and maintenance helpers
```

Several example, benchmark, and diagnostic directories contain local `data/`
and `figures/` subdirectories for generated CSV files, MAT files, PDFs, and
PNGs.

## Main Components

The package entry point is `src/VoltOpt.jl`, which includes and exports the
main routines from the implementation subdirectories.

### Smooth Volterra Matrices

Smooth-kernel Volterra matrix construction is implemented in:

- `src/VoltMat/voltmat_cheb.jl`
- `src/VoltMat/voltmat_leg.jl`
- `src/VoltMat/voltmat_jacobi.jl`

The corresponding exported module aliases and constructors include
`VoltMat_Cheb`, `VoltMat_Leg`, `VoltMat_Jacobi`, `voltmat_cheb`,
`voltmat_cheb_square`, `voltmat_square_from_coeffmat_fft_convolution`,
`voltmat_leg`, and `voltmat_jacobi`.

These routines discretize Volterra operators of the form

```math
(\mathcal{V}_K u)(x) = \int_{-1}^{x} K(x,y)u(y)\,dy
```

in polynomial coefficient bases.

### Weakly Singular Volterra Matrices

Weakly singular construction is implemented in:

- `src/VoltMat/weak_voltmat_cheb.jl`

The exported interface includes `Weak_VoltMat_Cheb`, `weak_voltmat_cheb`,
`PackedLowerHessenberg`, `packed_to_dense`, `phi_beta`, `invphi_beta`, and
`validate_weak_voltmat_columns`.  This component targets kernels with a
separated weak singular factor, for example

```math
\int_{-1}^{x} (x-y)^{-\mu}K(x,y)u(y)\,dy,
\qquad 0 \le \mu < 1.
```

### Restriction and Chebyshev Utilities

Restriction matrices are implemented in:

- `src/RestMat/restmat_cheb.jl`
- `src/RestMat/restmat_jacobi.jl`

Chebyshev transform and interpolation helpers are provided by
`src/chebtechs.jl`, `src/vals2cheb.jl`, and `src/cheb2vals.jl`.

### Composed and Nonlinear Volterra Actions

Fast composed Volterra actions and nonlinear Volterra--Urysohn utilities are in
`src/CompVolt/`.  Exported routines include `compVolt`, `compVolt_fast`,
`compVolt_matrix`, `compVolt_matrix_ultra`, `CompVoltGrid`,
`compVolt_nonlinear_fast`, `compVolt_nonlinear`, and Newton--GMRES helper
functions such as `compVolt_newton_jv!`.

The nonlinear examples in `examples/ex6_nonlinear/` use these routines for
Volterra--Urysohn equations.

## Installation and Environment

VoltOpt is developed with Julia 1.11.1 and uses the project environment defined
by `Project.toml` and `Manifest.toml`.

From the project root:

```bash
julia --project=.
```

On first use, instantiate the environment:

```julia
import Pkg
Pkg.instantiate()
```

Then load the package:

```julia
using VoltOpt
```

## Running Examples

Run example scripts from the project root with the project environment:

```bash
julia --project=. examples/path/to/script.jl
```

Current example directories and recommended commands are:

When an example lists multiple Julia scripts, run them in the order shown before
running the MATLAB plotting script.

### Ex1 Warmup

- Directory: `examples/ex1_warmup/`
- Main Julia entry: `examples/ex1_warmup/ex1_run.jl`
- Method scripts: `ex1_error.jl`, `ex1_timing.jl`
- MATLAB plot script: `examples/ex1_warmup/ex1_plot.m`
- Data directory: `examples/ex1_warmup/data/`
- Figures directory: `examples/ex1_warmup/figures/`
- The `N=1` rows in `ex1_error_N1_100.csv` are expected boundary diagnostics:
  the solvers require `N >= 2`, and the plotting script uses only
  `status=success` rows for formal error and figure output.

```bash
julia --project=. examples/ex1_warmup/ex1_run.jl
matlab -batch "run('examples/ex1_warmup/ex1_plot.m')"
```

### Ex2 Second-Kind Volterra Equation

- Directory: `examples/ex2_2ndVolterraEqn/`
- Main Julia entry: `examples/ex2_2ndVolterraEqn/ex2_run.jl`
- Method scripts: `ex2_VoltMat.jl`, `ex2_GO.jl`
- MATLAB plot script: `examples/ex2_2ndVolterraEqn/ex2_plot.m`
- Data directory: `examples/ex2_2ndVolterraEqn/data/`
- Figures directory: `examples/ex2_2ndVolterraEqn/figures/`

```bash
julia --project=. examples/ex2_2ndVolterraEqn/ex2_run.jl
matlab -batch "run('examples/ex2_2ndVolterraEqn/ex2_plot.m')"
```

### Ex3 Pantograph

- Directory: `examples/ex3_pantograph/`
- Main Julia entry: `examples/ex3_pantograph/ex3_run.jl`
- Secondary Julia script: `examples/ex3_pantograph/ex3_pantograph_firstkind.jl`
- MATLAB plot script: `examples/ex3_pantograph/ex3_plot.m`
- Data directory for `ex3_run.jl`: `examples/ex3_pantograph/data/`
- Figures directory for `ex3_plot.m`: `examples/ex3_pantograph/figures/`

```bash
julia --project=. examples/ex3_pantograph/ex3_run.jl
matlab -batch "run('examples/ex3_pantograph/ex3_plot.m')"
```

### Ex4 Oscillatory Kernel

- Directory: `examples/ex4_oscillatory/`
- Julia scripts: `examples/ex4_oscillatory/ex4_VoltMat.jl`,
  `examples/ex4_oscillatory/ex4_GO.jl`
- MATLAB plot script: `examples/ex4_oscillatory/ex4_plot.m`
- Data directory: `examples/ex4_oscillatory/data/`
- Figures directory: `examples/ex4_oscillatory/figures/`

```bash
julia --project=. examples/ex4_oscillatory/ex4_VoltMat.jl
julia --project=. examples/ex4_oscillatory/ex4_GO.jl
matlab -batch "run('examples/ex4_oscillatory/ex4_plot.m')"
```

### Ex5 Pseudospectra

- Directory: `examples/ex5_psespectra/`
- Julia script: `examples/ex5_psespectra/ex5_volterra_schur_besselj1.jl`
- MATLAB plot script: `examples/ex5_psespectra/ex5_plot.m`
- Data directory: `examples/ex5_psespectra/data/`
- Figures directory: `examples/ex5_psespectra/figures/`
- The Julia script writes pseudospectrum CSV data to local `data/` and its
  generated PDF/PNG figure to local `figures/`.
- The MATLAB plot script reads the local CSV data and writes PDF/PNG figures
  to local `figures/`.

```bash
julia --project=. examples/ex5_psespectra/ex5_volterra_schur_besselj1.jl
matlab -batch "run('examples/ex5_psespectra/ex5_plot.m')"
```

### Ex6 Nonlinear Volterra-Urysohn

- Directory: `examples/ex6_nonlinear/`
- Julia scripts: `examples/ex6_nonlinear/ex6_nonlinear_picard.jl`,
  `examples/ex6_nonlinear/ex6_fixed10_varyN.jl`
- MATLAB plot script: `examples/ex6_nonlinear/ex6_plot.m`
- Data directory: `examples/ex6_nonlinear/data/`
- Figures directory: `examples/ex6_nonlinear/figures/`

```bash
julia --project=. examples/ex6_nonlinear/ex6_nonlinear_picard.jl
julia --project=. examples/ex6_nonlinear/ex6_fixed10_varyN.jl
matlab -batch "run('examples/ex6_nonlinear/ex6_plot.m')"
```

### Ex7 Weakly Singular Volterra Equation

- Directory: `examples/ex7_weaksingular/`
- Julia scripts: `examples/ex7_weaksingular/ex7_VoltMat.jl`,
  `examples/ex7_weaksingular/ex7_FracCollocation.jl`
- MATLAB plot script: `examples/ex7_weaksingular/ex7_plot.m`
- Data directory: `examples/ex7_weaksingular/data/`
- Figures directory: `examples/ex7_weaksingular/figures/`

```bash
julia --project=. examples/ex7_weaksingular/ex7_VoltMat.jl
julia --project=. examples/ex7_weaksingular/ex7_FracCollocation.jl
matlab -batch "run('examples/ex7_weaksingular/ex7_plot.m')"
```

## Running Benchmarks

Benchmark scripts should be run from the project root with `--project=.`.
The `test/` directory currently stores benchmark scripts and plotting helpers;
`Pkg.test()` only runs lightweight smoke tests. Benchmark scripts remain manual
because they can take substantially longer and overwrite generated artifacts.

Benchmark CSV schemas are directory-specific. Smooth, weakly singular, and
nonlinear benchmarks are read by their corresponding MATLAB plotting scripts
rather than through a single shared CSV schema.

### Smooth-Kernel Benchmarks

```bash
julia --project=. test/smooth_bench/bench_end2end_fixedC_varyN.jl
julia --project=. test/smooth_bench/bench_end2end_fixedN_varyC.jl
```

The smooth benchmark scripts write CSV files to:

```text
test/smooth_bench/data/
```

Older root-level smooth CSV snapshots with the previous aggregate schema are
kept under:

```text
test/smooth_bench/legacy/
```

The MATLAB plotting script reads those CSV files and writes figures to
`test/smooth_bench/figures/`:

```bash
matlab -batch "run('test/smooth_bench/plot_end2end_smooth_bench.m')"
```

### Weakly Singular Benchmarks

```bash
julia --project=. test/weaksingular_bench/bench_weak_fixedC_varyN.jl
julia --project=. test/weaksingular_bench/bench_weak_fixedN_varyC.jl
```

The weakly singular benchmark scripts write CSV files to:

```text
test/weaksingular_bench/data/
```

The MATLAB plotting script reads those CSV files and writes figures to
`test/weaksingular_bench/figures/`:

```bash
matlab -batch "run('test/weaksingular_bench/plot_weaksingular_bench.m')"
```

### Nonlinear Volterra Action Benchmarks

```bash
julia --project=. test/compvolt_adaptive/bench_nonlinear_volterra_clenshaw_vs_compvolt.jl
```

The nonlinear benchmark script writes CSV files to:

```text
test/compvolt_adaptive/data/
```

This benchmark directly includes `src/CompVolt/clenshaw_valuemat.jl` as an
internal benchmark helper for comparison; that helper is not currently part of
the exported package API.

The MATLAB plotting script reads those CSV files and writes figures to
`test/compvolt_adaptive/figures/`:

```bash
matlab -batch "run('test/compvolt_adaptive/plot_nonlinear_bench.m')"
```

The same directory contains the adaptive-grid regression and oscillation
diagnostic:

```bash
julia --project=. test/compvolt_adaptive/runtests.jl
julia --project=. test/compvolt_adaptive/run_adaptive_oscillation.jl
matlab -batch "run('test/compvolt_adaptive/plot_adaptive_oscillation.m')"
```

Adaptive CSV data, figures, and the experiment report are stored in the local
`data/`, `figures/`, and `report/` subdirectories.

### Sparse Volterra Matrix Spy Diagnostics

The `test/sparse_spy/` directory stores a two-stage diagnostic workflow for
colored sparsity plots of Volterra coefficient matrices in Chebyshev,
Legendre, and Jacobi bases. It is not a benchmark sweep.

The Julia script constructs the matrices and writes a MATLAB data file:

```bash
julia --project=. test/sparse_spy/spy_VoltMat.jl
```

Output:

```text
test/sparse_spy/data/spy_VoltMat.mat
```

The MATLAB script reads that MAT file, adds the machine-epsilon perturbation
used by the legacy Julia spy plot, and writes PDF/PNG figures:

```bash
matlab -batch "run('test/sparse_spy/spy_VoltMat.m')"
```

Output:

```text
test/sparse_spy/figures/
```

## Data and Figures

CSV, MAT, PDF, and PNG files under `examples/*/data/`,
`examples/*/figures/`, `test/*_bench/data/`, `test/*_bench/figures/`,
`test/sparse_spy/data/`, and `test/sparse_spy/figures/` are reproducibility
snapshots or regenerated outputs, not stable public APIs. Rerunning the
corresponding Julia or MATLAB scripts may overwrite them.

The current layout convention is:

- Julia example and benchmark scripts write CSV/MAT data to local `data/`
  subdirectories.
- MATLAB plotting scripts and Julia figure-generation scripts write figures to
  local `figures/` subdirectories.
- Diagnostic scripts under `test/sparse_spy/` follow the same local `data/`
  and `figures/` convention.
- Older retained snapshots should live in local `legacy/` subdirectories.

Some plotting scripts add tail reference lines or complexity labels to timing
figures, for example visual guides such as `O(N)`, `O(N^2 log N)`, or `O(N^3)`.
These guides are not additional measured data. Their placement and scaling are
controlled by the reference-line settings in the corresponding MATLAB scripts,
including `test/smooth_bench/plot_end2end_smooth_bench.m`,
`test/compvolt_adaptive/plot_nonlinear_bench.m`, and
`examples/ex4_oscillatory/ex4_plot.m`.

## Notes on Reproducibility

- Run scripts from the project root using `julia --project=.` unless a script
  states otherwise.
- Benchmark scripts use `BenchmarkTools` where appropriate.
- Timing results depend on the machine, BLAS/FFT libraries, Julia compilation
  state, and current system load.
- Large benchmark cases, especially varying-kernel-size comparisons, may take
  significant time.
- MATLAB plotting scripts assume that the corresponding CSV or MAT files
  already exist in the local `data/` subdirectory.
- Run `julia --project=. -e "using Pkg; Pkg.test()"` for lightweight package
  smoke tests only. This does not run the benchmark scripts.

## Development Notes

- Do not modify `src/` when only adding examples, benchmarks, or plotting
  scripts.
- Put benchmark CSV output in the benchmark directory's local `data/` folder.
- Put generated benchmark figures in the benchmark directory's local
  `figures/` folder.
- Put diagnostic MAT output and generated diagnostic figures in the diagnostic
  directory's local `data/` and `figures/` folders.
- Prefer explicit paths based on `@__DIR__` in Julia scripts.
- Keep generated data paths local to the example or benchmark directory that
  produced them.
- Run the static layout audit with:

```bash
julia --project=. tools/audit_layout.jl
```

The audit checks expected directories, key scripts, root-level generated
artifacts, and stale README paths. It does not run examples or benchmarks.
