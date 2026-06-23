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
Project.toml                 Julia project environment
Manifest.toml                pinned package manifest
README.md                    project overview and usage notes
src/                         package source code
src/VoltOpt.jl               main module entry point and exports
src/VoltMat/                 Volterra matrix construction routines
src/CompVolt/                composed and nonlinear Volterra actions
src/RestMat/                 coefficient-space restriction matrices
examples/                    reproducible numerical examples and comparisons
examples/ex1_warmup/         low-order warmup comparison scripts
examples/ex2_2ndVolterraEqn/ second-kind smooth Volterra example and comparisons
examples/ex3_pantograph/     pantograph first-kind Volterra example
examples/ex4_oscillatory/    oscillatory-kernel comparison scripts
examples/ex5_psespectra/     Legendre Volterra pseudospectrum experiment
examples/ex6_nonlinear/      nonlinear Volterra--Urysohn examples
examples/ex7_weaksingular/   weakly singular Volterra comparison scripts
test/                        benchmark directories
test/smooth_bench/           smooth-kernel benchmark scripts and plots
test/weaksingular_bench/     weakly singular benchmark scripts and plots
test/nonlinear_bench/        nonlinear Volterra action benchmark scripts and plots
others/                      external/reference comparison code
others/GutlebOlver/          Gutleb--Olver-type comparison implementation
others/FracCollocation/      fractional-collocation comparison implementation
```

Several example and benchmark directories contain local `data/` and `figures/`
subdirectories for generated CSV files, MAT files, PDFs, and PNGs.

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

### Ex1 Warmup

- Directory: `examples/ex1_warmup/`
- Main Julia entry: `examples/ex1_warmup/ex1_run.jl`
- Method scripts: `ex1_error.jl`, `ex1_timing.jl`
- MATLAB plot script: `examples/ex1_warmup/ex1_plot.m`
- Data directory: `examples/ex1_warmup/data/`
- Figures directory: `examples/ex1_warmup/figures/`

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
- Current checked-in artifacts are in `examples/ex5_psespectra/`.
- Current Julia script writes generated CSV/PDF/PNG files under
  `examples/figures/`.

```bash
julia --project=. examples/ex5_psespectra/ex5_volterra_schur_besselj1.jl
matlab -batch "cd('examples/ex5_psespectra'); ex5_plot"
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

### Smooth-Kernel Benchmarks

```bash
julia --project=. test/smooth_bench/bench_end2end_fixedC_varyN.jl
julia --project=. test/smooth_bench/bench_end2end_fixedN_varyC.jl
```

The smooth benchmark scripts write CSV files to:

```text
test/smooth_bench/data/
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
julia --project=. test/nonlinear_bench/bench_nonlinear_volterra_clenshaw_vs_compvolt.jl
```

The nonlinear benchmark script writes CSV files to:

```text
test/nonlinear_bench/data/
```

The MATLAB plotting script reads those CSV files and writes figures to
`test/nonlinear_bench/figures/`:

```bash
matlab -batch "run('test/nonlinear_bench/plot_nonlinear_bench.m')"
```

## Data and Figures

Benchmark CSV data are written to local `data/` subdirectories inside the
corresponding benchmark directories.  Most example CSV data are also written
to local `data/` subdirectories inside the corresponding example directories.
Plotting scripts generally write generated figures to local `figures/`
subdirectories.

The current exception is `examples/ex5_psespectra/`: its checked-in
pseudospectrum artifacts are local to that directory, while the Julia script
currently writes generated output under `examples/figures/`.

Generated data and figures may be overwritten when the benchmark or plotting
scripts are rerun.  Existing checked-in CSV, PDF, PNG, and MAT files should be
treated as generated numerical artifacts rather than source interfaces.

## Notes on Reproducibility

- Run scripts from the project root using `julia --project=.` unless a script
  states otherwise.
- Benchmark scripts use `BenchmarkTools` where appropriate.
- Timing results depend on the machine, BLAS/FFT libraries, Julia compilation
  state, and current system load.
- Large benchmark cases, especially varying-kernel-size comparisons, may take
  significant time.
- MATLAB plotting scripts assume that the corresponding CSV files already exist
  in the local `data/` subdirectory.

## Development Notes

- Do not modify `src/` when only adding examples, benchmarks, or plotting
  scripts.
- Put benchmark CSV output in the benchmark directory's local `data/` folder.
- Put generated benchmark figures in the benchmark directory's local
  `figures/` folder.
- Prefer explicit paths based on `@__DIR__` in Julia scripts.
- Keep generated data paths local to the example or benchmark directory that
  produced them.
