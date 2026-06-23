# Ex5 Volterra Schur Bessel J1

This folder contains the pseudospectrum example for the kernel

```julia
K(x, y) = -im * besselj(1, 2 * (x - y) + x * y)
```

Run the Julia script from the repository root:

```powershell
julia --project=. examples/ex5_psespectra/ex5_volterra_schur_besselj1.jl
```

The Julia script writes:

- data: `examples/ex5_psespectra/data/ex5_volterra_schur_besselj1_pseudospectrum.csv`
- figures: `examples/ex5_psespectra/figures/ex5_volterra_schur_besselj1.pdf`
- figures: `examples/ex5_psespectra/figures/ex5_volterra_schur_besselj1.png`

To regenerate the MATLAB figure from the CSV, run from the repository root:

```powershell
matlab -batch "run('examples/ex5_psespectra/ex5_plot.m')"
```

The MATLAB script reads from `examples/ex5_psespectra/data/` and writes
`ex5_pseudospectrum.pdf` and `ex5_pseudospectrum.png` to
`examples/ex5_psespectra/figures/`.

Julia packages used by the script: `Plots` and `SpecialFunctions`. The rest
are Julia standard libraries.

The local `VoltMat_Leg.jl` file is retained intentionally as an ex5-specific
complex-valued Legendre support copy for reproduction history. The current
Julia entry point uses the project environment and imports
`src/VoltMat/voltmat_leg.jl` from the repository root.
