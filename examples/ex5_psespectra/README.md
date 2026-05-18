# Ex5 Volterra Schur Bessel J1

This folder is a self-contained reproduction script for the kernel

```julia
K(x, y) = -im * besselj(1, 2 * (x - y) + x * y)
```

Run from this folder:

```powershell
julia ex5_volterra_schur_besselj1.jl
```

Julia packages used by the script: `Plots` and `SpecialFunctions`. The rest are Julia standard libraries.

Outputs:

- `ex5_volterra_schur_besselj1_pseudospectrum.csv`
- `ex5_volterra_schur_besselj1.pdf`

The local `VoltMat_Leg.jl` is copied from `../op/VoltMat_Leg.jl` and extended in this folder only so that complex-valued Legendre interpolation and complex coefficient matrices are accepted directly.
