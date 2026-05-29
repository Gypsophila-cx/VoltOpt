# VoltOpt

VoltOpt is a Julia codebase for fast spectral discretization of Volterra integral
operators and Volterra-type equations.  The implementation centers on polynomial
bases, restriction matrices, and fast assembly/application routines for kernels
on triangular Volterra domains.

## Main Features

- Chebyshev utilities:
  - value/coefficients transforms with DCT-I/FFT,
  - two-dimensional Chebyshev interpolation helpers,
  - Chebyshev integration and multiplication matrices.
- Restriction matrices:
  - `restmat_cheb(a, b, N)` for Chebyshev coefficients,
  - `restmat_jacobi(a, b, N, alpha, beta)` for Jacobi coefficients.
- Volterra matrix assembly:
  - `voltmat_cheb` / `VoltMat_Cheb.voltmat_cheb` for Chebyshev-basis square Volterra matrices,
  - `weak_voltmat_cheb` / `Weak_VoltMat_Cheb.weak_voltmat_cheb` for weakly singular Chebyshev/JFP Volterra matrices,
  - `voltmat_leg` / `VoltMat_Leg.voltmat_leg` for Legendre-basis matrices,
  - `voltmat_jacobi` / `VoltMat_Jacobi.voltmat_jacobi` for Jacobi-basis matrices.
- Fast composed/nonlinear Volterra actions:
  - `compVolt`, `compVolt_fast`,
  - `compVolt_nonlinear_fast`,
  - Newton-GMRES matrix-free helpers such as `compVolt_newton_jv!`.
- Example scripts for first-kind and second-kind Volterra equations, stiff
  integro-differential equations, pseudospectra, and nonlinear Volterra-Urysohn
  Picard iteration.

## Package Usage

From the repository root:

```powershell
julia --project=.
```

Then in Julia:

```julia
using VoltOpt

K(x, t) = exp(-(x - t)^2)

B_cheb = voltmat_cheb(K, 32)
B_leg  = voltmat_leg(K, 32)
R      = restmat_cheb(-1.0, 0.0, 32)
```

For weakly singular Volterra kernels, VoltOpt also supports operators of the
form

```math
(\mathcal{V}_K u)(x)
=
\int_{-1}^{x}
(x-y)^{-\mu} K(x,y) u(y)\,dy,
```

where `K(x,y)` is smooth and the algebraic singularity is handled in a
Chebyshev/JFP-type weighted basis.  The constructor lives in
`src/VoltMat/weak_voltmat_cheb.jl` and is exported as `weak_voltmat_cheb`:

```julia
using VoltOpt

K(x, y) = exp(-x * y)

V, info = weak_voltmat_cheb(
    K;
    mu = 0.25,
    sigma = 0.0,
    beta = 1.25,
    N = 32,
)
```

You can also access the implementation modules directly:

```julia
using VoltOpt

B = VoltMat_Cheb.voltmat_cheb(K, 32)
W = Weak_VoltMat_Cheb.weak_voltmat_cheb(K; mu=0.25, sigma=0.0, beta=1.25, N=32)
L = VoltMat_Leg.voltmat_leg(K, 32)
J = VoltMat_Jacobi.voltmat_jacobi(K, 32, 0.0, 0.0)
```

For nonlinear Volterra actions:

```julia
using VoltOpt

grid = CompVoltGrid(64)
cu = zeros(65)
cu[1] = 1.0

Knl(x, t, u) = u / (1 + (x - t) * u^2)
Vu = compVolt_nonlinear_fast(Knl, cu; grid=grid, truncate_output=true)
```

## Examples

Run examples from the repository root with the project environment:

```powershell
julia --project=. examples/ex1_2ndVolterraEqn/ex1_aging_maxwell.jl
julia --project=. examples/ex2_1stVolterraEqn/ex2_firstkind_direct_vs_differentiated.jl
julia --project=. examples/ex3_pantograph/ex3_pantograph_firstkind.jl
julia --project=. examples/ex4_SPVIDE/ex4_SPVIDE.jl
julia --project=. examples/ex5_psespectra/ex5_volterra_schur_besselj1.jl
julia --project=. examples/ex6_nonlinear/ex6_nonlinear_picard.jl
```

All maintained example outputs are written to:

```text
examples/figures
```

with filenames beginning with the example number, such as `ex1_...`,
`ex2_...`, and so on.

## Source Layout

```text
src/VoltOpt.jl               package entry point and exports
src/CompVolt/                fast composed and nonlinear Volterra actions
src/VoltMat/                 Chebyshev, Legendre, and Jacobi Volterra matrices
src/VoltMat/weak_voltmat_cheb.jl
                              weakly singular Chebyshev/JFP Volterra matrices
src/RestMat/                 restriction matrices
src/cheb*.jl, src/vals2cheb.jl
                              Chebyshev transform and helper routines
examples/                    reproducible numerical examples
examples/figures             generated CSV/PDF outputs
```

## Notes

- The main package entry point now imports and re-exports the primary routines.
- Smooth-kernel Volterra matrix constructors handle nonsingular kernels.  The
  weak Chebyshev/JFP constructor handles endpoint/history-variable weak
  singular weights; the resulting matrices may use lower-banded upper
  Hessenberg or related sparse storage rather than a simple approximately
  banded smooth-kernel structure.
- Some example scripts are computationally heavy and may take tens of seconds or
  longer depending on the machine.
- `FastAlmostBandedMatrices` is used opportunistically by some examples; when
  the optional fast path is unavailable, those scripts fall back to sparse LU.
