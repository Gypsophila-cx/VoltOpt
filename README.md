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

Start Julia from the repository root with the project environment activated:

```powershell
julia --project=.
```

Then load the package:

```julia
using VoltOpt
```

### Smooth Volterra operators

For a smooth kernel $K(x,t)$, the package constructs coefficient-space matrices for Volterra integral operators of the form

$$
(\mathcal{V}_K u)(x)
=
\int_{-1}^{x} K(x,t)u(t)\,dt .
$$

If

```math
u(x) \approx \sum_{k=0}^{N} u_k \phi_k(x),
```

then the constructed matrix (V) satisfies

```math
\mathcal{V}_K u
\approx
\sum_{\ell=0}^{N} (Vu)_\ell \phi_\ell(x).
```

For example:

```julia
K(x, t) = exp(-(x - t)^2)

B_cheb = voltmat_cheb(K, 32)
B_leg  = voltmat_leg(K, 32)
R      = restmat_cheb(-1.0, 0.0, 32)
```

Here `voltmat_cheb` constructs the matrix of (\mathcal{V}_K) in the Chebyshev basis

```math
\phi_k(x)=T_k(x),
```

while `voltmat_leg` constructs the corresponding matrix in the Legendre basis

```math
\phi_k(x)=P_k(x).
```

The restriction matrix `restmat_cheb(a,b,N)` represents the affine restriction

```math
u(x)\mapsto u\left(\frac{b-a}{2}x+\frac{a+b}{2}\right),
\qquad x\in[-1,1],
```

in Chebyshev coefficients. For example,

```julia
R = restmat_cheb(-1.0, 0.0, 32)
```

constructs the coefficient-space restriction from ([-1,1]) to the subinterval ([-1,0]).

### Weakly singular Volterra operators

The package also supports weakly singular Volterra operators of the form

```math
(\mathcal{V}_K^\mu u)(x)
=
\int_{-1}^{x}
(x-y)^{-\mu} K(x,y) u(y)\,dy,
\qquad 0 \le \mu < 1,
```

where (K(x,y)) is smooth and the singular factor is explicitly separated as ((x-y)^{-\mu}).

The weakly singular constructor uses a mapped Chebyshev/JFP-type basis

```math
Q_n^{\sigma,\beta}(x)
=
\left(\frac{x+1}{2}\right)^\sigma
T_n\left(
2\left(\frac{x+1}{2}\right)^\beta-1
\right).
```

Thus, if

```math
u(x)\approx
\sum_{k=0}^{N} u_k Q_k^{\sigma,\beta}(x),
```

then `weak_voltmat_cheb` constructs a matrix (V) such that

```math
\mathcal{V}_K^\mu u
\approx
\sum_{\ell=0}^{N} (Vu)_\ell Q_\ell^{\sigma,\beta}(x).
```

In this convention, `mu` is the positive singularity strength. For example,

```julia
mu = 0.5
```

corresponds to

```math
(x-y)^{-\mu}=(x-y)^{-1/2}.
```

After the Duffy transformation

```math
y=\frac{x+1}{2}t+\frac{x-1}{2},
```

the endpoint factor becomes

```math
\left(\frac{x+1}{2}\right)^{1-\mu}.
```

Therefore, a natural mapped-basis choice is

```julia
beta = 1 - mu
```

because this makes the final endpoint multiplier simple in the mapped variable

```math
z=2\left(\frac{x+1}{2}\right)^\beta-1.
```

For example:

```julia
K(x, y) = exp(-x * y)

V, info = weak_voltmat_cheb(
    K;
    mu = 0.5,
    sigma = 0.0,
    beta = 0.5,
    N = 32,
)
```

This constructs the coefficient matrix for

```math
u \mapsto
\int_{-1}^{x}
(x-y)^{-1/2} K(x,y) u(y)\,dy
```

in the mapped Chebyshev basis

```math
Q_n^{0,1/2}(x)
=
T_n\left(
2\sqrt{\frac{x+1}{2}}-1
\right).
```

### Direct access to implementation modules

The main constructors are exported by `VoltOpt`, but the implementation modules can also be accessed directly:

```julia
K(x, y) = exp(-x * y)

B = VoltMat_Cheb.voltmat_cheb(K, 32)

W = Weak_VoltMat_Cheb.weak_voltmat_cheb(
    K;
    mu = 0.5,
    sigma = 0.0,
    beta = 0.5,
    N = 32,
)

L = VoltMat_Leg.voltmat_leg(K, 32)
J = VoltMat_Jacobi.voltmat_jacobi(K, 32, 0.0, 0.0)
```

Mathematically, these constructors correspond to the same Volterra operator

```math
(\mathcal{V}_K u)(x)
=
\int_{-1}^{x}K(x,y)u(y)\,dy,
```

but represented in different coefficient bases:

```math
u(x)=\sum_{k=0}^{N}u_k T_k(x)
\quad\text{for Chebyshev},
```

```math
u(x)=\sum_{k=0}^{N}u_k P_k(x)
\quad\text{for Legendre},
```

and more generally

```math
u(x)=\sum_{k=0}^{N}u_k P_k^{(\alpha,\beta)}(x)
\quad\text{for Jacobi}.
```

The weakly singular version instead represents

```math
(\mathcal{V}_K^\mu u)(x)
=
\int_{-1}^{x}(x-y)^{-\mu}K(x,y)u(y)\,dy
```

in the mapped weighted Chebyshev basis (Q_n^{\sigma,\beta}).

### Nonlinear Volterra actions

For nonlinear Volterra integral actions, the package provides a fast evaluator through `compVolt_nonlinear_fast`. It computes operators of the form

```math
(\mathcal{N}u)(x)
=
\int_{-1}^{x} K(x,t,u(t))\,dt,
```

where the kernel depends nonlinearly on the unknown function value (u(t)).

For example:

```julia
grid = CompVoltGrid(64)

cu = zeros(65)
cu[1] = 1.0

Knl(x, t, u) = u / (1 + (x - t) * u^2)

Vu = compVolt_nonlinear_fast(
    Knl,
    cu;
    grid = grid,
    truncate_output = true,
)
```

This corresponds to the nonlinear Volterra action

```math
(\mathcal{N}u)(x)
=
\int_{-1}^{x}
\frac{u(t)}{1+(x-t)u(t)^2}
\,dt.
```

Here `cu` stores the Chebyshev coefficients of (u), and `Vu` returns the Chebyshev coefficients of (\mathcal{N}u).


## Examples

This project currently provides seven example studies.  The scripts write their
main numerical outputs to `examples/figures`; the previews below use checked-in
PNG files for GitHub rendering.

### Example 1. Aging Maxwell second-kind Volterra equation

`examples/ex1_2ndVolterraEqn/ex1_aging_maxwell.jl` solves a manufactured
second-kind Volterra equation

```math
u(x) + \int_{-1}^{x} K(x,y)u(y)\,dy = f(x),
```

with an aging Maxwell-type smooth kernel
`K(x,y)=exp(-(x-y)/tau(x,y))/tau(x,y)` and an exact solution containing two
narrow smoothed step layers.  The script assembles `voltmat_cheb_square`,
checks convergence in values and coefficients, and compares the numerical and
exact solutions.

| Solution | Error convergence |
|:---:|:---:|
| <img src="examples/figures/ex1_aging_maxwell_solution.png" width="100%" /> | <img src="examples/figures/ex1_aging_maxwell_error_vs_N.png" width="100%" /> |
| The numerical solution matches the manufactured exact solution. | The error decreases as the truncation dimension \(N\) increases. |

| Coefficients | Right-hand side |
|:---:|:---:|
| <img src="examples/figures/ex1_aging_maxwell_coefficients.png" width="100%" /> | <img src="examples/figures/ex1_aging_maxwell_rhs.png" width="100%" /> |
| The coefficient magnitudes show spectral decay. | The plot shows the manufactured forcing term used in the equation. |

### Example 2. First-kind Volterra equation: direct vs differentiated

`examples/ex2_1stVolterraEqn/ex2_firstkind_direct_vs_differentiated.jl`
compares solving a first-kind equation directly,

```math
V_K u = f,
```

against differentiating it to a second-kind equation,

```math
(I + V_{K_x})u = f',
```

for two smooth Gaussian-type kernels.  The figures show max-error convergence
for both approaches.

| Kernel K1 | Kernel K2 |
|:---:|:---:|
| <img src="examples/figures/ex2_compare_K1_direct_vs_differentiated.png" width="100%" /> | <img src="examples/figures/ex2_compare_K2_direct_vs_differentiated.png" width="100%" /> |
| The curve compares direct first-kind solving with the differentiated second-kind formulation. | The curve repeats the direct-versus-differentiated comparison for the second kernel. |

### Example 3. Pantograph first-kind Volterra equation

`examples/ex3_pantograph/ex3_pantograph_firstkind.jl` solves a pantograph
first-kind Volterra problem with delay ratio `q=0.5`.  It uses Chebyshev
restriction matrices `restmat_cheb`, Chebyshev multiplication, and
`voltmat_square_from_coeffmat_fft_convolution` to assemble the coupled system.
The script also checks matrix structure and writes a sparsity plot for the
`N=100` system matrix.

| Matrix sparsity | BigFloat RHS big-truth error |
|:---:|:---:|
| <img src="examples/figures/ex3_pantograph_cheb_A100_spy.png" width="60%" /> | <img src="examples/figures/ex3_pantograph_cheb_error_curve_big_rhs_bigtruth.png" width="100%" /> |
| The sparsity plot shows the assembled \(N=100\) coupled system. | The curve compares the high-precision solution against the big-truth reference. |

### Example 4. Singularly perturbed Volterra integro-differential equation

`examples/ex4_SPVIDE/ex4_SPVIDE.jl` solves a singularly perturbed
Volterra integro-differential equation with `eps=1e-7`, `mu=-1`, and exact
solution `sin(t) + exp(-t/eps)`.  The script builds a rectangular
Chebyshev/ultraspherical tau system, uses `voltmat_cheb_square` for the
Volterra term, and reports true and Cauchy-type errors as `N` grows.

| Solution | Cauchy coefficient error |
|:---:|:---:|
| <img src="examples/figures/ex4_SPVIDE_solution_eps1eminus07_muminus1.png" width="100%" /> | <img src="examples/figures/ex4_SPVIDE_cauchy_eps1eminus07_muminus1.png" width="100%" /> |
| The numerical solution is compared with the exact boundary-layer solution. | The Cauchy coefficient error monitors convergence of the coefficient tail. |

### Example 5. Legendre Volterra pseudospectrum

`examples/ex5_psespectra/ex5_volterra_schur_besselj1.jl` assembles a complex
Legendre-basis Volterra matrix for
`K(x,y)=-im*besselj(1, 2*(x-y)+x*y)`, normalizes it, forms a Schur
triangular factor, and computes a shifted-SVD pseudospectrum grid.

| Pseudospectrum |
|:---:|
| <img src="examples/figures/ex5_volterra_schur_besselj1.png" width="70%" /> |
| The contour plot shows the pseudospectrum of the Legendre based Volterra matrix. |

### Example 6. Nonlinear Volterra-Urysohn Picard iteration

`examples/ex6_nonlinear/ex6_nonlinear_picard.jl` solves a nonlinear
saturated-memory Volterra-Urysohn equation with

```math
K(x,t,u) = \frac{u}{1+(x-t)u^2}.
```

It uses `compVolt_nonlinear_fast` inside Picard iteration and compares the
computed solution with a manufactured Lambert-W-based exact solution.  The
iteration-history plot shows error and update decay.

| Solution | Picard error decay |
|:---:|:---:|
| <img src="examples/figures/ex6_solution_lambertW_picard_N1024.png" width="100%" /> | <img src="examples/figures/ex6_picard_error_decay_N1024.png" width="100%" /> |
| The Picard solution is compared with the Lambert-W manufactured exact solution. | The iteration history shows both solution error and Picard update decay. |

### Example 7. Weakly singular rational-kernel Volterra equation

`examples/ex7_weaksingular/ex7_weaksingular.jl` solves a weakly singular
second-kind Volterra equation using `weak_voltmat_cheb`.  The smooth part is

```math
K(x,y)=\frac{1}{(x+1)^2+(y-1)^2},
```

with `mu=0.5`, `beta=1-mu=0.5`, and a Joukowski/Fourier-polynomial-style
mapped Chebyshev basis.  Here `mu=0.5` denotes the weak kernel factor
`(x-y)^(-1/2)`.  A BigFloat tanh-sinh quadrature is used to project the
right-hand side, and the figures show convergence plus the numerical/exact
solution comparison.

| Solution | Error convergence |
|:---:|:---:|
| <img src="examples/figures/ex7_weak_singular_rational_kernel_solution.png" width="100%" /> | <img src="examples/figures/ex7_weak_singular_rational_kernel_error_vs_N.png" width="100%" /> |
| The numerical solution is compared with the exact weakly singular solution. | The maximum error is shown as the truncation dimension \(N\) increases. |

## Benchmark Figures

The following timing figures summarize matrix-construction behavior from the
benchmark scripts in `test/`.  They include fixed-kernel-complexity/varying-`N`,
fixed-`N`/varying-bandwidth, and weakly singular `weak_voltmat_cheb` timings.

### Smooth-kernel assembly timings

| Runtime vs \(N\) | Runtime vs kernel size |
|:---:|:---:|
| <img src="test/figures/bench_fixedC_varyN_time.png" width="100%" /> | <img src="test/figures/bench_fixedN_varyC_time.png" width="100%" /> |
| Assembly time is shown as the matrix dimension \(N\) increases. | Assembly time is shown as the kernel expansion size increases. |

### Weakly singular assembly timings

| Runtime vs \(N\) | Runtime vs kernel size |
|:---:|:---:|
| <img src="test/figures/bench_weak_fixedC_varyN_time.png" width="100%" /> | <img src="test/figures/bench_weak_fixedN_varyC_time.png" width="100%" /> |
| Weak-kernel assembly time is shown as the matrix dimension \(N\) increases. | Weak-kernel assembly time is shown as the bandwidth or coefficient size increases. |

## Reproducing the Examples

Run examples from the repository root with the project environment:

```powershell
julia --project=. examples/ex1_2ndVolterraEqn/ex1_aging_maxwell.jl
julia --project=. examples/ex2_1stVolterraEqn/ex2_firstkind_direct_vs_differentiated.jl
julia --project=. examples/ex3_pantograph/ex3_pantograph_firstkind.jl
julia --project=. examples/ex4_SPVIDE/ex4_SPVIDE.jl
julia --project=. examples/ex5_psespectra/ex5_volterra_schur_besselj1.jl
julia --project=. examples/ex6_nonlinear/ex6_nonlinear_picard.jl
julia --project=. examples/ex7_weaksingular/ex7_weaksingular.jl
```

The standalone example scripts are the seven files above.  The file
`examples/ex5_psespectra/VoltMat_Leg.jl` is an auxiliary local helper, not a
separate example.  All maintained example outputs are written to:

```text
examples/figures
```

with filenames beginning with the example number, such as `ex1_...`,
`ex2_...`, and so on.

Some examples are computationally heavy and may take tens of seconds or longer
depending on the machine.

## Reproducing the Benchmarks

Benchmark data are written to `test/data`, and the plotting scripts write PDFs
and PNGs to `test/figures`:

```powershell
julia --project=. test/bench_fixedC_varyN.jl
julia --project=. test/bench_fixedN_varyC.jl
julia --project=. test/bench_weak_fixedC_varyN.jl
julia --project=. test/bench_weak_fixedN_varyC.jl

julia --project=. test/plot_varyNtime.jl
julia --project=. test/plot_Ctime.jl
julia --project=. test/plot_weak_varyNtime.jl
julia --project=. test/plot_weak_Ctime.jl
```

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
examples/figures             generated CSV/PDF/PNG outputs
test/                        benchmark scripts
test/data                    benchmark CSV outputs
test/figures                 benchmark timing figures
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
- This project is implemented in Julia 1.11.1.
