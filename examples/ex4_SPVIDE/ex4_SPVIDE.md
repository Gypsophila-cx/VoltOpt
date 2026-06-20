# Example 4: SPVIDE with a non-convolution oscillatory Volterra kernel

This example considers a singularly perturbed Volterra integro-differential equation (SPVIDE) whose right-hand side is explicitly prescribed, while the exact solution is not used.  The purpose of the example is to test the convergence of the numerical solution as the polynomial degree `N` increases, using Cauchy-type errors between two numerical solutions.

## Physical problem on \([0,1]\)

Let \(0 < \varepsilon \ll 1\).  We solve

\[
\varepsilon v'(t) + (1+t)v(t)
+ \int_0^t \cos(\omega t s) v(s)\,ds
= f(t), \qquad 0\le t\le 1,
\]

with the initial condition

\[
v(0)=1.
\]

The right-hand side is explicitly prescribed as

\[
f(t)=2+\sin(\pi t).
\]

The Volterra kernel is

\[
K(t,s)=\cos(\omega t s), \qquad 0\le s\le t\le 1.
\]

This kernel is oscillatory when \(\omega\) is moderately large.  It is also non-convolutional because it depends on the product \(ts\), rather than on the difference \(t-s\).

## Boundary-layer mechanism

The small parameter \(\varepsilon\) multiplies the derivative term.  Formally setting \(\varepsilon=0\) gives the reduced Volterra equation

\[
(1+t)v_0(t)+\int_0^t \cos(\omega t s)v_0(s)\,ds = 2+\sin(\pi t).
\]

At \(t=0\), the integral vanishes, so the reduced equation requires

\[
v_0(0)=f(0)=2.
\]

However, the imposed initial condition is

\[
v(0)=1.
\]

Thus the initial condition is incompatible with the reduced problem.  This mismatch produces an initial boundary layer near \(t=0\) when \(\varepsilon\) is small.  The layer is expected to have thickness of order \(O(\varepsilon)\), while away from \(t=0\) the solution is governed mainly by the reduced Volterra equation.

## Mapping to \([-1,1]\)

The Julia code solves the problem in the Chebyshev variable

\[
t=\frac{x+1}{2}, \qquad x\in[-1,1].
\]

Define

\[
u(x)=v\!\left(\frac{x+1}{2}\right).
\]

Since

\[
v'(t)=2u'(x),
\]

and

\[
\int_0^t K(t,s)v(s)\,ds
= \frac12\int_{-1}^x
\cos\!\left(\omega\frac{(x+1)(y+1)}{4}\right)u(y)\,dy,
\]

we obtain the mapped problem

\[
2\varepsilon u'(x)+\frac{x+3}{2}u(x)
+\frac12\int_{-1}^x
\cos\!\left(\omega\frac{(x+1)(y+1)}{4}\right)u(y)\,dy
=2+\sin\!\left(\pi\frac{x+1}{2}\right),
\]

with boundary condition

\[
u(-1)=1.
\]

## Numerical method

The unknown \(u(x)\) is expanded in Chebyshev polynomials:

\[
u_N(x)=\sum_{k=0}^N c_k T_k(x).
\]

The differential part is discretized using the ultraspherical spectral method.  The non-derivative terms are converted into the \(C^{(1)}\) ultraspherical basis, and the boundary condition \(u(-1)=1\) is imposed by a tau row.

The Volterra integral operator is constructed by the existing `VoltMat_Cheb.voltmat_cheb_square` routine.  The code builds a square Volterra matrix of degree `N+1`, then keeps the first `N+1` columns so that the non-derivative operator maps Chebyshev modes `0:N` to output modes `0:N+1`.

## Cauchy error diagnostics

Because the exact solution is not known, the code compares numerical solutions at two nearby polynomial degrees,

\[
N_p = \max(N+1, \operatorname{round}(1.01N)).
\]

Two Cauchy diagnostics are reported.

### Coefficient-space Cauchy error

The coefficient vectors are zero-padded to the same length:

\[
E^{\mathrm{coeff}}_N
=\left\|\operatorname{pad}(c^{(N)})-
\operatorname{pad}(c^{(N_p)})\right\|_\infty.
\]

### Grid-space Cauchy error

Both numerical solutions are evaluated on the same Chebyshev-Lobatto diagnostic grid:

\[
E^{\mathrm{grid}}_N
=\max_j \left|u_N(x_j)-u_{N_p}(x_j)\right|.
\]

The grid-space Cauchy error is often the more direct diagnostic for function-level convergence.

## Output files

The script writes two CSV files to `examples/data`:

1. `ex4_SPVIDE_error_history_eps*_omega*.csv`

   This file contains

   ```text
   N,Np,cauchy_coeff_error_inf,cauchy_grid_error_inf,tail_coeff_norm_inf,residual_inf
   ```

2. `ex4_SPVIDE_coefficients_N*_eps*_omega*.csv`

   This file stores the Chebyshev coefficients of the final solution.

## Parameters to vary

The most important parameters are:

```julia
const EPS = 1.0e-7
const OMEGA = 50.0
const N_FINAL = 32_000
```

Increasing `OMEGA` makes the kernel more oscillatory and may require larger `N`.  Decreasing `EPS` makes the initial layer thinner and also requires larger `N` to resolve the solution near `t=0`, equivalently near `x=-1`.

For debugging, it is recommended to start with

```julia
const EPS = 1.0e-3
const N_FINAL = 1000
```

and then increase `N_FINAL` after the script is confirmed to run correctly.
