# Gutleb--Olver diagnostics for ex4_SPVIDE

This note summarizes the current Gutleb--Olver sparse/banded diagnostics for
`examples/ex4_SPVIDE`.  The purpose is to explain why the original
`eps = 1e-7` GO benchmark row is kept as `failed` in the current practical
benchmark range, even though the sparse operator assembly itself is not a dense
collocation fallback.

## Original equation

The original problem is posed on `x in [-1,1]`:

```text
2eps*u_x(x) + (x+3)/2*u(x)
  + 1/2 int_{-1}^x exp(mu*(x+1)*(y+1)/4) u(y) dy
  = f((x+1)/2),

u(-1) = 1.
```

The exact solution used by the original script is

```text
u(x) = sin((x+1)/2) + exp(-(x+1)/(2eps)).
```

## Variable transform

Use

```text
x = 2s - 1,
y = 2r - 1,
s,r in [0,1].
```

The transformed solution is

```text
u_go(s) = sin(s) + exp(-s/eps).
```

The original `1/2` integral prefactor cancels the Jacobian `dy = 2dr`, giving

```text
eps*u_s(s) + (1+s)u(s)
  + int_0^s exp(mu*s*r) u(r) dr
  = f(s),

u(0) = 1.
```

Thus the transformed kernel is

```text
K_go(s,r) = exp(mu*s*r).
```

## Basis convention

The current GO SPVIDE attempt uses:

```text
solution basis: shifted Jacobi P^(0,1)
equation output/RHS basis: shifted Jacobi P^(1,2)
tau row: endpoint evaluation at s=0 in the P^(0,1) solution basis
```

The sparse/semi-banded body is assembled as

```text
eps * Derivative(P01 -> P12)
+ Conversion(P01 -> P12) * Multiplication(1+s in P01)
+ Conversion(P00 -> P12) * Reflect * WLowering * VK
```

## Volterra chain diagnostics

The raw strict-band GO matrix `VK` is not directly the physical
`int_0^s` operator.  The diagnostic script
`debug_go_volterra_basis_chain.jl` verifies low-order manufactured polynomial
cases and finds:

```text
physical int_0^s operator: Reflect * WLowering * VK
P^(1,2) output chain:
  Conversion(P00 -> P12) * Reflect * WLowering * VK
```

For polynomial cases such as `K=1`, `K=s`, `K=r`, and
`K=1+s+r` with polynomial inputs, the verified chain reaches about
machine precision (`~1e-15`) when `flip=true`.

## Other operator diagnostics

The derivative and multiplication checks in the current convention pass at
machine precision for polynomial inputs:

```text
derivative_error       ~2.66e-15
multiplication_error   ~5.33e-15
deriv_plus_mult_error  ~8.88e-15
```

RHS projection into `P^(1,2)` is also correct for polynomial RHS tests:

```text
1, s, s^2, 1+s+s^2: errors ~1e-15
```

For the ex4 RHS, projection is also accurate for moderate `eps`, for example
`eps=1e-1` and, from moderate N onward, `eps=1e-2`.

## Row scheme

The system residual diagnostic checks several tau/body row choices.  The most
consistent square truncation is:

```text
Scheme A:
A = [tau row; body rows 1:N-1]
b = [1; rhs rows 1:N-1]
```

Schemes that use body rows `2:N` or replace the first body equation by the tau
row are much less reliable for this GO formulation.

## Eps scan and boundary layer

The diagnostic script `debug_go_spvide_system_residual.jl` checks projection
of the exact solution

```text
u(s) = sin(s) + exp(-s/eps)
```

into the `P^(0,1)` solution basis.

Representative projection errors:

```text
eps=1e-1:
  N=32,64,128,256 -> ~4e-16

eps=1e-2:
  N=32 -> ~2e-4
  N=64,128,256 -> ~4e-15

eps=1e-4:
  N=32  -> ~8.94e-1
  N=64  -> ~6.51e-1
  N=128 -> ~3.14e-1
  N=256 -> ~8.76e-3

eps=1e-7:
  N=32,64,128,256 -> ~1
```

This shows that the original `eps=1e-7` boundary layer is not resolved at the
currently tested N.  It does not prove that the GO basis can never resolve the
layer; it indicates that the tested benchmark range is insufficient.  Resolving
the original layer may require `O(10^4)` to about `3e4` polynomial modes, which
is outside the practical scope of the current GO benchmark run.

## Why the original GO row is not success

The original `eps=1e-7` GO row remains `failed` because the exact solution's
boundary layer is unresolved at the tested N, not because the implementation
falls back to dense collocation or because the verified operator chain is
mathematically inconsistent.

Specifically:

```text
not a dense collocation path
not a dense final system
not a dense solve
not a Volterra-chain failure
not a derivative/multiplication operator failure
not a polynomial RHS projection failure
```

The current failure is a practical resolution issue for an extremely thin
endpoint boundary layer in a global polynomial Jacobi basis.

## Future directions

Useful next steps for a meaningful GO-vs-voltmat comparison include:

```text
1. Compare moderate-eps SPVIDE cases, such as eps=1e-1 or eps=1e-2.
2. Use much larger N only in a purpose-built experiment, not in the default benchmark.
3. Introduce a variable mapping adapted to the endpoint layer.
4. Use a piecewise spectral method with a refined layer subinterval.
5. Use endpoint-fitted or layer-adapted basis functions.
6. Keep the current GO path for smooth or moderate-layer SPVIDE comparisons.
```

Large-N projection or solve experiments for the original `eps=1e-7` case
should be optional and disabled by default:

```text
This is optional and disabled by default.
The original eps=1e-7 layer may require O(10^4)-O(3e4) modes.
```

## Large-scale runner

The dedicated script

```text
examples/ex4_SPVIDE/run_go_large_scale_eps1e-7.jl
```

writes append-only/resumable diagnostics to

```text
examples/data/ex4_go_large_scale_eps1e-7.csv
```

It supports projection-only and solve-only modes through environment
variables.  Projection-only should be run first.  Solve runs should only be
started after projection errors show meaningful decrease, or by explicit user
request.

The first projection-only run through `N=32768` used an adaptive ApproxFun
construction padded/truncated to `N+1`, followed by chunked boundary-layer-grid
evaluation.  That diagnostic did not show decreasing projection error; the CSV
notes mark that this is not yet a guaranteed fixed-N Jacobi projection.  This
result should therefore be read as a diagnostic of the current projection
pipeline, not as a proof that `N=32768` is mathematically insufficient.
