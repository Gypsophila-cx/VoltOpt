# ex1 Gutleb--Olver Diagnostics

## Problem

Original equation in `ex1_aging_maxwell.jl`:

```math
u(x) + \int_{-1}^{x} K(x,y)u(y)\,dy = f(x), \qquad x \in [-1,1].
```

This is a second-kind Volterra integral equation.  It is not a VIDE: there are
no derivative terms and no initial/tau rows.

The manufactured exact solution is

```math
u_*(x) = 1 + A_1 S_{\delta_1}(x,x_1) + A_2 S_{\delta_2}(x,x_2),
```

with `x1 = -0.35`, `x2 = 0.42`, `A1 = 1.0`, `A2 = -0.6`,
`delta1 = 1.0e-3`, and `delta2 = 1.5e-3`.  The RHS is built by split
Gauss quadrature:

```math
f(x) = u_*(x) + \int_{-1}^{x} K(x,y)u_*(y)\,dy.
```

The original script uses `evenly_spaced_integer_nodes(2, 40000, 100)` and
reports physical-grid maximum absolute error.

## Transform

Gutleb--Olver works on `[0,1]`, so ex1 is mapped by

```math
x = 2s - 1, \qquad y = 2r - 1.
```

The transformed unknown and RHS are

```math
u_{go}(s) = u_*(2s-1), \qquad g_{go}(s) = f(2s-1).
```

The integral term becomes

```math
\int_{-1}^{x} K(x,y)u(y)\,dy
=
\int_0^s K_{go}(s,r)u_{go}(r)\,dr,
```

where

```math
K_{go}(s,r) = 2K(2s-1,2r-1).
```

There is no derivative scaling in ex1.

## GO Basis And System

The debug script uses the sparse/banded Gutleb--Olver construction from
`others/GutlebOlver`:

- solution basis: GO shifted Jacobi `P^(1,0)` on `[0,1]`;
- raw Volterra operator: strict-banded `VK` from triangle Jacobi/Proriol
  kernel coefficients, assembled as `Q_y K(1-X,Y) E_y` with `flip=true`;
- physical output basis for the second-kind equation:
  `S10_from_00 * reflectPabtoPba * L00_from_10`;
- final system tested for ex1:
  `A = I + S10_from_00 * Reflect * L00_from_10 * VK`.

The public GO Jacobi parameters use the paper convention.  Internally,
ApproxFun stores the shifted endpoint parameters in the opposite order, so
`P^(1,0)` is represented by `Jacobi(0,1,0..1)`.

No tau rows are used because ex1 has no initial or boundary condition.

## Current Debug Result

Command:

```powershell
julia --project=others/GutlebOlver others/GutlebOlver/test/compare/debug_go_ex1_system.jl
```

Observed diagnostic run after enforcing the second-kind physical chain:

- `N = 24`
- `kerneldepth` scan: `8, 12, 16, 24, 32`
- `VK storage = BandedMatrix`
- `A storage = BandedMatrix`
- `raw_operator = Q_y K(1-X,Y) E_y`
- `final_chain = S10_from_00 * R * L00_from_10 * VK`
- `final_system = A = I + V_phys`
- `flip_policy = strict-band constructor flip=true; K_go is not pre-flipped`
- `dense_fallback = false`
- `raw_vk_added_to_identity = false`
- `tau_rows = false`

| kerneldepth | residual max | residual rel | V_phys physical error | solve physical error | A bandwidth |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8  | 2.130698e-03 | 1.168325e-03 | 1.153396e-02 | 5.151597e-01 | 8/8 |
| 12 | 2.130294e-03 | 1.168103e-03 | 1.148298e-02 | 5.152084e-01 | 12/12 |
| 16 | 2.130294e-03 | 1.168103e-03 | 1.148354e-02 | 5.152080e-01 | 16/16 |
| 24 | 2.130294e-03 | 1.168103e-03 | 1.148443e-02 | 5.152073e-01 | 23/23 |
| 32 | 2.130294e-03 | 1.168103e-03 | 1.158396e-02 | 5.151116e-01 | 23/23 |

The sparse/banded storage requirement is satisfied for the tested operator and
system.  The ex1 GO benchmark row is marked `success` only when all of the
following pass:

- RHS projection built by `build_go_rhs_coefficients(...; alpha=1,beta=0,
  method=:fixed_gauss_jacobi)` validates and has an acceptable tail;
- exact-solution projection to GO `P^(1,0)` validates;
- exact-coefficient residual `A*u_exact_coeff - rhs_coeff` is small;
- `V_phys*u_exact_coeff` evaluates to
  `int_0^s K_go(s,r)u_go(r)dr` on a physical grid;
- solved coefficients have small physical-grid error.

Dense collocation fallback and dense final systems remain rejected.

The current `N = 24` diagnostic is not certified: RHS projection validation is
about `5.171804e-01`, exact-solution projection validation is about
`5.096791e-01`, and the solve physical-grid error is about `5.15e-01`.
Therefore the GO ex1 CSV status remains `failed`/not certified.
