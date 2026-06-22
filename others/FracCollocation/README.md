# FracCollocation

Independent Julia implementation of the Hou-Lin-Azaiez-Xu fractional Jacobi
collocation idea for weakly singular second-kind Volterra integral equations:

```math
u(x) = g(x) + \int_0^x (x-s)^{-\mu} K(x,s) u(s)\,ds,
\qquad 0 < \mu < 1,\quad x\in[0,1].
```

The optional parameter `sigma` solves

```math
u(x) = g(x) + \sigma \int_0^x (x-s)^{-\mu} K(x,s) u(s)\,ds.
```

## N convention

This package uses an engineering convention:

```text
N = number of solution nodes = number of collocation points = number of quadrature points
```

Therefore:

```julia
length(sol.x) == N
length(sol.values) == N
length(sys.theta) == N
size(sys.A) == (N, N)
```

The fractional polynomial space is

```math
\operatorname{span}\{1, x^\lambda, x^{2\lambda}, \ldots, x^{(N-1)\lambda}\}.
```

This differs from papers that use degree `N` and therefore `N+1` nodes.

## Nodes

Let `z_i` be shifted Jacobi-Gauss nodes on `[0,1]`, i.e. zeros of
`J_N^{alpha,beta}(2z-1)`. The collocation nodes are

```math
x_i = z_i^{1/\lambda}.
```

The fractional Lagrange functions are evaluated as

```math
h_j^{alpha,beta,\lambda}(x) = \ell_j(x^\lambda),
```

where `ell_j` is the ordinary Lagrange cardinal basis on the `z` nodes.
The code uses barycentric Lagrange evaluation, not a Vandermonde matrix.

## Integral discretization

At each collocation point `x_i`, use

```math
s = x_i \theta^{1/\lambda}.
```

The quadrature nodes and weights `(theta_k, w_k)` are shifted Jacobi-Gauss
nodes for the weight

```math
(1-\theta)^{-\mu}\theta^{1/\lambda-1}.
```

Thus the quadrature Jacobi parameters are

```julia
alpha_q = -mu
beta_q = 1/lambda - 1
```

and

```math
\bar K(x_i, x_i\theta^{1/\lambda})
=
\frac{x_i^{1-\mu}}{\lambda}
\left(
\frac{1-\theta^{1/\lambda}}{1-\theta}
\right)^{-\mu}
K(x_i, x_i\theta^{1/\lambda}).
```

The integral matrix is

```math
V_{ij} =
\sum_{k=1}^N
w_k \bar K(x_i, x_i\theta_k^{1/\lambda})
\ell_j(z_i\theta_k).
```

The solved linear system is

```math
(I - \sigma V)u = g.
```

Matrix construction is a dense `O(N^3)` triple loop. It uses exactly `N`
quadrature nodes and reuses one length-`N` scratch vector for the Lagrange
basis values.

## Basic usage

```julia
using FracCollocation

prob = FracVIEProblem(
    mu = 0.5,
    lambda = 0.5,
    alpha = -0.5,
    beta = -0.5,
    K = (x, s) -> exp(x - s),
    g = x -> 1.0,
)

sys = build_system(prob, 40)
sol = solve(sys)

xgrid = range(0.0, 1.0; length=1001)
uvals = evaluate(sol, xgrid)
```

or directly:

```julia
sol = solve(prob, 40)
```

System fields include:

```julia
sys.x
sys.z
sys.theta
sys.wtheta
sys.V
sys.A
sys.rhs
```

`sol.values` stores the solution values at the collocation nodes.

The exported diagnostic helper

```julia
collocation_residual(sys, sol.values)
```

returns `sol.values - sys.rhs - sys.problem.sigma * sys.V * sol.values`,
which is the same residual as `sys.A * sol.values - sys.rhs`.

## Run tests

```bash
cd others/FracCollocation
julia --project=. -e "using Pkg; Pkg.instantiate(); Pkg.test()"
```

## Run examples

```bash
julia --project=. examples/example_basic.jl
julia --project=. examples/example_convergence.jl
julia --project=. examples/diagnose_basic.jl
julia --project=. examples/example_paper41_like.jl
```

The convergence example prints

```text
N, max_error, build_time, solve_time
```

and writes `examples/data/convergence.csv`.

`diagnose_basic.jl` writes `examples/data/diagnose_basic.csv` with residual,
condition number, matrix size, and solution-size diagnostics for the basic
`K(x,s)=exp(x-s)`, `g(x)=1` case. `example_paper41_like.jl` writes
`examples/data/paper41_like.csv` using an `Nref=100` numerical reference for
the paper Example 4.1 style parameter choices.
