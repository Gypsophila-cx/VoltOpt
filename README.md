# VoltOpt

VoltOpt is a Julia research codebase for fast spectral construction and
application of Volterra integral operators.  The current implementation focuses
on polynomial coefficient representations, fast matrix assembly, and
reproducible numerical experiments for smooth and weakly singular kernels.

The project includes:

- smooth Volterra operator construction in Chebyshev (Duffy-transformed kernel,
  recurrence construction, and FFT-accelerated Toeplitz--Hankel assembly),
  Legendre (a dedicated strictly banded construction), and general Jacobi
  bases (product linearization, recurrent/compressed moment tables, and
  top-dense/band-tail construction);
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
|_ tools/                       shared paper-figure and repository helpers
|  |_ voltopt_plot_style.m      shared MATLAB paper-figure style and fixed geometry
|  |_ voltopt_export_figure.m   common fixed-canvas PDF/PNG export helper
|  |_ audit_layout.jl           repository/static layout audit
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

These routines discretize Volterra operators of the form

```math
(\mathcal{V}_K u)(x) = \int_{-1}^{x} K(x,y)u(y)\,dy
```

in polynomial coefficient bases.

The public API is grouped by basis and purpose:

- Chebyshev (`VoltMat_Cheb`): `voltmat_cheb` constructs the square operator
  matrix from a kernel or its transformed coefficient matrix;
  `voltmat_cheb_square` and
  `voltmat_square_from_coeffmat_fft_convolution` are aliases of this constructor.
- Legendre (`VoltMat_Leg`): `voltmat_leg` constructs the operator, while
  `legendre_volterra_symband_from_C` assembles the band from known coefficients.
  `legendre_interp_coeffs_Khat` and `adaptive_legendre_interp_coeffs_Khat`
  construct transformed-kernel coefficients. `legendre_values`,
  `legendre_values!`, and `gausslegendre_rule` provide basis values and
  quadrature nodes/weights.
- Jacobi (`VoltMat_Jacobi`): `voltmat_jacobi` uses the integrated
  `voltmat_jacobi_full` constructor; `jacobi_volterra_topdense_from_C` constructs
  the dense leading block. `jacobi_interp_coeffs_Khat` and
  `adaptive_jacobi_interp_coeffs_Khat` construct transformed-kernel coefficients.
  `jacobi_values`, `jacobi_values!`, and `gaussjacobi_nodes` provide basis values
  and interpolation nodes.

For example:

```julia
using VoltOpt

K(x, y) = exp(-x*y)
N = 200

Bcheb = voltmat_cheb(K, N)
Bleg  = voltmat_leg(K, N)
Bjac  = voltmat_jacobi(K, N, 0.3, 0.2)
```

#### Common Duffy formulation

All three smooth implementations start with the collapsed-coordinate map

```math
\psi(x,t)=\frac{x+1}{2}t+\frac{x-1}{2},
\qquad
\widehat K(x,t)=K\bigl(x,\psi(x,t)\bigr).
```

The Duffy transform moves kernel construction from the Volterra triangle to a
tensor-product approximation on the standard square, giving

```math
(\mathcal V_K u)(x)
=\frac{1+x}{2}\int_{-1}^{1}
\widehat K(x,t)u\bigl(\psi(x,t)\bigr)\,dt.
```

The factor `(1+x)/2` is included in coefficient assembly. For Jacobi
parameters `alpha, beta > -1`, write

```math
p_q(x)=P_q^{(\alpha,\beta)}(x),
\qquad
\widehat K(x,t)\approx
\sum_{i=0}^{n}\sum_{j=0}^{m}C_{ij}p_i(x)p_j(t).
```

Here `i` and `j` index the outer and transformed integration modes, and `n`,
`m` are their maximum retained degrees. Thus `C` has size `(n+1, m+1)`, with
mathematical entry `C_ij` stored as `C[i+1,j+1]`. It contains coefficients of
the transformed/absorbed kernel `K(x, psi(x,t))`, not direct tensor-product
coefficients of the original `K(x,y)`. This is the meaning of `C` in
`voltmat_jacobi(C, ...)` and `voltmat_leg(C, ...)` (and the analogous Chebyshev
interface). The final matrix is denoted by `B`; its column `k` acts on input
mode `k`, and row `ell` is output mode `ell`. The square truncation retains
modes `0:N`.

#### General Jacobi construction

##### Key structural identities

The Duffy--Chebyshev construction extends to Legendre and general Jacobi
bases through the same two essential algebraic ingredients: affine
restriction has an explicit shifted-Jacobi representation, and polynomial
products linearize back into the same Jacobi basis.

Use the standard normalization and weight

```math
\begin{gathered}
P_n^{(\alpha,\beta)}(1)=\frac{(\alpha+1)_n}{n!},\qquad \alpha,\beta>-1,\\
w_{\alpha,\beta}(x)=(1-x)^\alpha(1+x)^\beta,\qquad \lambda=\alpha+\beta.
\end{gathered}
```

Here `(a)_n` is the Pochhammer symbol and `B(a,b)` below is the beta function.
The indices `j` and `i` denote input and restricted degrees. The map

```math
\psi(t,x)=\frac{t+1}{2}x+\frac{t-1}{2}=\psi(x,t)
```

maps `[-1,1]` to `[-1,t]` in `x`. Its symmetry makes this notation consistent
with the preceding Duffy formulation. For `0 <= i <= j`,

```math
\begin{gathered}
\int_{-1}^{1}P_i^{(\alpha,\beta)}(x)
P_j^{(\alpha,\beta)}\!\left(\frac{t+1}{2}x+\frac{t-1}{2}\right)
w_{\alpha,\beta}(x)\,dx
=\kappa_{ij}^{(\alpha,\beta)}
\left(\frac{t+1}{2}\right)^i P_{j-i}^{(-1,\lambda+2i+1)}(t),\\
\kappa_{ij}^{(\alpha,\beta)}=2^{\lambda+1}\frac{(j+\lambda+1)_i}{i!}
B(\alpha+i+1,\beta+i+1)
\frac{(\beta+i+1)_{j-i}}{(\lambda+2i+2)_{j-i}}.
\end{gathered}
```

For `i>j` the integral vanishes by orthogonality: restriction preserves degree
`j`. Define the squared Jacobi norm and restriction coefficients by

```math
\begin{gathered}
h_i^{(\alpha,\beta)}=\int_{-1}^{1}[P_i^{(\alpha,\beta)}(x)]^2w_{\alpha,\beta}(x)\,dx,\\
P_j^{(\alpha,\beta)}(\psi(t,x))
=\sum_{i=0}^{j}R_{ij}^{(\alpha,\beta)}(-1,t)P_i^{(\alpha,\beta)}(x).
\end{gathered}
```

Dividing the projection by the norm gives the actual matrix coefficient:

```math
\begin{gathered}
R_{ij}^{(\alpha,\beta)}(-1,t)=\rho_{ij}^{(\alpha,\beta)}
\left(\frac{t+1}{2}\right)^i P_{j-i}^{(-1,\lambda+2i+1)}(t),\\
\rho_{ij}^{(\alpha,\beta)}=\frac{\kappa_{ij}^{(\alpha,\beta)}}{h_i^{(\alpha,\beta)}}
=\frac{(j+\lambda+1)_i}{(i+\lambda+1)_i}
\frac{(\beta+i+1)_{j-i}}{(\lambda+2i+2)_{j-i}}.
\end{gathered}
```

Each coefficient is an elementary power `((1+t)/2)^i` times a single shifted
Jacobi polynomial. Restriction stays in a parameterized Jacobi family, the
structure exploited by Volterra assembly, rather than an unstructured dense
operation. `restmat_jacobi` constructs the same restriction by coefficient-space
three-term recurrence, avoiding direct evaluation of this analytic closed form.

For `alpha=beta=-1/2`, conversion to Chebyshev normalization gives, for
`j >= 1` and `0 <= i <= j`,

```math
\begin{gathered}
T_n=\frac{n!}{(1/2)_n}P_n^{(-1/2,-1/2)},\\
\int_{-1}^{1}T_i(x)T_j\!\left(\frac{t+1}{2}x+\frac{t-1}{2}\right)
\frac{dx}{\sqrt{1-x^2}}
=\frac{\pi j}{i+j}\left(\frac{t+1}{2}\right)^i P_{j-i}^{(-1,2i)}(t).
\end{gathered}
```

The paper's Chebyshev identity is thus the normalized Jacobi specialization,
not merely an analogous formula. Handle `j=0` separately: the constant mode
`T_0=1` has `i=0` projection `pi`.

For `alpha=beta=0` and `lambda=0`, the same formula gives

```math
R_{ij}^{\mathrm{Leg}}(-1,t)
=\frac{(j+1)_i}{(i+1)_i}\frac{(i+1)_{j-i}}{(2i+2)_{j-i}}
\left(\frac{1+t}{2}\right)^i P_{j-i}^{(-1,2i+1)}(t).
```

This restriction structure underlies the dedicated Legendre construction.
The second identity linearizes modes `p` and `q` into output modes `r`:

```math
\begin{gathered}
P_p^{(\alpha,\beta)}(x)P_q^{(\alpha,\beta)}(x)
=\sum_{r=|p-q|}^{p+q}\mu_{p,q,r}^{(\alpha,\beta)}P_r^{(\alpha,\beta)}(x),\\
\mu_{p,q,r}^{(\alpha,\beta)}=\frac{1}{h_r^{(\alpha,\beta)}}\int_{-1}^{1}
P_p^{(\alpha,\beta)}(x)P_q^{(\alpha,\beta)}(x)P_r^{(\alpha,\beta)}(x)
w_{\alpha,\beta}(x)\,dx.
\end{gathered}
```

Coefficients vanish outside `|p-q| <= r <= p+q`. A nonsymmetric Jacobi
weight generally has no further parity restriction; for symmetric families,
including Legendre, parity removes every other mode. In contrast,

```math
T_p(x)T_q(x)=\frac12\left(T_{p+q}(x)+T_{|p-q|}(x)\right).
```

Chebyshev's exceptional two-term linearization produces its Toeplitz--Hankel
FFT convolution. General Jacobi products have finite local support but many
modes, so this acceleration does not carry over directly. This is a central
reason for the different paper-level assembly estimates given below.

##### Recurrence construction

`jacobi_interp_coeffs_Khat` samples the transformed kernel on Gauss--Jacobi
nodes and solves with Jacobi Vandermonde matrices in both coordinates to obtain
`C`. The adaptive version chops the two-dimensional coefficient tails. The
current general Jacobi matrix constructor requires `size(C,1) >= size(C,2)`;
the adaptive constructor enforces this by default.

For a polynomial degree `q`, multiplication by the coordinate has the
three-term form

```math
xP_q^{(\alpha,\beta)}
=A_qP_{q+1}^{(\alpha,\beta)}
+B_qP_q^{(\alpha,\beta)}
+C_qP_{q-1}^{(\alpha,\beta)}.
```

This tridiagonal relation drives both product linearization and the final
local multiplication by `(1+x)/2`. The coefficients used in the code, for
`q >= 1`, are

```math
A_q=\frac{2(q+1)(q+\lambda+1)}
{(2q+\lambda+1)(2q+\lambda+2)},
\qquad
B_q=\frac{\beta^2-\alpha^2}{(2q+\lambda)(2q+\lambda+2)},
\qquad
C_q=\frac{2(q+\alpha)(q+\beta)}
{(2q+\lambda)(2q+\lambda+1)}.
```

These are the coefficients of `_jacobi_x_coeffs`; the subscripted scalar
`C_q` is distinct from the kernel matrix `C`. The degree-zero case is handled
separately:

```math
A_0=\frac{2}{\alpha+\beta+2},
\qquad
B_0=\frac{\beta-\alpha}{\alpha+\beta+2},
\qquad C_0=0.
```

This initializes the recurrence without referring to a negative-degree
polynomial. Product linearization then takes the form

```math
P_i^{(\alpha,\beta)}(x)P_\ell^{(\alpha,\beta)}(x)
=\sum_s L_s^{(i,\ell)}P_s^{(\alpha,\beta)}(x).
```

The index `s` is the resulting outer Jacobi mode. Starting from the constant
factor, the multiplication recurrence gives

```math
L_s^{(0,\ell)}=\delta_{s\ell},\qquad
L_s^{(i+1,\ell)}=\frac{1}{A_i}\left[
A_{s-1}L_{s-1}^{(i,\ell)}+(B_s-B_i)L_s^{(i,\ell)}
+C_{s+1}L_{s+1}^{(i,\ell)}-C_iL_s^{(i-1,\ell)}\right].
```

Out-of-range terms are zero. `_apply_x_local` and `_combine_next` implement
these adjacent-mode contributions; `_linearization_cache` generates the local
coefficients needed for band assembly, avoiding a global dense tensor of `mu`.

After the Duffy transform, symmetry of `psi` permits outer-variable restriction:

```math
P_k^{(\alpha,\beta)}(\psi(x,t))
=\sum_{s=0}^{k}R_{sk}^{(\alpha,\beta)}(-1,t)P_s^{(\alpha,\beta)}(x).
```

Kernel expansion introduces products `P_i(x)P_s(x)`; linearization returns
them to the Jacobi basis, leaving the one-dimensional projected moments below.
**Restriction and product linearization are the two fundamental identities
behind the Jacobi extension; the remaining recurrences efficiently evaluate
and assemble the coefficients implied by these identities.**

For the banded tail, let `r = k-s >= 0` be the offset from outer mode `s` to
input mode `k`, and define auxiliary functions

```math
z=\frac{1+t}{2},
\qquad \Psi_{s,0}(t)=z^s,
\qquad
\Psi_{s,r}(t)=(t-1)z^sP_{r-1}^{(1,\lambda+2s+1)}(t)
\quad (r\ge1).
```

These encode the restriction/projection moments needed at fixed `s`. Write the
projected moment table (called `G` internally) as

```math
\mathcal G_{j,r}^{(s)}
=\gamma_{s,r}\int_{-1}^{1}
P_j^{(\alpha,\beta)}(t)\Psi_{s,r}(t)\,dt.
```

The scale `gamma` converts the auxiliary Jacobi representation to the target
Jacobi projection; `_projection_scale` implements the normalization from
Jacobi norms and projection constants. These integrals use ordinary `dt`;
the target Jacobi weight enters the projection normalization.

The default production backend is `G_backend=:recurrence`. Its first two
Jacobi layers use the scalar moments

```math
J_{0,r}^{(s)}=\int_{-1}^{1}\Psi_{s,r}(t)\,dt,
\qquad
M_r^{(s)}=\int_{-1}^{1}z\Psi_{s,r}(t)\,dt.
```

They initialize the `j=0` and `j=1` layers before applying the projection
scale. With fixed `s` suppressed, the recurrence initialization is

```math
\begin{aligned}
r=0:&\quad J_{0,0}=\frac{2}{s+1},
&&M_0=\frac{2}{s+2},\\
r=1:&\quad J_{0,1}=-\frac{4}{(s+1)(s+2)},
&&M_1=-\frac{4}{(s+2)(s+3)}.
\end{aligned}
```

These explicit seeds start the stable adjacent-ratio recurrence used for
`r >= 2`:

```math
\frac{J_{0,r}}{J_{0,r-1}}
=-\frac{r}{r-1}\frac{\lambda+s+r-1}{s+r+1},
\qquad
\frac{M_r}{M_{r-1}}
=-\frac{r}{r-1}\frac{\lambda+s+r-2}{s+r+2}.
```

The code multiplies the previous moments by these ratios, avoiding monomial
expansion. The first Jacobi polynomial gives the next layer:

```math
P_1^{(\alpha,\beta)}(t)=(\lambda+2)z-(\beta+1),
\qquad
J_{1,r}=(\lambda+2)M_r-(\beta+1)J_{0,r}.
```

Higher `j` layers follow by combining Jacobi multiplication recurrences for
the target and auxiliary polynomials. With `init_layer_backend=:auto`, the
initial layers use log-gamma evaluation when `alpha+beta > 2`, and recurrence
initialization otherwise. `G_backend=:moments` is an alternative direct
moment-evaluation path for validation or use in place of the forward
recurrence; it is not the default.

The kernel and moment tables are compressed along `j` before matrix assembly:

```math
(C\mathcal G)_s(i,r)=\sum_{j=0}^{m}C_{ij}\mathcal G_{j,r}^{(s)}.
```

These cached `CG` tables avoid repeating the full kernel-mode summation for
each matrix entry. To obtain the intermediate coefficient in row `ell`,
column `k` of `Phi`, product linearization supplies mode `s`, the compressed
table supplies its restriction moment at `r=k-s`, and division by the squared Jacobi norm of output mode
`ell` completes the coefficient projection. Tail assembly evaluates only the
entries needed for the requested band and its adjacent intermediate rows.
The final relation is

```math
B=\frac12(I+M_x)\Phi.
```

Here `M_x` is the tridiagonal coefficient matrix for multiplication by `x` in
the target Jacobi basis. Assembly retains the necessary neighboring `Phi`
rows before truncating the output to modes `0:N`.

For non-Legendre Jacobi parameters, the final matrix is almost banded: the
constructor explicitly combines dense leading rows with a symmetric banded
tail. The defaults are `n_top = min(size(C,1), N+1)` and half-bandwidth
`min(size(C,1), N)`. Here symmetric band means equal upper and lower bandwidths,
not a symmetric matrix.

The dense top block has a separate `(s,k)` recurrence, rather than an
extrapolation of the tail recurrence. Its unscaled quantities are

```math
T_{j,k}^{s}=\int_{-1}^{1}P_j^{(\alpha,\beta)}(t)
\left(\frac{1+t}{2}\right)^s
P_{k-s}^{(-1,\lambda+2s+1)}(t)\,dt,
\qquad k\ge s.
```

These auxiliary moments cover all input columns needed by the leading rows.
The implemented recurrence advances in `s`:

```math
(k+s+\lambda+1)T_{j,k}^{s+1}
+(k-s-2)T_{j,k-1}^{s+1}
=(k-s)T_{j,k}^{s}
+(k+s+\lambda+1)T_{j,k-1}^{s}.
```

It builds successive layers from the `s=0` layer; projection then uses

```math
\mathcal H_{j,k}^{s}=\rho_{s,k}T_{j,k}^{s}.
```

Here the projected-moment scale `rho` is the factor `kappa` above, before
division by the Jacobi norm. After kernel compression, the same product
linearization and norm division assemble dense top `Phi`, then `(I+M_x)/2`.

The production `s=0` initialization is the mixed Jacobi recurrence in
`_tilde_H0_mixed!`, with the same recurrence/log-gamma initialization selection.
A BigFloat initialization helper remains in the source, but it is not the
default construction; `init_precision` is retained primarily for API
compatibility. General Jacobi construction does not require BigFloat.

#### Legendre specialization

The Legendre implementation specializes the same two structural identities to

```math
\alpha=\beta=0,\qquad P_q^{(0,0)}=P_q.
```

VoltOpt uses the dedicated `src/VoltMat/voltmat_leg.jl` implementation to
exploit this case's additional symmetry and strict bandedness. Its restriction
factor and parity-restricted product linearization are

```math
\begin{gathered}
\left(\frac{1+t}{2}\right)^i P_{j-i}^{(-1,2i+1)}(t),\\
P_p(x)P_q(x)=\sum_{a=0}^{\min(p,q)}
\mu_{p,q,p+q-2a}^{(0,0)}P_{p+q-2a}(x).
\end{gathered}
```

The prefactor is given above; product modes have the parity of `p+q`. These
structures underlie `H`/`CH` and strict-band assembly. The transformed kernel is

```math
\widehat K(x,t)\approx
\sum_{i=0}^{n}\sum_{j=0}^{m}C_{ij}P_i(x)P_j(t).
```

Here `C` is obtained by Gauss--Legendre interpolation of the same Duffy kernel.
The dedicated construction uses auxiliary moments denoted by `H` below, in
the normalization of `voltmat_leg.jl`. For each fixed `s`, it builds
`H[j+1,r+1] = H_{j,s+r}^s`. The `j=0` layer is explicit in `_H0_entry`:

```math
H_{0,s}^{s}=\frac{2}{s+1}.
```

This is the separate `r=0` seed, including `s=0`. For even offsets `r=2p >= 2`,
the seed is

```math
H_{0,s+2p}^{s}
=\frac{2(2s+2p+1)}{(s+2p)(s+2p+1)}.
```

This branch is used only for positive even offsets. For odd offsets
`r=2p+1`, with `p >= 0`, the seed is

```math
H_{0,s+2p+1}^{s}
=-\frac{4(s+p+1)}{(s+2p+1)(s+2p+2)}.
```

Together these initialize the full `j=0` layer. Subsequent layers are
generated recurrently using the Legendre relation

```math
P_{j+1}(t)=\frac{2j+1}{j+1}tP_j(t)
-\frac{j}{j+1}P_{j-1}(t).
```

`_fill_H_for_s!` combines this with multiplication by `t` for the auxiliary
Jacobi polynomials. The resulting moments are compressed with the kernel:

```math
CH_s(i,r)=\sum_{j=0}^{m}C_{ij}H_{j,s+r}^{s}.
```

The implementation provides dense and ragged caches for these quantities;
`ch_backend=:auto` selects storage using problem size and estimated memory.
For kernel mode `i` and output mode `ell`, the product formula above gives
`s=i+ell-2a`. Its Legendre product-linearization (Adams) coefficients are
generated by recurrence and cached; they are not a separate numerical
integration formula.
`alpha_backend=:auto` likewise chooses dense or ragged coefficient storage.
Combining the coefficients with `CH` and the projection normalization gives
the intermediate matrix `Phi`.

Finally, multiplication by `(1+x)/2` uses

```math
xP_q(x)=\frac{q+1}{2q+1}P_{q+1}(x)
+\frac{q}{2q+1}P_{q-1}(x).
```

Reading this as a coefficient transformation, output row `ell` receives
contributions from intermediate rows `ell-1`, `ell`, and `ell+1`:

```math
B_{\ell k}
=\frac12\Phi_{\ell k}
+\mathbf 1_{\ell\ge1}\frac{\ell}{2(2\ell-1)}\Phi_{\ell-1,k}
+\frac{\ell+1}{2(2\ell+3)}\Phi_{\ell+1,k}.
```

This is the row indexing used by `_assemble_symband`, including the separate
lower boundary at `ell=0` and any extra intermediate row needed at the upper
truncation boundary.

Under the Duffy-compatible transformed-kernel construction used here, the
Legendre specialization has no dense leading correction. The general Jacobi
constructor therefore sets `n_top = 0` for `alpha=beta=0` and identifies the
structure as `:strict_band`. The dedicated `voltmat_leg` constructs only the
symmetric band, with default half-bandwidth `size(C,1)` capped at `N`, and
further exploits the Legendre-specific `H` recurrence, `CH` compression, and
product linearization. This structure statement concerns coefficients of the
Duffy-transformed kernel used here; it does not assert that an arbitrary
tensor-product matrix `C` represents a smooth original Volterra kernel.

#### Matrix structure and implementation notes

The three paths share the Duffy formulation but have different assembly
structures:

- Chebyshev matrices are almost banded. Recurrence construction and
  Toeplitz--Hankel convolution support the FFT-accelerated assembly discussed
  in the paper; this is the default `assembly=:fft` path.
- General non-Legendre Jacobi matrices have a dense top and banded tail.
  There is no corresponding Toeplitz--Hankel convolution in general. The
  current implementation organizes work through recurrent product
  linearization, recurrent/compressed moment tables, and sparse assembly.
- Legendre matrices are strictly banded. For fixed kernel degree, storage and
  matrix application scale linearly with `N`. This dedicated construction
  does not use the Chebyshev Toeplitz--Hankel FFT path.

For equal transformed-kernel degrees `m=n`, the paper-level assembly estimates
are

```math
\text{Chebyshev: }O(n^2\log n\,N),
\qquad
\text{generic Jacobi: }O(n^3N).
```

These distinguish the FFT structure from generic Jacobi assembly; the cache
and recurrence organization described above is not a claim of a new, better
general-Jacobi asymptotic bound.

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

The main `ex6_nonlinear_picard.jl` experiment solves a manufactured nonlinear
Volterra--Urysohn equation by plain Picard/fixed-point iteration. Its fixed
manufactured right-hand side is constructed independently by adaptive
Gauss--Legendre quadrature. The working polynomial degree starts at `N0=32`;
if a complete nonlinear update is not sufficiently resolved, the degree is
refined without advancing the Picard state. The CSV records intermediate
attempts/refinements as well as accepted Picard steps. `ex6_plot.m` uses this
history to plot the error together with the current working degree `N`.

`ex6_fixed10_varyN.jl` is a separate fixed-10-Picard degree study, sweeping
`N = 10, 20, 40, 80, 160, 320, 640, 1024, 2048, 4096`.

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

The current diagnostic uses `K(x,y)=exp(-x*y)` and Jacobi parameters
`(alpha,beta)=(0.3,0.2)`. Its structure plots visualize the almost-banded
Chebyshev, strictly banded Legendre, and top-dense/band-tail general Jacobi
constructions described above. They provide diagnostic confirmation of the
computed structure, not a mathematical proof.

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

Most MATLAB paper figures use the shared helpers
`tools/voltopt_plot_style.m` and `tools/voltopt_export_figure.m` to enforce
common fonts, axes geometry, figure canvas sizes, and PDF/PNG export settings.
The supported fixed layout classes include `standard`, `standard-fixed`,
`square`, `phase`, and `multipanel`. Individual example/benchmark scripts
remain responsible for data content and graphical elements; the helpers
provide the common paper geometry, style, and export behavior.

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
