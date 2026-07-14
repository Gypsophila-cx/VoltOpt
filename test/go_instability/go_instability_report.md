# Gutleb-Olver Operator-Valued Clenshaw Instability

## 1. Problem and notation

The finite system is

```math
T_{d,M} z = b,\qquad V = P_{Nout,M} z.
```

`d` is the triangular kernel depth, `M` is the internal Clenshaw state size,
and `Nout` is the cropped output size.  The final output map extracts the
level-1 first block, applies weighted lowering, reflection,
`Jacobi(0,0) -> Jacobi(0,1)` conversion, and crops the leading
`Nout x Nout` block.

All high-precision work in this directory uses `setprecision(BigFloat, 128)`.

## 2. Padding and finite-section error

The production sweep uses `Nout=72`, `d=8,16,24,32,36`, and
`M=Nout+d+8`.  These are padding-converged against the largest tested
`M=Nout+2d+8` for `mixed_cos20xy`:

| d | Nout | selected M | padding error to max |
|---:|---:|---:|---:|
| 8 | 72 | 88 | 0.000e0 |
| 16 | 72 | 96 | 0.000e0 |
| 24 | 72 | 104 | 0.000e0 |
| 32 | 72 | 112 | 0.000e0 |
| 36 | 72 | 116 | 0.000e0 |

Data: `data/go_padding_validation.csv`.  Figure:
`figures/go_padding_convergence.pdf`.

## 3. Independent construction of T and P

The construction validation builds four separate systems from formulas:

```text
T64  = Float64 analytic reconstruction
T128 = BigFloat 128 analytic reconstruction
P64  = Float64 analytic reconstruction
P128 = BigFloat 128 analytic reconstruction
```

`T128` and `P128` are not `BigFloat.(T64)` or `BigFloat.(P64)`.  The 128-bit
path independently rebuilds the Jacobi multiplication coefficients, triangle
recurrence blocks, `B_n`, `B_n^+`, preconditioned coupling blocks, lowering,
reflection, Jacobi conversion, and crop/extraction stages.  A lifted Float64
system is used only to represent the exact data produced by Float64
construction when isolating construction error from application rounding.

Data: `data/go_construction_validation.csv`.

## 4. Stability of T construction

Small explicit padding-respecting matrices use `Nout=d+2`,
`M=Nout+d+8`, `d=2,3,4,5`:

| d | M | relative T Frobenius error | error / eps(Float64) |
|---:|---:|---:|---:|
| 2 | 14 | 4.932e-17 | 2.221e-1 |
| 3 | 16 | 8.059e-17 | 3.629e-1 |
| 4 | 18 | 9.256e-17 | 4.168e-1 |
| 5 | 20 | 9.490e-17 | 4.274e-1 |

The largest local block relative construction error, excluding the explicitly
reported left-inverse defect matrix, is `1.689e-16` at
`preconditioned_second_coupling_C`, level 8.  This is `0.761 eps(Float64)`.

The largest recorded Float64 left-inverse defect is
`||B_n^+B_n-I||_F = 3.331e-16` at level 30.  Recomputing the same Float64 data
in 128-bit gives `2.923e-16`; the independently constructed 128-bit defect is
`5.878e-39`.

Production matrix-free probes give max-over-probes T action errors:

| d | M | construction action | application rounding | total |
|---:|---:|---:|---:|---:|
| 8 | 88 | 1.226e-16 | 1.604e-16 | 2.010e-16 |
| 16 | 96 | 1.397e-16 | 1.589e-16 | 2.086e-16 |
| 24 | 104 | 1.498e-16 | 1.660e-16 | 2.223e-16 |
| 32 | 112 | 1.591e-16 | 1.608e-16 | 2.278e-16 |
| 36 | 116 | 1.615e-16 | 1.744e-16 | 2.399e-16 |

Figure: `figures/go_T_construction_error_vs_depth.pdf`.

## 5. Stability of P construction

Explicit P construction is also at normal Float64 precision:

| d | M | relative P Frobenius error | error / eps(Float64) |
|---:|---:|---:|---:|
| 2 | 14 | 7.719e-17 | 3.476e-1 |
| 3 | 16 | 7.396e-17 | 3.331e-1 |
| 4 | 18 | 7.989e-17 | 3.598e-1 |
| 5 | 20 | 7.915e-17 | 3.565e-1 |

P stage validation for `d=36`, `Nout=72`, `M=116`:

| stage | construction action | application rounding | total |
|---|---:|---:|---:|
| extraction | 0.000e0 | 0.000e0 | 0.000e0 |
| lowering | 0.000e0 | 0.000e0 | 0.000e0 |
| reflection | 0.000e0 | 0.000e0 | 0.000e0 |
| conversion | 4.809e-17 | 5.668e-17 | 6.910e-17 |
| crop | 4.520e-17 | 5.534e-17 | 7.114e-17 |
| full P | 4.520e-17 | 5.534e-17 | 7.114e-17 |

Thus extraction, lowering, reflection, and crop as an index operation are exact
in this test.  The only visible P construction contribution comes from the
Jacobi conversion, still only `0.217 eps(Float64)` in action error.

Production max-over-probes P construction action reaches `5.450e-16`
(`2.455 eps(Float64)`) at `d=36`; total P action error is `6.758e-16`.
Figure: `figures/go_P_construction_error_vs_depth.pdf`.

## 6. Nonnormality

Small dense diagnostics show all tested eigenvalues remain at 1 while
singular values and the output-aware inverse degrade:

| d | dim(T) | sigma_min(T) | cond_2(T) | ||P T^{-1}||_2 |
|---:|---:|---:|---:|---:|
| 2 | 68 | 4.720e-1 | 4.488e0 | 6.255e-1 |
| 3 | 214 | 1.598e-1 | 2.081e1 | 1.373e0 |
| 4 | 510 | 4.105e-2 | 1.146e2 | 4.523e0 |
| 5 | 1030 | 9.344e-3 | 6.625e2 | 1.782e1 |

Data: `data/go_nonnormality_diagnostics.csv`.  Figures:
`go_sigma_min_vs_depth.pdf`, `go_condition_vs_depth.pdf`,
`go_output_gain_vs_depth.pdf`.

## 7. Backward and forward errors

For padding-converged `mixed_cos20xy`, `Nout=72`, `d=36`, `M=116`:

| quantity | value |
|---|---:|
| native Float64 vs 128 output error | 1.068e-4 |
| relative output error | 2.944e-4 |
| componentwise backward error | 1.572e-16 |
| backward error / eps(Float64) | 7.080e-1 |

The solve is backward stable at machine precision, but the forward map is
sensitive.

## 8. Exact error-source decomposition

For fixed-b isolation, the same Float64 coefficient vector is used and lifted,
so `Deltab=0`.  The identity

```math
P_hat z_hat - P_star z_star =
P_star T_star^{-1}(-Delta T z_hat)
+ P_star T_star^{-1} r_solve
+ P_star T_star^{-1} Delta b
+ Delta P z_hat
```

is verified with relative residual `2.719e-23`.

| contribution | norm | relative to true output error | relative to ||Vstar|| |
|---|---:|---:|---:|
| DeltaV_T | 3.413e-10 | 3.196e-6 | 9.409e-10 |
| DeltaV_solve | 1.068e-4 | 1.000e0 | 2.944e-4 |
| DeltaV_b | 0.000e0 | 0.000e0 | 0.000e0 |
| DeltaV_P | 2.166e-17 | 2.028e-13 | 5.971e-17 |

The dominant source is `solve_rounding`.  The measured solve-residual output
gain is `2.002e8`.  Figure:
`figures/go_error_contribution_breakdown.pdf`.

The full-pipeline high-precision kernel projection is not available in the
current Gutleb-Olver projection path; the CSV records
`status=high_precision_projection_unavailable`.  Therefore the final T/P
conclusion uses fixed-b isolation.

## 9. Layerwise generation and propagation of rounding errors

The state numbering is the numbering actually used by the strict-band code.
`level_index=code_index=K` runs from `1` to `d`, and the corresponding
triangular total degree is `K-1`.  Thus `K=1` (total degree 0) is closest to
the final output and supplies the block extracted by `P`; `K=d` (total degree
`d-1`) is closest to the highest retained kernel degree.  The recurrence is
executed from `K=d` down to `K=1`.  All three layerwise figures use total
degree, not code index, on the horizontal axis.

For the padding-converged primary case `d=36`, `Nout=72`, `M=116`, the largest
local residual and the largest propagated contribution occur at different
layers:

| diagnostic | code level K | total degree | local residual | actual output gain | output contribution |
|---|---:|---:|---:|---:|---:|
| largest local residual | 15 | 14 | 2.958e-13 | 1.120e6 | 3.313e-7 |
| largest actual gain and contribution | 35 | 34 | 1.894e-24 | 3.485e19 | 6.600e-5 |

The degree-34 source alone has contribution-norm ratio `0.617913` relative to
`||DeltaV_solve||=1.068052e-4`, alignment `0.872868`, and signed projection
fraction `0.539357`.  The terminal `K=36`, degree-35 equation is copied exactly
as `z_d=b_d` in this solve and has zero local residual; the most influential
nonzero source is therefore the next layer, not the terminal layer itself.

Propagation distance is decisive.  Across the shallowest nine layers, the
nonzero actual gains range from `6.131e-3` to `4.707e2`; across the deepest
nine layers they range from `4.743e14` to `3.485e19`.  Gain grows strongly but
not strictly monotonically with total degree, so the data do not support a
rule that the terminal or every successively deeper layer must dominate.
Figure `go_layer_output_gain.pdf` shows the measured layer-by-layer behavior.

The complete-error recurrence separates the inherited terms as
`-A_n e_{n+1}` and `-C_n e_{n+2}`.  Summing their step norms gives
`1.81545e-3` through the A channel and `7.45385e-5` through the C channel:
`96.056%` versus `3.944%`.  A is the dominant propagation channel for this
case.  These are norm-mass fractions for the geometric diagnostic, not an
additive output-error decomposition.

Layer directions show material cancellation.  The 17 positive signed
projections sum to `1.376168`, while the 18 negative projections sum to
`-0.376168`; their total is one to `2.939e-39`.  Consequently it would be
incorrect to add layer norms as percentages.  The cumulative matrix norm
first reaches 50%, 90%, and 99% of the final norm after respectively 2, 9,
and 10 deepest layers.  The deepest nine layers give `0.945620` of the final
norm, whereas the shallowest nine give only `1.612e-7`.  The cumulative curve
then overshoots and retreats (for example `1.047596` after 12 layers and
`0.939523` after 14), directly exposing enhancement and cancellation.

Every layer remains locally backward stable.  The maximum recorded
componentwise backward error is `2.402e-16`, or `1.082 eps(Float64)`, at
`K=25` (total degree 24).  The independently reevaluated residual blocks
reconstruct `rsolve` with relative error zero.  More importantly,
`sum_k DeltaV_k` reconstructs `DeltaV_solve` with relative error
`2.257e-38`, and the signed projection fractions sum to one with error
`2.939e-39`.

The small explicit systems also distinguish actual from worst-case layer
directions.  For example, at `d=5`, `K=4`, the actual gain is `8.096e-1`
against worst-case gain `6.514e0`, a ratio `0.1243`; actual residuals need not
align with the most sensitive singular direction of their layer.

The depth sweep shows the dominant propagated layer moving toward higher
total degree, although not monotonically at every intermediate depth:

| d | dominant residual degree | dominant contribution degree | max actual gain | layers for 50% / 90% / 99% |
|---:|---:|---:|---:|---:|
| 16 | 6 | 14 | 1.394e7 | 2 / 3 / 3 |
| 24 | 9 | 18 | 5.538e11 | 2 / 10 / 14 |
| 32 | 9 | 30 | 1.276e17 | 3 / 9 / 14 |
| 36 | 14 | 34 | 3.485e19 | 2 / 9 / 10 |

Thus the layer producing the largest raw residual stays in the middle of the
recurrence, while the layer dominating final error moves from degree 14 to
degree 34 as depth increases.  Data:
`data/go_layerwise_diagnostics.csv`.  Figures:
`go_layer_residual_contribution.pdf`, `go_layer_output_gain.pdf`, and
`go_layer_cumulative_error.pdf`.

## 10. Padding-converged depth sweep

With `Nout=72` and `M=Nout+d+8`:

| d | M | final output error | relative error | backward error / eps |
|---:|---:|---:|---:|---:|
| 8 | 88 | 9.087e-14 | 2.626e-13 | 4.894e-1 |
| 16 | 96 | 1.298e-9 | 3.569e-9 | 5.919e-1 |
| 24 | 104 | 3.488e-7 | 9.614e-7 | 6.381e-1 |
| 32 | 112 | 3.019e-5 | 8.324e-5 | 7.025e-1 |
| 36 | 116 | 1.068e-4 | 2.944e-4 | 7.080e-1 |

Data: `data/go_depth_validation.csv`.  Figure:
`figures/go_operator_error_vs_depth.pdf`.

## 11. Deterministic conclusion

1. `T_{d,M}` entries and local blocks are stably constructed in Float64 over
   the tested range.  Global explicit errors are below `0.43 eps(Float64)`,
   local block errors stay below `0.77 eps(Float64)`, and
   `B_n^+B_n-I` behaves at the expected rounding level.
2. `P_{Nout,M}` is stably constructed.  Extraction, lowering, reflection, and
   crop are exact in the tested action; Jacobi conversion is the only visible
   stage and remains below `0.22 eps(Float64)` in the stage action test.
3. The final error is dominated by Clenshaw solve rounding:
   `||DeltaV_solve|| = 1.068e-4`.  T construction contributes
   `3.413e-10`; P construction contributes `2.166e-17`; fixed-b isolation has
   `DeltaV_b=0`.
4. This is not evidence of an unstable T or P construction algorithm.  The
   tested behavior is stable construction of a forward-ill-conditioned map:
   a machine-level backward residual is amplified by `P T^{-1}`.
5. The layerwise decomposition identifies degree 34 as the primary propagated
   source even though degree 14 has the largest local residual.  A-channel
   inheritance supplies `96.056%` of the measured propagation norm mass, and
   signed layer projections show `37.617%` cancellation against the positive
   projected mass.

## 12. Scope

- Only 128-bit `BigFloat` references are used.
- No higher-than-128-bit experiment is used or retained.
- Full-pipeline high-precision kernel projection is unavailable in the current
  code path, so kernel-coefficient projection error is not independently
  decomposed here.
- No asymptotic exponential growth theorem is claimed.
- Conclusions are limited to the tested depths, padding rule, kernel, and
  output size.

## 13. Reproduction

Run all Julia validations:

```bash
julia --project=others/GutlebOlver test/go_instability/run_all_final_validations.jl
```

The intended MATLAB command is:

```bash
matlab -batch "run('test/go_instability/plot_go_instability.m')"
```

In this run, MATLAB failed during startup with a settings-plugin error, so the
checked-in figures were generated from the same CSV files with the retained
Julia `Plots.jl` fallback:

```bash
julia --project=@v1.11 test/go_instability/plot_go_instability.jl
```

## 14. Final file inventory

```text
test/go_instability/
  go_instability_core.jl
  run_padding_validation.jl
  run_construction_validation.jl
  run_nonnormality_validation.jl
  run_output_error_validation.jl
  run_layerwise_diagnostics.jl
  run_all_final_validations.jl
  plot_go_instability.m
  plot_go_instability.jl
  go_instability_report.md
  data/
    go_construction_validation.csv
    go_depth_validation.csv
    go_nonnormality_diagnostics.csv
    go_output_error_validation.csv
    go_padding_validation.csv
    go_layerwise_diagnostics.csv
  figures/
    go_condition_vs_depth.pdf
    go_condition_vs_depth.png
    go_error_contribution_breakdown.pdf
    go_error_contribution_breakdown.png
    go_operator_error_vs_depth.pdf
    go_operator_error_vs_depth.png
    go_output_gain_vs_depth.pdf
    go_output_gain_vs_depth.png
    go_P_construction_error_vs_depth.pdf
    go_P_construction_error_vs_depth.png
    go_padding_convergence.pdf
    go_padding_convergence.png
    go_sigma_min_vs_depth.pdf
    go_sigma_min_vs_depth.png
    go_T_construction_error_vs_depth.pdf
    go_T_construction_error_vs_depth.png
    go_layer_residual_contribution.pdf
    go_layer_residual_contribution.png
    go_layer_output_gain.pdf
    go_layer_output_gain.png
    go_layer_cumulative_error.pdf
    go_layer_cumulative_error.png
```
