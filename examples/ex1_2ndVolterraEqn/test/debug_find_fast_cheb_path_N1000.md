# debug_find_fast_cheb_path_N1000

## Git Status
```text
 M Manifest.toml
 M Project.toml
 M examples/data/ex1_aging_maxwell_abs_error.csv
 M examples/figures/vide_low_order_cheb_leg_errors.pdf
 D examples/warmup/bench_gutleb_olver_vs_voltmat.jl
 D examples/warmup/plot_vide_low_order_accuracy_compare.jl
 D examples/warmup/solve_vide_low_order_cheb_leg.jl
 M others/GutlebOlver/GO_sparse_workflow_111.md
 M others/GutlebOlver/Project.toml
 M others/GutlebOlver/README.md
 M others/GutlebOlver/VOLTOPT_INTEGRATION_STATUS.md
 M others/GutlebOlver/examples/solve_section522_k100_sparse.jl
 D others/GutlebOlver/src/SparseGOWorkflow111.jl
 D others/GutlebOlver/src/triQEgen.jl
 D others/GutlebOlver/src/triVolterraFullKernel_strictband.jl
 M others/GutlebOlver/src/workflows.jl
 M others/GutlebOlver/test/runtests.jl
 M others/GutlebOlver/test/test_section522_k100_sparse.jl
 M test/compare/gutleb_olver/README.md
 M test/compare/gutleb_olver/bench_section522_k100_sparse.jl
 M test/compare/gutleb_olver/bench_section522_k100_sparse_repeated.jl
?? examples/data/solve_vide_low_order_gutleb_olver.csv
?? examples/ex1_2ndVolterraEqn/compare_large_2nd_volterra.jl
?? examples/ex1_2ndVolterraEqn/debug_compare_cheb_call_paths.csv
?? examples/ex1_2ndVolterraEqn/debug_compare_cheb_call_paths.jl
?? examples/ex1_2ndVolterraEqn/debug_find_fast_cheb_path_N1000.jl
?? examples/ex1_2ndVolterraEqn/debug_gutleb_olver_rhs_projection.csv
?? examples/ex1_2ndVolterraEqn/debug_gutleb_olver_rhs_projection.jl
?? examples/ex1_2ndVolterraEqn/debug_piecewise_jacobi_projection.csv
?? examples/ex1_2ndVolterraEqn/debug_piecewise_jacobi_projection.jl
?? examples/ex1_2ndVolterraEqn/debug_reproduce_aging_vk_timing_N1000.csv
?? examples/ex1_2ndVolterraEqn/debug_reproduce_aging_vk_timing_N1000.jl
?? examples/ex1_2ndVolterraEqn/debug_trace_aging_tVK_definition.csv
?? examples/ex1_2ndVolterraEqn/debug_trace_aging_tVK_definition.jl
?? examples/ex1_2ndVolterraEqn/debug_trace_aging_tVK_definition.md
?? examples/ex1_2ndVolterraEqn/debug_voltmat_cheb_build_time.csv
?? examples/ex1_2ndVolterraEqn/debug_voltmat_cheb_build_time.jl
?? examples/ex1_2ndVolterraEqn/debug_voltmat_cheb_fft_assembly.csv
?? examples/ex1_2ndVolterraEqn/debug_voltmat_cheb_fft_assembly.jl
?? examples/ex1_2ndVolterraEqn/ex1_2nd_volterra_common.jl
?? examples/ex1_2ndVolterraEqn/ex1_2nd_volterra_large_compare.csv
?? examples/ex5_psespectra/ex5_plot.m
?? examples/ex5_psespectra/ex5_pseudospectrum.pdf
?? examples/warmup/compare_vide_low_order_error.jl
?? examples/warmup/compare_vide_low_order_timing.jl
?? examples/warmup/vide_low_order_compare_common.jl
?? examples/warmup/vide_low_order_error_N1_100.csv
?? examples/warmup/vide_low_order_error_decay.pdf
?? examples/warmup/vide_low_order_median_elapsed_time.pdf
?? examples/warmup/vide_low_order_timing_N10_1000_loggrid.csv
?? examples/warmup/warmup_plot.m
?? others/GutlebOlver/examples/solve_simple_secondkind_sparse.jl
?? others/GutlebOlver/src/GutlebOlver.jl
?? others/GutlebOlver/src/diagnostics.jl
?? others/GutlebOlver/src/equation_assembly.jl
?? others/GutlebOlver/src/operators.jl
?? others/GutlebOlver/src/rhs_projection.jl
?? others/GutlebOlver/src/triangular_kernel_projection.jl
?? others/GutlebOlver/src/volterra_strictband.jl
?? others/GutlebOlver/test/test_dense_path_scan.jl
?? others/GutlebOlver/test/test_operators.jl
?? others/GutlebOlver/test/test_rhs_projection.jl
?? others/GutlebOlver/test/test_simple_secondkind_sparse.jl
?? others/GutlebOlver/test/test_triangular_kernel_projection.jl
?? test/compare/gutleb_olver/bench_section522_k100_sparse_varyN.jl
?? test/compare/gutleb_olver/bench_section522_k100_sparse_vary_kerneldepth.jl
?? test/compare/gutleb_olver/data/section522_k100_sparse_varyN_results.csv
?? test/compare/gutleb_olver/data/section522_k100_sparse_vary_kerneldepth_results.csv
?? test/compare/gutleb_olver/data/warmup_gutleb_olver_results.csv
?? test/compare/gutleb_olver/figures/
?? test/compare/gutleb_olver/solve_warmup_with_gutleb_olver.jl
?? test/smooth_bench/
```

## git diff -- src/VoltMat/voltmat_cheb.jl
```diff
(no diff)
```

## git diff -- examples/ex1_2ndVolterraEqn/ex1_aging_maxwell.jl
```diff
(no diff)
```

## Loaded Methods
- function-K method: `voltmat_cheb(K, N::Int64; assembly, use_fft_top, tol, tail_half_bandwidth) @ Main.VoltMat_Cheb D:\Projects\VoltOpt\src\VoltMat\voltmat_cheb.jl:1594`
- precomputed-C method: `voltmat_cheb(C::Matrix{Float64}, N::Int64; assembly, use_fft_top, tol, tail_half_bandwidth) @ Main.VoltMat_Cheb D:\Projects\VoltOpt\src\VoltMat\voltmat_cheb.jl:1664`

## Precomputed C
- C size: `(28, 17)`
- t_C_build: `6.636900e-03` s
- C was built with `VoltMat_Cheb.adaptive_cheb2_coeffmat((x,y)->K(x,(1+x)*(1+y)/2-1))` via `ex1_build_adaptive_cheb_kernel_coeff_metadata()`.

## Timing Results
| N | path | assembly | uses K | uses C | C size | t_C_build | t_VK | nnz | top/l/u | err | status |
|---:|---|---|---:|---:|---|---:|---:|---:|---|---:|---|
| 810 | raw_function_default_fft | default_fft | true | false |  | 0.000000e+00 | 1.980621e+00 | 46154 | 26/28/18 | 1.614e-01 | success |
| 810 | raw_function_explicit_fft | fft | true | false |  | 0.000000e+00 | 2.117249e+00 | 46154 | 26/28/18 | 1.614e-01 | success |
| 810 | raw_function_compressed_rankupdate | compressed_rankupdate | true | false |  | 0.000000e+00 | 3.543530e-02 | 46153 | 26/28/18 | 1.614e-01 | success |
| 810 | raw_function_compressed | compressed | true | false |  | 0.000000e+00 | 7.609230e-02 | 46154 | 26/28/18 | 1.614e-01 | success |
| 810 | precomputed_C_fft | fft | false | true | 28x17 | 6.636900e-03 | 1.883721e+00 | 46154 | 26/28/18 | 1.614e-01 | success |
| 810 | precomputed_C_compressed_rankupdate | compressed_rankupdate | false | true | 28x17 | 6.636900e-03 | 3.088190e-02 | 46153 | 26/28/18 | 1.614e-01 | success |
| 810 | ex1_aging_current | default_fft | true | false |  | 0.000000e+00 | 2.023409e+00 | 46154 | 26/28/18 | 1.614e-01 | success |
| 1000 | raw_function_default_fft | default_fft | true | false |  | 0.000000e+00 | 2.443757e+00 | 56553 | 26/28/18 | 2.119e-01 | success |
| 1000 | raw_function_explicit_fft | fft | true | false |  | 0.000000e+00 | 2.528246e+00 | 56553 | 26/28/18 | 2.119e-01 | success |
| 1000 | raw_function_compressed_rankupdate | compressed_rankupdate | true | false |  | 0.000000e+00 | 5.043790e-02 | 56552 | 26/28/18 | 2.119e-01 | success |
| 1000 | raw_function_compressed | compressed | true | false |  | 0.000000e+00 | 1.014473e-01 | 56554 | 26/28/18 | 2.119e-01 | success |
| 1000 | precomputed_C_fft | fft | false | true | 28x17 | 6.636900e-03 | 2.446953e+00 | 56553 | 26/28/18 | 2.119e-01 | success |
| 1000 | precomputed_C_compressed_rankupdate | compressed_rankupdate | false | true | 28x17 | 6.636900e-03 | 4.114480e-02 | 56552 | 26/28/18 | 2.119e-01 | success |
| 1000 | ex1_aging_current | default_fft | true | false |  | 0.000000e+00 | 2.513928e+00 | 56553 | 26/28/18 | 2.119e-01 | success |
| 1214 | raw_function_default_fft | default_fft | true | false |  | 0.000000e+00 | 2.983562e+00 | 68109 | 26/28/18 | 8.156e-02 | success |
| 1214 | raw_function_explicit_fft | fft | true | false |  | 0.000000e+00 | 2.992041e+00 | 68109 | 26/28/18 | 8.156e-02 | success |
| 1214 | raw_function_compressed_rankupdate | compressed_rankupdate | true | false |  | 0.000000e+00 | 6.126170e-02 | 68108 | 26/28/18 | 8.156e-02 | success |
| 1214 | raw_function_compressed | compressed | true | false |  | 0.000000e+00 | 1.268250e-01 | 68110 | 26/28/18 | 8.156e-02 | success |
| 1214 | precomputed_C_fft | fft | false | true | 28x17 | 6.636900e-03 | 2.952225e+00 | 68109 | 26/28/18 | 8.156e-02 | success |
| 1214 | precomputed_C_compressed_rankupdate | compressed_rankupdate | false | true | 28x17 | 6.636900e-03 | 4.388360e-02 | 68108 | 26/28/18 | 8.156e-02 | success |
| 1214 | ex1_aging_current | default_fft | true | false |  | 0.000000e+00 | 2.966152e+00 | 68109 | 26/28/18 | 8.156e-02 | success |

## Fastest Path By N
- N=810: `precomputed_C_compressed_rankupdate` with `t_VK=3.088190e-02` s
- N=1000: `precomputed_C_compressed_rankupdate` with `t_VK=4.114480e-02` s
- N=1214: `precomputed_C_compressed_rankupdate` with `t_VK=4.388360e-02` s

## Profile
Raw default profile saved to `debug_find_fast_cheb_path_N1000_profile.txt`.

## Interpretation
- The historical attachment's tens-of-milliseconds `t_VK` near N=1000 cannot correspond to the current raw function-K default FFT path if this run's timings are representative.
- If a precomputed-C compressed path lands near that scale, it is a candidate for the historical timing boundary: fixed kernel coefficients outside the N loop, then VK assembly only.
- Do not use the slow `precomputed_C_fft` path with an untrimmed or incompatible C-interface as evidence that all FFT assembly is slow.
