# debug_stepwise_from_aging_N1000

- N: 1000
- tol: 2.220446049250313e-16
- CSV: D:\Projects\VoltOpt\examples\ex1_2ndVolterraEqn\debug_stepwise_from_aging_N1000.csv
- Profile note: Step 0 already slow; profiled Step 0 only.
- Slowdown note: Step 0 is already slow, so the slowdown is not introduced by common/include/wrapper steps.

## VK timings

| step | name | assembly | K source | C | t_VK | status |
|---:|---|---|---|---|---:|---|
| 0 | aging_minimal_copy | default | direct_copy_from_ex1_aging_maxwell |  | 2.479406 | success |
| 1 | common_include_aging_K | default | direct_copy_from_ex1_aging_maxwell_after_common_include |  | 2.493674 | success |
| 2 | common_ex1_kernel_default | default | ex1_2nd_volterra_common.ex1_kernel |  | 2.722429 | success |
| 3 | absorbed_kernel_as_function_K_default | default | ex1_absorbed_kernel_passed_to_function_K_interface |  | 2.664043 | success |
| 40 | common_ex1_kernel_explicit_fft | fft | ex1_2nd_volterra_common.ex1_kernel |  | 2.674131 | success |
| 41 | common_ex1_kernel_explicit_compressed_rankupdate | compressed_rankupdate | ex1_2nd_volterra_common.ex1_kernel |  | 0.045336 | success |
| 42 | common_ex1_kernel_explicit_compressed | compressed | ex1_2nd_volterra_common.ex1_kernel |  | 0.091894 | success |
| 50 | current_cheb_builder_VK_only_aging_maxwell_fft | default | ex1_solve_cheb aging_maxwell_fft VK branch |  | 2.547566 | success |
| 51 | current_cheb_builder_VK_only_manual_C_fft | fft | ex1_solve_cheb manual_C_fft VK branch | 32x32 | 4.757515 | success |
| 52 | current_cheb_builder_VK_only_compressed_rankupdate | compressed_rankupdate | ex1_solve_cheb compressed_rankupdate VK branch | 32x32 | 0.085756 | success |
| 60 | postprocess_from_current_compressed_rankupdate | n/a | postprocess_from_current_compressed_rankupdate_VK | missingxmissing | NaN | success |

## Notes
- step 0 aging_minimal_copy: No common include; no RHS/solve/error/bandwidth conversion.
- step 1 common_include_aging_K: Only common include was added; VK still uses the direct aging K copy.
- step 2 common_ex1_kernel_default: K is replaced by ex1_kernel from common.
- step 3 absorbed_kernel_as_function_K_default: Diagnostic wrapper test. Passing the absorbed kernel to the function-K interface applies VoltMat_Cheb's internal absorption again.
- step 40 common_ex1_kernel_explicit_fft: Same as common ex1_kernel, but assembly=:fft is explicit.
- step 41 common_ex1_kernel_explicit_compressed_rankupdate: Function-K interface with internal C construction and compressed rank-update assembly.
- step 42 common_ex1_kernel_explicit_compressed: Function-K interface with internal C construction and compressed no-FFT Clenshaw tail.
- step 50 current_cheb_builder_VK_only_aging_maxwell_fft: Matches ex1_solve_cheb(:aging_maxwell_fft) VK-only branch. Adaptive C metadata is built outside but not passed to VK.
- step 51 current_cheb_builder_VK_only_manual_C_fft: Matches ex1_solve_cheb(:manual_C_fft) VK-only branch with fixed 32x32 absorbed-kernel C.
- step 52 current_cheb_builder_VK_only_compressed_rankupdate: Matches ex1_solve_cheb(:compressed_rankupdate) VK-only branch and reuses the fixed 32x32 C.
- step 60 postprocess_from_current_compressed_rankupdate: Postprocessing is timed separately from VK construction. max_abs_error=0.21190332455725702; residual_inf=1.1102230246251565e-15
