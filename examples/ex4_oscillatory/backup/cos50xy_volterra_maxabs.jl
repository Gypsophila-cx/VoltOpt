# cos50xy_volterra_maxabs.jl
#
# Construct the Gutleb--Olver triangular-basis Volterra matrix
# for K(x,y) = cos(50*x*y), and print max(abs(V)) for N = 500.
#
# Put this file in:
#   examples/warmup/cos50xy_volterra_maxabs.jl
#
# Run from VoltOpt root:
#   julia --project=. examples/warmup/cos50xy_volterra_maxabs.jl

using LinearAlgebra
using Printf
using SparseArrays

include("../warmup/vide_low_order_compare_common.jl")

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------

N = 500

# cos(50xy) is more oscillatory than the warmup example.
# You can reduce this to 15 to match the original warmup setting,
# but 80 is safer for the triangular expansion.
kerneldepth = 80

# ------------------------------------------------------------
# Original Volterra operator on [-1,1]:
#
#   (V_K u)(x) = int_{-1}^x K(x,y) u(y) dy,
#   K(x,y) = cos(50*x*y).
#
# Map x = 2s - 1, y = 2r - 1, dy = 2 dr:
#
#   (V_K u)(2s-1)
#     = int_0^s 2*K(2s-1,2r-1) U(r) dr.
#
# So the triangular-kernel expansion should use
#
#   K_go(s,r) = 2*cos(50*(2s-1)*(2r-1)).
# ------------------------------------------------------------

from_go_s(s) = 2.0s - 1.0

K_original(x, y) = cos(50.0 * x * y)

K_go(s, r) = 2.0 * K_original(from_go_s(s), from_go_s(r))

# ------------------------------------------------------------
# 1. Expand the mapped kernel on the triangular region.
# 2. Construct the raw Gutleb--Olver Volterra coefficient matrix.
# ------------------------------------------------------------

@printf("N           = %d\n", N)
@printf("kerneldepth = %d\n", kerneldepth)

kernel_coeffs = nothing
Vraw = nothing

t_coeff = @elapsed begin
    plan = precompute_triangular_kernel_plan(kerneldepth, true)
    global kernel_coeffs = triangular_kernel_coeffs_from_plan(plan, K_go)
end

t_matrix = @elapsed begin
    global Vraw = triVolterraFullKernelOpP01_strictband_from_coeffs(
        kernel_coeffs,
        N,
        true,
    )
end

maxabs_Vraw = matrix_maxabs(Vraw)

@printf("kernel_coeff_time       = %.6e sec\n", t_coeff)
@printf("volterra_construct_time = %.6e sec\n", t_matrix)
@printf("matrix_type             = %s\n", matrix_type_name(Vraw))
@printf("matrix_size             = %d x %d\n", size(Vraw, 1), size(Vraw, 2))
@printf("matrix_nnz              = %d\n", local_safe_nnz(Vraw))
@printf("maxabs_Vraw             = %.16e\n", maxabs_Vraw)