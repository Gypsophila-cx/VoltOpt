using BenchmarkTools
using SparseArrays
using LinearAlgebra

# 路径按脚本位于 examples/ex1_2ndVolterraEqn 下设置
ROOT = normpath(joinpath(@__DIR__, "..", ".."))

include(joinpath(ROOT, "src", "VoltMat", "voltmat_cheb.jl"))

const N = 1000

# ============================================================
# Copy these definitions from ex1_aging_maxwell.jl
# ============================================================

# Example:
# const tol = 1e-14
# K = (x, y) -> ...

tol = 1e-14

tau_xy(x, y) = 1.5 + x - 0.5 * y

function K(x, y)
    τ = tau_xy(x, y)
    return exp(-(x - y) / τ) / τ
end

# ============================================================

function build_VK(K, N, tol)
    return VoltMat_Cheb.voltmat_cheb_square(K, N - 1; tol=tol)
end

function matrix_nnz(A)
    return isa(A, SparseMatrixCSC) ? nnz(A) : count(!iszero, A)
end

println("Benchmark ex1 aging Maxwell")
println("N = ", N)
println("tol = ", tol)
println("method = ", @which VoltMat_Cheb.voltmat_cheb_square(K, N - 1; tol=tol))
println()

println("First-call timing with @timed:")
GC.gc()
first_run = @timed build_VK(K, N, tol)
VK = first_run.value

println("time       = ", first_run.time, " sec")
println("alloc      = ", first_run.bytes / 1024^2, " MiB")
println("gc time    = ", first_run.gctime, " sec")
println("size       = ", size(VK))
println("nnz        = ", matrix_nnz(VK))
println("norm       = ", norm(VK))
println()

println("BenchmarkTools timing after warmup:")
bench = @benchmark build_VK($K, $N, $tol) seconds=20 samples=20 evals=1

display(bench)
println()
println("minimum = ", minimum(bench).time / 1e9, " sec")
println("median  = ", median(bench).time / 1e9, " sec")
println("mean    = ", mean(bench).time / 1e9, " sec")