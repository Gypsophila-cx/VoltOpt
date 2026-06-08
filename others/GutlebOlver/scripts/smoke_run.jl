using BandedMatrices
using SparseVolterraNew

Kexp(x, y) = exp(-x * y)

N = 32
kerneldepth = 6
config = GutlebOlverConfig(N=N, kerneldepth=kerneldepth, flip=true)
A = build_gutleb_operator(Kexp, config)
stats = matrix_sparse_stats(A)

println("SparseVolterraNew smoke test")
println("N = ", N)
println("kerneldepth = ", kerneldepth)
println("size = ", size(A))
println("bandwidths = ", bandwidths(A))
println("nnz = ", stats.nnz)
println("basic_check = ", operator_basic_check(A, N))
