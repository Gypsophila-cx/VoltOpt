using BenchmarkTools
using BandedMatrices
using Printf
using SparseVolterraNew

Kexp(x, y) = exp(-x * y)

function bench_one(N::Integer, kerneldepth::Integer; flip::Bool=true)
    coeffs = build_gutleb_kernel_coefficients(Kexp, kerneldepth; flip=flip)
    A = build_gutleb_operator_from_coefficients(coeffs, N; flip=flip)
    basic_ok = operator_basic_check(A, N)
    stats = matrix_sparse_stats(A)
    lower_band, upper_band = bandwidths(A)

    GC.gc()
    trial = @benchmark build_gutleb_operator_from_coefficients($coeffs, $N; flip=$flip) samples=3 evals=1
    construct_time = minimum(trial).time / 1.0e9

    @printf(
        "N=%d kerneldepth=%d construct_time_sec=%.6g size=%dx%d nnz=%d bandwidth=(%d,%d) basic_check=%s\n",
        N,
        kerneldepth,
        construct_time,
        size(A, 1),
        size(A, 2),
        stats.nnz,
        lower_band,
        upper_band,
        string(basic_ok),
    )
end

function main()
    for N in (16, 32, 64, 128)
        bench_one(N, 6)
    end
end

main()
