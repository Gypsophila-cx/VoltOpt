using BandedMatrices
using SparseVolterraNew
using Test

Kexp(x, y) = exp(-x * y)

@testset "strict-banded construction smoke" begin
    for N in (16, 32, 64)
        kerneldepth = 6
        config = GutlebOlverConfig(N=N, kerneldepth=kerneldepth, flip=true)
        A = build_gutleb_operator(Kexp, config)
        @test size(A) == (N, N)
        @test all(isfinite, A)
        @test operator_basic_check(A, N)

        lower_band, upper_band = bandwidths(A)
        theory = theoretical_bandwidth_P01(kerneldepth; flip=true)
        @test lower_band <= theory.lower_band
        @test upper_band <= theory.upper_band

        stats = matrix_sparse_stats(A)
        @test stats.nnz > 0
        @test stats.lower_band <= theory.lower_band
        @test stats.upper_band <= theory.upper_band
    end
end

@testset "precompute and assembly-only interface" begin
    coeffs = build_gutleb_kernel_coefficients(Kexp, 5; flip=true)
    @test coeffs isa TriangularKernelCoefficients
    A = build_gutleb_operator_from_coefficients(coeffs, 32)
    @test size(A) == (32, 32)
    @test all(isfinite, A)
end
