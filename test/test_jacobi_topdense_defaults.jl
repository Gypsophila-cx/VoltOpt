using Test

const PROJECT_ROOT = dirname(@__DIR__)

include(joinpath(PROJECT_ROOT, "src", "VoltOpt.jl"))
using .VoltOpt

@testset "Jacobi top-dense default size" begin
    N = 80
    alpha = 0.3
    beta = 0.4
    C = randn(13, 7)

    B, info = VoltOpt.voltmat_jacobi_full(C, N, alpha, beta; return_info=true)

    @test info.n_top == size(C, 1)
    @test info.bandwidth == size(C, 1)
    @test info.structure == :topdense_bandtail
    @test size(B) == (N + 1, N + 1)

    Btop = VoltOpt.jacobi_volterra_topdense_from_C(C, N, alpha, beta)
    @test size(Btop, 1) == size(C, 1)
    @test size(Btop, 2) == N + 1
end

@testset "Public voltmat_jacobi defaults to full top-dense matrix" begin
    N = 80
    alpha = 0.3
    beta = 0.4
    C = randn(13, 7)

    B, info = VoltOpt.voltmat_jacobi(C, N, alpha, beta; return_info=true)
    Bfull, info_full = VoltOpt.voltmat_jacobi_full(C, N, alpha, beta; return_info=true)

    @test B == Bfull
    @test info.n_top == size(C, 1)
    @test info.bandwidth == size(C, 1)
    @test info.structure == :topdense_bandtail
    @test info == info_full
end

@testset "Jacobi top-dense default truncation for small N" begin
    N = 5
    alpha = 0.3
    beta = 0.4
    C = randn(13, 7)

    B, info = VoltOpt.voltmat_jacobi_full(C, N, alpha, beta; return_info=true)

    @test info.n_top == N + 1
    @test info.bandwidth == N
    @test size(B) == (N + 1, N + 1)
end

@testset "Legendre remains strictly banded" begin
    N = 80
    alpha = 0.0
    beta = 0.0
    C = randn(13, 7)

    B, info = VoltOpt.voltmat_jacobi_full(C, N, alpha, beta; return_info=true)

    @test info.n_top == 0
    @test info.bandwidth == size(C, 1)
    @test info.structure == :strict_band
    @test size(B) == (N + 1, N + 1)

    Btop = VoltOpt.jacobi_volterra_topdense_from_C(C, N, alpha, beta)
    @test size(Btop, 1) == 0
    @test size(Btop, 2) == N + 1
end

@testset "Explicit n_top and bandwidth are respected" begin
    N = 80
    alpha = 0.3
    beta = 0.4
    C = randn(13, 7)

    B, info = VoltOpt.voltmat_jacobi_full(
        C,
        N,
        alpha,
        beta;
        n_top=4,
        bandwidth=6,
        return_info=true,
    )

    @test info.n_top == 4
    @test info.bandwidth == 6
    @test size(B) == (N + 1, N + 1)
end
