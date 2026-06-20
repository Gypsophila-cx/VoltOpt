using GutlebOlver
using Test

@testset "RHS projection" begin
    functions = (
        x -> 1.0,
        x -> x,
        x -> x^2,
        x -> 1.0 + 2.0 * x + 0.5 * x^2,
    )

    for f in functions
        rhs, info = build_go_rhs_coefficients(
            f,
            16;
            alpha=0,
            beta=1,
            method=:fixed_gauss_jacobi,
        )
        @test length(rhs) == 16
        @test info.validation_error < 1e-12
        @test info.status == "pass"
    end

    rhs, info = build_go_rhs_coefficients(
        x -> sin(x),
        32;
        alpha=0,
        beta=1,
        method=:fixed_gauss_jacobi,
    )
    @test length(rhs) == 32
    @test info.validation_error < 1e-12
    @test info.tail_norm < 1e-12

    for (alpha, beta) in ((1, 0), (2, 1))
        rhs, info = build_go_rhs_coefficients(
            x -> 1.0 + x + x^2,
            16;
            alpha=alpha,
            beta=beta,
            method=:fixed_gauss_jacobi,
        )
        @test length(rhs) == 16
        @test occursin("Jacobi", info.basis)
        @test info.validation_error < 1e-11
    end

    rhs, info = build_go_rhs_coefficients(
        [1.0, 2.0, 0.5],
        16;
        alpha=0,
        beta=1,
        method=:exact_polynomial,
    )
    @test length(rhs) == 16
    @test info.validation_error < 1e-12
end
