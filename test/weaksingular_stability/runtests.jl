using Test

module Ex7VoltMatTestSupport
include(joinpath(@__DIR__, "..", "..", "examples", "ex7_weaksingular", "ex7_VoltMat.jl"))
end

const Ex7Weak = Ex7VoltMatTestSupport

@testset "Weak_VoltMat stable error path" begin
    polynomial(z) = 1.25 - 0.5 * z + 0.75 * (2.0 * z^2 - 1.0)
    projected = Ex7Weak.cheb_project_lobatto(polynomial, 16, 64)
    @test projected[1:3] ≈ [1.25, -0.5, 0.75] atol = 8eps(Float64) rtol = 0.0
    @test maximum(abs, projected[4:end]) <= 8eps(Float64)

    for x in (-1.0, -0.75, 0.0, 0.625, 1.0)
        @test Ex7Weak.rhs_basis_factor_x(x, 1.0 / 3.0) == Ex7Weak.rhs_factor_x(x)
    end

    cN = [0.75, -0.2, 3.0e-16]
    cref = [0.75, -0.2, 2.0e-16, -4.0e-17]
    grid = Ex7Weak.composite_test_grid(101, 51)
    error, _ = Ex7Weak.grid_cauchy(cN, cref, grid, 1.0 / 3.0, 0.5)
    difference = -copy(cref)
    difference[1:length(cN)] .+= cN
    expected = maximum(
        s -> abs(Ex7Weak.eval_jfp_s(difference, s, 1.0 / 3.0, 0.5)),
        grid.s,
    )
    @test error == expected
end
