using ApproxFun
using BandedMatrices
using GutlebOlver
using Test

@testset "basic sparse operators" begin
    N = 12

    QE = triQEygenP01(N)
    @test QE isa BandedMatrix
    @test safe_bandwidth(QE) == (0, 0)

    R = reflectPabtoPba(N)
    @test R isa BandedMatrix
    @test safe_bandwidth(R) == (0, 0)

    W = WLoweringP01P00(N)
    @test W isa BandedMatrix
    @test safe_bandwidth(W) == (1, 0)

    V = triVolterraFullKernelOpP01_strictband_from_coeffs(
        triangular_kernel_coeffs_from_plan(precompute_triangular_kernel_plan(1, true), (x, y) -> 1.0),
        N,
        true,
    )
    C01 = apply_conversion_00_to_01_strictband(apply_reflect_lowering_strictband(V, N), N)
    C12 = apply_conversion_00_to_12_strictband(apply_reflect_lowering_strictband(V, N), N)
    @test C01 isa BandedMatrix
    @test C12 isa BandedMatrix
    @test safe_bandwidth(C01)[1] <= 2
    @test safe_bandwidth(C12)[1] <= 2

    D = build_derivative_P01_to_P12(N)
    f = Fun(x -> x^2, GutlebOlver.SIMPLE_SOLUTION_SPACE, N)
    dcoeff = D * GutlebOlver.pad_to_length(f.coefficients, N)
    df = Fun(GutlebOlver.SECTION522_OUTPUT_SPACE, dcoeff)
    @test maximum(abs(df(x) - 2x) for x in range(0.0, 1.0; length=40)) < 1e-10

    row = build_initial_evaluation_row(N)
    g = Fun(x -> 1.0 + 2.0 * x + x^2, GutlebOlver.SIMPLE_SOLUTION_SPACE, N)
    gcoeff = GutlebOlver.pad_to_length(g.coefficients, N)
    @test abs(sum(row[j] * gcoeff[j] for j = 1:N) - 1.0) < 1e-12
end
