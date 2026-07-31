using LinearAlgebra
using SparseArrays
using Test
using VoltOpt

include("pantograph_period_common.jl")
using .PantographPeriod

const FIGURE2_T_TEST = 1000.0
const FIGURE2_Q_TEST = 1.0 / 6.0

@testset "Iserles Figure 2 ultraspherical solver" begin
    @testset "hard degree limit and coefficient operators" begin
        @test MAX_SOLVE_DEGREE == 5000
        @test_throws ArgumentError ultraspherical_operators(5001)
        N = 12
        operators = ultraspherical_operators(N)
        @test all(
            size(operator) == (N + 1, N + 1) for
            operator in values(operators)
        )
        @test all(
            issparse(operator) for operator in values(operators)
        )
    end

    @testset "RestMat direction and q=1/6 chain rule" begin
        N = 12
        coefficients = ComplexF64[
            complex(cos(0.31n), sin(0.47n)) / (n + 1)^2 for n in 0:N
        ]
        R = VoltOpt.restmat_cheb(
            -1.0,
            -2.0 / 3.0,
            N;
            T=Float64,
            check=true,
        )
        restricted = R * coefficients
        points = collect(range(-1.0, 1.0; length=41))
        direction_error = maximum(
            abs(
                chebyshev_evaluate(restricted, x) -
                chebyshev_evaluate(coefficients, (x - 5.0) / 6.0),
            ) for x in points
        )
        @test direction_error < 5.0e-14
        @test restriction_derivative_crosscheck(
            coefficients,
            points;
            q=FIGURE2_Q_TEST,
        ) < 1.0e-10
    end

    @testset "explicit Figure 2 operator" begin
        N = 14
        T = 11.5
        operators = ultraspherical_operators(N)
        R = VoltOpt.restmat_cheb(
            -1.0,
            -2.0 / 3.0,
            N;
            T=Float64,
        )
        expected =
            operators.D2 +
            (T^2 / 2.0) * (operators.S1 * operators.S0) -
            15.0im * T * (operators.S1 * operators.D1 * R)
        actual = build_scaled_pantograph_operator(
            N,
            T;
            omega2=2.0,
            gamma=5.0im,
            q=FIGURE2_Q_TEST,
        )
        @test eltype(parent(actual.matrix)) === ComplexF64
        @test actual.delay_scale == -15.0im * T
        @test norm(
            Matrix(actual.matrix) - ComplexF64.(expected),
            Inf,
        ) < 2.0e-12
    end

    @testset "T=1000 adaptive prepared solver and complex linearity" begin
        prepared = prepare_figure2_solver(
            T=FIGURE2_T_TEST;
            N0=512,
            verbose=false,
        )
        @test prepared.T == 1000.0
        @test prepared.N <= MAX_SOLVE_DEGREE
        @test prepared.N > prepared.coarse_N
        @test prepared.converged
        @test prepared.factorization_count_per_level == 1
        @test prepared.factorization_reused

        phi0_initial = initial_condition_errors(
            prepared.coefficients_phi0,
            prepared.T;
            y0=1.0,
            dy0=0.0,
        )
        phi1_initial = initial_condition_errors(
            prepared.coefficients_phi1,
            prepared.T;
            y0=0.0,
            dy0=1.0,
        )
        @test phi0_initial.value <= 1.0e-11
        @test phi0_initial.derivative <= 1.0e-11
        @test phi1_initial.value <= 1.0e-11
        @test phi1_initial.derivative <= 1.0e-11
        @test prepared.phi0.absolute_residual <= 1.0e-9
        @test prepared.phi1.absolute_residual <= 1.0e-9
        @test prepared.phi0.relative_residual <= 1.0e-11
        @test prepared.phi1.relative_residual <= 1.0e-11
        @test prepared.phi0.cauchy_error <= 1.0e-10
        @test prepared.phi1.cauchy_error <= 1.0e-10

        y0 = 0.7 + 1.2im
        dy0 = -0.3 + 0.4im
        combined = solve_prepared_figure2(prepared, y0, dy0)
        @test combined.converged
        @test combined.initial_value_error <= 1.0e-11
        @test combined.initial_derivative_error <= 1.0e-11
        @test combined.absolute_residual <= 1.0e-9
        @test combined.relative_residual <= 1.0e-11
        @test combined.cauchy_error <= 1.0e-10

        current = solve_prepared_figure2(
            prepared,
            1.0 + 1.0im,
            0.9 - 0.91im,
        )
        @test current.converged
        @test current.absolute_residual <= 1.0e-9
        @test current.relative_residual <= 1.0e-11
        @test current.initial_value_error <= 1.0e-11
        @test current.initial_derivative_error <= 1.0e-11

        direct = solve_figure2_pantograph(
            1.0 + 1.0im,
            0.9 - 0.91im;
            T=FIGURE2_T_TEST,
            N=prepared.N,
            adaptive=false,
        )
        @test direct.T == 1000.0
        @test direct.N == prepared.N
        @test norm(direct.coefficients - current.coefficients) /
              norm(current.coefficients) < 5.0e-12

        alpha = 0.4 - 0.7im
        beta = -0.2 + 0.3im
        initial_a = (y0=0.6 + 0.2im, dy0=-0.1 + 0.8im)
        initial_b = (y0=-0.3 + 0.9im, dy0=0.5 - 0.4im)
        solution_a = solve_prepared_figure2(
            prepared,
            initial_a.y0,
            initial_a.dy0,
        )
        solution_b = solve_prepared_figure2(
            prepared,
            initial_b.y0,
            initial_b.dy0,
        )
        mixed_y0 = alpha * initial_a.y0 + beta * initial_b.y0
        mixed_dy0 = alpha * initial_a.dy0 + beta * initial_b.dy0
        mixed = solve_prepared_figure2(
            prepared,
            mixed_y0,
            mixed_dy0,
        )
        linear_combination =
            alpha .* solution_a.coefficients .+
            beta .* solution_b.coefficients
        @test norm(mixed.coefficients - linear_combination) /
              norm(linear_combination) < 5.0e-13
    end

    @testset "test suite is independent of generated figures" begin
        @test isfile(joinpath(@__DIR__, "data", ".gitkeep"))
        @test isfile(joinpath(@__DIR__, "figures", ".gitkeep"))
    end
end
