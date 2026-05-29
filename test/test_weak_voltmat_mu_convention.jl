using LinearAlgebra
using Test

const PROJECT_ROOT = dirname(@__DIR__)

include(joinpath(PROJECT_ROOT, "src", "VoltOpt.jl"))
using .VoltOpt

const WeakCheb = VoltOpt.Weak_VoltMat_Cheb

dense_matrix(V) = V isa VoltOpt.PackedLowerHessenberg ? VoltOpt.packed_to_dense(V) : Matrix(V)

function max_value_error(c, f; sigma::Real, beta::Real)
    xs = range(-0.95, 0.95; length = 41)
    return maximum(xs) do x
        abs(WeakCheb.eval_jfp_basis_expansion(c, x; sigma = sigma, beta = beta) - f(x))
    end
end

@testset "weak_voltmat_cheb public mu convention" begin
    Kone(x, y) = 1.0
    C = ones(1, 1)
    sigma = 0.0

    @testset "mu=0.5 means (x-y)^(-1/2)" begin
        mu = 0.5
        beta = 1.0 - mu
        N = 24

        Vfast, info_fast = VoltOpt.weak_voltmat_cheb(
            C;
            C_is_coeffs = true,
            mu = mu,
            sigma = sigma,
            beta = beta,
            N = N,
            diagnostics = false,
            return_info = true,
        )

        Vdirect, info_direct = VoltOpt.weak_voltmat_cheb(
            C;
            C_is_coeffs = true,
            mu = mu,
            sigma = sigma,
            beta = beta,
            N = N,
            diagnostics = false,
            init_method = :direct,
            return_info = true,
        )

        @test info_fast.eta ≈ 1.0 atol = 10eps()
        @test info_direct.eta ≈ 1.0 atol = 10eps()

        Dfast = dense_matrix(Vfast)
        Ddirect = dense_matrix(Vdirect)
        @test norm(Dfast .- Ddirect, Inf) < 1e-10

        c = Dfast[:, 1]
        @test c[1] ≈ sqrt(2.0) rtol = 1e-11 atol = 1e-12
        @test c[2] ≈ sqrt(2.0) rtol = 1e-11 atol = 1e-12
        @test norm(@view(c[3:end]), Inf) < 1e-10

        analytic(x) = 2.0 * sqrt(x + 1.0)
        @test max_value_error(c, analytic; sigma = sigma, beta = beta) < 1e-10

        check = VoltOpt.validate_weak_voltmat_columns(
            Kone,
            Vfast;
            mu = mu,
            sigma = sigma,
            beta = beta,
            cols = (0,),
            xs = range(-0.95, 0.95; length = 21),
            Qref = 300,
        )
        @test check.max_abs < 1e-10
    end

    @testset "mu=0 degenerates to nonsingular Volterra integration" begin
        mu = 0.0
        beta = 1.0
        N = 16

        V, info = VoltOpt.weak_voltmat_cheb(
            C;
            C_is_coeffs = true,
            mu = mu,
            sigma = sigma,
            beta = beta,
            N = N,
            diagnostics = false,
            return_info = true,
        )

        @test info.eta ≈ 1.0 atol = 10eps()

        c = dense_matrix(V)[:, 1]
        @test c[1] ≈ 1.0 rtol = 1e-12 atol = 1e-12
        @test c[2] ≈ 1.0 rtol = 1e-12 atol = 1e-12
        @test norm(@view(c[3:end]), Inf) < 1e-12

        analytic(x) = x + 1.0
        @test max_value_error(c, analytic; sigma = sigma, beta = beta) < 1e-12
    end
end
