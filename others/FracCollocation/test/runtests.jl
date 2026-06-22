using FracCollocation
using SpecialFunctions: beta
using Test

@testset "shifted Jacobi-Gauss beta moments" begin
    params = ((-0.5, -0.5), (-0.2, 1.0), (-0.8, 3.0))

    for (alpha, beta_exp) in params
        for N in (8, 16)
            z, w = shifted_jacobi_gauss(N, alpha, beta_exp)

            @test length(z) == N
            @test length(w) == N
            @test issorted(z)

            for p in 0:12
                exact = beta(p + beta_exp + 1, alpha + 1)
                quad = sum(w .* (z .^ p))
                @test isapprox(quad, exact; rtol = 5e-12, atol = 5e-13)
            end
        end
    end
end

@testset "barycentric Lagrange basis" begin
    N = 14
    z, _ = shifted_jacobi_gauss(N, -0.4, 0.7)
    bary = FracCollocation.barycentric_weights(z)
    ell = zeros(N)

    ztests = [0.0, 0.1, 0.37, 0.9, 1.0, z[1], z[div(N, 2)], z[end]]

    for z_eval in ztests
        FracCollocation.lagrange_values!(ell, z, bary, z_eval)
        @test isapprox(sum(ell), 1.0; rtol = 1e-13, atol = 1e-13)

        for p in 0:min(N - 1, 5)
            interp = sum(ell[j] * z[j]^p for j in 1:N)
            @test isapprox(interp, z_eval^p; rtol = 1e-12, atol = 1e-12)
        end
    end
end

@testset "K = 0 returns g at nodes" begin
    prob = FracVIEProblem(
        mu = 0.35,
        lambda = 0.6,
        alpha = -0.25,
        beta = 0.2,
        K = (x, s) -> 0.0,
        g = x -> sin(x) + x^0.3,
    )

    sol = solve(prob, 12)
    @test isapprox(sol.values, prob.g.(sol.x); atol = 1e-13, rtol = 1e-13)
end

@testset "N consistency" begin
    prob = FracVIEProblem(
        mu = 0.5,
        lambda = 0.5,
        alpha = 0.0,
        beta = 0.0,
        K = (x, s) -> 0.25,
        g = x -> 1.0 + x,
    )

    for N in (1, 4, 9)
        sys = build_system(prob, N)
        sol = solve(sys)

        @test length(sys.x) == N
        @test length(sys.z) == N
        @test length(sys.theta) == N
        @test length(sys.wtheta) == N
        @test length(sol.values) == N
        @test size(sys.V) == (N, N)
        @test size(sys.A) == (N, N)
    end
end

@testset "residual consistency" begin
    prob = FracVIEProblem(
        mu = 0.45,
        lambda = 0.5,
        alpha = -0.2,
        beta = 0.3,
        K = (x, s) -> 1 + x * s,
        g = x -> 1 + x^0.5,
        sigma = 0.75,
    )

    sys = build_system(prob, 12)
    sol = solve(sys)
    linear_r = sys.A * sol.values - sys.rhs
    colloc_r = collocation_residual(sys, sol.values)

    @test isapprox(colloc_r, linear_r; rtol = 1e-12, atol = 1e-12)
    @test maximum(abs.(colloc_r)) < 1e-11
end

@testset "lambda = 1 manufactured polynomial" begin
    for mu in (0.2, 0.5, 0.8)
        for p in 0:3
            u_exact(x) = x^p

            prob = FracVIEProblem(
                mu = mu,
                lambda = 1.0,
                alpha = 0.0,
                beta = 0.0,
                K = (x, s) -> 1.0,
                g = x -> u_exact(x) - x^(p + 1 - mu) * beta(p + 1, 1 - mu),
            )

            sol = solve(prob, 10)
            err = maximum(abs.(sol.values .- u_exact.(sol.x)))
            @test err < 5e-11
        end
    end
end

@testset "sigma = 0 returns g at nodes" begin
    prob = FracVIEProblem(
        mu = 0.6,
        lambda = 0.4,
        alpha = -0.5,
        beta = -0.25,
        K = (x, s) -> exp(x + s) + 2.0,
        g = x -> cos(x) + x^0.25,
        sigma = 0.0,
    )

    sol = solve(prob, 14)
    @test isapprox(sol.values, prob.g.(sol.x); atol = 1e-13, rtol = 1e-13)
end

@testset "non-unit lambda convergence sanity" begin
    mu = 0.4
    lambda = 0.5
    gamma = 1.25
    u_exact(x) = x^gamma

    prob = FracVIEProblem(
        mu = mu,
        lambda = lambda,
        alpha = 0.0,
        beta = 0.0,
        K = (x, s) -> 1.0,
        g = x -> u_exact(x) - x^(gamma + 1 - mu) * beta(gamma + 1, 1 - mu),
    )

    sol8 = solve(prob, 8)
    sol16 = solve(prob, 16)
    err8 = maximum(abs.(sol8.values .- u_exact.(sol8.x)))
    err16 = maximum(abs.(sol16.values .- u_exact.(sol16.x)))

    @test err16 < err8
    @test err16 < 5e-5
end

@testset "evaluate returns nodal values" begin
    prob = FracVIEProblem(
        mu = 0.5,
        lambda = 0.5,
        alpha = -0.5,
        beta = -0.5,
        K = (x, s) -> 0.0,
        g = x -> 1.0 + x^0.5,
    )

    sol = solve(prob, 10)
    @test isapprox(evaluate(sol, sol.x), sol.values; atol = 1e-12, rtol = 1e-12)
end
