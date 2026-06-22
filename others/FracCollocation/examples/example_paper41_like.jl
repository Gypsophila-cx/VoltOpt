using FracCollocation
using LinearAlgebra: cond
using Printf

const CASES = (
    (0.2, 1 / 5),
    (0.2, 1 / 10),
    (0.5, 1 / 2),
    (0.5, 1 / 4),
)
const NS = collect(5:5:60)
const NREF = 100

function make_problem(mu, lambda)
    return FracVIEProblem(
        mu = mu,
        lambda = lambda,
        alpha = -0.5,
        beta = -0.5,
        K = (x, s) -> exp(x - s),
        g = x -> 1.0,
        sigma = 1.0,
    )
end

function build_and_solve(prob, N)
    sys = nothing
    build_time = @elapsed begin
        sys = build_system(prob, N)
    end

    sol = nothing
    solve_time = @elapsed begin
        sol = solve(sys)
    end

    return sys, sol, build_time, solve_time
end

function main()
    data_dir = joinpath(@__DIR__, "data")
    mkpath(data_dir)
    csv_path = joinpath(data_dir, "paper41_like.csv")
    xgrid = collect(range(0.0, 1.0; length = 1001))

    open(csv_path, "w") do io
        println(
            io,
            "mu,lambda,N,Nref,err_inf,build_time,solve_time,residual_inf,cond_A,maxabs_u",
        )
        println("mu, lambda, N, Nref, err_inf, build_time, solve_time, residual_inf, cond_A, maxabs_u")

        for (mu, lambda) in CASES
            prob = make_problem(mu, lambda)
            sys_ref, sol_ref, ref_build_time, ref_solve_time = build_and_solve(prob, NREF)
            uref_grid = evaluate(sol_ref, xgrid)
            ref_residual = maximum(abs.(collocation_residual(sys_ref, sol_ref.values)))
            @printf(
                "# reference mu=%.3f lambda=%.3f Nref=%d build=%.4e solve=%.4e residual=%.4e\n",
                mu,
                lambda,
                NREF,
                ref_build_time,
                ref_solve_time,
                ref_residual,
            )

            for N in NS
                sys, sol, build_time, solve_time = build_and_solve(prob, N)
                err_inf = maximum(abs.(evaluate(sol, xgrid) - uref_grid))
                residual_inf = maximum(abs.(collocation_residual(sys, sol.values)))
                cond_A = cond(sys.A)
                maxabs_u = maximum(abs.(sol.values))

                @printf(
                    "%.3f, %.3f, %d, %d, %.6e, %.6e, %.6e, %.6e, %.6e, %.6e\n",
                    mu,
                    lambda,
                    N,
                    NREF,
                    err_inf,
                    build_time,
                    solve_time,
                    residual_inf,
                    cond_A,
                    maxabs_u,
                )
                @printf(
                    io,
                    "%.16e,%.16e,%d,%d,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e\n",
                    mu,
                    lambda,
                    N,
                    NREF,
                    err_inf,
                    build_time,
                    solve_time,
                    residual_inf,
                    cond_A,
                    maxabs_u,
                )
            end
        end
    end

    println("wrote ", csv_path)
end

main()
