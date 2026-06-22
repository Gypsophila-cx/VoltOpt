using FracCollocation
using Printf
using SpecialFunctions: beta

function main()
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

    data_dir = joinpath(@__DIR__, "data")
    mkpath(data_dir)
    csv_path = joinpath(data_dir, "convergence.csv")

    open(csv_path, "w") do io
        println(io, "N,max_error,build_time,solve_time")
        println("N, max_error, build_time, solve_time")
        for N in (8, 12, 16, 24, 32)
            sys = nothing
            build_time = @elapsed begin
                sys = build_system(prob, N)
            end

            sol = nothing
            solve_time = @elapsed begin
                sol = solve(sys)
            end

            err = maximum(abs.(sol.values .- u_exact.(sol.x)))
            @printf("%d, %.6e, %.6e, %.6e\n", N, err, build_time, solve_time)
            @printf(io, "%d, %.16e, %.16e, %.16e\n", N, err, build_time, solve_time)
        end
    end

    println("wrote ", csv_path)
end

main()
