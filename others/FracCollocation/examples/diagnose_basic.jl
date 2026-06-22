using FracCollocation
using LinearAlgebra: cond
using Printf

const NS = [10, 20, 30, 40, 50, 60, 80]

function basic_problem(sigma)
    return FracVIEProblem(
        mu = 0.5,
        lambda = 0.5,
        alpha = -0.5,
        beta = -0.5,
        K = (x, s) -> exp(x - s),
        g = x -> 1.0,
        sigma = sigma,
    )
end

function status_label(residual_inf, cond_A, maxabs_u)
    flags = String[]
    residual_inf > 1e-8 && push!(flags, "large_residual")
    cond_A > 1e10 && push!(flags, "ill_conditioned")
    maxabs_u > 1e4 && push!(flags, "large_solution")
    return isempty(flags) ? "ok" : join(flags, "+")
end

function run_case(prob, sigma, N)
    sys = nothing
    build_time = @elapsed begin
        sys = build_system(prob, N)
    end

    sol = nothing
    solve_time = @elapsed begin
        sol = solve(sys)
    end

    residual_inf = maximum(abs.(collocation_residual(sys, sol.values)))
    linear_residual_inf = maximum(abs.(sys.A * sol.values - sys.rhs))
    cond_A = cond(sys.A)
    maxabs_V = maximum(abs.(sys.V))
    maxabs_A = maximum(abs.(sys.A))
    maxabs_u = maximum(abs.(sol.values))
    min_x = minimum(sys.x)
    max_x = maximum(sys.x)
    status = status_label(residual_inf, cond_A, maxabs_u)

    return (
        sigma = sigma,
        N = N,
        build_time = build_time,
        solve_time = solve_time,
        residual_inf = residual_inf,
        linear_residual_inf = linear_residual_inf,
        cond_A = cond_A,
        maxabs_V = maxabs_V,
        maxabs_A = maxabs_A,
        maxabs_u = maxabs_u,
        min_x = min_x,
        max_x = max_x,
        status = status,
    )
end

function print_row(row)
    @printf(
        "%5.2f %4d %11.4e %11.4e %13.4e %19.4e %11.4e %11.4e %11.4e %11.4e %11.4e %11.4e %s\n",
        row.sigma,
        row.N,
        row.build_time,
        row.solve_time,
        row.residual_inf,
        row.linear_residual_inf,
        row.cond_A,
        row.maxabs_V,
        row.maxabs_A,
        row.maxabs_u,
        row.min_x,
        row.max_x,
        row.status,
    )
end

function write_row(io, row)
    @printf(
        io,
        "%.16e,%d,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%s\n",
        row.sigma,
        row.N,
        row.build_time,
        row.solve_time,
        row.residual_inf,
        row.linear_residual_inf,
        row.cond_A,
        row.maxabs_V,
        row.maxabs_A,
        row.maxabs_u,
        row.min_x,
        row.max_x,
        row.status,
    )
end

function main()
    data_dir = joinpath(@__DIR__, "data")
    mkpath(data_dir)
    csv_path = joinpath(data_dir, "diagnose_basic.csv")

    sigmas = (1.0, 0.1)

    open(csv_path, "w") do io
        println(
            io,
            "sigma,N,build_time,solve_time,residual_inf,linear_residual_inf,cond_A,maxabs_V,maxabs_A,maxabs_u,min_x,max_x,status",
        )
        println(
            "sigma    N  build_time  solve_time  residual_inf  linear_residual_inf      cond_A    maxabs_V    maxabs_A    maxabs_u       min_x       max_x status",
        )
        for sigma in sigmas
            prob = basic_problem(sigma)
            for N in NS
                row = run_case(prob, sigma, N)
                print_row(row)
                write_row(io, row)
            end
        end
    end

    println("wrote ", csv_path)
end

main()
