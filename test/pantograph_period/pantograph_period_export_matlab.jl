include(joinpath(@__DIR__, "pantograph_period_common.jl"))

using .PantographPeriod
using Printf

# ==========================================================
# User parameters
# ==========================================================

T = 1000.0

y0 = 1.0 + 1.0im
dy0 = 0.9 - 0.91im

adaptive = true
N = nothing
N0 = 512
Nmax = 5000

Mplot = 100001

curve_csv = joinpath(
    @__DIR__,
    "data",
    "pantograph_period_T1000_curve.csv",
)

summary_csv = joinpath(
    @__DIR__,
    "data",
    "pantograph_period_T1000_summary.csv",
)

# ==========================================================
# End of user parameters; no edits are normally needed below
# ==========================================================

mkpath(dirname(curve_csv))

println("Preparing the verified Figure 2 solver for MATLAB export...")
prepared = prepare_figure2_solver(
    T=T;
    N=N,
    N0=N0,
    Nmax=Nmax,
    adaptive=adaptive,
    verbose=true,
)
prepared.converged ||
    error("The T=1000 fundamental solutions failed validation.")

solution = solve_prepared_figure2(
    prepared,
    y0,
    dy0;
    throw_on_failure=true,
)
solution.converged ||
    error("The requested complex initial-value solution failed validation.")

tgrid = collect(range(0.0, T; length=Mplot))
values = evaluate_pantograph(
    solution.coefficients,
    T,
    tgrid,
)
all(isfinite, real.(values)) ||
    error("Non-finite real samples prevent export.")
all(isfinite, imag.(values)) ||
    error("Non-finite imaginary samples prevent export.")

open(curve_csv, "w") do io
    println(io, "t,re_y,im_y")
    for j in eachindex(tgrid, values)
        @printf(
            io,
            "%.17g,%.17g,%.17g\n",
            tgrid[j],
            real(values[j]),
            imag(values[j]),
        )
    end
end

open(summary_csv, "w") do io
    println(
        io,
        "T,N,y0_real,y0_imag,dy0_real,dy0_imag," *
        "equation_residual,relative_residual,initial_value_error," *
        "initial_derivative_error,coefficient_tail,cauchy_error," *
        "Mplot,preparation_time,validation_time,status",
    )
    @printf(
        io,
        "%.17g,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%d,%.17g,%.17g,%s\n",
        solution.T,
        solution.N,
        real(solution.y0),
        imag(solution.y0),
        real(solution.dy0),
        imag(solution.dy0),
        solution.absolute_residual,
        solution.relative_residual,
        solution.initial_value_error,
        solution.initial_derivative_error,
        solution.coefficient_tail,
        solution.cauchy_error,
        Mplot,
        prepared.solve_time,
        solution.validation_time,
        solution.converged ? "passed" : "failed",
    )
end

println("\nMATLAB curve export completed")
@printf("  T=%.16g, N=%d, Mplot=%d\n", solution.T, solution.N, Mplot)
println("  y(0)  = ", solution.y0)
println("  y'(0) = ", solution.dy0)
@printf(
    "  residual: absolute=%.6e, relative=%.6e\n",
    solution.absolute_residual,
    solution.relative_residual,
)
@printf(
    "  initial errors: value=%.6e, derivative=%.6e\n",
    solution.initial_value_error,
    solution.initial_derivative_error,
)
@printf("  coefficient tail=%.6e\n", solution.coefficient_tail)
@printf("  Cauchy error=%.6e\n", solution.cauchy_error)
@printf(
    "  preparation time=%.3fs, combination/validation time=%.3fs\n",
    prepared.solve_time,
    solution.validation_time,
)
println("  curve CSV:   ", curve_csv)
println("  summary CSV: ", summary_csv)
