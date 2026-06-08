using Printf
using Plots

include(joinpath(@__DIR__, "ex1_aging_maxwell.jl"))

const DEFAULT_SOLUTION_N = 30000

function savefig_pdf_png(fig, pdf_path::AbstractString)
    png_path = replace(pdf_path, r"\.pdf$" => ".png")
    savefig(fig, pdf_path)
    savefig(fig, png_path)
    return png_path
end

function solution_degree_from_args(args::Vector{String})
    if isempty(args)
        return DEFAULT_SOLUTION_N
    elseif length(args) == 1
        N = parse(Int, args[1])
        N >= 2 || error("N must be at least 2.")
        return N
    else
        error("Usage: julia --project=. examples/ex1_2ndVolterraEqn/ex1_aging_maxwell_plot_solution.jl [N]")
    end
end

function plot_solution_comparison(N::Int)
    @printf("Solving aging Maxwell example for solution plot, N = %d\n", N)

    result = solve_volterra_once(
        N;
        tol = tol,
        keep_solution = true,
        use_fastalmostbanded = true,
        top_rows = nothing,
        lower_bw = nothing,
        upper_bw = nothing,
        detect_droptol = 0.0,
        convert_droptol = 0.0,
    )

    uplot_num = cheb_eval_many(result.cu_num, xplot)
    uplot_exact = u_exact.(xplot)
    plot_grid_abs_error = maximum(abs.(uplot_num .- uplot_exact))

    p = plot(
        xplot,
        uplot_exact;
        xlabel = "x",
        ylabel = "u(x)",
        linewidth = 2.0,
        label = "exact",
        framestyle = :box,
        grid = false,
        tickfontsize = 8,
        guidefontsize = 13,
        legendfontsize = 10,
        left_margin = 6Plots.mm,
        bottom_margin = 5Plots.mm,
        color = :black,
    )

    plot!(
        p,
        xplot,
        uplot_num;
        linewidth = 1.2,
        linestyle = :dash,
        label = "numerical",
        color = :gray,
    )

    mkpath(FIGURES_DIR)
    pdf_path = joinpath(FIGURES_DIR, "ex1_aging_maxwell_solution_N.pdf")
    png_path = savefig_pdf_png(p, pdf_path)

    @printf("Saved solution comparison figure:\n")
    @printf("  %s\n", pdf_path)
    @printf("  %s\n", png_path)
    @printf("Max absolute error on plot grid = %.6e\n", plot_grid_abs_error)
    @printf("Max absolute error on error grid = %.6e\n", result.err_inf)

    return p
end

plot_solution_comparison(solution_degree_from_args(ARGS))
