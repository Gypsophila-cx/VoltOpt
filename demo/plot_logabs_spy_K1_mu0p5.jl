#!/usr/bin/env julia

# Build and plot the final weakly singular Volterra matrix for K(x,t)=1,
# mu=0.5, matrix size 500x500, using the stable paper-DE transformed kernel.
# This script writes figures under demo/output/ and does not modify project
# dependency files.

using Printf

try
    @eval import Plots
catch err
    error("""
    Plots.jl is required to generate this PDF.
    Please run this script in a Julia environment where Plots.jl is already
    available. This script will not modify Project.toml or Manifest.toml.

    Original error:
    $err
    """)
end

include(joinpath(@__DIR__, "weak_voltmat_decheb.jl"))
using .Weak_VoltMat_DE

const LOG_FLOOR_FACTOR = 1e-18

function logabs_plot_matrix(A::AbstractMatrix{<:Real})
    nrows, ncols = size(A)
    maxabs = maximum(abs, A)
    floor_value = LOG_FLOOR_FACTOR * maxabs
    L = Matrix{Float64}(undef, nrows, ncols)
    vmin = Inf
    vmax = -Inf

    for j in 1:ncols, i in 1:nrows
        value = log10(max(abs(A[i, j]), floor_value))
        L[i, j] = value
        vmin = min(vmin, value)
        vmax = max(vmax, value)
    end

    return L, vmin, vmax
end

function colorbar_tick_spec(vmin::Float64, vmax::Float64; nticks=7)
    ticks = collect(range(vmin, vmax; length=nticks))
    labels = [@sprintf("%.2f", t) for t in ticks]
    return ticks, labels
end

function spy_coordinates(A::AbstractMatrix{<:Real}, threshold::Float64)
    rows = Int[]
    cols = Int[]
    for j in axes(A, 2), i in axes(A, 1)
        if abs(A[i, j]) > threshold
            push!(rows, i)
            push!(cols, j)
        end
    end
    return rows, cols
end

function min_positive_abs(A::AbstractMatrix{<:Real})
    best = Inf
    for x in A
        ax = abs(x)
        if ax > 0.0 && ax < best
            best = ax
        end
    end
    return best
end

function save_logabs_spy_plot(A::AbstractMatrix{<:Real}, pdf_path::AbstractString, png_path::AbstractString)
    L, vmin, vmax = logabs_plot_matrix(A)
    nrows, ncols = size(A)
    ticks, labels = colorbar_tick_spec(vmin, vmax)
    maxabs = maximum(abs, A)
    threshold = 1e-14 * maxabs
    rows, cols = spy_coordinates(A, threshold)
    title_text = "K=1, mu=0.5, paper DE, omega=4, C=100x100, N=500"

    Plots.gr()
    p1 = Plots.heatmap(
        1:ncols,
        1:nrows,
        L;
        title="log10(|A_ij|), " * title_text,
        xlabel="column index",
        ylabel="row index",
        yflip=true,
        aspect_ratio=:equal,
        color=:viridis,
        clims=(vmin, vmax),
        colorbar=true,
        colorbar_ticks=(ticks, labels),
        background_color=:white,
        background_color_inside=:white,
        grid=false,
        framestyle=:box,
        margin=6Plots.mm,
    )

    p2 = Plots.scatter(
        cols,
        rows;
        title=@sprintf("spy: |A_ij| > %.2e", threshold),
        xlabel="column index",
        ylabel="row index",
        yflip=true,
        aspect_ratio=:equal,
        markersize=1.2,
        markerstrokewidth=0,
        color=:black,
        label=false,
        xlims=(1, ncols),
        ylims=(1, nrows),
        background_color=:white,
        background_color_inside=:white,
        grid=false,
        framestyle=:box,
        margin=6Plots.mm,
    )

    p = Plots.plot(p1, p2; layout=(1, 2), size=(1500, 720), dpi=300)
    Plots.savefig(p, pdf_path)
    Plots.savefig(p, png_path)
    return (; pdf_path, png_path, vmin, vmax, threshold, plotted=length(rows))
end

function main()
    K(x, t) = 1.0
    mu = 0.5
    N = 500
    omega = 4.0

    A, info = Weak_VoltMat_DE.weak_voltmat_decheb(
        K;
        mu=mu,
        N=N,
        omega=omega,
        tol=1e-14,
        return_info=true,
        Csize_y=100,
        Csize_eta=100,
    )

    output_dir = joinpath(@__DIR__, "output")
    mkpath(output_dir)
    base = "logabs_spy_K1_mu0p5_DEomega4_C100_N500"
    pdf_path = joinpath(output_dir, base * ".pdf")
    png_path = joinpath(output_dir, base * ".png")
    fig = save_logabs_spy_plot(A, pdf_path, png_path)
    rel_pdf = replace(relpath(fig.pdf_path, pwd()), "\\" => "/")
    rel_png = replace(relpath(fig.png_path, pwd()), "\\" => "/")

    println("plot type: logabs heatmap spy")
    println("kernel: K(x,t)=1")
    println("mu: ", mu)
    println("paper DE omega: ", omega)
    println("matrix size: ", size(A, 1), " x ", size(A, 2))
    println("size(A): ", size(A))
    println("all finite: ", all(isfinite, A))
    println("maximum(abs,A): ", @sprintf("%.6e", maximum(abs, A)))
    println("minimum positive abs(A): ", @sprintf("%.6e", min_positive_abs(A)))
    println("matrix used for plot: final matrix A")
    println("spy threshold: ", @sprintf("%.6e", fig.threshold))
    println("output PDF: ", rel_pdf)
    println("output PNG: ", rel_png)
    println("stable DE Ktilde: ", info.stable_de_ktilde)
    println("C size: ", info.C_size[1], " x ", info.C_size[2])
    println("spy plotted entries: ", fig.plotted)
    println("colorbar ticks: ", join((@sprintf("%.2f", x) for x in colorbar_tick_spec(fig.vmin, fig.vmax)[1]), ", "))
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
