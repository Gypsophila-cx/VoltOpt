using DelimitedFiles
using Printf
using Plots

const THIS_DIR = @__DIR__

function find_project_root(start_dir::AbstractString=@__DIR__)
    dir = abspath(start_dir)
    while true
        if isfile(joinpath(dir, "Project.toml")) && isdir(joinpath(dir, "src"))
            return dir
        end
        parent = dirname(dir)
        parent == dir && error("Could not find the VoltOpt project root from $(start_dir).")
        dir = parent
    end
end

const VOLTERRA_ROOT = find_project_root()
const FIGURES_DIR = joinpath(VOLTERRA_ROOT, "examples", "figures")
const DATA_DIR = joinpath(VOLTERRA_ROOT, "examples", "data")
const ABS_ERROR_CSV = joinpath(DATA_DIR, "ex1_aging_maxwell_abs_error.csv")
const ABS_ERROR_PDF = joinpath(FIGURES_DIR, "ex1_aging_maxwell_abs_error_vs_N.pdf")

function savefig_pdf_png(fig, pdf_path::AbstractString)
    png_path = replace(pdf_path, r"\.pdf$" => ".png")
    savefig(fig, pdf_path)
    savefig(fig, png_path)
    return png_path
end

function require_column(headers::AbstractVector{<:AbstractString}, name::AbstractString)
    idx = findfirst(==(name), headers)
    idx === nothing && error("Missing required CSV column: $(name).")
    return idx
end

function read_abs_error_csv(filename::AbstractString)
    isfile(filename) || error(
        "Absolute-error CSV does not exist: $(filename). " *
        "Run examples/ex1_2ndVolterraEqn/ex1_aging_maxwell.jl first.",
    )

    data, header = readdlm(filename, ',', Float64; header=true)
    data = data isa AbstractVector ? reshape(data, 1, :) : data
    headers = vec(String.(header))

    n_col = require_column(headers, "N")
    err_col = require_column(headers, "abs_error_inf")

    Ns = Int.(data[:, n_col])
    abs_errors = Float64.(data[:, err_col])

    return Ns, abs_errors
end

function log_limits(values::AbstractVector{<:Real})
    positives = [Float64(v) for v in values if isfinite(v) && v > 0.0]
    isempty(positives) && error("Cannot draw a log-scale plot: all absolute errors are zero or non-finite.")

    ymin = minimum(positives)
    ymax = maximum(positives)

    if ymin == ymax
        return max(ymin / 10.0, floatmin(Float64)), ymax * 10.0
    end

    return (
        max(10.0^(floor(Int, log10(ymin)) - 1), floatmin(Float64)),
        10.0^(ceil(Int, log10(ymax)) + 1),
    )
end

function power_ticks(ymin::Real, ymax::Real; step::Int=2)
    kmin = floor(Int, log10(ymin))
    kmax = ceil(Int, log10(ymax))
    ks = collect(kmin:step:kmax)
    vals = 10.0 .^ ks
    labels = ["10^{$k}" for k in ks]
    return vals, labels
end

function plot_abs_error(
    Ns::AbstractVector{<:Integer},
    abs_errors::AbstractVector{<:Real};
    filename::AbstractString=ABS_ERROR_PDF,
)
    ymin, ymax = log_limits(abs_errors)
    yticks = power_ticks(ymin, ymax; step=2)

    p = plot(
        Ns,
        abs_errors;
        yaxis = :log10,
        ylims = (1e-15,1),
        #yticks = yticks,
        marker = false,
        linewidth = 2.0,
        xlabel = "N",
        ylabel = "error",
        label = nothing,
        framestyle = :box,
        grid = false,
        minorgrid = false,
        #minorticks = 5,
        tickfontsize = 10,
        guidefontsize = 13,
        legendfontsize = 10,
        left_margin = 6Plots.mm,
        bottom_margin = 5Plots.mm,
        color = :black,
    )

    png_path = savefig_pdf_png(p, filename)

    @printf("Saved absolute-error figure:\n")
    @printf("  %s\n", filename)
    @printf("  %s\n", png_path)

    return p
end

Ns, abs_errors = read_abs_error_csv(ABS_ERROR_CSV)
plot_abs_error(Ns, abs_errors)
