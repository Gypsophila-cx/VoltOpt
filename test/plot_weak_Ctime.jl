using Printf
using Plots

gr()

# ============================================================
# Paths
# ============================================================

script_dir = @__DIR__
project_root = dirname(script_dir)
data_dir = joinpath(project_root, "test", "data")
fig_dir = joinpath(project_root, "test", "figures")
mkpath(fig_dir)

# ============================================================
# CSV readers
# ============================================================

function read_csv_rows(csvfile::AbstractString)
    isfile(csvfile) || error("CSV file not found: $(csvfile)")

    lines = readlines(csvfile)
    isempty(lines) && error("Empty CSV file: $(csvfile)")

    header = String.(strip.(split(lines[1], ",")))

    rows = Vector{Vector{String}}()
    for line in lines[2:end]
        isempty(strip(line)) && continue
        push!(rows, strip.(split(line, ","; keepempty = true)))
    end

    col = Dict(header[i] => i for i in eachindex(header))
    return rows, col
end

function parse_float_cell(row, idx)
    idx > length(row) && return NaN
    v = tryparse(Float64, strip(row[idx]))
    return v === nothing ? NaN : v
end

function require_column(col::Dict{String,Int}, names::AbstractVector{String})
    for name in names
        haskey(col, name) && return name
    end
    error("CSV does not contain any of these required columns: $(join(names, ", "))")
end

function read_weak_fixedN_varyC_data(csvfile::AbstractString)
    rows, col = read_csv_rows(csvfile)

    x_col = require_column(col, ["bandwidth", "C_size", "C_rows"])
    t_col = require_column(col, ["t_build"])

    x = Float64[]
    t = Float64[]

    for row in rows
        xi = parse_float_cell(row, col[x_col])
        ti = parse_float_cell(row, col[t_col])

        if isfinite(xi) && isfinite(ti) && xi > 0 && ti > 0
            push!(x, xi)
            push!(t, ti)
        end
    end

    isempty(x) && error("No valid weak fixed-N timing data found in $(csvfile).")

    p = sortperm(x)
    return x[p], t[p], x_col
end

# ============================================================
# Reference curves
# ============================================================

reference_linear(x, x0, y0) = y0 .* (x ./ x0)

function tail_range(v; ntail::Int = 4)
    n = length(v)
    lo = max(1, n - ntail + 1)
    return lo:n
end

# ============================================================
# Main plot
# ============================================================

function plot_weak_fixedN_varyC_time(
    csvfile::AbstractString;
    output_pdf::String = joinpath(fig_dir, "bench_weak_fixedN_varyC_time.pdf"),
    output_png::String = replace(output_pdf, r"\.pdf$" => ".png"),
)
    x, t_build, x_col = read_weak_fixedN_varyC_data(csvfile)
    xlabel_text = x_col == "bandwidth" ? "lower bandwidth n" :
                  x_col == "C_size" ? "coefficient matrix size" :
                  "coefficient rows"

    plt = plot(
        x,
        t_build;
        xscale = :log10,
        yscale = :log10,
        marker = :circle,
        markersize = 5,
        linewidth = 2.5,
        label = "Weak Chebyshev/JFP",
        xlabel = xlabel_text,
        ylabel = "construction time / s",
        legend = :topleft,
        framestyle = :box,
        dpi = 300,
    )

    ydata_for_limits = copy(t_build)

    if length(x) >= 2
        r = tail_range(x; ntail = min(3, length(x)))
        idx = first(r)
        ref = reference_linear(x, x[idx], t_build[idx] * 0.8)

        plot!(
            plt,
            x[r],
            ref[r];
            linestyle = :dash,
            linewidth = 1.3,
            color = :black,
            label = nothing,
        )
        annotate!(plt, x[last(r)], ref[last(r)] * 0.6, text("O(n)", 10, :right))
        append!(ydata_for_limits, ref)
    end

    xticks!(plt, x, string.(Int.(round.(x))))

    ymin = minimum(ydata_for_limits)
    ymax = maximum(ydata_for_limits)
    #ylims!(plt, 10.0^floor(log10(ymin)), 10.0^ceil(log10(ymax)))

    savefig(plt, output_pdf)
    println("PDF saved to: ", output_pdf)
    savefig(plt, output_png)
    println("PNG saved to: ", output_png)

    return plt
end

# ============================================================
# Run
# ============================================================

csvfile = joinpath(data_dir, "bench_weak_fixedN_varyC.csv")
println("Using CSV: ", csvfile)

plot_weak_fixedN_varyC_time(
    csvfile;
    output_pdf = joinpath(fig_dir, "bench_weak_fixedN_varyC_time.pdf"),
)
