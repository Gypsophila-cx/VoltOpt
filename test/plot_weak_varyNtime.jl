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

function read_weak_fixedC_varyN_data(csvfile::AbstractString)
    rows, col = read_csv_rows(csvfile)

    n_col = require_column(col, ["N"])
    t_col = require_column(col, ["t_build"])

    N = Float64[]
    t = Float64[]

    for row in rows
        Ni = parse_float_cell(row, col[n_col])
        ti = parse_float_cell(row, col[t_col])

        if isfinite(Ni) && isfinite(ti) && Ni > 0 && ti > 0
            push!(N, Ni)
            push!(t, ti)
        end
    end

    isempty(N) && error("No valid weak vary-N timing data found in $(csvfile).")

    p = sortperm(N)
    return N[p], t[p]
end

# ============================================================
# Reference curves
# ============================================================

reference_power(x, x0, y0, p) = y0 .* (x ./ x0).^p

function tail_range(v; ntail::Int = 4)
    n = length(v)
    lo = max(1, n - ntail + 1)
    return lo:n
end

# ============================================================
# Main plot
# ============================================================

function plot_weak_fixedC_varyN_time(
    csvfile::AbstractString;
    output_pdf::String = joinpath(fig_dir, "bench_weak_fixedC_varyN_time.pdf"),
    output_png::String = replace(output_pdf, r"\.pdf$" => ".png"),
    add_reference_line::Bool = true,
)
    N, t_build = read_weak_fixedC_varyN_data(csvfile)

    plt = plot(
        N,
        t_build;
        xscale = :log10,
        yscale = :log10,
        marker = false,
        markersize = 5,
        linewidth = 2.5,
        label = nothing,
        xlabel = "matrix size N",
        ylabel = "construction time (sec)",
        legend = :topleft,
        framestyle = :box,
        dpi = 300,
    )

    ydata_for_limits = copy(t_build)

    if add_reference_line && length(N) >= 2
        r = tail_range(N; ntail = min(8, length(N)))
        idx = first(r)
        ref = reference_power(N, N[idx], t_build[idx] * 0.7, 2)

        plot!(
            plt,
            N[r],
            ref[r];
            linestyle = :dash,
            linewidth = 1.3,
            color = :black,
            label = nothing,
        )
        annotate!(plt, N[last(r)], ref[last(r)] * 0.4, text("O(N²)", 10, :right))
        append!(ydata_for_limits, ref)
    end

    ymin = minimum(ydata_for_limits)
    ymax = maximum(ydata_for_limits)
    ylims!(plt, 10.0^floor(log10(ymin)), 10.0^ceil(log10(ymax)))

    savefig(plt, output_pdf)
    println("PDF saved to: ", output_pdf)
    savefig(plt, output_png)
    println("PNG saved to: ", output_png)

    return plt
end

# ============================================================
# Run
# ============================================================

csvfile = joinpath(data_dir, "bench_weak_fixedC_varyN.csv")
println("Using CSV: ", csvfile)

plot_weak_fixedC_varyN_time(
    csvfile;
    output_pdf = joinpath(fig_dir, "bench_weak_fixedC_varyN_time.pdf"),
    add_reference_line = true,
)
