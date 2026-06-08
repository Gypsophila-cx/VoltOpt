# Plot build-time benchmark results for the coefficient-space weak Volterra
# matrix and the Muntz/fractional Jacobi collocation matrix.
#
# Run:
#   julia --project=. test/compare/three_methods/plot_build_ours_vs_muntz.jl

using Printf
using Statistics

const SCRIPT_DIR = @__DIR__
const DATA_DIR = joinpath(SCRIPT_DIR, "data")
const FIGURES_DIR = joinpath(SCRIPT_DIR, "figures")
const RESULTS_CSV = joinpath(DATA_DIR, "build_ours_vs_muntz_results.csv")
mkpath(FIGURES_DIR)

const METHOD_LABELS = Dict(
    "our_coeffspace_packed" => "Our coefficient matrix",
    "our_coeffspace_dense" => "Our coefficient matrix (dense)",
    "muntz_collocation" => "Müntz collocation matrix",
)

const METHOD_COLORS = Dict(
    "our_coeffspace_packed" => :black,
    "our_coeffspace_dense" => :gray35,
    "muntz_collocation" => :red,
)

const METHOD_MARKERS = Dict(
    "our_coeffspace_packed" => :star5,
    "our_coeffspace_dense" => :circle,
    "muntz_collocation" => :diamond,
)

function parse_bool(raw)
    s = lowercase(strip(string(raw)))
    s in ("1", "true", "yes", "on") && return true
    s in ("0", "false", "no", "off") && return false
    error("Invalid boolean value: $(raw)")
end

const PLOT_OURS_DENSE = parse_bool(get(ENV, "PLOT_OURS_DENSE", "0"))

function read_simple_csv(path::AbstractString)
    isfile(path) || error("CSV file not found: $(path)")
    lines = readlines(path)
    isempty(lines) && return String[], Vector{Dict{String,String}}()
    headers = split(lines[1], ",")
    rows = Dict{String,String}[]
    for line in lines[2:end]
        isempty(strip(line)) && continue
        vals = split(line, ","; keepempty = true)
        row = Dict{String,String}()
        for (i, h) in pairs(headers)
            row[h] = i <= length(vals) ? vals[i] : ""
        end
        push!(rows, row)
    end
    return headers, rows
end

parse_csv_float(row, key) = tryparse(Float64, get(row, key, "NaN"))

function plot_methods()
    methods = ["our_coeffspace_packed", "muntz_collocation"]
    PLOT_OURS_DENSE && push!(methods, "our_coeffspace_dense")
    return methods
end

function finite_rows(rows, method, ykey)
    out = Dict{String,String}[]
    for row in rows
        get(row, "method", "") == method || continue
        N = parse_csv_float(row, "N")
        y = parse_csv_float(row, ykey)
        if N !== nothing && y !== nothing && isfinite(N) && isfinite(y) && N > 0 && y > 0
            push!(out, row)
        end
    end
    sort!(out; by = r -> parse_csv_float(r, "N"))
    return out
end

function local_slopes(mrows)
    xs = Float64[]
    ps = Float64[]
    length(mrows) < 2 && return xs, ps
    Ns = [parse_csv_float(r, "N") for r in mrows]
    ts = [parse_csv_float(r, "build_time_min") for r in mrows]
    for i in 1:(length(Ns) - 1)
        if Ns[i] > 0 && Ns[i + 1] > Ns[i] && ts[i] > 0 && ts[i + 1] > 0
            push!(xs, sqrt(Ns[i] * Ns[i + 1]))
            push!(ps, log(ts[i + 1] / ts[i]) / log(Ns[i + 1] / Ns[i]))
        end
    end
    return xs, ps
end

function save_both(fig, fig_dir, stem)
    pdf = joinpath(fig_dir, stem * ".pdf")
    png = joinpath(fig_dir, stem * ".png")
    Base.invokelatest(Plots.savefig, fig, pdf)
    Base.invokelatest(Plots.savefig, fig, png)
    println("Saved figure: ", pdf)
    println("Saved figure: ", png)
end

function add_time_series!(p, rows, method; xkey = "N", ykey = "build_time_min")
    mrows = finite_rows(rows, method, ykey)
    isempty(mrows) && return
    xs = [parse_csv_float(r, xkey) for r in mrows]
    ys = [parse_csv_float(r, ykey) for r in mrows]
    Base.invokelatest(Plots.plot!, p, xs, ys;
        label = METHOD_LABELS[method],
        color = METHOD_COLORS[method],
        marker = METHOD_MARKERS[method],
        linewidth = 2.2,
        markersize = 4.2)
end

function add_reference_slopes!(p, rows)
    packed = finite_rows(rows, "our_coeffspace_packed", "build_time_min")
    muntz = finite_rows(rows, "muntz_collocation", "build_time_min")
    if length(packed) >= 2
        Ns = [parse_csv_float(r, "N") for r in packed]
        ts = [parse_csv_float(r, "build_time_min") for r in packed]
        i0 = max(1, round(Int, 0.55 * length(Ns)))
        x0 = Ns[i0]
        y0 = ts[i0] * 0.6
        xref = [x0, maximum(Ns)]
        yref = y0 .* (xref ./ x0).^2
        Base.invokelatest(Plots.plot!, p, xref, yref;
            color = :black, linestyle = :dash, linewidth = 1.4, label = "O(N^2)")
    end
    if length(muntz) >= 2
        Ns = [parse_csv_float(r, "N") for r in muntz]
        ts = [parse_csv_float(r, "build_time_min") for r in muntz]
        i0 = max(1, round(Int, 0.50 * length(Ns)))
        x0 = Ns[i0]
        y0 = ts[i0] * 1.8
        xref = [x0, maximum(Ns)]
        yref = y0 .* (xref ./ x0).^3
        Base.invokelatest(Plots.plot!, p, xref, yref;
            color = :red, linestyle = :dash, linewidth = 1.4, label = "O(N^3)")
    end
end

function plot_build_ours_vs_muntz(results_csv::AbstractString = RESULTS_CSV,
    fig_dir::AbstractString = FIGURES_DIR)
    _, rows = read_simple_csv(results_csv)
    isempty(rows) && error("No rows found in $(results_csv)")
    mkpath(fig_dir)

    try
        @eval import Plots
        Base.invokelatest(Plots.gr)

        methods = plot_methods()
        common_loglog = (
            xscale = :log10,
            yscale = :log10,
            framestyle = :box,
            grid = false,
            legend = :outerright,
            dpi = 300,
            size = (840, 520),
        )

        p1 = Base.invokelatest(Plots.plot, Float64[], Float64[];
            common_loglog..., xlabel = "Degree N", ylabel = "Build time (s)")
        for method in methods
            add_time_series!(p1, rows, method)
        end
        add_reference_slopes!(p1, rows)
        save_both(p1, fig_dir, "build_ours_vs_muntz_time_vs_N_loglog")

        p2 = Base.invokelatest(Plots.plot, Float64[], Float64[];
            yscale = :log10,
            framestyle = :box,
            grid = false,
            legend = :outerright,
            dpi = 300,
            size = (840, 520),
            xlabel = "Degree N",
            ylabel = "Build time (s)")
        for method in methods
            add_time_series!(p2, rows, method)
        end
        save_both(p2, fig_dir, "build_ours_vs_muntz_time_vs_N")

        p3 = Base.invokelatest(Plots.plot, Float64[], Float64[];
            framestyle = :box,
            grid = false,
            legend = :outerright,
            dpi = 300,
            size = (840, 520),
            xlabel = "Degree N",
            ylabel = "Local log-log slope")
        for method in methods
            mrows = finite_rows(rows, method, "build_time_min")
            xs, ps = local_slopes(mrows)
            isempty(xs) && continue
            Base.invokelatest(Plots.plot!, p3, xs, ps;
                xscale = :log10,
                label = METHOD_LABELS[method],
                color = METHOD_COLORS[method],
                marker = METHOD_MARKERS[method],
                linewidth = 2.0)
        end
        Base.invokelatest(Plots.hline!, p3, [2.0]; color = :black, linestyle = :dash,
            linewidth = 1.2, label = "slope 2")
        Base.invokelatest(Plots.hline!, p3, [3.0]; color = :red, linestyle = :dash,
            linewidth = 1.2, label = "slope 3")
        save_both(p3, fig_dir, "build_ours_vs_muntz_slope")

        p4 = Base.invokelatest(Plots.plot, Float64[], Float64[];
            xscale = :log10,
            yscale = :log10,
            framestyle = :box,
            grid = false,
            legend = :outerright,
            dpi = 300,
            size = (840, 520),
            xlabel = "Degree N",
            ylabel = "Estimated memory (GB)")
        for method in methods
            mrows = finite_rows(rows, method, "estimated_total_bytes")
            isempty(mrows) && continue
            xs = [parse_csv_float(r, "N") for r in mrows]
            ys = [parse_csv_float(r, "estimated_total_bytes") / 1024.0^3 for r in mrows]
            Base.invokelatest(Plots.plot!, p4, xs, ys;
                label = METHOD_LABELS[method],
                color = METHOD_COLORS[method],
                marker = METHOD_MARKERS[method],
                linewidth = 2.2,
                markersize = 4.2)
        end
        save_both(p4, fig_dir, "build_ours_vs_muntz_memory_estimate")

        return true
    catch err
        @warn "Plotting failed; CSV files are available." exception = (err, catch_backtrace())
        return false
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    plot_build_ours_vs_muntz()
end
