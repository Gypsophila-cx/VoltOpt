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
        push!(rows, String.(strip.(split(line, ","; keepempty = true))))
    end

    col = Dict(header[i] => i for i in eachindex(header))
    return rows, col
end

function parse_float_cell(row, idx)
    idx > length(row) && return NaN
    s = strip(row[idx])
    isempty(s) && return NaN
    v = tryparse(Float64, s)
    return v === nothing ? NaN : v
end

function parse_string_cell(row, idx)
    idx > length(row) && return ""
    return strip(row[idx])
end

function require_column(col::Dict{String,Int}, names::AbstractVector{String})
    for name in names
        haskey(col, name) && return name
    end

    available = sort!(collect(keys(col)))
    error(
        "CSV does not contain any of these required columns: " *
        "$(join(names, ", ")). Available columns: $(join(available, ", "))"
    )
end

function optional_column(col::Dict{String,Int}, names::AbstractVector{String})
    for name in names
        haskey(col, name) && return name
    end
    return nothing
end

function read_xy_data(
    csvfile::AbstractString;
    x_names::AbstractVector{String},
    y_names::AbstractVector{String},
)
    return read_timing_curve(csvfile, x_names, y_names)
end

function read_timing_curve(
    csvfile::AbstractString,
    x_candidates::AbstractVector{String},
    y_candidates::AbstractVector{String},
)
    rows, col = read_csv_rows(csvfile)

    x_col = require_column(col, x_candidates)
    y_col = require_column(col, y_candidates)
    status_col = optional_column(col, ["status"])

    x = Float64[]
    y = Float64[]

    for row in rows
        if status_col !== nothing &&
           lowercase(parse_string_cell(row, col[status_col])) != "success"
            continue
        end

        xi = parse_float_cell(row, col[x_col])
        yi = parse_float_cell(row, col[y_col])

        if isfinite(xi) && isfinite(yi) && xi > 0 && yi > 0
            push!(x, xi)
            push!(y, yi)
        end
    end

    isempty(x) && error("No valid timing data found in $(csvfile).")

    p = sortperm(x)
    return x[p], y[p]
end

# ============================================================
# Reference curves
# ============================================================

reference_power(x, x0, y0, p) = y0 .* (x ./ x0).^p

function reference_n2logn(x, x0, y0)
    return y0 .* (x ./ x0).^2 .* (log.(x) ./ log(x0))
end

function tail_range(v; ntail::Int = 4)
    n = length(v)
    lo = max(1, n - ntail + 1)
    return lo:n
end

function set_log_ylims!(plt, ydata)
    ymin = minimum(ydata)
    ymax = maximum(ydata)
    ylims!(plt, 10.0^floor(log10(ymin)), 10.0^ceil(log10(ymax)))
    return plt
end

function set_log10_xticks!(plt, xdata)
    xmin = minimum(xdata)
    xmax = maximum(xdata)
    pmin = ceil(Int, log10(xmin))
    pmax = floor(Int, log10(xmax))
    ticks = [10.0^p for p in pmin:pmax]
    labels = ["10$(superscript_int(p))" for p in pmin:pmax]
    isempty(ticks) || xticks!(plt, ticks, labels)
    return plt
end

function superscript_int(n::Int)
    chars = Dict(
        '-' => '⁻',
        '0' => '⁰',
        '1' => '¹',
        '2' => '²',
        '3' => '³',
        '4' => '⁴',
        '5' => '⁵',
        '6' => '⁶',
        '7' => '⁷',
        '8' => '⁸',
        '9' => '⁹',
    )
    return join(chars[c] for c in string(n))
end

# ============================================================
# Main plots
# ============================================================

function plot_fixedN1000_vary_bandwidth(
    gutleb_csv::AbstractString,
    cheb_csv::AbstractString;
    output_pdf::String = joinpath(fig_dir, "compare_fixedN1000_vary_bandwidth_time.pdf"),
    output_png::String = joinpath(fig_dir, "compare_fixedN1000_vary_bandwidth_time.png"),
)
    n_gutleb, t_gutleb = read_xy_data(
        gutleb_csv;
        x_names = ["input_bandwidth"],
        y_names = ["old_construct_time_min", "construct_from_coeffs_time_min"],
    )

    n_cheb, t_cheb = read_xy_data(
        cheb_csv;
        x_names = ["input_bandwidth"],
        y_names = ["construct_from_coeffs_time_min"],
    )

    plt = plot(
        n_gutleb,
        t_gutleb;
        xscale = :log10,
        yscale = :log10,
        linewidth = 2.5,
        label = "Gutleb-Olver",
        xlabel = "bandwidth n",
        ylabel = "construction time / s",
        legend = :topleft,
        framestyle = :box,
        title = "",
        dpi = 300,
    )

    plot!(
        plt,
        n_cheb,
        t_cheb;
        linewidth = 2.5,
        label = "Chebyshev-Duffy",
    )

    ydata_for_limits = [t_gutleb; t_cheb]

    if length(n_gutleb) >= 2
        r = tail_range(n_gutleb; ntail = min(10, length(n_gutleb)))
        idx = first(r)
        ref = reference_power(n_gutleb, n_gutleb[idx], t_gutleb[idx] * 0.65, 3)
        plot!(
            plt,
            n_gutleb[r],
            ref[r];
            linestyle = :dash,
            linewidth = 1.2,
            color = :black,
            label = nothing,
        )
        annotate!(plt, n_gutleb[last(r)-1], ref[last(r)], text("O(n³)", 10, :right))
        append!(ydata_for_limits, ref[r])
    end

    if length(n_cheb) >= 2
        r = tail_range(n_cheb; ntail = min(10, length(n_cheb)))
        idx = first(r)
        ref = reference_n2logn(n_cheb, n_cheb[idx], t_cheb[idx] * 0.6)
        plot!(
            plt,
            n_cheb[r],
            ref[r];
            linestyle = :dash,
            linewidth = 1.2,
            color = :black,
            label = nothing,
        )
        annotate!(plt, n_cheb[last(r)], ref[last(r)] * 0.1, text("O(n² log n)", 10, :right))
        append!(ydata_for_limits, ref[r])
    end

    set_log_ylims!(plt, ydata_for_limits)
    set_log10_xticks!(plt, [n_gutleb; n_cheb])

    savefig(plt, output_pdf)
    println("PDF saved to: ", output_pdf)
    savefig(plt, output_png)
    println("PNG saved to: ", output_png)

    return plt
end

function plot_fixed_bandwidth20_varyN(
    gutleb_csv::AbstractString,
    cheb_csv::AbstractString;
    output_pdf::String = joinpath(fig_dir, "compare_fixed_bandwidth20_varyN_time.pdf"),
    output_png::String = joinpath(fig_dir, "compare_fixed_bandwidth20_varyN_time.png"),
)
    N_gutleb, t_gutleb = read_xy_data(
        gutleb_csv;
        x_names = ["N"],
        y_names = ["construct_from_coeffs_time_min"],
    )

    N_cheb, t_cheb = read_xy_data(
        cheb_csv;
        x_names = ["N"],
        y_names = ["construct_from_coeffs_time_min"],
    )

    plt = plot(
        N_gutleb,
        t_gutleb;
        xscale = :log10,
        yscale = :log10,
        linewidth = 2.5,
        label = "Gutleb-Olver",
        xlabel = "truncation dimension N",
        ylabel = "construction time (sec)",
        legend = :topleft,
        framestyle = :box,
        title = "",
        dpi = 300,
    )

    plot!(
        plt,
        N_cheb,
        t_cheb;
        linewidth = 2.5,
        label = "Chebyshev-Duffy",
    )

    ydata_for_limits = [t_gutleb; t_cheb]

    if length(N_gutleb) >= 2
        r = tail_range(N_gutleb; ntail = min(10, length(N_gutleb)))
        idx = first(r)
        ref = reference_power(N_gutleb, N_gutleb[idx], t_gutleb[idx] * 0.8, 1)
        plot!(
            plt,
            N_gutleb[r],
            ref[r];
            linestyle = :dash,
            linewidth = 1.2,
            color = :black,
            label = nothing,
        )
        annotate!(plt, N_gutleb[last(r)], ref[last(r)]*1.5, text("O(N)", 10, :right))
        append!(ydata_for_limits, ref[r])
    end

    if length(N_cheb) >= 2
        r = tail_range(N_cheb; ntail = min(10, length(N_cheb)))
        idx = first(r)
        ref = reference_power(N_cheb, N_cheb[idx], t_cheb[idx] * 0.8, 1)
        plot!(
            plt,
            N_cheb[r],
            ref[r];
            linestyle = :dash,
            linewidth = 1.2,
            color = :black,
            label = nothing,
        )
        annotate!(plt, N_cheb[last(r)], ref[last(r)]*0.6, text("O(N)", 10, :right))
        append!(ydata_for_limits, ref[r])
    end

    set_log_ylims!(plt, ydata_for_limits)
    set_log10_xticks!(plt, [N_gutleb; N_cheb])

    savefig(plt, output_pdf)
    println("PDF saved to: ", output_pdf)
    savefig(plt, output_png)
    println("PNG saved to: ", output_png)

    return plt
end

function plot_fixedN1000_vary_bandwidth_end_to_end(
    gutleb_csv::AbstractString,
    cheb_csv::AbstractString;
    output_pdf::String = joinpath(fig_dir, "compare_fixedN1000_vary_bandwidth_end_to_end_time.pdf"),
    output_png::String = joinpath(fig_dir, "compare_fixedN1000_vary_bandwidth_end_to_end_time.png"),
)
    n_gutleb, t_gutleb = read_timing_curve(
        gutleb_csv,
        ["input_bandwidth"],
        [
            "fresh_total_sum_from_old_construct_min",
            "fresh_total_sum_min",
            "fresh_end_to_end_time_min",
            "end_to_end_time_min",
        ],
    )

    n_cheb, t_cheb = read_timing_curve(
        cheb_csv,
        ["input_bandwidth"],
        ["end_to_end_time_min", "total_time_sum_min"],
    )

    plt = plot(
        n_gutleb,
        t_gutleb;
        xscale = :log10,
        yscale = :log10,
        linewidth = 2.5,
        label = "Gutleb-Olver",
        xlabel = "bandwidth n",
        ylabel = "time (sec)",
        legend = :topleft,
        framestyle = :box,
        title = "",
        dpi = 300,
    )

    plot!(
        plt,
        n_cheb,
        t_cheb;
        linewidth = 2.5,
        label = "Chebyshev-Duffy",
    )

    ydata_for_limits = [t_gutleb; t_cheb]

    if length(n_gutleb) >= 2
        r = tail_range(n_gutleb; ntail = min(10, length(n_gutleb)))
        idx = first(r)
        ref = reference_power(n_gutleb, n_gutleb[idx], t_gutleb[idx] * 0.65, 3)
        plot!(
            plt,
            n_gutleb[r],
            ref[r];
            linestyle = :dash,
            linewidth = 1.2,
            color = :black,
            label = nothing,
        )
        annotate!(plt, n_gutleb[last(r)], ref[last(r)] * 0.2, text("O(n³)", 10, :right))
        append!(ydata_for_limits, ref[r])
    end

    if length(n_cheb) >= 2
        r = tail_range(n_cheb; ntail = min(10, length(n_cheb)))
        idx = first(r)
        ref = reference_n2logn(n_cheb, n_cheb[idx], t_cheb[idx] * 0.6)
        plot!(
            plt,
            n_cheb[r],
            ref[r];
            linestyle = :dash,
            linewidth = 1.2,
            color = :black,
            label = nothing,
        )
        annotate!(plt, n_cheb[last(r)], ref[last(r)] * 0.1, text("O(n² log n)", 10, :right))
        append!(ydata_for_limits, ref[r])
    end

    set_log_ylims!(plt, ydata_for_limits)
    set_log10_xticks!(plt, [n_gutleb; n_cheb])

    savefig(plt, output_pdf)
    println("PDF saved to: ", output_pdf)
    savefig(plt, output_png)
    println("PNG saved to: ", output_png)

    return plt
end

function plot_fixed_bandwidth20_varyN_end_to_end(
    gutleb_csv::AbstractString,
    cheb_csv::AbstractString;
    output_pdf::String = joinpath(fig_dir, "compare_fixed_bandwidth20_varyN_end_to_end_time.pdf"),
    output_png::String = joinpath(fig_dir, "compare_fixed_bandwidth20_varyN_end_to_end_time.png"),
)
    N_gutleb, t_gutleb = read_timing_curve(
        gutleb_csv,
        ["N"],
        [
            "cached_end_to_end_time_min",
            "cached_total_sum_min",
            "end_to_end_time_min",
            "total_time_sum_min",
        ],
    )

    N_cheb, t_cheb = read_timing_curve(
        cheb_csv,
        ["N"],
        ["end_to_end_time_min", "total_time_sum_min"],
    )

    plt = plot(
        N_gutleb,
        t_gutleb;
        xscale = :log10,
        yscale = :log10,
        linewidth = 2.5,
        label = "Gutleb-Olver",
        xlabel = "truncation dimension N",
        ylabel = "time (sec)",
        legend = :topleft,
        framestyle = :box,
        title = "",
        dpi = 300,
    )

    plot!(
        plt,
        N_cheb,
        t_cheb;
        linewidth = 2.5,
        label = "Chebyshev-Duffy",
    )

    ydata_for_limits = [t_gutleb; t_cheb]

    if length(N_gutleb) >= 2
        r = tail_range(N_gutleb; ntail = min(10, length(N_gutleb)))
        idx = first(r)
        ref = reference_power(N_gutleb, N_gutleb[idx], t_gutleb[idx] * 0.9, 1)
        plot!(
            plt,
            N_gutleb[r],
            ref[r];
            linestyle = :dash,
            linewidth = 1.2,
            color = :black,
            label = nothing,
        )
        annotate!(plt, N_gutleb[last(r)], ref[last(r)] * 0.65, text("O(N)", 10, :right))
        append!(ydata_for_limits, ref[r])
    end

    if length(N_cheb) >= 2
        r = tail_range(N_cheb; ntail = min(10, length(N_cheb)))
        idx = first(r)
        ref = reference_power(N_cheb, N_cheb[idx], t_cheb[idx] * 0.8, 1)
        plot!(
            plt,
            N_cheb[r],
            ref[r];
            linestyle = :dash,
            linewidth = 1.2,
            color = :black,
            label = nothing,
        )
        annotate!(plt, N_cheb[last(r)], ref[last(r)] * 0.5, text("O(N)", 10, :right))
        append!(ydata_for_limits, ref[r])
    end

    set_log_ylims!(plt, ydata_for_limits)
    set_log10_xticks!(plt, [N_gutleb; N_cheb])

    savefig(plt, output_pdf)
    println("PDF saved to: ", output_pdf)
    savefig(plt, output_png)
    println("PNG saved to: ", output_png)

    return plt
end

# ============================================================
# Run
# ============================================================

gutleb_fixedN_csv = joinpath(data_dir, "gutleb_refined_light_fixedN1000_vary_bandwidth_time.csv")
cheb_fixedN_csv = joinpath(data_dir, "voltmat_cheb_fixedN1000_vary_bandwidth_time.csv")
gutleb_varyN_csv = joinpath(data_dir, "gutleb_refined_fixed_bandwidth20_varyN_time.csv")
cheb_varyN_csv = joinpath(data_dir, "voltmat_cheb_fixed_bandwidth20_varyN_time.csv")

println("Using Gutleb fixed-N CSV: ", gutleb_fixedN_csv)
println("Using Chebyshev fixed-N CSV: ", cheb_fixedN_csv)
plot_fixedN1000_vary_bandwidth(
    gutleb_fixedN_csv,
    cheb_fixedN_csv;
    output_pdf = joinpath(fig_dir, "compare_fixedN1000_vary_bandwidth_time.pdf"),
    output_png = joinpath(fig_dir, "compare_fixedN1000_vary_bandwidth_time.png"),
)

println("Using Gutleb varying-N CSV: ", gutleb_varyN_csv)
println("Using Chebyshev varying-N CSV: ", cheb_varyN_csv)
plot_fixed_bandwidth20_varyN(
    gutleb_varyN_csv,
    cheb_varyN_csv;
    output_pdf = joinpath(fig_dir, "compare_fixed_bandwidth20_varyN_time.pdf"),
    output_png = joinpath(fig_dir, "compare_fixed_bandwidth20_varyN_time.png"),
)

println("Using Gutleb fixed-N end-to-end CSV: ", gutleb_fixedN_csv)
println("Using Chebyshev fixed-N end-to-end CSV: ", cheb_fixedN_csv)
plot_fixedN1000_vary_bandwidth_end_to_end(
    gutleb_fixedN_csv,
    cheb_fixedN_csv;
    output_pdf = joinpath(fig_dir, "compare_fixedN1000_vary_bandwidth_end_to_end_time.pdf"),
    output_png = joinpath(fig_dir, "compare_fixedN1000_vary_bandwidth_end_to_end_time.png"),
)

println("Using Gutleb varying-N end-to-end CSV: ", gutleb_varyN_csv)
println("Using Chebyshev varying-N end-to-end CSV: ", cheb_varyN_csv)
plot_fixed_bandwidth20_varyN_end_to_end(
    gutleb_varyN_csv,
    cheb_varyN_csv;
    output_pdf = joinpath(fig_dir, "compare_fixed_bandwidth20_varyN_end_to_end_time.pdf"),
    output_png = joinpath(fig_dir, "compare_fixed_bandwidth20_varyN_end_to_end_time.png"),
)

println("Finished comparison plots.")
