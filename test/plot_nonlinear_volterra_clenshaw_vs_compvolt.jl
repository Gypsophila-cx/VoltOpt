using Printf
using Plots
using LaTeXStrings

gr()

# ============================================================
# Paths
# ============================================================

script_dir = @__DIR__
project_root = dirname(script_dir)
data_dir = joinpath(project_root, "test", "data")
fig_dir = joinpath(project_root, "test", "figures")
mkpath(data_dir)
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

function require_column(col::Dict{String,Int}, name::String)
    haskey(col, name) && return col[name]
    error("CSV is missing required column: $(name)")
end

function parse_float_cell(row, idx)
    idx > length(row) && return NaN
    v = tryparse(Float64, strip(row[idx]))
    return v === nothing ? NaN : v
end

function read_bench_data(csvfile::AbstractString)
    rows, col = read_csv_rows(csvfile)

    method_col = require_column(col, "method")
    n_col = require_column(col, "N")
    time_col = require_column(col, "time")

    data = Dict{String,Tuple{Vector{Float64},Vector{Float64}}}()
    for row in rows
        method_col > length(row) && continue
        method = strip(row[method_col])
        N = parse_float_cell(row, n_col)
        t = parse_float_cell(row, time_col)

        if !isempty(method) && isfinite(N) && isfinite(t) && N > 0 && t > 0
            pair = get!(data, method, (Float64[], Float64[]))
            push!(pair[1], N)
            push!(pair[2], t)
        end
    end

    for method in ("clenshaw_valuemat", "compVolt")
        haskey(data, method) || error("No timing rows found for method: $(method)")
        N, t = data[method]
        p = sortperm(N)
        data[method] = (N[p], t[p])
    end

    return data
end

# ============================================================
# Main plot
# ============================================================

function tail_range(v; ntail::Int = 4)
    n = length(v)
    lo = max(1, n - ntail + 1)
    return lo:n
end

reference_power(N, N0, t0, p) = t0 .* (N ./ N0) .^ p

function reference_n2logn(N, N0, t0)
    return t0 .* (N ./ N0) .^ 2 .* (log.(N) ./ log(N0))
end

function plot_nonlinear_volterra_bench(
    csvfile::AbstractString;
    output_pdf::String = joinpath(fig_dir, "bench_nonlinear_volterra_clenshaw_vs_compvolt.pdf"),
)
    data = read_bench_data(csvfile)

    N_clenshaw, t_clenshaw = data["clenshaw_valuemat"]
    N_compvolt, t_compvolt = data["compVolt"]

    plt = plot(
        N_clenshaw,
        t_clenshaw;
        xscale = :log10,
        yscale = :log10,
        marker = :circle,
        markersize = 5,
        linewidth = 2.5,
        label = "clenshaw",
        xlabel = "N",
        ylabel = "time (sec)",
        legend = :topleft,
        framestyle = :box,
        grid = true,
        title = "",
        dpi = 300,
    )

    plot!(
        plt,
        N_compvolt,
        t_compvolt;
        marker = :diamond,
        markersize = 5,
        linewidth = 2.5,
        label = "fast",
    )

    ydata_for_limits = [t_clenshaw; t_compvolt]

    if length(N_clenshaw) >= 2
        r = tail_range(N_clenshaw; ntail = min(4, length(N_clenshaw)))
        idx = first(r)
        clenshaw_ref = reference_power(N_clenshaw, N_clenshaw[idx], t_clenshaw[idx] * 0.85, 3)

        plot!(
            plt,
            N_clenshaw[r],
            clenshaw_ref[r];
            linestyle = :dash,
            linewidth = 1.5,
            color = :black,
            label = nothing,
        )
        annotate!(
            plt,
            N_clenshaw[last(r)],
            clenshaw_ref[last(r)] * 0.2,
            text("O(N³)", 10, :right),
        )
        append!(ydata_for_limits, clenshaw_ref[r])
    end

    if length(N_compvolt) >= 2
        r = tail_range(N_compvolt; ntail = min(4, length(N_compvolt)))
        idx = first(r)
        compvolt_ref = reference_n2logn(N_compvolt, N_compvolt[idx], t_compvolt[idx] * 0.8)

        plot!(
            plt,
            N_compvolt[r],
            compvolt_ref[r];
            linestyle = :dash,
            linewidth = 1.5,
            color = :black,
            label = nothing,
        )
        annotate!(
            plt,
            N_compvolt[last(r)],
            compvolt_ref[last(r)] * 0.1,
            text("O(N² log N)", 10, :right),
        )
        append!(ydata_for_limits, compvolt_ref[r])
    end

    xtick_values = Float64.(sort(unique(vcat(N_clenshaw, N_compvolt))))
    xticks!(plt, xtick_values, string.(Int.(round.(xtick_values))))

    ylims!(plt, 10.0^floor(log10(minimum(ydata_for_limits))), 10.0^ceil(log10(maximum(ydata_for_limits))))

    savefig(plt, output_pdf)
    println("PDF saved to: ", output_pdf)

    return plt
end

# ============================================================
# Run
# ============================================================

csvfile = joinpath(data_dir, "bench_nonlinear_volterra_clenshaw_vs_compvolt.csv")
println("Using CSV: ", csvfile)

plot_nonlinear_volterra_bench(
    csvfile;
    output_pdf = joinpath(fig_dir, "bench_nonlinear_volterra_clenshaw_vs_compvolt.pdf"),
)
