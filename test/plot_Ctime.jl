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

function read_fixedN_varyC_data(csvfile::AbstractString; use_median::Bool = false)
    rows, col = read_csv_rows(csvfile)

    c_col = require_column(col, ["C_rows", "n"])

    cheb_col = if haskey(col, use_median ? "cheb_median_ms" : "cheb_min_ms")
        use_median ? "cheb_median_ms" : "cheb_min_ms"
    else
        require_column(col, ["cheb_time_sec"])
    end

    jacobi_col = if haskey(col, use_median ? "jacobi_median_ms" : "jacobi_min_ms")
        use_median ? "jacobi_median_ms" : "jacobi_min_ms"
    else
        require_column(col, ["jacobi_time_sec"])
    end

    C = Float64[]
    cheb_time = Float64[]
    jacobi_time = Float64[]

    for row in rows
        csize = parse_float_cell(row, col[c_col])
        cheb_raw = parse_float_cell(row, col[cheb_col])
        jacobi_raw = parse_float_cell(row, col[jacobi_col])

        cheb_sec = endswith(cheb_col, "_ms") ? cheb_raw / 1000.0 : cheb_raw
        jacobi_sec = endswith(jacobi_col, "_ms") ? jacobi_raw / 1000.0 : jacobi_raw

        if isfinite(csize) && isfinite(cheb_sec) && isfinite(jacobi_sec) &&
           csize > 0 && cheb_sec > 0 && jacobi_sec > 0
            push!(C, csize)
            push!(cheb_time, cheb_sec)
            push!(jacobi_time, jacobi_sec)
        end
    end

    isempty(C) && error("No valid timing data found in $(csvfile).")

    p = sortperm(C)
    return C[p], cheb_time[p], jacobi_time[p]
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

function plot_fixedN_varyC_time(
    csvfile::AbstractString;
    output_pdf::String = joinpath(fig_dir, "bench_fixedN_varyC_time.pdf"),
    use_median::Bool = false,
    add_reference_lines::Bool = true,
)
    C, cheb_time, jacobi_time = read_fixedN_varyC_data(
        csvfile;
        use_median = use_median,
    )

    plt = plot(
        C,
        cheb_time;
        xscale = :log10,
        yscale = :log10,
        marker = :circle,
        markersize = 5,
        linewidth = 2.5,
        label = "Chebyshev Duffy",
        xlabel = "band width n",
        ylabel = "construction time / s",
        legend = :topleft,
        framestyle = :box,
        title = "",
        dpi = 300,
    )

    plot!(
        plt,
        C,
        jacobi_time;
        marker = :diamond,
        markersize = 5,
        linewidth = 2.5,
        label = "Jacobi Duffy",
    )

    ydata_for_limits = [cheb_time; jacobi_time]

    if add_reference_lines && length(C) >= 2
        r = tail_range(C; ntail = min(4, length(C)))
        idx = first(r)

        cheb_ref = reference_power(C, C[idx], cheb_time[idx] * 0.8, 2)
        plot!(
            plt,
            C[r],
            cheb_ref[r];
            linestyle = :dash,
            linewidth = 1.2,
            color = :black,
            label = nothing,
        )
        annotate!(plt, C[last(r)], cheb_ref[last(r)] * 0.2, text("O(n² log n)", 10, :right))

        jacobi_ref = reference_power(C, C[idx], jacobi_time[idx] * 0.45, 3)
        plot!(
            plt,
            C[r],
            jacobi_ref[r];
            linestyle = :dash,
            linewidth = 1.2,
            color = :black,
            label = nothing,
        )
        annotate!(plt, (C[last(r)-1]+C[last(r)])/2, jacobi_ref[last(r)]*0.6, text("O(n³)", 10, :right))

        append!(ydata_for_limits, cheb_ref)
        append!(ydata_for_limits, jacobi_ref)
    end

    ymin = minimum(ydata_for_limits)
    ymax = maximum(ydata_for_limits)
    ylims!(plt, 10.0^floor(log10(ymin)), 10.0^ceil(log10(ymax)))

    xtick_candidates = Float64[]
    for v in C
        iv = round(Int, v)
        if iv in (4, 8, 16, 32, 64, 128, 256, 512, 1024)
            push!(xtick_candidates, v)
        end
    end
    isempty(xtick_candidates) || xticks!(plt, xtick_candidates, string.(Int.(round.(xtick_candidates))))

    savefig(plt, output_pdf)
    println("PDF saved to: ", output_pdf)

    return plt
end

# ============================================================
# Run
# ============================================================

csvfile = joinpath(data_dir, "bench_fixedN_varyC.csv")
println("Using CSV: ", csvfile)

plot_fixedN_varyC_time(
    csvfile;
    output_pdf = joinpath(fig_dir, "bench_fixedN_varyC_time.pdf"),
    use_median = false,
    add_reference_lines = true,
)
