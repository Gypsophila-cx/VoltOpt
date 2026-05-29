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

function read_fixedC_varyN_data(csvfile::AbstractString; use_median::Bool = false)
    rows, col = read_csv_rows(csvfile)

    n_col = require_column(col, ["N"])

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

    N = Float64[]
    cheb_time = Float64[]
    jacobi_time = Float64[]

    for row in rows
        Ni = parse_float_cell(row, col[n_col])
        cheb_raw = parse_float_cell(row, col[cheb_col])
        jacobi_raw = parse_float_cell(row, col[jacobi_col])

        cheb_sec = endswith(cheb_col, "_ms") ? cheb_raw / 1000.0 : cheb_raw
        jacobi_sec = endswith(jacobi_col, "_ms") ? jacobi_raw / 1000.0 : jacobi_raw

        if isfinite(Ni) && isfinite(cheb_sec) && isfinite(jacobi_sec) &&
           Ni > 0 && cheb_sec > 0 && jacobi_sec > 0
            push!(N, Ni)
            push!(cheb_time, cheb_sec)
            push!(jacobi_time, jacobi_sec)
        end
    end

    isempty(N) && error("No valid timing data found in $(csvfile).")

    p = sortperm(N)
    return N[p], cheb_time[p], jacobi_time[p]
end

# ============================================================
# Reference curves
# ============================================================

reference_linear(x, x0, y0) = y0 .* (x ./ x0)

function tail_range(v; ntail::Int = 6)
    n = length(v)
    lo = max(1, n - ntail + 1)
    return lo:n
end

# ============================================================
# Main plot
# ============================================================

function plot_fixedC_varyN_time(
    csvfile::AbstractString;
    output_pdf::String = joinpath(fig_dir, "bench_fixedC_varyN_time.pdf"),
    use_median::Bool = false,
    add_reference_lines::Bool = true,
)
    N, cheb_time, jacobi_time = read_fixedC_varyN_data(
        csvfile;
        use_median = use_median,
    )

    plt = plot(
        N,
        cheb_time;
        xscale = :log10,
        yscale = :log10,
        marker = :circle,
        markersize = 5,
        linewidth = 2.5,
        label = "Chebyshev Duffy",
        xlabel = "matrix size N",
        ylabel = "construction time / s",
        legend = :topleft,
        framestyle = :box,
        title = "",
        dpi = 300,
    )

    plot!(
        plt,
        N,
        jacobi_time;
        marker = :diamond,
        markersize = 5,
        linewidth = 2.5,
        label = "Jacobi Duffy",
    )

    ydata_for_limits = [cheb_time; jacobi_time]

    if add_reference_lines && length(N) >= 2
        r = tail_range(N; ntail = min(6, length(N)))
        idx = first(r)

        cheb_ref = reference_linear(N, N[idx], cheb_time[idx] * 0.9)
        plot!(
            plt,
            N[r],
            cheb_ref[r];
            linestyle = :dash,
            linewidth = 1.5,
            color = :black,
            label = nothing,
        )
        annotate!(plt, N[last(r)], cheb_ref[last(r)] * 0.6, text("O(N)", 10, :right))

        jacobi_ref = reference_linear(N, N[idx], jacobi_time[idx] * 0.9)
        plot!(
            plt,
            N[r],
            jacobi_ref[r];
            linestyle = :dash,
            linewidth = 1.5,
            color = :black,
            label = nothing,
        )
        annotate!(plt, N[max(first(r), last(r) - 1)], jacobi_ref[last(r)]*1.1 , text("O(N)", 10, :right))

        append!(ydata_for_limits, cheb_ref)
        append!(ydata_for_limits, jacobi_ref)
    end

    ymin = minimum(ydata_for_limits)
    ymax = maximum(ydata_for_limits)
    ylims!(plt, 10.0^floor(log10(ymin)), 10.0^ceil(log10(ymax)))

    savefig(plt, output_pdf)
    println("PDF saved to: ", output_pdf)

    return plt
end

# ============================================================
# Run
# ============================================================

csvfile = joinpath(data_dir, "bench_fixedC_varyN.csv")
println("Using CSV: ", csvfile)

plot_fixedC_varyN_time(
    csvfile;
    output_pdf = joinpath(fig_dir, "bench_fixedC_varyN_time.pdf"),
    use_median = false,
    add_reference_lines = true,
)
