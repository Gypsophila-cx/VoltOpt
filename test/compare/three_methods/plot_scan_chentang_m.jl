using Printf

const SCRIPT_DIR = @__DIR__
const DATA_DIR = joinpath(SCRIPT_DIR, "data")
const FIGURES_DIR = joinpath(SCRIPT_DIR, "figures")
const RESULTS_CSV = joinpath(DATA_DIR, "chentang_m_scan_results.csv")
const SUMMARY_CSV = joinpath(DATA_DIR, "chentang_m_scan_summary.csv")

mkpath(FIGURES_DIR)

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
parse_csv_int(row, key) = tryparse(Int, get(row, key, ""))

function finite_rows(rows, ykey)
    out = Dict{String,String}[]
    for row in rows
        dof = parse_csv_float(row, "dof")
        y = parse_csv_float(row, ykey)
        m = parse_csv_int(row, "m")
        if dof !== nothing && y !== nothing && m !== nothing &&
           isfinite(dof) && isfinite(y) && dof > 0 && y > 0
            push!(out, row)
        end
    end
    return out
end

function plot_scan_chentang_m(results_csv::AbstractString = RESULTS_CSV,
    summary_csv::AbstractString = SUMMARY_CSV, fig_dir::AbstractString = FIGURES_DIR)
    _, rows = read_simple_csv(results_csv)
    _, summary = read_simple_csv(summary_csv)
    isempty(rows) && error("No scan rows found in $(results_csv)")
    isempty(summary) && error("No summary rows found in $(summary_csv)")
    mkpath(fig_dir)

    try
        @eval import Plots
        Base.invokelatest(Plots.gr)

        markers = [:circle, :diamond, :utriangle, :dtriangle, :square, :star5, :hexagon, :cross, :xcross]
        mvals = sort(unique(parse_csv_int(r, "m") for r in rows if parse_csv_int(r, "m") !== nothing))

        function save(fig, stem)
            pdf = joinpath(fig_dir, stem * ".pdf")
            png = joinpath(fig_dir, stem * ".png")
            Base.invokelatest(Plots.savefig, fig, pdf)
            Base.invokelatest(Plots.savefig, fig, png)
            println("Saved figure: ", pdf)
            println("Saved figure: ", png)
        end

        common_loglog = (
            xscale = :log10,
            yscale = :log10,
            framestyle = :box,
            grid = false,
            legend = :outerright,
            dpi = 300,
            size = (900, 560),
        )

        function plot_by_m(stem, ykey, ylabel)
            fig = Base.invokelatest(Plots.plot, Float64[], Float64[];
                common_loglog..., xlabel = "Degrees of freedom", ylabel = ylabel)
            for (idx, m) in pairs(mvals)
                mrows = [r for r in finite_rows(rows, ykey) if parse_csv_int(r, "m") == m]
                sort!(mrows; by = r -> parse_csv_float(r, "dof"))
                isempty(mrows) && continue
                xs = [parse_csv_float(r, "dof") for r in mrows]
                ys = [parse_csv_float(r, ykey) for r in mrows]
                Base.invokelatest(Plots.plot!, fig, xs, ys;
                    marker = markers[mod1(idx, length(markers))],
                    linewidth = 2.0,
                    label = "m=$(m)")
            end
            save(fig, stem)
        end

        plot_by_m("chentang_m_error_vs_N", "linf_error_full_grid", "L^infinity error")
        plot_by_m("chentang_m_transformed_error_vs_N", "transformed_linf_error", "Transformed L^infinity error")

        srows = [r for r in summary if parse_csv_int(r, "m") !== nothing]
        sort!(srows; by = r -> parse_csv_int(r, "m"))
        ms = [parse_csv_float(r, "m") for r in srows]

        function summary_plot(stem, ykey, ylabel)
            ys = [parse_csv_float(r, ykey) for r in srows]
            keep = [i for i in eachindex(ms, ys) if ys[i] !== nothing && isfinite(ys[i]) && ys[i] > 0]
            fig = Base.invokelatest(Plots.plot, ms[keep], ys[keep];
                yscale = :log10,
                framestyle = :box,
                grid = false,
                legend = false,
                marker = :circle,
                linewidth = 2.2,
                dpi = 300,
                size = (760, 480),
                xlabel = "m",
                ylabel = ylabel)
            save(fig, stem)
        end

        summary_plot("chentang_m_best_original_error_vs_m", "best_original_error", "Best original L^infinity error")
        summary_plot("chentang_m_best_transformed_error_vs_m", "best_transformed_error", "Best transformed L^infinity error")
        summary_plot("chentang_m_amplification_vs_m", "best_amplification_indicator", "Amplification indicator")

        return true
    catch err
        @warn "Chen--Tang m scan plotting failed; CSV files are available." exception = (err, catch_backtrace())
        return false
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    plot_scan_chentang_m()
end
