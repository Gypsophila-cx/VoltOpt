using Printf

const SCRIPT_DIR = @__DIR__
const DATA_DIR = joinpath(SCRIPT_DIR, "data")
const FIGURES_DIR = joinpath(SCRIPT_DIR, "figures")
const EXAMPLE = lowercase(strip(get(ENV, "EXAMPLE", "oscillatory_sqrt")))
EXAMPLE in ("oscillatory_sqrt", "low_order_sqrt_poly") ||
    error("Unsupported EXAMPLE=$(EXAMPLE).")
output_stem(base::AbstractString) = EXAMPLE == "low_order_sqrt_poly" ? base : base * "_" * EXAMPLE
const RESULTS_CSV = joinpath(DATA_DIR, output_stem("muntz_lambda_scan") * "_results.csv")
const SUMMARY_CSV = joinpath(DATA_DIR, output_stem("muntz_lambda_scan") * "_summary.csv")

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

function plot_scan_muntz_lambda(results_csv::AbstractString = RESULTS_CSV,
    summary_csv::AbstractString = SUMMARY_CSV, fig_dir::AbstractString = FIGURES_DIR)
    _, rows = read_simple_csv(results_csv)
    _, summary = read_simple_csv(summary_csv)
    isempty(rows) && error("No rows found in $(results_csv)")
    isempty(summary) && error("No rows found in $(summary_csv)")
    mkpath(fig_dir)

    try
        @eval import Plots
        Base.invokelatest(Plots.gr)
        markers = [:circle, :diamond, :utriangle, :dtriangle, :square, :star5]

        function save(fig, stem)
            pdf = joinpath(fig_dir, stem * ".pdf")
            png = joinpath(fig_dir, stem * ".png")
            Base.invokelatest(Plots.savefig, fig, pdf)
            Base.invokelatest(Plots.savefig, fig, png)
            println("Saved figure: ", pdf)
            println("Saved figure: ", png)
        end

        srows = [r for r in summary if parse_csv_float(r, "muntz_lambda") !== nothing]
        sort!(srows; by = r -> parse_csv_float(r, "muntz_lambda"))
        lambdas = [parse_csv_float(r, "muntz_lambda") for r in srows]
        best_errs = [parse_csv_float(r, "best_linf_error") for r in srows]
        keep = [i for i in eachindex(lambdas, best_errs) if best_errs[i] !== nothing &&
                isfinite(best_errs[i]) && best_errs[i] > 0]
        p1 = Base.invokelatest(Plots.plot, lambdas[keep], best_errs[keep];
            yscale = :log10,
            framestyle = :box,
            grid = false,
            marker = :circle,
            linewidth = 2.3,
            legend = false,
            dpi = 300,
            size = (760, 480),
            xlabel = "Muntz lambda",
            ylabel = "Best Linf error")
        save(p1, output_stem("muntz_lambda") * "_best_error_vs_lambda")

        lam_vals = sort(unique(parse_csv_float(r, "muntz_lambda") for r in rows
            if parse_csv_float(r, "muntz_lambda") !== nothing))
        p2 = Base.invokelatest(Plots.plot, Float64[], Float64[];
            xscale = :log10,
            yscale = :log10,
            framestyle = :box,
            grid = false,
            legend = :outerright,
            dpi = 300,
            size = (900, 560),
            xlabel = "Degrees of freedom",
            ylabel = "Linf error")
        for (idx, lam) in pairs(lam_vals)
            lrows = [r for r in rows if parse_csv_float(r, "muntz_lambda") == lam &&
                     parse_csv_float(r, "linf_error") !== nothing &&
                     isfinite(parse_csv_float(r, "linf_error")) &&
                     parse_csv_float(r, "linf_error") > 0]
            sort!(lrows; by = r -> parse_csv_float(r, "dof"))
            isempty(lrows) && continue
            xs = [parse_csv_float(r, "dof") for r in lrows]
            ys = [parse_csv_float(r, "linf_error") for r in lrows]
            Base.invokelatest(Plots.plot!, p2, xs, ys;
                marker = markers[mod1(idx, length(markers))],
                linewidth = 2.0,
                label = @sprintf("lambda=%.6g", lam))
        end
        save(p2, output_stem("muntz_lambda") * "_error_vs_N")
        return true
    catch err
        @warn "Muntz lambda scan plotting failed; CSV files are available." exception = (err, catch_backtrace())
        return false
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    plot_scan_muntz_lambda()
end
