using Printf

const SCRIPT_DIR = @__DIR__
const DATA_DIR = joinpath(SCRIPT_DIR, "data")
const FIGURES_DIR = joinpath(SCRIPT_DIR, "figures")
const RESULTS_CSV = joinpath(DATA_DIR, "ours_muntz_cauchy_results.csv")
const SOLUTIONS_CSV = joinpath(DATA_DIR, "ours_muntz_cauchy_solutions.csv")
mkpath(FIGURES_DIR)

const METHOD_LABELS = Dict(
    "our_coeffspace" => "Our coefficient method",
    "muntz_collocation" => "Müntz collocation",
)

const METHOD_MARKERS = Dict(
    "our_coeffspace" => :star5,
    "muntz_collocation" => :diamond,
)

const METHOD_COLORS = Dict(
    "our_coeffspace" => :black,
    "muntz_collocation" => :red,
)

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

function finite_method_rows(rows, method, ykey)
    out = Dict{String,String}[]
    for row in rows
        get(row, "method", "") == method || continue
        dof = parse_csv_float(row, "dof")
        y = parse_csv_float(row, ykey)
        t = parse_csv_float(row, "total_time")
        if dof !== nothing && y !== nothing && t !== nothing &&
           isfinite(dof) && isfinite(y) && isfinite(t) && dof > 0 && y > 0 && t > 0
            push!(out, row)
        end
    end
    sort!(out; by = r -> parse_csv_float(r, "dof"))
    return out
end

function plot_ours_muntz_cauchy(results_csv::AbstractString = RESULTS_CSV,
    solutions_csv::AbstractString = SOLUTIONS_CSV, fig_dir::AbstractString = FIGURES_DIR)
    _, rows = read_simple_csv(results_csv)
    _, solrows = read_simple_csv(solutions_csv)
    isempty(rows) && error("No result rows found in $(results_csv)")
    isempty(solrows) && error("No solution rows found in $(solutions_csv)")
    mkpath(fig_dir)

    try
        @eval import Plots
        Base.invokelatest(Plots.gr)
        methods = ["our_coeffspace", "muntz_collocation"]
        common = (
            xscale = :log10,
            yscale = :log10,
            framestyle = :box,
            grid = false,
            legend = :outerright,
            dpi = 300,
            size = (820, 520),
        )

        function add_series!(p, method, xkey, ykey)
            mrows = finite_method_rows(rows, method, ykey)
            isempty(mrows) && return
            xs = [parse_csv_float(r, xkey) for r in mrows]
            ys = [parse_csv_float(r, ykey) for r in mrows]
            Base.invokelatest(Plots.plot!, p, xs, ys;
                marker = METHOD_MARKERS[method],
                color = METHOD_COLORS[method],
                linewidth = 2.2,
                label = METHOD_LABELS[method])
        end

        function save(fig, stem)
            pdf = joinpath(fig_dir, stem * ".pdf")
            png = joinpath(fig_dir, stem * ".png")
            Base.invokelatest(Plots.savefig, fig, pdf)
            Base.invokelatest(Plots.savefig, fig, png)
            println("Saved figure: ", pdf)
            println("Saved figure: ", png)
        end

        p1 = Base.invokelatest(Plots.plot, Float64[], Float64[];
            common..., xlabel = "Degrees of freedom", ylabel = "Reference Cauchy Linf error")
        for m in methods
            add_series!(p1, m, "dof", "cauchy_ref_linf")
        end
        save(p1, "ours_muntz_cauchy_error_vs_N")

        p2 = Base.invokelatest(Plots.plot, Float64[], Float64[];
            common..., xlabel = "Degrees of freedom", ylabel = "Total time (sec)")
        for m in methods
            add_series!(p2, m, "dof", "total_time")
        end
        save(p2, "ours_muntz_cauchy_time_vs_N")

        p3 = Base.invokelatest(Plots.plot, Float64[], Float64[];
            common..., xlabel = "Total time (sec)", ylabel = "Reference Cauchy Linf error")
        for m in methods
            add_series!(p3, m, "total_time", "cauchy_ref_linf")
        end
        save(p3, "ours_muntz_cauchy_error_vs_time")

        x = [parse_csv_float(r, "x") for r in solrows]
        uo = [parse_csv_float(r, "u_ours") for r in solrows]
        um = [parse_csv_float(r, "u_muntz") for r in solrows]
        diff = [parse_csv_float(r, "difference") for r in solrows]
        p4a = Base.invokelatest(Plots.plot, x, uo;
            framestyle = :box, grid = false, label = "Our coefficient method",
            color = :black, linewidth = 2.0, xlabel = "x", ylabel = "u(x)")
        Base.invokelatest(Plots.plot!, p4a, x, um; label = "Müntz collocation",
            color = :red, linestyle = :dash, linewidth = 2.0)
        p4b = Base.invokelatest(Plots.plot, x, diff;
            framestyle = :box, grid = false, label = false, color = :blue,
            linewidth = 1.8, xlabel = "x", ylabel = "difference")
        p4 = Base.invokelatest(Plots.plot, p4a, p4b; layout = (2, 1), size = (860, 720), dpi = 300)
        save(p4, "ours_muntz_cauchy_solutions")
        return true
    catch err
        @warn "Plotting failed; CSV files are available." exception = (err, catch_backtrace())
        return false
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    plot_ours_muntz_cauchy()
end
