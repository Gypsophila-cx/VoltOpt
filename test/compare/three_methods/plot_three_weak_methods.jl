using Printf

function find_project_root(start_dir::AbstractString = @__DIR__)
    dir = abspath(start_dir)
    while true
        if isfile(joinpath(dir, "Project.toml")) && isdir(joinpath(dir, "src"))
            return dir
        end
        parent = dirname(dir)
        parent == dir && error("Could not find project root from $(start_dir).")
        dir = parent
    end
end

const SCRIPT_DIR = @__DIR__
const DATA_DIR = joinpath(SCRIPT_DIR, "data")
const FIGURES_DIR = joinpath(SCRIPT_DIR, "figures")
const EXAMPLE = lowercase(strip(get(ENV, "EXAMPLE", "oscillatory_sqrt")))
EXAMPLE in ("oscillatory_sqrt", "low_order_sqrt_poly") ||
    error("Unsupported EXAMPLE=$(EXAMPLE).")
output_stem(base::AbstractString) = EXAMPLE == "low_order_sqrt_poly" ? base : base * "_" * EXAMPLE
const DEFAULT_CSV = joinpath(DATA_DIR, output_stem("three_weak_methods") * "_results.csv")

mkpath(FIGURES_DIR)

const METHOD_LABELS = Dict(
    "our_coeffspace" => "Our coefficient method",
    "brunner_graded" => "Brunner graded collocation",
    "muntz_collocation" => "Müntz collocation",
    "direct_jacobi_collocation" => "Direct Jacobi collocation",
    "chentang_transformed_collocation" => "Chen--Tang transformed collocation",
)

const METHOD_MARKERS = Dict(
    "our_coeffspace" => :star5,
    "brunner_graded" => :circle,
    "muntz_collocation" => :diamond,
    "direct_jacobi_collocation" => :square,
    "chentang_transformed_collocation" => :utriangle,
)

const METHOD_COLORS = Dict(
    "our_coeffspace" => :black,
    "brunner_graded" => :blue,
    "muntz_collocation" => :red,
    "direct_jacobi_collocation" => :gray35,
    "chentang_transformed_collocation" => :purple,
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

function finite_method_rows(rows, method)
    out = Dict{String,String}[]
    for row in rows
        get(row, "method", "") == method || continue
        dof = parse_csv_float(row, "dof")
        t = parse_csv_float(row, "total_time")
        err = parse_csv_float(row, "linf_error")
        if dof !== nothing && t !== nothing && err !== nothing &&
           isfinite(dof) && isfinite(t) && isfinite(err) && dof > 0 && t > 0 && err > 0
            push!(out, row)
        end
    end
    sort!(out; by = r -> parse_csv_float(r, "dof"))
    return out
end

function plot_three_weak_methods(csv_path::AbstractString = DEFAULT_CSV,
    fig_dir::AbstractString = FIGURES_DIR)
    _, rows = read_simple_csv(csv_path)
    isempty(rows) && error("No rows found in $(csv_path)")
    mkpath(fig_dir)

    try
        @eval import Plots
        Base.invokelatest(Plots.gr)

        methods = ["our_coeffspace", "brunner_graded"]
        if lowercase(strip(get(ENV, "PLOT_MUNTZ", "1"))) in ("1", "true", "yes", "on")
            push!(methods, "muntz_collocation")
        end
        if lowercase(strip(get(ENV, "PLOT_DIRECT_JACOBI", "0"))) in ("1", "true", "yes", "on")
            push!(methods, "direct_jacobi_collocation")
        end
        if lowercase(strip(get(ENV, "PLOT_CHENTANG", "0"))) in ("1", "true", "yes", "on")
            push!(methods, "chentang_transformed_collocation")
        end
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
            mrows = finite_method_rows(rows, method)
            isempty(mrows) && return
            xs = [parse_csv_float(r, xkey) for r in mrows]
            ys = [parse_csv_float(r, ykey) for r in mrows]
            Base.invokelatest(Plots.plot!, p, xs, ys;
                marker = METHOD_MARKERS[method],
                color = METHOD_COLORS[method],
                linewidth = 2.2,
                label = METHOD_LABELS[method],
            )
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
            common..., xlabel = "degrees of freedom", ylabel = "Linf error")
        for method in methods
            add_series!(p1, method, "dof", "linf_error")
        end
        save(p1, output_stem("three_methods") * "_error_vs_N")

        p2 = Base.invokelatest(Plots.plot, Float64[], Float64[];
            common..., xlabel = "degrees of freedom", ylabel = "total time (sec)")
        for method in methods
            add_series!(p2, method, "dof", "total_time")
        end
        save(p2, output_stem("three_methods") * "_time_vs_N")

        p3 = Base.invokelatest(Plots.plot, Float64[], Float64[];
            common..., xlabel = "total time (sec)", ylabel = "Linf error")
        for method in methods
            add_series!(p3, method, "total_time", "linf_error")
        end
        save(p3, output_stem("three_methods") * "_error_vs_time")

        return true
    catch err
        @warn "Plotting failed; CSV is still available." exception = (err, catch_backtrace())
        return false
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    plot_three_weak_methods()
end
