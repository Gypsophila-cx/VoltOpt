ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

using Printf
using Plots

gr()

# ============================================================
# Paths
# ============================================================

function find_project_root(start_dir::AbstractString = @__DIR__)
    dir = abspath(start_dir)
    while true
        if isfile(joinpath(dir, "Project.toml")) && isdir(joinpath(dir, "src"))
            return dir
        end
        parent = dirname(dir)
        parent == dir && error("Could not find the VoltOpt project root from $(start_dir).")
        dir = parent
    end
end

const PROJECT_ROOT = find_project_root()
const DATA_DIR = joinpath(PROJECT_ROOT, "examples", "data")
const FIGURES_DIR = joinpath(PROJECT_ROOT, "examples", "figures")

const CHEB_LEG_CSV = joinpath(DATA_DIR, "vide_low_order_cheb_leg_errors.csv")
const GUTLEB_CSV = joinpath(DATA_DIR, "vide_low_order_gutleb_errors.csv")
const PDF_PATH = joinpath(FIGURES_DIR, "vide_low_order_accuracy_compare.pdf")
const PNG_PATH = joinpath(FIGURES_DIR, "vide_low_order_accuracy_compare.png")

# ============================================================
# CSV readers
# ============================================================

function split_csv_line(line::AbstractString)
    cells = String[]
    buf = IOBuffer()
    in_quotes = false
    i = firstindex(line)

    while i <= lastindex(line)
        c = line[i]
        if c == '"'
            next_i = nextind(line, i)
            if in_quotes && next_i <= lastindex(line) && line[next_i] == '"'
                print(buf, '"')
                i = nextind(line, next_i)
                continue
            end
            in_quotes = !in_quotes
        elseif c == ',' && !in_quotes
            push!(cells, strip(String(take!(buf))))
        else
            print(buf, c)
        end
        i = nextind(line, i)
    end

    in_quotes && error("Unclosed quoted field in CSV line: $(line)")
    push!(cells, strip(String(take!(buf))))
    return cells
end

function read_csv_rows(csvfile::AbstractString)
    isfile(csvfile) || error("CSV file does not exist: $(csvfile)")

    header = nothing
    rows = Vector{Vector{String}}()

    for raw_line in readlines(csvfile)
        isempty(strip(raw_line)) && continue
        cells = split_csv_line(raw_line)
        if header === nothing
            header = cells
        else
            push!(rows, cells)
        end
    end

    header === nothing && error("CSV file is empty: $(csvfile)")

    col = Dict{String,Int}()
    for (idx, name) in enumerate(header)
        key = strip(name)
        isempty(key) && continue
        haskey(col, key) && error("Duplicate CSV column $(repr(key)) in $(csvfile).")
        col[key] = idx
    end

    return (header = header, col = col, rows = rows)
end

function require_column(col::Dict{String,Int}, names::AbstractVector{String})
    for name in names
        if haskey(col, name)
            return col[name]
        end
    end
    error("Missing required CSV column; expected one of: $(join(names, ", ")).")
end

function optional_column(col::Dict{String,Int}, names::AbstractVector{String})
    for name in names
        if haskey(col, name)
            return col[name]
        end
    end
    return nothing
end

function parse_float_cell(row, idx)
    idx === nothing && return missing
    idx > length(row) && return missing

    s = strip(row[idx])
    isempty(s) && return missing

    value = tryparse(Float64, s)
    value === nothing && return missing
    return value
end

function parse_int_cell(row, idx)
    idx === nothing && return missing
    idx > length(row) && return missing

    s = strip(row[idx])
    isempty(s) && return missing

    value = tryparse(Int, s)
    value !== nothing && return value

    float_value = tryparse(Float64, s)
    if float_value !== nothing && isfinite(float_value) && isinteger(float_value)
        return Int(float_value)
    end

    return missing
end

function parse_string_cell(row, idx)
    idx === nothing && return ""
    idx > length(row) && return ""
    return strip(row[idx])
end

function finite_positive(value)
    return value !== missing && isfinite(value) && value > 0.0
end

# ============================================================
# Data loading
# ============================================================

function valid_point(N, err)
    return N !== missing && finite_positive(err)
end

function load_cheb_leg_errors(csvfile::AbstractString)
    csv = read_csv_rows(csvfile)

    n_col = require_column(csv.col, ["N"])
    cheb_err_col = require_column(csv.col, ["cheb_err_inf"])
    leg_err_col = require_column(csv.col, ["leg_err_inf"])
    cheb_status_col = require_column(csv.col, ["cheb_status"])
    leg_status_col = require_column(csv.col, ["leg_status"])

    optional_column(csv.col, ["cheb_error_message"])
    optional_column(csv.col, ["leg_error_message"])

    cheb_points = NamedTuple[]
    leg_points = NamedTuple[]

    for row in csv.rows
        N = parse_int_cell(row, n_col)

        cheb_status = parse_string_cell(row, cheb_status_col)
        cheb_err = parse_float_cell(row, cheb_err_col)
        if cheb_status == "success" && valid_point(N, cheb_err)
            push!(cheb_points, (N = Int(N), err = Float64(cheb_err)))
        end

        leg_status = parse_string_cell(row, leg_status_col)
        leg_err = parse_float_cell(row, leg_err_col)
        if leg_status == "success" && valid_point(N, leg_err)
            push!(leg_points, (N = Int(N), err = Float64(leg_err)))
        end
    end

    sort!(cheb_points; by = p -> p.N)
    sort!(leg_points; by = p -> p.N)
    return cheb_points, leg_points
end

function load_gutleb_errors(csvfile::AbstractString)
    csv = read_csv_rows(csvfile)

    n_col = require_column(csv.col, ["N"])
    err_col = require_column(csv.col, ["err_inf"])
    status_col = require_column(csv.col, ["status"])

    optional_column(csv.col, ["error_message"])

    points = NamedTuple[]
    for row in csv.rows
        N = parse_int_cell(row, n_col)
        status = parse_string_cell(row, status_col)
        err = parse_float_cell(row, err_col)

        if status == "success" && valid_point(N, err)
            push!(points, (N = Int(N), err = Float64(err)))
        end
    end

    sort!(points; by = p -> p.N)
    return points
end

# ============================================================
# Plotting
# ============================================================

function require_nonempty(points, label::AbstractString)
    isempty(points) && error("No valid data points for $(label).")
    return points
end

function point_vectors(points)
    return [p.N for p in points], [p.err for p in points]
end

function savefig_pdf_png(fig, pdf_path::AbstractString, png_path::AbstractString)
    savefig(fig, pdf_path)
    savefig(fig, png_path)
    return pdf_path, png_path
end

function plot_accuracy_compare(cheb_points, leg_points, gutleb_points)
    require_nonempty(cheb_points, "Chebyshev--Duffy")
    require_nonempty(leg_points, "Legendre--Duffy")
    require_nonempty(gutleb_points, "Gutleb--Olver")

    cheb_N, cheb_err = point_vectors(cheb_points)
    leg_N, leg_err = point_vectors(leg_points)
    gutleb_N, gutleb_err = point_vectors(gutleb_points)

    plt = plot(
        cheb_N,
        cheb_err;
        label = "Chebyshev-Duffy",
        marker = false,
        linewidth = 2.0,
        xlabel = "N",
        ylabel = "error",
        yscale = :log10,
        framestyle = :box,
        grid = false,
        dpi = 300,
        legend = :topright,
        title = "",
        ylims = (1e-15, 1.0)
    )

    plot!(
        plt,
        leg_N,
        leg_err;
        label = "Legendre-Duffy",
        marker = false,
        linewidth = 2.0,
    )

    plot!(
        plt,
        gutleb_N,
        gutleb_err;
        label = "Gutleb-Olver",
        marker = false,
        linewidth = 2.0,
    )

    return plt
end

# ============================================================
# Run
# ============================================================

function main()
    @printf("Using Chebyshev/Legendre error CSV: %s\n", CHEB_LEG_CSV)
    @printf("Using Gutleb error CSV: %s\n", GUTLEB_CSV)

    mkpath(FIGURES_DIR)

    cheb_points, leg_points = load_cheb_leg_errors(CHEB_LEG_CSV)
    gutleb_points = load_gutleb_errors(GUTLEB_CSV)

    plt = plot_accuracy_compare(cheb_points, leg_points, gutleb_points)
    savefig_pdf_png(plt, PDF_PATH, PNG_PATH)

    @printf("PDF saved to: %s\n", PDF_PATH)
    @printf("PNG saved to: %s\n", PNG_PATH)
    println("Finished VIDE accuracy comparison plot.")
end

main()
