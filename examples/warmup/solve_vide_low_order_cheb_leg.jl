ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

using LinearAlgebra
using SparseArrays
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
const FIGURES_DIR = joinpath(PROJECT_ROOT, "examples", "figures")
const DATA_DIR = joinpath(PROJECT_ROOT, "examples", "data")

mkpath(FIGURES_DIR)
mkpath(DATA_DIR)

include(joinpath(PROJECT_ROOT, "src", "VoltOpt.jl"))
using .VoltOpt

const CSV_PATH = joinpath(DATA_DIR, "vide_low_order_cheb_leg_errors.csv")
const PDF_PATH = joinpath(FIGURES_DIR, "vide_low_order_cheb_leg_errors.pdf")
const PNG_PATH = joinpath(FIGURES_DIR, "vide_low_order_cheb_leg_errors.png")

# ============================================================
# Problem definition
# ============================================================

K(x, t) = ((x + 1) * (x + 2) / 4) * exp(((t + 1) * (x - t)) / 4)
rhs_fun(x) = x + 2
uexact(x) = exp(((x + 1)^2) / 4)

# If U(x)=exp((x+1)^2/4), then the transformed equation is satisfied exactly.

const DEFAULT_TOL = eps()
const DEFAULT_NS = collect(1:100)

# ============================================================
# Ultraspherical operators
# ============================================================

function cheb_derivative_to_c1(N::Integer)
    Nint = Int(N)
    D = spzeros(Float64, Nint, Nint)
    for k in 1:(Nint - 1)
        D[k, k + 1] = k
    end
    return D
end

function cheb_to_c1_conversion(N::Integer)
    Nint = Int(N)
    S = spzeros(Float64, Nint, Nint)

    if Nint >= 1
        S[1, 1] = 1.0
    end
    if Nint >= 2
        S[2, 2] = 0.5
    end
    for j in 3:Nint
        S[j, j] += 0.5
        S[j - 2, j] -= 0.5
    end

    return S
end

function leg_derivative_to_c32(N::Integer)
    Nint = Int(N)
    D = spzeros(Float64, Nint, Nint)
    for k in 1:(Nint - 1)
        D[k, k + 1] = 1.0
    end
    return D
end

function leg_to_c32_conversion(N::Integer)
    Nint = Int(N)
    S = spzeros(Float64, Nint, Nint)

    if Nint >= 1
        S[1, 1] = 1.0
    end
    for j in 2:Nint
        deg = j - 1
        c = 1.0 / (2deg + 1)
        S[j, j] += c
        if j >= 3
            S[j - 2, j] -= c
        end
    end

    return S
end

function rhs_coefficients(N::Integer)
    Nint = Int(N)
    f = zeros(Float64, Nint)
    Nint >= 1 && (f[1] = 2.0)
    Nint >= 2 && (f[2] = 1.0)
    return f
end

# ============================================================
# Series evaluation
# ============================================================

function eval_cheb_series(a::AbstractVector, x::Real)
    n = length(a)
    n == 0 && return zero(eltype(a))
    n == 1 && return a[1]

    b1 = zero(eltype(a))
    b2 = zero(eltype(a))
    xf = Float64(x)

    for k in n:-1:2
        b0 = 2.0 * xf * b1 - b2 + a[k]
        b2 = b1
        b1 = b0
    end

    return a[1] + xf * b1 - b2
end

function eval_leg_series(a::AbstractVector, x::Real)
    n = length(a)
    n == 0 && return zero(eltype(a))

    xf = Float64(x)
    acc = a[1]
    n == 1 && return acc

    pkm1 = 1.0
    pk = xf
    acc += a[2] * pk

    for k in 1:(n - 2)
        pkp1 = ((2k + 1) * xf * pk - k * pkm1) / (k + 1)
        acc += a[k + 2] * pkp1
        pkm1, pk = pk, pkp1
    end

    return acc
end

# ============================================================
# Solvers
# ============================================================

function solve_cheb_tau(N::Integer; tol::Float64 = DEFAULT_TOL)
    Nint = Int(N)
    Nint >= 2 || throw(ArgumentError("N must be at least 2."))

    # VoltOpt constructors use the maximum polynomial degree; N coefficients means degree N-1.
    V = sparse(VoltOpt.voltmat_cheb(K, Nint - 1; tol = tol))
    size(V) == (Nint, Nint) ||
        error("voltmat_cheb returned size $(size(V)); expected $((Nint, Nint)).")

    D = cheb_derivative_to_c1(Nint)
    S = cheb_to_c1_conversion(Nint)
    f = rhs_coefficients(Nint)

    A = Matrix(2.0 * D + S - S * V)
    rhs = Vector(S * f)

    for j in 1:Nint
        A[end, j] = (-1.0)^(j - 1)
    end
    rhs[end] = 1.0

    return qr(A) \ rhs
end

function solve_leg_tau(N::Integer; tol::Float64 = DEFAULT_TOL)
    Nint = Int(N)
    Nint >= 2 || throw(ArgumentError("N must be at least 2."))

    # VoltOpt constructors use the maximum polynomial degree; N coefficients means degree N-1.
    V = sparse(VoltOpt.voltmat_leg(K, Nint - 1; tol = tol))
    size(V) == (Nint, Nint) ||
        error("voltmat_leg returned size $(size(V)); expected $((Nint, Nint)).")

    D = leg_derivative_to_c32(Nint)
    S = leg_to_c32_conversion(Nint)
    f = rhs_coefficients(Nint)

    A = Matrix(2.0 * D + S - S * V)
    rhs = Vector(S * f)

    for j in 1:Nint
        A[end, j] = (-1.0)^(j - 1)
    end
    rhs[end] = 1.0

    return qr(A) \ rhs
end

# ============================================================
# Error tests
# ============================================================

function max_error_cheb(a::AbstractVector; ngrid::Integer = 2001)
    xgrid = range(-1.0, 1.0; length = Int(ngrid))
    return maximum(x -> abs(eval_cheb_series(a, x) - uexact(x)), xgrid)
end

function max_error_leg(a::AbstractVector; ngrid::Integer = 2001)
    xgrid = range(-1.0, 1.0; length = Int(ngrid))
    return maximum(x -> abs(eval_leg_series(a, x) - uexact(x)), xgrid)
end

function success_result(err::Real)
    return (err = Float64(err), status = "success", error_message = "")
end

function failed_result(err)
    return (err = NaN, status = "failed", error_message = sprint(showerror, err))
end

function run_one_basis(N::Integer, basis::Symbol; tol::Float64 = DEFAULT_TOL)
    try
        coeffs = if basis === :cheb
            solve_cheb_tau(N; tol = tol)
        elseif basis === :leg
            solve_leg_tau(N; tol = tol)
        else
            throw(ArgumentError("basis must be :cheb or :leg."))
        end

        err = basis === :cheb ? max_error_cheb(coeffs) : max_error_leg(coeffs)
        return success_result(err)
    catch err
        return failed_result(err)
    end
end

function run_error_tests(Ns::AbstractVector{<:Integer}; tol::Float64 = DEFAULT_TOL)
    rows = NamedTuple[]

    for N in Ns
        @printf("N = %d\n", N)

        cheb = run_one_basis(N, :cheb; tol = tol)
        leg = run_one_basis(N, :leg; tol = tol)

        push!(
            rows,
            (
                N = Int(N),
                cheb_err_inf = cheb.err,
                leg_err_inf = leg.err,
                cheb_status = cheb.status,
                leg_status = leg.status,
                cheb_error_message = cheb.error_message,
                leg_error_message = leg.error_message,
            ),
        )

        cheb_msg = cheb.status == "success" ? @sprintf("%.3e", cheb.err) : cheb.error_message
        leg_msg = leg.status == "success" ? @sprintf("%.3e", leg.err) : leg.error_message
        println("  Chebyshev: ", cheb.status, ", error = ", cheb_msg)
        println("  Legendre:  ", leg.status, ", error = ", leg_msg)
    end

    return rows
end

# ============================================================
# Plotting
# ============================================================

const CSV_HEADERS = [
    "N",
    "cheb_err_inf",
    "leg_err_inf",
    "cheb_status",
    "leg_status",
    "cheb_error_message",
    "leg_error_message",
]

function csv_escape(x)
    if x isa AbstractFloat && isnan(x)
        return ""
    end

    s = string(x)
    if occursin(",", s) || occursin("\"", s) || occursin("\n", s) || occursin("\r", s)
        return "\"" * replace(s, "\"" => "\"\"") * "\""
    end
    return s
end

function write_error_csv(path::AbstractString, rows::Vector{<:NamedTuple})
    open(path, "w") do io
        println(io, join(CSV_HEADERS, ","))
        for row in rows
            println(io, join((csv_escape(getproperty(row, Symbol(h))) for h in CSV_HEADERS), ","))
        end
    end
    return path
end

function savefig_pdf_png(fig, pdf_path::AbstractString, png_path::AbstractString)
    savefig(fig, pdf_path)
    savefig(fig, png_path)
    return pdf_path, png_path
end

function plot_errors(rows::Vector{<:NamedTuple}; pdf_path::AbstractString = PDF_PATH, png_path::AbstractString = PNG_PATH)
    cheb_rows = filter(row -> row.cheb_status == "success" && isfinite(row.cheb_err_inf), rows)
    leg_rows = filter(row -> row.leg_status == "success" && isfinite(row.leg_err_inf), rows)

    isempty(cheb_rows) && isempty(leg_rows) &&
        error("No successful error data points are available for plotting.")

    plt = plot(
        xlabel = "truncation dimension N",
        ylabel = "maximum error",
        yscale = :log10,
        framestyle = :box,
        dpi = 300,
        legend = :topright,
        grid = false,
        minorgrid = false,
        title = "",
        ylims = (1e-15, 1.0),
    )

    if !isempty(cheb_rows)
        plot!(
            plt,
            [row.N for row in cheb_rows],
            [row.cheb_err_inf for row in cheb_rows];
            label = "Chebyshev",
            marker = false,
            linewidth = 2,
        )
    end

    if !isempty(leg_rows)
        plot!(
            plt,
            [row.N for row in leg_rows],
            [row.leg_err_inf for row in leg_rows];
            label = "Legendre",
            marker = false,
            linewidth = 2,
        )
    end

    savefig_pdf_png(plt, pdf_path, png_path)
    return plt
end

# ============================================================
# Run
# ============================================================

function main(; Ns::Vector{Int} = DEFAULT_NS, tol::Float64 = DEFAULT_TOL)
    rows = run_error_tests(Ns; tol = tol)
    write_error_csv(CSV_PATH, rows)
    plot_errors(rows; pdf_path = PDF_PATH, png_path = PNG_PATH)

    println("Saved CSV: ", CSV_PATH)
    println("Saved PDF: ", PDF_PATH)
    println("Saved PNG: ", PNG_PATH)

    return rows
end

main()