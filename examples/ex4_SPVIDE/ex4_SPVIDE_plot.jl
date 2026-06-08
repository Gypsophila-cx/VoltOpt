using DelimitedFiles
using FFTW
using Printf
using Plots

function find_project_root(start_dir::AbstractString=@__DIR__)
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

const VOLTERRA_ROOT = find_project_root()
const FIGURES_DIR = joinpath(VOLTERRA_ROOT, "examples", "figures")
const DATA_DIR = joinpath(VOLTERRA_ROOT, "examples", "data")

const EPS = 1.0e-7
const MU = -1.0
const N_FINAL = 32_000

function spvide_output_tag()
    tag = @sprintf("eps%.0e_mu%.0f", EPS, MU)
    return replace(tag, "+" => "", "-" => "minus")
end

function savefig_pdf_png(fig, pdf_path::AbstractString)
    png_path = replace(pdf_path, r"\.pdf$" => ".png")
    savefig(fig, pdf_path)
    savefig(fig, png_path)
    return png_path
end

function require_column(headers::AbstractVector{<:AbstractString}, name::AbstractString)
    idx = findfirst(==(name), headers)
    idx === nothing && error("Missing required CSV column: $(name).")
    return idx
end

function read_numeric_csv(filename::AbstractString)
    isfile(filename) || error("CSV file does not exist: $(filename). Run examples/ex4_SPVIDE/ex4_SPVIDE.jl first.")

    data, header = readdlm(filename, ',', Float64; header=true)
    data = data isa AbstractVector ? reshape(data, 1, :) : data
    headers = vec(String.(header))

    return data, headers
end

function read_error_history_csv(filename::AbstractString)
    data, headers = read_numeric_csv(filename)

    n_col = require_column(headers, "N")
    true_col = require_column(headers, "true_error_inf")
    cauchy_col = require_column(headers, "cauchy_coeff_error_inf")

    Ns = Int.(data[:, n_col])
    true_errors = Float64.(data[:, true_col])
    cauchy_errors = Float64.(data[:, cauchy_col])

    return Ns, true_errors, cauchy_errors
end

function read_coefficients_csv(filename::AbstractString)
    data, headers = read_numeric_csv(filename)
    coeff_col = require_column(headers, "c_n")
    return Float64.(data[:, coeff_col])
end

function cheb_lobatto_nodes(M::Int)
    return cos.(pi .* (0:M) ./ M)
end

function cheb_values_on_lobatto_grid(c::AbstractVector{<:Real}, M::Int)
    N = length(c) - 1
    @assert M >= N

    cp = zeros(Float64, M + 1)
    cp[1:N + 1] .= c

    tmp = copy(cp)
    tmp[1] *= 2.0
    tmp[end] *= 2.0

    vals = FFTW.r2r(tmp, FFTW.REDFT00) ./ 2.0
    return vals
end

@inline function v_exact_t(t)
    return sin(t) + exp(-t / EPS)
end

function plot_solution_default(
    c::AbstractVector{<:Real};
    filename=joinpath(FIGURES_DIR, "ex4_SPVIDE_solution.pdf"),
)
    N = length(c) - 1
    M = max(100_000, 5N)

    x_desc = cheb_lobatto_nodes(M)
    u_desc = cheb_values_on_lobatto_grid(c, M)

    t_desc = 0.5 .* (x_desc .+ 1.0)
    v_exact_desc = v_exact_t.(t_desc)

    t_plot = reverse(t_desc)
    v_num_plot = reverse(u_desc)
    v_exact_plot = reverse(v_exact_desc)

    p = plot(
        t_plot,
        v_num_plot;
        lw=2,
        label="numerical",
        xlabel="t",
        ylabel="v(t)",
        framestyle=:box,
        legend=:bottomright,
        grid=false,
        minorgrid=false,
    )

    plot!(
        p,
        t_plot,
        v_exact_plot;
        lw=1.5,
        ls=:dash,
        label="exact",
    )

    savefig_pdf_png(p, filename)
    return p
end

function dense_log10_ticks(vmin, vmax; minor=true)
    @assert vmin > 0
    @assert vmax > 0

    kmin = floor(Int, log10(vmin))
    kmax = ceil(Int, log10(vmax))

    vals = Float64[]
    labels = String[]

    for k in kmin:2:kmax+2
        push!(vals, 10.0^k)
        push!(labels, "10^{$k}")

        if minor && k < kmax
            for a in 2:9
                push!(vals, a * 10.0^k)
                push!(labels, "")
            end
        end
    end

    keep = (vals .>= vmin) .& (vals .<= vmax)
    return vals[keep], labels[keep]
end

function plot_true_error_default(
    Ns,
    true_errors;
    filename=joinpath(FIGURES_DIR, "ex4_SPVIDE_inferror.pdf"),
)
    ymin = minimum(true_errors)
    ymax = maximum(true_errors)

    ymin_plot = 10.0^(floor(log10(ymin)) - 0.2)
    ymax_plot = 10.0^(ceil(log10(ymax)) + 0.2)

    yt = dense_log10_ticks(ymin_plot, ymax_plot; minor=false)

    p = plot(
        Ns,
        true_errors;
        yscale=:log10,
        yticks=yt,
        framestyle=:box,
        grid=false,
        marker=false,
        lw=2,
        xlabel="N",
        ylabel="||u_N - u_exact||_inf",
        label="true error",
        legend=:topright,
        color=:black,
    )

    savefig_pdf_png(p, filename)
    return p
end

function fixed_power_ticks(kmin::Int, kmax::Int)
    vals = [10.0^k for k in kmin:2:kmax]
    labels = ["10^{$k}" for k in kmin:2:kmax]
    return vals, labels
end

function plot_cauchy_error_default(
    Ns,
    cauchy_errors;
    filename=joinpath(FIGURES_DIR, "ex4_SPVIDE_cauchy_error.pdf"),
)
    yt = fixed_power_ticks(-16, -1)

    p = plot(
        Ns,
        cauchy_errors;
        yscale=:log10,
        ylim=(1e-16, 1e-1),
        #yticks=yt,
        framestyle=:box,
        grid=false,
        marker=false,
        lw=2,
        xlabel="N",
        ylabel="Cauchy Error",
        legend=false,
        color=:black,
    )

    savefig_pdf_png(p, filename)
    return p
end

function main()
    tag = spvide_output_tag()

    history_csv = joinpath(DATA_DIR, "ex4_SPVIDE_error_history_$(tag).csv")
    coeff_csv = joinpath(DATA_DIR, "ex4_SPVIDE_coefficients_N$(N_FINAL)_$(tag).csv")

    solution_pdf = joinpath(FIGURES_DIR, "ex4_SPVIDE_solution_$(tag).pdf")
    error_pdf = joinpath(FIGURES_DIR, "ex4_SPVIDE_error_$(tag).pdf")
    cauchy_pdf = joinpath(FIGURES_DIR, "ex4_SPVIDE_cauchy_$(tag).pdf")

    Ns, true_errors, cauchy_errors = read_error_history_csv(history_csv)
    c_final = read_coefficients_csv(coeff_csv)

    mkpath(FIGURES_DIR)

    plot_solution_default(
        c_final;
        filename=solution_pdf,
    )

    plot_true_error_default(
        Ns,
        true_errors;
        filename=error_pdf,
    )

    plot_cauchy_error_default(
        Ns,
        cauchy_errors;
        filename=cauchy_pdf,
    )

    @printf("\nDone.\n")
    @printf("Saved figures:\n")
    @printf("  %s\n", solution_pdf)
    @printf("  %s\n", replace(solution_pdf, r"\.pdf$" => ".png"))
    @printf("  %s\n", error_pdf)
    @printf("  %s\n", replace(error_pdf, r"\.pdf$" => ".png"))
    @printf("  %s\n", cauchy_pdf)
    @printf("  %s\n", replace(cauchy_pdf, r"\.pdf$" => ".png"))
end

main()
