ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

const THIS_DIR = @__DIR__

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

const PROJECT_ROOT = find_project_root()
const SRC_DIR = joinpath(PROJECT_ROOT, "src")
const DATA_DIR = joinpath(THIS_DIR, "data")
const FIGURES_DIR = joinpath(THIS_DIR, "figures")

include(joinpath(SRC_DIR, "VoltMat", "voltmat_leg.jl"))

using .VoltMat_Leg
using LinearAlgebra
using Plots
using Plots.PlotMeasures
using Printf
using SparseArrays
using SpecialFunctions

const T = ComplexF64

const matrix_n = 64
const interp_n = 30
const interp_m = 30

const nptx = 320
const npty = 300
const ax = (-0.65, 0.65)
const ay = (-0.72, 0.34)

const csv_path = joinpath(DATA_DIR, "ex5_volterra_schur_besselj1_pseudospectrum.csv")
const pdf_path = joinpath(FIGURES_DIR, "ex5_volterra_schur_besselj1.pdf")

function savefig_pdf_png(fig, pdf_path::AbstractString)
    png_path = replace(pdf_path, r"\.pdf$" => ".png")
    savefig(fig, pdf_path)
    savefig(fig, png_path)
    return png_path
end

kernel(x::Real, y::Real) = -im * besselj(1, 2 * (x - y) + x * y)

kernel_real(x::Real, y::Real) = real(kernel(x, y))
kernel_imag(x::Real, y::Real) = imag(kernel(x, y))

function legendre_interp_coeffs_Khat_complex(K, nrow::Integer, mcol::Integer)
    C_real = VoltMat_Leg.legendre_interp_coeffs_Khat(
        (x, y) -> real(K(x, y)),
        nrow,
        mcol,
    )
    C_imag = VoltMat_Leg.legendre_interp_coeffs_Khat(
        (x, y) -> imag(K(x, y)),
        nrow,
        mcol,
    )
    return complex.(C_real, C_imag)
end

function voltmat_leg_complex_from_coeffs(C::AbstractMatrix{<:Complex}, N::Integer; kwargs...)
    B_real, info_real = VoltMat_Leg.voltmat_leg(real.(C), N; return_info=true, kwargs...)
    B_imag, info_imag = VoltMat_Leg.voltmat_leg(imag.(C), N; return_info=true, kwargs...)
    B = complex.(B_real) + im .* complex.(B_imag)
    info = (
        real_info = info_real,
        imag_info = info_imag,
        matrix_size = size(B),
        nnz = nnz(B),
    )
    return B, info
end

function normalized_legendre_matrix(B_leg)
    n = size(B_leg, 1)
    degrees = 0:n-1
    w_left = spdiagm(n, n, sqrt.(2.0 ./ (2 .* degrees .+ 1)))
    w_right = spdiagm(n, n, sqrt.((2 .* degrees .+ 1) ./ 2.0))

    return Matrix(w_left * B_leg * w_right)
end

function schur_min_singular_grid(schur_T::AbstractMatrix{T}, ptx, pty) where {T<:Complex}
    n = size(schur_T, 1)
    I_n = Matrix{T}(I, n, n)
    pse = zeros(real(T), length(pty), length(ptx))

    @inbounds for i in eachindex(ptx)
        println("grid column $(i)/$(length(ptx))")

        for j in eachindex(pty)
            z = T(ptx[i] + im * pty[j])
            pse[j, i] = svdvals(schur_T - z * I_n)[end]
        end
    end

    return pse
end

function write_pseudospectrum_csv(path, ptx, pty, pse, logpse)
    open(path, "w") do io
        println(io, "x,y,pse,log10_pse")

        @inbounds for i in eachindex(ptx)
            for j in eachindex(pty)
                @printf(io, "%.17g,%.17g,%.17g,%.17g\n", ptx[i], pty[j], pse[j, i], logpse[j, i])
            end
        end
    end
end

function plot_pdf(path, ptx, pty, logpse)
    levels = -13:0.5:-0.25
    colorbar_ticks = -12.5:2.5:-2.5
    colorbar_y = collect(range(first(levels), last(levels), length = 512))
    colorbar_z = repeat(colorbar_y, 1, 2)

    p_main = contour(
        ptx,
        pty,
        logpse;
        levels = levels,
        color = cgrad(:turbo),
        clims = (first(levels), last(levels)),
        colorbar = false,
        linewidth = 0.9,
        aspect_ratio = :equal,
        grid = false,
        xlabel = "",
        ylabel = "",
        title = "",
        framestyle = :box,
        right_margin = 1mm,
    )

    p_bar = heatmap(
        [0.0, 1.0],
        colorbar_y,
        colorbar_z;
        color = cgrad(:turbo),
        clims = (first(levels), last(levels)),
        colorbar = false,
        xaxis = false,
        xticks = false,
        yticks = colorbar_ticks,
        ymirror = true,
        ytickfontsize = 8,
        grid = false,
        xlabel = "",
        ylabel = "",
        title = "",
        framestyle = :box,
        left_margin = 1mm,
        right_margin = 8mm,
    )

    plot_layout = @layout [a{0.95w} b{0.02w}]
    plt = plot(p_main, p_bar; layout = plot_layout, size = (720, 480))
    savefig_pdf_png(plt, path)

    return plt
end

mkpath(DATA_DIR)
mkpath(FIGURES_DIR)

println("interpolating Khat(x,t) in a tensor-product Legendre basis")
interp_time = @elapsed begin
    global C_Khat = legendre_interp_coeffs_Khat_complex(kernel, interp_n, interp_m)
end

println("assembling the Legendre-basis Volterra coefficient matrix")
assembly_time = @elapsed begin
    global B_leg, assembly_info = voltmat_leg_complex_from_coeffs(
        C_Khat,
        matrix_n - 1;
        bandwidth = size(C_Khat, 1),
        dropzeros_tol = 1.0e-13,
        return_info = true,
    )
    global A_volterra = normalized_legendre_matrix(B_leg)
    global norm_scale = opnorm(A_volterra)
    A_volterra ./= norm_scale
end

println("forming the Schur triangular matrix")
schur_time = @elapsed begin
    global schur_factor = schur(A_volterra)
    global schur_T = Matrix{T}(schur_factor.T)
end

ptx = Vector(range(ax[1], ax[2], nptx))
pty = Vector(range(ay[1], ay[2], npty))

println("computing the pseudospectrum by shifted Schur triangular SVDs")
pse_time = @elapsed begin
    global pse = schur_min_singular_grid(schur_T, ptx, pty)
end

logpse = log10.(pse)

println("writing CSV data")
csv_time = @elapsed write_pseudospectrum_csv(csv_path, ptx, pty, pse, logpse)

println("writing PDF/PNG figure")
plot_time = @elapsed plot_pdf(pdf_path, ptx, pty, logpse)

println("interp time (s): ", interp_time)
println("assembly time (s): ", assembly_time)
println("Schur time (s): ", schur_time)
println("pseudospectrum time (s): ", pse_time)
println("CSV time (s): ", csv_time)
println("plot time (s): ", plot_time)
println("Volterra matrix size: ", size(B_leg), ", nnz: ", assembly_info.nnz)
println("Khat coefficient size: ", size(C_Khat))
println("normalization scale: ", norm_scale)
println("saved CSV: ", csv_path)
println("saved PDF: ", pdf_path)
println("saved PNG: ", replace(pdf_path, r"\.pdf$" => ".png"))
