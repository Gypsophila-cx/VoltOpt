# spy_VoltMat.jl
#
# Construct sparsity and magnitude data for Volterra coefficient matrices.
# This script compares Chebyshev, Legendre, and Jacobi bases for the same
# Volterra kernel and truncation settings used by the legacy Julia spy plot.
# Julia only saves a MATLAB .mat data file; plotting is handled by
# spy_VoltMat.m.
#
# Recommended run from the repository root:
#   julia --project=. test/sparse_spy/spy_VoltMat.jl

using Printf
using SparseArrays

const scriptDir = @__DIR__
const projectRoot = dirname(dirname(scriptDir))
const dataDir = joinpath(scriptDir, "data")
mkpath(dataDir)

include(joinpath(projectRoot, "src", "VoltOpt.jl"))
using .VoltOpt

const Nplot = 200
const Nmat = 300
const jacobiAlpha = 0.3
const jacobiBeta = 0.2
const chebLegTol = 1.0e-20
const outputPath = joinpath(dataDir, "spy_VoltMat.mat")

K(x, y) = exp(-x * y)

extract_matrix(B_raw) = B_raw isa Tuple ? B_raw[1] : B_raw

function matrix_nnz(A::SparseMatrixCSC)
    return nnz(A)
end

function matrix_nnz(A::AbstractMatrix)
    return count(!iszero, A)
end

function nonzero_abs_values(A::SparseMatrixCSC)
    return abs.(nonzeros(A))
end

function nonzero_abs_values(A::AbstractMatrix)
    vals = Float64[]
    sizehint!(vals, matrix_nnz(A))

    for a in A
        if !iszero(a)
            push!(vals, abs(Float64(a)))
        end
    end

    return vals
end

function print_matrix_summary(label::AbstractString, A::AbstractMatrix, path::AbstractString)
    nzabs = nonzero_abs_values(A)
    max_abs = isempty(nzabs) ? 0.0 : maximum(nzabs)
    positive = nzabs[nzabs .> 0.0]
    min_positive_abs = isempty(positive) ? 0.0 : minimum(positive)

    println("------------------------------------------------------------")
    println(label)
    @printf("matrix size       = %d x %d\n", size(A, 1), size(A, 2))
    @printf("nnz               = %d\n", matrix_nnz(A))
    @printf("max abs           = %.17e\n", max_abs)
    @printf("min positive abs  = %.17e\n", min_positive_abs)
    println("output path       = ", path)

    return nothing
end

# Minimal MATLAB Level-4 writer for real Float64 matrices and ASCII text.
# This avoids adding MAT.jl to the project. Matrices are stored dense because
# the chosen N is small enough for this diagnostic data file.
function write_mat_v4_matrix(io, name::AbstractString, A)
    A64 = Matrix{Float64}(A)
    name_bytes = Vector{UInt8}(codeunits(name))

    write(io, Int32(0))                  # MOPT: little-endian IEEE, Float64, full matrix
    write(io, Int32(size(A64, 1)))       # mrows
    write(io, Int32(size(A64, 2)))       # ncols
    write(io, Int32(0))                  # imagf = 0
    write(io, Int32(length(name_bytes) + 1))
    write(io, name_bytes)
    write(io, UInt8(0))
    write(io, vec(A64))                  # Julia and MATLAB are both column-major

    return nothing
end

function write_mat_v4_text(io, name::AbstractString, text::AbstractString)
    name_bytes = Vector{UInt8}(codeunits(name))
    text_bytes = Vector{UInt8}(codeunits(text))

    write(io, Int32(51))                 # MOPT: little-endian IEEE, UInt8, text matrix
    write(io, Int32(1))                  # mrows
    write(io, Int32(length(text_bytes))) # ncols
    write(io, Int32(0))                  # imagf = 0
    write(io, Int32(length(name_bytes) + 1))
    write(io, name_bytes)
    write(io, UInt8(0))
    write(io, text_bytes)

    return nothing
end

function write_mat_v4_var(io, name::AbstractString, value)
    if value isa AbstractString
        write_mat_v4_text(io, name, value)
    else
        write_mat_v4_matrix(io, name, value)
    end

    return nothing
end

function write_mat_v4(path::AbstractString, vars::Pair...)
    open(path, "w") do io
        for var in vars
            write_mat_v4_var(io, String(var.first), var.second)
        end
    end

    return path
end

function scalar_mat(x)
    return reshape([Float64(x)], 1, 1)
end

println("============================================================")
println("Volterra matrix sparsity data")
println("Kernel: K(x, y) = exp(-x*y)")
println("Nmat = ", Nmat)
println("Nplot metadata = ", Nplot)
println("Data directory: ", dataDir)
println("============================================================")

println("Constructing Chebyshev Volterra matrix...")
A_cheb = extract_matrix(VoltOpt.voltmat_cheb(K, Nmat; tol=chebLegTol))

println("Constructing Legendre Volterra matrix...")
A_leg = extract_matrix(VoltOpt.voltmat_leg(K, Nmat; tol=chebLegTol))

println("Constructing Jacobi Volterra matrix...")
A_jac = extract_matrix(VoltOpt.voltmat_jacobi(K, Nmat, jacobiAlpha, jacobiBeta))

print_matrix_summary("Chebyshev", A_cheb, outputPath)
print_matrix_summary("Legendre", A_leg, outputPath)
print_matrix_summary("Jacobi($(jacobiAlpha), $(jacobiBeta))", A_jac, outputPath)

write_mat_v4(
    outputPath,
    "A_cheb" => A_cheb,
    "A_leg" => A_leg,
    "A_jac" => A_jac,
    "N" => scalar_mat(Nmat),
    "Nplot" => scalar_mat(Nplot),
    "jacobi_alpha" => scalar_mat(jacobiAlpha),
    "jacobi_beta" => scalar_mat(jacobiBeta),
    "cheb_leg_tol" => scalar_mat(chebLegTol),
    "kernel_name" => "K(x,y) = exp(-x*y)",
    "basis_names" => "Chebyshev; Legendre; Jacobi(0.3,0.2)",
    "description" => "Volterra coefficient matrices for colored spy plots. Julia saved data only; use test/sparse_spy/spy_VoltMat.m for MATLAB plotting.",
)

println("============================================================")
println("Saved MAT data:")
println("  ", outputPath)
println("MAT variables:")
println("  A_cheb, A_leg, A_jac, N, Nplot, jacobi_alpha, jacobi_beta,")
println("  cheb_leg_tol, kernel_name, basis_names, description")
println("============================================================")
