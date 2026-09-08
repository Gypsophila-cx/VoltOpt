# ex7_condition_numbers.jl
#
# Compare the Float64 Euclidean 2-norm condition numbers of the actual linear
# systems assembled by the current Example 7 Weak Duffy--CFP (Weak VoltMat)
# and FracCollocation drivers.
#
# The two matrices are expressed in each method's natural discrete coordinates.
# Matrix condition numbers depend on the basis, nodes, coordinate map, and
# scaling, so these results measure the numerical sensitivity of the algebraic
# systems used by the current implementations.  They are not basis-independent
# condition numbers of the continuous Volterra integral operator.
#
# Run from the VoltOpt project root:
#   julia --project=. examples/ex7_weaksingular/ex7_condition_numbers.jl
#
# Optionally override the default degrees with a comma-separated list:
#   EX7_COND_NS=10,20,40,80,160,320,640,1000
#
# Output:
#   examples/ex7_weaksingular/data/ex7_condition_numbers.csv

using LinearAlgebra
using Printf
using SparseArrays

const SCRIPT_DIR = @__DIR__
const PROJECT_ROOT = normpath(joinpath(SCRIPT_DIR, "..", ".."))

include(joinpath(PROJECT_ROOT, "src", "VoltOpt.jl"))
using .VoltOpt

include(joinpath(
    PROJECT_ROOT,
    "others",
    "FracCollocation",
    "src",
    "FracCollocation.jl",
))
import .FracCollocation

const DEFAULT_NS = [10, 20, 40, 80, 160, 320, 480, 640]
const KERNEL_I = 80
const KERNEL_J = 64

function kernel_xy(x::Real, y::Real)
    xf = Float64(x)
    yf = Float64(y)
    return 1.0 / ((xf + 1.0)^2 + (yf - 1.0)^2)
end

function mapped_kernel(t::Real, s::Real)
    # Match ex7_FracCollocation.jl: evaluate directly in [0,1] coordinates
    # instead of forming x=2t-1 and then recovering the small value x+1.
    tf = Float64(t)
    sf = Float64(s)
    return 1.0 / ((2.0 * tf)^2 + (2.0 * sf - 2.0)^2)
end

function mapped_rhs(t::Real)
    # build_system also forms its RHS even though condition numbers do not use
    # it.  Keep exactly the stable mapped RHS from ex7_FracCollocation.jl.
    tf = Float64(t)
    x = 2.0 * tf - 1.0
    return tf^(1.0 / 3.0) *
           (1.0 + sin(pi * x) / (1000.0 * x^2 + 1.0))
end

function make_frac_problem(
    mu::Float64,
    muntz_lambda::Float64,
    muntz_jacobi_a::Float64,
    muntz_jacobi_b::Float64,
)
    return FracCollocation.FracVIEProblem(
        mu,
        muntz_lambda,
        muntz_jacobi_a,
        muntz_jacobi_b,
        mapped_kernel,
        mapped_rhs,
        -0.5 * 2.0^(1.0 - mu),
    )
end

function build_kernel_coefficients(beta::Float64)
    values = VoltOpt.Weak_VoltMat_Cheb.sample_duffy_kernel(
        kernel_xy,
        KERNEL_I,
        KERNEL_J,
        beta,
    )
    return VoltOpt.Weak_VoltMat_Cheb.cheb_coeffs_lobatto_2d(values)
end

function packed_to_sparse(A::VoltOpt.PackedLowerHessenberg)
    n = A.n
    nnz_band = A.colptr[end] - 1
    rows = Vector{Int}(undef, nnz_band)
    cols = Vector{Int}(undef, nnz_band)
    vals = Vector{Float64}(undef, nnz_band)

    p = 1
    @inbounds for col in 1:n
        len = min(n, col + A.lower_bw)
        src = A.colptr[col]
        for row in 1:len
            rows[p] = row
            cols[p] = col
            vals[p] = A.values[src + row - 1]
            p += 1
        end
    end

    S = sparse(rows, cols, vals, n, n)
    dropzeros!(S)
    return S
end

function operator_to_sparse(V, n::Int)
    if V isa VoltOpt.PackedLowerHessenberg
        return packed_to_sparse(V)
    elseif V isa SparseMatrixCSC
        S = copy(V)
        dropzeros!(S)
        return S
    elseif V isa AbstractMatrix
        size(V) == (n, n) ||
            throw(DimensionMismatch("operator matrix has wrong size"))
        S = sparse(V)
        dropzeros!(S)
        return S
    else
        D = VoltOpt.packed_to_dense(V)
        S = sparse(D)
        dropzeros!(S)
        return S
    end
end

function system_matrix(VK::SparseMatrixCSC)
    n = size(VK, 1)
    size(VK, 2) == n ||
        throw(DimensionMismatch("operator matrix must be square"))
    A = 0.5 .* copy(VK)
    @inbounds for i in 1:n
        A[i, i] += 1.0
    end
    dropzeros!(A)
    return A
end

function build_weak_system_matrix(
    N::Int,
    kernel_coeffs::AbstractMatrix,
    mu::Float64,
    alpha::Float64,
    beta::Float64,
)
    Vraw, _info = VoltOpt.weak_voltmat_cheb(
        kernel_coeffs;
        C_is_coeffs = true,
        mu = mu,
        sigma = alpha,
        beta = beta,
        N = N,
        diagnostics = false,
        output_format = :packed_lower_hessenberg,
        return_info = true,
    )
    VK = operator_to_sparse(Vraw, N + 1)

    # Preserve the physical endpoint-weight normalization used by the current
    # ex7_VoltMat.jl driver before forming I + 0.5*V_K.
    VK .*= 2.0^alpha
    return system_matrix(VK)
end

function parse_ns()
    raw = get(ENV, "EX7_COND_NS", "")
    isempty(strip(raw)) && return copy(DEFAULT_NS)

    parts = strip.(split(raw, ','; keepempty = true))
    any(isempty, parts) &&
        throw(ArgumentError("EX7_COND_NS must be a comma-separated list of integers"))

    Ns = parse.(Int, parts)
    all(N -> N >= 0, Ns) ||
        throw(ArgumentError("all EX7_COND_NS values must be nonnegative"))
    length(unique(Ns)) == length(Ns) ||
        throw(ArgumentError("EX7_COND_NS must not contain duplicate values"))
    return Ns
end

function count_actual_nonzeros(A::SparseMatrixCSC)
    return nnz(A)
end

function count_actual_nonzeros(A::AbstractMatrix)
    return count(x -> !iszero(x), A)
end

function svd_condition_metrics(A::AbstractMatrix, label::AbstractString)
    dense = Matrix{Float64}(A)
    if !all(isfinite, dense)
        return (
            sigma_max = NaN,
            sigma_min = NaN,
            cond2 = Inf,
            log10_cond2 = Inf,
            status = "$(label)_matrix_nonfinite",
        )
    end

    singular_values = try
        svdvals(dense)
    catch err
        return (
            sigma_max = NaN,
            sigma_min = NaN,
            cond2 = Inf,
            log10_cond2 = Inf,
            status = "$(label)_svd_failed_$(nameof(typeof(err)))",
        )
    end

    sigma_max = maximum(singular_values)
    sigma_min = minimum(singular_values)

    if !isfinite(sigma_max) || sigma_max <= 0.0
        return (
            sigma_max = sigma_max,
            sigma_min = sigma_min,
            cond2 = Inf,
            log10_cond2 = Inf,
            status = "$(label)_sigma_max_invalid",
        )
    elseif !isfinite(sigma_min)
        return (
            sigma_max = sigma_max,
            sigma_min = sigma_min,
            cond2 = Inf,
            log10_cond2 = Inf,
            status = "$(label)_sigma_min_nonfinite",
        )
    elseif sigma_min < 0.0
        return (
            sigma_max = sigma_max,
            sigma_min = sigma_min,
            cond2 = Inf,
            log10_cond2 = Inf,
            status = "$(label)_sigma_min_negative",
        )
    elseif sigma_min == 0.0
        return (
            sigma_max = sigma_max,
            sigma_min = sigma_min,
            cond2 = Inf,
            log10_cond2 = Inf,
            status = "$(label)_sigma_min_zero",
        )
    end

    cond2 = sigma_max / sigma_min
    if !isfinite(cond2)
        return (
            sigma_max = sigma_max,
            sigma_min = sigma_min,
            cond2 = Inf,
            log10_cond2 = Inf,
            status = "$(label)_cond2_overflow",
        )
    end

    return (
        sigma_max = sigma_max,
        sigma_min = sigma_min,
        cond2 = cond2,
        log10_cond2 = log10(cond2),
        status = "success",
    )
end

function matrix_result(N::Int, kernel_coeffs, frac_problem, mu, alpha, beta)
    dimension = N + 1

    A_weak = build_weak_system_matrix(N, kernel_coeffs, mu, alpha, beta)
    frac_system = FracCollocation.build_system(frac_problem, dimension)
    A_frac = frac_system.A

    size(A_weak) == (dimension, dimension) ||
        throw(DimensionMismatch("Weak matrix size does not match N+1 for N=$(N)"))
    size(A_frac) == (dimension, dimension) ||
        throw(DimensionMismatch("Frac matrix size does not match N+1 for N=$(N)"))

    weak = svd_condition_metrics(A_weak, "weak")
    frac = svd_condition_metrics(A_frac, "frac")

    weak_nnz = count_actual_nonzeros(A_weak)
    frac_nnz = count_actual_nonzeros(A_frac)
    weak_density = weak_nnz / length(A_weak)
    frac_density = frac_nnz / length(A_frac)
    ratio = frac.cond2 / weak.cond2

    statuses = String[]
    weak.status == "success" || push!(statuses, weak.status)
    frac.status == "success" || push!(statuses, frac.status)
    if isempty(statuses) && (!isfinite(ratio) || ratio < 0.0)
        push!(statuses, "condition_ratio_nonfinite")
    end
    status = isempty(statuses) ? "success" : join(statuses, ";")

    return (
        N = N,
        dimension = dimension,
        weak_sigma_max = weak.sigma_max,
        weak_sigma_min = weak.sigma_min,
        weak_cond2 = weak.cond2,
        weak_log10_cond2 = weak.log10_cond2,
        weak_nnz = weak_nnz,
        weak_density = weak_density,
        frac_sigma_max = frac.sigma_max,
        frac_sigma_min = frac.sigma_min,
        frac_cond2 = frac.cond2,
        frac_log10_cond2 = frac.log10_cond2,
        frac_nnz = frac_nnz,
        frac_density = frac_density,
        cond2_ratio_frac_over_weak = ratio,
        status = status,
    )
end

function csv_float(value::Real)
    x = Float64(value)
    return isfinite(x) ? @sprintf("%.17e", x) : string(x)
end

function csv_string(value::AbstractString)
    return "\"" * replace(value, "\"" => "\"\"") * "\""
end

function write_csv(path::AbstractString, rows)
    header = (
        "N,dimension,weak_sigma_max,weak_sigma_min,weak_cond2," *
        "weak_log10_cond2,weak_nnz,weak_density,frac_sigma_max," *
        "frac_sigma_min,frac_cond2,frac_log10_cond2,frac_nnz," *
        "frac_density,cond2_ratio_frac_over_weak,status"
    )

    open(path, "w") do io
        println(io, header)
        for row in rows
            fields = (
                string(row.N),
                string(row.dimension),
                csv_float(row.weak_sigma_max),
                csv_float(row.weak_sigma_min),
                csv_float(row.weak_cond2),
                csv_float(row.weak_log10_cond2),
                string(row.weak_nnz),
                csv_float(row.weak_density),
                csv_float(row.frac_sigma_max),
                csv_float(row.frac_sigma_min),
                csv_float(row.frac_cond2),
                csv_float(row.frac_log10_cond2),
                string(row.frac_nnz),
                csv_float(row.frac_density),
                csv_float(row.cond2_ratio_frac_over_weak),
                csv_string(row.status),
            )
            println(io, join(fields, ','))
        end
    end
    return path
end

function main()
    mu = 0.5
    alpha = 1.0 / 3.0
    beta = 0.5
    muntz_lambda = 1.0 / 6.0
    muntz_jacobi_a = -0.5
    muntz_jacobi_b = -0.5
    Ns = parse_ns()

    println("ex7_condition_numbers.jl")
    println("Ns = ", join(Ns, ','))
    println("condition metric = exact dense Float64 SVD 2-norm condition number")
    @printf(
        "mu=%.16g, weak(alpha,beta)=(%.16g,%.16g), Muntz(lambda,a,b)=(%.16g,%.16g,%.16g)\n",
        mu,
        alpha,
        beta,
        muntz_lambda,
        muntz_jacobi_a,
        muntz_jacobi_b,
    )

    # These two method-specific objects are independent of N and are reused.
    kernel_coeffs = build_kernel_coefficients(beta)
    frac_problem = make_frac_problem(
        mu,
        muntz_lambda,
        muntz_jacobi_a,
        muntz_jacobi_b,
    )

    rows = NamedTuple[]
    sizehint!(rows, length(Ns))
    for N in Ns
        row = matrix_result(
            N,
            kernel_coeffs,
            frac_problem,
            mu,
            alpha,
            beta,
        )
        push!(rows, row)
        @printf(
            "N=%d dimension=%d weak_cond2=%.6e frac_cond2=%.6e ratio=%.6e status=%s\n",
            row.N,
            row.dimension,
            row.weak_cond2,
            row.frac_cond2,
            row.cond2_ratio_frac_over_weak,
            row.status,
        )
    end

    data_dir = joinpath(SCRIPT_DIR, "data")
    mkpath(data_dir)
    csv_path = joinpath(data_dir, "ex7_condition_numbers.csv")
    write_csv(csv_path, rows)
    println("wrote ", csv_path)
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
