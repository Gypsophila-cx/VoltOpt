using LinearAlgebra
using SparseArrays
using Printf

# ============================================================
# Example 7: Weak_VoltMat_Cheb for a weakly singular VIE
#
#   u(x) + lambda * int_{-1}^x (x-y)^(-mu) K(x,y) u(y) dy = g(x),
#   K(x,y) = 1 / ((x+1)^2 + (y-1)^2),
#   g(x) = 1 + sin(pi*x)/(1000*x^2 + 1),
#   lambda = 0.5, mu = 0.5.
#
# The script computes Cauchy errors between degree N and
# M = max(N+1, round(Int, 1.01*N)) JFP/Chebyshev approximations.
# Output:
#   ex7_weaksingular/data/ex7_weaksingular_cauchy.csv
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

function parse_int_arg(args, name::String, default::Int)
    prefix = "--" * name * "="
    for a in args
        startswith(a, prefix) && return parse(Int, a[(lastindex(prefix)+1):end])
    end
    return default
end

function parse_float_arg(args, name::String, default::Float64)
    prefix = "--" * name * "="
    for a in args
        startswith(a, prefix) && return parse(Float64, a[(lastindex(prefix)+1):end])
    end
    return default
end

function parse_symbol_arg(args, name::String, default::Symbol)
    prefix = "--" * name * "="
    for a in args
        startswith(a, prefix) && return Symbol(a[(lastindex(prefix)+1):end])
    end
    return default
end

project_root = find_project_root()
include(joinpath(project_root, "src", "VoltOpt.jl"))
using .VoltOpt

WeakCheb = VoltOpt.Weak_VoltMat_Cheb

# ============================================================
# Problem data and basis map
# ============================================================

mu = 0.5
jfp_beta = 1.0 - mu
sigma = 0.0
lambda_vie = 0.5

K_rational(x, y) = 1.0 / ((x + 1.0)^2 + (y - 1.0)^2)
g_rhs(x) = 1.0 + sin(pi * x) / (1000.0 * x^2 + 1.0)

phi_beta(x, beta) = 2.0 * ((1.0 + Float64(x)) / 2.0)^beta - 1.0
invphi_beta(z, beta) = 2.0 * ((1.0 + Float64(z)) / 2.0)^(1.0 / beta) - 1.0

function cheb_eval(c::AbstractVector{<:Real}, z::Real)
    n = length(c) - 1
    n < 0 && throw(ArgumentError("coefficient vector must be nonempty."))
    n == 0 && return Float64(c[1])

    zz = Float64(z)
    bkp1 = 0.0
    bkp2 = 0.0
    @inbounds for k in n:-1:1
        bk = 2.0 * zz * bkp1 - bkp2 + Float64(c[k + 1])
        bkp2 = bkp1
        bkp1 = bk
    end
    return zz * bkp1 - bkp2 + Float64(c[1])
end

eval_jfp(c::AbstractVector{<:Real}, x::Real, beta::Float64) = cheb_eval(c, phi_beta(x, beta))

# Chebyshev projection of h(z) = g(invphi_beta(z)).
# This is a DCT-I implementation using Lobatto samples.
function cheb_project_lobatto(f, coeff_degree::Int, sample_degree::Int)
    coeff_degree <= sample_degree ||
        throw(ArgumentError("coeff_degree must not exceed sample_degree."))
    sample_degree >= 1 || throw(ArgumentError("sample_degree must be positive."))

    M = sample_degree
    vals = Vector{Float64}(undef, M + 1)
    @inbounds for j in 0:M
        z = cos(pi * j / M)
        vals[j + 1] = f(z)
    end

    c = zeros(Float64, coeff_degree + 1)
    @inbounds for k in 0:coeff_degree
        s = 0.0
        for j in 0:M
            wj = (j == 0 || j == M) ? 0.5 : 1.0
            s += wj * vals[j + 1] * cos(pi * k * j / M)
        end
        c[k + 1] = 2.0 * s / M
    end
    c[1] *= 0.5

    tail_count = min(20, length(c))
    tail_inf = maximum(abs, @view c[(end - tail_count + 1):end])
    return c, (sample_degree = M, coeff_degree = coeff_degree,
               tail_count = tail_count, tail_inf = tail_inf)
end

function build_rhs_coefficients(max_degree::Int, sample_degree::Int, check_degree::Int, beta::Float64)
    sample_degree = max(sample_degree, max_degree)
    check_degree = max(check_degree, max_degree)

    h = z -> g_rhs(invphi_beta(z, beta))

    main_time = @elapsed begin
        c, info = cheb_project_lobatto(h, max_degree, sample_degree)
    end

    check_time = 0.0
    diff_inf = NaN
    diff_rel = NaN
    check_info = (sample_degree = 0, coeff_degree = max_degree,
                  tail_count = 0, tail_inf = NaN)

    if check_degree > 0
        check_time = @elapsed begin
            c_check, check_info = cheb_project_lobatto(h, max_degree, check_degree)
        end
        diff_inf = norm(c .- c_check, Inf)
        diff_rel = diff_inf / max(norm(c, Inf), floatmin(Float64))
    end

    return c, (main = info, check = check_info,
               coeff_diff_inf = diff_inf, coeff_diff_rel = diff_rel,
               main_time = main_time, check_time = check_time)
end

function build_kernel_coefficients(kernel_i::Int, kernel_j::Int, beta::Float64)
    t = @elapsed begin
        values = WeakCheb.sample_duffy_kernel(K_rational, kernel_i, kernel_j, beta)
        C = WeakCheb.cheb_coeffs_lobatto_2d(values)
    end
    tail = WeakCheb.coefficient_tail_info(C)
    return C, t, tail
end

function maybe_call(fsym::Symbol, args...)
    isdefined(VoltOpt, fsym) || return nothing
    f = getfield(VoltOpt, fsym)
    return f(args...)
end

function operator_to_sparse_or_dense(V, n::Int; prefer_sparse::Bool = true)
    if V isa SparseMatrixCSC
        A = copy(V)
        dropzeros!(A)
        return A
    elseif V isa AbstractMatrix
        if prefer_sparse
            A = sparse(V)
            dropzeros!(A)
            return A
        else
            return Matrix(V)
        end
    end

    # PackedLowerHessenberg path, if the current VoltOpt has converters.
    if isdefined(VoltOpt, :PackedLowerHessenberg)
        T = getfield(VoltOpt, :PackedLowerHessenberg)
        if V isa T
            if prefer_sparse && isdefined(VoltOpt, :packed_to_sparse)
                A = getfield(VoltOpt, :packed_to_sparse)(V)
                dropzeros!(A)
                return A
            elseif isdefined(VoltOpt, :packed_to_dense)
                D = getfield(VoltOpt, :packed_to_dense)(V)
                if prefer_sparse
                    A = sparse(D)
                    dropzeros!(A)
                    return A
                else
                    return Matrix(D)
                end
            end
        end
    end

    # Last-resort conversion if Matrix(V) is defined.
    try
        D = Matrix(V)
        if prefer_sparse
            A = sparse(D)
            dropzeros!(A)
            return A
        else
            return D
        end
    catch err
        error("Cannot convert operator of type $(typeof(V)) to a matrix. Original error: $(err)")
    end
end

function add_identity_scaled(Vmat, scale::Float64)
    n = size(Vmat, 1)
    size(Vmat, 2) == n || error("operator matrix must be square")

    if Vmat isa SparseMatrixCSC
        A = scale .* copy(Vmat)
        @inbounds for i in 1:n
            A[i, i] += 1.0
        end
        dropzeros!(A)
        return A
    else
        A = Matrix{Float64}(I, n, n)
        A .+= scale .* Vmat
        return A
    end
end

function build_system_matrix(N::Int, C::AbstractMatrix;
        output_format::Symbol, prefer_sparse::Bool,
        tried_formats::Vector{Symbol})

    formats = Symbol[]
    for fmt in (output_format, :sparse, :packed, :dense)
        if !(fmt in formats)
            push!(formats, fmt)
        end
    end

    last_error = nothing
    for fmt in formats
        push!(tried_formats, fmt)
        try
            matrix_time = @elapsed begin
                V, info = VoltOpt.weak_voltmat_cheb(
                    C;
                    C_is_coeffs = true,
                    mu = mu,
                    sigma = sigma,
                    beta = jfp_beta,
                    N = N,
                    diagnostics = false,
                    output_format = fmt,
                    return_info = true,
                )
                Vmat = operator_to_sparse_or_dense(V, N + 1; prefer_sparse = prefer_sparse)
                A = add_identity_scaled(Vmat, lambda_vie)
            end
            return A, info, matrix_time, fmt
        catch err
            last_error = err
        end
    end

    error("All operator formats failed for N=$(N). Last error: $(last_error)")
end

function solve_for_N(N::Int, C::AbstractMatrix, rhs_coeffs::AbstractVector;
        output_format::Symbol, prefer_sparse::Bool)

    rhsN = rhs_coeffs[1:(N + 1)]
    tried_formats = Symbol[]
    A, info, matrix_time, used_format = build_system_matrix(
        N, C;
        output_format = output_format,
        prefer_sparse = prefer_sparse,
        tried_formats = tried_formats,
    )

    solve_time = @elapsed begin
        c = A \ rhsN
    end

    residual_inf = norm(A * c - rhsN, Inf)
    nnzA = A isa SparseMatrixCSC ? nnz(A) : count(!iszero, A)

    return (
        N = N,
        coeffs = c,
        residual_inf = residual_inf,
        matrix_time = matrix_time,
        solve_time = solve_time,
        used_format = used_format,
        tried_formats = copy(tried_formats),
        matrix_storage = A isa SparseMatrixCSC ? "sparse" : "dense",
        nnzA = nnzA,
        info = info,
    )
end

function cauchy_M(N::Int)
    return max(N + 1, round(Int, 1.01 * N))
end

function make_N_values(Nmin::Int, Nmax::Int, numN::Int)
    numN >= 2 || return [Nmax]
    return unique(round.(Int, range(Nmin, Nmax; length = numN)))
end

function evaluation_grid(grid_size::Int, beta::Float64)
    x_uniform = collect(range(-1.0, 1.0; length = grid_size))
    z_uniform = collect(range(-1.0, 1.0; length = grid_size))
    x_jfp = invphi_beta.(z_uniform, beta)
    return sort(unique(vcat(x_uniform, x_jfp)))
end

function coefficient_cauchy(cN::AbstractVector, cM::AbstractVector)
    nN = length(cN)
    nM = length(cM)
    nN <= nM || error("coefficient_cauchy expects length(cN) <= length(cM)")

    err = norm(cN .- @view(cM[1:nN]), Inf)
    if nM > nN
        err = max(err, norm(@view(cM[(nN + 1):nM]), Inf))
    end
    return err
end

function grid_cauchy(cN::AbstractVector, cM::AbstractVector, xgrid, beta::Float64)
    err = 0.0
    @inbounds for x in xgrid
        e = abs(eval_jfp(cN, x, beta) - eval_jfp(cM, x, beta))
        err = max(err, e)
    end
    return err
end

function write_cauchy_csv(path::AbstractString, rows::Vector{<:NamedTuple})
    mkpath(dirname(path))
    open(path, "w") do io
        println(io,
            "N,M,coeff_cauchy_inf,grid_cauchy_inf," *
            "residual_inf_N,residual_inf_M," *
            "matrix_time_N,solve_time_N,total_time_N," *
            "matrix_time_M,solve_time_M,total_time_M," *
            "nnz_A_N,nnz_A_M,storage_N,storage_M,format_N,format_M,status")
        for r in rows
            @printf(io,
                "%d,%d,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%d,%d,%s,%s,%s,%s,%s\n",
                r.N, r.M,
                r.coeff_cauchy_inf, r.grid_cauchy_inf,
                r.residual_inf_N, r.residual_inf_M,
                r.matrix_time_N, r.solve_time_N, r.total_time_N,
                r.matrix_time_M, r.solve_time_M, r.total_time_M,
                r.nnz_A_N, r.nnz_A_M,
                r.storage_N, r.storage_M,
                string(r.format_N), string(r.format_M), r.status)
        end
    end
end

function write_info_csv(path::AbstractString, info::NamedTuple)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "key,value")
        for name in propertynames(info)
            value = getproperty(info, name)
            println(io, string(name), ",", string(value))
        end
    end
end

function main(args = ARGS)
    Nmin = parse_int_arg(args, "Nmin", 10)
    Nmax = parse_int_arg(args, "Nmax", 1500)
    numN = parse_int_arg(args, "numN", 100)
    kernel_i = parse_int_arg(args, "kernel-i", 80)
    kernel_j = parse_int_arg(args, "kernel-j", 64)
    rhs_sample = parse_int_arg(args, "rhs-sample", 8192)
    rhs_check = parse_int_arg(args, "rhs-check", 4096)
    grid_size = parse_int_arg(args, "grid-size", 5001)
    output_format = parse_symbol_arg(args, "format", :sparse)
    prefer_sparse_int = parse_int_arg(args, "prefer-sparse", 1)
    prefer_sparse = prefer_sparse_int != 0

    example_dir = basename(abspath(@__DIR__)) == "ex7_weaksingular" ?
                  abspath(@__DIR__) : joinpath(project_root, "examples", "ex7_weaksingular")
    data_dir = joinpath(example_dir, "data")
    mkpath(data_dir)

    Ns = make_N_values(Nmin, Nmax, numN)
    Ms = cauchy_M.(Ns)
    max_degree = maximum(Ms)

    println("============================================================")
    println("Example 7: Weak_VoltMat_Cheb Cauchy test")
    @printf("mu = %.16g, lambda = %.16g, JFP beta = %.16g\n", mu, lambda_vie, jfp_beta)
    @printf("N range: %d to %d, numN = %d, max M = %d\n", first(Ns), last(Ns), length(Ns), max_degree)
    @printf("kernel coefficients: %d x %d\n", kernel_i, kernel_j)
    @printf("rhs sample/check degrees: %d / %d\n", rhs_sample, rhs_check)
    @printf("operator format request: %s, prefer_sparse = %s\n", string(output_format), string(prefer_sparse))
    println("data dir = ", data_dir)
    println("============================================================")

    C, kernel_time, kernel_tail = build_kernel_coefficients(kernel_i, kernel_j, jfp_beta)
    @printf("kernel coefficient build time = %.6f s\n", kernel_time)
    @printf("kernel tail row_rel = %.3e, col_rel = %.3e, max_coeff = %.3e\n",
        kernel_tail.row_rel, kernel_tail.col_rel, kernel_tail.max_coeff)

    rhs_coeffs, rhs_info = build_rhs_coefficients(max_degree, rhs_sample, rhs_check, jfp_beta)
    @printf("rhs coefficient projection time = %.6f s\n", rhs_info.main_time)
    if rhs_check > 0
        @printf("rhs check time = %.6f s, coeff diff inf = %.3e, rel = %.3e\n",
            rhs_info.check_time, rhs_info.coeff_diff_inf, rhs_info.coeff_diff_rel)
    end
    @printf("rhs tail last %d modes = %.3e\n", rhs_info.main.tail_count, rhs_info.main.tail_inf)

    xgrid = evaluation_grid(grid_size, jfp_beta)
    @printf("evaluation grid size = %d\n", length(xgrid))
    println("------------------------------------------------------------")

    rows = NamedTuple[]
    for (idx, N) in enumerate(Ns)
        M = cauchy_M(N)
        status = "ok"
        try
            resN = solve_for_N(N, C, rhs_coeffs;
                output_format = output_format,
                prefer_sparse = prefer_sparse)
            resM = solve_for_N(M, C, rhs_coeffs;
                output_format = output_format,
                prefer_sparse = prefer_sparse)

            coeff_err = coefficient_cauchy(resN.coeffs, resM.coeffs)
            grid_err = grid_cauchy(resN.coeffs, resM.coeffs, xgrid, jfp_beta)

            push!(rows, (
                N = N,
                M = M,
                coeff_cauchy_inf = coeff_err,
                grid_cauchy_inf = grid_err,
                residual_inf_N = resN.residual_inf,
                residual_inf_M = resM.residual_inf,
                matrix_time_N = resN.matrix_time,
                solve_time_N = resN.solve_time,
                total_time_N = resN.matrix_time + resN.solve_time,
                matrix_time_M = resM.matrix_time,
                solve_time_M = resM.solve_time,
                total_time_M = resM.matrix_time + resM.solve_time,
                nnz_A_N = resN.nnzA,
                nnz_A_M = resM.nnzA,
                storage_N = resN.matrix_storage,
                storage_M = resM.matrix_storage,
                format_N = resN.used_format,
                format_M = resM.used_format,
                status = status,
            ))

            @printf("[%3d/%3d] N=%4d M=%4d coeff=%.3e grid=%.3e resN=%.3e resM=%.3e fmt=(%s,%s) storage=(%s,%s)\n",
                idx, length(Ns), N, M, coeff_err, grid_err,
                resN.residual_inf, resM.residual_inf,
                string(resN.used_format), string(resM.used_format),
                resN.matrix_storage, resM.matrix_storage)
        catch err
            status = "failed: " * sprint(showerror, err)
            @printf("[%3d/%3d] N=%4d M=%4d FAILED: %s\n", idx, length(Ns), N, M, status)
            push!(rows, (
                N = N,
                M = M,
                coeff_cauchy_inf = NaN,
                grid_cauchy_inf = NaN,
                residual_inf_N = NaN,
                residual_inf_M = NaN,
                matrix_time_N = NaN,
                solve_time_N = NaN,
                total_time_N = NaN,
                matrix_time_M = NaN,
                solve_time_M = NaN,
                total_time_M = NaN,
                nnz_A_N = 0,
                nnz_A_M = 0,
                storage_N = "failed",
                storage_M = "failed",
                format_N = :failed,
                format_M = :failed,
                status = status,
            ))
        end
    end

    cauchy_csv = joinpath(data_dir, "ex7_weaksingular_cauchy.csv")
    write_cauchy_csv(cauchy_csv, rows)

    info_csv = joinpath(data_dir, "ex7_weaksingular_runinfo.csv")
    write_info_csv(info_csv, (
        mu = mu,
        lambda = lambda_vie,
        jfp_beta = jfp_beta,
        sigma = sigma,
        Nmin = Nmin,
        Nmax = Nmax,
        numN = length(Ns),
        maxM = max_degree,
        kernel_i = kernel_i,
        kernel_j = kernel_j,
        kernel_time = kernel_time,
        kernel_tail_row_rel = kernel_tail.row_rel,
        kernel_tail_col_rel = kernel_tail.col_rel,
        kernel_tail_max_coeff = kernel_tail.max_coeff,
        rhs_sample_degree = rhs_info.main.sample_degree,
        rhs_check_degree = rhs_info.check.sample_degree,
        rhs_coeff_tail_inf = rhs_info.main.tail_inf,
        rhs_coeff_diff_inf = rhs_info.coeff_diff_inf,
        rhs_coeff_diff_rel = rhs_info.coeff_diff_rel,
        rhs_time = rhs_info.main_time,
        rhs_check_time = rhs_info.check_time,
        grid_size = length(xgrid),
        requested_format = output_format,
        prefer_sparse = prefer_sparse,
    ))

    println("------------------------------------------------------------")
    println("Saved:")
    println(cauchy_csv)
    println(info_csv)
    println("============================================================")
end

main()
