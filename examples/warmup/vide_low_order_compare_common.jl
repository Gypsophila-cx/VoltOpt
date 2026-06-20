# Common implementation for the low-order VIDE warmup comparison scripts.
#
# The exported entry points in this file deliberately keep the same test problem,
# basis choices, kernel truncation sizes, error grid, and diagnostics as the
# unified comparison script.

using LinearAlgebra
using Printf
using SparseArrays
using Statistics
using FFTW
using Pkg
using FastAlmostBandedMatrices

function find_project_root(start_dir::AbstractString)
    dir = abspath(start_dir)
    while true
        if isfile(joinpath(dir, "Project.toml")) &&
           isdir(joinpath(dir, "examples", "warmup")) &&
           isdir(joinpath(dir, "others", "GutlebOlver"))
            return dir
        end
        parent = dirname(dir)
        parent == dir && error("Could not locate VoltOpt project root from $start_dir")
        dir = parent
    end
end

const PROJECT_ROOT = find_project_root(@__DIR__)
const WARMUP_DIR = joinpath(PROJECT_ROOT, "examples", "warmup")

include(joinpath(PROJECT_ROOT, "src", "VoltOpt.jl"))
using .VoltOpt

const GUTLEB_OLVER_PROJECT = joinpath(PROJECT_ROOT, "others", "GutlebOlver")
Pkg.activate(GUTLEB_OLVER_PROJECT; io=devnull)

using ApproxFun
using BandedMatrices
using GutlebOlver

const DEFAULT_TOL = eps()
const KERNELDEPTH = 15
const VOLTMAT_C_ROWS = 15
const VOLTMAT_C_COLS = 15
const TARGET_KERNEL_BANDWIDTH = 15
const ERROR_GRID = collect(range(-1.0, 1.0; length=2001))
const GO_SOLUTION_SPACE = Jacobi(0, 1, 0..1)
const GO_OUTPUT_SPACE = Jacobi(1, 2, 0..1)

const METHOD_SPECS = (
    (method="VoltMat", basis="Chebyshev", runner=:cheb, kerneldepth="", C_rows=VOLTMAT_C_ROWS, C_cols=VOLTMAT_C_COLS),
    (method="VoltMat", basis="Legendre", runner=:leg, kerneldepth="", C_rows=VOLTMAT_C_ROWS, C_cols=VOLTMAT_C_COLS),
    (method="GutlebOlver", basis="shifted Jacobi P^(1,0)", runner=:go, kerneldepth=KERNELDEPTH, C_rows="", C_cols=""),
)

# Original x-domain problem:
#   2u_x(x) + u(x) - int_{-1}^x K(x,t)u(t)dt = x + 2,  u(-1)=1.
kernel_fun(x, t) = ((x + 1.0) * (x + 2.0) / 4.0) * exp(((t + 1.0) * (x - t)) / 4.0)
rhs_fun(x) = x + 2.0
uexact(x) = exp(((x + 1.0)^2) / 4.0)

# Mapped Gutleb--Olver s-domain equation:
#   u_s(s) + u(s) - int_0^s K_go(s,r)u(r)dr = 2s + 1, u(0)=1.
to_go_s(x) = (x + 1.0) / 2.0
from_go_s(s) = 2.0s - 1.0
kernel_go(s, r) = 2.0 * kernel_fun(from_go_s(s), from_go_s(r))
rhs_go(s) = 2.0s + 1.0
uexact_go(s) = uexact(from_go_s(s))

function fixed_cheb_coeffs_Khat(K, nrow::Int, mcol::Int)
    nrow >= 2 || throw(ArgumentError("nrow must be at least 2"))
    mcol >= 2 || throw(ArgumentError("mcol must be at least 2"))

    xs = [cos((i - 1) * pi / (nrow - 1)) for i in 1:nrow]
    ys = [cos((j - 1) * pi / (mcol - 1)) for j in 1:mcol]
    Khat(x, y) = K(x, (1.0 + x) * (1.0 + y) / 2.0 - 1.0)
    values = [Khat(x, y) for x in xs, y in ys]

    C = FFTW.r2r(values, FFTW.REDFT00)
    C ./= (nrow - 1) * (mcol - 1)
    C[1, :] ./= 2.0
    C[end, :] ./= 2.0
    C[:, 1] ./= 2.0
    C[:, end] ./= 2.0
    return Matrix{Float64}(C)
end

function fixed_legendre_coeffs_Khat(K, nrow::Int, mcol::Int)
    return Matrix{Float64}(VoltOpt.legendre_interp_coeffs_Khat(K, nrow, mcol))
end

function cheb_derivative_to_c1(N::Int)
    D = spzeros(Float64, N, N)
    for k in 1:(N - 1)
        D[k, k + 1] = k
    end
    return D
end

function cheb_to_c1_conversion(N::Int)
    S = spzeros(Float64, N, N)
    N >= 1 && (S[1, 1] = 1.0)
    N >= 2 && (S[2, 2] = 0.5)
    for j in 3:N
        S[j, j] += 0.5
        S[j - 2, j] -= 0.5
    end
    return S
end

function leg_derivative_to_c32(N::Int)
    D = spzeros(Float64, N, N)
    for k in 1:(N - 1)
        D[k, k + 1] = 1.0
    end
    return D
end

function leg_to_c32_conversion(N::Int)
    S = spzeros(Float64, N, N)
    N >= 1 && (S[1, 1] = 1.0)
    for j in 2:N
        deg = j - 1
        c = 1.0 / (2deg + 1)
        S[j, j] += c
        j >= 3 && (S[j - 2, j] -= c)
    end
    return S
end

function rhs_coefficients(N::Int)
    f = zeros(Float64, N)
    N >= 1 && (f[1] = 2.0)
    N >= 2 && (f[2] = 1.0)
    return f
end

function eval_cheb_series(c::AbstractVector, x::Real)
    n = length(c)
    n == 0 && return 0.0
    n == 1 && return c[1]
    bkp1 = 0.0
    bkp2 = 0.0
    for j in n:-1:2
        bk = 2.0 * x * bkp1 - bkp2 + c[j]
        bkp2 = bkp1
        bkp1 = bk
    end
    return c[1] + x * bkp1 - bkp2
end

function eval_leg_series(c::AbstractVector, x::Real)
    n = length(c)
    n == 0 && return 0.0
    p0 = 1.0
    y = c[1] * p0
    n == 1 && return y
    p1 = x
    y += c[2] * p1
    for k in 1:(n - 2)
        p2 = ((2k + 1) * x * p1 - k * p0) / (k + 1)
        y += c[k + 2] * p2
        p0, p1 = p1, p2
    end
    return y
end

max_error_cheb(coeffs::AbstractVector) =
    maximum(abs(eval_cheb_series(coeffs, x) - uexact(x)) for x in ERROR_GRID)

max_error_leg(coeffs::AbstractVector) =
    maximum(abs(eval_leg_series(coeffs, x) - uexact(x)) for x in ERROR_GRID)

function max_error_go(coeffs::AbstractVector)
    ufun = Fun(GO_SOLUTION_SPACE, coeffs)
    return maximum(abs(ufun(to_go_s(x)) - uexact(x)) for x in ERROR_GRID)
end

matrix_type_name(A) = A isa Matrix ? "Matrix" : string(nameof(typeof(A)))

function matrix_maxabs(A)
    if A isa SparseMatrixCSC
        vals = nonzeros(A)
        return isempty(vals) ? 0.0 : maximum(abs, vals)
    end
    mx = 0.0
    for j in axes(A, 2), i in axes(A, 1)
        mx = max(mx, abs(A[i, j]))
    end
    return mx
end

function default_band_tol(A)
    scale = matrix_maxabs(A)
    scale == 0.0 && return 0.0
    return max(100.0 * eps(Float64) * scale, 1.0e-14 * scale)
end

function local_safe_nnz(A)
    A isa SparseMatrixCSC && return nnz(A)
    try
        return GutlebOlver.safe_nnz(A)
    catch
    end
    count = 0
    for j in axes(A, 2), i in axes(A, 1)
        count += !iszero(A[i, j])
    end
    return count
end

function pattern_bandwidths(A; row_range=nothing, band_tol=nothing)
    rows = row_range === nothing ? axes(A, 1) : row_range
    isempty(rows) && return 0, 0
    tol = band_tol === nothing ? default_band_tol(A) : Float64(band_tol)
    lband = 0
    uband = 0
    found = false

    if A isa SparseMatrixCSC
        I, J, V = findnz(A)
        @inbounds for k in eachindex(V)
            i = I[k]
            if i in rows && abs(V[k]) > tol
                j = J[k]
                lband = max(lband, Int(i - j))
                uband = max(uband, Int(j - i))
                found = true
            end
        end
    else
        @inbounds for j in axes(A, 2), i in rows
            if abs(A[i, j]) > tol
                lband = max(lband, Int(i - j))
                uband = max(uband, Int(j - i))
                found = true
            end
        end
    end

    return found ? (lband, uband) : (0, 0)
end

function local_safe_bandwidths(A; row_range=nothing)
    if row_range === nothing && occursin("Banded", string(typeof(A)))
        try
            bw = BandedMatrices.bandwidths(A)
            return Int(bw[1]), Int(bw[2])
        catch
        end
    end
    row_range !== nothing && return pattern_bandwidths(A; row_range=row_range)
    try
        bw = GutlebOlver.safe_bandwidth(A)
        return Int(bw[1]), Int(bw[2])
    catch
    end
    return pattern_bandwidths(A)
end

function cheb_top_dense_plus_band_diagnostics(A; top_dense_rows::Int=VOLTMAT_C_ROWS,
                                               bottom_dense_rows::Int=0)
    m, _ = size(A)
    top = min(top_dense_rows, m)
    last_body_row = max(0, m - bottom_dense_rows)
    body_rows = last_body_row >= 1 ? (1:last_body_row) : (1:0)
    below_top_rows = top + 1 <= last_body_row ? ((top + 1):last_body_row) : (1:0)

    global_lband, global_uband = pattern_bandwidths(A)
    lower_lband, lower_uband = pattern_bandwidths(A; row_range=below_top_rows)

    return (
        structure = bottom_dense_rows == 0 ?
            "top_dense_plus_lower_banded" :
            "top_dense_plus_lower_banded_with_tau_row",
        top_dense_rows = top,
        lower_bandwidth_below_top = lower_lband,
        global_lower_bandwidth = global_lband,
        global_upper_bandwidth = global_uband,
        structural_lband = lower_lband,
        structural_uband = global_uband,
        band_lband = lower_lband,
        band_uband = lower_uband,
    )
end

function strict_banded_diagnostics(A; bottom_dense_rows::Int=0)
    m, _ = size(A)
    last_body_row = max(0, m - bottom_dense_rows)
    body_rows = last_body_row >= 1 ? (1:last_body_row) : (1:0)
    global_lband, global_uband = pattern_bandwidths(A)
    body_lband, body_uband = pattern_bandwidths(A; row_range=body_rows)
    return (
        structure = bottom_dense_rows == 0 ? "strict_banded" : "strict_banded_body_with_tau_row",
        top_dense_rows = 0,
        lower_bandwidth_below_top = body_lband,
        global_lower_bandwidth = global_lband,
        global_upper_bandwidth = global_uband,
        structural_lband = body_lband,
        structural_uband = body_uband,
        band_lband = body_lband,
        band_uband = body_uband,
    )
end

function top_dense_plus_band_diagnostics(A; top_dense_rows::Int,
                                         structure::String="top_dense_plus_banded")
    m, _ = size(A)
    top = min(top_dense_rows, m)
    below_top_rows = top + 1 <= m ? ((top + 1):m) : (1:0)
    global_lband, global_uband = pattern_bandwidths(A; band_tol=0.0)
    band_lband, band_uband = pattern_bandwidths(A; row_range=below_top_rows, band_tol=0.0)
    return (
        structure = structure,
        top_dense_rows = top,
        lower_bandwidth_below_top = band_lband,
        global_lower_bandwidth = global_lband,
        global_upper_bandwidth = global_uband,
        structural_lband = band_lband,
        structural_uband = global_uband,
        band_lband = band_lband,
        band_uband = band_uband,
    )
end

function generic_matrix_diagnostics(A; structure::String="")
    lband, uband = local_safe_bandwidths(A)
    return (
        structure = isempty(structure) ? matrix_type_name(A) : structure,
        top_dense_rows = "",
        lower_bandwidth_below_top = "",
        global_lower_bandwidth = lband,
        global_upper_bandwidth = uband,
        structural_lband = lband,
        structural_uband = uband,
        band_lband = lband,
        band_uband = uband,
    )
end

function residual_ratios(residual::AbstractVector, rhs::AbstractVector, body_rows)
    rhs_norm = norm(rhs)
    residual_relative = rhs_norm == 0 ? norm(residual) : norm(residual) / rhs_norm
    if isempty(body_rows)
        body_residual_relative = residual_relative
    else
        body_rhs_norm = norm(rhs[body_rows])
        body_residual_relative = body_rhs_norm == 0 ? norm(residual[body_rows]) :
                                 norm(residual[body_rows]) / body_rhs_norm
    end
    return residual_relative, body_residual_relative
end

function dense_from_matvec(A, n::Integer)
    m, cols = size(A)
    Int(n) == cols || throw(ArgumentError("requested $n columns, but matrix has $cols"))
    T = promote_type(eltype(A), Float64)
    B = zeros(T, m, cols)
    e = zeros(T, cols)
    @inbounds for j in 1:cols
        e[j] = one(T)
        B[:, j] = A * e
        e[j] = zero(T)
    end
    return Matrix(B)
end

function dense_for_comparison(A, n::Integer)
    try
        return Matrix(A)
    catch
        return dense_from_matvec(A, n)
    end
end

function to_fast_almost_banded(A; top_dense_rows::Integer=0, lband::Integer=0, uband::Integer=0)
    m, n = size(A)
    m == n || throw(ArgumentError("Fast almost-banded solve expects a square matrix, got $(size(A))"))

    fill_rows = clamp(Int(top_dense_rows), 0, m)
    body_rows = fill_rows + 1 <= m ? ((fill_rows + 1):m) : (1:0)
    exact_lband, exact_uband = pattern_bandwidths(A; row_range=body_rows, band_tol=0.0)
    band_l = max(Int(lband), exact_lband, fill_rows - 1, 0)
    band_u = max(Int(uband), exact_uband, 0)

    bands = BandedMatrix(zeros(Float64, m, n), (band_l, band_u))
    @inbounds for j in 1:n
        i1 = max(1, j - band_u)
        i2 = min(m, j + band_l)
        for i in i1:i2
            if fill_rows == 0 || i > fill_rows
                bands[i, j] = A[i, j]
            end
        end
    end

    if fill_rows == 0
        return bands, (
            solver_backend = "fastalmostbanded",
            fast_matrix_type = matrix_type_name(bands),
            fast_matrix_lband = band_l,
            fast_matrix_uband = band_u,
            fast_matrix_fill_rows = 0,
            fallback_reason = "",
        )
    end

    fill = zeros(Float64, fill_rows, n)
    @inbounds for j in 1:n, i in 1:fill_rows
        fill[i, j] = A[i, j]
    end

    # AlmostBandedMatrix stores the top fill rows with an overlap in the banded
    # part. Its constructor copies fill values into the overlapping band entries.
    fast_A = FastAlmostBandedMatrices.AlmostBandedMatrix(bands, fill)
    return fast_A, (
        solver_backend = "fastalmostbanded",
        fast_matrix_type = matrix_type_name(fast_A),
        fast_matrix_lband = band_l,
        fast_matrix_uband = band_u,
        fast_matrix_fill_rows = fill_rows,
        fallback_reason = "",
    )
end

function solve_vide_system(Afinal, rhs; solver::Symbol=:fastalmostbanded,
                           top_dense_rows::Integer=0,
                           lband::Integer=0,
                           uband::Integer=0,
                           allow_sparse_fallback::Bool=true)
    if solver === :dense
        local sol
        solve_time = @elapsed begin
            sol = Matrix(Afinal) \ rhs
        end
        return sol, (
            solver = "dense",
            solver_backend = "dense",
            fast_matrix_type = "",
            fast_matrix_lband = "",
            fast_matrix_uband = "",
            fast_matrix_fill_rows = "",
            fallback_reason = "",
            solve_time = solve_time,
        )
    elseif solver === :sparse
        local sol
        solve_time = @elapsed begin
            sol = sparse(Afinal) \ rhs
        end
        return sol, (
            solver = "sparse",
            solver_backend = "sparse",
            fast_matrix_type = "",
            fast_matrix_lband = "",
            fast_matrix_uband = "",
            fast_matrix_fill_rows = "",
            fallback_reason = "",
            solve_time = solve_time,
        )
    elseif solver !== :fastalmostbanded
        throw(ArgumentError("unsupported solver $(solver); expected :fastalmostbanded, :dense, or :sparse"))
    end

    try
        fast_A, fast_info = to_fast_almost_banded(
            Afinal;
            top_dense_rows=top_dense_rows,
            lband=lband,
            uband=uband,
        )
        local sol
        solve_time = @elapsed begin
            sol = fast_A \ rhs
        end
        return sol, merge(fast_info, (solver="fastalmostbanded", solve_time=solve_time))
    catch err
        allow_sparse_fallback || rethrow()
        fallback_reason = sprint(showerror, err)
        local sol
        solve_time = @elapsed begin
            sol = sparse(Afinal) \ rhs
        end
        return sol, (
            solver = "fastalmostbanded",
            solver_backend = "sparse_fallback",
            fast_matrix_type = "",
            fast_matrix_lband = "",
            fast_matrix_uband = "",
            fast_matrix_fill_rows = "",
            fallback_reason = fallback_reason,
            solve_time = solve_time,
        )
    end
end

function build_voltmat_system(N::Int, basis::Symbol)
    N >= 2 || error("VoltMat warmup solve requires N >= 2")

    C = nothing
    V = nothing
    A = nothing
    rhs = nothing
    coeff_time = NaN
    construct_time = NaN
    final_time = NaN

    coeff_time = @elapsed begin
        C = basis === :cheb ?
            fixed_cheb_coeffs_Khat(kernel_fun, VOLTMAT_C_ROWS, VOLTMAT_C_COLS) :
            fixed_legendre_coeffs_Khat(kernel_fun, VOLTMAT_C_ROWS, VOLTMAT_C_COLS)
    end
    size(C) == (VOLTMAT_C_ROWS, VOLTMAT_C_COLS) ||
        error("fixed VoltMat C has size $(size(C)); expected $((VOLTMAT_C_ROWS, VOLTMAT_C_COLS))")

    construct_time = @elapsed begin
        if basis === :cheb
            V = sparse(VoltOpt.voltmat_cheb(
                C,
                N - 1;
                tol=DEFAULT_TOL,
                tail_half_bandwidth=VOLTMAT_C_ROWS,
            ))
        elseif basis === :leg
            V = sparse(VoltOpt.voltmat_leg(
                C,
                N - 1;
                bandwidth=VOLTMAT_C_ROWS,
            ))
        else
            error("unknown VoltMat basis $basis")
        end
    end

    final_time = @elapsed begin
        D = basis === :cheb ? cheb_derivative_to_c1(N) : leg_derivative_to_c32(N)
        S = basis === :cheb ? cheb_to_c1_conversion(N) : leg_to_c32_conversion(N)
        f = rhs_coefficients(N)
        body = Matrix(2.0 * D + S - S * V)
        body_rhs = Vector(S * f)

        # FastAlmostBandedMatrices stores dense correction rows at the top.
        # This is the same linear system as the original bottom tau row,
        # only with the initial condition row moved before the body equations.
        A = zeros(Float64, N, N)
        for j in 1:N
            A[1, j] = (-1.0)^(j - 1)
        end
        A[2:N, :] .= body[1:(N - 1), :]

        rhs = zeros(Float64, N)
        rhs[1] = 1.0
        rhs[2:N] .= body_rhs[1:(N - 1)]
    end

    raw_diag = basis === :cheb ?
        cheb_top_dense_plus_band_diagnostics(V; top_dense_rows=VOLTMAT_C_ROWS) :
        strict_banded_diagnostics(V)
    final_diag = basis === :cheb ?
        top_dense_plus_band_diagnostics(
            A;
            top_dense_rows=min(N, VOLTMAT_C_ROWS + 1),
            structure="top_tau_plus_cheb_top_dense_plus_lower_banded",
        ) :
        top_dense_plus_band_diagnostics(
            A;
            top_dense_rows=1,
            structure="top_tau_plus_strict_banded",
        )

    return (
        C=C,
        V=V,
        A=A,
        rhs=rhs,
        coeff_time=coeff_time,
        construct_time=construct_time,
        final_time=final_time,
        raw_diag=raw_diag,
        final_diag=final_diag,
        solver_top_dense_rows=final_diag.top_dense_rows,
        solver_lband=final_diag.band_lband,
        solver_uband=final_diag.band_uband,
    )
end

function run_voltmat_case(N::Int, basis::Symbol; solver::Symbol=:fastalmostbanded)
    start_time_ns = time_ns()
    sys = build_voltmat_system(N, basis)
    coeffs, solve_info = solve_vide_system(
        sys.A,
        sys.rhs;
        solver=solver,
        top_dense_rows=sys.solver_top_dense_rows,
        lband=sys.solver_lband,
        uband=sys.solver_uband,
    )

    residual = sys.A * coeffs - sys.rhs
    residual_relative, body_residual_relative = residual_ratios(residual, sys.rhs, 2:N)
    err = basis === :cheb ? max_error_cheb(coeffs) : max_error_leg(coeffs)
    basis_name = basis === :cheb ? "Chebyshev" : "Legendre"
    fallback_message = solve_info.solver_backend == "sparse_fallback" ?
        "sparse_fallback: $(solve_info.fallback_reason)" : ""

    return (
        method="VoltMat",
        basis=basis_name,
        N=N,
        kerneldepth="",
        C_rows=size(sys.C, 1),
        C_cols=size(sys.C, 2),
        target_kernel_bandwidth=TARGET_KERNEL_BANDWIDTH,
        coeff_precompute_time=sys.coeff_time,
        kernel_coeff_time=sys.coeff_time,
        volterra_matrix_construct_time=sys.construct_time,
        final_matrix_assembly_time=sys.final_time,
        solve_time=solve_info.solve_time,
        elapsed_time=(time_ns() - start_time_ns) * 1.0e-9,
        max_abs_error=err,
        residual_relative=residual_relative,
        body_residual_relative=body_residual_relative,
        raw_matrix_type=matrix_type_name(sys.V),
        raw_matrix_nnz=local_safe_nnz(sys.V),
        raw_matrix_lband=sys.raw_diag.structural_lband,
        raw_matrix_uband=sys.raw_diag.structural_uband,
        raw_structure=sys.raw_diag.structure,
        raw_global_lband=sys.raw_diag.global_lower_bandwidth,
        raw_global_uband=sys.raw_diag.global_upper_bandwidth,
        final_matrix_type=matrix_type_name(sys.A),
        final_matrix_nnz=local_safe_nnz(sys.A),
        final_matrix_lband=sys.final_diag.structural_lband,
        final_matrix_uband=sys.final_diag.structural_uband,
        final_structure=sys.final_diag.structure,
        final_global_lband=sys.final_diag.global_lower_bandwidth,
        final_global_uband=sys.final_diag.global_upper_bandwidth,
        solver=solve_info.solver,
        solver_backend=solve_info.solver_backend,
        fast_matrix_type=solve_info.fast_matrix_type,
        fast_matrix_lband=solve_info.fast_matrix_lband,
        fast_matrix_uband=solve_info.fast_matrix_uband,
        fast_matrix_fill_rows=solve_info.fast_matrix_fill_rows,
        status="success",
        error_message=fallback_message,
    )
end

function build_gutleb_olver_system(N::Int; kerneldepth::Int=KERNELDEPTH)
    N >= 2 || error("Gutleb--Olver warmup solve requires N >= 2")

    kernel_coeffs = nothing
    rhs_coeff = nothing
    rhs_info = nothing
    Vraw = nothing
    A = nothing
    final_rhs = nothing
    sol = nothing
    precompute_time = NaN
    rhs_time = NaN
    volterra_time = NaN
    final_time = NaN

    precompute_time = @elapsed begin
        plan = precompute_triangular_kernel_plan(kerneldepth, true)
        kernel_coeffs = triangular_kernel_coeffs_from_plan(plan, kernel_go)
    end

    rhs_time = @elapsed begin
        rhs_coeff, rhs_info = build_go_rhs_coefficients(
            rhs_go,
            N;
            alpha=1,
            beta=2,
            method=:fixed_gauss_jacobi,
            projection_tol=1e-12,
        )
    end
    rhs_status = Symbol(rhs_info.status)
    rhs_status in (:ok, :validated, :pass) ||
        error("RHS projection status $(rhs_info.status)")

    volterra_time = @elapsed begin
        Vraw = triVolterraFullKernelOpP01_strictband_from_coeffs(kernel_coeffs, N, true)
    end

    final_time = @elapsed begin
        Vreflect_lower = apply_reflect_lowering_strictband(Vraw, N)
        Vconv = apply_conversion_00_to_12_strictband(Vreflect_lower, N)
        D = build_derivative_P01_to_P12(N)
        Csolution = Conversion(GO_SOLUTION_SPACE, GO_OUTPUT_SPACE)[1:N, 1:N]
        body = D + Csolution - Vconv
        evaluation_row = build_initial_evaluation_row(N)
        A = assemble_semibanded_system(body, N; evaluation_row=evaluation_row)
        final_rhs = zeros(Float64, N)
        final_rhs[1] = 1.0
        final_rhs[2:N] .= rhs_coeff[1:(N - 1)]
    end

    raw_diag = generic_matrix_diagnostics(Vraw; structure="strict_banded")
    final_diag = top_dense_plus_band_diagnostics(
        A;
        top_dense_rows=1,
        structure="semibanded_evaluation_row",
    )

    return (
        Vraw=Vraw,
        A=A,
        rhs=final_rhs,
        precompute_time=precompute_time,
        rhs_time=rhs_time,
        volterra_time=volterra_time,
        final_time=final_time,
        raw_diag=raw_diag,
        final_diag=final_diag,
        solver_top_dense_rows=final_diag.top_dense_rows,
        solver_lband=final_diag.band_lband,
        solver_uband=final_diag.band_uband,
    )
end

function run_gutleb_olver_case(N::Int; kerneldepth::Int=KERNELDEPTH,
                               solver::Symbol=:fastalmostbanded)
    start_time_ns = time_ns()
    sys = build_gutleb_olver_system(N; kerneldepth=kerneldepth)
    sol, solve_info = solve_vide_system(
        sys.A,
        sys.rhs;
        solver=solver,
        top_dense_rows=sys.solver_top_dense_rows,
        lband=sys.solver_lband,
        uband=sys.solver_uband,
    )

    residual = sys.A * sol - sys.rhs
    residual_relative, body_residual_relative = residual_ratios(residual, sys.rhs, 2:N)
    err = max_error_go(sol)
    fallback_message = solve_info.solver_backend == "sparse_fallback" ?
        "sparse_fallback: $(solve_info.fallback_reason)" : ""

    return (
        method="GutlebOlver",
        basis="shifted Jacobi P^(1,0)",
        N=N,
        kerneldepth=kerneldepth,
        C_rows="",
        C_cols="",
        target_kernel_bandwidth=kerneldepth,
        coeff_precompute_time=sys.precompute_time,
        kernel_coeff_time=sys.precompute_time,
        rhs_time=sys.rhs_time,
        volterra_matrix_construct_time=sys.volterra_time,
        final_matrix_assembly_time=sys.final_time,
        solve_time=solve_info.solve_time,
        elapsed_time=(time_ns() - start_time_ns) * 1.0e-9,
        max_abs_error=err,
        residual_relative=residual_relative,
        body_residual_relative=body_residual_relative,
        raw_matrix_type=matrix_type_name(sys.Vraw),
        raw_matrix_nnz=local_safe_nnz(sys.Vraw),
        raw_matrix_lband=sys.raw_diag.structural_lband,
        raw_matrix_uband=sys.raw_diag.structural_uband,
        raw_structure=sys.raw_diag.structure,
        raw_global_lband=sys.raw_diag.global_lower_bandwidth,
        raw_global_uband=sys.raw_diag.global_upper_bandwidth,
        final_matrix_type=matrix_type_name(sys.A),
        final_matrix_nnz=local_safe_nnz(sys.A),
        final_matrix_lband=sys.final_diag.structural_lband,
        final_matrix_uband=sys.final_diag.structural_uband,
        final_structure=sys.final_diag.structure,
        final_global_lband=sys.final_diag.global_lower_bandwidth,
        final_global_uband=sys.final_diag.global_upper_bandwidth,
        solver=solve_info.solver,
        solver_backend=solve_info.solver_backend,
        fast_matrix_type=solve_info.fast_matrix_type,
        fast_matrix_lband=solve_info.fast_matrix_lband,
        fast_matrix_uband=solve_info.fast_matrix_uband,
        fast_matrix_fill_rows=solve_info.fast_matrix_fill_rows,
        status="success",
        error_message=fallback_message,
    )
end

function build_system_for_spec(spec, N::Int)
    spec.runner === :cheb && return build_voltmat_system(N, :cheb)
    spec.runner === :leg && return build_voltmat_system(N, :leg)
    spec.runner === :go && return build_gutleb_olver_system(N; kerneldepth=KERNELDEPTH)
    error("unknown method runner $(spec.runner)")
end

function validate_fast_solver_consistency(; Ns=(10, 20, 50, 100), rtol=1.0e-12)
    for N in Ns
        for spec in METHOD_SPECS
            sys = build_system_for_spec(spec, N)
            fast_A, _ = to_fast_almost_banded(
                sys.A;
                top_dense_rows=sys.solver_top_dense_rows,
                lband=sys.solver_lband,
                uband=sys.solver_uband,
            )
            A_dense = Matrix(sys.A)
            A_fast_dense = dense_for_comparison(fast_A, N)
            matrix_diff = A_fast_dense - A_dense
            max_abs_matrix_diff = maximum(abs, matrix_diff)
            rel_matrix_diff = norm(matrix_diff) / max(norm(A_dense), eps(Float64))
            u_fast = fast_A \ sys.rhs
            u_ref = A_dense \ sys.rhs
            relerr = norm(u_fast - u_ref, Inf) / max(1.0, norm(u_ref, Inf))
            rhs_norm = max(norm(sys.rhs), eps(Float64))
            residual_on_original_A = norm(sys.A * u_fast - sys.rhs) / rhs_norm
            residual_on_fast_A = norm(fast_A * u_fast - sys.rhs) / rhs_norm
            matrix_abs_limit = rtol * max(1.0, matrix_maxabs(A_dense))
            if !(relerr < rtol &&
                 max_abs_matrix_diff <= matrix_abs_limit &&
                 rel_matrix_diff < rtol &&
                 residual_on_original_A < rtol &&
                 residual_on_fast_A < rtol)
                error(
                    "FastAlmostBanded consistency failed for " *
                    "method=$(spec.method), basis=$(spec.basis), N=$N: " *
                    "solution_relerr=$relerr, " *
                    "max_abs_matrix_diff=$max_abs_matrix_diff, " *
                    "rel_matrix_diff=$rel_matrix_diff, " *
                    "residual_on_original_A=$residual_on_original_A, " *
                    "residual_on_fast_A=$residual_on_fast_A"
                )
            end
        end
    end
    return true
end

function run_method_case(spec, N::Int; solver::Symbol=:fastalmostbanded)
    spec.runner === :cheb && return run_voltmat_case(N, :cheb; solver=solver)
    spec.runner === :leg && return run_voltmat_case(N, :leg; solver=solver)
    spec.runner === :go && return run_gutleb_olver_case(N; kerneldepth=KERNELDEPTH, solver=solver)
    error("unknown method runner $(spec.runner)")
end

function failed_case_row(spec, N::Int, err)
    return (
        method=spec.method,
        basis=spec.basis,
        N=N,
        kerneldepth=spec.kerneldepth,
        C_rows=spec.C_rows,
        C_cols=spec.C_cols,
        target_kernel_bandwidth=spec.runner === :go ? KERNELDEPTH : TARGET_KERNEL_BANDWIDTH,
        coeff_precompute_time=NaN,
        kernel_coeff_time=NaN,
        rhs_time=NaN,
        volterra_matrix_construct_time=NaN,
        final_matrix_assembly_time=NaN,
        solve_time=NaN,
        elapsed_time=NaN,
        max_abs_error=NaN,
        residual_relative=NaN,
        body_residual_relative=NaN,
        raw_matrix_type="",
        raw_matrix_nnz="",
        raw_matrix_lband="",
        raw_matrix_uband="",
        raw_structure="",
        raw_global_lband="",
        raw_global_uband="",
        final_matrix_type="",
        final_matrix_nnz="",
        final_matrix_lband="",
        final_matrix_uband="",
        final_structure="",
        final_global_lband="",
        final_global_uband="",
        solver="fastalmostbanded",
        solver_backend="failed",
        fast_matrix_type="",
        fast_matrix_lband="",
        fast_matrix_uband="",
        fast_matrix_fill_rows="",
        status="failed",
        error_message=err isa AbstractString ? err : sprint(showerror, err),
    )
end

function csv_escape(x)
    if x === missing || x === nothing
        return ""
    end
    s = string(x)
    if occursin('"', s) || occursin(',', s) || occursin('\n', s) || occursin('\r', s)
        return "\"" * replace(s, "\"" => "\"\"") * "\""
    end
    return s
end

function write_csv_rows(rows, path::AbstractString, header)
    tmp_path = path * ".tmp"
    open(tmp_path, "w") do io
        println(io, join(string.(header), ","))
        for row in rows
            println(io, join((csv_escape(getproperty(row, h)) for h in header), ","))
        end
    end
    last_error = nothing
    for _ in 1:20
        try
            mv(tmp_path, path; force=true)
            return path
        catch err
            last_error = err
            sleep(0.25)
        end
    end
    @warn "Could not replace locked CSV; leaving temporary output in place" path tmp_path exception=last_error
    return tmp_path
end

function stats_or_nan(values::Vector{Float64})
    isempty(values) && return (min=NaN, median=NaN, mean=NaN, std=NaN)
    return (
        min=minimum(values),
        median=median(values),
        mean=mean(values),
        std=length(values) == 1 ? 0.0 : std(values),
    )
end
