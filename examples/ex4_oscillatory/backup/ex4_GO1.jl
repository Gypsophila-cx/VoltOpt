# ex4_GO.jl
#
# Gutleb--Olver solve for the first-order VIDE
#
#   kappa^{-1} u'(x) + u(x)
#   + int_0^x cos(omega*x*y) u(y) dy = f(x),
#   x in [0,1], u(0)=1.
#
# Exact solution:
#
#   u(x) = cos(kappa*x), kappa = 1000, omega = 50.
#
# This implementation follows the VIDE workflow in
# vide_low_order_compare_common.jl:
#
#   Vraw
#   -> apply_reflect_lowering_strictband
#   -> apply_conversion_00_to_12_strictband
#   -> assemble_semibanded_system
#   -> FastAlmostBandedMatrices.AlmostBandedMatrix
#
# It does NOT use build_VIDE_LHS_sec522_strictband.
# It does NOT call VoltMat.
#
# Output:
#
#   data/ex4_GO_error.csv
#   data/ex4_GO_time.csv

using LinearAlgebra
using Printf
using SparseArrays
using Pkg
using FastAlmostBandedMatrices

# ============================================================
# Locate project and activate Gutleb--Olver environment
# ============================================================

function find_project_root(start_dir::AbstractString)
    dir = abspath(start_dir)
    while true
        if isfile(joinpath(dir, "Project.toml")) &&
           isdir(joinpath(dir, "others", "GutlebOlver"))
            return dir
        end
        parent = dirname(dir)
        parent == dir && error("Could not locate VoltOpt project root from $start_dir")
        dir = parent
    end
end

project_root = find_project_root(@__DIR__)
go_project = joinpath(project_root, "others", "GutlebOlver")

Pkg.activate(go_project; io=devnull)

using ApproxFun
using BandedMatrices
using GutlebOlver

# ============================================================
# Parameters
# ============================================================

omega = 50.0
kappa = 1000.0

N_min = 10
N_max = 2000
num_N = 100

# Here N means number of coefficients, as in vide_low_order_compare_common.jl.
Ns_full = unique(round.(Int, range(N_min, N_max; length=num_N)))

quick_mode = lowercase(get(ENV, "EX4_QUICK", "0")) in ("1", "true", "yes", "on")

function parse_int_list_env(name::AbstractString)
    raw = strip(get(ENV, name, ""))
    isempty(raw) && return nothing
    return [parse(Int, strip(s)) for s in split(raw, ",") if !isempty(strip(s))]
end

Ns_custom = parse_int_list_env("EX4_N_LIST")
Ns = Ns_custom === nothing ? (quick_mode ? [10, 50, 100] : Ns_full) : Ns_custom

kerneldepth = parse(Int, get(ENV, "EX4_KERNELDEPTH", "15"))
rhs_fixed_N = parse(Int, get(ENV, "EX4_RHS_FIXED_N", "8192"))
grid_M = parse(Int, get(ENV, "EX4_GRID_M", "20000"))

solution_space = Jacobi(0, 1, 0..1)
output_space = Jacobi(1, 2, 0..1)

data_dir = joinpath(@__DIR__, "data")
mkpath(data_dir)

error_csv = joinpath(data_dir, "ex4_GO_error.csv")
time_csv  = joinpath(data_dir, "ex4_GO_time.csv")

# ============================================================
# Exact solution, kernel, and RHS
# ============================================================

function sinc_stable(z::Float64)
    az = abs(z)
    if az < 1.0e-4
        z2 = z*z
        return 1.0 - z2/6.0 + z2*z2/120.0 - z2*z2*z2/5040.0
    else
        return sin(z)/z
    end
end

function exact_integral(x::Real; omega=omega, kappa=kappa)
    xf = Float64(x)
    z_minus = (omega*xf - kappa)*xf
    z_plus  = (omega*xf + kappa)*xf
    return 0.5*xf*(sinc_stable(z_minus) + sinc_stable(z_plus))
end

exact_solution(x::Real) = cos(kappa*Float64(x))

function rhs_fun(x::Real)
    xf = Float64(x)
    return -sin(kappa*xf) + cos(kappa*xf) + exact_integral(xf)
end

function kernel_go(x::Real, y::Real)
    return cos(omega*Float64(x)*Float64(y))
end

# ============================================================
# Matrix diagnostics and fast almost-banded utilities
# ============================================================

matrix_type_name(A) = A isa Matrix ? "Matrix" : string(nameof(typeof(A)))

function matrix_maxabs(A)
    if A isa SparseMatrixCSC
        vals = nonzeros(A)
        return isempty(vals) ? 0.0 : maximum(abs, vals)
    end

    try
        l, u = bandwidths(A)
        n, m = size(A)
        mx = 0.0
        @inbounds for j in 1:m
            i1 = max(1, j - u)
            i2 = min(n, j + l)
            for i in i1:i2
                v = Float64(A[i, j])
                if isnan(v)
                    return NaN
                elseif isinf(v)
                    mx = Inf
                else
                    mx = max(mx, abs(v))
                end
            end
        end
        return mx
    catch
    end

    mx = 0.0
    for j in axes(A, 2), i in axes(A, 1)
        v = Float64(A[i, j])
        if isnan(v)
            return NaN
        elseif isinf(v)
            mx = Inf
        else
            mx = max(mx, abs(v))
        end
    end
    return mx
end

function local_safe_nnz(A)
    A isa SparseMatrixCSC && return nnz(A)
    try
        return GutlebOlver.safe_nnz(A)
    catch
    end
    count = 0
    try
        l, u = bandwidths(A)
        n, m = size(A)
        @inbounds for j in 1:m
            i1 = max(1, j - u)
            i2 = min(n, j + l)
            for i in i1:i2
                count += !iszero(A[i, j])
            end
        end
        return count
    catch
    end
    for j in axes(A, 2), i in axes(A, 1)
        count += !iszero(A[i, j])
    end
    return count
end

function default_band_tol(A)
    scale = matrix_maxabs(A)
    scale == 0.0 && return 0.0
    return max(100.0 * eps(Float64) * scale, 1.0e-14 * scale)
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
        try
            l, u = bandwidths(A)
            n, m = size(A)
            @inbounds for j in 1:m
                i1 = max(first(rows), max(1, j - u))
                i2 = min(last(rows), min(n, j + l))
                for i in i1:i2
                    if abs(A[i, j]) > tol
                        lband = max(lband, Int(i - j))
                        uband = max(uband, Int(j - i))
                        found = true
                    end
                end
            end
        catch
            @inbounds for j in axes(A, 2), i in rows
                if abs(A[i, j]) > tol
                    lband = max(lband, Int(i - j))
                    uband = max(uband, Int(j - i))
                    found = true
                end
            end
        end
    end

    return found ? (lband, uband) : (0, 0)
end

function top_dense_plus_band_diagnostics(A; top_dense_rows::Int,
                                         structure::String="top_dense_plus_banded")
    m, _ = size(A)
    top = min(top_dense_rows, m)
    below_top_rows = top + 1 <= m ? ((top + 1):m) : (1:0)

    global_lband, global_uband = pattern_bandwidths(A; band_tol=0.0)
    band_lband, band_uband = pattern_bandwidths(A; row_range=below_top_rows, band_tol=0.0)

    return (
        structure=structure,
        top_dense_rows=top,
        lower_bandwidth_below_top=band_lband,
        global_lower_bandwidth=global_lband,
        global_upper_bandwidth=global_uband,
        structural_lband=band_lband,
        structural_uband=global_uband,
        band_lband=band_lband,
        band_uband=band_uband,
    )
end

function generic_matrix_diagnostics(A; structure::String="")
    try
        l, u = bandwidths(A)
        return (
            structure=isempty(structure) ? matrix_type_name(A) : structure,
            top_dense_rows="",
            lower_bandwidth_below_top="",
            global_lower_bandwidth=Int(l),
            global_upper_bandwidth=Int(u),
            structural_lband=Int(l),
            structural_uband=Int(u),
            band_lband=Int(l),
            band_uband=Int(u),
        )
    catch
        l, u = pattern_bandwidths(A)
        return (
            structure=isempty(structure) ? matrix_type_name(A) : structure,
            top_dense_rows="",
            lower_bandwidth_below_top="",
            global_lower_bandwidth=l,
            global_upper_bandwidth=u,
            structural_lband=l,
            structural_uband=u,
            band_lband=l,
            band_uband=u,
        )
    end
end

function to_fast_almost_banded(A; top_dense_rows::Integer=0,
                               lband::Integer=0,
                               uband::Integer=0)
    m, n = size(A)
    m == n || throw(ArgumentError("Almost-banded solve expects a square matrix, got $(size(A))"))

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
            solver_backend="fastalmostbanded",
            fast_matrix_type=matrix_type_name(bands),
            fast_matrix_lband=band_l,
            fast_matrix_uband=band_u,
            fast_matrix_fill_rows=0,
            fallback_reason="",
        )
    end

    fill = zeros(Float64, fill_rows, n)
    @inbounds for j in 1:n, i in 1:fill_rows
        fill[i, j] = A[i, j]
    end

    fast_A = FastAlmostBandedMatrices.AlmostBandedMatrix(bands, fill)

    return fast_A, (
        solver_backend="fastalmostbanded",
        fast_matrix_type=matrix_type_name(fast_A),
        fast_matrix_lband=band_l,
        fast_matrix_uband=band_u,
        fast_matrix_fill_rows=fill_rows,
        fallback_reason="",
    )
end

function solve_vide_system(A, rhs; top_dense_rows::Integer,
                           lband::Integer,
                           uband::Integer)
    fast_A, fast_info = to_fast_almost_banded(
        A;
        top_dense_rows=top_dense_rows,
        lband=lband,
        uband=uband,
    )

    local sol
    solve_time = @elapsed begin
        sol = fast_A \ rhs
    end

    return sol, merge(fast_info, (solver="fastalmostbanded", solve_time=solve_time))
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

# ============================================================
# RHS projection
# ============================================================

function build_rhs_coefficients_fixed(Nfixed::Int)
    try
        rhs_coeff, rhs_info = build_go_rhs_coefficients(
            rhs_fun,
            Nfixed;
            alpha=1,
            beta=2,
            method=:fixed_gauss_jacobi,
            projection_tol=1.0e-12,
        )
        return Vector{Float64}(rhs_coeff), rhs_info
    catch
        rhs_coeff, rhs_info = build_go_rhs_coefficients(
            rhs_fun,
            Nfixed;
            alpha=1,
            beta=2,
            method=:fixed_gauss_jacobi,
        )
        return Vector{Float64}(rhs_coeff), rhs_info
    end
end

function build_rhs_from_cache(N::Int, rhs_cache::AbstractVector{Float64})
    N >= 2 || error("N must be at least 2")
    N - 1 <= length(rhs_cache) ||
        error("rhs_fixed_N=$(length(rhs_cache)) is too small for N=$N")

    final_rhs = zeros(Float64, N)
    final_rhs[1] = 1.0
    final_rhs[2:N] .= rhs_cache[1:(N - 1)]

    return final_rhs
end

# ============================================================
# GO VIDE system construction: new uploaded-file path
# ============================================================

function build_kernel_coefficients(kerneldepth::Int)
    plan = precompute_triangular_kernel_plan(kerneldepth, true)
    kernel_coeffs = triangular_kernel_coeffs_from_plan(plan, kernel_go)
    return plan, kernel_coeffs
end

function build_gutleb_olver_system(N::Int, kernel_coeffs, rhs_cache)
    N >= 2 || error("Gutleb--Olver VIDE solve requires N >= 2")

    local Vraw
    local Vreflect_lower
    local Vconv
    local D
    local Csolution
    local body
    local A
    local final_rhs

    volterra_time = @elapsed begin
        Vraw = triVolterraFullKernelOpP01_strictband_from_coeffs(kernel_coeffs, N, true)
    end

    final_time = @elapsed begin
        Vreflect_lower = apply_reflect_lowering_strictband(Vraw, N)
        Vconv = apply_conversion_00_to_12_strictband(Vreflect_lower, N)

        D = build_derivative_P01_to_P12(N)
        Csolution = Conversion(solution_space, output_space)[1:N, 1:N]

        # Equation:
        #
        #   kappa^{-1} u' + u + V_K u = f.
        #
        # Uploaded warmup file uses D + Csolution - Vconv for
        # u_s + u - V_K u = f.
        #
        # Here the sign is plus, and the derivative coefficient is 1/kappa.
        body = (1.0/kappa) * D + Csolution + Vconv

        evaluation_row = build_initial_evaluation_row(N)
        A = assemble_semibanded_system(body, N; evaluation_row=evaluation_row)

        final_rhs = build_rhs_from_cache(N, rhs_cache)
    end

    raw_diag = generic_matrix_diagnostics(Vraw; structure="strict_banded")
    final_diag = top_dense_plus_band_diagnostics(
        A;
        top_dense_rows=1,
        structure="semibanded_evaluation_row",
    )

    return (
        Vraw=Vraw,
        Vreflect_lower=Vreflect_lower,
        Vconv=Vconv,
        D=D,
        Csolution=Csolution,
        body=body,
        A=A,
        rhs=final_rhs,
        volterra_time=volterra_time,
        final_time=final_time,
        raw_diag=raw_diag,
        final_diag=final_diag,
        solver_top_dense_rows=final_diag.top_dense_rows,
        solver_lband=final_diag.band_lband,
        solver_uband=final_diag.band_uband,
    )
end

# ============================================================
# Error
# ============================================================

function max_error_go(coeffs::AbstractVector)
    ufun = Fun(solution_space, coeffs)
    err = 0.0
    for x in range(0.0, 1.0; length=grid_M+1)
        err = max(err, abs(Float64(ufun(x)) - exact_solution(x)))
    end
    return err
end

# ============================================================
# Run one case
# ============================================================

function run_gutleb_olver_case(N::Int, kernel_coeffs, rhs_cache;
                               kernel_coeff_time::Float64,
                               rhs_precompute_time::Float64)

    start_time_ns = time_ns()

    sys = build_gutleb_olver_system(N, kernel_coeffs, rhs_cache)

    sol, solve_info = solve_vide_system(
        sys.A,
        sys.rhs;
        top_dense_rows=sys.solver_top_dense_rows,
        lband=sys.solver_lband,
        uband=sys.solver_uband,
    )

    residual = sys.A * sol - sys.rhs
    residual_relative, body_residual_relative = residual_ratios(residual, sys.rhs, 2:N)

    err = max_error_go(sol)

    return (
        method="GutlebOlver",
        basis="shifted Jacobi P^(1,0)",
        N=N,
        kerneldepth=kerneldepth,
        omega=omega,
        kappa=kappa,
        rhs_fixed_N=rhs_fixed_N,
        kernel_coeff_time=kernel_coeff_time,
        rhs_precompute_time=rhs_precompute_time,
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
        maxabs_Vraw=matrix_maxabs(sys.Vraw),
        maxabs_Vreflect_lower=matrix_maxabs(sys.Vreflect_lower),
        maxabs_Vconv=matrix_maxabs(sys.Vconv),
        maxabs_D=matrix_maxabs(sys.D),
        maxabs_C=matrix_maxabs(sys.Csolution),
        maxabs_body=matrix_maxabs(sys.body),
        maxabs_A=matrix_maxabs(sys.A),
        solver=solve_info.solver,
        solver_backend=solve_info.solver_backend,
        fast_matrix_type=solve_info.fast_matrix_type,
        fast_matrix_lband=solve_info.fast_matrix_lband,
        fast_matrix_uband=solve_info.fast_matrix_uband,
        fast_matrix_fill_rows=solve_info.fast_matrix_fill_rows,
        status="success",
        error_message="",
    )
end

function failed_case_row(N::Int, err)
    return (
        method="GutlebOlver",
        basis="shifted Jacobi P^(1,0)",
        N=N,
        kerneldepth=kerneldepth,
        omega=omega,
        kappa=kappa,
        rhs_fixed_N=rhs_fixed_N,
        kernel_coeff_time=NaN,
        rhs_precompute_time=NaN,
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
        maxabs_Vraw=NaN,
        maxabs_Vreflect_lower=NaN,
        maxabs_Vconv=NaN,
        maxabs_D=NaN,
        maxabs_C=NaN,
        maxabs_body=NaN,
        maxabs_A=NaN,
        solver="fastalmostbanded",
        solver_backend="failed",
        fast_matrix_type="",
        fast_matrix_lband="",
        fast_matrix_uband="",
        fast_matrix_fill_rows="",
        status="failed",
        error_message=sprint(showerror, err),
    )
end

# ============================================================
# CSV
# ============================================================

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

error_header = (
    :method,
    :basis,
    :N,
    :omega,
    :kappa,
    :kerneldepth,
    :rhs_fixed_N,
    :max_abs_error,
    :residual_relative,
    :body_residual_relative,
    :maxabs_Vraw,
    :maxabs_Vconv,
    :maxabs_body,
    :maxabs_A,
    :solver_backend,
    :status,
    :error_message,
)

time_header = (
    :method,
    :basis,
    :N,
    :omega,
    :kappa,
    :kerneldepth,
    :rhs_fixed_N,
    :kernel_coeff_time,
    :rhs_precompute_time,
    :volterra_matrix_construct_time,
    :final_matrix_assembly_time,
    :solve_time,
    :elapsed_time,
    :raw_matrix_type,
    :raw_matrix_nnz,
    :raw_matrix_lband,
    :raw_matrix_uband,
    :raw_structure,
    :raw_global_lband,
    :raw_global_uband,
    :final_matrix_type,
    :final_matrix_nnz,
    :final_matrix_lband,
    :final_matrix_uband,
    :final_structure,
    :final_global_lband,
    :final_global_uband,
    :maxabs_Vraw,
    :maxabs_Vreflect_lower,
    :maxabs_Vconv,
    :maxabs_D,
    :maxabs_C,
    :maxabs_body,
    :maxabs_A,
    :solver,
    :solver_backend,
    :fast_matrix_type,
    :fast_matrix_lband,
    :fast_matrix_uband,
    :fast_matrix_fill_rows,
    :status,
    :error_message,
)

function write_csv_rows(rows, path::AbstractString, header)
    open(path, "w") do io
        println(io, join(string.(header), ","))
        for row in rows
            println(io, join((csv_escape(getproperty(row, h)) for h in header), ","))
        end
    end
end

function write_error_csv(rows)
    write_csv_rows(rows, error_csv, error_header)
end

function write_time_csv(rows)
    write_csv_rows(rows, time_csv, time_header)
end

function print_row(row)
    println("  N                         = ", row.N)
    @printf("  err_inf                   = %.6e\n", row.max_abs_error)
    @printf("  residual_relative         = %.6e\n", row.residual_relative)
    @printf("  body_residual_relative    = %.6e\n", row.body_residual_relative)
    @printf("  maxabs_Vraw               = %.6e\n", row.maxabs_Vraw)
    @printf("  maxabs_Vconv              = %.6e\n", row.maxabs_Vconv)
    @printf("  maxabs_A                  = %.6e\n", row.maxabs_A)
    @printf("  volterra_construct_time   = %.6e\n", row.volterra_matrix_construct_time)
    @printf("  final_assembly_time       = %.6e\n", row.final_matrix_assembly_time)
    @printf("  solve_time                = %.6e\n", row.solve_time)
    println("  solver_backend            = ", row.solver_backend)
    println("  fast_matrix_type          = ", row.fast_matrix_type)
    println("  status                    = ", row.status)
    if row.status != "success"
        println("  error                     = ", row.error_message)
    end
end

# ============================================================
# Main
# ============================================================

println("============================================================")
println("ex4 Gutleb--Olver first-order VIDE solve")
println("Path: Vraw -> reflect/lowering -> conversion_00_to_12 -> tau -> fast almost-banded")
println("Does not use build_VIDE_LHS_sec522_strictband")
println("Equation: kappa^{-1} u' + u + V_K u = f")
println("K(x,y) = cos(omega*x*y)")
println("project_root       = ", project_root)
println("go_project         = ", go_project)
println("omega              = ", omega)
println("kappa              = ", kappa)
println("kerneldepth        = ", kerneldepth)
println("rhs_fixed_N        = ", rhs_fixed_N)
println("grid_M             = ", grid_M)
println("quick_mode         = ", quick_mode)
println("Ns count           = ", length(Ns))
println("data_dir           = ", data_dir)
println("============================================================")

println()
println("Precomputing triangular kernel coefficients ...")
local kernel_plan
local kernel_coeffs
kernel_coeff_time = @elapsed begin
    kernel_plan, kernel_coeffs = build_kernel_coefficients(kerneldepth)
end
@printf("kernel_coeff_time = %.6e sec\n", kernel_coeff_time)

println()
println("Precomputing RHS coefficients ...")
local rhs_cache
local rhs_info
rhs_precompute_time = @elapsed begin
    rhs_cache, rhs_info = build_rhs_coefficients_fixed(rhs_fixed_N)
end
@printf("rhs_precompute_time = %.6e sec\n", rhs_precompute_time)

rows = []

for (idx, N) in enumerate(Ns)
    println()
    @printf("Solving %3d / %3d: N = %d\n", idx, length(Ns), N)

    row = try
        run_gutleb_olver_case(
            N,
            kernel_coeffs,
            rhs_cache;
            kernel_coeff_time=kernel_coeff_time,
            rhs_precompute_time=rhs_precompute_time,
        )
    catch err
        failed_case_row(N, err)
    end

    push!(rows, row)
    print_row(row)

    write_error_csv(rows)
    write_time_csv(rows)
end

println()
println("============================================================")
println("Finished.")
println("Wrote:")
println("  ", error_csv)
println("  ", time_csv)
println("============================================================")