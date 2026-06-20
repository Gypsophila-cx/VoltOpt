using Pkg

const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const GUTLEB_OLVER_PROJECT = joinpath(ROOT, "others", "GutlebOlver")
const OUTPUT_CSV = joinpath(@__DIR__, "data", "warmup_gutleb_olver_results.csv")
const N_LIST = (16, 32, 64, 128)
const KERNELDEPTH_LIST = (4, 8, 12, 16, 20)
const SOLUTION_GRID_SIZE = 2001

Pkg.activate(GUTLEB_OLVER_PROJECT; io=devnull)

using ApproxFun
using BandedMatrices
using GutlebOlver
using LinearAlgebra
using Printf

const SOLUTION_SPACE = Jacobi(0, 1, 0..1)
const OUTPUT_SPACE = Jacobi(1, 2, 0..1)

const CSV_COLUMNS = (
    :problem,
    :equation_type,
    :N,
    :kerneldepth,
    :rhs_validation_error,
    :rhs_tail_norm,
    :rhs_status,
    :rhs_basis,
    :rhs_method,
    :tau_residual,
    :residual_relative,
    :body_residual_relative,
    :max_abs_error,
    :raw_type,
    :raw_nnz,
    :raw_lower_bw,
    :raw_upper_bw,
    :body_type,
    :body_nnz,
    :body_lower_bw,
    :body_upper_bw,
    :final_type,
    :final_nnz,
    :final_lower_bw,
    :final_upper_bw,
    :final_density,
    :kernel_coeff_time,
    :operator_time,
    :rhs_time,
    :assembly_time,
    :solve_time,
    :evaluation_time,
    :status,
    :notes,
    :error_message,
)

original_kernel(x, t) = ((x + 1.0) * (x + 2.0) / 4.0) * exp(((t + 1.0) * (x - t)) / 4.0)
original_exact(x) = exp(((x + 1.0)^2) / 4.0)
original_rhs(x) = x + 2.0

to_original_x(s) = 2.0 * s - 1.0
warmup_exact(s) = original_exact(to_original_x(s))
warmup_rhs(s) = original_rhs(to_original_x(s))
warmup_kernel(s, r) = 2.0 * original_kernel(to_original_x(s), to_original_x(r))

matrix_type(A) = string(nameof(typeof(A)))
bandwidth_part(bw, index::Integer) = bw === missing ? missing : Tuple(bw)[index]

function pad_to_length(c, n::Integer)
    out = zeros(Float64, Int(n))
    m = min(length(c), Int(n))
    if m > 0
        out[1:m] .= c[1:m]
    end
    return out
end

function error_text(err)
    io = IOBuffer()
    showerror(io, err)
    return String(take!(io))
end

function csv_escape(value)
    text = value === missing ? "" : string(value)
    if occursin(',', text) || occursin('"', text) || occursin('\n', text) || occursin('\r', text)
        return "\"" * replace(text, "\"" => "\"\"") * "\""
    end
    return text
end

function write_header(path::AbstractString)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(string.(CSV_COLUMNS), ","))
    end
    return path
end

function append_csv_row(path::AbstractString, row::NamedTuple)
    open(path, "a") do io
        println(io, join((csv_escape(getproperty(row, col)) for col in CSV_COLUMNS), ","))
    end
    return path
end

function failed_row(N::Integer, kerneldepth::Integer, err)
    return (
        problem="warmup_vide_low_order",
        equation_type="mapped_first_order_vide",
        N=Int(N),
        kerneldepth=Int(kerneldepth),
        rhs_validation_error=missing,
        rhs_tail_norm=missing,
        rhs_status=missing,
        rhs_basis=missing,
        rhs_method=missing,
        tau_residual=missing,
        residual_relative=missing,
        body_residual_relative=missing,
        max_abs_error=missing,
        raw_type=missing,
        raw_nnz=missing,
        raw_lower_bw=missing,
        raw_upper_bw=missing,
        body_type=missing,
        body_nnz=missing,
        body_lower_bw=missing,
        body_upper_bw=missing,
        final_type=missing,
        final_nnz=missing,
        final_lower_bw=missing,
        final_upper_bw=missing,
        final_density=missing,
        kernel_coeff_time=missing,
        operator_time=missing,
        rhs_time=missing,
        assembly_time=missing,
        solve_time=missing,
        evaluation_time=missing,
        status="failed",
        notes="adapter failed before completing sparse solve",
        error_message=error_text(err),
    )
end

function assemble_warmup_body(Vraw, n::Integer)
    Vreflect_lower = apply_reflect_lowering_strictband(Vraw, n)
    Vconv = apply_conversion_00_to_12_strictband(Vreflect_lower, n)
    D = build_derivative_P01_to_P12(n)
    Csolution = Conversion(SOLUTION_SPACE, OUTPUT_SPACE)[1:n, 1:n]
    return D + Csolution - Vconv
end

function solve_one(N::Integer, kerneldepth::Integer)
    n = Int(N)
    kerneldepth_int = Int(kerneldepth)
    n >= 2 || throw(ArgumentError("N must be at least 2"))

    local coeffs
    local Vraw
    local rhs_coeff
    local rhs_info
    local body
    local final_system
    local solution_coeff
    local max_abs_error

    kernel_coeff_time = @elapsed begin
        plan = precompute_triangular_kernel_plan(kerneldepth_int, true)
        coeffs = triangular_kernel_coeffs_from_plan(plan, warmup_kernel)
    end

    operator_time = @elapsed begin
        Vraw = triVolterraFullKernelOpP01_strictband_from_coeffs(coeffs, n, true)
    end

    rhs_time = @elapsed begin
        rhs_coeff, rhs_info = build_go_rhs_coefficients(
            warmup_rhs,
            n;
            alpha=1,
            beta=2,
            method=:fixed_gauss_jacobi,
            projection_tol=1.0e-12,
        )
    end

    exact_coeff = pad_to_length(Fun(warmup_exact, SOLUTION_SPACE, n).coefficients, n)

    final_rhs = zeros(Float64, n)
    final_rhs[1] = warmup_exact(0.0)
    final_rhs[2:n] .= rhs_coeff[1:(n - 1)]

    assembly_time = @elapsed begin
        body = assemble_warmup_body(Vraw, n)
        final_system = assemble_semibanded_system(body, n; evaluation_row=build_initial_evaluation_row(n))
    end

    residual = final_system * exact_coeff - final_rhs
    body_residual = residual[2:n]
    body_rhs = final_rhs[2:n]
    residual_relative = norm(residual) / norm(final_rhs)
    body_residual_relative = norm(body_residual) / norm(body_rhs)
    tau_residual = abs(residual[1])

    solve_time = @elapsed begin
        solution_coeff = final_system \ final_rhs
    end

    evaluation_time = @elapsed begin
        u_num = Fun(SOLUTION_SPACE, solution_coeff)
        grid = range(0.0, 1.0; length=SOLUTION_GRID_SIZE)
        max_abs_error = maximum(abs(u_num(s) - warmup_exact(s)) for s in grid)
    end

    raw_bw = safe_bandwidth(Vraw)
    body_bw = safe_bandwidth(body)
    final_bw = safe_bandwidth(final_system)
    passed = rhs_info.status == "pass" &&
        rhs_info.validation_error <= 1.0e-10 &&
        body_residual_relative <= 1.0e-7 &&
        max_abs_error <= 1.0e-7

    return (
        problem="warmup_vide_low_order",
        equation_type="mapped_first_order_vide",
        N=n,
        kerneldepth=kerneldepth_int,
        rhs_validation_error=rhs_info.validation_error,
        rhs_tail_norm=rhs_info.tail_norm,
        rhs_status=rhs_info.status,
        rhs_basis=rhs_info.basis,
        rhs_method=rhs_info.method,
        tau_residual=tau_residual,
        residual_relative=residual_relative,
        body_residual_relative=body_residual_relative,
        max_abs_error=max_abs_error,
        raw_type=matrix_type(Vraw),
        raw_nnz=safe_nnz(Vraw),
        raw_lower_bw=bandwidth_part(raw_bw, 1),
        raw_upper_bw=bandwidth_part(raw_bw, 2),
        body_type=matrix_type(body),
        body_nnz=safe_nnz(body),
        body_lower_bw=bandwidth_part(body_bw, 1),
        body_upper_bw=bandwidth_part(body_bw, 2),
        final_type=matrix_type(final_system),
        final_nnz=safe_nnz(final_system),
        final_lower_bw=bandwidth_part(final_bw, 1),
        final_upper_bw=bandwidth_part(final_bw, 2),
        final_density=safe_density(final_system),
        kernel_coeff_time=kernel_coeff_time,
        operator_time=operator_time,
        rhs_time=rhs_time,
        assembly_time=assembly_time,
        solve_time=solve_time,
        evaluation_time=evaluation_time,
        status=passed ? "pass" : "check",
        notes=passed ?
            "mapped warmup VIDE solved by GutlebOlver sparse adapter" :
            "completed solve but one acceptance diagnostic is above threshold",
        error_message="",
    )
end

function print_row(row)
    println(
        "N=", row.N,
        " kerneldepth=", row.kerneldepth,
        " rhs_validation_error=", row.rhs_validation_error,
        " residual_relative=", row.residual_relative,
        " body_residual_relative=", row.body_residual_relative,
        " max_abs_error=", row.max_abs_error,
        " raw=", row.raw_type, " ", row.raw_nnz, "/(", row.raw_lower_bw, ",", row.raw_upper_bw, ")",
        " final=", row.final_type, " ", row.final_nnz, "/(", row.final_lower_bw, ",", row.final_upper_bw, ")",
        " status=", row.status,
    )
end

function main()
    println("GutlebOlver warmup VIDE adapter")
    println("project = ", GUTLEB_OLVER_PROJECT)
    println("csv = ", OUTPUT_CSV)
    println("original: 2u_x(x) + u(x) - int_{-1}^x K(x,t)u(t)dt = x + 2, u(-1)=1")
    println("mapped: u_s(s) + u(s) - int_0^s K_go(s,r)u(r)dr = 2s + 1, u(0)=1")

    write_header(OUTPUT_CSV)
    rows = NamedTuple[]
    for N in N_LIST
        for kerneldepth in KERNELDEPTH_LIST
            row = try
                solve_one(N, kerneldepth)
            catch err
                failed_row(N, kerneldepth, err)
            end
            append_csv_row(OUTPUT_CSV, row)
            push!(rows, row)
            print_row(row)
        end
    end

    println("wrote csv = ", OUTPUT_CSV)
    return rows
end

main()
