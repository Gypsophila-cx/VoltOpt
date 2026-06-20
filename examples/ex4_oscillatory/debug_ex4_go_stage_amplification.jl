# debug_ex4_go_stage_amplification.jl
#
# Diagnose where the Gutleb--Olver strict-band Volterra construction amplifies
# an otherwise resolved triangular kernel expansion for ex4.

using LinearAlgebra
using Printf
using SparseArrays

ex4_dir = @__DIR__
project_root = normpath(joinpath(ex4_dir, "..", ".."))
go_project = joinpath(project_root, "others", "GutlebOlver")

pushfirst!(LOAD_PATH, project_root)
pushfirst!(LOAD_PATH, go_project)

using ApproxFun
using BandedMatrices
using BlockArrays
using GutlebOlver

omega_default = 50.0
kappa = 1000.0
go_flip = true

function parse_int_list_env(name::AbstractString, default::Vector{Int})
    raw = strip(get(ENV, name, ""))
    isempty(raw) && return default
    return [parse(Int, strip(x)) for x in split(raw, ",") if !isempty(strip(x))]
end

function parse_float_list_env(name::AbstractString, default::Vector{Float64})
    raw = strip(get(ENV, name, ""))
    isempty(raw) && return default
    return [parse(Float64, strip(x)) for x in split(raw, ",") if !isempty(strip(x))]
end

kerneldepth = parse(Int, get(ENV, "EX4_KERNELDEPTH", "96"))
Ns = parse_int_list_env("EX4_STAGE_NLIST", [100, 500, 1000, 2000])
omega_scan_values = parse_float_list_env("EX4_OMEGA_SCAN", [1.0, 5.0, 10.0, 20.0, 50.0])

solution_space = Jacobi(0, 1, 0..1)
body_space = Jacobi(1, 2, 0..1)

data_dir = joinpath(ex4_dir, "data")
mkpath(data_dir)

coeff_csv = joinpath(data_dir, "ex4_GO_kernel_coeff_diagnostics.csv")
stage_csv = joinpath(data_dir, "ex4_GO_stage_amplification.csv")
omega_csv = joinpath(data_dir, "ex4_GO_omega_stage_scan.csv")

kernel_go(x, y; omega=omega_default) = cos(omega * Float64(x) * Float64(y))
kernel_const(x, y) = 1.0

# ------------------------------------------------------------
# CSV helpers
# ------------------------------------------------------------

function csv_escape(x)
    s = x === missing ? "" : string(x)
    if occursin(",", s) || occursin("\"", s) || occursin("\n", s) || occursin("\r", s)
        return "\"" * replace(s, "\"" => "\"\"") * "\""
    end
    return s
end

csv_value(x) = x isa AbstractFloat ? @sprintf("%.17e", x) : csv_escape(x)

function row_value(row, col::Symbol)
    return hasproperty(row, col) ? getproperty(row, col) : missing
end

function write_rows_csv(path::AbstractString, columns, rows)
    open(path, "w") do io
        println(io, join(string.(columns), ","))
        for row in rows
            println(io, join((csv_value(row_value(row, col)) for col in columns), ","))
        end
    end
end

# ------------------------------------------------------------
# Numeric collection for triangular kernel coefficients
# ------------------------------------------------------------

function collect_numbers!(out::Vector{Float64}, x)
    if x isa Number
        push!(out, Float64(x))
    elseif x isa AbstractArray{<:Number}
        for v in x
            push!(out, Float64(v))
        end
    elseif x isa NamedTuple
        for v in values(x)
            collect_numbers!(out, v)
        end
    elseif x isa Tuple
        for v in x
            collect_numbers!(out, v)
        end
    end
    return out
end

function coefficient_values(coeffs)
    vals = Float64[]
    try
        cfs = getfield(coeffs, :cfs)
        collect_numbers!(vals, cfs)
        return vals, "ok"
    catch err
        println("Could not directly read coeffs.cfs; fieldnames(typeof(coeffs)) = ",
                fieldnames(typeof(coeffs)))
        collect_numbers!(vals, coeffs)
        return vals, "fallback_recursive: " * sprint(showerror, err)
    end
end

function coefficient_tail_values(coeffs)
    vals = Float64[]
    try
        cfs = getfield(coeffs, :cfs)
        depth = Int(getfield(coeffs, :kerneldepth))
        nblocks = min(depth, max(16, ceil(Int, 0.10 * depth)))
        first_tail = depth - nblocks + 1
        for block in first_tail:depth
            collect_numbers!(vals, view(cfs, Block(block)))
        end
        return vals
    catch
        all_vals, _ = coefficient_values(coeffs)
        isempty(all_vals) && return Float64[]
        ntail = min(length(all_vals), max(16, ceil(Int, 0.10 * length(all_vals))))
        return all_vals[(length(all_vals) - ntail + 1):end]
    end
end

function finite_abs_values(vals::AbstractVector{<:Real})
    out = Float64[]
    for v in vals
        vf = Float64(v)
        isfinite(vf) && !iszero(vf) && push!(out, abs(vf))
    end
    return out
end

function summarize_coefficients(kernel_name::AbstractString, omega, depth::Int, coeffs)
    vals, status = coefficient_values(coeffs)
    absvals = finite_abs_values(vals)
    tail_absvals = finite_abs_values(coefficient_tail_values(coeffs))
    nonfinite_count = count(v -> !isfinite(v), vals)
    top = sort(absvals; rev=true)[1:min(10, length(absvals))]
    l2 = 0.0
    for v in absvals
        l2 = hypot(l2, v)
    end

    return (
        kernel_name=kernel_name,
        omega=omega,
        kerneldepth=depth,
        coeff_count=length(vals),
        coeff_maxabs=isempty(absvals) ? 0.0 : maximum(absvals),
        coeff_minabs_nonzero=isempty(absvals) ? 0.0 : minimum(absvals),
        coeff_l1norm=sum(absvals),
        coeff_l2norm=l2,
        coeff_tail_maxabs=isempty(tail_absvals) ? 0.0 : maximum(tail_absvals),
        coeff_nonfinite_count=nonfinite_count,
        coeff_top10_abs=join((@sprintf("%.17e", v) for v in top), ";"),
        status=status,
    )
end

# ------------------------------------------------------------
# Structural matrix scanning
# ------------------------------------------------------------

function bandwidth_tuple(A)
    if A isa SparseMatrixCSC
        lower = 0
        upper = 0
        @inbounds for j in 1:size(A, 2)
            for p in A.colptr[j]:(A.colptr[j + 1] - 1)
                i = A.rowval[p]
                lower = max(lower, i - j)
                upper = max(upper, j - i)
            end
        end
        return lower, upper
    end
    try
        bw = BandedMatrices.bandwidths(A)
        return Int(bw[1]), Int(bw[2])
    catch
        return missing, missing
    end
end

function _structural_entries(visit, A)
    if A isa SparseMatrixCSC
        @inbounds for j in 1:size(A, 2)
            for p in A.colptr[j]:(A.colptr[j + 1] - 1)
                visit(A.rowval[p], j, A.nzval[p])
            end
        end
        return nothing
    end

    bw = try
        BandedMatrices.bandwidths(A)
    catch
        nothing
    end
    if bw !== nothing
        nrow, ncol = size(A)
        lower, upper = Int(bw[1]), Int(bw[2])
        @inbounds for j in 1:ncol
            ilo = max(1, j - upper)
            ihi = min(nrow, j + lower)
            for i in ilo:ihi
                visit(i, j, A[i, j])
            end
        end
        return nothing
    end

    axes(A, 1) isa Base.OneTo && axes(A, 2) isa Base.OneTo ||
        throw(ArgumentError("unsupported finite axes for $(typeof(A))"))
    nrow, ncol = size(A)
    @inbounds for j in 1:ncol
        for i in 1:nrow
            visit(i, j, A[i, j])
        end
    end
    return nothing
end

function matrix_structural_stats(A)
    nrow, ncol = size(A)
    row_sums = zeros(Float64, nrow)
    col_sums = zeros(Float64, ncol)
    row_nan = falses(nrow)
    col_nan = falses(ncol)
    maxabs = 0.0
    minabs_nonzero = Inf
    l1 = 0.0
    l2 = 0.0
    nonfinite_count = 0
    structural_count = 0
    saw_nan = false

    _structural_entries(A) do i, j, v
        structural_count += 1
        vf = Float64(v)
        if !isfinite(vf)
            nonfinite_count += 1
            if isnan(vf)
                saw_nan = true
                row_nan[i] = true
                col_nan[j] = true
            else
                maxabs = Inf
                row_sums[i] = Inf
                col_sums[j] = Inf
            end
            return
        end

        av = abs(vf)
        if !iszero(av)
            maxabs = isinf(maxabs) ? Inf : max(maxabs, av)
            minabs_nonzero = min(minabs_nonzero, av)
            l1 += av
            l2 = hypot(l2, av)
        end
        row_sums[i] += av
        col_sums[j] += av
    end

    lower, upper = bandwidth_tuple(A)
    row_sum_inf = any(isinf, row_sums) ? Inf : (any(row_nan) ? NaN : maximum(row_sums))
    col_sum_inf = any(isinf, col_sums) ? Inf : (any(col_nan) ? NaN : maximum(col_sums))

    return (
        matrix_type=string(typeof(A)),
        nrow=nrow,
        ncol=ncol,
        nnz_or_structural_nnz=structural_count,
        bandwidth_lower=lower,
        bandwidth_upper=upper,
        maxabs=saw_nan && !isinf(maxabs) ? NaN : maxabs,
        minabs_nonzero=isinf(minabs_nonzero) ? 0.0 : minabs_nonzero,
        l1_nonzero=l1,
        l2_nonzero=l2,
        row_sum_inf=row_sum_inf,
        col_sum_inf=col_sum_inf,
        nonfinite_count=nonfinite_count,
        status="ok",
    )
end

maxabs_matrix_structural(A) = matrix_structural_stats(A).maxabs
minabs_nonzero_matrix_structural(A) = matrix_structural_stats(A).minabs_nonzero
l1_matrix_structural(A) = matrix_structural_stats(A).l1_nonzero
l2_matrix_structural(A) = matrix_structural_stats(A).l2_nonzero
row_sum_inf_structural(A) = matrix_structural_stats(A).row_sum_inf
col_sum_inf_structural(A) = matrix_structural_stats(A).col_sum_inf
nonfinite_count_structural(A) = matrix_structural_stats(A).nonfinite_count

function failed_stage_row(kernel_name, omega, N, ncoeff, depth, stage, err)
    msg = "scan_failed: " * sprint(showerror, err)
    return (
        kernel_name=kernel_name,
        omega=omega,
        N=N,
        ncoeff=ncoeff,
        kerneldepth=depth,
        stage=stage,
        matrix_type="",
        nrow=missing,
        ncol=missing,
        nnz_or_structural_nnz=missing,
        bandwidth_lower=missing,
        bandwidth_upper=missing,
        maxabs=missing,
        minabs_nonzero=missing,
        l1_nonzero=missing,
        l2_nonzero=missing,
        row_sum_inf=missing,
        col_sum_inf=missing,
        nonfinite_count=missing,
        status=msg,
    )
end

function stage_row(kernel_name, omega, N, ncoeff, depth, stage, A)
    try
        s = matrix_structural_stats(A)
        return (
            kernel_name=kernel_name,
            omega=omega,
            N=N,
            ncoeff=ncoeff,
            kerneldepth=depth,
            stage=stage,
            matrix_type=s.matrix_type,
            nrow=s.nrow,
            ncol=s.ncol,
            nnz_or_structural_nnz=s.nnz_or_structural_nnz,
            bandwidth_lower=s.bandwidth_lower,
            bandwidth_upper=s.bandwidth_upper,
            maxabs=s.maxabs,
            minabs_nonzero=s.minabs_nonzero,
            l1_nonzero=s.l1_nonzero,
            l2_nonzero=s.l2_nonzero,
            row_sum_inf=s.row_sum_inf,
            col_sum_inf=s.col_sum_inf,
            nonfinite_count=s.nonfinite_count,
            status=s.status,
        )
    catch err
        return failed_stage_row(kernel_name, omega, N, ncoeff, depth, stage, err)
    end
end

function safe_bandwidths(A)
    bw = BandedMatrices.bandwidths(A)
    return Int(bw[1]), Int(bw[2])
end

function build_manual_lowering(Vraw, ncoeff)
    W = WLoweringP01P00(ncoeff)
    lower_v, upper_v = safe_bandwidths(Vraw)
    return GutlebOlver._strictband_leftmul_banded(W, Vraw, lower_v + 1, upper_v)
end

function build_manual_reflect_lowering(Vraw, Vlower, ncoeff)
    R = reflectPabtoPba(ncoeff)
    lower_v, upper_v = safe_bandwidths(Vraw)
    return GutlebOlver._strictband_leftmul_banded(R, Vlower, lower_v + 1, upper_v)
end

function construct_stage_rows(kernel_name, omega, coeffs, N, depth)
    ncoeff = N + 1
    rows = []
    stage_values = Dict{String,Any}()
    local Vraw

    try
        Vraw = triVolterraFullKernelOpP01_strictband_from_coeffs(coeffs, ncoeff, go_flip)
        stage_values["Vraw"] = Vraw
        push!(rows, stage_row(kernel_name, omega, N, ncoeff, depth, "Vraw", Vraw))
    catch err
        push!(rows, failed_stage_row(kernel_name, omega, N, ncoeff, depth, "Vraw", err))
        return rows, stage_values
    end

    try
        W = WLoweringP01P00(ncoeff)
        push!(rows, stage_row(kernel_name, omega, N, ncoeff, depth, "WLoweringP01P00", W))
    catch err
        push!(rows, failed_stage_row(kernel_name, omega, N, ncoeff, depth, "WLoweringP01P00", err))
    end

    try
        R = reflectPabtoPba(ncoeff)
        push!(rows, stage_row(kernel_name, omega, N, ncoeff, depth, "reflectPabtoPba", R))
    catch err
        push!(rows, failed_stage_row(kernel_name, omega, N, ncoeff, depth, "reflectPabtoPba", err))
    end

    local Vlower
    try
        Vlower = build_manual_lowering(Vraw, ncoeff)
        stage_values["Vlower_manual"] = Vlower
        push!(rows, stage_row(kernel_name, omega, N, ncoeff, depth, "Vlower_manual", Vlower))
    catch err
        push!(rows, failed_stage_row(kernel_name, omega, N, ncoeff, depth, "Vlower_manual", err))
    end

    try
        if @isdefined(Vlower)
            Vreflect_manual = build_manual_reflect_lowering(Vraw, Vlower, ncoeff)
            stage_values["Vreflect_lower_manual"] = Vreflect_manual
            push!(rows, stage_row(kernel_name, omega, N, ncoeff, depth, "Vreflect_lower_manual", Vreflect_manual))
        end
    catch err
        push!(rows, failed_stage_row(kernel_name, omega, N, ncoeff, depth, "Vreflect_lower_manual", err))
    end

    local Vreflect_lower
    try
        Vreflect_lower = apply_reflect_lowering_strictband(Vraw, ncoeff)
        stage_values["Vreflect_lower"] = Vreflect_lower
        push!(rows, stage_row(kernel_name, omega, N, ncoeff, depth, "Vreflect_lower", Vreflect_lower))
    catch err
        push!(rows, failed_stage_row(kernel_name, omega, N, ncoeff, depth, "Vreflect_lower", err))
    end

    try
        if @isdefined(Vreflect_lower)
            V01 = apply_conversion_00_to_01_strictband(Vreflect_lower, ncoeff)
            stage_values["V01"] = V01
            push!(rows, stage_row(kernel_name, omega, N, ncoeff, depth, "V01", V01))
        end
    catch err
        push!(rows, failed_stage_row(kernel_name, omega, N, ncoeff, depth, "V01", err))
    end

    try
        if @isdefined(Vreflect_lower)
            V12 = apply_conversion_00_to_12_strictband(Vreflect_lower, ncoeff)
            stage_values["V12"] = V12
            push!(rows, stage_row(kernel_name, omega, N, ncoeff, depth, "V12", V12))
        end
    catch err
        push!(rows, failed_stage_row(kernel_name, omega, N, ncoeff, depth, "V12", err))
    end

    try
        D = build_derivative_P01_to_P12(ncoeff)
        stage_values["D"] = D
        push!(rows, stage_row(kernel_name, omega, N, ncoeff, depth, "D", D))
    catch err
        push!(rows, failed_stage_row(kernel_name, omega, N, ncoeff, depth, "D", err))
    end

    try
        C = Conversion(solution_space, body_space)[1:ncoeff, 1:ncoeff]
        stage_values["C"] = C
        push!(rows, stage_row(kernel_name, omega, N, ncoeff, depth, "C", C))
    catch err
        push!(rows, failed_stage_row(kernel_name, omega, N, ncoeff, depth, "C", err))
    end

    try
        body_minus_public = build_VIDE_LHS_sec522_strictband(Vraw, ncoeff)
        stage_values["body_minus_public"] = body_minus_public
        push!(rows, stage_row(kernel_name, omega, N, ncoeff, depth, "body_minus_public", body_minus_public))

        D = get(stage_values, "D", build_derivative_P01_to_P12(ncoeff))
        C = get(stage_values, "C", Conversion(solution_space, body_space)[1:ncoeff, 1:ncoeff])
        VK_public = D - body_minus_public
        body_public = (1.0 / kappa) * D + C + VK_public
        A_public = assemble_semibanded_system(
            body_public,
            ncoeff;
            evaluation_row=build_initial_evaluation_row(ncoeff),
        )
        stage_values["VK_public"] = VK_public
        stage_values["body_public"] = body_public
        stage_values["A_public"] = A_public
        push!(rows, stage_row(kernel_name, omega, N, ncoeff, depth, "VK_public", VK_public))
        push!(rows, stage_row(kernel_name, omega, N, ncoeff, depth, "body_public", body_public))
        push!(rows, stage_row(kernel_name, omega, N, ncoeff, depth, "A_public", A_public))
    catch err
        push!(rows, failed_stage_row(kernel_name, omega, N, ncoeff, depth, "public_VIDE_chain", err))
    end

    return rows, stage_values
end

function build_kernel_coefficients(K)
    plan = precompute_triangular_kernel_plan(kerneldepth, go_flip)
    coeffs = triangular_kernel_coeffs_from_plan(plan, K)
    return coeffs
end

function first_stage_over_threshold(rows, threshold::Float64)
    for row in rows
        row.status == "ok" || continue
        row.maxabs isa Real || continue
        row.maxabs > threshold && return row.stage
    end
    return ""
end

function row_maxabs(rows, stage::String)
    for row in rows
        if row.stage == stage && row.status == "ok" && row.maxabs isa Real
            return Float64(row.maxabs)
        end
    end
    return NaN
end

function run_kernel_case(kernel_name::String, omega, K, Ns)
    println()
    println("Kernel case: ", kernel_name, ", omega = ", omega)
    coeffs = build_kernel_coefficients(K)
    coeff_row = summarize_coefficients(kernel_name, omega, kerneldepth, coeffs)
    @printf("  coeff_count          = %d\n", coeff_row.coeff_count)
    @printf("  coeff_maxabs         = %.6e\n", coeff_row.coeff_maxabs)
    @printf("  coeff_tail_maxabs    = %.6e\n", coeff_row.coeff_tail_maxabs)
    @printf("  coeff_nonfinite      = %d\n", coeff_row.coeff_nonfinite_count)

    all_stage_rows = []
    for N in Ns
        @printf("  Building stages for N = %d\n", N)
        rows, _ = construct_stage_rows(kernel_name, omega, coeffs, N, kerneldepth)
        append!(all_stage_rows, rows)
        jump_stage = first_stage_over_threshold(rows, 1.0e20)
        println("    first maxabs > 1e20: ", isempty(jump_stage) ? "none" : jump_stage)
    end
    return coeff_row, all_stage_rows
end

function run_omega_scan()
    rows = []
    N = 500
    for om in omega_scan_values
        println()
        @printf("Omega scan: omega = %.6g, N = %d\n", om, N)
        K = (x, y) -> kernel_go(x, y; omega=om)
        coeffs = build_kernel_coefficients(K)
        coeff_row = summarize_coefficients("cos_omega_xy", om, kerneldepth, coeffs)
        stage_rows, _ = construct_stage_rows("cos_omega_xy", om, coeffs, N, kerneldepth)
        status = any(row -> row.status != "ok", stage_rows) ? "stage_failure" : "ok"
        push!(rows, (
            omega=om,
            N=N,
            kerneldepth=kerneldepth,
            coeff_maxabs=coeff_row.coeff_maxabs,
            coeff_tail_maxabs=coeff_row.coeff_tail_maxabs,
            maxabs_Vraw=row_maxabs(stage_rows, "Vraw"),
            maxabs_after_reflect_lower=row_maxabs(stage_rows, "Vreflect_lower"),
            maxabs_V12=row_maxabs(stage_rows, "V12"),
            maxabs_public_VK=row_maxabs(stage_rows, "VK_public"),
            status=status,
        ))
    end
    return rows
end

const COEFF_COLUMNS = (
    :kernel_name,
    :omega,
    :kerneldepth,
    :coeff_count,
    :coeff_maxabs,
    :coeff_minabs_nonzero,
    :coeff_l1norm,
    :coeff_l2norm,
    :coeff_tail_maxabs,
    :coeff_nonfinite_count,
    :coeff_top10_abs,
    :status,
)

const STAGE_COLUMNS = (
    :kernel_name,
    :omega,
    :N,
    :ncoeff,
    :kerneldepth,
    :stage,
    :matrix_type,
    :nrow,
    :ncol,
    :nnz_or_structural_nnz,
    :bandwidth_lower,
    :bandwidth_upper,
    :maxabs,
    :minabs_nonzero,
    :l1_nonzero,
    :l2_nonzero,
    :row_sum_inf,
    :col_sum_inf,
    :nonfinite_count,
    :status,
)

const OMEGA_COLUMNS = (
    :omega,
    :N,
    :kerneldepth,
    :coeff_maxabs,
    :coeff_tail_maxabs,
    :maxabs_Vraw,
    :maxabs_after_reflect_lower,
    :maxabs_V12,
    :maxabs_public_VK,
    :status,
)

println("============================================================")
println("ex4 GO stage amplification diagnostics")
println("project_root        = ", project_root)
println("go_project          = ", go_project)
println("kerneldepth         = ", kerneldepth)
println("Ns                  = ", join(Ns, ","))
println("go_flip             = ", go_flip)
println("data_dir            = ", data_dir)
println("============================================================")

coeff_rows = []
stage_rows = []

cos_coeff_row, cos_stage_rows = run_kernel_case(
    "cos_50xy",
    omega_default,
    (x, y) -> kernel_go(x, y; omega=omega_default),
    Ns,
)
push!(coeff_rows, cos_coeff_row)
append!(stage_rows, cos_stage_rows)

const_coeff_row, const_stage_rows = run_kernel_case(
    "constant",
    0.0,
    kernel_const,
    Ns,
)
push!(coeff_rows, const_coeff_row)
append!(stage_rows, const_stage_rows)

omega_rows = run_omega_scan()

write_rows_csv(coeff_csv, COEFF_COLUMNS, coeff_rows)
write_rows_csv(stage_csv, STAGE_COLUMNS, stage_rows)
write_rows_csv(omega_csv, OMEGA_COLUMNS, omega_rows)

println()
println("Finished.")
println("Wrote:")
println("  ", coeff_csv)
println("  ", stage_csv)
println("  ", omega_csv)
println("============================================================")
