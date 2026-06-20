# debug_ex4_go_bessel_kernel.jl
#
# Diagnose Gutleb--Olver strict-band Volterra construction for
# K(x,y) = J_2(a*(x-y)) on 0 <= y <= x <= 1.

using LinearAlgebra
using Printf
using SparseArrays

ex4_dir = @__DIR__
project_root = normpath(joinpath(ex4_dir, "..", ".."))
go_project = joinpath(project_root, "others", "GutlebOlver")

pushfirst!(LOAD_PATH, project_root)
pushfirst!(LOAD_PATH, go_project)

using BandedMatrices
using BlockArrays
using GutlebOlver
using SpecialFunctions

go_flip = true
kerneldepth = parse(Int, get(ENV, "EX4_KERNELDEPTH", "96"))

function parse_int_list_env(name::AbstractString, default::Vector{Int})
    raw = strip(get(ENV, name, ""))
    isempty(raw) && return default
    return [parse(Int, strip(x)) for x in split(raw, ",") if !isempty(strip(x))]
end

Ns = parse_int_list_env("EX4_STAGE_NLIST", [100, 500, 1000, 2000])
a_default = 20.0
a_values = [1.0, 5.0, 10.0, 20.0, 50.0]

data_dir = joinpath(ex4_dir, "data")
mkpath(data_dir)

coeff_csv = joinpath(data_dir, "ex4_GO_bessel_kernel_coeff_diagnostics.csv")
stage_csv = joinpath(data_dir, "ex4_GO_bessel_stage_amplification.csv")
a_scan_csv = joinpath(data_dir, "ex4_GO_bessel_a_scan.csv")

kernel_bessel(x, y; a=a_default) = besselj(2, a * (Float64(x) - Float64(y)))
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
row_value(row, col::Symbol) = hasproperty(row, col) ? getproperty(row, col) : missing

function write_rows_csv(path::AbstractString, columns, rows)
    open(path, "w") do io
        println(io, join(string.(columns), ","))
        for row in rows
            println(io, join((csv_value(row_value(row, col)) for col in columns), ","))
        end
    end
end

# ------------------------------------------------------------
# Coefficient diagnostics
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

function summarize_coefficients(kernel_name::AbstractString, a, depth::Int, coeffs)
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
        a=a,
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

function build_kernel_coefficients(K)
    plan = precompute_triangular_kernel_plan(kerneldepth, go_flip)
    return triangular_kernel_coeffs_from_plan(plan, K)
end

# ------------------------------------------------------------
# Structural matrix diagnostics
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

function failed_stage_row(kernel_name, a, N, ncoeff, depth, stage, err)
    return (
        kernel_name=kernel_name,
        a=a,
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
        status="scan_failed: " * sprint(showerror, err),
    )
end

function stage_row(kernel_name, a, N, ncoeff, depth, stage, A)
    try
        s = matrix_structural_stats(A)
        return (
            kernel_name=kernel_name,
            a=a,
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
        return failed_stage_row(kernel_name, a, N, ncoeff, depth, stage, err)
    end
end

function row_maxabs(rows, stage::String)
    for row in rows
        row.stage == stage && row.status == "ok" && row.maxabs isa Real && return Float64(row.maxabs)
    end
    return NaN
end

function row_nonfinite(rows, stage::String)
    for row in rows
        row.stage == stage && row.status == "ok" && row.nonfinite_count isa Integer &&
            return Int(row.nonfinite_count)
    end
    return 0
end

function construct_stage_rows(kernel_name::String, a, coeffs, N)
    ncoeff = N + 1
    rows = []
    local Vraw

    try
        Vraw = triVolterraFullKernelOpP01_strictband_from_coeffs(coeffs, ncoeff, go_flip)
        push!(rows, stage_row(kernel_name, a, N, ncoeff, kerneldepth, "Vraw", Vraw))
    catch err
        push!(rows, failed_stage_row(kernel_name, a, N, ncoeff, kerneldepth, "Vraw", err))
        return rows
    end

    try
        V_secondkind = build_secondkind_volterra_strictband(Vraw, ncoeff)
        push!(rows, stage_row(kernel_name, a, N, ncoeff, kerneldepth, "V_secondkind", V_secondkind))
    catch err
        push!(rows, failed_stage_row(kernel_name, a, N, ncoeff, kerneldepth, "V_secondkind", err))
    end

    local body_minus
    try
        body_minus = build_VIDE_LHS_sec522_strictband(Vraw, ncoeff)
        push!(rows, stage_row(kernel_name, a, N, ncoeff, kerneldepth, "body_minus", body_minus))
    catch err
        push!(rows, failed_stage_row(kernel_name, a, N, ncoeff, kerneldepth, "body_minus", err))
    end

    try
        D = build_derivative_P01_to_P12(ncoeff)
        push!(rows, stage_row(kernel_name, a, N, ncoeff, kerneldepth, "D", D))
        if @isdefined(body_minus)
            V_vide = D - body_minus
            push!(rows, stage_row(kernel_name, a, N, ncoeff, kerneldepth, "V_vide", V_vide))
        end
    catch err
        push!(rows, failed_stage_row(kernel_name, a, N, ncoeff, kerneldepth, "V_vide", err))
    end

    return rows
end

function run_kernel_case(kernel_name::String, a, K)
    println()
    println("Kernel case: ", kernel_name, ", a = ", a)
    coeffs = build_kernel_coefficients(K)
    coeff_row = summarize_coefficients(kernel_name, a, kerneldepth, coeffs)
    @printf("  coeff_count          = %d\n", coeff_row.coeff_count)
    @printf("  coeff_maxabs         = %.6e\n", coeff_row.coeff_maxabs)
    @printf("  coeff_tail_maxabs    = %.6e\n", coeff_row.coeff_tail_maxabs)
    @printf("  coeff_nonfinite      = %d\n", coeff_row.coeff_nonfinite_count)

    all_stage_rows = []
    for N in Ns
        @printf("  Building stages for N = %d\n", N)
        rows = construct_stage_rows(kernel_name, a, coeffs, N)
        append!(all_stage_rows, rows)
        @printf("    maxabs Vraw/secondkind/VIDE = %.6e, %.6e, %.6e\n",
                row_maxabs(rows, "Vraw"),
                row_maxabs(rows, "V_secondkind"),
                row_maxabs(rows, "V_vide"))
    end
    return coeff_row, all_stage_rows
end

function run_a_scan()
    rows = []
    N = 500
    for aval in a_values
        println()
        @printf("a scan: a = %.6g, N = %d\n", aval, N)
        K = (x, y) -> kernel_bessel(x, y; a=aval)
        coeffs = build_kernel_coefficients(K)
        coeff_row = summarize_coefficients("bessel_J2_a", aval, kerneldepth, coeffs)
        stage_rows = construct_stage_rows("bessel_J2_a", aval, coeffs, N)
        status = any(row -> row.status != "ok", stage_rows) ? "stage_failure" : "ok"
        nbad = row_nonfinite(stage_rows, "Vraw") +
               row_nonfinite(stage_rows, "V_secondkind") +
               row_nonfinite(stage_rows, "V_vide")
        push!(rows, (
            a=aval,
            N=N,
            kerneldepth=kerneldepth,
            coeff_maxabs=coeff_row.coeff_maxabs,
            coeff_tail_maxabs=coeff_row.coeff_tail_maxabs,
            maxabs_Vraw=row_maxabs(stage_rows, "Vraw"),
            maxabs_secondkind=row_maxabs(stage_rows, "V_secondkind"),
            maxabs_vide=row_maxabs(stage_rows, "V_vide"),
            nonfinite_count=nbad,
            status=status,
        ))
    end
    return rows
end

const COEFF_COLUMNS = (
    :kernel_name,
    :a,
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
    :a,
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

const A_SCAN_COLUMNS = (
    :a,
    :N,
    :kerneldepth,
    :coeff_maxabs,
    :coeff_tail_maxabs,
    :maxabs_Vraw,
    :maxabs_secondkind,
    :maxabs_vide,
    :nonfinite_count,
    :status,
)

println("============================================================")
println("ex4 GO Bessel-kernel diagnostics")
println("project_root        = ", project_root)
println("go_project          = ", go_project)
println("kerneldepth         = ", kerneldepth)
println("Ns                  = ", join(Ns, ","))
println("go_flip             = ", go_flip)
println("data_dir            = ", data_dir)
println("============================================================")

coeff_rows = []
stage_rows = []

bessel_coeff_row, bessel_stage_rows = run_kernel_case(
    "bessel_J2_20_xminusy",
    a_default,
    (x, y) -> kernel_bessel(x, y; a=a_default),
)
push!(coeff_rows, bessel_coeff_row)
append!(stage_rows, bessel_stage_rows)

const_coeff_row, const_stage_rows = run_kernel_case("constant", 0.0, kernel_const)
push!(coeff_rows, const_coeff_row)
append!(stage_rows, const_stage_rows)

a_scan_rows = run_a_scan()

write_rows_csv(coeff_csv, COEFF_COLUMNS, coeff_rows)
write_rows_csv(stage_csv, STAGE_COLUMNS, stage_rows)
write_rows_csv(a_scan_csv, A_SCAN_COLUMNS, a_scan_rows)

println()
println("Finished.")
println("Wrote:")
println("  ", coeff_csv)
println("  ", stage_csv)
println("  ", a_scan_csv)
println("============================================================")
