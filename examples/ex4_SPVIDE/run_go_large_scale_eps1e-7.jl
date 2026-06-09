############################################################
# Large-scale diagnostic runner for the ex4 Gutleb--Olver path.
#
# This script is intentionally separate from the normal benchmark.  It is
# designed to test whether the original eps=1e-7 boundary layer starts to be
# resolved only at large N.  Results are appended after each N so the run can
# be resumed.
#
# Default N list:
#   512, 1024, 2048, 4096, 8192, 16384, 32768
#
# Useful environment variables:
#   GO_LARGE_N_LIST              comma-separated N list
#   GO_LARGE_MAX_N               default 32768
#   GO_LARGE_START_N             skip N below this value
#   GO_LARGE_PROJECTION_ONLY     run only projection phase
#   GO_LARGE_SOLVE_ONLY          run only solve phase
#   GO_LARGE_FORCE_RERUN         rerun rows already present in CSV
#   GO_LARGE_KERNEL_DEPTH        default ex4 safe value min(8,ceil(sqrt(N)))
#   GO_LARGE_GRID_LAYER_WIDTH    default 1e-5
#   GO_LARGE_GRID_LAYER_POINTS   default 4000
#   GO_LARGE_GRID_OUTER_POINTS   default 4000
#   GO_LARGE_EVAL_CHUNK          default 1000
#   GO_LARGE_COMPUTE_CAUCHY      default 0; solve reference_N only when 1
############################################################

using Pkg

function find_project_root(start_dir::AbstractString = @__DIR__)
    dir = abspath(start_dir)
    while true
        if isfile(joinpath(dir, "Project.toml")) && isdir(joinpath(dir, "src"))
            return dir
        end
        parent = dirname(dir)
        parent == dir && error("Could not find project root from $(start_dir).")
        dir = parent
    end
end

const PROJECT_ROOT = find_project_root()
const GO_PROJECT = joinpath(PROJECT_ROOT, "others", "GutlebOlver")
const DATA_DIR = joinpath(PROJECT_ROOT, "examples", "data")
mkpath(DATA_DIR)

Pkg.activate(PROJECT_ROOT; io=devnull)

using FastGaussQuadrature
using LinearAlgebra
using Printf
using SparseArrays

Pkg.activate(GO_PROJECT; io=devnull)

using ApproxFun
using BandedMatrices
using SparseVolterraNew

const EPS_EX4 = 1.0e-7
const MU_EX4 = -1.0
const P01 = Jacobi(0, 1, 0..1)
const P00 = Jacobi(0, 0, 0..1)
const P12 = Jacobi(1, 2, 0..1)
const OUT_CSV = joinpath(DATA_DIR, "ex4_go_large_scale_eps1e-7.csv")

const CSV_HEADERS = [
    :phase,
    :projection_method,
    :basis_family,
    :basis_parameters,
    :coefficient_length,
    :tail_window,
    :eps,
    :N,
    :reference_N,
    :status,
    :projection_error,
    :exact_error,
    :cauchy_error,
    :rhs_time,
    :kernel_approx_time,
    :matrix_construct_time,
    :system_assembly_time,
    :solve_time,
    :evaluation_time,
    :total_time,
    :projection_time,
    :go_storage_type,
    :go_operator_nnz,
    :go_system_nnz,
    :go_lower_bandwidth,
    :go_upper_bandwidth,
    :go_density,
    :solution_coeff_norm,
    :solution_coeff_tail_norm,
    :voltmat_exact_error_boundary_grid,
    :voltmat_cauchy_error_original,
    :voltmat_total_time,
    :evaluation_grid_description,
    :notes,
]

phi1_ex4(z) = abs(z) < 1.0e-8 ? 1.0 + z / 2.0 + z^2 / 6.0 + z^3 / 24.0 + z^4 / 120.0 : expm1(z) / z
u_exact(s) = sin(s) + exp(-s / EPS_EX4)
K_go(s, r) = exp(MU_EX4 * s * r)

function rhs_go(s)
    layer = exp(-s / EPS_EX4)
    v = sin(s) + layer
    diff_part = EPS_EX4 * cos(s) - layer
    local_part = (1.0 + s) * v
    a = MU_EX4 * s
    I_sin = (exp(a * s) * (a * sin(s) - cos(s)) + 1.0) / (a^2 + 1.0)
    I_layer = s * phi1_ex4(MU_EX4 * s^2 - s / EPS_EX4)
    return diff_part + local_part + I_sin + I_layer
end

envflag(name::AbstractString) = get(ENV, name, "0") in ("1", "true", "TRUE", "yes", "YES")

function parse_n_list()
    raw = strip(get(ENV, "GO_LARGE_N_LIST", ""))
    if isempty(raw)
        Ns = [512, 1024, 2048, 4096, 8192, 16384, 32768]
    else
        Ns = parse.(Int, split(raw, ","))
    end
    maxN = parse(Int, get(ENV, "GO_LARGE_MAX_N", "32768"))
    startN = parse(Int, get(ENV, "GO_LARGE_START_N", "0"))
    return [N for N in Ns if startN <= N <= maxN]
end

function kerneldepth_for_N(N::Int)
    raw = strip(get(ENV, "GO_LARGE_KERNEL_DEPTH", ""))
    isempty(raw) || return parse(Int, raw)
    return min(8, max(2, ceil(Int, sqrt(N))))
end

function build_grid()
    layer_width = parse(Float64, get(ENV, "GO_LARGE_GRID_LAYER_WIDTH", "1e-5"))
    layer_points = parse(Int, get(ENV, "GO_LARGE_GRID_LAYER_POINTS", "4000"))
    outer_points = parse(Int, get(ENV, "GO_LARGE_GRID_OUTER_POINTS", "4000"))
    p1 = max(10, layer_points ÷ 2)
    p2 = max(10, layer_points ÷ 2)
    p3 = max(10, layer_points)
    grid = unique(sort(vcat(
        collect(range(0.0, 1.0e-7; length=p1)),
        collect(range(1.0e-7, 1.0e-6; length=p2)),
        collect(range(1.0e-6, layer_width; length=p3)),
        collect(range(layer_width, 1.0; length=outer_points)),
    )))
    return grid, "union([0,1e-7], [1e-7,1e-6], [1e-6,$layer_width], [$layer_width,1]) with $(length(grid)) points"
end

function csv_escape(x)
    if x isa AbstractFloat && isnan(x)
        return "NaN"
    end
    s = string(x)
    if occursin(",", s) || occursin("\"", s) || occursin("\n", s)
        return "\"" * replace(s, "\"" => "\"\"") * "\""
    end
    return s
end

function ensure_csv()
    expected = join(string.(CSV_HEADERS), ",")
    if isfile(OUT_CSV)
        firstline = open(OUT_CSV, "r") do io
            eof(io) ? "" : readline(io)
        end
        if firstline != expected
            backup = OUT_CSV * ".obsolete"
            k = 1
            while isfile(backup)
                backup = OUT_CSV * ".obsolete$(k)"
                k += 1
            end
            mv(OUT_CSV, backup)
            println("Existing CSV header is obsolete; moved old file to ", backup)
        end
    end
    if !isfile(OUT_CSV)
        open(OUT_CSV, "w") do io
            println(io, expected)
        end
    end
end

function append_row(row)
    ensure_csv()
    open(OUT_CSV, "a") do io
        println(io, join((csv_escape(getproperty(row, h)) for h in CSV_HEADERS), ","))
    end
end

function completed_keys()
    isfile(OUT_CSV) || return Set{Tuple{String,Int}}()
    rows = readlines(OUT_CSV)
    length(rows) <= 1 && return Set{Tuple{String,Int}}()
    header = split(rows[1], ",")
    phase_idx = findfirst(==("phase"), header)
    n_idx = findfirst(==("N"), header)
    (phase_idx === nothing || n_idx === nothing) && return Set{Tuple{String,Int}}()
    keys = Set{Tuple{String,Int}}()
    for line in rows[2:end]
        isempty(strip(line)) && continue
        parts = split(line, ",")
        length(parts) >= max(phase_idx, n_idx) || continue
        N = tryparse(Int, parts[n_idx])
        N === nothing && continue
        push!(keys, (parts[phase_idx], N))
    end
    return keys
end

function rewrite_without_keys(remove_keys::Set{Tuple{String,Int}})
    isfile(OUT_CSV) || return
    rows = readlines(OUT_CSV)
    isempty(rows) && return
    header = split(rows[1], ",")
    phase_idx = findfirst(==("phase"), header)
    n_idx = findfirst(==("N"), header)
    (phase_idx === nothing || n_idx === nothing) && return
    kept = String[rows[1]]
    for line in rows[2:end]
        parts = split(line, ",")
        if length(parts) >= max(phase_idx, n_idx)
            N = tryparse(Int, parts[n_idx])
            if N !== nothing && (parts[phase_idx], N) in remove_keys
                continue
            end
        end
        push!(kept, line)
    end
    open(OUT_CSV, "w") do io
        for line in kept
            println(io, line)
        end
    end
end

function banded_structural_nnz(n::Int, l::Int, u::Int)
    total = 0
    for j in 1:n
        total += max(0, min(n, j + l) - max(1, j - u) + 1)
    end
    return total
end

function storage_stats(A)
    if A isa Matrix
        error("dense path rejected: matrix storage is dense Matrix")
    elseif A isa SparseMatrixCSC
        rows, cols, vals = findnz(A)
        actual = 0
        lb = 0
        ub = 0
        for (i, j, v) in zip(rows, cols, vals)
            iszero(v) && continue
            actual += 1
            lb = max(lb, i - j)
            ub = max(ub, j - i)
        end
        return (storage=string(parentmodule(typeof(A)), ".", nameof(typeof(A))),
                nnz=actual, lower=lb, upper=ub, density=actual / prod(size(A)))
    elseif A isa BandedMatrices.AbstractBandedMatrix
        l, u = bandwidths(A)
        actual = banded_structural_nnz(size(A, 1), Int(l), Int(u))
        return (storage=string(typeof(A)), nnz=actual, lower=Int(l), upper=Int(u),
                density=actual / prod(size(A)))
    else
        error("dense path rejected: unsupported matrix storage $(typeof(A))")
    end
end

function coeff_norms(c; tail_window::Int=min(50, length(c)))
    tail_len = min(tail_window, length(c))
    return norm(c, Inf), norm(c[end-tail_len+1:end], Inf)
end

function jacobi_p01_values(s::AbstractVector, n::Int)
    m = length(s)
    V = Matrix{Float64}(undef, m, n)
    n == 0 && return V
    x = @. 2.0 * s - 1.0
    V[:, 1] .= 1.0
    n == 1 && return V
    V[:, 2] .= -0.5 .+ 1.5 .* x
    for k in 1:(n - 2)
        alpha = 0.0
        beta = 1.0
        kk = Float64(k)
        a1 = 2.0 * (kk + 1.0) * (kk + alpha + beta + 1.0) *
            (2.0 * kk + alpha + beta)
        a2 = (2.0 * kk + alpha + beta + 1.0) * (alpha^2 - beta^2)
        a3 = (2.0 * kk + alpha + beta) *
            (2.0 * kk + alpha + beta + 1.0) *
            (2.0 * kk + alpha + beta + 2.0)
        a4 = 2.0 * (kk + alpha) * (kk + beta) *
            (2.0 * kk + alpha + beta + 2.0)
        @views V[:, k + 2] .= ((a2 .+ a3 .* x) .* V[:, k + 1] .- a4 .* V[:, k]) ./ a1
    end
    return V
end

p01_norms(n::Int) = [1.0 / (2.0 * (k + 1.0)) for k in 0:(n - 1)]

function fixedN_p01_projection(f, n::Int)
    max_direct = parse(Int, get(ENV, "GO_LARGE_FIXED_PROJECTION_MAX_N", "2048"))
    if n - 1 > max_direct
        error("fixed-N direct Gauss-Jacobi projection rejected for N=$(n - 1): O(N^2) direct transform would be too expensive; provide a fixed-N fast Jacobi transform or raise GO_LARGE_FIXED_PROJECTION_MAX_N explicitly")
    end
    qfactor = parse(Int, get(ENV, "GO_LARGE_PROJECTION_QFACTOR", "2"))
    qn = max(n + 8, qfactor * n)
    x, w = gaussjacobi(qn, 0.0, 1.0)
    s = @. 0.5 * (x + 1.0)
    vals = f.(s)
    V = jacobi_p01_values(s, n)
    h = p01_norms(n)
    c = Vector{Float64}(undef, n)
    for j in 1:n
        c[j] = 0.25 * dot(w, vals .* V[:, j]) / h[j]
    end
    return c
end

function eval_p01_fixed_chunked(c, grid)
    chunk = parse(Int, get(ENV, "GO_LARGE_EVAL_CHUNK", "1000"))
    vals = Vector{Float64}(undef, length(grid))
    for lo in 1:chunk:length(grid)
        hi = min(length(grid), lo + chunk - 1)
        V = jacobi_p01_values(collect(view(grid, lo:hi)), length(c))
        vals[lo:hi] .= V * c
    end
    return vals
end

function eval_fun_chunked(c, basis, grid)
    chunk = parse(Int, get(ENV, "GO_LARGE_EVAL_CHUNK", "1000"))
    f = Fun(basis, c)
    vals = Vector{Float64}(undef, length(grid))
    for lo in 1:chunk:length(grid)
        hi = min(length(grid), lo + chunk - 1)
        vals[lo:hi] .= f.(view(grid, lo:hi))
    end
    return vals
end

function project_solution(N::Int, grid)
    n = N + 1
    local c
    tproj = @elapsed begin
        c = fixedN_p01_projection(u_exact, n)
    end
    local err
    teval = @elapsed begin
        vals = eval_p01_fixed_chunked(c, grid)
        exact = u_exact.(grid)
        err = maximum(abs.(vals .- exact))
    end
    tail_window = min(50, length(c))
    cn, tn = coeff_norms(c; tail_window=tail_window)
    return (; coeffs=c, projection_error=err, projection_time=tproj,
        evaluation_time=teval, solution_coeff_norm=cn, solution_coeff_tail_norm=tn,
        tail_window=tail_window)
end

function tau_row(n::Int)
    vals = zeros(Float64, n)
    for j in 1:n
        ej = zeros(Float64, n)
        ej[j] = 1.0
        vals[j] = Fun(P01, ej)(0.0)
    end
    return sparse(ones(Int, n), collect(1:n), vals, 1, n)
end

function assemble_system(N::Int)
    n = N + 1
    kd = kerneldepth_for_N(N)
    local coeffs
    tkernel = @elapsed begin
        coeffs = build_gutleb_kernel_coefficients(K_go, kd; flip=true)
    end
    local VK
    tmatrix = @elapsed begin
        VK = build_gutleb_operator_from_coefficients(coeffs, n; flip=true)
    end
    vk_stats = storage_stats(VK)

    local A, rhs
    trhs = 0.0
    tassemble = @elapsed begin
        Vrl = apply_reflect_lowering_strictband(VK, n)
        Vconv = apply_conversion_00_to_12_strictband(Vrl, n)
        D = Derivative(P01, 1)[1:n, 1:n]
        C01to12 = Conversion(P01, P12)[1:n, 1:n]
        Mlocal = Multiplication(Fun(s -> 1.0 + s, P01), P01)[1:n, 1:n]
        body = EPS_EX4 * D + C01to12 * Mlocal + Vconv
        tau = tau_row(n)
        A = vcat(tau, sparse(body[1:(n - 1), :]))
        trhs = @elapsed begin
            rhs_full = pad(Fun(rhs_go, P12).coefficients, n)
            rhs = Vector{Float64}(undef, n)
            rhs[1] = 1.0
            rhs[2:end] .= rhs_full[1:(n - 1)]
        end
    end
    a_stats = storage_stats(A)
    return (; A, rhs, VK, vk_stats, a_stats, rhs_time=trhs,
        kernel_approx_time=tkernel, matrix_construct_time=tmatrix,
        system_assembly_time=tassemble)
end

function solve_coefficients(N::Int)
    sys = assemble_system(N)
    local c
    tsolve = @elapsed begin
        c = sys.A \ sys.rhs
    end
    return merge(sys, (; coeffs=c, solve_time=tsolve))
end

function projection_row(N, grid, grid_desc)
    try
        result = project_solution(N, grid)
        status = isfinite(result.projection_error) ? "success" : "failed"
        notes = "fixed-N Jacobi-Gauss projection to shifted P^(0,1), evaluated on boundary-layer-aware grid; dense Vandermonde is used only inside the direct fixed-N projection up to GO_LARGE_FIXED_PROJECTION_MAX_N; no dense GO operator/system materialization"
        return make_row(phase="projection", N=N, status=status,
            projection_error=result.projection_error,
            projection_time=result.projection_time,
            evaluation_time=result.evaluation_time,
            solution_coeff_norm=result.solution_coeff_norm,
            solution_coeff_tail_norm=result.solution_coeff_tail_norm,
            coefficient_length=N + 1,
            tail_window=result.tail_window,
            evaluation_grid_description=grid_desc,
            notes=notes)
    catch err
        return make_row(phase="projection", N=N, status="failed",
            evaluation_grid_description=grid_desc,
            notes="projection failed: " * sprint(showerror, err))
    end
end

function solve_row(N, grid, grid_desc)
    try
        result = solve_coefficients(N)
        local exact_error
        teval = @elapsed begin
            vals = eval_fun_chunked(result.coeffs, P01, grid)
            exact_error = maximum(abs.(vals .- u_exact.(grid)))
        end
        cn, tn = coeff_norms(result.coeffs)
        refN = min(max(round(Int, 1.01 * N), N + 8), N + 512)
        cauchy_error = NaN
        notes = "semi-banded sparse system with one dense tau row; cauchy skipped by default due to cost; set GO_LARGE_COMPUTE_CAUCHY=1 to compute"
        if envflag("GO_LARGE_COMPUTE_CAUCHY")
            ref = solve_coefficients(refN)
            ref_vals = eval_fun_chunked(ref.coeffs, P01, grid)
            vals = eval_fun_chunked(result.coeffs, P01, grid)
            cauchy_error = maximum(abs.(vals .- ref_vals))
            notes = "semi-banded sparse system with one dense tau row; physical-grid cauchy computed against reference_N=$refN"
        end
        return make_row(phase="solve", N=N, reference_N=refN, status="success",
            exact_error=exact_error, cauchy_error=cauchy_error,
            rhs_time=result.rhs_time,
            kernel_approx_time=result.kernel_approx_time,
            matrix_construct_time=result.matrix_construct_time,
            system_assembly_time=result.system_assembly_time,
            solve_time=result.solve_time,
            evaluation_time=teval,
            total_time=result.rhs_time + result.kernel_approx_time +
                result.matrix_construct_time + result.system_assembly_time +
                result.solve_time + teval,
            go_storage_type=result.a_stats.storage,
            go_operator_nnz=result.vk_stats.nnz,
            go_system_nnz=result.a_stats.nnz,
            go_lower_bandwidth=result.a_stats.lower,
            go_upper_bandwidth=result.a_stats.upper,
            go_density=result.a_stats.density,
            solution_coeff_norm=cn,
            solution_coeff_tail_norm=tn,
            evaluation_grid_description=grid_desc,
            notes=notes)
    catch err
        return make_row(phase="solve", N=N, status="failed",
            evaluation_grid_description=grid_desc,
            notes="solve failed: " * sprint(showerror, err))
    end
end

function make_row(; phase, N, reference_N="missing", status,
    projection_method=phase == "projection" ? "fixed_N_jacobi_gauss_projection" : "not_applicable",
    basis_family="shifted Jacobi",
    basis_parameters="solution P^(0,1); output/RHS P^(1,2)",
    coefficient_length="missing",
    tail_window="missing",
    projection_error=NaN, exact_error=NaN, cauchy_error=NaN,
    rhs_time=NaN, kernel_approx_time=NaN, matrix_construct_time=NaN,
    system_assembly_time=NaN, solve_time=NaN, evaluation_time=NaN,
    total_time=NaN, projection_time=NaN, go_storage_type="missing",
    go_operator_nnz="missing", go_system_nnz="missing",
    go_lower_bandwidth="missing", go_upper_bandwidth="missing",
    go_density="missing", solution_coeff_norm=NaN,
    solution_coeff_tail_norm=NaN, voltmat_exact_error_boundary_grid=NaN,
    voltmat_cauchy_error_original=NaN, voltmat_total_time=NaN,
    evaluation_grid_description, notes)
    return (
        phase=phase,
        projection_method=projection_method,
        basis_family=basis_family,
        basis_parameters=basis_parameters,
        coefficient_length=coefficient_length,
        tail_window=tail_window,
        eps=EPS_EX4,
        N=N,
        reference_N=reference_N,
        status=status,
        projection_error=projection_error,
        exact_error=exact_error,
        cauchy_error=cauchy_error,
        rhs_time=rhs_time,
        kernel_approx_time=kernel_approx_time,
        matrix_construct_time=matrix_construct_time,
        system_assembly_time=system_assembly_time,
        solve_time=solve_time,
        evaluation_time=evaluation_time,
        total_time=total_time,
        projection_time=projection_time,
        go_storage_type=go_storage_type,
        go_operator_nnz=go_operator_nnz,
        go_system_nnz=go_system_nnz,
        go_lower_bandwidth=go_lower_bandwidth,
        go_upper_bandwidth=go_upper_bandwidth,
        go_density=go_density,
        solution_coeff_norm=solution_coeff_norm,
        solution_coeff_tail_norm=solution_coeff_tail_norm,
        voltmat_exact_error_boundary_grid=voltmat_exact_error_boundary_grid,
        voltmat_cauchy_error_original=voltmat_cauchy_error_original,
        voltmat_total_time=voltmat_total_time,
        evaluation_grid_description=evaluation_grid_description,
        notes=notes,
    )
end

function main()
    Ns = parse_n_list()
    projection_only = envflag("GO_LARGE_PROJECTION_ONLY")
    solve_only = envflag("GO_LARGE_SOLVE_ONLY")
    force = envflag("GO_LARGE_FORCE_RERUN")
    grid, grid_desc = build_grid()
    ensure_csv()
    done = completed_keys()

    println("Output CSV: ", OUT_CSV)
    println("N list: ", Ns)
    println("Grid: ", grid_desc)

    phases = if projection_only
        ["projection"]
    elseif solve_only
        ["solve"]
    else
        ["projection"]
    end

    if force
        rewrite_without_keys(Set((phase, N) for phase in phases for N in Ns))
        done = completed_keys()
    end

    for phase in phases
        for N in Ns
            if !force && (phase, N) in done
                println("Skipping existing $phase N=$N")
                continue
            end
            println("Running $phase N=$N")
            row = phase == "projection" ? projection_row(N, grid, grid_desc) :
                solve_row(N, grid, grid_desc)
            append_row(row)
            println("  status=", row.status,
                " projection_error=", row.projection_error,
                " exact_error=", row.exact_error,
                " notes=", row.notes)
            GC.gc()
        end
    end
end

main()
