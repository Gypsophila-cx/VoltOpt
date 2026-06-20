# ex4_GO.jl
#
# Gutleb--Olver solve for
#
#   kappa^{-1} u'(x) + u(x) + int_0^x cos(omega*x*y) u(y) dy = f(x),
#   x in [0,1], u(0)=1.
#
# Exact solution:
#
#   u(x) = cos(kappa*x), kappa = 1000, omega = 50.
#
# The implementation follows the sparse Gutleb--Olver workflow used by the
# verified ex1 script: solution coefficients live in Jacobi(0,1,0..1), body
# equations live in Jacobi(1,2,0..1), the Volterra term is built through the
# public VIDE operator workflow, and the tau row enforces u(0)=1.

using LinearAlgebra
using Printf
using SparseArrays

# ------------------------------------------------------------
# Locate package environments
# ------------------------------------------------------------

ex4_dir = @__DIR__
project_root = normpath(joinpath(ex4_dir, "..", ".."))
go_project = joinpath(project_root, "others", "GutlebOlver")

pushfirst!(LOAD_PATH, project_root)
pushfirst!(LOAD_PATH, go_project)

using ApproxFun
using BandedMatrices
using BlockArrays
using FastAlmostBandedMatrices
using GutlebOlver

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------

omega = 50.0
kappa = 1000.0

N_min = 10
N_max = 2000
num_N = 100
Ns_full = unique(round.(Int, range(N_min, N_max, length=num_N)))

function parse_int_list_env(name::AbstractString)
    raw = strip(get(ENV, name, ""))
    isempty(raw) && return nothing
    return [parse(Int, strip(x)) for x in split(raw, ",") if !isempty(strip(x))]
end

quick_mode = lowercase(get(ENV, "EX4_QUICK", "0")) in ("1", "true", "yes", "on")
Ns_custom = parse_int_list_env("EX4_N_LIST")
Ns_diag = parse_int_list_env("EX4_DIAG_NLIST")
Ns = Ns_diag !== nothing ? Ns_diag :
     (Ns_custom === nothing ? (quick_mode ? [10, 50, 100] : Ns_full) : Ns_custom)

grid_M = 20000
action_grid_M = 2000
N_rhs_fixed = 8192

const DEFAULT_KERNELDEPTH = 96
const AUTO_KERNELDEPTH_CANDIDATES = [96, 128, 160, 192, 256, 320, 384, 512, 640, 768, 1024]
const KERNEL_TAIL_TARGET_TOL = 100eps(Float64)
const KERNEL_PROJECTION_TARGET_TOL = 1.0e-14
const KERNEL_PROJECTION_GRID_LEN = 81

kerneldepth_raw = strip(get(ENV, "EX4_KERNELDEPTH", string(DEFAULT_KERNELDEPTH)))
kerneldepth_mode = lowercase(kerneldepth_raw) == "auto" ? "auto" : "fixed"
requested_kerneldepth = kerneldepth_mode == "auto" ? missing : parse(Int, kerneldepth_raw)
go_flip = true

solution_space = Jacobi(0, 1, 0..1)
body_space = Jacobi(1, 2, 0..1)

data_dir = joinpath(@__DIR__, "data")
mkpath(data_dir)

error_csv = joinpath(data_dir, "ex4_GO_error.csv")
time_csv = joinpath(data_dir, "ex4_GO_time.csv")
kerneldepth_scan_csv = joinpath(data_dir, "ex4_GO_kerneldepth_scan.csv")
matrix_diagnostics_csv = joinpath(data_dir, "ex4_GO_matrix_diagnostics.csv")

solver_name = "almost_banded"

# ------------------------------------------------------------
# Manufactured solution and RHS
# ------------------------------------------------------------

function sinc_stable(z::Float64)
    az = abs(z)
    if az < 1.0e-4
        z2 = z*z
        return 1.0 - z2/6.0 + z2*z2/120.0 - z2*z2*z2/5040.0
    end
    return sin(z)/z
end

function exact_integral_original(x::Real; omega=omega, kappa=kappa)
    xf = Float64(x)
    z_minus = (omega*xf - kappa)*xf
    z_plus = (omega*xf + kappa)*xf
    return 0.5*xf*(sinc_stable(z_minus) + sinc_stable(z_plus))
end

function exact_original(x::Real; kappa=kappa)
    return cos(kappa*Float64(x))
end

function rhs_original(x::Real; omega=omega, kappa=kappa)
    xf = Float64(x)
    return -sin(kappa*xf) +
           cos(kappa*xf) +
           exact_integral_original(xf; omega=omega, kappa=kappa)
end

kernel_target(x::Real, y::Real) = cos(omega*Float64(x)*Float64(y))
kernel_one(x::Real, y::Real) = 1.0

# ------------------------------------------------------------
# Generic helpers
# ------------------------------------------------------------

function pad_to_length_local(c, n::Integer)
    n < 0 && throw(ArgumentError("target length must be non-negative"))
    out = zeros(Float64, Int(n))
    m = min(length(c), Int(n))
    m > 0 && (out[1:m] .= c[1:m])
    return out
end

function assert_all_finite(label::AbstractString, x)
    if x isa Number
        isfinite(x) || error(label * " is not finite: $(x)")
        return nothing
    end
    all(isfinite, x) || error(label * " contains NaN or Inf")
    return nothing
end

function count_nnz(A)
    try
        return safe_nnz(A)
    catch
        return nnz(A)
    end
end

function bandwidth_string(A)
    try
        bw = safe_bandwidth(A)
        return "($(bw[1]),$(bw[2]))"
    catch
        return ""
    end
end

function projection_coefficients(f, n::Int; alpha::Int, beta::Int)
    coeffs, info = build_go_rhs_coefficients(
        f,
        n;
        alpha=alpha,
        beta=beta,
        domain=0..1,
        method=:fixed_gauss_jacobi,
        validate=false,
    )
    return Vector{Float64}(coeffs), info
end

function kernel_coefficients_tail_indicator(coeffs; tail_fraction::Float64=0.10, min_tail_blocks::Int=16)
    depth = Int(coeffs.kerneldepth)
    depth < 1 && return NaN

    nblocks = min(depth, max(min_tail_blocks, ceil(Int, tail_fraction * depth)))
    first_tail_block = depth - nblocks + 1
    tailmax = 0.0

    try
        for b in first_tail_block:depth
            for c in view(coeffs.cfs, Block(b))
                v = abs(Float64(c))
                isfinite(v) || return Inf
                tailmax = max(tailmax, v)
            end
        end
        return tailmax
    catch
        flat = collect(coeffs.cfs)
        isempty(flat) && return NaN
        ntail = min(length(flat), max(min_tail_blocks, ceil(Int, tail_fraction * length(flat))))
        first_tail = length(flat) - ntail + 1
        for i in first_tail:length(flat)
            v = abs(Float64(flat[i]))
            isfinite(v) || return Inf
            tailmax = max(tailmax, v)
        end
        return tailmax
    end
end

function triangular_kernel_coeffs_eval(coeffs, tx::Real, ty::Real)
    cfs = coeffs.cfs
    depth = Int(coeffs.kerneldepth)
    depth < 1 && return NaN
    depth == 1 && return Float64(first(view(cfs, Block(1))))

    recurrence = coeffs.recurrence
    Bx = getfield(recurrence, 2)
    By = getfield(recurrence, 3)
    B = getfield(recurrence, 4)
    Ctail = getfield(recurrence, 5)
    x = Float64(tx)
    y = Float64(ty)

    b2 = Vector{Float64}(view(cfs, Block(depth)))
    b1 = Vector{Float64}(view(cfs, Block(depth - 1)))
    b1 .+= x .* (Bx[depth - 1] * b2)
    b1 .+= y .* (By[depth - 1] * b2)
    b1 .-= B[depth - 1] * b2

    for level = (depth - 2):-1:1
        next = Vector{Float64}(view(cfs, Block(level)))
        next .+= x .* (Bx[level] * b1)
        next .+= y .* (By[level] * b1)
        next .-= B[level] * b1
        if level <= length(Ctail)
            next .-= Ctail[level] * b2
        end
        b2 = b1
        b1 = next
    end

    return b1[1]
end

function kernel_projection_grid_error(plan, coeffs, K; grid_len::Int=KERNEL_PROJECTION_GRID_LEN)
    err = 0.0
    xs = range(0.0, 1.0; length=grid_len)
    ys = range(0.0, 1.0; length=grid_len)

    for x in xs
        xf = Float64(x)
        for y in ys
            yf = Float64(y)
            yf <= xf || continue

            tx = coeffs.flip ? 1.0 - xf : xf
            ty = yf
            if tx < -1.0e-12 || ty < -1.0e-12 || tx + ty > 1.0 + 1.0e-12
                continue
            end

            approx = triangular_kernel_coeffs_eval(coeffs, tx, ty)
            target = Float64(K(xf, yf))
            err = max(err, abs(approx - target))
        end
    end

    return err
end

function kernel_resolution_status(projection_error, tail_indicator;
                                  target_tol::Float64=KERNEL_TAIL_TARGET_TOL,
                                  projection_tol::Float64=KERNEL_PROJECTION_TARGET_TOL)
    tail_ok = isfinite(tail_indicator) && tail_indicator <= target_tol
    projection_ok = isfinite(projection_error) && projection_error <= projection_tol
    return (tail_ok || projection_ok) ? "kernel_machine_resolved" : "kernel_not_machine_resolved"
end

function kernel_projection_diagnostics(depth::Int, K; flip::Bool=go_flip,
                                       target_tol::Float64=KERNEL_TAIL_TARGET_TOL,
                                       projection_tol::Float64=KERNEL_PROJECTION_TARGET_TOL)
    try
        local plan
        local coeffs
        local projection_error = NaN
        local tail_indicator = NaN
        local status = "kernel_projection_failed"

        projection_sec = @elapsed begin
            plan = precompute_triangular_kernel_plan(depth, flip)
            coeffs = triangular_kernel_coeffs_from_plan(plan, K)
            tail_indicator = kernel_coefficients_tail_indicator(coeffs)
            projection_error = kernel_projection_grid_error(plan, coeffs, K)
            status = kernel_resolution_status(
                projection_error,
                tail_indicator;
                target_tol=target_tol,
                projection_tol=projection_tol,
            )
        end

        return (
            kerneldepth=depth,
            realized_kerneldepth=coeffs.kerneldepth,
            kernel_projection_error=projection_error,
            kernel_tail_indicator=tail_indicator,
            kernel_projection_sec=projection_sec,
            status=status,
            plan=plan,
            coeffs=coeffs,
        )
    catch err
        return (
            kerneldepth=depth,
            realized_kerneldepth=missing,
            kernel_projection_error=NaN,
            kernel_tail_indicator=NaN,
            kernel_projection_sec=NaN,
            status="kernel_projection_failed: " * sprint(showerror, err),
            plan=nothing,
            coeffs=nothing,
        )
    end
end

function choose_kerneldepth_for_kernel(; target_tol::Float64=KERNEL_TAIL_TARGET_TOL,
                                       projection_tol::Float64=KERNEL_PROJECTION_TARGET_TOL,
                                       max_depth::Int=1024)
    depths = if kerneldepth_mode == "auto"
        [d for d in AUTO_KERNELDEPTH_CANDIDATES if d <= max_depth]
    else
        [Int(requested_kerneldepth)]
    end
    isempty(depths) && error("No kerneldepth candidates available")

    scan_rows = []
    selected = nothing
    kernel_select_sec = @elapsed begin
        for depth in depths
            diag = kernel_projection_diagnostics(
                depth,
                kernel_target;
                flip=go_flip,
                target_tol=target_tol,
                projection_tol=projection_tol,
            )
            push!(scan_rows, diag)
            if diag.coeffs !== nothing && diag.status == "kernel_machine_resolved"
                selected = diag
                break
            end
        end
    end

    if selected === nothing
        for diag in reverse(scan_rows)
            if diag.coeffs !== nothing
                selected = diag
                break
            end
        end
    end
    selected === nothing && error("Unable to construct any triangular kernel projection")

    return (
        selected_kerneldepth=selected.kerneldepth,
        realized_kerneldepth=selected.realized_kerneldepth,
        kernel_projection_error=selected.kernel_projection_error,
        kernel_tail_indicator=selected.kernel_tail_indicator,
        kernel_resolution_status=selected.status,
        kernel_select_sec=kernel_select_sec,
        scan_rows=scan_rows,
        plan=selected.plan,
        coeffs=selected.coeffs,
    )
end

function precompute_rhs_cache()
    coeffs, info = projection_coefficients(
        rhs_original,
        N_rhs_fixed;
        alpha=1,
        beta=2,
    )
    assert_all_finite("rhs_cache", coeffs)
    return coeffs, info
end

function precompute_solution_cache()
    coeffs, info = projection_coefficients(
        exact_original,
        N_rhs_fixed;
        alpha=0,
        beta=1,
    )
    assert_all_finite("exact_solution_cache", coeffs)
    return coeffs, info
end

function precompute_constant_cache()
    coeffs, info = projection_coefficients(
        x -> 1.0,
        N_rhs_fixed;
        alpha=0,
        beta=1,
    )
    assert_all_finite("constant_solution_cache", coeffs)
    return coeffs, info
end

function truncate_coefficients(cache::AbstractVector{<:Real}, n::Int)
    n <= length(cache) && return Vector{Float64}(@view cache[1:n])
    out = zeros(Float64, n)
    out[1:length(cache)] .= cache
    return out
end

function build_rhs_vector(ncoeff::Int, rhs_cache::AbstractVector{<:Real})
    rhs = zeros(Float64, ncoeff)
    rhs[1] = 1.0
    rhs[2:ncoeff] .= rhs_cache[1:(ncoeff - 1)]
    assert_all_finite("rhs", rhs)
    return rhs
end

# ------------------------------------------------------------
# Structural matrix diagnostics
# ------------------------------------------------------------

function _maxabs_value_stream(values)
    mx = 0.0
    saw_nan = false
    for v in values
        vf = Float64(v)
        if isnan(vf)
            saw_nan = true
        elseif isinf(vf)
            return Inf
        else
            mx = max(mx, abs(vf))
        end
    end
    return saw_nan ? NaN : mx
end

function _nonfinite_count_value_stream(values)
    nbad = 0
    for v in values
        isfinite(Float64(v)) || (nbad += 1)
    end
    return nbad
end

function _add_abs_to_row_sum!(row_sums::Vector{Float64}, row_has_nan::AbstractVector{Bool}, i::Int, v)
    vf = Float64(v)
    if isnan(vf)
        row_has_nan[i] = true
    elseif isinf(vf)
        row_sums[i] = Inf
    else
        row_sums[i] += abs(vf)
    end
    return nothing
end

function _finish_row_sum_inf(row_sums::Vector{Float64}, row_has_nan::AbstractVector{Bool})
    any(isinf, row_sums) && return Inf
    mx = isempty(row_sums) ? 0.0 : maximum(row_sums)
    return any(row_has_nan) ? NaN : mx
end

function _bandwidths_or_nothing(A)
    try
        return BandedMatrices.bandwidths(A)
    catch
        return nothing
    end
end

function _is_almostbanded_like(A)
    names = fieldnames(typeof(A))
    return (:bands in names) && (:fill in names)
end

function _scan_banded_structural(visit, A)
    n, m = size(A)
    bw = _bandwidths_or_nothing(A)
    bw === nothing && throw(ArgumentError("matrix does not expose banded bandwidths"))
    l, u = Int(bw[1]), Int(bw[2])
    @inbounds for j in 1:m
        ilo = max(1, j - u)
        ihi = min(n, j + l)
        for i in ilo:ihi
            visit(i, j, A[i, j])
        end
    end
    return nothing
end

function _scan_almostbanded_structural(visit, A)
    bands = getfield(A, :bands)
    fill = getfield(A, :fill)
    n, m = size(A)
    p = min(size(fill, 1), n)

    @inbounds for j in 1:m
        for i in 1:p
            visit(i, j, fill[i, j])
        end
    end

    bw = _bandwidths_or_nothing(bands)
    bw === nothing && throw(ArgumentError("AlmostBandedMatrix bands do not expose bandwidths"))
    l, u = Int(bw[1]), Int(bw[2])
    @inbounds for j in 1:m
        ilo = max(p + 1, j - u)
        ihi = min(n, j + l)
        for i in ilo:ihi
            visit(i, j, bands[i, j])
        end
    end
    return nothing
end

maxabs_matrix_structural(v::AbstractVector) = _maxabs_value_stream(v)
nonfinite_count_structural(v::AbstractVector) = _nonfinite_count_value_stream(v)
row_sum_inf_structural(v::AbstractVector) = _maxabs_value_stream(v)

function maxabs_matrix_structural(A::SparseMatrixCSC)
    return _maxabs_value_stream(A.nzval)
end

function nonfinite_count_structural(A::SparseMatrixCSC)
    return _nonfinite_count_value_stream(A.nzval)
end

function row_sum_inf_structural(A::SparseMatrixCSC)
    row_sums = zeros(Float64, size(A, 1))
    row_has_nan = falses(size(A, 1))
    @inbounds for j in 1:size(A, 2)
        for p in A.colptr[j]:(A.colptr[j + 1] - 1)
            _add_abs_to_row_sum!(row_sums, row_has_nan, A.rowval[p], A.nzval[p])
        end
    end
    return _finish_row_sum_inf(row_sums, row_has_nan)
end

function maxabs_matrix_structural(A::AbstractMatrix)
    if _is_almostbanded_like(A)
        mx_ref = Ref(0.0)
        saw_nan = Ref(false)
        _scan_almostbanded_structural(A) do _, _, v
            vf = Float64(v)
            if isnan(vf)
                saw_nan[] = true
            elseif isinf(vf)
                mx_ref[] = Inf
            elseif !isinf(mx_ref[])
                mx_ref[] = max(mx_ref[], abs(vf))
            end
        end
        return isinf(mx_ref[]) ? Inf : (saw_nan[] ? NaN : mx_ref[])
    end

    bw = _bandwidths_or_nothing(A)
    bw === nothing && throw(ArgumentError("unsupported matrix type for structural maxabs: $(typeof(A))"))
    mx_ref = Ref(0.0)
    saw_nan = Ref(false)
    _scan_banded_structural(A) do _, _, v
        vf = Float64(v)
        if isnan(vf)
            saw_nan[] = true
        elseif isinf(vf)
            mx_ref[] = Inf
        elseif !isinf(mx_ref[])
            mx_ref[] = max(mx_ref[], abs(vf))
        end
    end
    return isinf(mx_ref[]) ? Inf : (saw_nan[] ? NaN : mx_ref[])
end

function nonfinite_count_structural(A::AbstractMatrix)
    if _is_almostbanded_like(A)
        nbad = Ref(0)
        _scan_almostbanded_structural(A) do _, _, v
            isfinite(Float64(v)) || (nbad[] += 1)
        end
        return nbad[]
    end

    bw = _bandwidths_or_nothing(A)
    bw === nothing && throw(ArgumentError("unsupported matrix type for structural nonfinite count: $(typeof(A))"))
    nbad = Ref(0)
    _scan_banded_structural(A) do _, _, v
        isfinite(Float64(v)) || (nbad[] += 1)
    end
    return nbad[]
end

function row_sum_inf_structural(A::AbstractMatrix)
    row_sums = zeros(Float64, size(A, 1))
    row_has_nan = falses(size(A, 1))

    if _is_almostbanded_like(A)
        _scan_almostbanded_structural(A) do i, _, v
            _add_abs_to_row_sum!(row_sums, row_has_nan, i, v)
        end
        return _finish_row_sum_inf(row_sums, row_has_nan)
    end

    bw = _bandwidths_or_nothing(A)
    bw === nothing && throw(ArgumentError("unsupported matrix type for structural row sum: $(typeof(A))"))
    _scan_banded_structural(A) do i, _, v
        _add_abs_to_row_sum!(row_sums, row_has_nan, i, v)
    end
    return _finish_row_sum_inf(row_sums, row_has_nan)
end

# ------------------------------------------------------------
# Almost-banded conversion
# ------------------------------------------------------------

function estimate_almostbanded_structure(A::SparseMatrixCSC; dense_frac::Float64=0.02)
    n, m = size(A)
    n == m || throw(DimensionMismatch("A must be square"))

    I, J, V = findnz(A)
    row_nnz = zeros(Int, n)

    @inbounds for k in eachindex(V)
        !iszero(V[k]) && (row_nnz[I[k]] += 1)
    end

    dense_threshold = max(64, ceil(Int, dense_frac*n))
    dense_rows = findall(row_nnz .>= dense_threshold)
    top_rows = isempty(dense_rows) ? 1 : maximum(dense_rows)
    top_rows = max(top_rows, 1)

    lower_bw = 0
    upper_bw = 0
    @inbounds for k in eachindex(V)
        if !iszero(V[k])
            i = I[k]
            j = J[k]
            if i > top_rows
                lower_bw = max(lower_bw, i - j)
                upper_bw = max(upper_bw, j - i)
            end
        end
    end

    lower_bw = max(lower_bw, top_rows - 1)
    return (
        top_rows=top_rows,
        lower_bw=lower_bw,
        upper_bw=upper_bw,
        dense_threshold=dense_threshold,
    )
end

function sparse_to_almostbanded(
    A::SparseMatrixCSC;
    top_rows::Int,
    lower_bw::Int,
    upper_bw::Int,
)
    n, m = size(A)
    n == m || throw(DimensionMismatch("A must be square"))

    p = max(top_rows, 1)
    l = max(lower_bw, p - 1)
    u = max(upper_bw, 0)
    T = eltype(A)

    bands = BandedMatrix{T}(undef, (n, n), (l, u))
    fill!(bands, zero(T))
    fillpart = zeros(T, p, n)

    dropped = 0
    max_dropped = 0.0
    I, J, V = findnz(A)

    @inbounds for k in eachindex(V)
        i = I[k]
        j = J[k]
        v = V[k]
        if i <= p
            fillpart[i, j] = v
        elseif i - j <= l && j - i <= u
            bands[i, j] = v
        elseif !iszero(v)
            dropped += 1
            max_dropped = max(max_dropped, abs(v))
        end
    end

    dropped == 0 ||
        error("Almost-banded conversion dropped $dropped entries; max dropped = $max_dropped")

    return AlmostBandedMatrix(bands, fillpart), (
        top_rows=p,
        lower_bw=l,
        upper_bw=u,
    )
end

function to_fast_almost_banded(A::SparseMatrixCSC; dense_frac::Float64=0.02)
    structure = estimate_almostbanded_structure(A; dense_frac=dense_frac)
    AB, info = sparse_to_almostbanded(
        A;
        top_rows=structure.top_rows,
        lower_bw=structure.lower_bw,
        upper_bw=structure.upper_bw,
    )
    return AB, merge(structure, info)
end

# ------------------------------------------------------------
# Gutleb--Olver operator and system assembly
# ------------------------------------------------------------

function build_raw_volterra_operator_from_coeffs(coeffs, ncoeff::Int)
    Vraw = triVolterraFullKernelOpP01_strictband_from_coeffs(coeffs, ncoeff, go_flip)
    return Vraw
end

function build_vide_volterra_operator_from_raw(Vraw, ncoeff::Int)
    D = build_derivative_P01_to_P12(ncoeff)

    # Public GO VIDE helper.  Its semantics are D - V_K in the
    # Jacobi(1,2,0..1) body space, so subtract from D to recover V_K.
    body_minus = build_VIDE_LHS_sec522_strictband(Vraw, ncoeff)
    VK = D - body_minus
    return VK, D, body_minus
end

function build_vide_volterra_operator_from_coeffs(coeffs, ncoeff::Int)
    Vraw = build_raw_volterra_operator_from_coeffs(coeffs, ncoeff)
    VK, D, body_minus = build_vide_volterra_operator_from_raw(Vraw, ncoeff)
    return (
        VK=VK,
        D=D,
        Vraw=Vraw,
        body_minus=body_minus,
        coeffs=coeffs,
    )
end

function build_vide_system(VK, D, ncoeff::Int)
    C = Conversion(solution_space, body_space)[1:ncoeff, 1:ncoeff]
    body = (1.0/kappa)*D + C + VK
    A = assemble_semibanded_system(
        body,
        ncoeff;
        evaluation_row=build_initial_evaluation_row(ncoeff),
    )
    return A, body, C
end

# ------------------------------------------------------------
# Operator action validation
# ------------------------------------------------------------

function action_error_from_values(vals::AbstractVector{<:Real}, exact_fn)
    all(isfinite, vals) || return Inf
    fnum = Fun(body_space, vals)
    err = 0.0
    for x in range(0.0, 1.0; length=action_grid_M + 1)
        err = max(err, abs(fnum(x) - exact_fn(x)))
    end
    return err
end

function action_error(VK, coeffs::AbstractVector{<:Real}, exact_fn)
    vals = VK * coeffs
    return action_error_from_values(vals, exact_fn)
end

function solution_projection_error(coeffs::AbstractVector{<:Real}, exact_fn)
    fnum = Fun(solution_space, coeffs)
    err = 0.0
    for x in range(0.0, 1.0; length=action_grid_M + 1)
        err = max(err, abs(fnum(x) - exact_fn(x)))
    end
    return err
end

function action_diagnostics(
    target_ops,
    ncoeff::Int,
    exact_cache::AbstractVector{<:Real},
    constant_cache::AbstractVector{<:Real},
    one_kernel_coeffs,
)
    one_ops = build_vide_volterra_operator_from_coeffs(one_kernel_coeffs, ncoeff)
    u_one = truncate_coefficients(constant_cache, ncoeff)
    u_exact = truncate_coefficients(exact_cache, ncoeff)
    Vu_coeff = target_ops.VK * u_exact

    action_err_K1_u1 = action_error(one_ops.VK, u_one, x -> Float64(x))
    u_projection_err = solution_projection_error(u_exact, exact_original)
    action_err_target = action_error_from_values(Vu_coeff, x -> exact_integral_original(x))

    return action_err_K1_u1, action_err_target, u_projection_err, one_ops, u_exact, Vu_coeff
end

# ------------------------------------------------------------
# Solve one N
# ------------------------------------------------------------

function solve_one_N(
    N::Int,
    rhs_cache::AbstractVector{<:Real},
    exact_cache::AbstractVector{<:Real},
    constant_cache::AbstractVector{<:Real},
    target_kernel_coeffs,
    one_kernel_coeffs,
    kernel_selection,
)
    ncoeff = N + 1

    local target_ops
    local one_ops
    local action_err_K1_u1
    local action_err_target
    local u_projection_err
    local u_exact
    local Vu_coeff
    operator_build_sec = @elapsed begin
        target_ops = build_vide_volterra_operator_from_coeffs(target_kernel_coeffs, ncoeff)
        action_err_K1_u1, action_err_target, u_projection_err, one_ops, u_exact, Vu_coeff =
            action_diagnostics(
                target_ops,
                ncoeff,
                exact_cache,
                constant_cache,
                one_kernel_coeffs,
            )
    end

    local A
    local body
    local C
    system_build_sec = @elapsed begin
        A, body, C = build_vide_system(target_ops.VK, target_ops.D, ncoeff)
    end

    local rhs
    rhs_build_sec = @elapsed begin
        rhs = build_rhs_vector(ncoeff, rhs_cache)
    end

    local AB
    local ab_info
    ab_convert_sec = @elapsed begin
        AB, ab_info = to_fast_almost_banded(A)
    end

    local coeffs
    solve_sec = @elapsed begin
        coeffs = AB \ rhs
    end
    assert_all_finite("solution coefficients", coeffs)

    local err_inf = 0.0
    eval_sec = @elapsed begin
        u_num = Fun(solution_space, coeffs)
        for x in range(0.0, 1.0; length=grid_M + 1)
            err_inf = max(err_inf, abs(u_num(x) - exact_original(x)))
        end
    end

    total_sec = operator_build_sec +
                system_build_sec +
                rhs_build_sec +
                ab_convert_sec +
                solve_sec +
                eval_sec

    return (
        N=N,
        ncoeff=ncoeff,
        err_inf=err_inf,
        action_err_K1_u1=action_err_K1_u1,
        action_err_target=action_err_target,
        u_projection_err=u_projection_err,
        maxabs_Vraw=maxabs_matrix_structural(target_ops.Vraw),
        maxabs_VK=maxabs_matrix_structural(target_ops.VK),
        maxabs_D=maxabs_matrix_structural(target_ops.D),
        maxabs_C=maxabs_matrix_structural(C),
        maxabs_body=maxabs_matrix_structural(body),
        maxabs_A=maxabs_matrix_structural(A),
        maxabs_rhs=maxabs_matrix_structural(rhs),
        maxabs_u_coeff=maxabs_matrix_structural(u_exact),
        maxabs_Vu_coeff=maxabs_matrix_structural(Vu_coeff),
        row_sum_body_inf=row_sum_inf_structural(body),
        row_sum_A_inf=row_sum_inf_structural(A),
        nonfinite_VK=nonfinite_count_structural(target_ops.VK),
        nonfinite_body=nonfinite_count_structural(body),
        nonfinite_A=nonfinite_count_structural(A),
        operator_build_sec=operator_build_sec,
        system_build_sec=system_build_sec,
        rhs_build_sec=rhs_build_sec,
        ab_convert_sec=ab_convert_sec,
        solve_sec=solve_sec,
        eval_sec=eval_sec,
        total_sec=total_sec,
        nnz_VK=count_nnz(target_ops.VK),
        nnz_A=count_nnz(A),
        body_bandwidth=bandwidth_string(body),
        final_bandwidth=bandwidth_string(A),
        solver=solver_name,
        status="ok_almost_banded",
        kerneldepth=kernel_selection.selected_kerneldepth,
        selected_kerneldepth=kernel_selection.selected_kerneldepth,
        realized_kerneldepth=target_ops.coeffs.kerneldepth,
        kernel_projection_error=kernel_selection.kernel_projection_error,
        kernel_tail_indicator=kernel_selection.kernel_tail_indicator,
        kernel_resolution_status=kernel_selection.kernel_resolution_status,
        kernel_select_sec=kernel_selection.kernel_select_sec,
        omega=omega,
        kappa=kappa,
        N_rhs_fixed=N_rhs_fixed,
        operator_type=string(typeof(target_ops.VK)),
        system_type=string(typeof(A)),
        ab_type=string(typeof(AB)),
        ab_top_rows=ab_info.top_rows,
        ab_lower_bw=ab_info.lower_bw,
        ab_upper_bw=ab_info.upper_bw,
        k1_operator_type=string(typeof(one_ops.VK)),
    )
end

function failure_row(N::Int, err, kernel_selection)
    msg = sprint(showerror, err)
    return (
        N=N,
        ncoeff=N + 1,
        err_inf=NaN,
        action_err_K1_u1=NaN,
        action_err_target=NaN,
        u_projection_err=NaN,
        maxabs_Vraw=missing,
        maxabs_VK=missing,
        maxabs_D=missing,
        maxabs_C=missing,
        maxabs_body=missing,
        maxabs_A=missing,
        maxabs_rhs=missing,
        maxabs_u_coeff=missing,
        maxabs_Vu_coeff=missing,
        row_sum_body_inf=missing,
        row_sum_A_inf=missing,
        nonfinite_VK=missing,
        nonfinite_body=missing,
        nonfinite_A=missing,
        operator_build_sec=NaN,
        system_build_sec=NaN,
        rhs_build_sec=NaN,
        ab_convert_sec=NaN,
        solve_sec=NaN,
        eval_sec=NaN,
        total_sec=NaN,
        nnz_VK=missing,
        nnz_A=missing,
        body_bandwidth="",
        final_bandwidth="",
        solver=solver_name,
        status="failed: " * msg,
        kerneldepth=kernel_selection.selected_kerneldepth,
        selected_kerneldepth=kernel_selection.selected_kerneldepth,
        realized_kerneldepth=missing,
        kernel_projection_error=kernel_selection.kernel_projection_error,
        kernel_tail_indicator=kernel_selection.kernel_tail_indicator,
        kernel_resolution_status=kernel_selection.kernel_resolution_status,
        kernel_select_sec=kernel_selection.kernel_select_sec,
        omega=omega,
        kappa=kappa,
        N_rhs_fixed=N_rhs_fixed,
        operator_type="",
        system_type="",
        ab_type="",
        ab_top_rows=missing,
        ab_lower_bw=missing,
        ab_upper_bw=missing,
        k1_operator_type="",
    )
end

# ------------------------------------------------------------
# CSV writing
# ------------------------------------------------------------

function csv_escape(x)
    s = x === missing ? "" : string(x)
    if occursin(",", s) || occursin("\"", s) || occursin("\n", s) || occursin("\r", s)
        return "\"" * replace(s, "\"" => "\"\"") * "\""
    end
    return s
end

csv_float(x) = x isa AbstractFloat ? @sprintf("%.17e", x) : csv_escape(x)

const ERROR_COLUMNS = (
    :N,
    :ncoeff,
    :err_inf,
    :action_err_K1_u1,
    :action_err_target,
    :u_projection_err,
    :status,
    :kerneldepth,
    :selected_kerneldepth,
    :realized_kerneldepth,
    :kernel_projection_error,
    :kernel_tail_indicator,
    :kernel_resolution_status,
    :omega,
    :kappa,
    :N_rhs_fixed,
)

const TIME_COLUMNS = (
    :N,
    :ncoeff,
    :operator_build_sec,
    :system_build_sec,
    :rhs_build_sec,
    :ab_convert_sec,
    :solve_sec,
    :eval_sec,
    :total_sec,
    :nnz_VK,
    :nnz_A,
    :maxabs_Vraw,
    :maxabs_VK,
    :maxabs_D,
    :maxabs_C,
    :maxabs_body,
    :maxabs_A,
    :maxabs_rhs,
    :maxabs_u_coeff,
    :maxabs_Vu_coeff,
    :row_sum_body_inf,
    :row_sum_A_inf,
    :nonfinite_VK,
    :nonfinite_body,
    :nonfinite_A,
    :body_bandwidth,
    :final_bandwidth,
    :solver,
    :status,
    :kerneldepth,
    :selected_kerneldepth,
    :realized_kerneldepth,
    :kernel_select_sec,
    :kernel_projection_error,
    :kernel_tail_indicator,
    :kernel_resolution_status,
    :omega,
    :kappa,
    :N_rhs_fixed,
    :operator_type,
    :system_type,
    :ab_type,
    :ab_top_rows,
    :ab_lower_bw,
    :ab_upper_bw,
)

const KERNELDEPTH_SCAN_COLUMNS = (
    :kerneldepth,
    :realized_kerneldepth,
    :kernel_projection_error,
    :kernel_tail_indicator,
    :kernel_projection_sec,
    :status,
)

const MATRIX_DIAGNOSTIC_COLUMNS = (
    :N,
    :ncoeff,
    :kerneldepth,
    :maxabs_Vraw,
    :maxabs_VK,
    :maxabs_D,
    :maxabs_C,
    :maxabs_body,
    :maxabs_A,
    :maxabs_rhs,
    :maxabs_u_coeff,
    :maxabs_Vu_coeff,
    :row_sum_body_inf,
    :row_sum_A_inf,
    :nonfinite_VK,
    :nonfinite_body,
    :nonfinite_A,
    :action_err_target,
    :err_inf,
    :solver,
    :ab_type,
    :status,
)

function row_value(row, col::Symbol)
    return getproperty(row, col)
end

function write_rows_csv(path::AbstractString, columns, rows)
    open(path, "w") do io
        println(io, join(string.(columns), ","))
        for r in rows
            println(io, join((csv_float(row_value(r, c)) for c in columns), ","))
        end
    end
end

write_error_csv(path, rows) = write_rows_csv(path, ERROR_COLUMNS, rows)
write_time_csv(path, rows) = write_rows_csv(path, TIME_COLUMNS, rows)
write_kerneldepth_scan_csv(path, rows) = write_rows_csv(path, KERNELDEPTH_SCAN_COLUMNS, rows)
write_matrix_diagnostics_csv(path, rows) = write_rows_csv(path, MATRIX_DIAGNOSTIC_COLUMNS, rows)

function sci_string(x)
    x === missing && return "missing"
    x isa Real && return @sprintf("%.6e", Float64(x))
    return string(x)
end

function print_row(r)
    println("  N                    = ", r.N)
    @printf("  action_err_K1_u1     = %.6e\n", r.action_err_K1_u1)
    @printf("  action_err_target    = %.6e\n", r.action_err_target)
    @printf("  u_projection_err     = %.6e\n", r.u_projection_err)
    @printf("  err_inf              = %.6e\n", r.err_inf)
    @printf("  operator_build_sec   = %.6e\n", r.operator_build_sec)
    @printf("  rhs_build_sec        = %.6e\n", r.rhs_build_sec)
    @printf("  ab_convert_sec       = %.6e\n", r.ab_convert_sec)
    @printf("  solve_sec            = %.6e\n", r.solve_sec)
    @printf("  total_sec            = %.6e\n", r.total_sec)
    println("  nnz(VK), nnz(A)      = ", r.nnz_VK, ", ", r.nnz_A)
    println("  maxabs_VK/body/A     = ",
            sci_string(r.maxabs_VK), ", ", sci_string(r.maxabs_body), ", ", sci_string(r.maxabs_A))
    println("  maxabs_u/Vu coeff    = ",
            sci_string(r.maxabs_u_coeff), ", ", sci_string(r.maxabs_Vu_coeff))
    println("  row_sum_body/A_inf   = ",
            sci_string(r.row_sum_body_inf), ", ", sci_string(r.row_sum_A_inf))
    println("  nonfinite VK/body/A  = ", r.nonfinite_VK, ", ", r.nonfinite_body, ", ", r.nonfinite_A)
    println("  body/final bandwidth = ", r.body_bandwidth, ", ", r.final_bandwidth)
    println("  solver               = ", r.solver)
    println("  system_type          = ", r.system_type)
    println("  ab_type              = ", r.ab_type)
    println("  selected_kerneldepth = ", r.selected_kerneldepth)
    @printf("  kernel_projection_err= %.6e\n", r.kernel_projection_error)
    @printf("  kernel_tail_indicator= %.6e\n", r.kernel_tail_indicator)
    println("  kernel_status        = ", r.kernel_resolution_status)
    println("  status               = ", r.status)
end

# ------------------------------------------------------------
# Main run
# ------------------------------------------------------------

println("Selecting Gutleb--Olver triangular kerneldepth")
println("EX4_KERNELDEPTH     = ", kerneldepth_raw)
println("kerneldepth_mode    = ", kerneldepth_mode)
kernel_selection = choose_kerneldepth_for_kernel()
target_kernel_coeffs = kernel_selection.coeffs
one_kernel_plan = precompute_triangular_kernel_plan(1, go_flip)
one_kernel_coeffs = triangular_kernel_coeffs_from_plan(one_kernel_plan, kernel_one)
write_kerneldepth_scan_csv(kerneldepth_scan_csv, kernel_selection.scan_rows)
println("selected_kerneldepth= ", kernel_selection.selected_kerneldepth)
println("realized_kerneldepth= ", kernel_selection.realized_kerneldepth)
@printf("kernel_select_sec   = %.6e\n", kernel_selection.kernel_select_sec)
@printf("kernel_projection_error = %.6e\n", kernel_selection.kernel_projection_error)
@printf("kernel_tail_indicator   = %.6e\n", kernel_selection.kernel_tail_indicator)
println("kernel_status       = ", kernel_selection.kernel_resolution_status)
println("kerneldepth_scan_csv= ", kerneldepth_scan_csv)
println()

println("============================================================")
println("ex4 Gutleb--Olver + first-order tau method")
println("Equation: kappa^{-1} u' + u + V_K u = f, u(0)=1")
println("project_root        = ", project_root)
println("go_project          = ", go_project)
println("omega               = ", omega)
println("kappa               = ", kappa)
println("N range             = ", first(Ns), " : ", last(Ns), ", count = ", length(Ns))
println("quick_mode          = ", quick_mode)
println("EX4_DIAG_NLIST      = ", Ns_diag === nothing ? "" : join(Ns_diag, ","))
println("grid_M              = ", grid_M)
println("action_grid_M       = ", action_grid_M)
println("N_rhs_fixed         = ", N_rhs_fixed)
println("kerneldepth_mode    = ", kerneldepth_mode)
println("selected_kerneldepth= ", kernel_selection.selected_kerneldepth)
println("requested_kerneldepth = ", requested_kerneldepth)
println("flip                = ", go_flip)
println("solver              = ", solver_name)
println("solution basis      = Jacobi(0,1,0..1)")
println("body basis          = Jacobi(1,2,0..1)")
println("data_dir            = ", data_dir)
println("============================================================")

println()
println("Precomputing fixed RHS and action coefficient caches")
rhs_timed = @timed precompute_rhs_cache()
rhs_cache, rhs_info = rhs_timed.value
exact_timed = @timed precompute_solution_cache()
exact_cache, exact_info = exact_timed.value
constant_timed = @timed precompute_constant_cache()
constant_cache, constant_info = constant_timed.value
println("rhs_cache_sec       = ", rhs_timed.time)
println("exact_cache_sec     = ", exact_timed.time)
println("constant_cache_sec  = ", constant_timed.time)
println("rhs_basis           = ", rhs_info.basis)
println("solution_basis      = ", exact_info.basis)
println("constant_basis      = ", constant_info.basis)
println("rhs_cache_norm      = ", norm(rhs_cache))
println("exact_cache_norm    = ", norm(exact_cache))

rows = []

for (idx, N) in enumerate(Ns)
    println()
    @printf("Solving %3d / %3d\n", idx, length(Ns))

    row = try
        solve_one_N(
            N,
            rhs_cache,
            exact_cache,
            constant_cache,
            target_kernel_coeffs,
            one_kernel_coeffs,
            kernel_selection,
        )
    catch err
        failure_row(N, err, kernel_selection)
    end

    push!(rows, row)
    print_row(row)

    write_error_csv(error_csv, rows)
    write_time_csv(time_csv, rows)
    write_matrix_diagnostics_csv(matrix_diagnostics_csv, rows)
end

println()
println("============================================================")
println("Finished.")
println("Wrote:")
println("  ", error_csv)
println("  ", time_csv)
println("  ", kerneldepth_scan_csv)
println("  ", matrix_diagnostics_csv)
println("============================================================")
