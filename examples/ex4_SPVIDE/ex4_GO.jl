############################################################
# ex4_GO.jl
#
# Solve the same oscillatory SPVIDE as ex4_VoltMat.jl,
# but use the Gutleb--Olver shifted-Jacobi Volterra machinery.
#
# Physical problem on t in [0,1]:
#
#   eps*v'(t) + (1+t)*v(t)
#     + ∫_0^t cos(omega*t*s) v(s) ds
#     = 2 + sin(pi*t),       v(0)=1.
#
# To use a pure second-kind Volterra Gutleb--Olver solve, this
# script integrates the equation from 0 to t.  Since v(0)=1,
# the equivalent second-kind equation is
#
#   v(t) + ∫_0^t K2(t,s) v(s) ds = g(t),
#
# where
#
#   g(t)  = 1 + [2t + (1-cos(pi*t))/pi] / eps,
#
#   K2(t,s) = [(1+s) + ∫_s^t cos(omega*r*s) dr] / eps.
#
# The inner integral is evaluated by the stable identity
#
#   ∫_s^t cos(omega*r*s) dr
#     = (t-s) cos(omega*s*(t+s)/2)
#       sinc(omega*s*(t-s)/2),       sinc(z)=sin(z)/z.
#
# RHS policy:
#   Before any N-loop solve, g(t) is projected once into the
#   shifted Jacobi P^(1,0) basis to a fixed long coefficient
#   vector.  Each solve then only truncates or zero-pads this
#   fixed vector.
#
# Conversion in ASCII notation:
#
#   v(t) + int_0^t K2(t,s) v(s) ds = g(t),
#
#   g(t) = 1 + (int_0^t [2 + sin(pi*r)] dr) / eps
#        = 1 + [2t + (1 - cos(pi*t))/pi] / eps,
#
#   K2(t,s) = ((1+s) + int_s^t cos(omega*r*s) dr) / eps,
#
#   int_s^t cos(omega*r*s) dr
#        = (t-s) cos(omega*s*(t+s)/2) sinc(omega*s*(t-s)/2).
#
# This is exactly the original SPVIDE integrated from 0 to t,
# with v(0)=1 used in the integration constant.
############################################################

using LinearAlgebra
using SparseArrays
using Printf
using FastAlmostBandedMatrices

script_dir = @__DIR__
include(joinpath(script_dir, "ex4_common.jl"))

function find_project_root(start_dir::AbstractString=@__DIR__)
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

VOLTERRA_ROOT = find_project_root()
DATA_DIR = joinpath(script_dir, "data")

# ============================================================
# 0. Parameters
# ============================================================

EPS = ex4_env_float("EX4_EPS", 1.0e-7)
OMEGA = ex4_env_float("EX4_OMEGA", 1.0)

N_FINAL = ex4_env_int("EX4_N_FINAL", 32_000)
N_COUNT = ex4_env_int("EX4_N_COUNT", 100)
# For quick local checks:
#   $env:EX4_N_FINAL = "32"; $env:EX4_N_COUNT = "4"

DROP_TOL = eps(Float64)
CAUCHY_REF_FACTOR = 1.01

# Gutleb--Olver kernel truncation.  Increase for larger OMEGA.
KERNELDEPTH = ex4_env_int("EX4_GO_KERNELDEPTH", 4)
FLIP = true

# Fixed shifted-Jacobi RHS cache.  It must be at least as long as
# the largest requested coefficient vector; extra modes are used only
# to verify that the RHS tail is at machine precision.
RHS_EXTRA_MODES = ex4_env_int("EX4_RHS_EXTRA_MODES", 512)
RHS_TAIL_LEN = 64
RHS_TOL = 100.0 * eps(Float64)
GO_RHS_FIXED_PROJECTION_LEN = ex4_env_int("EX4_GO_RHS_FIXED_PROJECTION_LEN", 4096)

# Grid-space diagnostics in shifted Jacobi coefficients use a common
# physical Chebyshev-Lobatto diagnostic grid as VoltMat.
COMPUTE_GRID_ERROR = true
GRID_M = ex4_env_int("EX4_GRID_M", 4096)
SHAPE_POINTS = ex4_env_int("EX4_SHAPE_POINTS", 1001)
WRITE_SHAPE = ex4_env_bool("EX4_WRITE_SHAPE", false)
SUMMARY_CSV = get(ENV, "EX4_SUMMARY_CSV", "")
SHAPE_CSV = get(ENV, "EX4_SHAPE_CSV", joinpath(DATA_DIR, "ex4_solution_shape.csv"))

N_LIST = ex4_default_n_list(N_FINAL, N_COUNT)

function reference_degree(N::Int; factor=CAUCHY_REF_FACTOR)
    return ex4_reference_degree(N; factor=factor)
end

function nextpow2_int(n::Integer)
    n <= 1 && return 1
    p = 1
    while p < n
        p <<= 1
    end
    return p
end

# ============================================================
# 1. Locate and load Gutleb--Olver implementation
# ============================================================

function include_or_load_gutleb_olver(root::AbstractString)
    go_project = joinpath(root, "others", "GutlebOlver")
    if isfile(joinpath(go_project, "Project.toml")) && !(go_project in LOAD_PATH)
        pushfirst!(LOAD_PATH, go_project)
    end

    try
        @eval using GutlebOlver
        return go_project
    catch err
        # Fall through to local includes.
    end

    candidates = String[
        joinpath(root, "others", "GutlebOlver", "src", "GutlebOlver.jl"),
        joinpath(root, "others", "GutlebOlver", "GutlebOlver.jl"),
        joinpath(root, "src", "GutlebOlver.jl"),
    ]

    for file in candidates
        if isfile(file)
            include(file)
            if isdefined(Main, :GutlebOlver)
                @eval using .GutlebOlver
            end
            return file
        end
    end

    error("Could not load GutlebOlver.  Tried package import and local paths: $(candidates)")
end

GO_SOURCE = include_or_load_gutleb_olver(VOLTERRA_ROOT)
@printf("Loaded GutlebOlver from: %s\n", GO_SOURCE)

function go_has(name::Symbol)
    if isdefined(Main, :GutlebOlver) && isdefined(getfield(Main, :GutlebOlver), name)
        return true
    end
    return isdefined(Main, name)
end

function go_get(name::Symbol)
    if isdefined(Main, :GutlebOlver) && isdefined(getfield(Main, :GutlebOlver), name)
        return getfield(getfield(Main, :GutlebOlver), name)
    elseif isdefined(Main, name)
        return getfield(Main, name)
    else
        error("GutlebOlver function $(name) is not available in this environment.")
    end
end

# ============================================================
# 2. Equivalent integrated second-kind problem
# ============================================================

@inline function sinc_unpi(z)
    az = abs(z)
    if az < 1.0e-8
        z2 = z * z
        return 1.0 - z2 / 6.0 + z2 * z2 / 120.0 - z2 * z2 * z2 / 5040.0
    else
        return sin(z) / z
    end
end

@inline function integrated_cos_kernel(t, s)
    # Stable form of ∫_s^t cos(omega*r*s) dr.
    half_delta = 0.5 * OMEGA * s * (t - s)
    mid_phase = 0.5 * OMEGA * s * (t + s)
    return (t - s) * cos(mid_phase) * sinc_unpi(half_delta)
end

@inline function K2_phys(t, s)
    # Kernel of v + V_{K2}v = g after division by eps.
    return ((1.0 + s) + integrated_cos_kernel(t, s)) / EPS
end

@inline function g_phys(t)
    F = 2.0 * t + (1.0 - cos(pi * t)) / pi
    return 1.0 + F / EPS
end

# ============================================================
# 3. Fixed shifted-Jacobi P^(1,0) RHS coefficient cache
# ============================================================

struct FixedGORHSCache
    coeffs::Vector{Float64}
    coeff_len::Int
    tail_norm::Float64
    relative_tail_norm::Float64
    refine_diff::Float64
    info::Any
end

function parse_rhs_projection_result(ret)
    if ret isa Tuple
        rhs = Vector{Float64}(ret[1])
        info = length(ret) >= 2 ? ret[2] : nothing
        return rhs, info
    elseif hasproperty(ret, :coeffs)
        return Vector{Float64}(getproperty(ret, :coeffs)), ret
    elseif hasproperty(ret, :rhs)
        return Vector{Float64}(getproperty(ret, :rhs)), ret
    else
        return Vector{Float64}(ret), nothing
    end
end

function build_fixed_go_rhs_cache_legacy(f, coeff_len::Int;
                                         alpha::Int=1,
                                         beta::Int=0,
                                         tail_len::Int=RHS_TAIL_LEN)
    build_rhs = go_get(:build_go_rhs_coefficients)

    ret = try
        build_rhs(f, coeff_len; alpha=alpha, beta=beta, method=:fixed_gauss_jacobi)
    catch err1
        try
            build_rhs(f, coeff_len; α=alpha, β=beta, method=:fixed_gauss_jacobi)
        catch err2
            @error "Failed to call build_go_rhs_coefficients with the expected keyword signatures." err1 err2
            rethrow(err2)
        end
    end

    rhs, info = parse_rhs_projection_result(ret)
    lt = min(tail_len, length(rhs))
    tail = norm(rhs[(end - lt + 1):end], Inf)
    rel_tail = tail / max(norm(rhs, Inf), eps(Float64))

    @printf("Fixed GO RHS cache: coeff_len = %d, tail_inf = %.6e, rel_tail_inf = %.6e\n",
            length(rhs), tail, rel_tail)

    if rel_tail > 100.0 * eps(Float64)
        @warn "The fixed GO RHS tail is not at relative machine precision. Increase RHS_EXTRA_MODES or RHS_FIXED_LEN." rel_tail RHS_EXTRA_MODES
    end

    return FixedGORHSCache(rhs, length(rhs), tail, rel_tail, 0.0, info)
end

function project_go_rhs_once(f, coeff_len::Int; alpha::Int=0, beta::Int=1)
    build_rhs = go_get(:build_go_rhs_coefficients)
    ret = build_rhs(
        f,
        coeff_len;
        alpha=alpha,
        beta=beta,
        method=:fixed_gauss_jacobi,
        validate=false,
    )
    return parse_rhs_projection_result(ret)
end

function build_fixed_go_rhs_cache(
    f,
    min_coeff_len::Int;
    alpha::Int=0,
    beta::Int=1,
    tol::Float64=RHS_TOL,
    max_coeff_len::Int=1 << 22,
    tail_len::Int=RHS_TAIL_LEN,
)
    min_coeff_len >= 1 || error("min_coeff_len must be positive.")
    if GO_RHS_FIXED_PROJECTION_LEN > 0
        coeff_len = nextpow2_int(max(128, GO_RHS_FIXED_PROJECTION_LEN))
        coeff_len <= max_coeff_len || error("Requested GO RHS projection length $(coeff_len) exceeds max_coeff_len=$(max_coeff_len).")

        prev_len = max(128, coeff_len ÷ 2)
        prev_rhs, _ = project_go_rhs_once(f, prev_len; alpha=alpha, beta=beta)
        rhs, info = project_go_rhs_once(f, coeff_len; alpha=alpha, beta=beta)

        lt = min(tail_len, length(rhs))
        tail = norm(rhs[(end - lt + 1):end], Inf)
        scale = max(norm(rhs, Inf), 1.0)
        rel_tail = tail / scale
        L = min(length(prev_rhs), length(rhs))
        refine = norm(prev_rhs[1:L] .- rhs[1:L], Inf) / scale

        @printf("  RHS shifted-Jacobi fixed projection length = %d, requested padded length = %d\n",
                length(rhs), min_coeff_len)
        @printf("  RHS fixed projection tail_inf = %.6e, rel_tail_inf = %.6e, refine_rel = %.6e\n",
                tail, rel_tail, refine)

        if !(isfinite(rel_tail) && rel_tail <= tol)
            @warn "The fixed GO RHS projection tail is not at relative machine precision." rel_tail tol coeff_len
        end
        if !(isfinite(refine) && refine <= 10.0 * tol)
            @warn "The fixed GO RHS projection refinement check is above tolerance." refine tol coeff_len
        end

        return FixedGORHSCache(rhs, length(rhs), tail, rel_tail, refine, info)
    end

    coeff_len = nextpow2_int(max(128, min_coeff_len + RHS_EXTRA_MODES))
    coeff_len <= max_coeff_len || error("Requested GO RHS length $(coeff_len) exceeds max_coeff_len=$(max_coeff_len).")

    prev_rhs = nothing
    last_info = nothing

    while true
        rhs, info = project_go_rhs_once(f, coeff_len; alpha=alpha, beta=beta)
        last_info = info

        lt = min(tail_len, length(rhs))
        tail = norm(rhs[(end - lt + 1):end], Inf)
        scale = max(norm(rhs, Inf), 1.0)
        rel_tail = tail / scale
        refine = Inf

        if prev_rhs !== nothing
            L = min(length(prev_rhs), length(rhs))
            refine = norm(prev_rhs[1:L] .- rhs[1:L], Inf) / scale
        end

        @printf("  RHS shifted-Jacobi cache trial length = %d, tail_inf = %.6e, rel_tail_inf = %.6e, refine_rel = %.6e\n",
                length(rhs), tail, rel_tail, refine)

        if coeff_len >= min_coeff_len && prev_rhs !== nothing && rel_tail <= tol && refine <= 10.0 * tol
            return FixedGORHSCache(rhs, length(rhs), tail, rel_tail, refine, info)
        end

        if coeff_len >= max_coeff_len
            @warn "GO RHS cache reached max_coeff_len before the requested tolerance was certified." coeff_len tail rel_tail refine tol
            return FixedGORHSCache(rhs, length(rhs), tail, rel_tail, refine, last_info)
        end

        prev_rhs = rhs
        coeff_len *= 2
    end
end

function fixed_coeff_slice(cfixed::AbstractVector{<:Real}, len::Int)
    out = zeros(Float64, len)
    m = min(len, length(cfixed))
    out[1:m] .= cfixed[1:m]
    return out
end

function rhs_go_coeffs_from_cache(cache::FixedGORHSCache, len::Int)
    return fixed_coeff_slice(cache.coeffs, len)
end

# ============================================================
# 4. Gutleb--Olver second-kind matrix construction
# ============================================================

function extract_matrix_from_go_object(obj)
    if obj isa AbstractMatrix
        return obj
    elseif obj isa Tuple
        for item in obj
            if item isa AbstractMatrix
                return item
            elseif hasproperty(item, :A)
                return getproperty(item, :A)
            elseif hasproperty(item, :matrix)
                return getproperty(item, :matrix)
            elseif hasproperty(item, :lhs)
                return getproperty(item, :lhs)
            end
        end
    else
        for name in (:A, :matrix, :lhs, :LHS)
            if hasproperty(obj, name)
                return getproperty(obj, name)
            end
        end
    end
    error("Could not extract a matrix from the GutlebOlver object of type $(typeof(obj)).")
end

function call_build_secondkind_system(Kfun, coeff_len::Int)
    build_secondkind = go_get(:build_secondkind_system_strictband)

    attempts = (
        () -> build_secondkind(Kfun, coeff_len; kerneldepth=KERNELDEPTH, flip=FLIP),
        () -> build_secondkind(Kfun, coeff_len; kerneldepth=KERNELDEPTH),
        () -> build_secondkind(coeff_len, Kfun; kerneldepth=KERNELDEPTH, flip=FLIP),
        () -> build_secondkind(coeff_len, Kfun; kerneldepth=KERNELDEPTH),
    )

    last_err = nothing
    for attempt in attempts
        try
            return attempt()
        catch err
            last_err = err
        end
    end

    @error "Could not call build_secondkind_system_strictband with supported signatures." coeff_len KERNELDEPTH FLIP last_err
    rethrow(last_err)
end

function call_lowlevel_go_volterra(Kfun, coeff_len::Int)
    # Fallback for older refactored versions where only the strict-band
    # triangular-kernel workflow is exported.
    planfun = go_get(:precompute_triangular_kernel_plan)
    coefffun = go_get(:triangular_kernel_coeffs_from_plan)
    opfun = go_get(:triVolterraFullKernelOpP01_strictband_from_coeffs)

    plan = try
        planfun(KERNELDEPTH, FLIP)
    catch
        planfun(KERNELDEPTH; flip=FLIP)
    end

    coeffs = try
        coefffun(plan, Kfun)
    catch
        coefffun(plan, Kfun; flip=FLIP)
    end

    V = try
        opfun(coeffs, coeff_len, FLIP)
    catch
        opfun(coeffs, coeff_len; flip=FLIP)
    end

    return spdiagm(0 => ones(Float64, coeff_len)) + sparse(Float64.(V))
end

function build_go_secondkind_matrix(coeff_len::Int)
    @printf("  Building Gutleb--Olver second-kind matrix, coeff_len = %d, kerneldepth = %d ...\n",
            coeff_len, KERNELDEPTH)

    if go_has(:build_secondkind_system_strictband)
        sys = call_build_secondkind_system(K2_phys, coeff_len)
        A = extract_matrix_from_go_object(sys)
        return sparse(Float64.(A))
    else
        @warn "build_secondkind_system_strictband not found; falling back to low-level strict-band Volterra operator."
        return call_lowlevel_go_volterra(K2_phys, coeff_len)
    end
end

struct FixedGOKernelCache
    coeffs::Any
    build_time_sec::Float64
end

function build_fixed_go_kernel_cache(Kfun)
    planfun = go_get(:precompute_triangular_kernel_plan)
    coefffun = go_get(:triangular_kernel_coeffs_from_plan)

    local coeffs
    build_time_sec = @elapsed begin
        plan = planfun(KERNELDEPTH, FLIP)
        coeffs = coefffun(plan, Kfun)
    end

    return FixedGOKernelCache(coeffs, build_time_sec)
end

function safe_nnz_go(A)
    if go_has(:safe_nnz)
        try
            return Int(go_get(:safe_nnz)(A))
        catch
            # Fall through to generic sparse diagnostics.
        end
    end

    try
        return nnz(A)
    catch
        return nnz(sparse(A))
    end
end

function build_go_secondkind_matrix(coeff_len::Int, kernel_cache::FixedGOKernelCache)
    @printf("  Building Gutleb--Olver second-kind matrix, coeff_len = %d, kerneldepth = %d ...\n",
            coeff_len, KERNELDEPTH)

    opfun = go_get(:triVolterraFullKernelOpP01_strictband_from_coeffs)
    build_secondkind = go_get(:build_secondkind_system_strictband)

    local Vraw, Aop, A
    build_operator_sec = @elapsed begin
        Vraw = try
            opfun(kernel_cache.coeffs, coeff_len, FLIP)
        catch
            opfun(kernel_cache.coeffs, coeff_len; flip=FLIP)
        end

        Aop = build_secondkind(Vraw, coeff_len; sign=1.0)
        A = sparse(Aop)
    end

    return A, safe_nnz_go(Vraw), build_operator_sec
end

# ============================================================
# 4b. FastAlmostBandedMatrices solve path
# ============================================================

function estimate_go_almostbanded_structure(
    A::SparseMatrixCSC;
    dense_frac::Float64=0.02,
    droptol::Float64=0.0,
)
    n, m = size(A)
    n == m || error("GO FastAlmostBanded solve expects a square matrix, got $(size(A)).")

    I, J, V = findnz(A)
    row_nnz = zeros(Int, n)

    @inbounds for k in eachindex(V)
        abs(V[k]) > droptol && (row_nnz[I[k]] += 1)
    end

    dense_threshold = max(64, ceil(Int, dense_frac * n))
    dense_rows = findall(row_nnz .>= dense_threshold)
    top_rows = isempty(dense_rows) ? 0 : maximum(dense_rows)

    lower_bw = 0
    upper_bw = 0

    @inbounds for k in eachindex(V)
        v = V[k]
        if abs(v) > droptol
            i = I[k]
            j = J[k]
            if i > top_rows
                lower_bw = max(lower_bw, i - j)
                upper_bw = max(upper_bw, j - i)
            end
        end
    end

    lower_bw = max(lower_bw, top_rows - 1, 0)
    upper_bw = max(upper_bw, 0)

    return (; top_rows, lower_bw, upper_bw, dense_threshold)
end

function sparse_to_go_fastalmostbanded(
    A::SparseMatrixCSC;
    top_rows::Int,
    lower_bw::Int,
    upper_bw::Int,
    droptol::Float64=0.0,
)
    n, m = size(A)
    n == m || error("GO FastAlmostBanded solve expects a square matrix, got $(size(A)).")

    T = eltype(A)
    p = clamp(top_rows, 0, n)
    l = max(lower_bw, p - 1, 0)
    u = max(upper_bw, 0)

    bands = BandedMatrix{T}(undef, (n, n), (l, u))
    fill!(bands, zero(T))
    fillpart = p == 0 ? zeros(T, 0, n) : zeros(T, p, n)

    dropped = 0
    max_dropped = 0.0
    I, J, V = findnz(A)

    @inbounds for k in eachindex(V)
        i = I[k]
        j = J[k]
        v = V[k]

        abs(v) <= droptol && continue

        if i <= p
            fillpart[i, j] = v
        elseif (i - j <= l) && (j - i <= u)
            bands[i, j] = v
        else
            dropped += 1
            max_dropped = max(max_dropped, abs(v))
        end
    end

    if dropped > 0
        error("FastAlmostBanded conversion dropped $(dropped) entries; max dropped = $(max_dropped).")
    end

    if p == 0
        return bands, (; top_rows=p, lower_bw=l, upper_bw=u, matrix_type=string(nameof(typeof(bands))))
    end

    Aab = AlmostBandedMatrix(bands, fillpart)
    return Aab, (; top_rows=p, lower_bw=l, upper_bw=u, matrix_type=string(nameof(typeof(Aab))))
end

function solve_go_fastalmostbanded(A::SparseMatrixCSC, rhs::AbstractVector{<:Real})
    structure = estimate_go_almostbanded_structure(A)
    @printf("  FastAlmostBandedMatrices path: top_rows = %d, lower_bw = %d, upper_bw = %d\n",
            structure.top_rows, structure.lower_bw, structure.upper_bw)

    local Afast, info, c
    convert_sec = @elapsed begin
        Afast, info = sparse_to_go_fastalmostbanded(
            A;
            top_rows=structure.top_rows,
            lower_bw=structure.lower_bw,
            upper_bw=structure.upper_bw,
        )
    end

    factor_solve_sec = @elapsed begin
        c = Afast \ rhs
    end

    @printf("  FastAlmostBanded matrix type = %s, convert_sec = %.6e, factor_solve_sec = %.6e\n",
            info.matrix_type, convert_sec, factor_solve_sec)

    return Vector{Float64}(c), convert_sec + factor_solve_sec, info
end

# ============================================================
# 5. Solve one degree
# ============================================================

function solve_one_N_go(
    N::Int;
    rhs_cache::FixedGORHSCache,
    kernel_cache::FixedGOKernelCache,
)
    @printf("\n============================================================\n")
    @printf("Gutleb--Olver solve, polynomial degree N = %d\n", N)
    @printf("eps = %.3e, omega = %.3e, kerneldepth = %d\n", EPS, OMEGA, KERNELDEPTH)

    try
        coeff_len = N + 1
        A, nnzVK, build_operator_sec = build_go_secondkind_matrix(coeff_len, kernel_cache)

        local rhs
        build_rhs_sec = @elapsed begin
            rhs = rhs_go_coeffs_from_cache(rhs_cache, coeff_len)
        end

        @printf("  size(A) = %d x %d, nnz(A) = %d\n", size(A, 1), size(A, 2), nnz(A))
        @printf("  FastAlmostBandedMatrices conversion and solve ...\n")

        local c
        c, solve_sec, solver_info = solve_go_fastalmostbanded(A, rhs)

        residual_inf = norm(A * c - rhs, Inf)
        status = "ok"
        if !all(isfinite, c) || !isfinite(residual_inf)
            status = "failed"
        end
        total_sec = build_operator_sec + build_rhs_sec + solve_sec
        @printf("  solver = fastalmostbanded, top_rows = %d, lower_bw = %d, upper_bw = %d\n",
                solver_info.top_rows, solver_info.lower_bw, solver_info.upper_bw)
        @printf("  residual_inf = %.6e, status = %s\n", residual_inf, status)

        return (; N, coeffs=c, residual_inf, nnzA=nnz(A), nnzVK,
                  build_operator_sec, build_rhs_sec, solve_sec, total_sec, status)
    catch err
        @warn "GO solve failed" N err
        coeff_len = N + 1
        c = fill(NaN, coeff_len)
        return (; N, coeffs=c, residual_inf=NaN, nnzA=0, nnzVK=0,
                  build_operator_sec=NaN, build_rhs_sec=NaN,
                  solve_sec=NaN, total_sec=NaN, status=ex4_status_from_error(err))
    end
end

# ============================================================
# 6. Cauchy diagnostics
# ============================================================

function cauchy_coeff_error_inf(cN::AbstractVector{<:Real},
                                cP::AbstractVector{<:Real})
    L = max(length(cN), length(cP))
    cN_pad = zeros(Float64, L)
    cP_pad = zeros(Float64, L)
    cN_pad[1:length(cN)] .= cN
    cP_pad[1:length(cP)] .= cP
    return norm(cN_pad .- cP_pad, Inf)
end

function tail_coeff_norm_inf(c::AbstractVector{<:Real}; fraction=0.10)
    L = length(c)
    tail_len = max(1, ceil(Int, fraction * L))
    first_tail = max(1, L - tail_len + 1)
    return norm(c[first_tail:end], Inf)
end

function shifted_jacobi_p10_eval(c::AbstractVector{<:Real}, t)
    M = length(c)
    M == 0 && return 0.0

    z = 2.0 * t - 1.0
    p_prev = 1.0
    acc = c[1] * p_prev

    M == 1 && return acc

    # P_1^(1,0)(z) = (1 + 3z)/2 = 3t - 1.
    p_curr = 0.5 * (1.0 + 3.0 * z)
    acc += c[2] * p_curr

    for n in 1:(M - 2)
        A = (n + 2.0) / (2.0 * n + 3.0)
        B = -1.0 / ((2.0 * n + 1.0) * (2.0 * n + 3.0))
        C = n / (2.0 * n + 1.0)
        p_next = ((z - B) * p_curr - C * p_prev) / A
        acc += c[n + 2] * p_next
        p_prev, p_curr = p_curr, p_next
    end

    return acc
end

function shifted_jacobi_values_on_lobatto_grid(c::AbstractVector{<:Real}, Mgrid::Int)
    t = ex4_diagnostic_physical_grid(Mgrid)
    vals = shifted_jacobi_values_on_t_grid(c, t)
    return vals
end

function shifted_jacobi_values_on_t_grid(c::AbstractVector{<:Real}, t::AbstractVector{<:Real})
    vals = Vector{Float64}(undef, length(t))
    for j in eachindex(t)
        vals[j] = shifted_jacobi_p10_eval(c, t[j])
    end
    return vals
end

function cauchy_grid_error_inf_go(cN::AbstractVector{<:Real},
                                  cP::AbstractVector{<:Real};
                                  Ndiag::Int)
    if !COMPUTE_GRID_ERROR
        return NaN
    end

    Mgrid = max(2, GRID_M)

    uN = shifted_jacobi_values_on_lobatto_grid(cN, Mgrid)
    uP = shifted_jacobi_values_on_lobatto_grid(cP, Mgrid)
    return maximum(abs.(uN .- uP))
end

# ============================================================
# 7. Run sequence
# ============================================================

function append_shape_rows_go(filename::AbstractString, N::Int, N_ref::Int,
                              cN::AbstractVector{<:Real}, cP::AbstractVector{<:Real})
    tgrid = ex4_shape_x_grid(SHAPE_POINTS)
    uN = shifted_jacobi_values_on_t_grid(cN, tgrid)
    uP = shifted_jacobi_values_on_t_grid(cP, tgrid)

    mkpath(dirname(filename))
    open(filename, "a") do io
        for i in eachindex(tgrid)
            row = ("GutlebOlver", N, tgrid[i], uN[i], uP[i], abs(uN[i] - uP[i]))
            println(io, join(ex4_csv_cell.(row), ","))
        end
        flush(io)
    end

    @printf("  shape rows appended to %s for N = %d, N_ref = %d\n", filename, N, N_ref)
end

function run_go_sequence(;
    N_list=N_LIST,
    result_csv::AbstractString=joinpath(DATA_DIR, "ex4_GO_results.csv"),
    summary_csv::AbstractString=SUMMARY_CSV,
    shape_csv::AbstractString=SHAPE_CSV,
    write_shape::Bool=WRITE_SHAPE,
)
    needed_N_sorted = ex4_collect_needed_degrees(N_list; factor=CAUCHY_REF_FACTOR)
    shape_N = ex4_env_int("EX4_SHAPE_N", maximum(N_list))

    @printf("\nDegrees to solve:\n")
    @printf("%s\n", string(needed_N_sorted))
    @printf("Diagnostic physical grid points = %d\n", GRID_M + 1)

    max_coeff_len = maximum(needed_N_sorted) + 1

    @printf("\nPrecomputing fixed shifted-Jacobi RHS coefficients, min coeff_len = %d ...\n", max_coeff_len)
    rhs_timed = @timed build_fixed_go_rhs_cache(g_phys, max_coeff_len; alpha=0, beta=1)
    rhs_cache = rhs_timed.value
    @printf("\nRHS fixed basis      = shifted Jacobi P^(1,0)\n")
    @printf("RHS fixed length     = %d\n", rhs_cache.coeff_len)
    @printf("RHS tail inf norm    = %.6e\n", rhs_cache.tail_norm)
    @printf("RHS tail relative    = %.6e\n", rhs_cache.relative_tail_norm)
    @printf("RHS refine relative  = %.6e\n", rhs_cache.refine_diff)
    @printf("RHS build time       = %.6e\n", rhs_timed.time)

    @printf("\nPrecomputing fixed Gutleb--Olver triangular kernel coefficients ...\n")
    kernel_cache = build_fixed_go_kernel_cache(K2_phys)
    @printf("Kernel fixed depth   = %d\n", KERNELDEPTH)
    @printf("Kernel build time    = %.6e\n", kernel_cache.build_time_sec)

    mkpath(DATA_DIR)
    ex4_write_csv_header(result_csv, ex4_result_columns)

    last_coeffs = Float64[]
    last_N = 0
    final_rows = NamedTuple[]

    for N in N_list
        N_ref = reference_degree(N)
        local solN, solP, e_coeff, e_grid, e_tail, pair_status

        pair_elapsed = @elapsed begin
            solN = solve_one_N_go(N; rhs_cache=rhs_cache, kernel_cache=kernel_cache)
            solP = solve_one_N_go(N_ref; rhs_cache=rhs_cache, kernel_cache=kernel_cache)

            pair_status = ex4_pair_status(solN.status, solP.status)
            e_coeff = cauchy_coeff_error_inf(solN.coeffs, solP.coeffs)
            e_grid = cauchy_grid_error_inf_go(solN.coeffs, solP.coeffs; Ndiag=N_ref)
            e_tail = tail_coeff_norm_inf(solN.coeffs)
        end

        build_operator_sec = solN.build_operator_sec + solP.build_operator_sec
        build_rhs_sec = solN.build_rhs_sec + solP.build_rhs_sec
        solve_sec = solN.solve_sec + solP.solve_sec
        nnzA = solN.nnzA
        nnzVK = solN.nnzVK

        row = (
            "GutlebOlver", N, N_ref, OMEGA, e_coeff, e_grid, e_tail,
            build_operator_sec, build_rhs_sec, solve_sec, pair_elapsed,
            nnzA, nnzVK, solN.residual_inf, pair_status,
        )
        ex4_append_csv_row(result_csv, row)

        if !isempty(summary_csv)
            ex4_append_csv_row(summary_csv, (
                "GutlebOlver", N, N_ref, e_coeff, e_grid, pair_elapsed, pair_status,
            ))
        end

        if write_shape && N == shape_N
            append_shape_rows_go(shape_csv, N, N_ref, solN.coeffs, solP.coeffs)
        end

        @printf(
            "\n[GutlebOlver] N = %d, N_ref = %d, cauchy_coeff = %.6e, cauchy_grid = %.6e, total_time_sec = %.6e, status = %s\n",
            N, N_ref, e_coeff, e_grid, pair_elapsed, pair_status,
        )
        @printf("[GutlebOlver] appended row to %s\n", result_csv)
        flush(stdout)

        last_coeffs = Vector{Float64}(solN.coeffs)
        last_N = N
        push!(final_rows, (; method="GutlebOlver", N, N_ref, cauchy_coeff_error=e_coeff,
            cauchy_grid_error=e_grid, total_time_sec=pair_elapsed, status=pair_status))
        GC.gc()
    end

    return (; result_csv, summary_csv, shape_csv, last_coeffs, last_N, final_rows,
              rhs_build_time_sec=rhs_timed.time,
              kernel_build_time_sec=kernel_cache.build_time_sec)
end

function write_coefficients_csv(filename, c)
    mkpath(dirname(filename))
    open(filename, "w") do io
        println(io, "n,c_n,abs_c_n")
        for (j, cj) in enumerate(c)
            @printf(io, "%d,%.17e,%.17e\n", j - 1, cj, abs(cj))
        end
        flush(io)
    end
    return filename
end

# ============================================================
# 8. Main
# ============================================================

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    out = run_go_sequence(N_list=N_LIST)

    tag = ex4_spvide_output_tag(EPS, OMEGA; kerneldepth=KERNELDEPTH)
    coeff_csv = joinpath(DATA_DIR, "ex4_GO_coefficients_N$(out.last_N)_$(tag).csv")

    write_coefficients_csv(coeff_csv, out.last_coeffs)

    @printf("\nDone.\n")
    @printf("Saved CSV outputs:\n")
    @printf("  %s\n", out.result_csv)
    @printf("  %s\n", coeff_csv)
    if !isempty(out.summary_csv)
        @printf("  %s\n", out.summary_csv)
    end
    if WRITE_SHAPE
        @printf("  %s\n", out.shape_csv)
    end
end
