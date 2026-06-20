using FFTW
using LinearAlgebra
using Printf
using SparseArrays

const EX1_DIR = @__DIR__

function ex1_find_project_root(start_dir::AbstractString=@__DIR__)
    dir = abspath(start_dir)
    while true
        if isfile(joinpath(dir, "Project.toml")) && isdir(joinpath(dir, "src"))
            return dir
        end
        parent = dirname(dir)
        parent == dir && error("Could not find VoltOpt project root from $(start_dir).")
        dir = parent
    end
end

const EX1_PROJECT_ROOT = ex1_find_project_root()
const EX1_SRC_DIR = joinpath(EX1_PROJECT_ROOT, "src")
const EX1_GO_PROJECT = joinpath(EX1_PROJECT_ROOT, "others", "GutlebOlver")

include(joinpath(EX1_SRC_DIR, "VoltMat", "voltmat_cheb.jl"))
using .VoltMat_Cheb

const EX1_HAS_GO = Ref(false)
try
    @eval using Pkg
    Pkg.activate(EX1_GO_PROJECT; io=devnull)
    @eval using GutlebOlver
    @eval using ApproxFun
    @eval using BandedMatrices
    Pkg.activate(EX1_PROJECT_ROOT; io=devnull)
    EX1_HAS_GO[] = true
catch err
    try
        @eval using Pkg
        Pkg.activate(EX1_PROJECT_ROOT; io=devnull)
    catch
    end
    @warn "GutlebOlver could not be loaded; GO rows will be marked failed." exception=(err, catch_backtrace())
end

const EX1_HAS_FAST_AB = Ref(false)
try
    @eval using FastAlmostBandedMatrices
    if !isdefined(Main, :BandedMatrix)
        @eval using BandedMatrices
    end
    EX1_HAS_FAST_AB[] = true
catch err
    @warn "FastAlmostBandedMatrices/BandedMatrices unavailable; Cheb path falls back to sparse LU." exception=err
end

const EX1_HAS_FASTTRANSFORMS = Ref(false)
try
    @eval using FastTransforms
    EX1_HAS_FASTTRANSFORMS[] = true
catch err
    @warn "FastTransforms unavailable; GO Cheb-to-Jacobi RHS fallback is disabled." exception=err
end

const EX1_HAS_FASTGAUSSQUAD = Ref(false)
try
    @eval using FastGaussQuadrature
    EX1_HAS_FASTGAUSSQUAD[] = true
catch err
    @warn "FastGaussQuadrature unavailable; large piecewise direct projection is disabled." exception=err
end

# ---------------------------------------------------------------------------
# Problem definition copied from ex1_aging_maxwell.jl, with ASCII names.
# This is the second-kind Volterra equation
#     u(x) + int_{-1}^x K(x,y)u(y)dy = f(x),  x in [-1,1].
# ---------------------------------------------------------------------------

const EX1_X1 = -0.35
const EX1_X2 = 0.42
const EX1_A1 = 1.0
const EX1_A2 = -0.6
const EX1_DELTA1 = 1.0e-3
const EX1_DELTA2 = 1.5e-3

ex1_step(x, x0, delta) = 0.5 * (1.0 + tanh((x - x0) / delta))

function ex1_u_exact(x)
    return 1.0 +
           EX1_A1 * ex1_step(x, EX1_X1, EX1_DELTA1) +
           EX1_A2 * ex1_step(x, EX1_X2, EX1_DELTA2)
end

ex1_tau_xy(x, y) = 1.5 + x - 0.5 * y

function ex1_kernel(x, y)
    tau = ex1_tau_xy(x, y)
    return exp(-(x - y) / tau) / tau
end

ex1_x_from_s(s) = 2.0 * s - 1.0
ex1_s_from_x(x) = 0.5 * (x + 1.0)
ex1_u_go(s) = ex1_u_exact(ex1_x_from_s(s))
ex1_kernel_go(s, r) = 2.0 * ex1_kernel(ex1_x_from_s(s), ex1_x_from_s(r))

# ---------------------------------------------------------------------------
# Split high-order RHS quadrature.
# The split points match ex1_aging_maxwell.jl and explicitly isolate the two
# narrow transition layers in u_exact.
# ---------------------------------------------------------------------------

const EX1_RHS_Q = 160
const EX1_RHS_SPLIT_L = 40.0

function ex1_gausslegendre_rule(n::Int)
    n <= 0 && throw(ArgumentError("n must be positive"))
    n == 1 && return [0.0], [2.0]
    beta = [k / sqrt(4.0 * k^2 - 1.0) for k in 1:(n - 1)]
    eig = eigen(SymTridiagonal(zeros(n), beta))
    x = collect(eig.values)
    w = 2.0 .* abs2.(eig.vectors[1, :])
    p = sortperm(x)
    return x[p], w[p]
end

const EX1_GL_X, EX1_GL_W = ex1_gausslegendre_rule(EX1_RHS_Q)

function ex1_gauss_integral_on_interval(f, a::Float64, b::Float64)
    b > a || return 0.0
    c = 0.5 * (a + b)
    h = 0.5 * (b - a)
    acc = 0.0
    @inbounds for q in eachindex(EX1_GL_X)
        y = c + h * EX1_GL_X[q]
        acc += EX1_GL_W[q] * f(y)
    end
    return h * acc
end

function ex1_split_points_x(x::Float64)
    pts = Float64[-1.0, x]
    for (x0, delta) in ((EX1_X1, EX1_DELTA1), (EX1_X2, EX1_DELTA2))
        a = x0 - EX1_RHS_SPLIT_L * delta
        c = x0
        b = x0 + EX1_RHS_SPLIT_L * delta
        -1.0 < a < x && push!(pts, a)
        -1.0 < c < x && push!(pts, c)
        -1.0 < b < x && push!(pts, b)
    end
    sort!(pts)
    out = Float64[]
    sizehint!(out, length(pts))
    lastp = NaN
    for p in pts
        if isempty(out) || abs(p - lastp) > 10eps(Float64) * max(1.0, abs(p))
            push!(out, p)
            lastp = p
        end
    end
    return out
end

function ex1_rhs_integral_direct(x::Real)
    xf = Float64(x)
    xf <= -1.0 && return 0.0
    pts = ex1_split_points_x(xf)
    total = 0.0
    @inbounds for j in 1:(length(pts) - 1)
        a = pts[j]
        b = pts[j + 1]
        total += ex1_gauss_integral_on_interval(y -> ex1_kernel(xf, y) * ex1_u_exact(y), a, b)
    end
    return total
end

ex1_f_exact(x::Real) = ex1_u_exact(x) + ex1_rhs_integral_direct(x)
ex1_rhs_go(s::Real) = ex1_f_exact(ex1_x_from_s(s))

function ex1_layer_validation_grid(n::Int)
    pts = collect(range(0.0, 1.0; length=max(401, min(4001, 4n))))
    for (x0, delta) in ((EX1_X1, EX1_DELTA1), (EX1_X2, EX1_DELTA2))
        for scale in (-40.0, -20.0, -10.0, -2.0, -1.0, 0.0, 1.0, 2.0, 10.0, 20.0, 40.0)
            s = ex1_s_from_x(x0 + scale * delta)
            0.0 <= s <= 1.0 && push!(pts, s)
        end
    end
    return sort!(unique!(pts))
end

function ex1_projection_breakpoints(; radius_factor::Float64=80.0)
    pts = Float64[0.0, 1.0]
    for (x0, delta) in ((EX1_X1, EX1_DELTA1), (EX1_X2, EX1_DELTA2))
        for scale in (-radius_factor, -40.0, -20.0, -10.0, -5.0, -2.0, -1.0,
                      0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, radius_factor)
            s = ex1_s_from_x(x0 + scale * delta)
            0.0 < s < 1.0 && push!(pts, s)
        end
    end
    sort!(pts)
    out = Float64[]
    sizehint!(out, length(pts))
    lastp = NaN
    for p in pts
        if isempty(out) || abs(p - lastp) > 10eps(Float64) * max(1.0, abs(p))
            push!(out, p)
            lastp = p
        end
    end
    return out
end

function ex1_make_test_grid(; Mglobal=8001, Mlocal=5001, radius_factor=80.0)
    delta_max = max(EX1_DELTA1, EX1_DELTA2)
    radius = radius_factor * delta_max
    xg = collect(range(-1.0, 1.0; length=Mglobal))
    xl1 = collect(range(max(-1.0, EX1_X1 - radius), min(1.0, EX1_X1 + radius); length=Mlocal))
    xl2 = collect(range(max(-1.0, EX1_X2 - radius), min(1.0, EX1_X2 + radius); length=Mlocal))
    return sort(unique(vcat(xg, xl1, xl2, [EX1_X1, EX1_X2])))
end

const EX1_XTEST = ex1_make_test_grid()
const EX1_STEST = ex1_s_from_x.(EX1_XTEST)
const EX1_UTEST_EXACT = ex1_u_exact.(EX1_XTEST)
const EX1_UTEST_SCALE = maximum(abs.(EX1_UTEST_EXACT))

# ---------------------------------------------------------------------------
# Chebyshev helpers from ex1_aging_maxwell.jl.
# ---------------------------------------------------------------------------

function ex1_chebpts_lobatto(N::Int)
    N >= 1 || throw(ArgumentError("N must be positive"))
    n = N - 1
    n == 0 && return [0.0]
    return [-cospi(j / n) for j in 0:n]
end

function ex1_chebcoeffs_from_values(vals_increasing::AbstractVector{<:Real})
    N = length(vals_increasing)
    n = N - 1
    vals_cos_order = reverse(collect(vals_increasing))
    n == 0 && return copy(vals_cos_order)
    C = FFTW.r2r(vals_cos_order, FFTW.REDFT00)
    c = C ./ n
    c[1] *= 0.5
    c[end] *= 0.5
    return c
end

const EX1_MAX_CHEB_KERNEL_COEFF_SIZE = 100
const EX1_DEFAULT_CHEB_KERNEL_COEFF_SIZE = 32
const EX1_DEFAULT_CHEB_PATH = :aging_maxwell_fft
const EX1_DEFAULT_CHEB_ASSEMBLY = :fft

function ex1_absorbed_kernel(x::Real, y::Real)
    return ex1_kernel(x, (1.0 + x) * (1.0 + y) / 2.0 - 1.0)
end

function ex1_fixed_cheb2_coeffmat(K, M::Int)
    2 <= M <= EX1_MAX_CHEB_KERNEL_COEFF_SIZE ||
        throw(ArgumentError("kernel coefficient size must be in 2:$(EX1_MAX_CHEB_KERNEL_COEFF_SIZE), got $M"))
    nodes = [cospi((i - 1) / (M - 1)) for i in 1:M]
    vals = Matrix{Float64}(undef, M, M)
    @inbounds for j in 1:M
        y = nodes[j]
        for i in 1:M
            vals[i, j] = K(nodes[i], y)
        end
    end
    C = FFTW.r2r(vals, FFTW.REDFT00)
    C ./= (M - 1)^2
    C[1, :] ./= 2.0
    C[end, :] ./= 2.0
    C[:, 1] ./= 2.0
    C[:, end] ./= 2.0
    return C
end

function ex1_build_cheb_kernel_coeff_cache(M::Int=EX1_DEFAULT_CHEB_KERNEL_COEFF_SIZE)
    local C
    t_coeff = @elapsed begin
        C = ex1_fixed_cheb2_coeffmat(ex1_absorbed_kernel, M)
    end
    rows, cols = size(C)
    return (
        C=C,
        kernel_coeff_rows=rows,
        kernel_coeff_cols=cols,
        kernel_coeff_time=t_coeff,
        requested_kernel_coeff_size=M,
        source="fixed_lobatto_dct_absorbed_kernel",
    )
end

function ex1_build_adaptive_cheb_kernel_coeff_metadata()
    local C
    t_coeff = @elapsed begin
        C = VoltMat_Cheb.adaptive_cheb2_coeffmat(ex1_absorbed_kernel)
    end
    rows, cols = size(C)
    return (
        C=C,
        kernel_coeff_rows=rows,
        kernel_coeff_cols=cols,
        kernel_coeff_time=t_coeff,
        requested_kernel_coeff_size=missing,
        source="adaptive_cheb2_coeffmat_absorbed_kernel_metadata_only",
    )
end

function ex1_kernel_cache_matrix(kernel_cache)
    if kernel_cache isa AbstractMatrix
        return Matrix{Float64}(kernel_cache)
    elseif hasproperty(kernel_cache, :C)
        return getproperty(kernel_cache, :C)
    end
    throw(ArgumentError("kernel_cache must be a coefficient matrix or a cache with field C"))
end

function ex1_kernel_cache_time(kernel_cache)
    return hasproperty(kernel_cache, :kernel_coeff_time) ? Float64(getproperty(kernel_cache, :kernel_coeff_time)) : 0.0
end

function ex1_manual_cheb_path_assembly(cheb_path::Symbol)
    if cheb_path === :manual_C_fft
        return :fft
    elseif cheb_path === :compressed_rankupdate
        return :compressed_rankupdate
    end
    throw(ArgumentError("unsupported manual Cheb path $cheb_path"))
end

function ex1_cheb_eval_many(c::AbstractVector{<:Real}, x::AbstractVector{<:Real})
    M = length(x)
    N = length(c)
    N == 1 && return fill(c[1], M)
    b1 = zeros(Float64, M)
    b2 = zeros(Float64, M)
    b0 = zeros(Float64, M)
    @inbounds for k in N:-1:2
        ck = c[k]
        for i in 1:M
            b0[i] = 2.0 * x[i] * b1[i] - b2[i] + ck
        end
        b2, b1, b0 = b1, b0, b2
    end
    vals = zeros(Float64, M)
    @inbounds for i in 1:M
        vals[i] = x[i] * b1[i] - b2[i] + c[1]
    end
    return vals
end

function ex1_sparse_bandwidth(A::SparseMatrixCSC)
    lower = 0
    upper = 0
    @inbounds for j in 1:size(A, 2)
        for p in A.colptr[j]:(A.colptr[j + 1] - 1)
            if !iszero(A.nzval[p])
                i = A.rowval[p]
                lower = max(lower, i - j)
                upper = max(upper, j - i)
            end
        end
    end
    return lower, upper
end

# ---------------------------------------------------------------------------
# Almost-banded conversion for the Chebyshev system.
# Top dense rows are stored only in the AlmostBanded fill part; the banded
# component is intentionally left zero on those rows to avoid duplicate entries.
# ---------------------------------------------------------------------------

function ex1_estimate_almostbanded_structure(A::SparseMatrixCSC; dense_frac::Float64=0.02)
    n, m = size(A)
    n == m || throw(DimensionMismatch("A must be square"))
    I, J, V = findnz(A)
    row_nnz = zeros(Int, n)
    @inbounds for k in eachindex(V)
        !iszero(V[k]) && (row_nnz[I[k]] += 1)
    end
    dense_threshold = max(64, ceil(Int, dense_frac * n))
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
        row_nnz=row_nnz,
        dense_threshold=dense_threshold,
    )
end

function ex1_sparse_to_almostbanded(A::SparseMatrixCSC; top_rows::Int, lower_bw::Int, upper_bw::Int)
    EX1_HAS_FAST_AB[] || throw(ArgumentError("FastAlmostBandedMatrices is unavailable"))
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
        else
            dropped += 1
            max_dropped = max(max_dropped, abs(v))
        end
    end
    dropped == 0 || error("ex1_sparse_to_almostbanded dropped $dropped entries, max dropped = $max_dropped")
    return AlmostBandedMatrix(bands, fillpart)
end

function ex1_solve_cheb(
    N::Int;
    tol=eps(),
    kernel_cache=nothing,
    charge_kernel_coeff_time::Bool=false,
    assembly::Symbol=EX1_DEFAULT_CHEB_ASSEMBLY,
    cheb_path::Symbol=EX1_DEFAULT_CHEB_PATH,
    return_work::Bool=false,
)
    local xnodes, fvals, rhs_coeff, VK, Aop, sol, residual
    local structure, solver_name, top_rows, lower_bw, upper_bw
    local active_kernel_cache, C
    local kernel_coeff_rows, kernel_coeff_cols, kernel_coeff_size, kernel_order
    local cached_kernel_coeff, t_kernel_coeff, cheb_assembly, path_notes

    if cheb_path === :aging_maxwell_fft
        cheb_assembly = "fft"
        cached_kernel_coeff = false
        t_kernel_coeff = 0.0
        if kernel_cache !== nothing && hasproperty(kernel_cache, :kernel_coeff_rows)
            kernel_coeff_rows = getproperty(kernel_cache, :kernel_coeff_rows)
            kernel_coeff_cols = getproperty(kernel_cache, :kernel_coeff_cols)
            kernel_coeff_size = "$(kernel_coeff_rows)x$(kernel_coeff_cols)"
            kernel_order = max(kernel_coeff_rows, kernel_coeff_cols) - 1
            path_notes = "aging_maxwell_fft uses VoltMat_Cheb.voltmat_cheb_square(ex1_kernel, N - 1; tol=tol), matching ex1_aging_maxwell.jl; C metadata was computed separately and is not passed into VK construction."
        else
            kernel_coeff_rows = missing
            kernel_coeff_cols = missing
            kernel_coeff_size = missing
            kernel_order = missing
            path_notes = "aging_maxwell_fft uses VoltMat_Cheb.voltmat_cheb_square(ex1_kernel, N - 1; tol=tol), matching ex1_aging_maxwell.jl; C_not_exposed_by_aging_maxwell_fft_path."
        end
    elseif cheb_path === :manual_C_fft || cheb_path === :compressed_rankupdate
        if kernel_cache === nothing
            active_kernel_cache = ex1_build_cheb_kernel_coeff_cache()
            charge_kernel_coeff_time = true
            cached_kernel_coeff = false
        else
            active_kernel_cache = kernel_cache
            cached_kernel_coeff = !charge_kernel_coeff_time
        end
        C = ex1_kernel_cache_matrix(active_kernel_cache)
        kernel_coeff_rows, kernel_coeff_cols = size(C)
        kernel_coeff_rows <= EX1_MAX_CHEB_KERNEL_COEFF_SIZE &&
            kernel_coeff_cols <= EX1_MAX_CHEB_KERNEL_COEFF_SIZE ||
            throw(ArgumentError("kernel coefficient matrix is $(size(C)); dimensions must be <= $(EX1_MAX_CHEB_KERNEL_COEFF_SIZE)"))
        t_kernel_coeff = charge_kernel_coeff_time ? ex1_kernel_cache_time(active_kernel_cache) : 0.0
        assembly = ex1_manual_cheb_path_assembly(cheb_path)
        cheb_assembly = string(assembly)
        kernel_coeff_size = "$(kernel_coeff_rows)x$(kernel_coeff_cols)"
        kernel_order = max(kernel_coeff_rows, kernel_coeff_cols) - 1
        path_notes =
            cheb_path === :manual_C_fft ?
            "manual_C_fft uses fixed untrimmed 32x32 C with assembly=:fft; this is not the same as ex1_aging_maxwell default function-kernel FFT path and should not be used as evidence that FFT assembly is slow." :
            "compressed_rankupdate uses fixed 32x32 C with assembly=:compressed_rankupdate as a diagnostic comparison path."
    else
        throw(ArgumentError("unsupported Cheb path $cheb_path"))
    end

    t_rhs = @elapsed begin
        xnodes = ex1_chebpts_lobatto(N)
        fvals = ex1_f_exact.(xnodes)
        rhs_coeff = ex1_chebcoeffs_from_values(fvals)
    end
    t_matrix = @elapsed begin
        if cheb_path === :aging_maxwell_fft
            VK = VoltMat_Cheb.voltmat_cheb_square(ex1_kernel, N - 1; tol=tol)
        else
            VK = VoltMat_Cheb.voltmat_cheb_square(C, N - 1; assembly=assembly, tol=tol)
        end
    end
    t_system = @elapsed begin
        Aop = spdiagm(0 => ones(N)) + VK
    end
    t_convert = 0.0
    t_factor = 0.0
    t_solve = 0.0
    try
        if EX1_HAS_FAST_AB[]
            local Aab, F
            t_convert = @elapsed begin
                structure = ex1_estimate_almostbanded_structure(Aop)
                top_rows = structure.top_rows
                lower_bw = structure.lower_bw
                upper_bw = structure.upper_bw
                Aab = ex1_sparse_to_almostbanded(Aop; top_rows=top_rows, lower_bw=lower_bw, upper_bw=upper_bw)
            end
            t_factor = @elapsed begin
                F = qr(Aab)
            end
            t_solve = @elapsed begin
                sol = F \ rhs_coeff
            end
            solver_name = "fast_almost_banded_qr"
        else
            throw(ArgumentError("FastAlmostBandedMatrices unavailable"))
        end
    catch err
        local Fsp
        @warn "Cheb fast almost-banded solve failed; falling back to sparse LU." exception=(err, catch_backtrace())
        t_factor = @elapsed begin
            Fsp = lu(Aop)
        end
        t_solve = @elapsed begin
            sol = Fsp \ rhs_coeff
        end
        solver_name = "sparse_lu_fallback"
        structure = ex1_estimate_almostbanded_structure(Aop)
        top_rows = structure.top_rows
        lower_bw = structure.lower_bw
        upper_bw = structure.upper_bw
    end
    local max_abs_error, rel_error, residual_inf, residual_rel, uvals
    t_error = @elapsed begin
        uvals = ex1_cheb_eval_many(sol, EX1_XTEST)
        max_abs_error = maximum(abs.(uvals .- EX1_UTEST_EXACT))
        rel_error = max_abs_error / EX1_UTEST_SCALE
        residual = Aop * sol - rhs_coeff
        residual_inf = norm(residual, Inf)
        residual_rel = residual_inf / max(norm(rhs_coeff, Inf), eps(Float64))
    end
    raw_lb, raw_ub = ex1_sparse_bandwidth(VK)
    final_lb, final_ub = ex1_sparse_bandwidth(Aop)
    total_time = t_kernel_coeff + t_rhs + t_matrix + t_system + t_convert + t_factor + t_solve + t_error
    build_time = t_kernel_coeff + t_rhs + t_matrix + t_system + t_convert + t_factor
    row = (
        method="VoltMat_Cheb",
        N=N,
        cheb_path=string(cheb_path),
        kernel_order=kernel_order,
        kerneldepth=missing,
        kernel_coeff_size=kernel_coeff_size,
        kernel_coeff_rows=kernel_coeff_rows,
        kernel_coeff_cols=kernel_coeff_cols,
        kernel_coeff_time=t_kernel_coeff,
        cached_kernel_coeff=cached_kernel_coeff,
        cheb_assembly=cheb_assembly,
        rhs_projection_method="split_gauss_cheb_lobatto_dct",
        rhs_projection_status="pass",
        rhs_projection_validation_error=0.0,
        rhs_projection_time=t_rhs,
        rhs_projection_error_estimate=0.0,
        status="success",
        build_time=build_time,
        solve_time=t_solve,
        total_time=total_time,
        rhs_time=t_rhs,
        kernel_time=t_kernel_coeff,
        matrix_time=t_matrix,
        t_VK=t_matrix,
        matrix_assembly_time=t_matrix,
        system_time=t_system,
        convert_time=t_convert,
        to_fast_almost_banded_time=t_convert,
        factor_time=t_factor,
        error_time=t_error,
        max_abs_error=max_abs_error,
        relative_error=rel_error,
        residual_inf=residual_inf,
        residual_rel=residual_rel,
        nnz=nnz(Aop),
        rows=size(Aop, 1),
        cols=size(Aop, 2),
        top_rows=top_rows,
        lower_bw=lower_bw,
        upper_bw=upper_bw,
        bw_lower=final_lb,
        bw_upper=final_ub,
        raw_bw_lower=raw_lb,
        raw_bw_upper=raw_ub,
        solver=solver_name,
        notes="RHS uses the split Gauss-Legendre quadrature from ex1_aging_maxwell.jl; $(path_notes)",
        error_message="",
    )
    if return_work
        return (
            row=row,
            sol=sol,
            Aop=Aop,
            VK=VK,
            rhs_coeff=rhs_coeff,
            xnodes=xnodes,
        )
    end
    return row
end

# ---------------------------------------------------------------------------
# Gutleb--Olver helpers.
# The default RHS path uses the package fixed-length shifted Jacobi P^(1,0)
# construction with a validation grid concentrated near the layers.  When that
# validation fails at small N, an optional split-sampled least-squares fallback
# is tried.  The fallback is capped because a dense sampled Vandermonde is not
# appropriate for the large-N GO benchmark.
# ---------------------------------------------------------------------------

function ex1_go_space()
    EX1_HAS_GO[] || error("GutlebOlver/ApproxFun unavailable")
    return Jacobi(0, 1, 0..1)
end

function ex1_go_values(coeffs::AbstractVector, sgrid::AbstractVector)
    return Fun(ex1_go_space(), coeffs).(sgrid)
end

function ex1_go_projection_error(coeffs::AbstractVector, f, grid)
    vals = ex1_go_values(coeffs, grid)
    truth = f.(grid)
    return maximum(abs.(vals .- truth))
end

function ex1_go_split_sample_grid(N::Int)
    base_n = max(2N + 50, 600)
    pts = collect(range(0.0, 1.0; length=base_n))
    for (x0, delta) in ((EX1_X1, EX1_DELTA1), (EX1_X2, EX1_DELTA2))
        center = ex1_s_from_x(x0)
        radius = 0.5 * EX1_RHS_SPLIT_L * delta
        local_n = max(301, min(2001, N))
        append!(pts, range(max(0.0, center - radius), min(1.0, center + radius); length=local_n))
        for scale in (-40.0, -20.0, -10.0, -2.0, -1.0, 0.0, 1.0, 2.0, 10.0, 20.0, 40.0)
            s = ex1_s_from_x(x0 + scale * delta)
            0.0 <= s <= 1.0 && push!(pts, s)
        end
    end
    return sort!(unique!(pts))
end

function ex1_go_basis_matrix(space, grid::AbstractVector, N::Int)
    V = Matrix{Float64}(undef, length(grid), N)
    coeff = zeros(Float64, N)
    @inbounds for j in 1:N
        coeff[j] = 1.0
        V[:, j] .= Fun(space, coeff).(grid)
        coeff[j] = 0.0
    end
    return V
end

function ex1_project_go_rhs_split_lstsq(f, N::Int)
    max_lstsq_N = parse(Int, get(ENV, "EX1_GO_LSTSQ_MAX_N", "200"))
    N <= max_lstsq_N || error("split least-squares GO RHS projection capped at N <= $max_lstsq_N")
    space = ex1_go_space()
    grid = ex1_go_split_sample_grid(N)
    values = f.(grid)
    V = ex1_go_basis_matrix(space, grid, N)
    coeffs = V \ values
    validation_grid = ex1_layer_validation_grid(N)
    validation_error = ex1_go_projection_error(coeffs, f, validation_grid)
    info = (
        method=:split_sampled_least_squares,
        basis="Jacobi(0,1,0..1) internal representation of shifted P^(1,0)",
        coefficient_length=length(coeffs),
        tail_norm=norm(view(coeffs, max(1, N - 9):N)),
        validation_error=validation_error,
        status=validation_error <= parse(Float64, get(ENV, "EX1_GO_PROJECTION_TOL", "1e-8")) ? "pass" : "check",
        notes=("Split-sampled least-squares projection on a layer-refined grid; dense fallback capped for large N.",),
    )
    return coeffs, info
end

function ex1_project_go_rhs_cheb2jac_normalized(f, N::Int)
    EX1_HAS_FASTTRANSFORMS[] || error("FastTransforms is unavailable")
    tnodes = ex1_chebpts_lobatto(N)
    values = [f(0.5 * (t + 1.0)) for t in tnodes]
    cheb_coeff = ex1_chebcoeffs_from_values(values)

    # Direct unnormalized cheb2jac can overflow/produce NaN around N=30000.
    # The normalized Jacobi conversion remains finite; for the Jacobi(0,1,0..1)
    # basis used by the GO operator, the matching unnormalized coefficient is
    # normalized_coeff[k] * sqrt(k/2), with k in Julia's 1-based indexing.
    normalized_coeff = cheb2jac(cheb_coeff, 1, 0; normjac=true)
    rhs_coeff = similar(normalized_coeff)
    @inbounds for k in eachindex(normalized_coeff)
        rhs_coeff[k] = normalized_coeff[k] * sqrt(0.5 * k)
    end

    validation_grid = ex1_layer_validation_grid(N)
    validation_error = all(isfinite, rhs_coeff) ?
        ex1_go_projection_error(rhs_coeff, f, validation_grid) :
        NaN
    projection_tol = parse(Float64, get(ENV, "EX1_GO_PROJECTION_TOL", "1e-8"))
    info = (
        method=:split_cheb_lobatto_dct_normalized_cheb2jac,
        basis="Jacobi(0,1,0..1) internal representation of shifted P^(1,0)",
        coefficient_length=length(rhs_coeff),
        tail_norm=all(isfinite, rhs_coeff) ? norm(view(rhs_coeff, max(1, N - 9):N)) : NaN,
        validation_error=validation_error,
        status=isfinite(validation_error) && validation_error <= projection_tol ? "pass" : "check",
        notes=(
            "RHS values use the split Gauss-Legendre quadrature from ex1_aging_maxwell.jl.",
            "Chebyshev-Lobatto DCT coefficients are converted with FastTransforms cheb2jac(normjac=true) and rescaled by sqrt(k/2) to avoid large-N unnormalized overflow.",
        ),
    )
    return rhs_coeff, info
end

function ex1_piecewise_direct_quad_rule(order::Int)
    order > 0 || throw(ArgumentError("quadrature order must be positive"))
    if EX1_HAS_FASTGAUSSQUAD[]
        x, w = gausslegendre(order)
        return collect(Float64, x), collect(Float64, w)
    end
    order <= 512 || error("FastGaussQuadrature is unavailable; fallback rule is capped at order <= 512")
    return ex1_gausslegendre_rule(order)
end

function ex1_default_piecewise_quad_order(N::Int)
    raw = get(ENV, "EX1_PIECEWISE_QUAD_ORDER", "")
    !isempty(strip(raw)) && return parse(Int, raw)
    return N <= 8000 ? 4096 : N <= 16000 ? 16384 : 32768
end

@inline function ex1_jacobi_p10_next(pnm2, pnm1, x, n::Int)
    # Return P_n^(1,0)(x) from P_{n-2}, P_{n-1}; n is 0-based and n >= 2.
    nf = Float64(n)
    a = 2.0 * nf * (nf + 1.0) * (2.0 * nf - 1.0)
    b = (2.0 * nf) * ((2.0 * nf + 1.0) * (2.0 * nf - 1.0) * x + 1.0)
    c = 2.0 * nf * (nf - 1.0) * (2.0 * nf + 1.0)
    return (b * pnm1 - c * pnm2) / a
end

@inline function ex1_kahan_add!(sum::Vector{Float64}, comp::Vector{Float64}, i::Int, value::Float64)
    y = value - comp[i]
    t = sum[i] + y
    comp[i] = (t - sum[i]) - y
    sum[i] = t
    return nothing
end

function ex1_piecewise_direct_accumulate_float64!(moments, compensation, f, N::Int, qx, qw, breaks)
    npieces = length(breaks) - 1
    mquad = length(qx)
    total_nodes = npieces * mquad
    s_nodes = Vector{Float64}(undef, total_nodes)
    weighted_values = Vector{Float64}(undef, total_nodes)
    idx = 0
    for piece in 1:npieces
        a = breaks[piece]
        b = breaks[piece + 1]
        b > a || continue
        center = 0.5 * (a + b)
        half = 0.5 * (b - a)
        @inbounds for j in 1:mquad
            idx += 1
            s = center + half * qx[j]
            s_nodes[idx] = s
            weighted_values[idx] = half * qw[j] * (1.0 - s) * f(s)
        end
    end
    if idx < total_nodes
        resize!(s_nodes, idx)
        resize!(weighted_values, idx)
    end

    m = length(s_nodes)
    pnm2 = ones(Float64, m)
    moments[1] = dot(weighted_values, pnm2)
    N == 1 && return moments

    pnm1 = Vector{Float64}(undef, m)
    @inbounds for j in 1:m
        pnm1[j] = 3.0 * s_nodes[j] - 1.0
    end
    moments[2] = dot(weighted_values, pnm1)

    pn = Vector{Float64}(undef, m)
    t_nodes = @. 2.0 * s_nodes - 1.0
    @inbounds for n0 in 2:(N - 1)
        for j in 1:m
            pn[j] = ex1_jacobi_p10_next(pnm2[j], pnm1[j], t_nodes[j], n0)
        end
        moments[n0 + 1] = dot(weighted_values, pn)
        pnm2, pnm1, pn = pnm1, pn, pnm2
    end
    return moments
end

function ex1_piecewise_direct_accumulate_bigfloat!(moments, f, N::Int, qx, qw, breaks)
    setprecision(parse(Int, get(ENV, "EX1_PIECEWISE_BIGFLOAT_PRECISION", "128"))) do
        for piece in 1:(length(breaks) - 1)
            a = BigFloat(breaks[piece])
            b = BigFloat(breaks[piece + 1])
            b > a || continue
            center = (a + b) / 2
            half = (b - a) / 2
            m = length(qx)
            s_nodes = Vector{BigFloat}(undef, m)
            weighted_values = Vector{BigFloat}(undef, m)
            @inbounds for j in 1:m
                s = center + half * BigFloat(qx[j])
                s_nodes[j] = s
                weighted_values[j] = half * BigFloat(qw[j]) * (1 - s) * BigFloat(f(Float64(s)))
            end

            pnm2 = ones(BigFloat, m)
            moments[1] += sum(weighted_values[j] * pnm2[j] for j in 1:m)
            N == 1 && continue

            pnm1 = Vector{BigFloat}(undef, m)
            @inbounds for j in 1:m
                pnm1[j] = 3 * s_nodes[j] - 1
            end
            moments[2] += sum(weighted_values[j] * pnm1[j] for j in 1:m)

            pn = Vector{BigFloat}(undef, m)
            t_nodes = @. 2 * s_nodes - 1
            @inbounds for n0 in 2:(N - 1)
                nf = BigFloat(n0)
                aden = 2 * nf * (nf + 1) * (2 * nf - 1)
                for j in 1:m
                    bnum = (2 * nf) * ((2 * nf + 1) * (2 * nf - 1) * t_nodes[j] + 1)
                    cnum = 2 * nf * (nf - 1) * (2 * nf + 1)
                    pn[j] = (bnum * pnm1[j] - cnum * pnm2[j]) / aden
                end
                moments[n0 + 1] += sum(weighted_values[j] * pn[j] for j in 1:m)
                pnm2, pnm1, pn = pnm1, pn, pnm2
            end
        end
    end
    return moments
end

function ex1_project_go_rhs_piecewise_direct(
    f,
    N::Int;
    quad_order::Int=ex1_default_piecewise_quad_order(N),
    precision::Symbol=Symbol(get(ENV, "EX1_PIECEWISE_PRECISION", "float64")),
    endpoint_correction::Symbol=Symbol(get(ENV, "EX1_PIECEWISE_ENDPOINT_CORRECTION", "right")),
    validation_grid=ex1_layer_validation_grid(N),
)
    qx, qw = ex1_piecewise_direct_quad_rule(quad_order)
    breaks = ex1_projection_breakpoints()
    if precision == :bigfloat_accumulation
        moments_big = zeros(BigFloat, N)
        ex1_piecewise_direct_accumulate_bigfloat!(moments_big, f, N, qx, qw, breaks)
        coeffs = Vector{Float64}(undef, N)
        @inbounds for k in 1:N
            coeffs[k] = Float64(2 * BigFloat(k) * moments_big[k])
        end
    elseif precision == :float64
        moments = zeros(Float64, N)
        compensation = zeros(Float64, N)
        ex1_piecewise_direct_accumulate_float64!(moments, compensation, f, N, qx, qw, breaks)
        coeffs = Vector{Float64}(undef, N)
        @inbounds for k in 1:N
            coeffs[k] = 2.0 * k * moments[k]
        end
    else
        throw(ArgumentError("unsupported projection precision $precision"))
    end

    if endpoint_correction == :right && N >= 1
        endpoint_value = ex1_go_values(coeffs, [1.0])[1]
        coeffs[end] += (f(1.0) - endpoint_value) / N
    elseif endpoint_correction == :both && N >= 2
        endpoint_values = ex1_go_values(coeffs, [0.0, 1.0])
        e0 = f(0.0) - endpoint_values[1]
        e1 = f(1.0) - endpoint_values[2]
        left_prev = (-1.0)^(N - 2)
        left_last = (-1.0)^(N - 1)
        det = left_prev * N - left_last * (N - 1)
        dprev = (N * e0 - left_last * e1) / det
        dlast = (left_prev * e1 - (N - 1) * e0) / det
        coeffs[end - 1] += dprev
        coeffs[end] += dlast
    elseif endpoint_correction == :none
        nothing
    else
        throw(ArgumentError("unsupported endpoint correction $endpoint_correction"))
    end

    validation_error = all(isfinite, coeffs) ?
        ex1_go_projection_error(coeffs, f, validation_grid) :
        NaN
    projection_tol = parse(Float64, get(ENV, "EX1_GO_PROJECTION_TOL", "1e-8"))
    info = (
        method=:piecewise_direct_jacobi_projection,
        basis="Jacobi(0,1,0..1) internal representation of shifted P^(1,0)",
        coefficient_length=length(coeffs),
        tail_norm=all(isfinite, coeffs) ? norm(view(coeffs, max(1, N - 9):N)) : NaN,
        validation_error=validation_error,
        status=isfinite(validation_error) && validation_error <= projection_tol ? "pass" : "check",
        notes=(
            "Direct orthogonal projection c_k = 2(k+1) int g(s) P_k^(1,0)(2s-1) (1-s) ds.",
            "The integral is split at layer-refined breakpoints and evaluated with Gauss-Legendre quadrature.",
            "quad_order=$quad_order; precision=$precision; endpoint_correction=$endpoint_correction; pieces=$(length(breaks) - 1)",
        ),
    )
    return coeffs, info
end

function ex1_projection_score(coeffs, info)
    all(isfinite, coeffs) || return Inf
    err = info.validation_error
    err isa Missing && return Inf
    return isfinite(err) ? Float64(err) : Inf
end

function ex1_project_go_rhs(f, N::Int; method::Symbol=Symbol(get(ENV, "EX1_GO_RHS_METHOD", "auto")))
    projection_tol = parse(Float64, get(ENV, "EX1_GO_PROJECTION_TOL", "1e-8"))
    validation_grid = ex1_layer_validation_grid(N)
    if method == :piecewise_direct_jacobi_projection
        return ex1_project_go_rhs_piecewise_direct(f, N; validation_grid=validation_grid)
    elseif method == :cheb2jac_norm_fallback
        return ex1_project_go_rhs_cheb2jac_normalized(f, N)
    elseif method == :fixed_gauss_jacobi
        return build_go_rhs_coefficients(
            f,
            N;
            alpha=0,
            beta=1,
            domain=0..1,
            method=:fixed_gauss_jacobi,
            projection_tol=projection_tol,
            validate=true,
            validation_grid=validation_grid,
        )
    elseif method != :auto
        throw(ArgumentError("unsupported GO RHS projection method $method"))
    end

    rhs_coeff, rhs_info = build_go_rhs_coefficients(
        f,
        N;
        alpha=0,
        beta=1,
        domain=0..1,
        method=:fixed_gauss_jacobi,
        projection_tol=projection_tol,
        validate=true,
        validation_grid=validation_grid,
    )
    best_coeff = rhs_coeff
    best_info = rhs_info
    best_score = ex1_projection_score(best_coeff, best_info)

    if best_score <= projection_tol
        return best_coeff, best_info
    end

    try
        alt_coeff, alt_info = ex1_project_go_rhs_cheb2jac_normalized(f, N)
        alt_score = ex1_projection_score(alt_coeff, alt_info)
        if alt_score < best_score
            best_coeff = alt_coeff
            best_info = alt_info
            best_score = alt_score
        end
    catch err
        # Keep the fixed Jacobi projection unless a later small-N least-squares
        # fallback improves it.
    end

    if best_score <= projection_tol
        return best_coeff, best_info
    end

    if all(isfinite, best_coeff) && best_score < Inf
        # For large N, returning the best finite projection is more informative
        # than forcing a dense least-squares fallback that cannot scale.
        N > parse(Int, get(ENV, "EX1_GO_LSTSQ_MAX_N", "200")) && return best_coeff, best_info
    end

    if !(rhs_info.validation_error isa Missing) && rhs_info.validation_error <= projection_tol
        return rhs_coeff, rhs_info
    end
    try
        ls_coeff, ls_info = ex1_project_go_rhs_split_lstsq(f, N)
        return ex1_projection_score(ls_coeff, ls_info) < best_score ?
            (ls_coeff, ls_info) :
            (best_coeff, best_info)
    catch err
        notes = (
            "Selected best finite RHS projection did not validate on the layer grid.",
            "Split-sampled least-squares fallback was unavailable or failed: $(sprint(showerror, err))",
        )
        info = (
            method=best_info.method,
            basis=best_info.basis,
            coefficient_length=best_info.coefficient_length,
            tail_norm=best_info.tail_norm,
            validation_error=best_info.validation_error,
            status="check",
            notes=notes,
        )
        return best_coeff, info
    end
end

function ex1_solve_gutleb_olver(
    N::Int;
    kerneldepth::Int=parse(Int, get(ENV, "EX1_GO_KERNELDEPTH", "32")),
    rhs_projection_method::Symbol=Symbol(get(ENV, "EX1_GO_RHS_METHOD", "auto")),
)
    EX1_HAS_GO[] || error("GutlebOlver is unavailable")
    local rhs_coeff, rhs_info, plan, coeffs, Vraw, A, sol, residual
    t_rhs = @elapsed begin
        rhs_coeff, rhs_info = ex1_project_go_rhs(ex1_rhs_go, N; method=rhs_projection_method)
    end
    if !all(isfinite, rhs_coeff)
        msg = "GO RHS projection produced nonfinite shifted Jacobi coefficients"
        return (
            method="GutlebOlver_P10",
            N=N,
            cheb_path=missing,
            kernel_order=missing,
            kerneldepth=kerneldepth,
            kernel_coeff_size=missing,
            kernel_coeff_rows=missing,
            kernel_coeff_cols=missing,
            kernel_coeff_time=NaN,
            cached_kernel_coeff=false,
            cheb_assembly=missing,
            rhs_projection_method=string(rhs_info.method),
            rhs_projection_status="nonfinite",
            rhs_projection_validation_error=rhs_info.validation_error,
            rhs_projection_time=t_rhs,
            rhs_projection_error_estimate=rhs_info.validation_error,
            status="rhs_projection_failed",
            build_time=t_rhs,
            solve_time=NaN,
            total_time=t_rhs,
            rhs_time=t_rhs,
            kernel_time=NaN,
            matrix_time=NaN,
            t_VK=NaN,
            matrix_assembly_time=NaN,
            system_time=NaN,
            convert_time=0.0,
            to_fast_almost_banded_time=0.0,
            factor_time=0.0,
            error_time=NaN,
            max_abs_error=NaN,
            relative_error=NaN,
            residual_inf=NaN,
            residual_rel=NaN,
            nnz=missing,
            rows=N,
            cols=N,
            top_rows=missing,
            lower_bw=missing,
            upper_bw=missing,
            bw_lower=missing,
            bw_upper=missing,
            raw_bw_lower=missing,
            raw_bw_upper=missing,
            solver="not_run",
            notes=join(string.(rhs_info.notes), "; "),
            error_message=msg,
        )
    end
    t_kernel = @elapsed begin
        plan = precompute_triangular_kernel_plan(kerneldepth, true)
        coeffs = triangular_kernel_coeffs_from_plan(plan, ex1_kernel_go)
    end
    t_matrix = @elapsed begin
        Vraw = triVolterraFullKernelOpP01_strictband_from_coeffs(coeffs, N, true)
    end
    t_system = @elapsed begin
        A = build_secondkind_system_strictband(Vraw, N; sign=1.0)
    end
    t_solve = @elapsed begin
        sol = A \ rhs_coeff
    end
    local max_abs_error, rel_error, residual_inf, residual_rel, uvals
    t_error = @elapsed begin
        uvals = ex1_go_values(sol, EX1_STEST)
        max_abs_error = maximum(abs.(uvals .- EX1_UTEST_EXACT))
        rel_error = max_abs_error / EX1_UTEST_SCALE
        residual = A * sol - rhs_coeff
        residual_inf = norm(residual, Inf)
        residual_rel = residual_inf / max(norm(rhs_coeff, Inf), eps(Float64))
    end
    bw_lower, bw_upper = safe_bandwidth(A)
    build_time = t_rhs + t_kernel + t_matrix + t_system
    total_time = build_time + t_solve + t_error
    projection_status = string(rhs_info.status)
    status = projection_status == "pass" ? "success" : "projection_check"
    notes = join(string.(rhs_info.notes), "; ")
    return (
        method="GutlebOlver_P10",
        N=N,
        cheb_path=missing,
        kernel_order=missing,
        kerneldepth=kerneldepth,
        kernel_coeff_size=coeffs.kerneldepth,
        kernel_coeff_rows=missing,
        kernel_coeff_cols=missing,
        kernel_coeff_time=t_kernel,
        cached_kernel_coeff=false,
        cheb_assembly=missing,
        rhs_projection_method=string(rhs_info.method),
        rhs_projection_status=projection_status,
        rhs_projection_validation_error=rhs_info.validation_error,
        rhs_projection_time=t_rhs,
        rhs_projection_error_estimate=rhs_info.validation_error,
        status=status,
        build_time=build_time,
        solve_time=t_solve,
        total_time=total_time,
        rhs_time=t_rhs,
        kernel_time=t_kernel,
        matrix_time=t_matrix,
        t_VK=t_matrix,
        matrix_assembly_time=t_matrix,
        system_time=t_system,
        convert_time=0.0,
        to_fast_almost_banded_time=0.0,
        factor_time=0.0,
        error_time=t_error,
        max_abs_error=max_abs_error,
        relative_error=rel_error,
        residual_inf=residual_inf,
        residual_rel=residual_rel,
        nnz=safe_nnz(A),
        rows=size(A, 1),
        cols=size(A, 2),
        top_rows=missing,
        lower_bw=missing,
        upper_bw=missing,
        bw_lower=bw_lower,
        bw_upper=bw_upper,
        raw_bw_lower=missing,
        raw_bw_upper=missing,
        solver="banded_backslash",
        notes=notes,
        error_message="",
    )
end

const EX1_COMPARE_COLUMNS = [
    :method,
    :N,
    :cheb_path,
    :kernel_order,
    :kerneldepth,
    :kernel_coeff_size,
    :kernel_coeff_rows,
    :kernel_coeff_cols,
    :kernel_coeff_time,
    :cached_kernel_coeff,
    :cheb_assembly,
    :rhs_projection_method,
    :rhs_projection_status,
    :rhs_projection_validation_error,
    :rhs_projection_time,
    :rhs_projection_error_estimate,
    :status,
    :build_time,
    :solve_time,
    :total_time,
    :rhs_time,
    :kernel_time,
    :matrix_time,
    :t_VK,
    :matrix_assembly_time,
    :system_time,
    :convert_time,
    :to_fast_almost_banded_time,
    :factor_time,
    :error_time,
    :max_abs_error,
    :relative_error,
    :residual_inf,
    :residual_rel,
    :nnz,
    :rows,
    :cols,
    :top_rows,
    :lower_bw,
    :upper_bw,
    :bw_lower,
    :bw_upper,
    :raw_bw_lower,
    :raw_bw_upper,
    :solver,
    :notes,
    :error_message,
]

function ex1_csv_escape(x)
    if x === missing
        return ""
    elseif x isa AbstractFloat
        return isfinite(x) ? @sprintf("%.17e", x) : string(x)
    elseif x isa Integer
        return string(x)
    else
        s = string(x)
        if occursin(",", s) || occursin("\"", s) || occursin("\n", s) || occursin("\r", s)
            return "\"" * replace(s, "\"" => "\"\"") * "\""
        end
        return s
    end
end

function ex1_write_csv(path::AbstractString, rows)
    open(path, "w") do io
        println(io, join(string.(EX1_COMPARE_COLUMNS), ","))
        for row in rows
            vals = [ex1_csv_escape(getproperty(row, col)) for col in EX1_COMPARE_COLUMNS]
            println(io, join(vals, ","))
        end
    end
    return path
end

function ex1_failure_row(method::AbstractString, N::Int, err; status::AbstractString="failed")
    msg = sprint(showerror, err)
    return (
        method=method,
        N=N,
        cheb_path=missing,
        kernel_order=missing,
        kerneldepth=missing,
        kernel_coeff_size=missing,
        kernel_coeff_rows=missing,
        kernel_coeff_cols=missing,
        kernel_coeff_time=NaN,
        cached_kernel_coeff=false,
        cheb_assembly=missing,
        rhs_projection_method=missing,
        rhs_projection_status=missing,
        rhs_projection_validation_error=NaN,
        rhs_projection_time=NaN,
        rhs_projection_error_estimate=NaN,
        status=status,
        build_time=NaN,
        solve_time=NaN,
        total_time=NaN,
        rhs_time=NaN,
        kernel_time=NaN,
        matrix_time=NaN,
        t_VK=NaN,
        matrix_assembly_time=NaN,
        system_time=NaN,
        convert_time=NaN,
        to_fast_almost_banded_time=NaN,
        factor_time=NaN,
        error_time=NaN,
        max_abs_error=NaN,
        relative_error=NaN,
        residual_inf=NaN,
        residual_rel=NaN,
        nnz=missing,
        rows=missing,
        cols=missing,
        top_rows=missing,
        lower_bw=missing,
        upper_bw=missing,
        bw_lower=missing,
        bw_upper=missing,
        raw_bw_lower=missing,
        raw_bw_upper=missing,
        solver=missing,
        notes="",
        error_message=msg,
    )
end

function ex1_parse_int_list_env(name::AbstractString, default::Vector{Int})
    raw = get(ENV, name, "")
    isempty(strip(raw)) && return default
    return [parse(Int, strip(x)) for x in split(raw, ",") if !isempty(strip(x))]
end
