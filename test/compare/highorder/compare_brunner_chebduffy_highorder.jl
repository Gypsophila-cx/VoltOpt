using LinearAlgebra
using Printf

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

const SCRIPT_DIR = @__DIR__
const PROJECT_ROOT = find_project_root(SCRIPT_DIR)
const DATA_DIR = joinpath(SCRIPT_DIR, "data")
const FIGURES_DIR = joinpath(SCRIPT_DIR, "figures")

mkpath(DATA_DIR)
mkpath(FIGURES_DIR)

include(joinpath(PROJECT_ROOT, "src", "VoltOpt.jl"))
using .VoltOpt

const WeakCheb = VoltOpt.Weak_VoltMat_Cheb

const MU = 0.5
const LAMBDA = 0.5
const BETA = 0.5
const SIGMA = 0.0
const LAMBDA_HAT = LAMBDA * 2.0^(1.0 - MU)

const RHS_MODE = get(ENV, "RHS_MODE", "ex7")
const BRUNNER_IMPL = lowercase(get(ENV, "BRUNNER_IMPL", "fast"))

const DEFAULT_BRUNNER_MS = [64, 128, 256, 512, 1024, 2048, 4096]
const BRUNNER_MS_CANDIDATES = [64, 128, 256, 512, 1024, 2048, 4096]
const DEFAULT_CHEB_NS = [32, 48, 64, 96, 128, 160, 192, 256, 320, 384, 512, 640, 768, 1024]
const DEFAULT_BRUNNER_M_VALUES = [2, 3, 4]
const DEFAULT_BRUNNER_Q_VALUES = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]
const DEFAULT_XGRID_N = parse(Int, get(ENV, "XGRID_N", "20001"))

const KERNEL_C_ROWS = parse(Int, get(ENV, "KERNEL_C_ROWS", "61"))
const KERNEL_C_COLS = parse(Int, get(ENV, "KERNEL_C_COLS", "41"))
const KERNEL_I = KERNEL_C_ROWS - 1
const KERNEL_J = KERNEL_C_COLS - 1

const RHS_PRECISION = parse(Int, get(ENV, "RHS_PRECISION", "256"))
const RHS_TS_H = get(ENV, "RHS_TS_H", "0.02")
const RHS_TS_ENDPOINT_EPS = get(ENV, "RHS_TS_ENDPOINT_EPS", string(eps(Float64)))
const RHS_TS_MAXK = parse(Int, get(ENV, "RHS_TS_MAXK", "100000"))
const RHS_SAMPLE_DEGREE_ENV = get(ENV, "RHS_SAMPLE_DEGREE", "")
const RHS_CHECK_SAMPLE_DEGREE_ENV = get(ENV, "RHS_CHECK_SAMPLE_DEGREE", "")
const RHS_TARGET_DEGREE = Ref(parse(Int, get(ENV, "RHS_TARGET_DEGREE", "0")))
const COND_MAX_N = parse(Int, get(ENV, "COND_MAX_N", "192"))

const _RHS_COEFFS = Ref{Union{Nothing,Vector{Float64}}}(nothing)
const _RHS_INFO = Ref{Any}(nothing)
const _RHS_VALUE_CACHE = Dict{Tuple{Float64,String},Float64}()
const _MESH_WARNING_KEYS = Set{Tuple{Int,Float64}}()

# ============================================================
# Exact solution and kernels
# ============================================================

function u_exact_x(x)
    xp1 = max(Float64(x) + 1.0, 0.0)
    return sqrt(xp1) * sin(100.0 * Float64(x))
end

function u_exact_s(s)
    sf = max(Float64(s), 0.0)
    return sqrt(2.0 * sf) * sin(100.0 * (2.0 * sf - 1.0))
end

Kfun(x, y) = inv((Float64(x) + 1.0)^2 + (Float64(y) - 1.0)^2)
Khat(s, t) = inv(4.0 * Float64(s)^2 + 4.0 * (Float64(t) - 1.0)^2)

function u_exact_s_big(s::BigFloat)
    s <= 0 && return zero(BigFloat)
    return sqrt(BigFloat(2) * s) * sin(BigFloat(100) * (BigFloat(2) * s - one(BigFloat)))
end

Khat_big(s::BigFloat, t::BigFloat) =
    inv(BigFloat(4) * s^2 + BigFloat(4) * (t - one(BigFloat))^2)

# ============================================================
# BigFloat tanh-sinh RHS in s variable
# ============================================================

function tanh_sinh_integrate01_big(f; hstr = RHS_TS_H, endpoint_eps_str = RHS_TS_ENDPOINT_EPS,
    maxK = RHS_TS_MAXK, precision = RHS_PRECISION)
    setprecision(precision) do
        h = BigFloat(hstr)
        endpoint_eps = BigFloat(endpoint_eps_str)
        half = BigFloat(0.5)
        pi2 = big(pi) / BigFloat(2)

        K = 0
        while true
            t = BigFloat(K) * h
            xip = tanh(pi2 * sinh(t))
            xim = tanh(pi2 * sinh(-t))
            if one(BigFloat) - xip <= endpoint_eps && xim + one(BigFloat) <= endpoint_eps
                break
            end
            K += 1
            K > maxK && error("tanh-sinh endpoint-distance truncation did not converge before maxK=$(maxK).")
        end

        total = zero(BigFloat)
        for k in -K:K
            t = BigFloat(k) * h
            a = pi2 * sinh(t)
            xi = tanh(a)
            z = (one(BigFloat) + xi) / BigFloat(2)
            (z <= 0 || z >= 1) && continue
            dzdt = half * pi2 * cosh(t) / cosh(a)^2
            total += f(z) * dzdt
        end
        value = h * total
        isfinite(Float64(value)) || error("Non-finite tanh-sinh integral.")
        return value, K
    end
end

function volterra_integral_s_big(s::BigFloat; mu = MU, hstr = RHS_TS_H)
    s <= 0 && return zero(BigFloat), 0
    setprecision(RHS_PRECISION) do
        mub = BigFloat(mu)
        alpha = inv(one(BigFloat) - mub)
        factor = s^(one(BigFloat) - mub) / (one(BigFloat) - mub)
        integral, K = tanh_sinh_integrate01_big(
            r -> begin
                t = s * (one(BigFloat) - r^alpha)
                Khat_big(s, t) * u_exact_s_big(t)
            end;
            hstr = hstr,
        )
        return factor * integral, K
    end
end

function G_manufactured_s_bigfloat(s; mu = MU, lambda = LAMBDA, hstr = RHS_TS_H)
    sf = Float64(s)
    sf <= 0.0 && return 0.0
    key = (sf, hstr)
    cached = get(_RHS_VALUE_CACHE, key, nothing)
    cached !== nothing && return cached

    val = setprecision(RHS_PRECISION) do
        sb = BigFloat(sf)
        I, _ = volterra_integral_s_big(sb; mu = mu, hstr = hstr)
        u_exact_s_big(sb) + BigFloat(lambda) * BigFloat(2)^(one(BigFloat) - BigFloat(mu)) * I
    end
    out = Float64(val)
    isfinite(out) || error("Non-finite manufactured RHS at s=$(sf).")
    _RHS_VALUE_CACHE[key] = out
    return out
end

G_manufactured_x(x) = G_manufactured_s_bigfloat((Float64(x) + 1.0) / 2.0)

function check_rhs_stability()
    println()
    println("Checking highorder RHS stability on 41 x-points...")
    x_test = collect(range(-1.0, 1.0; length = 41))
    vals_002 = [G_manufactured_s_bigfloat((x + 1.0) / 2.0; hstr = "0.02") for x in x_test]
    vals_0015 = [G_manufactured_s_bigfloat((x + 1.0) / 2.0; hstr = "0.015") for x in x_test]
    diff = maximum(abs.(vals_002 .- vals_0015))
    @printf("  max |G_h=0.02 - G_h=0.015| = %.6e\n", diff)
    if diff > 1e-10
        @warn "Highorder RHS h=0.02 differs from h=0.015 by more than 1e-10." diff
    end
    return (x_test = x_test, diff_002_0015 = diff)
end

# ============================================================
# Chebyshev/JFP utilities
# ============================================================

phi_beta_float(x::Real, beta::Real = BETA) =
    2.0 * ((1.0 + Float64(x)) / 2.0)^Float64(beta) - 1.0

phi_beta_s_float(s::Real, beta::Real = BETA) =
    2.0 * max(Float64(s), 0.0)^Float64(beta) - 1.0

invphi_s_float(z::Real, beta::Real = BETA) =
    ((1.0 + Float64(z)) / 2.0)^(1.0 / Float64(beta))

function cheb_eval(c::AbstractVector{<:Real}, z::Real)
    n = length(c) - 1
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

eval_jfp(c::AbstractVector{<:Real}, x::Real; beta::Real = BETA) =
    cheb_eval(c, phi_beta_float(x, beta))

eval_jfp_s(c::AbstractVector{<:Real}, s::Real; beta::Real = BETA) =
    cheb_eval(c, phi_beta_s_float(s, beta))

function cheb_project_lobatto_from_values(vals::AbstractVector{<:Real}, coeff_degree::Int)
    M = length(vals) - 1
    coeff_degree <= M || throw(ArgumentError("coeff_degree must not exceed sample degree."))
    c = zeros(Float64, coeff_degree + 1)
    @inbounds for k in 0:coeff_degree
        s = 0.0
        for j in 0:M
            wj = (j == 0 || j == M) ? 0.5 : 1.0
            s += wj * Float64(vals[j + 1]) * cos(pi * k * j / M)
        end
        c[k + 1] = 2.0 * s / M
    end
    c[1] /= 2.0
    return c
end

function rhs_coeffs_highorder(maxN::Int)
    cached = _RHS_COEFFS[]
    if cached !== nothing && length(cached) >= maxN + 1
        return cached[1:(maxN + 1)], _RHS_INFO[]
    end

    sample_degree = isempty(RHS_SAMPLE_DEGREE_ENV) ? max(2048, 2 * maxN) :
                    parse(Int, RHS_SAMPLE_DEGREE_ENV)
    sample_degree = max(sample_degree, maxN)
    check_degree = isempty(RHS_CHECK_SAMPLE_DEGREE_ENV) ? max(sample_degree ÷ 2, maxN) :
                   parse(Int, RHS_CHECK_SAMPLE_DEGREE_ENV)

    println()
    @printf("Generating highorder RHS coefficients: degree=%d, sample_degree=%d\n", maxN, sample_degree)
    rhs_stat = @timed begin
        vals = Vector{Float64}(undef, sample_degree + 1)
        @inbounds for j in 0:sample_degree
            z = cos(pi * j / sample_degree)
            s = invphi_s_float(z, BETA)
            vals[j + 1] = G_manufactured_s_bigfloat(s; hstr = RHS_TS_H)
        end
        vals
    end
    proj_stat = @timed cheb_project_lobatto_from_values(rhs_stat.value, maxN)
    coeffs = proj_stat.value

    check_coeffs = Float64[]
    check_diff = NaN
    if check_degree >= maxN && check_degree != sample_degree
        check_vals = Vector{Float64}(undef, check_degree + 1)
        @inbounds for j in 0:check_degree
            z = cos(pi * j / check_degree)
            s = invphi_s_float(z, BETA)
            check_vals[j + 1] = G_manufactured_s_bigfloat(s; hstr = RHS_TS_H)
        end
        check_coeffs = cheb_project_lobatto_from_values(check_vals, maxN)
        check_diff = maximum(abs.(coeffs .- check_coeffs))
    end

    tail_count = min(20, length(coeffs))
    info = (
        rhs_time = rhs_stat.time,
        projection_time = proj_stat.time,
        sample_degree = sample_degree,
        coeff_degree = maxN,
        check_sample_degree = check_degree,
        check_diff_inf = check_diff,
        tail_inf = maximum(abs, @view coeffs[(end - tail_count + 1):end]),
        tail_count = tail_count,
    )
    _RHS_COEFFS[] = coeffs
    _RHS_INFO[] = info
    @printf("  RHS BigFloat point generation time = %.6e s\n", info.rhs_time)
    @printf("  RHS Lobatto projection time = %.6e s\n", info.projection_time)
    @printf("  RHS tail last %d coeffs = %.3e\n", info.tail_count, info.tail_inf)
    isfinite(info.check_diff_inf) && @printf("  RHS high/check coeff diff inf = %.3e\n", info.check_diff_inf)
    return coeffs, info
end

function exact_solution_coeffs_jfp(N::Int)
    sample_degree = max(2048, 2 * N)
    vals = Vector{Float64}(undef, sample_degree + 1)
    @inbounds for j in 0:sample_degree
        z = cos(pi * j / sample_degree)
        vals[j + 1] = u_exact_s(invphi_s_float(z, BETA))
    end
    return cheb_project_lobatto_from_values(vals, N)
end

# ============================================================
# Fast Brunner implementation
# ============================================================

function radau_right_nodes(m::Int)
    m == 2 && return [1.0 / 3.0, 1.0]
    m == 3 && return [0.1550510257216822, 0.6449489742783178, 1.0]
    m == 4 && return [0.0885879595127042, 0.4094668644407347, 0.7876594617608471, 1.0]
    throw(ArgumentError("supported Brunner collocation orders are m = 2, 3, 4."))
end

validate_collocation_nodes(c) = Float64.(c)

function lagrange_basis(theta, j, c)
    v = 1.0
    cj = c[j]
    @inbounds for k in eachindex(c)
        k == j && continue
        v *= (theta - c[k]) / (cj - c[k])
    end
    return v
end

function lagrange_power_coeffs(c)
    cvec = validate_collocation_nodes(c)
    m = length(cvec)
    coeffs = zeros(Float64, m, m)
    @inbounds for j in 1:m
        poly = zeros(Float64, m)
        tmp = zeros(Float64, m)
        poly[1] = 1.0
        degree = 0
        denom = 1.0
        for k in 1:m
            k == j && continue
            fill!(tmp, 0.0)
            ck = cvec[k]
            for p in 0:degree
                tmp[p + 1] += -ck * poly[p + 1]
                tmp[p + 2] += poly[p + 1]
            end
            degree += 1
            copyto!(poly, tmp)
            denom *= cvec[j] - ck
        end
        for p in 0:(m - 1)
            coeffs[j, p + 1] = poly[p + 1] / denom
        end
    end
    return coeffs
end

function moment_power_asymptotic(delta::Float64, a::Float64, p::Int, mu::Float64)
    a <= 0.0 && return 0.0
    coeff = 1.0
    apow = a^(p + 1)
    dscale = delta^(-mu)
    acc = 0.0
    @inbounds for k in 0:80
        term = coeff * apow * dscale / (p + k + 1)
        acc += term
        abs(term) <= 4eps(Float64) * max(1.0, abs(acc)) && return acc
        coeff *= (mu + k) / (k + 1)
        apow *= a
        dscale /= delta
    end
    return acc
end

function moment_power(delta, a, p::Int, mu)
    deltaf = Float64(delta)
    af = Float64(a)
    muf = Float64(mu)
    af <= 0.0 && return 0.0
    if deltaf > 1e6 || af / deltaf <= 0.1
        return moment_power_asymptotic(deltaf, af, p, muf)
    end
    lower = deltaf - af
    lower = abs(lower) < 10eps(deltaf) ? 0.0 : max(lower, 0.0)
    acc = 0.0
    @inbounds for ridx in 0:p
        exponent = ridx + 1 - muf
        lower_term = lower == 0.0 ? 0.0 : lower^exponent
        signed_binom = isodd(ridx) ? -Float64(binomial(p, ridx)) : Float64(binomial(p, ridx))
        acc += signed_binom * deltaf^(p - ridx) * (deltaf^exponent - lower_term) / exponent
    end
    return acc
end

function product_weights_fast!(weights, moments, delta, theta_max, coeffs::AbstractMatrix{<:Real}, mu)
    m = size(coeffs, 1)
    @inbounds for p in 0:(m - 1)
        moments[p + 1] = moment_power(delta, theta_max, p, mu)
    end
    @inbounds for j in 1:m
        acc = 0.0
        for p in 1:m
            acc += Float64(coeffs[j, p]) * moments[p]
        end
        weights[j] = acc
    end
    return weights
end

function graded_mesh(M::Int, q::Real)
    [(n / M)^Float64(q) for n in 0:M]
end

function collocation_points(mesh, c)
    M = length(mesh) - 1
    m = length(c)
    nodes = Matrix{Float64}(undef, M, m)
    @inbounds for n in 1:M
        left = mesh[n]
        h = mesh[n + 1] - left
        for i in 1:m
            nodes[n, i] = left + c[i] * h
        end
    end
    return nodes
end

function solve_small_system!(x::Vector{Float64}, A::Matrix{Float64}, b::Vector{Float64}, m::Int)
    @inbounds begin
        for k in 1:m
            pivot = k
            pivot_abs = abs(A[k, k])
            for r in (k + 1):m
                ar = abs(A[r, k])
                if ar > pivot_abs
                    pivot = r
                    pivot_abs = ar
                end
            end
            pivot_abs > 0.0 || error("Singular local Brunner system.")
            if pivot != k
                for col in k:m
                    tmp = A[k, col]; A[k, col] = A[pivot, col]; A[pivot, col] = tmp
                end
                tmpb = b[k]; b[k] = b[pivot]; b[pivot] = tmpb
            end
            invpivot = inv(A[k, k])
            for r in (k + 1):m
                factor = A[r, k] * invpivot
                A[r, k] = 0.0
                for col in (k + 1):m
                    A[r, col] -= factor * A[k, col]
                end
                b[r] -= factor * b[k]
            end
        end
        for i in m:-1:1
            s = b[i]
            for col in (i + 1):m
                s -= A[i, col] * x[col]
            end
            x[i] = s / A[i, i]
        end
    end
    return x
end

struct BrunnerSolution
    mesh::Vector{Float64}
    c::Vector{Float64}
    Uvals::Matrix{Float64}
    M::Int
    m::Int
    q::Float64
    rhs_eval_time::Float64
    solve_time::Float64
    dof::Int
end

function solve_brunner_fast(M; m = 2, q = 2.0, rhs_coeffs)
    cvec = radau_right_nodes(m)
    mesh = graded_mesh(M, q)
    h = diff(mesh)
    hpow = h .^ (1.0 - MU)
    nodes = collocation_points(mesh, cvec)
    Uvals = zeros(Float64, M, m)
    power_coeffs = lagrange_power_coeffs(cvec)

    current_weights0 = Matrix{Float64}(undef, m, m)
    moments = zeros(Float64, m)
    row_weights = zeros(Float64, m)
    @inbounds for i in 1:m
        product_weights_fast!(row_weights, moments, cvec[i], cvec[i], power_coeffs, MU)
        for j in 1:m
            current_weights0[i, j] = row_weights[j]
        end
    end

    rhs_nodes = Matrix{Float64}(undef, M, m)
    rhs_eval_time = @elapsed begin
        @inbounds for n in 1:M
            for i in 1:m
                rhs_nodes[n, i] = eval_jfp_s(rhs_coeffs, nodes[n, i]; beta = BETA)
            end
        end
    end

    A = Matrix{Float64}(undef, m, m)
    b = Vector{Float64}(undef, m)
    xloc = Vector{Float64}(undef, m)
    hist_weights = zeros(Float64, m)
    hist_moments = zeros(Float64, m)

    solve_time = @elapsed begin
        @inbounds for n in 1:M
            for i in 1:m
                sni = nodes[n, i]
                history = 0.0
                for l in 1:(n - 1)
                    delta = (sni - mesh[l]) / h[l]
                    product_weights_fast!(hist_weights, hist_moments, delta, 1.0, power_coeffs, MU)
                    scale = hpow[l]
                    for j in 1:m
                        history += scale * hist_weights[j] * Khat(sni, nodes[l, j]) * Uvals[l, j]
                    end
                end
                b[i] = rhs_nodes[n, i] - LAMBDA_HAT * history
                for j in 1:m
                    A[i, j] = (i == j ? 1.0 : 0.0) +
                              LAMBDA_HAT * hpow[n] * current_weights0[i, j] * Khat(sni, nodes[n, j])
                end
            end
            solve_small_system!(xloc, A, b, m)
            for j in 1:m
                Uvals[n, j] = xloc[j]
            end
        end
    end

    return BrunnerSolution(mesh, cvec, Uvals, M, m, Float64(q), rhs_eval_time, solve_time, M * m)
end

function eval_brunner_solution(x_eval, sol::BrunnerSolution)
    s = clamp((Float64(x_eval) + 1.0) / 2.0, 0.0, 1.0)
    M = length(sol.mesh) - 1
    panel = clamp(searchsortedfirst(sol.mesh, s) - 1, 1, M)
    h = sol.mesh[panel + 1] - sol.mesh[panel]
    theta = (s - sol.mesh[panel]) / h
    u = 0.0
    @inbounds for j in eachindex(sol.c)
        u += sol.Uvals[panel, j] * lagrange_basis(theta, j, sol.c)
    end
    return u
end

# ============================================================
# Diagnostics, benchmarks, and IO
# ============================================================

function parse_env_int_list(env_name::AbstractString, default_values)
    raw = get(ENV, env_name, nothing)
    raw === nothing && return collect(default_values)
    vals = Int[]
    for piece in split(raw, ",")
        s = strip(piece)
        isempty(s) && continue
        v = tryparse(Int, s)
        v === nothing && error("Invalid integer in $(env_name): $(s)")
        push!(vals, v)
    end
    isempty(vals) && error("$(env_name) did not contain any values.")
    return vals
end

function parse_env_float_list(env_name::AbstractString, default_values)
    raw = get(ENV, env_name, nothing)
    raw === nothing && return collect(Float64, default_values)
    vals = Float64[]
    for piece in split(raw, ",")
        s = strip(piece)
        isempty(s) && continue
        v = tryparse(Float64, s)
        v === nothing && error("Invalid floating-point value in $(env_name): $(s)")
        push!(vals, v)
    end
    isempty(vals) && error("$(env_name) did not contain any values.")
    return vals
end

function env_limited_values(values, env_name::AbstractString; candidates = values)
    raw = get(ENV, env_name, nothing)
    raw === nothing && return values
    maxval = parse(Int, raw)
    filtered = [v for v in candidates if v <= maxval]
    isempty(filtered) && error("$(env_name)=$(maxval) removes all benchmark values.")
    return filtered
end

function mesh_diagnostics(M::Int, q::Real, c)
    mesh = graded_mesh(M, q)
    diffs = diff(mesh)
    hmin = minimum(diffs)
    hmax = maximum(diffs)
    mesh_valid = all(diffs .> 0.0)
    ratio = hmin > 0.0 ? hmax / hmin : Inf
    collapsed_x_count = 0
    min_positive_s = Inf
    min_positive_xplus1 = Inf
    node_count = M * length(c)
    @inbounds for n in 1:M
        left = mesh[n]
        h = mesh[n + 1] - left
        for ci in c
            s = left + ci * h
            s > 0.0 && (min_positive_s = min(min_positive_s, s))
            x = 2.0 * s - 1.0
            x == -1.0 && (collapsed_x_count += 1)
            xplus1 = x + 1.0
            xplus1 > 0.0 && (min_positive_xplus1 = min(min_positive_xplus1, xplus1))
        end
    end
    frac = collapsed_x_count / node_count
    return (
        hmin = hmin,
        hmax = hmax,
        mesh_ratio = ratio,
        mesh_valid = mesh_valid,
        collapsed_x_count = collapsed_x_count,
        collapsed_x_fraction = frac,
        min_positive_s = min_positive_s,
        min_positive_xplus1 = min_positive_xplus1,
        numerically_extreme = frac > 0.01,
    )
end

function evaluate_error(f, xgrid, exact_vals)
    max_error = 0.0
    sumsq = 0.0
    @inbounds for i in eachindex(xgrid, exact_vals)
        d = f(xgrid[i]) - exact_vals[i]
        ad = abs(d)
        max_error = max(max_error, ad)
        sumsq += d * d
    end
    return max_error, sqrt(sumsq / length(xgrid))
end

function build_kernel_coefficients()
    stat = @timed begin
        values = WeakCheb.sample_duffy_kernel(Kfun, KERNEL_I, KERNEL_J, BETA)
        WeakCheb.cheb_coeffs_lobatto_2d(values)
    end
    tail = try
        WeakCheb.coefficient_tail_info(stat.value)
    catch
        (row_rel = NaN, col_rel = NaN, max_coeff = maximum(abs, stat.value))
    end
    return stat.value, stat.time, tail
end

dense_matrix(V) = V isa VoltOpt.PackedLowerHessenberg ? VoltOpt.packed_to_dense(V) : Matrix(V)

function solve_chebduffy(N, C, rhs_coeffs, rhs_info, xgrid, exact_vals)
    rhsN = rhs_coeffs[1:(N + 1)]
    mat_stat = @timed begin
        V, _ = VoltOpt.weak_voltmat_cheb(
            C;
            C_is_coeffs = true,
            mu = MU,
            sigma = SIGMA,
            beta = BETA,
            N = N,
            diagnostics = false,
            output_format = :dense,
            return_info = true,
        )
        dense_matrix(V)
    end
    A = Matrix{Float64}(LinearAlgebra.I, N + 1, N + 1)
    A .+= LAMBDA .* mat_stat.value
    solve_stat = @timed A \ rhsN
    coeffs = solve_stat.value
    max_error, rms_error = evaluate_error(x -> eval_jfp(coeffs, x; beta = BETA), xgrid, exact_vals)
    method_time = rhs_info.projection_time + mat_stat.time + solve_stat.time
    total_time = rhs_info.rhs_time + method_time
    return (
        method = "Chebyshev-Duffy",
        dof = N + 1,
        N_or_M = N,
        N = N,
        m = missing,
        q = missing,
        mu = MU,
        lambda = LAMBDA,
        beta = BETA,
        rhs_time = rhs_info.rhs_time,
        projection_time = rhs_info.projection_time,
        construction_time = mat_stat.time,
        solve_time = solve_stat.time,
        method_time = method_time,
        total_time_including_rhs_generation = total_time,
        max_error = max_error,
        rms_error = rms_error,
        hmin = missing,
        hmax = missing,
        mesh_ratio = missing,
        mesh_valid = missing,
        collapsed_x_count = missing,
        collapsed_x_fraction = missing,
        min_positive_s = missing,
        min_positive_xplus1 = missing,
        numerically_extreme = missing,
    )
end

function run_brunner_sweep(Ms, m_values, q_values, rhs_coeffs, rhs_info, xgrid, exact_vals)
    rows = NamedTuple[]
    println()
    println("Running highorder Brunner fast sweep...")
    for m in m_values
        cvec = radau_right_nodes(m)
        for q in q_values
            solve_brunner_fast(8; m = m, q = q, rhs_coeffs = rhs_coeffs)
            for M in Ms
                diag = mesh_diagnostics(M, q, cvec)
                if !diag.mesh_valid
                    @warn "Skipping invalid Brunner mesh." M q m
                    continue
                end
                sol = solve_brunner_fast(M; m = m, q = q, rhs_coeffs = rhs_coeffs)
                max_error, rms_error = evaluate_error(x -> eval_brunner_solution(x, sol), xgrid, exact_vals)
                rhs_time = rhs_info.rhs_time + sol.rhs_eval_time
                method_time = sol.solve_time
                row = (
                    method = "Brunner",
                    dof = sol.dof,
                    N_or_M = M,
                    N = missing,
                    m = m,
                    q = Float64(q),
                    mu = MU,
                    lambda = LAMBDA,
                    beta = missing,
                    rhs_time = rhs_time,
                    projection_time = missing,
                    construction_time = missing,
                    solve_time = sol.solve_time,
                    method_time = method_time,
                    total_time_including_rhs_generation = rhs_time + method_time,
                    max_error = max_error,
                    rms_error = rms_error,
                    hmin = diag.hmin,
                    hmax = diag.hmax,
                    mesh_ratio = diag.mesh_ratio,
                    mesh_valid = diag.mesh_valid,
                    collapsed_x_count = diag.collapsed_x_count,
                    collapsed_x_fraction = diag.collapsed_x_fraction,
                    min_positive_s = diag.min_positive_s,
                    min_positive_xplus1 = diag.min_positive_xplus1,
                    numerically_extreme = diag.numerically_extreme,
                )
                push!(rows, row)
                @printf("m=%d, q=%.1f, M=%5d, dof=%6d, method_time=%.6e, max_error=%.6e, collapsed=%.3e\n",
                    m, q, M, row.dof, row.method_time, row.max_error, row.collapsed_x_fraction)
            end
        end
    end
    return rows
end

function run_cheb_benchmark(Ns, rhs_coeffs, rhs_info, xgrid, exact_vals)
    rows = NamedTuple[]
    println()
    println("Running highorder Chebyshev-Duffy benchmark...")
    C, kernel_time, kernel_tail = build_kernel_coefficients()
    @printf("Duffy kernel coefficient size = %d x %d, build %.6e s\n", size(C, 1), size(C, 2), kernel_time)
    @printf("kernel tail row_rel=%.3e, col_rel=%.3e\n", kernel_tail.row_rel, kernel_tail.col_rel)
    for warm_N in unique([first(Ns), min(maximum(Ns), max(64, first(Ns)))])
        solve_chebduffy(warm_N, C, rhs_coeffs, rhs_info, [-1.0, 0.0, 1.0], u_exact_x.([-1.0, 0.0, 1.0]))
    end
    for N in Ns
        row = solve_chebduffy(N, C, rhs_coeffs, rhs_info, xgrid, exact_vals)
        push!(rows, row)
        @printf("N=%5d, dof=%5d, method_time=%.6e, total_with_rhs=%.6e, max_error=%.6e, rms_error=%.6e\n",
            N, row.dof, row.method_time, row.total_time_including_rhs_generation, row.max_error, row.rms_error)
    end
    return rows
end

function brunner_best_envelope(rows)
    valid = [r for r in rows if !Bool(r.numerically_extreme) && Bool(r.mesh_valid) && isfinite(r.max_error)]
    env = NamedTuple[]
    for i in eachindex(valid)
        r = valid[i]
        dominated = false
        for j in eachindex(valid)
            i == j && continue
            s = valid[j]
            if s.max_error <= r.max_error && s.method_time <= r.method_time &&
               (s.max_error < r.max_error || s.method_time < r.method_time)
                dominated = true
                break
            end
        end
        dominated || push!(env, r)
    end
    return sort(env; by = r -> r.max_error, rev = true)
end

function csv_cell(x)
    x === missing && return "missing"
    if x isa AbstractFloat
        return isfinite(x) ? @sprintf("%.17e", x) : string(x)
    elseif x isa Integer || x isa Bool
        return string(x)
    else
        return string(x)
    end
end

function write_csv(path, headers, rows)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(headers, ","))
        for r in rows
            println(io, join((csv_cell(getfield(r, Symbol(h))) for h in headers), ","))
        end
    end
end

const COMMON_HEADERS = [
    "method", "dof", "N_or_M", "m", "q", "mu", "lambda", "beta",
    "rhs_time", "projection_time", "construction_time", "solve_time",
    "method_time", "total_time_including_rhs_generation",
    "max_error", "rms_error",
    "hmin", "hmax", "mesh_ratio", "mesh_valid",
    "collapsed_x_count", "collapsed_x_fraction", "min_positive_s", "min_positive_xplus1",
    "numerically_extreme",
]

function plot_results(brunner_rows, cheb_rows, envelope_rows)
    try
        @eval import Plots
        Base.invokelatest(Plots.gr)

        b_valid = [r for r in brunner_rows if Bool(r.mesh_valid) && isfinite(r.max_error)]
        c_dof = [Float64(r.dof) for r in cheb_rows]
        c_err = [Float64(r.max_error) for r in cheb_rows]
        c_time = [Float64(r.method_time) for r in cheb_rows]
        e_err = [Float64(r.max_error) for r in envelope_rows]
        e_time = [Float64(r.method_time) for r in envelope_rows]
        e_dof = [Float64(r.dof) for r in envelope_rows]

        function save(fig, stem)
            pdf = joinpath(FIGURES_DIR, stem * ".pdf")
            png = joinpath(FIGURES_DIR, stem * ".png")
            Base.invokelatest(Plots.savefig, fig, pdf)
            Base.invokelatest(Plots.savefig, fig, png)
            println("Saved figure: ", pdf)
            println("Saved figure: ", png)
        end

        common = (xscale = :log10, yscale = :log10, framestyle = :box, grid = false,
            legend = :outerright, dpi = 300, size = (900, 540))

        p1 = Base.invokelatest(Plots.scatter,
            [Float64(r.dof) for r in b_valid], [Float64(r.max_error) for r in b_valid];
            common..., markeralpha = 0.25, markersize = 3, label = "Brunner parameter points",
            xlabel = "degrees of freedom", ylabel = "max error")
        Base.invokelatest(Plots.plot!, p1, e_dof, e_err; marker = :circle, linewidth = 2.5, label = "Brunner best envelope")
        Base.invokelatest(Plots.plot!, p1, c_dof, c_err; marker = :star5, color = :black, linewidth = 2.5, linestyle = :dash, label = "Chebyshev--Duffy")
        save(p1, "compare_highorder_error_vs_dof")

        p2 = Base.invokelatest(Plots.scatter,
            [Float64(r.dof) for r in b_valid], [Float64(r.method_time) for r in b_valid];
            common..., markeralpha = 0.25, markersize = 3, label = "Brunner parameter points",
            xlabel = "degrees of freedom", ylabel = "method time (sec)")
        Base.invokelatest(Plots.plot!, p2, c_dof, c_time; marker = :star5, color = :black, linewidth = 2.5, linestyle = :dash, label = "Chebyshev--Duffy")
        save(p2, "compare_highorder_method_time_vs_dof")

        p3 = Base.invokelatest(Plots.scatter,
            [Float64(r.max_error) for r in b_valid], [Float64(r.method_time) for r in b_valid];
            common..., markeralpha = 0.25, markersize = 3, label = "Brunner parameter points",
            xlabel = "max error", ylabel = "method time (sec)")
        Base.invokelatest(Plots.plot!, p3, e_err, e_time; marker = :circle, linewidth = 2.5, label = "Brunner best envelope")
        Base.invokelatest(Plots.plot!, p3, c_err, c_time; marker = :star5, color = :black, linewidth = 2.5, linestyle = :dash, label = "Chebyshev--Duffy")
        save(p3, "compare_highorder_method_time_vs_error")
        save(p3, "compare_highorder_time_vs_error_with_envelope")

        Ms = sort(unique(r.N_or_M for r in b_valid))
        p4 = Base.invokelatest(Plots.plot, Float64[], Float64[];
            yscale = :log10, framestyle = :box, grid = false, legend = :outerright,
            xlabel = "q", ylabel = "best max error over m", dpi = 300, size = (860, 520), label = false)
        for M in Ms
            rowsM = [r for r in b_valid if r.N_or_M == M]
            qs = sort(unique(Float64(r.q) for r in rowsM))
            errs = [minimum(Float64(r.max_error) for r in rowsM if isapprox(Float64(r.q), q; atol = 1e-14)) for q in qs]
            Base.invokelatest(Plots.plot!, p4, qs, errs; marker = :circle, label = "M=$(M)")
        end
        save(p4, "brunner_highorder_error_vs_q_by_M")
        return true
    catch err
        @warn "Plotting skipped; CSV files are available." exception = (err, catch_backtrace())
        return false
    end
end

function run_operator_consistent(Ns, xgrid, exact_vals)
    rows = NamedTuple[]
    C, _, _ = build_kernel_coefficients()
    for N in Ns
        stat = @timed begin
            V, _ = VoltOpt.weak_voltmat_cheb(C; C_is_coeffs = true, mu = MU, sigma = SIGMA,
                beta = BETA, N = N, diagnostics = false, output_format = :dense, return_info = true)
            A = Matrix{Float64}(LinearAlgebra.I, N + 1, N + 1)
            A .+= LAMBDA .* dense_matrix(V)
            u_coeff = exact_solution_coeffs_jfp(N)
            g_coeff = A * u_coeff
            solve_stat = @timed A \ g_coeff
            coeffs = solve_stat.value
            max_error, rms_error = evaluate_error(x -> eval_jfp(coeffs, x; beta = BETA), xgrid, exact_vals)
            coeff_error = norm(coeffs .- u_coeff, Inf)
            cond_est = N <= COND_MAX_N ? cond(A) : NaN
            (max_error = max_error, rms_error = rms_error, coeff_error = coeff_error,
                solve_time = solve_stat.time, cond_est = cond_est)
        end
        r = stat.value
        push!(rows, (
            N = N, dof = N + 1, total_time = stat.time, max_error = r.max_error,
            rms_error = r.rms_error, coeff_error = r.coeff_error, cond_est = r.cond_est,
        ))
        @printf("operator N=%5d, max_error=%.6e, coeff_error=%.6e\n", N, r.max_error, r.coeff_error)
    end
    return rows
end

function plot_operator(rows)
    try
        @eval import Plots
        Base.invokelatest(Plots.gr)
        fig = Base.invokelatest(Plots.plot, [r.N for r in rows], [r.max_error for r in rows];
            xscale = :log10, yscale = :log10, marker = :circle, linewidth = 2.2,
            framestyle = :box, grid = false, label = false, xlabel = "N", ylabel = "max error",
            dpi = 300)
        pdf = joinpath(FIGURES_DIR, "chebduffy_highorder_operator_consistent_error_vs_N.pdf")
        png = joinpath(FIGURES_DIR, "chebduffy_highorder_operator_consistent_error_vs_N.png")
        Base.invokelatest(Plots.savefig, fig, pdf)
        Base.invokelatest(Plots.savefig, fig, png)
        return true
    catch err
        @warn "Operator diagnostic plot skipped." exception = (err, catch_backtrace())
        return false
    end
end

function main()
    RHS_MODE in ("ex7", "operator_consistent") ||
        error("Unsupported RHS_MODE=$(RHS_MODE).")
    BRUNNER_IMPL == "fast" ||
        @warn "This highorder script implements the optimized Brunner fast path; BRUNNER_IMPL is ignored." BRUNNER_IMPL

    brunner_Ms = env_limited_values(DEFAULT_BRUNNER_MS, "BRUNNER_MAX_M"; candidates = BRUNNER_MS_CANDIDATES)
    cheb_Ns = env_limited_values(DEFAULT_CHEB_NS, "CHEB_MAX_N")
    brunner_ms = parse_env_int_list("BRUNNER_MS", DEFAULT_BRUNNER_M_VALUES)
    brunner_qs = parse_env_float_list("BRUNNER_QS", DEFAULT_BRUNNER_Q_VALUES)
    RHS_TARGET_DEGREE[] = max(RHS_TARGET_DEGREE[], maximum(cheb_Ns))

    xgrid = collect(range(-1.0, 1.0; length = DEFAULT_XGRID_N))
    exact_vals = u_exact_x.(xgrid)

    println("============================================================")
    println("Highorder oscillatory manufactured weakly singular Volterra test")
    println("u_exact(x) = sqrt(x+1) * sin(100x)")
    @printf("mu = %.3f, lambda = %.3f, beta = %.3f\n", MU, LAMBDA, BETA)
    println("RHS_MODE = ", RHS_MODE)
    @printf("RHS BigFloat precision = %d bits, tanh-sinh h = %s\n", RHS_PRECISION, RHS_TS_H)
    @printf("Duffy kernel coefficient size = %d x %d\n", KERNEL_C_ROWS, KERNEL_C_COLS)
    println("Brunner Ms = ", brunner_Ms)
    println("Brunner m values = ", brunner_ms)
    println("Brunner q values = ", brunner_qs)
    println("Chebyshev-Duffy Ns = ", cheb_Ns)
    println("xgrid length = ", length(xgrid))
    println("============================================================")

    if RHS_MODE == "operator_consistent"
        rows = run_operator_consistent(cheb_Ns, xgrid, exact_vals)
        csv = joinpath(DATA_DIR, "chebduffy_highorder_operator_consistent.csv")
        write_csv(csv, ["N", "dof", "total_time", "max_error", "rms_error", "coeff_error", "cond_est"], rows)
        plot_operator(rows)
        println("Saved operator diagnostic CSV: ", csv)
        return rows
    end

    rhs_stability = check_rhs_stability()
    rhs_coeffs, rhs_info = rhs_coeffs_highorder(RHS_TARGET_DEGREE[])
    brunner_rows = run_brunner_sweep(brunner_Ms, brunner_ms, brunner_qs, rhs_coeffs, rhs_info, xgrid, exact_vals)
    cheb_rows = run_cheb_benchmark(cheb_Ns, rhs_coeffs, rhs_info, xgrid, exact_vals)
    envelope_rows = brunner_best_envelope(brunner_rows)
    combined = vcat(brunner_rows, cheb_rows)

    brunner_csv = joinpath(DATA_DIR, "brunner_highorder_fast.csv")
    cheb_csv = joinpath(DATA_DIR, "chebduffy_highorder.csv")
    compare_csv = joinpath(DATA_DIR, "compare_brunner_chebduffy_highorder.csv")
    env_csv = joinpath(DATA_DIR, "brunner_best_envelope_highorder.csv")
    write_csv(brunner_csv, COMMON_HEADERS, brunner_rows)
    write_csv(cheb_csv, COMMON_HEADERS, cheb_rows)
    write_csv(compare_csv, COMMON_HEADERS, combined)
    write_csv(env_csv, COMMON_HEADERS, envelope_rows)
    plot_results(brunner_rows, cheb_rows, envelope_rows)

    best_cheb = cheb_rows[argmin([r.max_error for r in cheb_rows])]
    best_brunner = envelope_rows[argmin([r.max_error for r in envelope_rows])]
    println()
    @printf("RHS stability max |h=.02 - h=.015| = %.6e\n", rhs_stability.diff_002_0015)
    @printf("Best Chebyshev-Duffy: N=%d, dof=%d, method_time=%.6e, max_error=%.6e\n",
        best_cheb.N_or_M, best_cheb.dof, best_cheb.method_time, best_cheb.max_error)
    @printf("Best Brunner envelope: m=%d, q=%.1f, M=%d, dof=%d, method_time=%.6e, max_error=%.6e, extreme=%s\n",
        best_brunner.m, best_brunner.q, best_brunner.N_or_M, best_brunner.dof,
        best_brunner.method_time, best_brunner.max_error, string(best_brunner.numerically_extreme))
    println("Saved CSV files:")
    println("  ", brunner_csv)
    println("  ", cheb_csv)
    println("  ", compare_csv)
    println("  ", env_csv)
    println("Recommended paper figure: ", joinpath(FIGURES_DIR, "compare_highorder_time_vs_error_with_envelope.pdf"))
    return (brunner = brunner_rows, chebduffy = cheb_rows, envelope = envelope_rows)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
