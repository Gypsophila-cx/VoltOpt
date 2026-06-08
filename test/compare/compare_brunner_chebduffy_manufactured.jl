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
const RHS_MODE = get(ENV, "RHS_MODE", "ex7")
const BRUNNER_IMPL = lowercase(get(ENV, "BRUNNER_IMPL", "fast"))

const DEFAULT_BRUNNER_MS = [32, 64, 128, 256, 512, 1024, 2048]
const BRUNNER_MS_CANDIDATES = [32, 64, 128, 256, 512, 1024, 2048, 4096]
const DEFAULT_CHEB_NS = [16, 24, 32, 48, 64, 96, 128, 160, 192, 256, 320, 384, 512]

const DEFAULT_BRUNNER_M = 2
const DEFAULT_BRUNNER_Q = 2.0
const DEFAULT_BRUNNER_M_VALUES = [2, 3, 4]
const DEFAULT_BRUNNER_Q_VALUES = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
const DEFAULT_BRUNNER_NQ = 64
const RHS_DE_H = parse(Float64, get(ENV, "RHS_DE_H", "0.02"))
const DEFAULT_RHS_DE_EPS_STOP = eps(Float64)
const DEFAULT_XGRID_N = 4001

const KERNEL_C_ROWS = parse(Int, get(ENV, "KERNEL_C_ROWS", "61"))
const KERNEL_C_COLS = parse(Int, get(ENV, "KERNEL_C_COLS", "41"))
const KERNEL_I = KERNEL_C_ROWS - 1
const KERNEL_J = KERNEL_C_COLS - 1

const RHS_PRECISION = parse(Int, get(ENV, "RHS_PRECISION", "256"))
const RHS_EX7_TARGET_DEGREE = Ref(parse(Int, get(ENV, "RHS_POINT_DEGREE", "512")))
const RHS_SAMPLE_DEGREE_ENV = get(ENV, "RHS_SAMPLE_DEGREE", "")
const RHS_CHECK_SAMPLE_DEGREE_ENV = get(ENV, "RHS_CHECK_SAMPLE_DEGREE", "")
const TS_H_STR = get(ENV, "RHS_TS_H", "0.125")
const TS_ATOL_STR = get(ENV, "RHS_TS_ATOL", "1e-62")
const TS_MAX_TERMS = parse(Int, get(ENV, "RHS_TS_MAX_TERMS", "180"))
const COND_MAX_N = parse(Int, get(ENV, "COND_MAX_N", "192"))

const _GL01_CACHE = Dict{Int,Tuple{Vector{Float64},Vector{Float64}}}()
const _RHS_EX7_COEFFS = Ref{Union{Nothing,Vector{Float64}}}(nothing)
const _RHS_EX7_INFO = Ref{Any}(nothing)

struct BrunnerManufacturedSolution
    mesh::Vector{Float64}
    c::Vector{Float64}
    Uvals::Matrix{Float64}
    M::Int
    m::Int
    q::Float64
    mu::Float64
    lambda::Float64
    lambda_hat::Float64
    nq::Int
    rhs_h::Float64
    impl::Symbol
    solve_time::Float64
    dof::Int
end

# ============================================================
# Manufactured problem
# ============================================================

function u_exact(x)
    xp1 = Float64(x) + 1.0
    xp1 = xp1 < 0.0 && xp1 > -100eps(Float64) ? 0.0 : xp1
    xp1 >= 0.0 || throw(ArgumentError("x must be in [-1, 1]."))
    sx = sqrt(xp1)
    return 1.0 + 0.2 * sx + 0.1 * xp1 + 0.05 * xp1 * sx
end

Kfun(x, y) = inv((Float64(x) + 1.0)^2 + (Float64(y) - 1.0)^2)
Khat(s, t) = Kfun(2.0 * Float64(s) - 1.0, 2.0 * Float64(t) - 1.0)

function u_exact_big(x::BigFloat)
    xp1 = x + one(BigFloat)
    return one(BigFloat) +
           BigFloat("0.2") * sqrt(xp1) +
           BigFloat("0.1") * xp1 +
           BigFloat("0.05") * xp1 * sqrt(xp1)
end

kernel_big(x::BigFloat, y::BigFloat) =
    inv((x + one(BigFloat))^2 + (y - one(BigFloat))^2)

function invphi_beta_big(z::BigFloat; beta = BETA)
    betab = BigFloat(beta)
    return BigFloat(2) * ((one(BigFloat) + z) / BigFloat(2))^(one(BigFloat) / betab) -
           one(BigFloat)
end

function gausslegendre01(nq)
    nq > 0 || throw(ArgumentError("nq must be positive."))

    cached = get(_GL01_CACHE, nq, nothing)
    cached !== nothing && return cached

    x = Vector{Float64}(undef, nq)
    w = Vector{Float64}(undef, nq)

    if nq == 1
        x[1] = 0.5
        w[1] = 1.0
    else
        beta = [i / sqrt(4.0 * i^2 - 1.0) for i in 1:(nq - 1)]
        T = SymTridiagonal(zeros(nq), beta)
        E = eigen(T)
        @inbounds for k in 1:nq
            x[k] = 0.5 * (E.values[k] + 1.0)
            w[k] = abs2(E.vectors[1, k])
        end
    end

    _GL01_CACHE[nq] = (x, w)
    return x, w
end

function tanhsinh_xi_dzdt(t::Float64)
    a = (pi / 2.0) * sinh(t)
    xi = tanh(a)
    sech2 = abs(a) > 350.0 ? 0.0 : inv(cosh(a)^2)
    dzdt = 0.5 * (pi / 2.0) * cosh(t) * sech2
    return xi, dzdt
end

function tanhsinh_truncation_info(; h = RHS_DE_H, eps_stop = DEFAULT_RHS_DE_EPS_STOP, maxK = 100000)
    h > 0.0 || throw(ArgumentError("h must be positive."))
    eps_stop > 0.0 || throw(ArgumentError("eps_stop must be positive."))
    maxK > 0 || throw(ArgumentError("maxK must be positive."))

    for K in 0:maxK
        t = K * Float64(h)
        xip, _ = tanhsinh_xi_dzdt(t)
        xim, _ = tanhsinh_xi_dzdt(-t)
        right_distance = 1.0 - xip
        left_distance = xim + 1.0
        if right_distance <= eps_stop && left_distance <= eps_stop
            return (
                K = K,
                h = Float64(h),
                eps_stop = Float64(eps_stop),
                right_distance = right_distance,
                left_distance = left_distance,
            )
        end
    end

    error("tanh-sinh endpoint-distance truncation did not converge before maxK=$(maxK).")
end

function tanhsinh_integral01(f; h = RHS_DE_H, eps_stop = DEFAULT_RHS_DE_EPS_STOP, maxK = 100000)
    info = tanhsinh_truncation_info(; h = h, eps_stop = eps_stop, maxK = maxK)
    hf = Float64(h)
    total = 0.0

    # K is selected by the endpoint-distance criterion above; this loop is
    # therefore not a fixed tmax or fixed truncation rule.
    for k in -info.K:info.K
        t = hf * k
        xi, dzdt = tanhsinh_xi_dzdt(t)
        z = 0.5 * (1.0 + xi)

        # Once z collapses exactly to an endpoint, evaluating endpoint-singular
        # integrands would create Inf/NaN. The DE weight has already decayed to
        # zero at these collapsed nodes, so they are omitted.
        (z <= 0.0 || z >= 1.0 || dzdt == 0.0) && continue

        val = f(z)
        contrib = val * dzdt
        isfinite(contrib) ||
            error("Non-finite tanh-sinh contribution at k=$(k), z=$(z), value=$(val), dzdt=$(dzdt).")
        total += contrib
    end

    result = hf * total
    isfinite(result) || error("Non-finite tanh-sinh integral result.")
    return result
end

function tanhsinh_integral01_right_weight(f, right_mu; h = RHS_DE_H, eps_stop = DEFAULT_RHS_DE_EPS_STOP, maxK = 100000)
    info = tanhsinh_truncation_info(; h = h, eps_stop = eps_stop, maxK = maxK)
    hf = Float64(h)
    muf = Float64(right_mu)
    total = 0.0

    for k in -info.K:info.K
        t = hf * k
        xi, dzdt = tanhsinh_xi_dzdt(t)
        z = 0.5 * (1.0 + xi)

        (z <= 0.0 || z >= 1.0 || dzdt <= 0.0) && continue

        logweight = -muf * log1p(-z) + log(dzdt)
        weight = exp(logweight)
        contrib = f(z) * weight
        isfinite(contrib) ||
            error("Non-finite weighted tanh-sinh contribution at k=$(k), z=$(z), weight=$(weight).")
        total += contrib
    end

    result = hf * total
    isfinite(result) || error("Non-finite weighted tanh-sinh integral result.")
    return result
end

function volterra_integral_rhs_DE(x; mu = MU, h = RHS_DE_H)
    xf = Float64(x)
    xp1 = xf + 1.0
    xp1 <= 100eps(Float64) && return 0.0

    muf = Float64(mu)
    singular_power = inv(1.0 - muf)
    singular_scale = inv(1.0 - muf)

    # Remove the right-endpoint algebraic singularity exactly before applying
    # tanh-sinh. For mu=1/2 this is z = 1 - r^2 and the singular factor becomes
    # the constant multiplier 2, avoiding the O(sqrt(eps)) endpoint tail that
    # appears if Float64 z-nodes are used directly at the singular endpoint.
    integral01 = tanhsinh_integral01(h = h) do r
        z = 1.0 - r^singular_power
        y = -1.0 + xp1 * z
        return singular_scale * Kfun(xf, y) * u_exact(y)
    end

    result = xp1^(1.0 - muf) * integral01
    isfinite(result) || error("Non-finite manufactured Volterra RHS integral at x=$(x).")
    return result
end

function g_manufactured_DE(x; mu = MU, lambda = LAMBDA, h = RHS_DE_H)
    xf = Float64(x)
    xf <= -1.0 + 100eps(Float64) && return u_exact(-1.0)
    result = u_exact(xf) + Float64(lambda) * volterra_integral_rhs_DE(xf; mu = mu, h = h)
    isfinite(result) || error("Non-finite manufactured RHS at x=$(x).")
    return result
end

Ghat(s; h = RHS_DE_H) =
    g_manufactured_DE(2.0 * Float64(s) - 1.0; mu = MU, lambda = LAMBDA, h = h)

# ============================================================
# ex7-style BigFloat tanh-sinh RHS generation
# ============================================================

mutable struct TanhSinhStats
    max_terms::Int
    max_last_term::BigFloat
    calls::Int
end

TanhSinhStats() = TanhSinhStats(0, zero(BigFloat), 0)

function tanh_sinh_q_dq(s::BigFloat)
    half = BigFloat(0.5)
    a = (big(pi) / BigFloat(2)) * sinh(s)
    q = half * (one(BigFloat) + tanh(a))
    sech2 = inv(cosh(a)^2)
    dq = half * (big(pi) / BigFloat(2)) * cosh(s) * sech2
    return q, dq
end

function tanh_sinh_integrate_01_big(
    f;
    h::BigFloat = parse(BigFloat, TS_H_STR),
    atol::BigFloat = parse(BigFloat, TS_ATOL_STR),
    max_terms::Int = TS_MAX_TERMS,
    stats::Union{Nothing,TanhSinhStats} = nothing,
)
    total = zero(BigFloat)
    max_term = zero(BigFloat)
    last_pair = typemax(BigFloat)
    small_count = 0
    terms_used = 0

    for k in 0:max_terms
        pair_sum = zero(BigFloat)

        if k == 0
            q, dq = tanh_sinh_q_dq(zero(BigFloat))
            if zero(BigFloat) < q < one(BigFloat)
                pair_sum += f(q) * dq
            end
        else
            for sgn in (-1, 1)
                q, dq = tanh_sinh_q_dq(BigFloat(sgn * k) * h)
                if zero(BigFloat) < q < one(BigFloat)
                    pair_sum += f(q) * dq
                end
            end
        end

        total += pair_sum
        last_pair = abs(pair_sum)
        max_term = max(max_term, last_pair)
        terms_used = k

        scale = max(one(BigFloat), abs(total), max_term)
        if k >= 8 && last_pair <= atol * scale
            small_count += 1
            small_count >= 8 && break
        else
            small_count = 0
        end
    end

    if stats !== nothing
        stats.calls += 1
        stats.max_terms = max(stats.max_terms, terms_used)
        stats.max_last_term = max(stats.max_last_term, last_pair)
    end

    return h * total
end

function volterra_manufactured_big(x::BigFloat, stats::TanhSinhStats)
    x <= -one(BigFloat) && return zero(BigFloat)

    xp1 = x + one(BigFloat)
    xp1 == 0 && return zero(BigFloat)

    integral = tanh_sinh_integrate_01_big(stats = stats) do xi
        y = -one(BigFloat) + xp1 * xi
        return (one(BigFloat) - xi)^(-BigFloat(MU)) * kernel_big(x, y) * u_exact_big(y)
    end

    return xp1^(one(BigFloat) - BigFloat(MU)) * integral
end

function rhs_big_at_x_manufactured(x::BigFloat, stats::TanhSinhStats)
    return u_exact_big(x) + BigFloat(LAMBDA) * volterra_manufactured_big(x, stats)
end

function cheb_project_lobatto_big(f, coeff_degree::Int, sample_degree::Int; precision::Int)
    coeff_degree <= sample_degree ||
        throw(ArgumentError("coeff_degree must not exceed sample_degree."))
    sample_degree >= 1 ||
        throw(ArgumentError("sample_degree must be positive."))

    stats = TanhSinhStats()

    coeffs, info = setprecision(BigFloat, precision) do
        M = sample_degree
        vals = Vector{BigFloat}(undef, M + 1)

        @inbounds for j in 0:M
            z = cos(big(pi) * BigFloat(j) / BigFloat(M))
            vals[j + 1] = f(z, stats)
        end

        c = zeros(BigFloat, coeff_degree + 1)

        @inbounds for k in 0:coeff_degree
            s = zero(BigFloat)
            for j in 0:M
                wj = (j == 0 || j == M) ? BigFloat(0.5) : one(BigFloat)
                s += wj * vals[j + 1] * cos(big(pi) * BigFloat(k * j) / BigFloat(M))
            end
            c[k + 1] = BigFloat(2) * s / BigFloat(M)
        end
        c[1] /= BigFloat(2)

        tail_count = min(20, coeff_degree + 1)
        tail_inf = maximum(abs, @view c[(end - tail_count + 1):end])
        info = (
            sample_degree = M,
            coeff_degree = coeff_degree,
            tail_count = tail_count,
            tail_inf = Float64(tail_inf),
            ts_calls = stats.calls,
            ts_max_terms = stats.max_terms,
            ts_max_last_term = Float64(stats.max_last_term),
        )

        Float64.(c), info
    end

    return coeffs, info
end

function rhs_coeffs_like_ex7_manufactured(
    coeff_degree::Int;
    beta = BETA,
    precision = RHS_PRECISION,
)
    sample_degree = isempty(RHS_SAMPLE_DEGREE_ENV) ?
                    max(512, 2 * coeff_degree) :
                    parse(Int, RHS_SAMPLE_DEGREE_ENV)
    check_sample_degree = isempty(RHS_CHECK_SAMPLE_DEGREE_ENV) ?
                          max(coeff_degree, cld(sample_degree, 2)) :
                          parse(Int, RHS_CHECK_SAMPLE_DEGREE_ENV)

    f_rhs = function (z::BigFloat, stats::TanhSinhStats)
        x = invphi_beta_big(z; beta = beta)
        return rhs_big_at_x_manufactured(x, stats)
    end

    t_hi = @elapsed begin
        coeffs_hi, info_hi = cheb_project_lobatto_big(
            f_rhs,
            coeff_degree,
            sample_degree;
            precision = precision,
        )
    end

    t_check = @elapsed begin
        coeffs_check, info_check = cheb_project_lobatto_big(
            f_rhs,
            coeff_degree,
            check_sample_degree;
            precision = precision,
        )
    end

    diff_inf = maximum(abs.(coeffs_hi .- coeffs_check))
    diff_rel = diff_inf / max(norm(coeffs_hi, Inf), floatmin(Float64))

    info = (
        high = info_hi,
        check = info_check,
        coeff_diff_inf = diff_inf,
        coeff_diff_rel = diff_rel,
        high_time = t_hi,
        check_time = t_check,
        precision = precision,
        ts_h = TS_H_STR,
        ts_atol = TS_ATOL_STR,
        ts_max_terms = TS_MAX_TERMS,
    )

    return coeffs_hi, info
end

function ensure_rhs_ex7_coeffs(degree::Int)
    degree = max(Int(degree), 0)
    cached = _RHS_EX7_COEFFS[]
    if cached !== nothing && length(cached) >= degree + 1
        return cached
    end

    coeffs, info = rhs_coeffs_like_ex7_manufactured(degree)
    _RHS_EX7_COEFFS[] = coeffs
    _RHS_EX7_INFO[] = info
    return coeffs
end

function g_manufactured_like_ex7(x)
    coeffs = ensure_rhs_ex7_coeffs(RHS_EX7_TARGET_DEGREE[])
    return eval_jfp(coeffs, x; beta = BETA)
end

function g_manufactured_s_like_ex7(s)
    coeffs = ensure_rhs_ex7_coeffs(RHS_EX7_TARGET_DEGREE[])
    return eval_jfp_s(coeffs, s; beta = BETA)
end

function g_manufactured_rhs(x; h = RHS_DE_H)
    if RHS_MODE == "ex7"
        return g_manufactured_like_ex7(x)
    elseif RHS_MODE == "old_DE"
        return g_manufactured_DE(x; mu = MU, lambda = LAMBDA, h = h)
    else
        error("Point RHS is not available for RHS_MODE=$(RHS_MODE).")
    end
end

function g_manufactured_s_rhs(s; h = RHS_DE_H)
    if RHS_MODE == "ex7"
        return g_manufactured_s_like_ex7(s)
    elseif RHS_MODE == "old_DE"
        return g_manufactured_DE(2.0 * Float64(s) - 1.0; mu = MU, lambda = LAMBDA, h = h)
    else
        error("Point RHS is not available for RHS_MODE=$(RHS_MODE).")
    end
end

Ghat_rhs(s; h = RHS_DE_H) = g_manufactured_s_rhs(s; h = h)

# ============================================================
# Brunner graded collocation / product integration
# ============================================================

function radau_right_nodes(m::Int)
    m == 2 && return [1.0 / 3.0, 1.0]
    m == 3 && return [0.1550510257216822, 0.6449489742783178, 1.0]
    m == 4 && return [0.0885879595127042, 0.4094668644407347, 0.7876594617608471, 1.0]
    throw(ArgumentError("supported Brunner collocation orders are m = 2, 3, 4."))
end

function validate_collocation_nodes(c)
    all(isfinite, c) || throw(ArgumentError("collocation nodes must be finite."))
    all(x -> 0.0 < x <= 1.0, c) ||
        throw(ArgumentError("collocation nodes must lie in (0, 1]."))
    all(diff(c) .> 0.0) ||
        throw(ArgumentError("collocation nodes must be strictly increasing."))
    return Float64.(c)
end

function lagrange_basis(theta, j, c)
    1 <= j <= length(c) || throw(BoundsError(c, j))
    v = 1.0
    cj = c[j]
    @inbounds for k in eachindex(c)
        k == j && continue
        v *= (theta - c[k]) / (cj - c[k])
    end
    return v
end

function product_weight(delta, theta_max, j, c, mu; nq = DEFAULT_BRUNNER_NQ)
    0.0 < mu < 1.0 || throw(ArgumentError("mu must satisfy 0 < mu < 1."))
    theta_max <= 0.0 && return 0.0
    delta + 1e-14 >= theta_max ||
        throw(ArgumentError("theta_max must not exceed delta."))

    xq, wq = gausslegendre01(nq)
    singular_current_panel =
        delta <= 1.0 + 1e-14 &&
        abs(theta_max - delta) <= 1e-12 * max(1.0, abs(delta))

    if singular_current_panel
        exponent = inv(1.0 - Float64(mu))
        acc = 0.0
        @inbounds for k in eachindex(xq)
            theta = delta * (1.0 - xq[k]^exponent)
            acc += wq[k] * lagrange_basis(theta, j, c)
        end
        return delta^(1.0 - Float64(mu)) * acc / (1.0 - Float64(mu))
    else
        acc = 0.0
        @inbounds for k in eachindex(xq)
            theta = theta_max * xq[k]
            acc += wq[k] * (delta - theta)^(-Float64(mu)) * lagrange_basis(theta, j, c)
        end
        return theta_max * acc
    end
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

    # Analytic expansion of (1 - theta/delta)^(-mu), used only when
    # a/delta is tiny enough that the finite-difference moment loses digits.
    @inbounds for k in 0:80
        term = coeff * apow * dscale / (p + k + 1)
        acc += term
        if abs(term) <= 4eps(Float64) * max(1.0, abs(acc))
            return acc
        end
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
    p >= 0 || throw(ArgumentError("p must be nonnegative."))
    af <= 0.0 && return 0.0
    deltaf + 1e-14 >= af ||
        throw(ArgumentError("moment upper limit a must not exceed delta."))

    if deltaf > 1e6 || af / deltaf <= 0.1
        return moment_power_asymptotic(deltaf, af, p, muf)
    end

    lower = deltaf - af
    if abs(lower) < 10eps(deltaf)
        lower = 0.0
    else
        lower = max(lower, 0.0)
    end

    acc = 0.0
    @inbounds for ridx in 0:p
        exponent = ridx + 1 - muf
        upper_term = deltaf^exponent
        lower_term = lower == 0.0 ? 0.0 : lower^exponent
        signed_binom = isodd(ridx) ? -Float64(binomial(p, ridx)) : Float64(binomial(p, ridx))
        acc += signed_binom *
               deltaf^(p - ridx) *
               (upper_term - lower_term) /
               exponent
    end
    return acc
end

function product_weight_fast(delta, theta_max, j, coeffs::AbstractMatrix{<:Real}, mu)
    m = size(coeffs, 1)
    1 <= j <= m || throw(BoundsError(coeffs, j))
    acc = 0.0
    @inbounds for p in 0:(m - 1)
        acc += Float64(coeffs[j, p + 1]) *
               moment_power(delta, theta_max, p, mu)
    end
    return acc
end

function product_weights_fast!(weights, moments, delta, theta_max, coeffs::AbstractMatrix{<:Real}, mu)
    m = size(coeffs, 1)
    length(weights) >= m || throw(DimensionMismatch("weights is too short."))
    length(moments) >= m || throw(DimensionMismatch("moments is too short."))

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

@inline khat01_fast(s::Float64, t::Float64) =
    inv(4.0 * s * s + 4.0 * (t - 1.0) * (t - 1.0))

function graded_mesh(M::Int, q::Real)
    M > 0 || throw(ArgumentError("M must be positive."))
    q > 0 || throw(ArgumentError("q must be positive."))
    return [(n / M)^Float64(q) for n in 0:M]
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

function local_smooth_weights!(weights, delta, c, mu, xq, wq)
    fill!(weights, 0.0)
    muf = Float64(mu)
    @inbounds for k in eachindex(xq)
        theta = xq[k]
        factor = wq[k] * (delta - theta)^(-muf)
        for j in eachindex(c)
            weights[j] += factor * lagrange_basis(theta, j, c)
        end
    end
    return weights
end

function solve_brunner_manufactured_old(
    M;
    m = DEFAULT_BRUNNER_M,
    q = DEFAULT_BRUNNER_Q,
    mu = MU,
    lambda = LAMBDA,
    nq = DEFAULT_BRUNNER_NQ,
    rhs_h = RHS_DE_H,
    c = nothing,
)
    cvec = validate_collocation_nodes(c === nothing ? radau_right_nodes(m) : c)
    length(cvec) == m || throw(ArgumentError("length(c) must match m."))

    muf = Float64(mu)
    lambdaf = Float64(lambda)
    lambda_hat = lambdaf * 2.0^(1.0 - muf)
    mesh = graded_mesh(M, q)
    h = diff(mesh)
    hpow = h .^ (1.0 - muf)
    nodes = collocation_points(mesh, cvec)
    Uvals = zeros(Float64, M, m)

    current_weights = Matrix{Float64}(undef, m, m)
    for i in 1:m
        for j in 1:m
            current_weights[i, j] =
                product_weight(cvec[i], cvec[i], j, cvec, muf; nq = nq)
        end
    end

    xq, wq = gausslegendre01(nq)
    smooth_weights = zeros(Float64, m)
    A = Matrix{Float64}(undef, m, m)
    b = Vector{Float64}(undef, m)

    solve_time = @elapsed begin
        rhs_nodes = Matrix{Float64}(undef, M, m)
        @inbounds for n in 1:M
            for i in 1:m
                rhs_nodes[n, i] = Ghat_rhs(nodes[n, i]; h = rhs_h)
            end
        end
        all(isfinite, rhs_nodes) || error("Non-finite Brunner RHS values detected.")

        @inbounds for n in 1:M
            fill!(A, 0.0)
            fill!(b, 0.0)

            for i in 1:m
                sni = nodes[n, i]
                history = 0.0

                for l in 1:(n - 1)
                    delta = (sni - mesh[l]) / h[l]
                    local_smooth_weights!(smooth_weights, delta, cvec, muf, xq, wq)
                    for j in 1:m
                        history += hpow[l] *
                                   smooth_weights[j] *
                                   Khat(sni, nodes[l, j]) *
                                   Uvals[l, j]
                    end
                end

                b[i] = rhs_nodes[n, i] - lambda_hat * history

                for j in 1:m
                    A[i, j] = (i == j ? 1.0 : 0.0) +
                              lambda_hat *
                              hpow[n] *
                              current_weights[i, j] *
                              Khat(sni, nodes[n, j])
                end
            end

            Uvals[n, :] .= A \ b
            all(isfinite, @view Uvals[n, :]) ||
                error("Non-finite Brunner solution values detected on panel $(n).")
        end
    end

    return BrunnerManufacturedSolution(
        mesh,
        cvec,
        Uvals,
        Int(M),
        Int(m),
        Float64(q),
        muf,
        lambdaf,
        lambda_hat,
        Int(nq),
        Float64(rhs_h),
        :old,
        solve_time,
        Int(M) * Int(m),
    )
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
                    tmp = A[k, col]
                    A[k, col] = A[pivot, col]
                    A[pivot, col] = tmp
                end
                tmpb = b[k]
                b[k] = b[pivot]
                b[pivot] = tmpb
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

function solve_brunner_fast(
    M;
    m = DEFAULT_BRUNNER_M,
    q = DEFAULT_BRUNNER_Q,
    mu = MU,
    lambda = LAMBDA,
    rhs_h = RHS_DE_H,
    c = nothing,
)
    cvec = validate_collocation_nodes(c === nothing ? radau_right_nodes(m) : c)
    length(cvec) == m || throw(ArgumentError("length(c) must match m."))

    muf = Float64(mu)
    lambdaf = Float64(lambda)
    lambda_hat = lambdaf * 2.0^(1.0 - muf)
    mesh = graded_mesh(M, q)
    h = Vector{Float64}(undef, M)
    hpow = Vector{Float64}(undef, M)
    @inbounds for n in 1:M
        hn = mesh[n + 1] - mesh[n]
        h[n] = hn
        hpow[n] = hn^(1.0 - muf)
    end

    nodes = collocation_points(mesh, cvec)
    Uvals = zeros(Float64, M, m)
    power_coeffs = lagrange_power_coeffs(cvec)

    current_weights0 = Matrix{Float64}(undef, m, m)
    moments = zeros(Float64, m)
    row_weights = zeros(Float64, m)
    @inbounds for i in 1:m
        product_weights_fast!(row_weights, moments, cvec[i], cvec[i], power_coeffs, muf)
        for j in 1:m
            current_weights0[i, j] = row_weights[j]
        end
    end

    A = Matrix{Float64}(undef, m, m)
    b = Vector{Float64}(undef, m)
    xloc = Vector{Float64}(undef, m)
    hist_weights = zeros(Float64, m)
    hist_moments = zeros(Float64, m)

    solve_time = @elapsed begin
        rhs_nodes = Matrix{Float64}(undef, M, m)
        @inbounds for n in 1:M
            for i in 1:m
                rhs_nodes[n, i] = Ghat_rhs(nodes[n, i]; h = rhs_h)
            end
        end
        all(isfinite, rhs_nodes) || error("Non-finite Brunner RHS values detected.")

        @inbounds for n in 1:M
            for i in 1:m
                sni = nodes[n, i]
                history = 0.0

                for l in 1:(n - 1)
                    delta = (sni - mesh[l]) / h[l]
                    product_weights_fast!(hist_weights, hist_moments, delta, 1.0, power_coeffs, muf)
                    scale = hpow[l]
                    for j in 1:m
                        history += scale *
                                   hist_weights[j] *
                                   khat01_fast(sni, nodes[l, j]) *
                                   Uvals[l, j]
                    end
                end

                b[i] = rhs_nodes[n, i] - lambda_hat * history
                hcur = hpow[n]
                for j in 1:m
                    A[i, j] = (i == j ? 1.0 : 0.0) +
                              lambda_hat *
                              hcur *
                              current_weights0[i, j] *
                              khat01_fast(sni, nodes[n, j])
                end
            end

            solve_small_system!(xloc, A, b, m)
            for j in 1:m
                Uvals[n, j] = xloc[j]
            end
            all(isfinite, @view Uvals[n, :]) ||
                error("Non-finite Brunner solution values detected on panel $(n).")
        end
    end

    return BrunnerManufacturedSolution(
        mesh,
        cvec,
        Uvals,
        Int(M),
        Int(m),
        Float64(q),
        muf,
        lambdaf,
        lambda_hat,
        0,
        Float64(rhs_h),
        :fast,
        solve_time,
        Int(M) * Int(m),
    )
end

function parse_brunner_impl(impl)
    if impl isa Symbol
        s = String(impl)
    else
        s = lowercase(String(impl))
    end
    s == "old" && return :old
    s == "fast" && return :fast
    s == "both" && return :fast
    throw(ArgumentError("BRUNNER_IMPL must be old, fast, or both."))
end

function active_brunner_impl()
    return parse_brunner_impl(BRUNNER_IMPL)
end

function solve_brunner_manufactured(
    M;
    m = DEFAULT_BRUNNER_M,
    q = DEFAULT_BRUNNER_Q,
    mu = MU,
    lambda = LAMBDA,
    nq = DEFAULT_BRUNNER_NQ,
    rhs_h = RHS_DE_H,
    c = nothing,
    impl = active_brunner_impl(),
)
    impl_sym = parse_brunner_impl(impl)
    if impl_sym === :old
        return solve_brunner_manufactured_old(
            M;
            m = m,
            q = q,
            mu = mu,
            lambda = lambda,
            nq = nq,
            rhs_h = rhs_h,
            c = c,
        )
    else
        return solve_brunner_fast(
            M;
            m = m,
            q = q,
            mu = mu,
            lambda = lambda,
            rhs_h = rhs_h,
            c = c,
        )
    end
end

function eval_brunner_solution(x_eval, sol::BrunnerManufacturedSolution)
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
# Chebyshev-Duffy solve
# ============================================================

phi_beta_float(x::Real, beta::Real = BETA) =
    2.0 * ((1.0 + Float64(x)) / 2.0)^Float64(beta) - 1.0

phi_beta_s_float(s::Real, beta::Real = BETA) =
    2.0 * max(Float64(s), 0.0)^Float64(beta) - 1.0

invphi_beta_float(z::Real, beta::Real = BETA) =
    2.0 * ((1.0 + Float64(z)) / 2.0)^(1.0 / Float64(beta)) - 1.0

function cheb_eval(c::AbstractVector{<:Real}, z::Real)
    n = length(c) - 1
    n < 0 && throw(ArgumentError("coefficient vector must be nonempty."))
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

function u_exact_s(s)
    sf = max(Float64(s), 0.0)
    xp1 = 2.0 * sf
    sx = sqrt(xp1)
    return 1.0 + 0.2 * sx + 0.1 * xp1 + 0.05 * xp1 * sx
end

function exact_solution_coeffs_jfp(N::Int; beta = BETA)
    abs(Float64(beta) - 0.5) <= 100eps(Float64) ||
        throw(ArgumentError("exact_solution_coeffs_jfp is implemented analytically for beta=0.5."))

    coeffs = zeros(Float64, N + 1)
    a = 0.2 / sqrt(2.0)
    b = 0.1 / 2.0
    c = 0.05 / (2.0 * sqrt(2.0))

    p0 = 1.0 + a + b + c
    p1 = a + 2.0 * b + 3.0 * c
    p2 = b + 3.0 * c
    p3 = c

    coeffs[1] = p0 + 0.5 * p2
    N >= 1 && (coeffs[2] = p1 + 0.75 * p3)
    N >= 2 && (coeffs[3] = 0.5 * p2)
    N >= 3 && (coeffs[4] = 0.25 * p3)
    return coeffs
end

function cheb_project_lobatto_float(f, coeff_degree::Int, sample_degree::Int)
    coeff_degree <= sample_degree ||
        throw(ArgumentError("coeff_degree must not exceed sample_degree."))
    sample_degree >= 1 || throw(ArgumentError("sample_degree must be positive."))

    M = sample_degree
    vals = Vector{Float64}(undef, M + 1)
    @inbounds for j in 0:M
        vals[j + 1] = f(cos(pi * j / M))
    end

    c = zeros(Float64, coeff_degree + 1)
    @inbounds for k in 0:coeff_degree
        s = 0.0
        for j in 0:M
            wj = (j == 0 || j == M) ? 0.5 : 1.0
            s += wj * vals[j + 1] * cos(pi * k * j / M)
        end
        c[k + 1] = 2.0 * s / M
    end
    c[1] /= 2.0

    tail_count = min(20, coeff_degree + 1)
    tail_inf = maximum(abs, @view c[(end - tail_count + 1):end])

    return c, (
        sample_degree = M,
        coeff_degree = coeff_degree,
        tail_count = tail_count,
        tail_inf = tail_inf,
    )
end

function compute_rhs_coefficients_old_DE(maxN::Int; beta = BETA, rhs_h = RHS_DE_H)
    sample_degree = max(2048, 4 * maxN)
    f_z = z -> g_manufactured_DE(invphi_beta_float(z, beta); mu = MU, lambda = LAMBDA, h = rhs_h)

    stat = @timed cheb_project_lobatto_float(f_z, maxN, sample_degree)
    coeffs, info = stat.value
    all(isfinite, coeffs) || error("Non-finite Chebyshev RHS coefficients detected.")
    return coeffs, merge(info, (time = stat.time, rhs_h = Float64(rhs_h)))
end

function compute_rhs_coefficients(maxN::Int; beta = BETA, rhs_h = RHS_DE_H)
    if RHS_MODE == "ex7"
        RHS_EX7_TARGET_DEGREE[] = max(RHS_EX7_TARGET_DEGREE[], Int(maxN))
        coeffs = ensure_rhs_ex7_coeffs(RHS_EX7_TARGET_DEGREE[])
        info = _RHS_EX7_INFO[]
        return coeffs[1:(maxN + 1)], merge(info, (mode = RHS_MODE,))
    elseif RHS_MODE == "old_DE"
        coeffs, info = compute_rhs_coefficients_old_DE(maxN; beta = beta, rhs_h = rhs_h)
        return coeffs, merge(info, (mode = RHS_MODE,))
    else
        error("compute_rhs_coefficients is not used for RHS_MODE=$(RHS_MODE).")
    end
end

function build_kernel_coefficients(; beta = BETA)
    stat = @timed begin
        values = WeakCheb.sample_duffy_kernel(Kfun, KERNEL_I, KERNEL_J, beta)
        WeakCheb.cheb_coeffs_lobatto_2d(values)
    end
    C = stat.value
    tail = try
        WeakCheb.coefficient_tail_info(C)
    catch
        (row_rel = NaN, col_rel = NaN, max_coeff = maximum(abs, C))
    end
    return C, stat.time, tail
end

function dense_matrix(V)
    return V isa VoltOpt.PackedLowerHessenberg ? VoltOpt.packed_to_dense(V) : Matrix(V)
end

function solve_chebduffy_manufactured(
    N::Int,
    C::AbstractMatrix,
    rhs_coeffs::AbstractVector,
    xgrid::AbstractVector,
    exact_vals::AbstractVector;
    mu = MU,
    lambda = LAMBDA,
    beta = BETA,
)
    rhsN = rhs_coeffs[1:(N + 1)]

    matrix_stat = @timed begin
        V, info = VoltOpt.weak_voltmat_cheb(
            C;
            C_is_coeffs = true,
            mu = mu,
            sigma = SIGMA,
            beta = beta,
            N = N,
            diagnostics = false,
            output_format = :dense,
            return_info = true,
        )
        dense_matrix(V), info
    end

    Vdense, info = matrix_stat.value
    A = Matrix{Float64}(LinearAlgebra.I, N + 1, N + 1)
    A .+= Float64(lambda) .* Vdense

    solve_stat = @timed begin
        A \ rhsN
    end
    coeffs = solve_stat.value
    all(isfinite, coeffs) || error("Non-finite Chebyshev-Duffy coefficients detected for N=$(N).")

    max_error, rms_error = evaluate_error(
        x -> eval_jfp(coeffs, x; beta = beta),
        xgrid,
        exact_vals,
    )

    return (
        method = "Chebyshev-Duffy",
        N = Int(N),
        dof = Int(N + 1),
        mu = Float64(mu),
        lambda = Float64(lambda),
        beta = Float64(beta),
        construction_time = matrix_stat.time,
        solve_time = solve_stat.time,
        total_time = matrix_stat.time + solve_stat.time,
        max_error = max_error,
        rms_error = rms_error,
        residual_inf = norm(A * coeffs - rhsN, Inf),
        coeffs = coeffs,
        info = info,
    )
end

# ============================================================
# Benchmark utilities
# ============================================================

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

function check_rhs_DE_stability(;
    x_test = [-0.99, -0.75, -0.5, 0.0, 0.5, 0.9, 1.0],
    hs = [0.05, 0.035, 0.025, 0.02, 0.015],
)
    println()
    println("Checking RHS stability for h = ", hs)
    values = Dict{Float64,Vector{Float64}}()

    for h in hs
        info = tanhsinh_truncation_info(h = h)
        vals = [g_manufactured_DE(x; mu = MU, lambda = LAMBDA, h = h) for x in x_test]
        all(isfinite, vals) || error("Non-finite RHS value detected in DE stability check.")
        values[Float64(h)] = vals
        @printf(
            "  h = %.3f, K = %d, endpoint distances = (%.3e, %.3e)\n",
            h,
            info.K,
            info.left_distance,
            info.right_distance,
        )
    end

    diffs = NamedTuple[]
    for i in 1:(length(hs) - 1)
        h1 = Float64(hs[i])
        h2 = Float64(hs[i + 1])
        diff = maximum(abs.(values[h1] .- values[h2]))
        push!(diffs, (h1 = h1, h2 = h2, max_diff = diff))
        @printf("  max difference h=%.3f vs h=%.3f: %.6e\n", h1, h2, diff)
    end

    diff_0025_002 = maximum(abs.(values[0.025] .- values[0.02]))
    diff_002_0015 = maximum(abs.(values[0.02] .- values[0.015]))
    @printf("max |g_h=0.025 - g_h=0.02| = %.6e\n", diff_0025_002)
    @printf("max |g_h=0.02  - g_h=0.015| = %.6e\n", diff_002_0015)
    if diff_002_0015 > 1e-12
        @warn "DE RHS h=0.02 differs from h=0.015 by more than 1e-12; consider decreasing RHS_DE_H." diff_002_0015
    else
        println("RHS DE stability appears sufficient for near machine-precision solve.")
    end

    return (
        x_test = x_test,
        hs = hs,
        values = values,
        diffs = diffs,
        diff_0025_002 = diff_0025_002,
        diff_002_0015 = diff_002_0015,
    )
end

function check_rhs_stability()
    if RHS_MODE == "old_DE"
        return check_rhs_DE_stability()
    elseif RHS_MODE == "ex7"
        coeffs = ensure_rhs_ex7_coeffs(RHS_EX7_TARGET_DEGREE[])
        info = _RHS_EX7_INFO[]
        x_test = [-0.99, -0.75, -0.5, 0.0, 0.5, 0.9, 1.0]
        vals = [g_manufactured_like_ex7(x) for x in x_test]
        all(isfinite, vals) || error("Non-finite ex7-style RHS values detected.")

        println()
        println("Checking ex7-style RHS projection stability...")
        @printf("  coefficient degree = %d\n", length(coeffs) - 1)
        @printf("  high/check coeff diff inf = %.6e, rel = %.6e\n",
            info.coeff_diff_inf,
            info.coeff_diff_rel,
        )
        @printf("  high sample degree = %d, check sample degree = %d\n",
            info.high.sample_degree,
            info.check.sample_degree,
        )
        @printf("  tanh-sinh max k = %d, max last pair = %.3e\n",
            info.high.ts_max_terms,
            info.high.ts_max_last_term,
        )
        return (
            mode = RHS_MODE,
            values = Dict(0.0 => vals),
            coeff_diff_inf = info.coeff_diff_inf,
            coeff_diff_rel = info.coeff_diff_rel,
            info = info,
        )
    elseif RHS_MODE == "operator_consistent"
        return (
            mode = RHS_MODE,
            values = Dict(0.0 => Float64[]),
        )
    else
        error("Unsupported RHS_MODE=$(RHS_MODE).")
    end
end

function check_product_weight_fast_against_old(; nq = 128)
    println()
    println("Checking fast weights against old quadrature weights...")
    max_diff = 0.0
    worst = nothing

    for m in DEFAULT_BRUNNER_M_VALUES
        cvec = radau_right_nodes(m)
        coeffs = lagrange_power_coeffs(cvec)
        cases = Tuple{Float64,Float64}[]
        for i in 1:m
            push!(cases, (cvec[i], cvec[i]))
        end
        append!(cases, [(1.0001, 1.0), (1.25, 1.0), (2.0, 1.0), (10.0, 1.0), (1.0e4, 1.0), (1.0e8, 1.0)])

        for (delta, theta_max) in cases
            for j in 1:m
                local_nq = (delta > theta_max && delta - theta_max < 1e-2) ? max(nq, 1024) : nq
                w_old = product_weight(delta, theta_max, j, cvec, MU; nq = local_nq)
                w_fast = product_weight_fast(delta, theta_max, j, coeffs, MU)
                diff = abs(w_old - w_fast)
                if diff > max_diff
                    max_diff = diff
                    worst = (m = m, delta = delta, theta_max = theta_max, j = j, old = w_old, fast = w_fast)
                end
            end
        end
    end

    @printf("  max |W_old - W_fast| = %.6e\n", max_diff)
    if worst !== nothing
        @printf(
            "  worst: m=%d, delta=%.6e, theta_max=%.6e, j=%d, old=%.16e, fast=%.16e\n",
            worst.m,
            worst.delta,
            worst.theta_max,
            worst.j,
            worst.old,
            worst.fast,
        )
    end
    if max_diff > 1e-10
        @warn "Fast analytic product weights differ from old quadrature weights by more than 1e-10." max_diff worst
    end
    return (max_diff = max_diff, worst = worst)
end

function check_brunner_fast_against_old(; M = 16, qs = [1.0, 2.0, 4.0])
    println()
    println("Checking Brunner fast implementation against old implementation...")
    rows = NamedTuple[]

    for m in DEFAULT_BRUNNER_M_VALUES
        for q in qs
            old = solve_brunner_manufactured_old(
                M;
                m = m,
                q = q,
                mu = MU,
                lambda = LAMBDA,
                nq = DEFAULT_BRUNNER_NQ,
                rhs_h = RHS_DE_H,
            )
            fast = solve_brunner_fast(
                M;
                m = m,
                q = q,
                mu = MU,
                lambda = LAMBDA,
                rhs_h = RHS_DE_H,
            )
            diff = maximum(abs.(old.Uvals .- fast.Uvals))
            all(isfinite, old.Uvals) || error("Non-finite old Brunner check solution.")
            all(isfinite, fast.Uvals) || error("Non-finite fast Brunner check solution.")
            push!(rows, (
                M = M,
                m = m,
                q = Float64(q),
                old_time = old.solve_time,
                fast_time = fast.solve_time,
                speedup = old.solve_time / fast.solve_time,
                max_abs_diff = diff,
            ))
            @printf(
                "  Brunner fast/old check: m=%d, q=%.1f, M=%d, max diff=%.6e, speedup=%.3f\n",
                m,
                q,
                M,
                diff,
                old.solve_time / fast.solve_time,
            )
            if diff > 1e-10
                @warn "Fast and old Brunner solutions differ by more than 1e-10." M m q diff
            end
        end
    end

    return rows
end

function run_brunner_old_fast_time_ratio(; Ms = [64, 128, 256, 512], m = 2, q = 2.0)
    println()
    println("Running Brunner old-vs-fast timing comparison...")
    rows = NamedTuple[]

    # Keep compile cost out of the reported rows.
    solve_brunner_manufactured_old(8; m = m, q = q, nq = min(DEFAULT_BRUNNER_NQ, 16), rhs_h = RHS_DE_H)
    solve_brunner_fast(8; m = m, q = q, rhs_h = RHS_DE_H)

    for M in Ms
        old = solve_brunner_manufactured_old(
            M;
            m = m,
            q = q,
            mu = MU,
            lambda = LAMBDA,
            nq = DEFAULT_BRUNNER_NQ,
            rhs_h = RHS_DE_H,
        )
        fast = solve_brunner_fast(
            M;
            m = m,
            q = q,
            mu = MU,
            lambda = LAMBDA,
            rhs_h = RHS_DE_H,
        )
        diff = maximum(abs.(old.Uvals .- fast.Uvals))
        speedup = old.solve_time / fast.solve_time
        row = (
            M = Int(M),
            dof = Int(M) * Int(m),
            m = Int(m),
            q = Float64(q),
            old_time = old.solve_time,
            fast_time = fast.solve_time,
            speedup = speedup,
            max_abs_diff = diff,
        )
        push!(rows, row)
        @printf(
            "  M = %5d, dof = %6d, old = %.6e, fast = %.6e, speedup = %.3f, max diff = %.6e\n",
            row.M,
            row.dof,
            row.old_time,
            row.fast_time,
            row.speedup,
            row.max_abs_diff,
        )
    end

    return rows
end

function plot_brunner_old_fast_time_ratio(rows; fig_dir = FIGURES_DIR)
    isempty(rows) && return false
    try
        @eval import Plots
        Base.invokelatest(Plots.gr)

        fig = Base.invokelatest(
            Plots.plot,
            [Float64(r.dof) for r in rows],
            [Float64(r.speedup) for r in rows];
            xscale = :log10,
            marker = :circle,
            linewidth = 2.2,
            markersize = 4.8,
            color = :black,
            label = nothing,
            xlabel = "degrees of freedom",
            ylabel = "old / fast solve time",
            framestyle = :box,
            grid = false,
            size = (760, 480),
            dpi = 300,
        )

        pdf_path = joinpath(fig_dir, "brunner_old_vs_fast_time_ratio.pdf")
        png_path = joinpath(fig_dir, "brunner_old_vs_fast_time_ratio.png")
        Base.invokelatest(Plots.savefig, fig, pdf_path)
        Base.invokelatest(Plots.savefig, fig, png_path)
        println("Saved figure: ", pdf_path)
        println("Saved figure: ", png_path)
        return true
    catch err
        @warn "Brunner old-vs-fast timing plot skipped." exception = (err, catch_backtrace())
        return false
    end
end

function csv_cell(x)
    x === missing && return "missing"
    if x isa AbstractFloat
        return isfinite(x) ? @sprintf("%.17e", x) : string(x)
    elseif x isa Integer
        return string(x)
    elseif x isa AbstractString
        if occursin(",", x) || occursin("\"", x) || occursin("\n", x)
            return "\"" * replace(x, "\"" => "\"\"") * "\""
        end
        return x
    else
        return string(x)
    end
end

function write_csv(path::AbstractString, headers::Vector{String}, rows)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(headers, ","))
        for r in rows
            println(io, join((csv_cell(getfield(r, Symbol(h))) for h in headers), ","))
        end
    end
end

function env_limited_values(values, env_name::AbstractString; candidates = values)
    raw = get(ENV, env_name, nothing)
    raw === nothing && return values
    maxval = tryparse(Int, raw)
    if maxval === nothing
        @warn "Ignoring invalid integer environment variable." env_name raw
        return values
    end
    filtered = [v for v in candidates if v <= maxval]
    isempty(filtered) &&
        error("$(env_name)=$(maxval) removes all benchmark values.")
    return filtered
end

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

function mesh_diagnostics(M::Int, q::Real, c)
    mesh = graded_mesh(M, q)
    diffs = diff(mesh)
    hmin = minimum(diffs)
    hmax = maximum(diffs)
    mesh_valid = all(diffs .> 0.0)
    ratio = hmin > 0.0 ? hmax / hmin : Inf
    cvec = validate_collocation_nodes(c)

    collapsed_x_count = 0
    min_positive_s = Inf
    min_positive_xplus1 = Inf
    node_count = M * length(cvec)
    @inbounds for n in 1:M
        left = mesh[n]
        h = mesh[n + 1] - left
        for ci in cvec
            s = left + ci * h
            if s > 0.0 && s < min_positive_s
                min_positive_s = s
            end
            x = 2.0 * s - 1.0
            if x == -1.0
                collapsed_x_count += 1
            end
            xplus1 = x + 1.0
            if xplus1 > 0.0 && xplus1 < min_positive_xplus1
                min_positive_xplus1 = xplus1
            end
        end
    end
    collapsed_x_fraction = collapsed_x_count / node_count
    numerically_extreme = collapsed_x_fraction > 0.01

    return (
        mesh = mesh,
        hmin = hmin,
        hmax = hmax,
        mesh_ratio = ratio,
        mesh_valid = mesh_valid,
        collapsed_x_count = collapsed_x_count,
        collapsed_x_fraction = collapsed_x_fraction,
        min_positive_s = min_positive_s,
        min_positive_xplus1 = min_positive_xplus1,
        numerically_extreme = numerically_extreme,
    )
end

function warn_mesh_if_needed(M::Int, q::Real, m::Int, diag)
    key = (M, Float64(q))
    key in _MESH_WARNING_KEYS && return nothing
    if !diag.mesh_valid
        @warn "Brunner graded mesh has repeated or decreasing points; skipping parameter group." M q hmin = diag.hmin hmax = diag.hmax mesh_ratio = diag.mesh_ratio
    elseif diag.hmin < 100eps(Float64)
        @warn "Brunner graded mesh has hmin < 100eps(Float64)." M q hmin = diag.hmin hmax = diag.hmax mesh_ratio = diag.mesh_ratio
    end
    if diag.mesh_ratio > 1e14
        @warn "Brunner graded mesh ratio is larger than 1e14." M q hmin = diag.hmin hmax = diag.hmax mesh_ratio = diag.mesh_ratio
    end
    if diag.collapsed_x_count > 0
        @warn "Some graded collocation nodes collapse to x=-1 in Float64." M q m collapsed_x_count = diag.collapsed_x_count collapsed_x_fraction = diag.collapsed_x_fraction min_positive_s = diag.min_positive_s min_positive_xplus1 = diag.min_positive_xplus1
    end
    if diag.numerically_extreme
        @warn "Brunner graded mesh is numerically extreme because more than 1% of x-nodes collapse to -1." M q m collapsed_x_fraction = diag.collapsed_x_fraction
    end
    if !diag.mesh_valid || diag.hmin < 100eps(Float64) || diag.mesh_ratio > 1e14 || diag.collapsed_x_count > 0
        push!(_MESH_WARNING_KEYS, key)
    end
    return nothing
end

is_valid_brunner_row(r) =
    hasfield(typeof(r), :mesh_valid) ? Bool(r.mesh_valid) && isfinite(r.max_error) : isfinite(r.max_error)

const _BRUNNER_QEXTENDED = Ref(false)
const _BRUNNER_QEXTREME = Ref(false)
const _MESH_WARNING_KEYS = Set{Tuple{Int,Float64}}()

function run_brunner_parameter_sweep(
    Ms,
    m_values,
    q_values,
    xgrid,
    exact_vals;
    impl = active_brunner_impl(),
)
    rows = NamedTuple[]
    println()
    impl_sym = parse_brunner_impl(impl)
    if impl_sym === :fast
        println("Running optimized Brunner parameter sweep...")
    else
        println("Running Brunner parameter sweep with old quadrature weights...")
    end

    for m in m_values
        cvec_diag = radau_right_nodes(Int(m))
        for q in q_values
            # Compile this local order/mesh path outside the reported rows.
            solve_brunner_manufactured(
                2;
                m = m,
                q = q,
                nq = min(DEFAULT_BRUNNER_NQ, 16),
                rhs_h = RHS_DE_H,
                impl = impl_sym,
            )

            for M in Ms
                mesh_diag = mesh_diagnostics(Int(M), q, cvec_diag)
                warn_mesh_if_needed(Int(M), q, Int(m), mesh_diag)
                if !mesh_diag.mesh_valid
                    row = (
                        method = "Brunner",
                        M = Int(M),
                        dof = Int(M) * Int(m),
                        m = Int(m),
                        q = Float64(q),
                        mu = MU,
                        lambda = LAMBDA,
                        nq = impl_sym === :fast ? 0 : DEFAULT_BRUNNER_NQ,
                        impl = String(impl_sym),
                        hmin = mesh_diag.hmin,
                        hmax = mesh_diag.hmax,
                        mesh_ratio = mesh_diag.mesh_ratio,
                        mesh_valid = false,
                        collapsed_x_count = mesh_diag.collapsed_x_count,
                        collapsed_x_fraction = mesh_diag.collapsed_x_fraction,
                        min_positive_s = mesh_diag.min_positive_s,
                        min_positive_xplus1 = mesh_diag.min_positive_xplus1,
                        numerically_extreme = mesh_diag.numerically_extreme,
                        solve_time = NaN,
                        total_time = NaN,
                        max_error = NaN,
                        rms_error = NaN,
                    )
                    push!(rows, row)
                    @printf(
                        "impl = %s, m = %d, q = %.1f, M = %5d, dof = %6d, mesh invalid, hmin = %.3e, hmax = %.3e, ratio = %.3e\n",
                        row.impl,
                        row.m,
                        row.q,
                        row.M,
                        row.dof,
                        row.hmin,
                        row.hmax,
                        row.mesh_ratio,
                    )
                    continue
                end

                sol = solve_brunner_manufactured(
                    M;
                    m = m,
                    q = q,
                    mu = MU,
                    lambda = LAMBDA,
                    nq = DEFAULT_BRUNNER_NQ,
                    rhs_h = RHS_DE_H,
                    impl = impl_sym,
                )
                max_error, rms_error =
                    evaluate_error(x -> eval_brunner_solution(x, sol), xgrid, exact_vals)

                row = (
                    method = "Brunner",
                    M = Int(M),
                    dof = sol.dof,
                    m = sol.m,
                    q = sol.q,
                    mu = sol.mu,
                    lambda = sol.lambda,
                    nq = sol.nq,
                    impl = String(sol.impl),
                    hmin = mesh_diag.hmin,
                    hmax = mesh_diag.hmax,
                    mesh_ratio = mesh_diag.mesh_ratio,
                    mesh_valid = true,
                    collapsed_x_count = mesh_diag.collapsed_x_count,
                    collapsed_x_fraction = mesh_diag.collapsed_x_fraction,
                    min_positive_s = mesh_diag.min_positive_s,
                    min_positive_xplus1 = mesh_diag.min_positive_xplus1,
                    numerically_extreme = mesh_diag.numerically_extreme,
                    solve_time = sol.solve_time,
                    total_time = sol.solve_time,
                    max_error = max_error,
                    rms_error = rms_error,
                )
                push!(rows, row)

                @printf(
                    "impl = %s, m = %d, q = %.1f, M = %5d, dof = %6d, time = %.6e, max_error = %.6e, rms_error = %.6e, hmin = %.3e, ratio = %.3e\n",
                    row.impl,
                    row.m,
                    row.q,
                    row.M,
                    row.dof,
                    row.solve_time,
                    row.max_error,
                    row.rms_error,
                    row.hmin,
                    row.mesh_ratio,
                )
            end
        end
    end
    return rows
end

function run_chebduffy_benchmark(Ns, xgrid, exact_vals)
    rows = NamedTuple[]

    println()
    println("Running Chebyshev-Duffy benchmark...")
    C, kernel_time, kernel_tail = build_kernel_coefficients(beta = BETA)
    @printf("Duffy-Chebyshev kernel coefficient matrix size = %d x %d\n", size(C, 1), size(C, 2))
    @printf("one-time kernel coefficient build time = %.6e s\n", kernel_time)
    @printf(
        "kernel tail row_rel = %.3e, col_rel = %.3e, max_coeff = %.3e\n",
        kernel_tail.row_rel,
        kernel_tail.col_rel,
        kernel_tail.max_coeff,
    )

    rhs_coeffs, rhs_info = compute_rhs_coefficients(maximum(Ns); beta = BETA, rhs_h = RHS_DE_H)
    if RHS_MODE == "ex7"
        @printf("RHS mode = ex7-style BigFloat tanh-sinh + Lobatto projection\n")
        @printf("RHS BigFloat precision = %d bits\n", rhs_info.precision)
        @printf("RHS tanh-sinh h = %s, atol = %s, max_terms = %d\n",
            rhs_info.ts_h,
            rhs_info.ts_atol,
            rhs_info.ts_max_terms,
        )
        @printf("RHS projection degree = %d, high sample degree = %d, check sample degree = %d\n",
            rhs_info.high.coeff_degree,
            rhs_info.high.sample_degree,
            rhs_info.check.sample_degree,
        )
        @printf("RHS coeff high/check diff inf = %.6e, rel = %.6e\n",
            rhs_info.coeff_diff_inf,
            rhs_info.coeff_diff_rel,
        )
        @printf("RHS tail last %d coefficients = %.3e\n",
            rhs_info.high.tail_count,
            rhs_info.high.tail_inf,
        )
        @printf("RHS tanh-sinh calls = %d, max k = %d, max last pair = %.3e\n",
            rhs_info.high.ts_calls,
            rhs_info.high.ts_max_terms,
            rhs_info.high.ts_max_last_term,
        )
        @printf("RHS projection time = %.6e s (check %.6e s)\n",
            rhs_info.high_time,
            rhs_info.check_time,
        )
    else
        @printf(
            "RHS projection degree = %d, sample degree = %d, rhs_h = %.3f, time = %.6e s\n",
            rhs_info.coeff_degree,
            rhs_info.sample_degree,
            rhs_info.rhs_h,
            rhs_info.time,
        )
        @printf("RHS tail last %d coefficients = %.3e\n", rhs_info.tail_count, rhs_info.tail_inf)
    end

    for warm_N in unique([first(Ns), min(maximum(Ns), max(64, first(Ns)))])
        solve_chebduffy_manufactured(
            warm_N,
            C,
            rhs_coeffs,
            [-1.0, 0.0, 1.0],
            u_exact.([-1.0, 0.0, 1.0]);
            mu = MU,
            lambda = LAMBDA,
            beta = BETA,
        )
    end

    for N in Ns
        result = solve_chebduffy_manufactured(
            N,
            C,
            rhs_coeffs,
            xgrid,
            exact_vals;
            mu = MU,
            lambda = LAMBDA,
            beta = BETA,
        )
        row = (
            method = result.method,
            N = result.N,
            dof = result.dof,
            mu = result.mu,
            lambda = result.lambda,
            beta = result.beta,
            construction_time = result.construction_time,
            solve_time = result.solve_time,
            total_time = result.total_time,
            max_error = result.max_error,
            rms_error = result.rms_error,
        )
        push!(rows, row)

        @printf(
            "N = %5d, dof = %5d, construct = %.6e, solve = %.6e, total = %.6e, max_error = %.6e, rms_error = %.6e\n",
            row.N,
            row.dof,
            row.construction_time,
            row.solve_time,
            row.total_time,
            row.max_error,
            row.rms_error,
        )
    end
    return rows
end

function run_operator_consistent_diagnostic(Ns, xgrid, exact_vals)
    rows = NamedTuple[]

    println()
    println("Running Chebyshev-Duffy operator-consistent diagnostic...")
    C, kernel_time, kernel_tail = build_kernel_coefficients(beta = BETA)
    @printf("Duffy-Chebyshev kernel coefficient matrix size = %d x %d\n", size(C, 1), size(C, 2))
    @printf("one-time kernel coefficient build time = %.6e s\n", kernel_time)
    @printf(
        "kernel tail row_rel = %.3e, col_rel = %.3e, max_coeff = %.3e\n",
        kernel_tail.row_rel,
        kernel_tail.col_rel,
        kernel_tail.max_coeff,
    )

    for warm_N in unique([first(Ns), min(maximum(Ns), max(64, first(Ns)))])
        V, _ = VoltOpt.weak_voltmat_cheb(
            C;
            C_is_coeffs = true,
            mu = MU,
            sigma = SIGMA,
            beta = BETA,
            N = warm_N,
            diagnostics = false,
            output_format = :dense,
            return_info = true,
        )
        dense_matrix(V)
    end

    for N in Ns
        stat = @timed begin
            V, info = VoltOpt.weak_voltmat_cheb(
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
            Vdense = dense_matrix(V)
            A = Matrix{Float64}(LinearAlgebra.I, N + 1, N + 1)
            A .+= LAMBDA .* Vdense
            u_coeff = exact_solution_coeffs_jfp(N; beta = BETA)
            g_coeff = A * u_coeff
            solve_stat = @timed A \ g_coeff
            u_num_coeff = solve_stat.value
            coeff_error = norm(u_num_coeff .- u_coeff, Inf)
            max_error, rms_error = evaluate_error(
                x -> eval_jfp(u_num_coeff, x; beta = BETA),
                xgrid,
                exact_vals,
            )
            cond_est = N <= COND_MAX_N ? cond(A) : NaN
            (
                info = info,
                solve_time = solve_stat.time,
                coeff_error = coeff_error,
                max_error = max_error,
                rms_error = rms_error,
                cond_est = cond_est,
            )
        end

        result = stat.value
        row = (
            N = Int(N),
            dof = Int(N + 1),
            total_time = stat.time,
            max_error = result.max_error,
            rms_error = result.rms_error,
            coeff_error = result.coeff_error,
            cond_est = result.cond_est,
        )
        push!(rows, row)

        @printf(
            "N = %5d, dof = %5d, total = %.6e, max_error = %.6e, rms_error = %.6e, coeff_error = %.6e, cond_est = %.3e\n",
            row.N,
            row.dof,
            row.total_time,
            row.max_error,
            row.rms_error,
            row.coeff_error,
            row.cond_est,
        )
    end

    return rows
end

function plot_operator_consistent_diagnostic(rows; fig_dir = FIGURES_DIR)
    try
        @eval import Plots
        Base.invokelatest(Plots.gr)

        Ns = [Float64(r.N) for r in rows]
        errs = [Float64(r.max_error) for r in rows]

        fig = Base.invokelatest(
            Plots.plot,
            Ns,
            errs;
            xscale = :log10,
            yscale = :log10,
            marker = :circle,
            linewidth = 2.2,
            markersize = 4.5,
            color = :black,
            label = nothing,
            xlabel = "N",
            ylabel = "max error",
            framestyle = :box,
            grid = false,
            size = (760, 480),
            dpi = 300,
        )

        pdf_path = joinpath(fig_dir, "chebduffy_operator_consistent_error_vs_N.pdf")
        png_path = joinpath(fig_dir, "chebduffy_operator_consistent_error_vs_N.png")
        Base.invokelatest(Plots.savefig, fig, pdf_path)
        Base.invokelatest(Plots.savefig, fig, png_path)
        println("Saved figure: ", pdf_path)
        println("Saved figure: ", png_path)
        return true
    catch err
        @warn "Operator-consistent diagnostic plotting skipped." exception = (err, catch_backtrace())
        return false
    end
end

function combined_rows(brunner_rows, cheb_rows)
    rows = NamedTuple[]
    for r in filter(is_valid_brunner_row, brunner_rows)
        push!(rows, (
            method = r.method,
            dof = r.dof,
            total_time = r.total_time,
            max_error = r.max_error,
            rms_error = r.rms_error,
            N_or_M = r.M,
            extra = @sprintf("m=%d,q=%.1f,impl=%s", r.m, r.q, r.impl),
            hmin = r.hmin,
            hmax = r.hmax,
            mesh_ratio = r.mesh_ratio,
            mesh_valid = r.mesh_valid,
            collapsed_x_count = r.collapsed_x_count,
            collapsed_x_fraction = r.collapsed_x_fraction,
            min_positive_s = r.min_positive_s,
            min_positive_xplus1 = r.min_positive_xplus1,
            numerically_extreme = r.numerically_extreme,
        ))
    end
    for r in cheb_rows
        push!(rows, (
            method = r.method,
            dof = r.dof,
            total_time = r.total_time,
            max_error = r.max_error,
            rms_error = r.rms_error,
            N_or_M = r.N,
            extra = @sprintf("beta=%.1f", r.beta),
            hmin = missing,
            hmax = missing,
            mesh_ratio = missing,
            mesh_valid = missing,
            collapsed_x_count = missing,
            collapsed_x_fraction = missing,
            min_positive_s = missing,
            min_positive_xplus1 = missing,
            numerically_extreme = missing,
        ))
    end
    return rows
end

function brunner_best_envelope(brunner_rows)
    valid_rows = filter(is_valid_brunner_row, brunner_rows)
    env = NamedTuple[]
    for i in eachindex(valid_rows)
        r = valid_rows[i]
        dominated = false
        for j in eachindex(valid_rows)
            i == j && continue
            s = valid_rows[j]
            if s.max_error <= r.max_error &&
               s.total_time <= r.total_time &&
               (s.max_error < r.max_error || s.total_time < r.total_time)
                dominated = true
                break
            end
        end

        if !dominated
            push!(env, (
                dof = r.dof,
                total_time = r.total_time,
                max_error = r.max_error,
                m = r.m,
                q = r.q,
                M = r.M,
                hmin = r.hmin,
                hmax = r.hmax,
                mesh_ratio = r.mesh_ratio,
                mesh_valid = r.mesh_valid,
                collapsed_x_count = r.collapsed_x_count,
                collapsed_x_fraction = r.collapsed_x_fraction,
                min_positive_s = r.min_positive_s,
                min_positive_xplus1 = r.min_positive_xplus1,
                numerically_extreme = r.numerically_extreme,
            ))
        end
    end

    return sort(env; by = r -> r.max_error, rev = true)
end

function brunner_group_rows(brunner_rows, m, q)
    rows = [r for r in brunner_rows if is_valid_brunner_row(r) && r.m == m && isapprox(r.q, q; rtol = 0.0, atol = 1e-14)]
    return sort(rows; by = r -> r.M)
end

marker_for_m(m::Int) = m == 2 ? :circle : (m == 3 ? :square : :utriangle)

function rhs_h_tag(h::Real)
    s = @sprintf("%.16g", Float64(h))
    s = replace(s, "." => "", "-" => "m", "+" => "")
    return "h" * s
end

function rhs_output_tag()
    if RHS_MODE == "ex7"
        return "ex7rhs"
    elseif RHS_MODE == "old_DE"
        return "oldDE_$(rhs_h_tag(RHS_DE_H))"
    elseif RHS_MODE == "operator_consistent"
        return "operator_consistent"
    else
        return replace(RHS_MODE, r"[^A-Za-z0-9_]+" => "_")
    end
end

function brunner_impl_output_tag()
    BRUNNER_IMPL == "both" && return "fast_bothcheck"
    impl = parse_brunner_impl(BRUNNER_IMPL)
    return impl === :old ? "old" : "fast"
end

function brunner_output_tag()
    tag = "$(rhs_output_tag())_$(brunner_impl_output_tag())"
    if _BRUNNER_QEXTREME[]
        tag *= "_qextreme"
    elseif _BRUNNER_QEXTENDED[]
        tag *= "_qextended"
    end
    return tag
end

function compare_output_filename(rhs_tag::AbstractString)
    impl_tag = brunner_impl_output_tag()
    if _BRUNNER_QEXTREME[]
        return "compare_brunner_$(impl_tag)_qextreme_chebduffy_manufactured_$(rhs_tag).csv"
    elseif _BRUNNER_QEXTENDED[]
        return "compare_brunner_$(impl_tag)_qextended_chebduffy_manufactured_$(rhs_tag).csv"
    end
    return "compare_brunner_$(impl_tag)_mq_chebduffy_manufactured_$(rhs_tag).csv"
end

function plot_comparison(brunner_rows, cheb_rows, envelope_rows; fig_dir = FIGURES_DIR)
    try
        @eval import Plots
        Base.invokelatest(Plots.gr)

        valid_brunner_rows = filter(is_valid_brunner_row, brunner_rows)
        c_dof = [Float64(r.dof) for r in cheb_rows]
        c_time = [Float64(r.total_time) for r in cheb_rows]
        c_err = [Float64(r.max_error) for r in cheb_rows]
        e_time = [Float64(r.total_time) for r in envelope_rows]
        e_err = [Float64(r.max_error) for r in envelope_rows]
        tag = brunner_output_tag()

        function save_pdf_png(fig, stem)
            pdf_path = joinpath(fig_dir, stem * ".pdf")
            png_path = joinpath(fig_dir, stem * ".png")
            Base.invokelatest(Plots.savefig, fig, pdf_path)
            Base.invokelatest(Plots.savefig, fig, png_path)
            println("Saved figure: ", pdf_path)
            println("Saved figure: ", png_path)
        end

        common = (
            xscale = :log10,
            yscale = :log10,
            linewidth = 1.35,
            markersize = 3.4,
            framestyle = :box,
            grid = false,
            legend = :outerright,
            legendfontsize = 6,
            size = (980, 560),
            dpi = 300,
        )

        combos = sort(unique((r.m, r.q) for r in valid_brunner_rows))

        p1 = Base.invokelatest(
            Plots.plot,
            Float64[],
            Float64[];
            common...,
            label = false,
            xlabel = "degrees of freedom",
            ylabel = "max error",
        )
        for (m, q) in combos
            rows = brunner_group_rows(brunner_rows, m, q)
            Base.invokelatest(
                Plots.plot!,
                p1,
                [Float64(r.dof) for r in rows],
                [Float64(r.max_error) for r in rows];
                marker = marker_for_m(m),
                label = @sprintf("Brunner m=%d, q=%.1f", m, q),
            )
        end
        Base.invokelatest(
            Plots.plot!,
            p1,
            c_dof,
            c_err;
            marker = :star5,
            color = :black,
            linestyle = :dash,
            linewidth = 2.8,
            markersize = 5.5,
            label = "Chebyshev--Duffy beta=0.5",
        )
        save_pdf_png(p1, "compare_mq_$(tag)_error_vs_dof")

        p2 = Base.invokelatest(
            Plots.plot,
            Float64[],
            Float64[];
            common...,
            label = false,
            xlabel = "degrees of freedom",
            ylabel = "total time (sec)",
        )
        for (m, q) in combos
            rows = brunner_group_rows(brunner_rows, m, q)
            Base.invokelatest(
                Plots.plot!,
                p2,
                [Float64(r.dof) for r in rows],
                [Float64(r.total_time) for r in rows];
                marker = marker_for_m(m),
                label = @sprintf("Brunner m=%d, q=%.1f", m, q),
            )
        end
        Base.invokelatest(
            Plots.plot!,
            p2,
            c_dof,
            c_time;
            marker = :star5,
            color = :black,
            linestyle = :dash,
            linewidth = 2.8,
            markersize = 5.5,
            label = "Chebyshev--Duffy beta=0.5",
        )
        save_pdf_png(p2, "compare_mq_$(tag)_time_vs_dof")

        p3 = Base.invokelatest(
            Plots.plot,
            Float64[],
            Float64[];
            common...,
            label = false,
            xlabel = "max error",
            ylabel = "total time (sec)",
        )
        for (m, q) in combos
            rows = brunner_group_rows(brunner_rows, m, q)
            Base.invokelatest(
                Plots.plot!,
                p3,
                [Float64(r.max_error) for r in rows],
                [Float64(r.total_time) for r in rows];
                marker = marker_for_m(m),
                label = @sprintf("Brunner m=%d, q=%.1f", m, q),
            )
        end
        Base.invokelatest(
            Plots.plot!,
            p3,
            c_err,
            c_time;
            marker = :star5,
            color = :black,
            linestyle = :dash,
            linewidth = 2.8,
            markersize = 5.5,
            label = "Chebyshev--Duffy beta=0.5",
        )
        save_pdf_png(p3, "compare_mq_$(tag)_time_vs_error")

        p4 = Base.invokelatest(
            Plots.scatter,
            [Float64(r.max_error) for r in valid_brunner_rows],
            [Float64(r.total_time) for r in valid_brunner_rows];
            xscale = :log10,
            yscale = :log10,
            marker = :circle,
            markersize = 3.3,
            markeralpha = 0.32,
            color = :steelblue,
            label = "Brunner parameter points",
            xlabel = "max error",
            ylabel = "total time (sec)",
            framestyle = :box,
            grid = false,
            legend = :outerright,
            legendfontsize = 7,
            size = (940, 540),
            dpi = 300,
        )
        Base.invokelatest(
            Plots.plot!,
            p4,
            e_err,
            e_time;
            marker = :circle,
            color = :navy,
            linewidth = 2.5,
            markersize = 4.5,
            label = "Brunner best envelope",
        )
        Base.invokelatest(
            Plots.plot!,
            p4,
            c_err,
            c_time;
            marker = :star5,
            color = :black,
            linestyle = :dash,
            linewidth = 2.8,
            markersize = 5.5,
            label = "Chebyshev--Duffy beta=0.5",
        )
        save_pdf_png(p4, "compare_$(tag)_time_vs_error_with_envelope")

        return true
    catch err
        @warn "Plotting skipped; CSV output is still available." exception = (err, catch_backtrace())
        return false
    end
end

function plot_brunner_error_vs_q_by_M(brunner_rows; fig_dir = FIGURES_DIR)
    valid_rows = filter(is_valid_brunner_row, brunner_rows)
    isempty(valid_rows) && return false

    try
        @eval import Plots
        Base.invokelatest(Plots.gr)

        Ms = sort(unique(r.M for r in valid_rows))
        fig = Base.invokelatest(
            Plots.plot,
            Float64[],
            Float64[];
            yscale = :log10,
            marker = :circle,
            linewidth = 1.7,
            markersize = 4.0,
            framestyle = :box,
            grid = false,
            legend = :outerright,
            legendfontsize = 7,
            xlabel = "q",
            ylabel = "best max error over m",
            size = (880, 520),
            dpi = 300,
            label = false,
        )

        for M in Ms
            rowsM = [r for r in valid_rows if r.M == M]
            qs = sort(unique(r.q for r in rowsM))
            qvals = Float64[]
            errs = Float64[]
            for q in qs
                rowsMq = [r for r in rowsM if isapprox(r.q, q; rtol = 0.0, atol = 1e-14)]
                isempty(rowsMq) && continue
                best = minimum(r.max_error for r in rowsMq)
                push!(qvals, Float64(q))
                push!(errs, Float64(best))
            end
            Base.invokelatest(
                Plots.plot!,
                fig,
                qvals,
                errs;
                marker = :circle,
                label = @sprintf("M=%d", M),
            )
        end

        suffix = _BRUNNER_QEXTREME[] ? "_qextreme" : ""
        pdf_path = joinpath(fig_dir, "brunner_fast_error_vs_q_by_M$(suffix).pdf")
        png_path = joinpath(fig_dir, "brunner_fast_error_vs_q_by_M$(suffix).png")
        Base.invokelatest(Plots.savefig, fig, pdf_path)
        Base.invokelatest(Plots.savefig, fig, png_path)
        println("Saved figure: ", pdf_path)
        println("Saved figure: ", png_path)
        return true
    catch err
        @warn "Brunner error-vs-q plotting skipped." exception = (err, catch_backtrace())
        return false
    end
end

function plot_brunner_collapsed_fraction_vs_q(brunner_rows; fig_dir = FIGURES_DIR)
    valid_rows = filter(is_valid_brunner_row, brunner_rows)
    isempty(valid_rows) && return false

    try
        @eval import Plots
        Base.invokelatest(Plots.gr)

        Ms = sort(unique(r.M for r in valid_rows))
        fig = Base.invokelatest(
            Plots.plot,
            Float64[],
            Float64[];
            marker = :circle,
            linewidth = 1.7,
            markersize = 4.0,
            framestyle = :box,
            grid = false,
            legend = :outerright,
            legendfontsize = 7,
            xlabel = "q",
            ylabel = "collapsed x-node fraction",
            size = (880, 520),
            dpi = 300,
            label = false,
        )

        for M in Ms
            rowsM = [r for r in valid_rows if r.M == M]
            qs = sort(unique(r.q for r in rowsM))
            qvals = Float64[]
            fracs = Float64[]
            for q in qs
                rowsMq = [r for r in rowsM if isapprox(r.q, q; rtol = 0.0, atol = 1e-14)]
                isempty(rowsMq) && continue
                push!(qvals, Float64(q))
                push!(fracs, maximum(Float64(r.collapsed_x_fraction) for r in rowsMq))
            end
            Base.invokelatest(
                Plots.plot!,
                fig,
                qvals,
                fracs;
                marker = :circle,
                label = @sprintf("M=%d", M),
            )
        end

        pdf_path = joinpath(fig_dir, "brunner_fast_collapsed_x_fraction_vs_q.pdf")
        png_path = joinpath(fig_dir, "brunner_fast_collapsed_x_fraction_vs_q.png")
        Base.invokelatest(Plots.savefig, fig, pdf_path)
        Base.invokelatest(Plots.savefig, fig, png_path)
        println("Saved figure: ", pdf_path)
        println("Saved figure: ", png_path)
        return true
    catch err
        @warn "Brunner collapsed-fraction plotting skipped." exception = (err, catch_backtrace())
        return false
    end
end

function sanity_checks(brunner_rows, cheb_rows, rhs_stability)
    valid_brunner_rows = filter(is_valid_brunner_row, brunner_rows)
    all_rows = vcat(valid_brunner_rows, cheb_rows)
    all(isfinite(r.max_error) && isfinite(r.rms_error) for r in all_rows) ||
        error("Non-finite error detected.")
    all(r.total_time > 0.0 && isfinite(r.total_time) for r in all_rows) ||
        error("Non-positive or non-finite timing detected.")

    all(isfinite, vcat(values(rhs_stability.values)...)) ||
        error("Non-finite RHS stability values detected.")

    g_left = g_manufactured_rhs(-1.0)
    u_left = u_exact(-1.0)
    if !(abs(g_left - u_left) <= 1e-12)
        @warn "manufactured RHS at x=-1 is not close to u_exact(-1)." RHS_MODE g_left u_left
    end

    combos = sort(unique((r.m, r.q) for r in valid_brunner_rows))
    for (m, q) in combos
        rows = brunner_group_rows(brunner_rows, m, q)
        berr = [r.max_error for r in rows]
        for i in 1:(length(berr) - 1)
            if berr[i + 1] > berr[i] * (1.0 + 1e-10)
                @warn "Brunner max_error is locally non-monotone." m q index = i previous = berr[i] next = berr[i + 1]
            end
        end
        if last(berr) >= first(berr)
            @warn "Brunner max_error did not decrease overall." m q first_error = first(berr) last_error = last(berr)
        end
    end

    cheb_best = minimum(r.max_error for r in cheb_rows)
    if cheb_best > 1e-8
        @warn "Chebyshev-Duffy did not reach 1e-8 max error; check RHS projection, DE h, or beta." cheb_best
    end

    return true
end

function main()
    RHS_MODE in ("ex7", "operator_consistent", "old_DE") ||
        error("Unsupported RHS_MODE=$(RHS_MODE). Use ex7, operator_consistent, or old_DE.")
    BRUNNER_IMPL in ("old", "fast", "both") ||
        error("Unsupported BRUNNER_IMPL=$(BRUNNER_IMPL). Use old, fast, or both.")
    brunner_impl = active_brunner_impl()

    brunner_Ms = env_limited_values(
        DEFAULT_BRUNNER_MS,
        "BRUNNER_MAX_M";
        candidates = BRUNNER_MS_CANDIDATES,
    )
    cheb_Ns = env_limited_values(DEFAULT_CHEB_NS, "CHEB_MAX_N")
    brunner_ms = parse_env_int_list("BRUNNER_MS", DEFAULT_BRUNNER_M_VALUES)
    brunner_qs = parse_env_float_list("BRUNNER_QS", DEFAULT_BRUNNER_Q_VALUES)
    all(m in DEFAULT_BRUNNER_M_VALUES for m in brunner_ms) ||
        error("BRUNNER_MS may only contain supported local orders $(DEFAULT_BRUNNER_M_VALUES).")
    all(q > 0.0 for q in brunner_qs) ||
        error("BRUNNER_QS values must be positive.")
    _BRUNNER_QEXTREME[] = any(q -> q > 6.0 + 1e-14, brunner_qs)
    _BRUNNER_QEXTENDED[] = !_BRUNNER_QEXTREME[] &&
                            (any(q -> q > 4.0 + 1e-14, brunner_qs) || haskey(ENV, "BRUNNER_QS"))
    RHS_EX7_TARGET_DEGREE[] = max(RHS_EX7_TARGET_DEGREE[], maximum(cheb_Ns))

    xgrid = collect(range(-1.0, 1.0; length = DEFAULT_XGRID_N))
    exact_vals = u_exact.(xgrid)

    println("============================================================")
    println("Manufactured weakly singular Volterra test")
    @printf("mu = %.16g, lambda = %.16g\n", MU, LAMBDA)
    println("exact u(x) = 1 + 0.2 sqrt(x+1) + 0.1(x+1) + 0.05(x+1)^(3/2)")
    println("K(x,y) = 1 / ((x+1)^2 + (y-1)^2)")
    println("RHS_MODE = ", RHS_MODE)
    println("BRUNNER_IMPL = ", BRUNNER_IMPL)
    if brunner_impl === :fast
        println("Brunner implementation: fast analytic product weights")
        println("product weights: analytic moment formula")
    else
        println("Brunner implementation: old quadrature product weights")
    end
    @printf("Duffy kernel coefficient size = %d x %d\n", KERNEL_C_ROWS, KERNEL_C_COLS)
    if RHS_MODE == "old_DE"
        println("RHS generated by compare-script tanh-sinh DE quadrature")
        println("DE truncation: endpoint distances to +/-1 below eps(Float64)")
        de_info = tanhsinh_truncation_info(h = RHS_DE_H)
        @printf(
            "RHS_DE_H = %.6g, K = %d, endpoint distances = (%.3e, %.3e)\n",
            RHS_DE_H,
            de_info.K,
            de_info.left_distance,
            de_info.right_distance,
        )
    elseif RHS_MODE == "ex7"
        println("RHS generated by ex7-style BigFloat tanh-sinh and JFP-Chebyshev Lobatto projection")
        @printf("RHS BigFloat precision = %d bits\n", RHS_PRECISION)
        @printf("RHS tanh-sinh h = %s, atol = %s, max_terms = %d\n",
            TS_H_STR,
            TS_ATOL_STR,
            TS_MAX_TERMS,
        )
        @printf("RHS target coefficient degree = %d\n", RHS_EX7_TARGET_DEGREE[])
    else
        println("RHS generated by operator consistency: g_coeff = (I + lambda*B_N) * u_exact_coeff")
    end
    println("Brunner m values = ", brunner_ms)
    println("Brunner q values = ", brunner_qs)
    println("Brunner Ms = ", brunner_Ms)
    println("Chebyshev-Duffy Ns = ", cheb_Ns)
    println("============================================================")

    if RHS_MODE == "operator_consistent"
        rows = run_operator_consistent_diagnostic(cheb_Ns, xgrid, exact_vals)
        csv_path = joinpath(DATA_DIR, "chebduffy_operator_consistent_diagnostic.csv")
        write_csv(
            csv_path,
            ["N", "dof", "total_time", "max_error", "rms_error", "coeff_error", "cond_est"],
            rows,
        )
        plotted = plot_operator_consistent_diagnostic(rows)

        println()
        println("Saved operator-consistent diagnostic CSV to:")
        println("  ", csv_path)
        if plotted
            println("Saved operator-consistent diagnostic figure to: ", FIGURES_DIR)
        end
        println("============================================================")
        return (operator_consistent = rows,)
    end

    rhs_stability = check_rhs_stability()

    weight_check = nothing
    fast_old_check = NamedTuple[]
    ratio_rows = NamedTuple[]
    ratio_plotted = false
    if brunner_impl === :fast
        weight_check = check_product_weight_fast_against_old()
        fast_old_check = check_brunner_fast_against_old()
        if BRUNNER_IMPL == "both"
            ratio_rows = run_brunner_old_fast_time_ratio()
            ratio_csv = joinpath(DATA_DIR, "brunner_old_vs_fast_time_ratio.csv")
            write_csv(
                ratio_csv,
                ["M", "dof", "m", "q", "old_time", "fast_time", "speedup", "max_abs_diff"],
                ratio_rows,
            )
            ratio_plotted = plot_brunner_old_fast_time_ratio(ratio_rows)
            println("Saved Brunner old-vs-fast timing CSV to:")
            println("  ", ratio_csv)
        end
    end

    brunner_rows = run_brunner_parameter_sweep(
        brunner_Ms,
        brunner_ms,
        brunner_qs,
        xgrid,
        exact_vals;
        impl = brunner_impl,
    )
    cheb_rows = run_chebduffy_benchmark(cheb_Ns, xgrid, exact_vals)
    combined = combined_rows(brunner_rows, cheb_rows)
    envelope_rows = brunner_best_envelope(brunner_rows)
    rhs_tag = rhs_output_tag()
    brunner_tag = brunner_output_tag()

    brunner_csv = joinpath(DATA_DIR, "brunner_mq_sweep_manufactured_$(brunner_tag).csv")
    cheb_csv = joinpath(DATA_DIR, "chebduffy_manufactured_$(rhs_tag)_results.csv")
    compare_csv = joinpath(DATA_DIR, compare_output_filename(rhs_tag))
    envelope_csv = joinpath(DATA_DIR, "brunner_best_envelope_$(brunner_tag).csv")

    write_csv(
        brunner_csv,
        ["method", "M", "dof", "m", "q", "mu", "lambda", "nq", "impl", "hmin", "hmax", "mesh_ratio", "mesh_valid", "collapsed_x_count", "collapsed_x_fraction", "min_positive_s", "min_positive_xplus1", "numerically_extreme", "solve_time", "total_time", "max_error", "rms_error"],
        brunner_rows,
    )
    write_csv(
        cheb_csv,
        ["method", "N", "dof", "mu", "lambda", "beta", "construction_time", "solve_time", "total_time", "max_error", "rms_error"],
        cheb_rows,
    )
    write_csv(
        compare_csv,
        ["method", "dof", "total_time", "max_error", "rms_error", "N_or_M", "extra", "hmin", "hmax", "mesh_ratio", "mesh_valid", "collapsed_x_count", "collapsed_x_fraction", "min_positive_s", "min_positive_xplus1", "numerically_extreme"],
        combined,
    )
    write_csv(
        envelope_csv,
        ["dof", "total_time", "max_error", "m", "q", "M", "hmin", "hmax", "mesh_ratio", "mesh_valid", "collapsed_x_count", "collapsed_x_fraction", "min_positive_s", "min_positive_xplus1", "numerically_extreme"],
        envelope_rows,
    )

    sanity_checks(brunner_rows, cheb_rows, rhs_stability)
    plotted = plot_comparison(brunner_rows, cheb_rows, envelope_rows)
    q_plot = plot_brunner_error_vs_q_by_M(brunner_rows)
    collapsed_plot = plot_brunner_collapsed_fraction_vs_q(brunner_rows)

    valid_brunner_rows = filter(is_valid_brunner_row, brunner_rows)
    best_brunner = valid_brunner_rows[argmin([r.max_error for r in valid_brunner_rows])]
    best_envelope = envelope_rows[argmin([r.max_error for r in envelope_rows])]
    best_cheb = cheb_rows[argmin([r.max_error for r in cheb_rows])]

    println()
    @printf(
        "Fast Brunner lowest max_error: impl=%s, m=%d, q=%.1f, M=%d, dof=%d, total=%.6e, max_error=%.6e\n",
        best_brunner.impl,
        best_brunner.m,
        best_brunner.q,
        best_brunner.M,
        best_brunner.dof,
        best_brunner.total_time,
        best_brunner.max_error,
    )
    @printf(
        "Fast Brunner best mesh diagnostics: collapsed_x_count=%d, collapsed_x_fraction=%.6e, min_positive_s=%.3e, min_positive_xplus1=%.3e, numerically_extreme=%s\n",
        best_brunner.collapsed_x_count,
        best_brunner.collapsed_x_fraction,
        best_brunner.min_positive_s,
        best_brunner.min_positive_xplus1,
        string(best_brunner.numerically_extreme),
    )
    @printf(
        "Brunner best envelope lowest error: m=%d, q=%.1f, M=%d, dof=%d, total=%.6e, max_error=%.6e\n",
        best_envelope.m,
        best_envelope.q,
        best_envelope.M,
        best_envelope.dof,
        best_envelope.total_time,
        best_envelope.max_error,
    )
    @printf(
        "Chebyshev-Duffy best result: N=%d, dof=%d, total=%.6e, max_error=%.6e\n",
        best_cheb.N,
        best_cheb.dof,
        best_cheb.total_time,
        best_cheb.max_error,
    )
    if !isempty(ratio_rows)
        speedups = [r.speedup for r in ratio_rows]
        @printf(
            "Brunner old/fast timing speedup: min=%.3f, max=%.3f, last=%.3f\n",
            minimum(speedups),
            maximum(speedups),
            last(speedups),
        )
    end
    if best_envelope.max_error > best_cheb.max_error && best_envelope.total_time > best_cheb.total_time
        println("Brunner best envelope remains slower and less accurate than Chebyshev-Duffy on this test.")
    end
    for qcheck in (4.0, 5.0, 6.0, 7.0, 8.0, 10.0)
        rowsq = [r for r in valid_brunner_rows if isapprox(r.q, qcheck; rtol = 0.0, atol = 1e-14)]
        if !isempty(rowsq)
            bestq = rowsq[argmin([r.max_error for r in rowsq])]
            @printf(
                "Best Brunner at q=%.1f: m=%d, M=%d, dof=%d, total=%.6e, max_error=%.6e, hmin=%.3e, ratio=%.3e, collapsed_fraction=%.3e, numerically_extreme=%s\n",
                qcheck,
                bestq.m,
                bestq.M,
                bestq.dof,
                bestq.total_time,
                bestq.max_error,
                bestq.hmin,
                bestq.mesh_ratio,
                bestq.collapsed_x_fraction,
                string(bestq.numerically_extreme),
            )
        end
    end
    invalid_count = count(r -> hasfield(typeof(r), :mesh_valid) && !r.mesh_valid, brunner_rows)
    invalid_count > 0 && @printf("Skipped %d Brunner parameter groups because the graded mesh was invalid.\n", invalid_count)
    println("Recommended paper figure: ", joinpath(FIGURES_DIR, "compare_$(brunner_tag)_time_vs_error_with_envelope.pdf"))
    if q_plot
        suffix = _BRUNNER_QEXTREME[] ? "_qextreme" : ""
        println("Saved Brunner q-scan figure: ", joinpath(FIGURES_DIR, "brunner_fast_error_vs_q_by_M$(suffix).pdf"))
    end
    if collapsed_plot
        println("Saved Brunner collapsed-node diagnostic figure: ", joinpath(FIGURES_DIR, "brunner_fast_collapsed_x_fraction_vs_q.pdf"))
    end

    println()
    println("Saved CSV files to:")
    println("  ", brunner_csv)
    println("  ", cheb_csv)
    println("  ", compare_csv)
    println("  ", envelope_csv)
    if plotted
        println("Saved figures to: ", FIGURES_DIR)
    else
        println("Plots.jl unavailable or failed; CSV files were still written.")
    end
    if ratio_plotted
        println("Saved Brunner old-vs-fast timing figure to: ", FIGURES_DIR)
    end
    println("============================================================")

    return (
        brunner = brunner_rows,
        chebduffy = cheb_rows,
        combined = combined,
        envelope = envelope_rows,
        rhs_stability = rhs_stability,
        weight_check = weight_check,
        fast_old_check = fast_old_check,
        old_fast_ratio = ratio_rows,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
