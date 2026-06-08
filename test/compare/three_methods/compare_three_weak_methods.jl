# Three-way benchmark for weakly singular Volterra second-kind equations.
#
# Run from the project root:
#   julia --project=. test/compare/three_methods/compare_three_weak_methods.jl
#   julia --project=. test/compare/three_methods/plot_three_weak_methods.jl
#
# Useful environment variables:
#   EXAMPLE=oscillatory_sqrt
#   N_LIST="32,48,64,96,128,192,256,384,512,768,1024"
#   BRUNNER_N_LIST="64,96,128,192,256,384,512"
#   MU=0.5
#   LAMBDA=0.5
#   RHS_QUAD_ORDER=128
#   CHECK_GRID_SIZE=8001
#   RUN_OURS=1
#   RUN_BRUNNER=1
#   RUN_MUNTZ=1
#   RUN_CHENTANG=0
#   RUN_DIRECT_JACOBI=0
#   CHENTANG_M=2
#   MUNTZ_LAMBDA=0.5

using FastGaussQuadrature
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

const EXAMPLE = lowercase(strip(get(ENV, "EXAMPLE", "oscillatory_sqrt")))
EXAMPLE in ("oscillatory_sqrt", "low_order_sqrt_poly") ||
    error("Unsupported EXAMPLE=$(EXAMPLE). Use oscillatory_sqrt or low_order_sqrt_poly.")

const MU = parse(Float64, get(ENV, "MU", "0.5"))
const LAMBDA = parse(Float64, get(ENV, "LAMBDA", "0.5"))
const BETA = parse(Float64, get(ENV, "BETA", "0.5"))
const SIGMA = 0.0
const LAMBDA_X = LAMBDA * 2.0^(MU - 1.0)
const KERNEL_EPS = parse(Float64, get(ENV, "KERNEL_EPS", "1e-14"))

const DEFAULT_RHS_QUAD_ORDER = EXAMPLE == "oscillatory_sqrt" ? "128" : "64"
const RHS_QUAD_ORDER = parse(Int, get(ENV, "RHS_QUAD_ORDER", DEFAULT_RHS_QUAD_ORDER))
const RHS_PRECISION = parse(Int, get(ENV, "RHS_PRECISION", string(max(256, 4 * RHS_QUAD_ORDER))))
const RHS_TS_H = @sprintf("%.17g", inv(Float64(RHS_QUAD_ORDER)))
const RHS_TS_H_CHECK = @sprintf("%.17g", inv(2.0 * Float64(RHS_QUAD_ORDER)))
const RHS_TS_ENDPOINT_EPS = get(ENV, "RHS_TS_ENDPOINT_EPS", string(eps(Float64)))
const RHS_TS_MAXK = parse(Int, get(ENV, "RHS_TS_MAXK", "100000"))
const RHS_SAMPLE_DEGREE_ENV = get(ENV, "RHS_SAMPLE_DEGREE", "")

const DEFAULT_CHECK_GRID_SIZE = EXAMPLE == "oscillatory_sqrt" ? "8001" : "2001"
const CHECK_GRID_SIZE = parse(Int, get(ENV, "CHECK_GRID_SIZE", DEFAULT_CHECK_GRID_SIZE))
const RESIDUAL_GRID_SIZE = parse(Int, get(ENV, "RESIDUAL_GRID_SIZE", string(min(CHECK_GRID_SIZE, 401))))
const RESIDUAL_QUAD_ORDER = parse(Int, get(ENV, "RESIDUAL_QUAD_ORDER", "96"))

const DEFAULT_N_LIST = EXAMPLE == "oscillatory_sqrt" ?
    [32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024] :
    [16, 24, 32, 48, 64, 96, 128, 192, 256]
const DEFAULT_BRUNNER_N_LIST = EXAMPLE == "oscillatory_sqrt" ?
    [64, 96, 128, 192, 256, 384, 512] :
    DEFAULT_N_LIST
const BRUNNER_LOCAL_M = parse(Int, get(ENV, "BRUNNER_LOCAL_M", "4"))
const BRUNNER_Q = parse(Float64, get(ENV, "BRUNNER_Q", "2.0"))
const CHENTANG_QUAD_PAD = parse(Int, get(ENV, "CHENTANG_QUAD_PAD", "16"))
const CHENTANG_M = parse(Int, get(ENV, "CHENTANG_M", "2"))
const CHENTANG_T = parse(Float64, get(ENV, "CHENTANG_T", "1.0"))
const MUNTZ_LAMBDA = parse(Float64, get(ENV, "MUNTZ_LAMBDA", string(1.0 - MU)))
const MUNTZ_ALPHA = parse(Float64, get(ENV, "MUNTZ_ALPHA", "-0.5"))
const MUNTZ_BETA = parse(Float64, get(ENV, "MUNTZ_BETA", "-0.5"))
const MUNTZ_QUAD_ORDER = parse(Int, get(ENV, "MUNTZ_QUAD_ORDER", string(RHS_QUAD_ORDER)))
const KERNEL_C_ROWS = parse(Int, get(ENV, "KERNEL_C_ROWS", "61"))
const KERNEL_C_COLS = parse(Int, get(ENV, "KERNEL_C_COLS", "41"))
const KERNEL_I = KERNEL_C_ROWS - 1
const KERNEL_J = KERNEL_C_COLS - 1

const _RHS_VALUE_CACHE = Dict{Tuple{Float64,String},Float64}()
const _TS_CACHE = Dict{String,Any}()
const _CT_KNOWN_INTEGRAL_CACHE = Dict{Tuple{Float64,String,Int},Float64}()
const _CT_F_CACHE = Dict{Tuple{Float64,String,Int},Float64}()

output_stem(base::AbstractString) = EXAMPLE == "low_order_sqrt_poly" ? base : base * "_" * EXAMPLE
results_csv_path() = joinpath(DATA_DIR, output_stem("three_weak_methods") * "_results.csv")

# ============================================================
# Environment parsing
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

function parse_bool(raw)
    s = lowercase(strip(string(raw)))
    s in ("1", "true", "yes", "on") && return true
    s in ("0", "false", "no", "off") && return false
    error("Invalid boolean value: $(raw)")
end

function run_flag(env_name::AbstractString, default::AbstractString = "1")
    raw = get(ENV, env_name, default)
    return parse_bool(raw)
end

# ============================================================
# Problem definition on [0, 1]
# ============================================================

function u_exact_t(t)
    tf = max(Float64(t), 0.0)
    if EXAMPLE == "oscillatory_sqrt"
        tf <= 0.0 && return 0.0
        return sqrt(2.0 * tf) * sin(100.0 * (2.0 * tf - 1.0))
    else
        return 1.0 + 0.2 * sqrt(tf) + 0.1 * tf + 0.05 * tf^(1.5)
    end
end

function u_exact_t_big(t::BigFloat)
    if EXAMPLE == "oscillatory_sqrt"
        t <= 0 && return zero(BigFloat)
        return sqrt(BigFloat(2) * t) *
               sin(BigFloat(100) * (BigFloat(2) * t - one(BigFloat)))
    else
        t <= 0 && return one(BigFloat)
        return one(BigFloat) + BigFloat("0.2") * sqrt(t) +
               BigFloat("0.1") * t + BigFloat("0.05") * t^(BigFloat(3) / BigFloat(2))
    end
end

function u_exact_x(x)
    xf = Float64(x)
    if EXAMPLE == "oscillatory_sqrt"
        xf <= -1.0 && return 0.0
        return sqrt(max(1.0 + xf, 0.0)) * sin(100.0 * xf)
    else
        return u_exact_t((xf + 1.0) / 2.0)
    end
end

exact_description() = EXAMPLE == "oscillatory_sqrt" ?
    "u_exact_x(x) = sqrt(1+x) * sin(100x), u_exact_t(t)=sqrt(2t)*sin(100*(2t-1))" :
    "u_exact(t) = 1 + 0.2sqrt(t) + 0.1t + 0.05t^(3/2)"

Kts(t, s) = inv(4.0 * Float64(t)^2 + 4.0 * (Float64(s) - 1.0)^2 + KERNEL_EPS)

function Kts_big(t::BigFloat, s::BigFloat)
    return inv(BigFloat(4) * t^2 + BigFloat(4) * (s - one(BigFloat))^2 + BigFloat(KERNEL_EPS))
end

Kx_mapped(x, y) = Kts((Float64(x) + 1.0) / 2.0, (Float64(y) + 1.0) / 2.0)

# ============================================================
# BigFloat tanh-sinh RHS generation
# ============================================================

function tanh_sinh_nodes_weights_big(hstr::AbstractString)
    cached = get(_TS_CACHE, hstr, nothing)
    cached !== nothing && return cached
    data = setprecision(RHS_PRECISION) do
        h = BigFloat(hstr)
        endpoint_eps = BigFloat(RHS_TS_ENDPOINT_EPS)
        pi2 = big(pi) / BigFloat(2)
        K = 0
        right_dist = one(BigFloat)
        left_dist = one(BigFloat)
        while true
            tau = BigFloat(K) * h
            xip = tanh(pi2 * sinh(tau))
            xim = tanh(pi2 * sinh(-tau))
            right_dist = one(BigFloat) - xip
            left_dist = xim + one(BigFloat)
            if right_dist <= endpoint_eps && left_dist <= endpoint_eps
                break
            end
            K += 1
            K > RHS_TS_MAXK && error("tanh-sinh endpoint truncation failed before K=$(RHS_TS_MAXK).")
        end

        nodes = BigFloat[]
        weights = BigFloat[]
        for k in -K:K
            tau = BigFloat(k) * h
            a = pi2 * sinh(tau)
            xi = tanh(a)
            r = (one(BigFloat) + xi) / BigFloat(2)
            (r <= 0 || r >= 1) && continue
            drdt = BigFloat("0.5") * pi2 * cosh(tau) / cosh(a)^2
            push!(nodes, r)
            push!(weights, h * drdt)
        end
        (nodes = nodes, weights = weights, K = K,
            right_dist = Float64(right_dist), left_dist = Float64(left_dist))
    end
    _TS_CACHE[hstr] = data
    return data
end

function tanh_sinh_integrate01_big(f; hstr = RHS_TS_H)
    data = tanh_sinh_nodes_weights_big(hstr)
    val = setprecision(RHS_PRECISION) do
        total = zero(BigFloat)
        @inbounds for i in eachindex(data.nodes)
            total += f(data.nodes[i]) * data.weights[i]
        end
        total
    end
    isfinite(Float64(val)) || error("Non-finite tanh-sinh integral.")
    return val
end

function volterra_integral_rhs_big(t::BigFloat; mu = MU, hstr = RHS_TS_H)
    t <= 0 && return zero(BigFloat)
    setprecision(RHS_PRECISION) do
        mub = BigFloat(mu)
        alpha = inv(one(BigFloat) - mub)
        factor = t^(one(BigFloat) - mub) / (one(BigFloat) - mub)
        integral = tanh_sinh_integrate01_big(hstr = hstr) do r
            s = t * (one(BigFloat) - r^alpha)
            Kts_big(t, s) * u_exact_t_big(s)
        end
        return factor * integral
    end
end

function g_manufactured_t(t; hstr = RHS_TS_H)
    tf = clamp(Float64(t), 0.0, 1.0)
    key = (tf, hstr)
    cached = get(_RHS_VALUE_CACHE, key, nothing)
    cached !== nothing && return cached
    val = setprecision(RHS_PRECISION) do
        tb = BigFloat(tf)
        u_exact_t_big(tb) + BigFloat(LAMBDA) * volterra_integral_rhs_big(tb; mu = MU, hstr = hstr)
    end
    out = Float64(val)
    isfinite(out) || error("Non-finite RHS at t=$(tf).")
    _RHS_VALUE_CACHE[key] = out
    return out
end

function check_rhs_stability()
    pts = collect(range(0.0, 1.0; length = 33))
    vals1 = [g_manufactured_t(t; hstr = RHS_TS_H) for t in pts]
    vals2 = [g_manufactured_t(t; hstr = RHS_TS_H_CHECK) for t in pts]
    diff = maximum(abs.(vals1 .- vals2))
    ts1 = tanh_sinh_nodes_weights_big(RHS_TS_H)
    ts2 = tanh_sinh_nodes_weights_big(RHS_TS_H_CHECK)
    println("RHS stability check:")
    @printf("  h=%s K=%d endpoint distances=(%.3e, %.3e)\n",
        RHS_TS_H, ts1.K, ts1.left_dist, ts1.right_dist)
    @printf("  h=%s K=%d endpoint distances=(%.3e, %.3e)\n",
        RHS_TS_H_CHECK, ts2.K, ts2.left_dist, ts2.right_dist)
    @printf("  max |g_h - g_h/2| on 33 points = %.6e\n", diff)
    diff > 1e-12 && @warn "RHS stability difference is larger than 1e-12." diff
    return (diff = diff, K = ts1.K, K_check = ts2.K,
        left_dist = ts1.left_dist, right_dist = ts1.right_dist)
end

# ============================================================
# Chebyshev/JFP utilities
# ============================================================

phi_beta_t(t::Real, beta::Real = BETA) = 2.0 * clamp(Float64(t), 0.0, 1.0)^Float64(beta) - 1.0
invphi_t(z::Real, beta::Real = BETA) = ((1.0 + Float64(z)) / 2.0)^(1.0 / Float64(beta))

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

eval_jfp_t(c::AbstractVector{<:Real}, t::Real; beta::Real = BETA) =
    cheb_eval(c, phi_beta_t(t, beta))

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

function rhs_coefficients(maxN::Int)
    sample_floor = EXAMPLE == "oscillatory_sqrt" ? 4096 : 1024
    sample_degree = isempty(RHS_SAMPLE_DEGREE_ENV) ? max(sample_floor, 4 * maxN) :
                    parse(Int, RHS_SAMPLE_DEGREE_ENV)
    sample_degree = max(sample_degree, maxN)
    println()
    @printf("Generating common RHS coefficients: degree=%d, sample_degree=%d\n", maxN, sample_degree)
    rhs_stat = @timed begin
        vals = Vector{Float64}(undef, sample_degree + 1)
        @inbounds for j in 0:sample_degree
            z = cos(pi * j / sample_degree)
            vals[j + 1] = g_manufactured_t(invphi_t(z, BETA); hstr = RHS_TS_H)
        end
        vals
    end
    proj_stat = @timed cheb_project_lobatto_from_values(rhs_stat.value, maxN)
    coeffs = proj_stat.value

    check_degree = max(2 * sample_degree, maxN)
    check_diff = NaN
    if check_degree != sample_degree
        check_vals = Vector{Float64}(undef, check_degree + 1)
        @inbounds for j in 0:check_degree
            z = cos(pi * j / check_degree)
            check_vals[j + 1] = g_manufactured_t(invphi_t(z, BETA); hstr = RHS_TS_H)
        end
        check_coeffs = cheb_project_lobatto_from_values(check_vals, maxN)
        check_diff = maximum(abs.(coeffs .- check_coeffs))
    end

    tail_count = min(20, length(coeffs))
    info = (
        rhs_time = rhs_stat.time,
        projection_time = proj_stat.time,
        sample_degree = sample_degree,
        check_degree = check_degree,
        check_diff_inf = check_diff,
        tail_inf = maximum(abs, @view coeffs[(end - tail_count + 1):end]),
    )
    @printf("  BigFloat RHS point time = %.6e s\n", info.rhs_time)
    @printf("  Lobatto projection time = %.6e s\n", info.projection_time)
    @printf("  RHS coeff tail last %d = %.3e\n", tail_count, info.tail_inf)
    isfinite(check_diff) && @printf("  RHS coeff sample/check diff = %.3e\n", check_diff)
    return coeffs, info
end

# ============================================================
# Error and residual utilities
# ============================================================

function evaluate_error(f, tgrid, exact_vals)
    linf = 0.0
    sumsq = 0.0
    @inbounds for i in eachindex(tgrid, exact_vals)
        d = f(tgrid[i]) - exact_vals[i]
        linf = max(linf, abs(d))
        sumsq += d * d
    end
    return linf, sqrt(sumsq / length(tgrid))
end

function residual_linf(f, g_coeffs, tgrid; qorder = RESIDUAL_QUAD_ORDER)
    theta, w = gaussjacobi(qorder, -MU, 0.0)
    maxr = 0.0
    @inbounds for t in tgrid
        tf = Float64(t)
        integral = 0.0
        if tf > 0.0
            factor = (tf / 2.0)^(1.0 - MU)
            for k in eachindex(theta)
                s = tf * (1.0 + theta[k]) / 2.0
                integral += w[k] * Kts(tf, s) * f(s)
            end
            integral *= factor
        end
        r = f(tf) + LAMBDA * integral - eval_jfp_t(g_coeffs, tf)
        maxr = max(maxr, abs(r))
    end
    return maxr
end

# ============================================================
# Our coefficient-space method
# ============================================================

dense_matrix(V) = V isa VoltOpt.PackedLowerHessenberg ? VoltOpt.packed_to_dense(V) : Matrix(V)

function build_kernel_coefficients()
    stat = @timed begin
        values = WeakCheb.sample_duffy_kernel(Kx_mapped, KERNEL_I, KERNEL_J, BETA)
        WeakCheb.cheb_coeffs_lobatto_2d(values)
    end
    return stat.value, stat.time
end

function solve_our_coeffspace(N, C, kernel_time, rhs_coeffs, tgrid, exact_vals, rgrid)
    rhsN = rhs_coeffs[1:(N + 1)]
    build_stat = @timed begin
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
        A = Matrix{Float64}(LinearAlgebra.I, N + 1, N + 1)
        A .+= LAMBDA_X .* dense_matrix(V)
        A
    end
    A = build_stat.value
    solve_stat = @timed A \ rhsN
    coeffs = solve_stat.value
    sol = t -> eval_jfp_t(coeffs, t)
    linf, l2 = evaluate_error(sol, tgrid, exact_vals)
    res = residual_linf(sol, rhs_coeffs, rgrid)
    return (
        build_time = kernel_time + build_stat.time,
        solve_time = solve_stat.time,
        total_time = kernel_time + build_stat.time + solve_stat.time,
        linf_error = linf,
        l2_grid_error = l2,
        residual_linf = res,
        notes = @sprintf("lambda_x=%.17e; beta=%.2f; kernel=%dx%d", LAMBDA_X, BETA, KERNEL_C_ROWS, KERNEL_C_COLS),
    )
end

# ============================================================
# Fast Brunner graded collocation/product integration
# ============================================================

function radau_right_nodes(m::Int)
    m == 2 && return [1.0 / 3.0, 1.0]
    m == 3 && return [0.1550510257216822, 0.6449489742783178, 1.0]
    m == 4 && return [0.0885879595127042, 0.4094668644407347, 0.7876594617608471, 1.0]
    throw(ArgumentError("supported Brunner local orders are m=2,3,4."))
end

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
    m = length(c)
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
            ck = c[k]
            for p in 0:degree
                tmp[p + 1] += -ck * poly[p + 1]
                tmp[p + 2] += poly[p + 1]
            end
            degree += 1
            copyto!(poly, tmp)
            denom *= c[j] - ck
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

function product_weights_fast!(weights, moments, delta, theta_max, coeffs, mu)
    m = size(coeffs, 1)
    @inbounds for p in 0:(m - 1)
        moments[p + 1] = moment_power(delta, theta_max, p, mu)
    end
    @inbounds for j in 1:m
        acc = 0.0
        for p in 1:m
            acc += coeffs[j, p] * moments[p]
        end
        weights[j] = acc
    end
    return weights
end

graded_mesh(M::Int, q::Real) = [(n / M)^Float64(q) for n in 0:M]

function solve_small_system!(x, A, b, m)
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
    build_time::Float64
    solve_time::Float64
    dof::Int
end

function solve_brunner_graded(M::Int, rhs_coeffs; m = BRUNNER_LOCAL_M, q = BRUNNER_Q)
    cvec = radau_right_nodes(m)
    build_stat = @timed begin
        mesh = graded_mesh(M, q)
        h = diff(mesh)
        nodes = Matrix{Float64}(undef, M, m)
        rhs_nodes = Matrix{Float64}(undef, M, m)
        @inbounds for n in 1:M
            left = mesh[n]
            hn = mesh[n + 1] - left
            for i in 1:m
                nodes[n, i] = left + cvec[i] * hn
                rhs_nodes[n, i] = eval_jfp_t(rhs_coeffs, nodes[n, i])
            end
        end
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
        (mesh = mesh, h = h, hpow = h .^ (1.0 - MU), nodes = nodes,
            rhs_nodes = rhs_nodes, power_coeffs = power_coeffs, current_weights0 = current_weights0)
    end
    data = build_stat.value
    Uvals = zeros(Float64, M, m)
    A = Matrix{Float64}(undef, m, m)
    b = Vector{Float64}(undef, m)
    xloc = Vector{Float64}(undef, m)
    hist_weights = zeros(Float64, m)
    hist_moments = zeros(Float64, m)
    solve_time = @elapsed begin
        @inbounds for n in 1:M
            for i in 1:m
                ti = data.nodes[n, i]
                history = 0.0
                for l in 1:(n - 1)
                    delta = (ti - data.mesh[l]) / data.h[l]
                    product_weights_fast!(hist_weights, hist_moments, delta, 1.0, data.power_coeffs, MU)
                    scale = data.hpow[l]
                    for j in 1:m
                        history += scale * hist_weights[j] * Kts(ti, data.nodes[l, j]) * Uvals[l, j]
                    end
                end
                b[i] = data.rhs_nodes[n, i] - LAMBDA * history
                for j in 1:m
                    A[i, j] = (i == j ? 1.0 : 0.0) +
                              LAMBDA * data.hpow[n] * data.current_weights0[i, j] *
                              Kts(ti, data.nodes[n, j])
                end
            end
            solve_small_system!(xloc, A, b, m)
            for j in 1:m
                Uvals[n, j] = xloc[j]
            end
        end
    end
    return BrunnerSolution(data.mesh, cvec, Uvals, M, m, Float64(q), build_stat.time, solve_time, M * m)
end

function eval_brunner_solution(t, sol::BrunnerSolution)
    tf = clamp(Float64(t), 0.0, 1.0)
    M = length(sol.mesh) - 1
    panel = clamp(searchsortedfirst(sol.mesh, tf) - 1, 1, M)
    h = sol.mesh[panel + 1] - sol.mesh[panel]
    theta = (tf - sol.mesh[panel]) / h
    u = 0.0
    @inbounds for j in eachindex(sol.c)
        u += sol.Uvals[panel, j] * lagrange_basis(theta, j, sol.c)
    end
    return u
end

# ============================================================
# Direct Jacobi collocation baseline
# ============================================================

function direct_jacobi_nodes(n::Int)
    n < 1 && throw(ArgumentError("n must be positive."))
    n == 1 && return [0.0]
    n == 2 && return [0.0, 1.0]
    z, _ = gaussjacobi(n - 2, 0.0, -MU)
    t = (z .+ 1.0) ./ 2.0
    nodes = vcat(0.0, sort!(collect(t)), 1.0)
    return nodes
end

function barycentric_weights(nodes)
    n = length(nodes)
    logs = zeros(Float64, n)
    signs = ones(Float64, n)
    @inbounds for j in 1:n
        xj = nodes[j]
        logabs = 0.0
        signv = 1.0
        for k in 1:n
            k == j && continue
            d = xj - nodes[k]
            signv *= sign(d)
            logabs += log(abs(d))
        end
        logs[j] = -logabs
        signs[j] = signv
    end
    maxlog = maximum(logs)
    return signs .* exp.(logs .- maxlog)
end

function barycentric_basis_values!(basis, nodes, weights, x)
    n = length(nodes)
    fill!(basis, 0.0)
    xf = Float64(x)
    tol = 256eps(Float64) * max(1.0, abs(xf))
    @inbounds for j in 1:n
        if abs(xf - nodes[j]) <= tol
            basis[j] = 1.0
            return basis
        end
    end
    denom = 0.0
    @inbounds for j in 1:n
        v = weights[j] / (xf - nodes[j])
        basis[j] = v
        denom += v
    end
    @inbounds for j in 1:n
        basis[j] /= denom
    end
    return basis
end

function barycentric_eval(values, nodes, weights, x)
    xf = Float64(x)
    tol = 256eps(Float64) * max(1.0, abs(xf))
    @inbounds for j in eachindex(nodes)
        if abs(xf - nodes[j]) <= tol
            return values[j]
        end
    end
    num = 0.0
    den = 0.0
    @inbounds for j in eachindex(nodes)
        v = weights[j] / (xf - nodes[j])
        num += v * values[j]
        den += v
    end
    return num / den
end

struct DirectJacobiSolution
    nodes::Vector{Float64}
    weights::Vector{Float64}
    values::Vector{Float64}
    build_time::Float64
    solve_time::Float64
    dof::Int
    quad_order::Int
end

function solve_direct_jacobi_collocation(N::Int, rhs_coeffs)
    dof = N + 1
    qorder = max(RHS_QUAD_ORDER, dof + CHENTANG_QUAD_PAD)
    build_stat = @timed begin
        nodes = direct_jacobi_nodes(dof)
        bw = barycentric_weights(nodes)
        theta, wq = gaussjacobi(qorder, -MU, 0.0)
        A = Matrix{Float64}(LinearAlgebra.I, dof, dof)
        b = Vector{Float64}(undef, dof)
        basis = zeros(Float64, dof)
        @inbounds for i in 1:dof
            ti = nodes[i]
            b[i] = eval_jfp_t(rhs_coeffs, ti)
            ti <= 0.0 && continue
            factor = (ti / 2.0)^(1.0 - MU)
            for k in eachindex(theta)
                s = ti * (1.0 + theta[k]) / 2.0
                barycentric_basis_values!(basis, nodes, bw, s)
                coeff = LAMBDA * factor * wq[k] * Kts(ti, s)
                for j in 1:dof
                    A[i, j] += coeff * basis[j]
                end
            end
        end
        (A = A, b = b, nodes = nodes, bw = bw)
    end
    data = build_stat.value
    solve_stat = @timed data.A \ data.b
    return DirectJacobiSolution(data.nodes, data.bw, solve_stat.value, build_stat.time,
        solve_stat.time, dof, qorder)
end

eval_direct_jacobi_solution(t, sol::DirectJacobiSolution) =
    barycentric_eval(sol.values, sol.nodes, sol.weights, clamp(Float64(t), 0.0, 1.0))

# ============================================================
# Hou--Lin--Azaiez--Xu-style Muntz/fractional Jacobi collocation
# ============================================================

function shifted_gaussjacobi(n::Int, alpha::Real, beta::Real)
    r, w = gaussjacobi(n, Float64(alpha), Float64(beta))
    z = (r .+ 1.0) ./ 2.0
    wshift = w ./ 2.0^(Float64(alpha) + Float64(beta) + 1.0)
    return z, wshift
end

function muntz_nodes(N::Int; lambda_frac = MUNTZ_LAMBDA,
    alpha_c = MUNTZ_ALPHA, beta_c = MUNTZ_BETA)
    dof = N + 1
    z, _ = shifted_gaussjacobi(dof, alpha_c, beta_c)
    t = z .^ inv(Float64(lambda_frac))
    return collect(t), collect(z)
end

function muntz_basis_values!(basis, z_nodes, bary_weights_z, t, lambda_frac)
    tf = clamp(Float64(t), 0.0, 1.0)
    z = tf^Float64(lambda_frac)
    return barycentric_basis_values!(basis, z_nodes, bary_weights_z, z)
end

function eval_muntz_solution(t, values, z_nodes, bary_weights_z, lambda_frac)
    tf = clamp(Float64(t), 0.0, 1.0)
    z = tf^Float64(lambda_frac)
    return barycentric_eval(values, z_nodes, bary_weights_z, z)
end

struct MuntzSolution
    tnodes::Vector{Float64}
    znodes::Vector{Float64}
    bary_weights_z::Vector{Float64}
    values::Vector{Float64}
    lambda_frac::Float64
    alpha_c::Float64
    beta_c::Float64
    build_time::Float64
    solve_time::Float64
    dof::Int
    qorder::Int
    residual_linf::Float64
    consistency_linf::Float64
    ratio_limit_count::Int
end

function solve_muntz_collocation(N::Int, rhs_coeffs; lambda_frac = MUNTZ_LAMBDA,
    alpha_c = MUNTZ_ALPHA, beta_c = MUNTZ_BETA)
    lambda_f = Float64(lambda_frac)
    lambda_f > 0.0 || throw(ArgumentError("Muntz lambda_frac must be positive."))
    dof = N + 1
    qorder = MUNTZ_QUAD_ORDER
    ratio_limit_count = Ref(0)
    build_stat = @timed begin
        tnodes, znodes = muntz_nodes(N; lambda_frac = lambda_f, alpha_c = alpha_c, beta_c = beta_c)
        bwz = barycentric_weights(znodes)
        alpha_q = -MU
        beta_q = inv(lambda_f) - 1.0
        theta, wq = shifted_gaussjacobi(qorder, alpha_q, beta_q)
        Aop = zeros(Float64, dof, dof)
        basis = zeros(Float64, dof)
        g_nodes = [eval_jfp_t(rhs_coeffs, t) for t in tnodes]

        @inbounds for i in 1:dof
            ti = tnodes[i]
            factor = ti^(1.0 - MU) / lambda_f
            for k in eachindex(theta)
                th = theta[k]
                thpow = th^inv(lambda_f)
                s = ti * thpow
                denom = 1.0 - th
                ratio = if abs(denom) <= sqrt(eps(Float64))
                    ratio_limit_count[] += 1
                    inv(lambda_f)
                else
                    (1.0 - thpow) / denom
                end
                Kbar = factor * ratio^(-MU) * Kts(ti, s)
                muntz_basis_values!(basis, znodes, bwz, s, lambda_f)
                coeff = wq[k] * Kbar
                for j in 1:dof
                    Aop[i, j] += coeff * basis[j]
                end
            end
        end
        A = Matrix{Float64}(LinearAlgebra.I, dof, dof)
        A .+= LAMBDA .* Aop
        U_exact_nodes = u_exact_t.(tnodes)
        consistency = norm(A * U_exact_nodes .- g_nodes, Inf)
        (A = A, g_nodes = g_nodes, tnodes = tnodes, znodes = znodes, bwz = bwz,
            consistency = consistency)
    end
    data = build_stat.value
    solve_stat = @timed data.A \ data.g_nodes
    values = solve_stat.value
    residual = norm(data.A * values .- data.g_nodes, Inf)
    return MuntzSolution(data.tnodes, data.znodes, data.bwz, values, lambda_f,
        Float64(alpha_c), Float64(beta_c), build_stat.time, solve_stat.time, dof,
        qorder, residual, data.consistency, ratio_limit_count[])
end

function run_muntz_collocation(N; mu = MU, lambda_coup = LAMBDA,
    lambda_frac = MUNTZ_LAMBDA, alpha_c = MUNTZ_ALPHA, beta_c = MUNTZ_BETA,
    rhs_coeffs = nothing, tgrid = nothing, exact_vals = nothing,
    rhs_stability = NaN)
    abs(Float64(mu) - MU) <= 10eps(Float64) ||
        throw(ArgumentError("This loaded script was configured with MU=$(MU), got mu=$(mu). Set ENV[\"MU\"] before including."))
    abs(Float64(lambda_coup) - LAMBDA) <= 10eps(Float64) ||
        throw(ArgumentError("This loaded script was configured with LAMBDA=$(LAMBDA), got lambda_coup=$(lambda_coup). Set ENV[\"LAMBDA\"] before including."))
    if rhs_coeffs === nothing
        rhs_coeffs, _ = rhs_coefficients(N)
    end
    if tgrid === nothing
        tgrid = collect(range(0.0, 1.0; length = CHECK_GRID_SIZE))
    end
    if exact_vals === nothing
        exact_vals = u_exact_t.(tgrid)
    end

    sol = solve_muntz_collocation(N, rhs_coeffs;
        lambda_frac = lambda_frac, alpha_c = alpha_c, beta_c = beta_c)
    f = t -> eval_muntz_solution(t, sol.values, sol.znodes, sol.bary_weights_z, sol.lambda_frac)
    linf, l2 = evaluate_error(f, tgrid, exact_vals)
    notes = @sprintf("Muntz/fractional Jacobi collocation; lambda_frac=%.12g; alpha=%.3f; beta=%.3f; qorder=%d; ratio_limit=%d; consistency=%.3e",
        sol.lambda_frac, sol.alpha_c, sol.beta_c, sol.qorder, sol.ratio_limit_count, sol.consistency_linf)
    return (
        method = "muntz_collocation",
        N = N,
        dof = sol.dof,
        build_time = sol.build_time,
        solve_time = sol.solve_time,
        total_time = sol.build_time + sol.solve_time,
        linf_error = linf,
        l2_grid_error = l2,
        residual_linf = sol.residual_linf,
        consistency_linf = sol.consistency_linf,
        rhs_stability = rhs_stability,
        muntz_lambda = sol.lambda_frac,
        muntz_alpha = sol.alpha_c,
        muntz_beta = sol.beta_c,
        notes = notes,
    )
end

# ============================================================
# Chen--Tang transformed Jacobi spectral-collocation baseline
# ============================================================

ct_a(m::Int = CHENTANG_M) = MU + Float64(m) - 1.0
y0_value() = g_manufactured_t(0.0; hstr = RHS_TS_H)

function ytilde_exact_t(t; m::Int = CHENTANG_M, y0 = u_exact_t(0.0))
    tf = clamp(Float64(t), 0.0, 1.0)
    tf == 0.0 && return 0.0
    return tf^ct_a(m) * (u_exact_t(tf) - Float64(y0))
end

function ct_known_integral_big(t::BigFloat; m::Int = CHENTANG_M, hstr = RHS_TS_H)
    t <= 0 && return zero(BigFloat)
    setprecision(RHS_PRECISION) do
        mub = BigFloat(MU)
        alpha = inv(one(BigFloat) - mub)
        factor = t^(one(BigFloat) - mub) / (one(BigFloat) - mub)
        integral = tanh_sinh_integrate01_big(hstr = hstr) do r
            s = t * (one(BigFloat) - r^alpha)
            -BigFloat(LAMBDA) * Kts_big(t, s)
        end
        return factor * integral
    end
end

function ct_known_integral(t; m::Int = CHENTANG_M, hstr = RHS_TS_H)
    tf = clamp(Float64(t), 0.0, 1.0)
    tf <= 0.0 && return 0.0
    key = (tf, hstr, m)
    cached = get(_CT_KNOWN_INTEGRAL_CACHE, key, nothing)
    cached !== nothing && return cached
    out = Float64(setprecision(RHS_PRECISION) do
        ct_known_integral_big(BigFloat(tf); m = m, hstr = hstr)
    end)
    isfinite(out) || error("Non-finite Chen--Tang known integral at t=$(tf).")
    _CT_KNOWN_INTEGRAL_CACHE[key] = out
    return out
end

function ct_F_t(t; m::Int = CHENTANG_M, y0 = y0_value(), hstr = RHS_TS_H)
    tf = clamp(Float64(t), 0.0, 1.0)
    tf <= 0.0 && return 0.0
    key = (tf, hstr, m)
    cached = get(_CT_F_CACHE, key, nothing)
    cached !== nothing && return cached
    a = ct_a(m)
    val = tf^a * (g_manufactured_t(tf; hstr = hstr) - Float64(y0)) +
          Float64(y0) * tf^a * ct_known_integral(tf; m = m, hstr = hstr)
    isfinite(val) || error("Non-finite Chen--Tang F at t=$(tf).")
    _CT_F_CACHE[key] = val
    return val
end

ct_nodes(dof::Int) = first(gaussjacobi(dof, -MU, -MU))

function ct_khat(x, tau, m::Int, protected_count::Base.RefValue{Int})
    # Chen--Tang transformed kernel on [-1,1]:
    #
    # U(x) = f(x) + int_{-1}^x (1+tau)^(-mu) (x-tau)^(-mu)
    #        Khat(x,tau) U(tau) dtau,
    #
    # Khat(x,tau) =
    #   (1+tau)^(1-m) (1+x)^(mu+m-1) (T/2)^(1-mu)
    #   Kct(T/2*(1+x), T/2*(1+tau)),
    #
    # where Kct(t,s) = -lambda*K(t,s) for our sign convention
    # y(t) = g(t) - lambda*int (t-s)^(-mu)K(t,s)y(s)ds.
    xp1 = 1.0 + Float64(x)
    tp1 = 1.0 + Float64(tau)
    if xp1 <= 0.0 || tp1 <= 0.0
        protected_count[] += 1
        xp1 = max(xp1, realmin(Float64))
        tp1 = max(tp1, realmin(Float64))
    end
    t = 0.5 * CHENTANG_T * xp1
    s = 0.5 * CHENTANG_T * tp1
    return tp1^(1 - m) * xp1^(MU + m - 1.0) *
           (CHENTANG_T / 2.0)^(1.0 - MU) *
           (-LAMBDA * Kts(t, s))
end

function ct_K1(x, tau, m::Int, protected_count::Base.RefValue{Int})
    xp1 = 1.0 + Float64(x)
    if xp1 <= 0.0
        protected_count[] += 1
        xp1 = realmin(Float64)
    end
    return (xp1 / 2.0)^(1.0 - 2.0 * MU) * ct_khat(x, tau, m, protected_count)
end

struct ChenTangTransformedSolution
    xnodes::Vector{Float64}
    weights::Vector{Float64}
    Uvals::Vector{Float64}
    y0::Float64
    m::Int
    a::Float64
    build_time::Float64
    solve_time::Float64
    dof::Int
    system_residual::Float64
    consistency::Float64
    max_abs_exact_nodes::Float64
    protected_count::Int
end

function solve_chentang_transformed_collocation(N::Int; m::Int = CHENTANG_M)
    dof = N + 1
    y0 = y0_value()
    a = ct_a(m)
    protected_count = Ref(0)
    build_stat = @timed begin
        xnodes = collect(ct_nodes(dof))
        qnodes, qw = gaussjacobi(dof, -MU, -MU)
        bw = barycentric_weights(xnodes)
        Mmat = zeros(Float64, dof, dof)
        f_nodes = Vector{Float64}(undef, dof)
        basis = zeros(Float64, dof)
        @inbounds for i in 1:dof
            x = xnodes[i]
            t = 0.5 * CHENTANG_T * (1.0 + x)
            f_nodes[i] = ct_F_t(t; m = m, y0 = y0)
            half_xp1 = (1.0 + x) / 2.0
            for k in eachindex(qnodes)
                tau = half_xp1 * qnodes[k] + (x - 1.0) / 2.0
                barycentric_basis_values!(basis, xnodes, bw, tau)
                coeff = qw[k] * ct_K1(x, tau, m, protected_count)
                for j in 1:dof
                    Mmat[i, j] += coeff * basis[j]
                end
            end
        end
        A = Matrix{Float64}(LinearAlgebra.I, dof, dof) .- Mmat
        U_exact_nodes = [ytilde_exact_t(0.5 * CHENTANG_T * (1.0 + x); m = m, y0 = y0) for x in xnodes]
        consistency = norm(A * U_exact_nodes .- f_nodes, Inf)
        max_abs_exact_nodes = maximum(abs, U_exact_nodes)
        (A = A, f_nodes = f_nodes, Mmat = Mmat, xnodes = xnodes, bw = bw,
            consistency = consistency, max_abs_exact_nodes = max_abs_exact_nodes)
    end
    data = build_stat.value
    solve_stat = @timed data.A \ data.f_nodes
    Uvals = solve_stat.value
    system_residual = norm(data.A * Uvals .- data.f_nodes, Inf)
    return ChenTangTransformedSolution(data.xnodes, data.bw, Uvals, y0, m, a,
        build_stat.time, solve_stat.time, dof, system_residual, data.consistency,
        data.max_abs_exact_nodes, protected_count[])
end

eval_chentang_U(x, sol::ChenTangTransformedSolution) =
    barycentric_eval(sol.Uvals, sol.xnodes, sol.weights, clamp(Float64(x), -1.0, 1.0))

function eval_chentang_transformed_solution(t, sol::ChenTangTransformedSolution)
    tf = clamp(Float64(t), 0.0, 1.0)
    tf == 0.0 && return sol.y0
    x = 2.0 * tf / CHENTANG_T - 1.0
    return sol.y0 + tf^(-sol.a) * eval_chentang_U(x, sol)
end

function transformed_linf_error(sol::ChenTangTransformedSolution, tgrid)
    err = 0.0
    @inbounds for t in tgrid
        x = 2.0 * Float64(t) / CHENTANG_T - 1.0
        d = eval_chentang_U(x, sol) - ytilde_exact_t(t; m = sol.m, y0 = sol.y0)
        err = max(err, abs(d))
    end
    return err
end

function linf_error_excluding_endpoint_layer(f, tgrid, exact_vals; cutoff = 1e-8)
    err = 0.0
    any_used = false
    @inbounds for i in eachindex(tgrid, exact_vals)
        Float64(tgrid[i]) >= cutoff || continue
        any_used = true
        err = max(err, abs(f(tgrid[i]) - exact_vals[i]))
    end
    return any_used ? err : NaN
end

function run_chentang_transformed(N; mu = MU, lambda = LAMBDA, m = CHENTANG_M,
    rhs_coeffs = nothing, tgrid = nothing, exact_vals = nothing, rgrid = nothing,
    rhs_stability = NaN, endpoint_cutoff = 1e-8)
    abs(Float64(mu) - MU) <= 10eps(Float64) ||
        throw(ArgumentError("This loaded script was configured with MU=$(MU), got mu=$(mu). Set ENV[\"MU\"] before including."))
    abs(Float64(lambda) - LAMBDA) <= 10eps(Float64) ||
        throw(ArgumentError("This loaded script was configured with LAMBDA=$(LAMBDA), got lambda=$(lambda). Set ENV[\"LAMBDA\"] before including."))

    if rhs_coeffs === nothing
        rhs_coeffs, _ = rhs_coefficients(N)
    end
    if tgrid === nothing
        tgrid = collect(range(0.0, 1.0; length = CHECK_GRID_SIZE))
    end
    if exact_vals === nothing
        exact_vals = u_exact_t.(tgrid)
    end
    if rgrid === nothing
        rgrid = collect(range(0.0, 1.0; length = RESIDUAL_GRID_SIZE))
    end

    sol = solve_chentang_transformed_collocation(N; m = m)
    f = t -> eval_chentang_transformed_solution(t, sol)
    linf, l2 = evaluate_error(f, tgrid, exact_vals)
    linf_excl = linf_error_excluding_endpoint_layer(f, tgrid, exact_vals; cutoff = endpoint_cutoff)
    trans_linf = transformed_linf_error(sol, tgrid)
    residual = residual_linf(f, rhs_coeffs, rgrid)
    amplification = linf / max(trans_linf, eps(Float64))
    notes = @sprintf("Chen--Tang transformed; m=%d; a=%.6f; endpoint_cutoff=%.1e; system_residual=%.3e; consistency=%.3e; max_abs_U_exact_nodes=%.3e; protected=%d",
        sol.m, sol.a, endpoint_cutoff, sol.system_residual, sol.consistency,
        sol.max_abs_exact_nodes, sol.protected_count)

    return (
        method = "chentang_transformed_collocation",
        N = N,
        dof = N + 1,
        mu = MU,
        lambda = LAMBDA,
        m = m,
        a = sol.a,
        build_time = sol.build_time,
        solve_time = sol.solve_time,
        total_time = sol.build_time + sol.solve_time,
        linf_error = linf,
        linf_error_full_grid = linf,
        linf_error_excluding_endpoint_layer = linf_excl,
        l2_grid_error = l2,
        transformed_linf_error = trans_linf,
        residual_linf = residual,
        consistency = sol.consistency,
        max_abs_U_exact_nodes = sol.max_abs_exact_nodes,
        amplification_indicator = amplification,
        rhs_stability = rhs_stability,
        notes = notes,
    )
end

# ============================================================
# CSV and benchmark runners
# ============================================================

function csv_cell(x)
    if x isa AbstractFloat
        return isfinite(x) ? @sprintf("%.17e", x) : string(x)
    elseif x isa Integer || x isa Bool
        return string(x)
    else
        return replace(string(x), "," => ";")
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

function failure_row(method, N, dof, rhs_stability, note)
    return (
        method = method,
        N = N,
        dof = dof,
        build_time = NaN,
        solve_time = NaN,
        total_time = NaN,
        linf_error = NaN,
        l2_grid_error = NaN,
        transformed_linf_error = NaN,
        residual_linf = NaN,
        consistency_linf = NaN,
        rhs_stability = rhs_stability,
        chentang_m = "missing",
        muntz_lambda = NaN,
        muntz_alpha = NaN,
        muntz_beta = NaN,
        notes = note,
    )
end

function success_row(method, N, dof, build_time, solve_time, linf, l2, transformed_linf,
    residual, rhs_stability, chentang_m, notes; consistency_linf = NaN,
    muntz_lambda = NaN, muntz_alpha = NaN, muntz_beta = NaN)
    return (
        method = method,
        N = N,
        dof = dof,
        build_time = build_time,
        solve_time = solve_time,
        total_time = build_time + solve_time,
        linf_error = linf,
        l2_grid_error = l2,
        transformed_linf_error = transformed_linf,
        residual_linf = residual,
        consistency_linf = consistency_linf,
        rhs_stability = rhs_stability,
        chentang_m = chentang_m,
        muntz_lambda = muntz_lambda,
        muntz_alpha = muntz_alpha,
        muntz_beta = muntz_beta,
        notes = notes,
    )
end

function run_our_method!(rows, Ns, rhs_coeffs, tgrid, exact_vals, rgrid, rhs_stability)
    println()
    println("Running our_coeffspace...")
    C, kernel_time = build_kernel_coefficients()
    @printf("  kernel coefficient size = %d x %d, build_time = %.6e s\n", size(C, 1), size(C, 2), kernel_time)
    try
        warmN = first(Ns)
        solve_our_coeffspace(warmN, C, kernel_time, rhs_coeffs,
            [0.0, 0.5, 1.0], u_exact_t.([0.0, 0.5, 1.0]), [0.0, 0.5, 1.0])
    catch
    end
    for N in Ns
        try
            result = solve_our_coeffspace(N, C, kernel_time, rhs_coeffs, tgrid, exact_vals, rgrid)
            push!(rows, success_row("our_coeffspace", N, N + 1, result.build_time, result.solve_time,
                result.linf_error, result.l2_grid_error, NaN, result.residual_linf, rhs_stability,
                "missing", result.notes))
            @printf("  N=%4d, dof=%4d, time=%.6e, Linf=%.6e, residual=%.6e\n",
                N, N + 1, result.build_time + result.solve_time, result.linf_error, result.residual_linf)
        catch err
            @warn "our_coeffspace failed." N exception = (err, catch_backtrace())
            push!(rows, failure_row("our_coeffspace", N, N + 1, rhs_stability,
                "failed: $(typeof(err))"))
        end
    end
end

function run_brunner_method!(rows, Ns, rhs_coeffs, tgrid, exact_vals, rgrid, rhs_stability)
    println()
    @printf("Running brunner_graded with local m=%d, q=%.3f...\n", BRUNNER_LOCAL_M, BRUNNER_Q)
    try
        solve_brunner_graded(max(2, first(Ns)), rhs_coeffs;
            m = BRUNNER_LOCAL_M, q = BRUNNER_Q)
    catch
    end
    for N in Ns
        try
            sol = solve_brunner_graded(N, rhs_coeffs; m = BRUNNER_LOCAL_M, q = BRUNNER_Q)
            f = t -> eval_brunner_solution(t, sol)
            linf, l2 = evaluate_error(f, tgrid, exact_vals)
            res = residual_linf(f, rhs_coeffs, rgrid)
            notes = @sprintf("M=%d; local m=%d; q=%.3f; analytic product weights", N, sol.m, sol.q)
            push!(rows, success_row("brunner_graded", N, sol.dof, sol.build_time, sol.solve_time,
                linf, l2, NaN, res, rhs_stability, "missing", notes))
            @printf("  M=%4d, dof=%4d, time=%.6e, Linf=%.6e, residual=%.6e\n",
                N, sol.dof, sol.build_time + sol.solve_time, linf, res)
        catch err
            @warn "brunner_graded failed." N exception = (err, catch_backtrace())
            push!(rows, failure_row("brunner_graded", N, N * BRUNNER_LOCAL_M, rhs_stability,
                "failed: $(typeof(err))"))
        end
    end
end

function run_muntz_method!(rows, Ns, rhs_coeffs, tgrid, exact_vals, rhs_stability)
    println()
    @printf("Running muntz_collocation with lambda_frac=%.12g, alpha=%.3f, beta=%.3f...\n",
        MUNTZ_LAMBDA, MUNTZ_ALPHA, MUNTZ_BETA)
    try
        run_muntz_collocation(max(2, first(Ns));
            rhs_coeffs = rhs_coeffs,
            tgrid = [0.0, 0.5, 1.0],
            exact_vals = u_exact_t.([0.0, 0.5, 1.0]),
            rhs_stability = rhs_stability)
    catch
    end
    for N in Ns
        try
            result = run_muntz_collocation(N;
                rhs_coeffs = rhs_coeffs,
                tgrid = tgrid,
                exact_vals = exact_vals,
                rhs_stability = rhs_stability)
            push!(rows, success_row("muntz_collocation", N, result.dof, result.build_time,
                result.solve_time, result.linf_error, result.l2_grid_error, NaN,
                result.residual_linf, rhs_stability, "missing", result.notes;
                consistency_linf = result.consistency_linf,
                muntz_lambda = result.muntz_lambda,
                muntz_alpha = result.muntz_alpha,
                muntz_beta = result.muntz_beta))
            @printf("  N=%4d, dof=%4d, time=%.6e, Linf=%.6e, residual=%.6e, consistency=%.6e\n",
                N, result.dof, result.total_time, result.linf_error,
                result.residual_linf, result.consistency_linf)
        catch err
            @warn "muntz_collocation failed." N exception = (err, catch_backtrace())
            push!(rows, failure_row("muntz_collocation", N, N + 1, rhs_stability,
                "failed: $(typeof(err))"))
        end
    end
end

function run_direct_jacobi_method!(rows, Ns, rhs_coeffs, tgrid, exact_vals, rgrid, rhs_stability)
    println()
    println("Running direct_jacobi_collocation...")
    try
        solve_direct_jacobi_collocation(max(2, first(Ns)), rhs_coeffs)
    catch
    end
    for N in Ns
        try
            sol = solve_direct_jacobi_collocation(N, rhs_coeffs)
            f = t -> eval_direct_jacobi_solution(t, sol)
            linf, l2 = evaluate_error(f, tgrid, exact_vals)
            res = residual_linf(f, rhs_coeffs, rgrid)
            notes = @sprintf("direct Jacobi collocation on original y(t); endpoint augmented nodes; GJ q=%d", sol.quad_order)
            push!(rows, success_row("direct_jacobi_collocation", N, sol.dof, sol.build_time,
                sol.solve_time, linf, l2, NaN, res, rhs_stability, "missing", notes))
            @printf("  N=%4d, dof=%4d, time=%.6e, Linf=%.6e, residual=%.6e\n",
                N, sol.dof, sol.build_time + sol.solve_time, linf, res)
        catch err
            @warn "direct_jacobi_collocation failed." N exception = (err, catch_backtrace())
            push!(rows, failure_row("direct_jacobi_collocation", N, N + 1, rhs_stability,
                "failed: $(typeof(err))"))
        end
    end
end

function run_chentang_transformed_method!(rows, Ns, rhs_coeffs, tgrid, exact_vals, rgrid, rhs_stability)
    println()
    a = ct_a(CHENTANG_M)
    @printf("Running chentang_transformed_collocation with m=%d, a=%.6f...\n", CHENTANG_M, a)
    try
        solve_chentang_transformed_collocation(max(2, first(Ns)); m = CHENTANG_M)
    catch
    end
    for N in Ns
        try
            sol = solve_chentang_transformed_collocation(N; m = CHENTANG_M)
            f = t -> eval_chentang_transformed_solution(t, sol)
            linf, l2 = evaluate_error(f, tgrid, exact_vals)
            trans_linf = transformed_linf_error(sol, tgrid)
            res = residual_linf(f, rhs_coeffs, rgrid)
            notes = @sprintf("Chen--Tang transformed; m=%d; a=%.6f; system_residual=%.3e; consistency=%.3e; max_abs_U_exact_nodes=%.3e; protected=%d",
                sol.m, sol.a, sol.system_residual, sol.consistency, sol.max_abs_exact_nodes, sol.protected_count)
            push!(rows, success_row("chentang_transformed_collocation", N, sol.dof,
                sol.build_time, sol.solve_time, linf, l2, trans_linf, res, rhs_stability,
                sol.m, notes; consistency_linf = sol.consistency))
            @printf("  N=%4d, dof=%4d, time=%.6e, transformed_Linf=%.6e, original_Linf=%.6e, consistency=%.6e\n",
                N, sol.dof, sol.build_time + sol.solve_time, trans_linf, linf, sol.consistency)
        catch err
            @warn "chentang_transformed_collocation failed." N exception = (err, catch_backtrace())
            push!(rows, failure_row("chentang_transformed_collocation", N, N + 1, rhs_stability,
                "failed: $(typeof(err))"))
        end
    end
end

const CSV_HEADERS = [
    "method", "N", "dof", "build_time", "solve_time", "total_time",
    "linf_error", "l2_grid_error", "transformed_linf_error",
    "residual_linf", "consistency_linf", "rhs_stability", "chentang_m",
    "muntz_lambda", "muntz_alpha", "muntz_beta", "notes",
]

function main()
    Ns = parse_env_int_list("N_LIST", DEFAULT_N_LIST)
    brunner_Ns = parse_env_int_list("BRUNNER_N_LIST", DEFAULT_BRUNNER_N_LIST)
    maxN = max(maximum(Ns), maximum(brunner_Ns))
    tgrid = collect(range(0.0, 1.0; length = CHECK_GRID_SIZE))
    exact_vals = u_exact_t.(tgrid)
    rgrid = collect(range(0.0, 1.0; length = RESIDUAL_GRID_SIZE))

    println("============================================================")
    println("Three-method weakly singular Volterra benchmark on [0,1]")
    println("EXAMPLE = ", EXAMPLE)
    @printf("mu = %.6f, lambda = %.6f, beta = %.6f\n", MU, LAMBDA, BETA)
    println(exact_description())
    println("K(t,s) = 1 / (4t^2 + 4(s-1)^2 + 1e-14)")
    println("N_LIST = ", Ns)
    println("BRUNNER_N_LIST = ", brunner_Ns)
    @printf("CHENTANG_M = %d, a = mu + m - 1 = %.6f\n", CHENTANG_M, ct_a(CHENTANG_M))
    @printf("MUNTZ_LAMBDA = %.12g, MUNTZ_ALPHA = %.6f, MUNTZ_BETA = %.6f\n",
        MUNTZ_LAMBDA, MUNTZ_ALPHA, MUNTZ_BETA)
    @printf("MUNTZ_QUAD_ORDER = %d\n", MUNTZ_QUAD_ORDER)
    @printf("RHS_QUAD_ORDER = %d, tanh-sinh h = %s, precision = %d bits\n",
        RHS_QUAD_ORDER, RHS_TS_H, RHS_PRECISION)
    println("CHECK_GRID_SIZE = ", CHECK_GRID_SIZE)
    println("RESIDUAL_GRID_SIZE = ", RESIDUAL_GRID_SIZE)
    println("============================================================")

    rhs_stab = check_rhs_stability()
    rhs_coeffs, rhs_info = rhs_coefficients(maxN)
    @printf("Common RHS generation: point_time=%.6e, projection_time=%.6e\n",
        rhs_info.rhs_time, rhs_info.projection_time)

    rows = NamedTuple[]
    run_flag("RUN_OURS", "1") && run_our_method!(rows, Ns, rhs_coeffs, tgrid, exact_vals, rgrid, rhs_stab.diff)
    run_flag("RUN_BRUNNER", "1") && run_brunner_method!(rows, brunner_Ns, rhs_coeffs, tgrid, exact_vals, rgrid, rhs_stab.diff)
    run_flag("RUN_MUNTZ", "1") && run_muntz_method!(rows, Ns, rhs_coeffs, tgrid, exact_vals, rhs_stab.diff)
    run_flag("RUN_DIRECT_JACOBI", "0") && run_direct_jacobi_method!(rows, Ns, rhs_coeffs, tgrid, exact_vals, rgrid, rhs_stab.diff)
    run_flag("RUN_CHENTANG", "0") && run_chentang_transformed_method!(rows, Ns, rhs_coeffs, tgrid, exact_vals, rgrid, rhs_stab.diff)

    csv_path = results_csv_path()
    write_csv(csv_path, CSV_HEADERS, rows)
    println()
    println("Saved CSV: ", csv_path)

    try
        include(joinpath(SCRIPT_DIR, "plot_three_weak_methods.jl"))
        Base.invokelatest(plot_three_weak_methods, csv_path, FIGURES_DIR)
    catch err
        @warn "Plotting from compare script failed. Run plot_three_weak_methods.jl separately." exception = (err, catch_backtrace())
    end

    for method in ("our_coeffspace", "brunner_graded", "muntz_collocation",
        "direct_jacobi_collocation", "chentang_transformed_collocation")
        mrows = [r for r in rows if r.method == method && isfinite(r.linf_error)]
        isempty(mrows) && continue
        best = mrows[argmin([r.linf_error for r in mrows])]
        @printf("Best %-30s: N=%d, dof=%d, time=%.6e, Linf=%.6e\n",
            method, best.N, best.dof, best.total_time, best.linf_error)
    end
    println("Figures directory: ", FIGURES_DIR)
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
