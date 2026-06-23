# ex3_run.jl
#
# Purpose:
#   Main Example 3 generator for the pantograph first-kind Volterra equation.
#
# Problem:
#   First-kind pantograph Volterra equation represented in a Chebyshev basis.
#   The script builds the coefficient-space operator and compares truncation
#   sizes against a fixed reference.
#
# Method:
#   Uses VoltMat-Cheb-style coefficient operations and rest-matrix machinery in
#   this file, with optional FFTW acceleration for Chebyshev transforms and a
#   high-precision RHS setup where required.
#
# Output files:
#   data/ex3_error_time.csv
#   data/ex3_A_Nmax.mat
#
# Run from the repository root:
#   julia --project=. examples/ex3_pantograph/ex3_run.jl

const THIS_DIR = @__DIR__

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

const VOLTERRA_ROOT = find_project_root()
const SRC_DIR = joinpath(VOLTERRA_ROOT, "src")
const VOLTERRA_PROJECT = joinpath(VOLTERRA_ROOT, "Project.toml")
const DATA_DIR = joinpath(THIS_DIR, "data")

if isfile(VOLTERRA_PROJECT) && !(VOLTERRA_ROOT in LOAD_PATH)
    pushfirst!(LOAD_PATH, VOLTERRA_ROOT)
end

using LinearAlgebra
using SparseArrays
using DelimitedFiles
using Printf
using Statistics

const HAS_FFTW = try
    @eval using FFTW
    true
catch err
    @warn "FFTW is not available; Chebyshev coefficient transforms will use a slow DCT-I fallback." exception=(err, catch_backtrace())
    false
end

try
    @eval using SpecialFunctions
catch err
    error(
        "SpecialFunctions is required for the complex error function in fprime_physical. " *
        "Run this script with the Volterra project environment, for example " *
        "`julia --project=$(VOLTERRA_ROOT) $(joinpath(THIS_DIR, basename(@__FILE__)))`. " *
        "Original error: $(err)",
    )
end

include(joinpath(SRC_DIR, "RestMat", "restmat_cheb.jl"))
include(joinpath(SRC_DIR, "VoltMat", "voltmat_cheb.jl"))
using .VoltMat_Cheb

const q_default = 0.5
const Ns = collect(2:1:100)
const tol_machine = eps(Float64)
const tol_volt = tol_machine
const tol_cheb = tol_machine
const sanity_N = 40
const diagnostic_Ns = (40, 60, 80, 100)
const use_high_precision_rhs = true
const rhs_precision = 256
const rhs_quad_n = 256
const N_rhs_check = 160

function _complex_big_erf_available()
    try
        setprecision(rhs_precision) do
            z = erf(Complex{BigFloat}(BigFloat(1), BigFloat(1)))
            isfinite(real(z)) && isfinite(imag(z))
        end
    catch
        false
    end
end

const HAS_COMPLEX_BIG_ERF = _complex_big_erf_available()

# -----------------------------------------------------------------------------
# Basic functions on the standard interval [-1, 1]
# -----------------------------------------------------------------------------

tau_q(X, q) = q * (X + 1) - 1

m_q_fun(X, q) = exp(-((1 - q)^2) * (X + 1)^2 / 4)

G1_fun(X, S) = -0.5 * (X - S) * exp(-((X - S)^2) / 4)

function G2_fun(X, S, q)
    Y = tau_q(S, q)
    return -0.5 * (X - Y) * exp(-((X - Y)^2) / 4)
end

utrue_standard(X) = exp((X + 1) / 2) * cos(10 * (X + 1))

function utrue_standard_big(X; precision::Integer = 256)
    setprecision(precision) do
        Xb = BigFloat(X)
        val = exp((Xb + 1) / 2) * cos(10 * (Xb + 1))
        return Float64(val)
    end
end

function fprime_physical(x, q)
    lambda = 1.0 + 20.0im
    z =
        sqrt(pi) / 2 *
        lambda *
        exp(lambda * x + lambda^2 / 4) *
        (
            erf((1 - q) * x + lambda / 2) -
            erf(lambda / 2)
        )

    endpoint_term =
        (1 - q) *
        exp(q * x - (1 - q)^2 * x^2) *
        cos(20 * q * x)

    return real(z) + endpoint_term
end

rhs_standard(X, q) = fprime_physical((X + 1) / 2, q)

duffy_S(X, T) = (X + 1) / 2 * T + (X - 1) / 2

# -----------------------------------------------------------------------------
# High-precision right-hand side helpers
# -----------------------------------------------------------------------------

const _gausslegendre_big_cache =
    Dict{Tuple{Int,Int},Tuple{Vector{BigFloat},Vector{BigFloat}}}()

function _legendre_p_and_derivative_big(n::Integer, x::BigFloat)
    if n == 0
        return one(BigFloat), zero(BigFloat)
    elseif n == 1
        return x, one(BigFloat)
    end

    p_nm1 = one(BigFloat)
    p_n = x

    for k in 1:(n - 1)
        kb = BigFloat(k)
        p_np1 = ((2 * kb + 1) * x * p_n - kb * p_nm1) / (kb + 1)
        p_nm1, p_n = p_n, p_np1
    end

    dp = BigFloat(n) * (x * p_n - p_nm1) / (x^2 - 1)

    return p_n, dp
end

function gausslegendre_big(n::Integer; precision::Integer = rhs_precision)
    n >= 1 || throw(ArgumentError("n must be positive."))

    key = (Int(n), Int(precision))

    if haskey(_gausslegendre_big_cache, key)
        return _gausslegendre_big_cache[key]
    end

    nodes, weights = setprecision(precision) do
        x = Vector{BigFloat}(undef, n)
        w = Vector{BigFloat}(undef, n)
        m = cld(n, 2)
        tol_newton = eps(BigFloat)

        for i in 1:m
            ib = BigFloat(i)
            nb = BigFloat(n)
            z = cos(big(pi) * (ib - BigFloat(0.25)) / (nb + BigFloat(0.5)))

            while true
                p, dp = _legendre_p_and_derivative_big(n, z)
                dz = p / dp
                z -= dz

                if abs(dz) <= tol_newton * max(one(BigFloat), abs(z))
                    break
                end
            end

            _, dp = _legendre_p_and_derivative_big(n, z)
            wi = 2 / ((1 - z^2) * dp^2)

            x[i] = -z
            x[n + 1 - i] = z
            w[i] = wi
            w[n + 1 - i] = wi
        end

        x, w
    end

    _gausslegendre_big_cache[key] = (nodes, weights)

    return nodes, weights
end

u_physical_big(x) = exp(x) * cos(BigFloat(20) * x)
K_physical_big(x, y) = exp(-((x - y)^2))
Kx_physical_big(x, y) = -BigFloat(2) * (x - y) * exp(-((x - y)^2))

function _integrate_0_x_big(f, xb::BigFloat, nodes, weights)
    xb == 0 && return zero(BigFloat)

    half = xb / 2
    center = xb / 2
    s = zero(BigFloat)

    @inbounds for i in eachindex(nodes)
        y = center + half * nodes[i]
        s += weights[i] * f(y)
    end

    return half * s
end

function _fprime_physical_big_quad_value(
    x,
    q;
    precision::Integer = rhs_precision,
    nquad::Integer = rhs_quad_n,
)
    setprecision(precision) do
        xb = BigFloat(x)
        qb = BigFloat(q)
        nodes, weights = gausslegendre_big(nquad; precision=precision)

        endpoint =
            u_physical_big(xb) -
            qb * K_physical_big(xb, qb * xb) * u_physical_big(qb * xb)

        int1 = _integrate_0_x_big(
            s -> Kx_physical_big(xb, s) * u_physical_big(s),
            xb,
            nodes,
            weights,
        )

        int2 = _integrate_0_x_big(
            s -> Kx_physical_big(xb, qb * s) * u_physical_big(qb * s),
            xb,
            nodes,
            weights,
        )

        endpoint + int1 - qb * int2
    end
end

function fprime_physical_big_quad(
    x,
    q;
    precision::Integer = rhs_precision,
    nquad::Integer = rhs_quad_n,
)
    return Float64(
        _fprime_physical_big_quad_value(
            x,
            q;
            precision = precision,
            nquad = nquad,
        ),
    )
end

function _fprime_physical_big_erf_value(x, q; precision::Integer = rhs_precision)
    setprecision(precision) do
        xb = BigFloat(x)
        qb = BigFloat(q)
        oneb = BigFloat(1)
        lambda = Complex{BigFloat}(oneb, BigFloat(20))

        term1 =
            sqrt(big(pi)) / 2 *
            lambda *
            exp(lambda * xb + lambda^2 / 4) *
            (
                erf((oneb - qb) * xb + lambda / 2) -
                erf(lambda / 2)
            )

        term2 =
            (oneb - qb) *
            exp(qb * xb - (oneb - qb)^2 * xb^2) *
            cos(BigFloat(20) * qb * xb)

        real(term1) + term2
    end
end

function _fprime_physical_big_value(
    x,
    q;
    precision::Integer = rhs_precision,
    nquad::Integer = rhs_quad_n,
)
    if HAS_COMPLEX_BIG_ERF
        return _fprime_physical_big_erf_value(x, q; precision=precision)
    end

    return _fprime_physical_big_quad_value(
        x,
        q;
        precision = precision,
        nquad = nquad,
    )
end

function fprime_physical_big(x, q; precision::Integer = rhs_precision)
    return Float64(_fprime_physical_big_value(x, q; precision=precision))
end

rhs_standard_big(X, q; precision::Integer = rhs_precision) =
    fprime_physical_big((X + 1) / 2, q; precision=precision)

function rhs_standard_big_value(X, q; precision::Integer = rhs_precision)
    setprecision(precision) do
        _fprime_physical_big_value(
            (BigFloat(X) + 1) / 2,
            q;
            precision = precision,
        )
    end
end

# -----------------------------------------------------------------------------
# Chebyshev utilities
# -----------------------------------------------------------------------------

function cheb_lobatto_nodes(M::Integer)
    M >= 1 || throw(ArgumentError("M must be positive."))
    return cos.(pi .* (0:M) ./ M)
end

function cheb_coeffs_from_lobatto_values(vals_in::AbstractVector)
    M = length(vals_in) - 1
    M >= 1 || throw(ArgumentError("M must be positive."))

    vals = Float64.(vals_in)

    if !all(isfinite, vals)
        throw(ArgumentError("Chebyshev samples contain non-finite values."))
    end

    if HAS_FFTW
        c = FFTW.r2r(vals, FFTW.REDFT00)
        c ./= M
        c[1] /= 2
        c[end] /= 2
        return c
    end

    c = zeros(Float64, M + 1)

    @inbounds for k in 0:M
        s = vals[1] + (iseven(k) ? vals[end] : -vals[end])
        for j in 1:(M - 1)
            s += 2 * vals[j + 1] * cos(pi * j * k / M)
        end
        c[k + 1] = s / M
    end

    c[1] /= 2
    c[end] /= 2

    return c
end

function cheb_coeffs_lobatto(f, M::Integer)
    xs = cheb_lobatto_nodes(M)
    vals = Float64[f(x) for x in xs]

    return cheb_coeffs_from_lobatto_values(vals)
end

function _nextpow2_ge(n::Integer)
    n >= 1 || throw(ArgumentError("n must be positive."))

    p = 1
    while p < n
        p <<= 1
    end

    return p
end

function _chop_cheb_coeffs(c::AbstractVector{Float64}, tol::Float64; buffer::Int = 4)
    scale = max(1.0, norm(c, Inf))
    threshold = tol * scale

    last_idx = findlast(k -> abs(c[k]) > threshold, eachindex(c))

    degree =
        last_idx === nothing ? 0 :
        min(length(c) - 1, last_idx - 1 + buffer)

    return collect(@view c[1:(degree + 1)]), degree, scale, threshold
end

function cheb_coeffs_adaptive(
    f;
    tol::Float64 = tol_machine,
    maxdeg::Integer = 4096,
    M0::Integer = 256,
    tail_count::Integer = 32,
    buffer::Integer = 4,
)
    tol > 0 || throw(ArgumentError("tol must be positive."))
    maxdeg >= 8 || throw(ArgumentError("maxdeg must be at least 8."))

    M = _nextpow2_ge(M0)
    M <= maxdeg || throw(ArgumentError("M0 must not exceed maxdeg."))

    last_c = Float64[]
    last_tail = Inf
    converged = false

    while M <= maxdeg
        c = cheb_coeffs_lobatto(f, M)
        scale = max(1.0, norm(c, Inf))

        ntail = min(length(c), max(Int(tail_count), max(8, cld(M, 8))))
        last_tail = maximum(abs, @view c[(end - ntail + 1):end])
        last_c = c

        if last_tail <= tol * scale
            converged = true
            break
        end

        M *= 2
    end

    c_chop, degree, scale, threshold = _chop_cheb_coeffs(last_c, tol; buffer=Int(buffer))

    info = (
        degree = degree,
        sample_degree = length(last_c) - 1,
        tail = last_tail,
        scale = scale,
        threshold = threshold,
        converged = converged,
        tol = tol,
    )

    return c_chop, info
end

function cheb_coeffs_lobatto_big(f, M::Integer; precision::Integer = rhs_precision)
    M >= 1 || throw(ArgumentError("M must be positive."))

    setprecision(precision) do
        vals = Vector{BigFloat}(undef, M + 1)

        @inbounds for j in 0:M
            xj = cos(big(pi) * BigFloat(j) / BigFloat(M))
            vals[j + 1] = BigFloat(f(xj))
        end

        coeffs = zeros(BigFloat, M + 1)

        @inbounds for k in 0:M
            s = zero(BigFloat)

            for j in 0:M
                wj = (j == 0 || j == M) ? BigFloat(0.5) : one(BigFloat)
                s +=
                    wj *
                    vals[j + 1] *
                    cos(big(pi) * BigFloat(k * j) / BigFloat(M))
            end

            coeffs[k + 1] = 2 * s / BigFloat(M)
        end

        coeffs[1] /= 2
        coeffs[end] /= 2

        return Float64.(coeffs)
    end
end

function cheb_coeffs_adaptive_big(
    f;
    tol::Float64 = tol_machine,
    maxdeg::Integer = 4096,
    M0::Integer = 256,
    tail_count::Integer = 32,
    buffer::Integer = 4,
    precision::Integer = rhs_precision,
)
    tol > 0 || throw(ArgumentError("tol must be positive."))
    maxdeg >= 8 || throw(ArgumentError("maxdeg must be at least 8."))

    M = _nextpow2_ge(M0)
    M <= maxdeg || throw(ArgumentError("M0 must not exceed maxdeg."))

    last_c = Float64[]
    last_tail = Inf
    converged = false

    while M <= maxdeg
        c = cheb_coeffs_lobatto_big(f, M; precision=precision)
        scale = max(1.0, norm(c, Inf))

        ntail = min(length(c), max(Int(tail_count), max(8, cld(M, 8))))
        last_tail = maximum(abs, @view c[(end - ntail + 1):end])
        last_c = c

        if last_tail <= tol * scale
            converged = true
            break
        end

        M *= 2
    end

    c_chop, degree, scale, threshold = _chop_cheb_coeffs(last_c, tol; buffer=Int(buffer))

    info = (
        degree = degree,
        sample_degree = length(last_c) - 1,
        tail = last_tail,
        scale = scale,
        threshold = threshold,
        converged = converged,
        tol = tol,
        precision = precision,
    )

    return c_chop, info
end

function check_big_cheb_coeffs_normalization()
    f(x) = 1 + x + 2x^2
    c_float = cheb_coeffs_lobatto(f, 16)
    c_big = cheb_coeffs_lobatto_big(f, 16; precision=rhs_precision)

    return norm(c_float - c_big, Inf)
end

function cheb2_coeffs_lobatto(f, Mx::Integer, Mt::Integer)
    Mx >= 1 || throw(ArgumentError("Mx must be positive."))
    Mt >= 1 || throw(ArgumentError("Mt must be positive."))

    xs = cheb_lobatto_nodes(Mx)
    ts = cheb_lobatto_nodes(Mt)
    vals = Matrix{Float64}(undef, Mx + 1, Mt + 1)

    @inbounds for j in 1:(Mt + 1)
        T = ts[j]
        for i in 1:(Mx + 1)
            vals[i, j] = f(xs[i], T)
        end
    end

    if !all(isfinite, vals)
        throw(ArgumentError("2D Chebyshev samples contain non-finite values."))
    end

    if HAS_FFTW
        C = FFTW.r2r(vals, FFTW.REDFT00)
        C ./= Mx * Mt
        C[1, :] ./= 2
        C[end, :] ./= 2
        C[:, 1] ./= 2
        C[:, end] ./= 2

        return C
    end

    tmp = Matrix{Float64}(undef, Mx + 1, Mt + 1)
    C = Matrix{Float64}(undef, Mx + 1, Mt + 1)

    @inbounds for j in 1:(Mt + 1)
        tmp[:, j] .= cheb_coeffs_from_lobatto_values(@view vals[:, j])
    end

    @inbounds for i in 1:(Mx + 1)
        C[i, :] .= cheb_coeffs_from_lobatto_values(@view tmp[i, :])
    end

    return C
end

function cheb2_tail(C::AbstractMatrix; tailcheck::Integer = 32)
    nr, nc = size(C)
    tr = min(nr, max(Int(tailcheck), max(1, cld(nr, 8))))
    tc = min(nc, max(Int(tailcheck), max(1, cld(nc, 8))))

    return max(
        maximum(abs, @view C[(end - tr + 1):end, :]),
        maximum(abs, @view C[:, (end - tc + 1):end]),
    )
end

function chop_cheb2_coeffs(C::AbstractMatrix{Float64}; tol::Float64 = tol_machine)
    maxabs = max(1.0, maximum(abs, C))
    threshold = tol * maxabs

    row_stop = findlast(i -> maximum(abs, @view C[i, :]) > threshold, axes(C, 1))
    col_stop = findlast(j -> maximum(abs, @view C[:, j]) > threshold, axes(C, 2))

    row_stop = row_stop === nothing ? first(axes(C, 1)) : row_stop
    col_stop = col_stop === nothing ? first(axes(C, 2)) : col_stop

    return collect(@view C[1:row_stop, 1:col_stop])
end

function cheb2_coeffs_adaptive(
    f;
    tol::Float64 = tol_machine,
    maxdeg::Integer = 1024,
    M0::Integer = 64,
    tailcheck::Integer = 32,
)
    tol > 0 || throw(ArgumentError("tol must be positive."))
    maxdeg >= 8 || throw(ArgumentError("maxdeg must be at least 8."))

    Mx = _nextpow2_ge(M0)
    Mt = _nextpow2_ge(M0)
    max(Mx, Mt) <= maxdeg || throw(ArgumentError("M0 must not exceed maxdeg."))

    last_C = Matrix{Float64}(undef, 0, 0)
    last_tail = Inf
    last_scale = 1.0
    converged = false

    while max(Mx, Mt) <= maxdeg
        C = cheb2_coeffs_lobatto(f, Mx, Mt)
        last_scale = max(1.0, maximum(abs, C))
        last_tail = cheb2_tail(C; tailcheck=tailcheck)
        last_C = C

        if last_tail <= tol * last_scale
            converged = true
            break
        end

        Mx *= 2
        Mt *= 2
    end

    if !converged
        @warn "Adaptive 2D Chebyshev coefficient computation reached maxdeg before the tail criterion was met." maxdeg last_tail tol
    end

    C_chop = chop_cheb2_coeffs(last_C; tol=tol)

    info = (
        sample_size = size(last_C),
        tail = last_tail,
        maxabs = last_scale,
        converged = converged,
        tol = tol,
    )

    return C_chop, info
end

function coeffs_to_degree(c::AbstractVector, N::Integer)
    N >= 0 || throw(ArgumentError("N must be nonnegative."))

    out = zeros(Float64, N + 1)
    L = min(length(c), N + 1)
    out[1:L] .= c[1:L]

    return out
end

function exact_solution_coeffs(N::Integer; tol::Float64 = tol_machine)
    c_raw, _ = cheb_coeffs_adaptive(
        utrue_standard;
        tol = tol,
        maxdeg = 4096,
        M0 = 256,
    )

    return coeffs_to_degree(c_raw, N)
end

function cheb_eval_clenshaw(c::AbstractVector, x)
    N = length(c) - 1

    if N == 0
        return c[1]
    end

    b1 = zero(promote_type(eltype(c), typeof(x)))
    b2 = zero(promote_type(eltype(c), typeof(x)))

    @inbounds for k in N:-1:1
        b0 = 2 * x * b1 - b2 + c[k + 1]
        b2 = b1
        b1 = b0
    end

    return x * b1 - b2 + c[1]
end

cheb_eval_many(c::AbstractVector, xs) = [cheb_eval_clenshaw(c, x) for x in xs]

function cheb_basis_values(N::Integer, x)
    N >= 0 || throw(ArgumentError("N must be nonnegative."))

    vals = Vector{Float64}(undef, N + 1)
    vals[1] = 1.0

    if N >= 1
        vals[2] = x
    end

    @inbounds for k in 1:(N - 1)
        vals[k + 2] = 2 * x * vals[k + 1] - vals[k]
    end

    return vals
end

function cheb2_eval(C::AbstractMatrix, X, T)
    tx = cheb_basis_values(size(C, 1) - 1, X)
    tt = cheb_basis_values(size(C, 2) - 1, T)

    return dot(tx, C * tt)
end

function check_cheb2_approx(C, f; ntest::Integer = 41)
    ntest >= 2 || throw(ArgumentError("ntest must be at least 2."))

    xs = range(-1.0, 1.0; length=ntest)
    ts = range(-1.0, 1.0; length=ntest)
    err = 0.0

    for X in xs
        for T in ts
            err = max(err, abs(cheb2_eval(C, X, T) - f(X, T)))
        end
    end

    return err
end

function chebmul_matrix_from_coeffs(a::AbstractVector, N::Integer)
    N >= 0 || throw(ArgumentError("N must be nonnegative."))

    m = min(length(a) - 1, 2N)
    M = zeros(Float64, N + 1, N + 1)

    @inbounds for j in 0:N
        M[j + 1, j + 1] += a[1]

        for k in 1:m
            ak = a[k + 1] / 2

            i1 = k + j
            if i1 <= N
                M[i1 + 1, j + 1] += ak
            end

            i2 = abs(k - j)
            if i2 <= N
                M[i2 + 1, j + 1] += ak
            end
        end
    end

    return M
end

# -----------------------------------------------------------------------------
# Sanity checks
# -----------------------------------------------------------------------------

function _check_square_size(name::AbstractString, A, N::Integer)
    expected = (N + 1, N + 1)
    size(A) == expected || error("$(name) has size $(size(A)); expected $(expected).")

    return nothing
end

function _strict_lower_inf(A)
    m, n = size(A)
    val = 0.0

    @inbounds for j in 1:n
        for i in (j + 1):m
            val = max(val, abs(A[i, j]))
        end
    end

    return val
end

function _check_upper_triangular(name::AbstractString, A)
    lower = _strict_lower_inf(A)
    scale = max(1.0, norm(A, Inf))
    tol = tol_machine * scale

    lower <= tol || error("$(name) is not approximately upper triangular: lower norm $(lower), tolerance $(tol).")

    return lower
end

function _restriction_sanity_error(N::Integer, q; tol::Float64 = tol_cheb)
    Rq = restmat_cheb(-1.0, 2.0 * q - 1.0, N)
    cU_raw, _ = cheb_coeffs_adaptive(utrue_standard; tol=tol, maxdeg=4096)
    cU = coeffs_to_degree(cU_raw, N)
    cU_tau = Rq * cU

    Xcheck = [-1.0, -0.75, -0.2, 0.0, 0.35, 0.8, 1.0]
    vals_R = cheb_eval_many(cU_tau, Xcheck)
    vals_exact = [utrue_standard(tau_q(X, q)) for X in Xcheck]

    return maximum(abs.(vals_R .- vals_exact))
end

function _run_system_sanity_checks(N, Rq, Mmq, V1, V2, A; q, tol_cheb)
    _check_square_size("Rq", Rq, N)
    _check_upper_triangular("Rq", Rq)
    _check_square_size("M[m_q]", Mmq, N)
    _check_square_size("V1", V1, N)
    _check_square_size("V2", V2, N)
    _check_square_size("A", A, N)

    if N == sanity_N
        restriction_err = _restriction_sanity_error(N, q; tol=tol_cheb)
        @printf("restriction sanity at N=%d: %.6e\n", N, restriction_err)
    end

    return nothing
end

# -----------------------------------------------------------------------------
# Assembly and solve
# -----------------------------------------------------------------------------

function build_volterra_kernel_coeffs(q; tol::Float64 = tol_machine, maxdeg::Integer = 1024)
    C1, info1 = cheb2_coeffs_adaptive(
        (X, T) -> G1_fun(X, duffy_S(X, T));
        tol = tol,
        maxdeg = maxdeg,
    )

    C2, info2 = cheb2_coeffs_adaptive(
        (X, T) -> G2_fun(X, duffy_S(X, T), q);
        tol = tol,
        maxdeg = maxdeg,
    )

    err_C1 = check_cheb2_approx(C1, (X, T) -> G1_fun(X, duffy_S(X, T)))
    err_C2 = check_cheb2_approx(C2, (X, T) -> G2_fun(X, duffy_S(X, T), q))

    @printf(
        "C1 size = %d x %d, maxabs = %.3e, tail = %.3e\n",
        size(C1, 1),
        size(C1, 2),
        maximum(abs, C1),
        info1.tail,
    )
    @printf(
        "C2 size = %d x %d, maxabs = %.3e, tail = %.3e\n",
        size(C2, 1),
        size(C2, 2),
        maximum(abs, C2),
        info2.tail,
    )
    @printf("C1 interpolation check error = %.3e\n", err_C1)
    @printf("C2 interpolation check error = %.3e\n", err_C2)

    return (
        C1 = C1,
        C2 = C2,
        info1 = info1,
        info2 = info2,
        err_C1 = err_C1,
        err_C2 = err_C2,
    )
end

function assemble_system(
    N::Integer,
    q;
    tol_volt::Float64 = tol_volt,
    tol_cheb::Float64 = tol_cheb,
    rhs_coeffs = nothing,
    kernel_coeffs = nothing,
)
    N >= 0 || throw(ArgumentError("N must be nonnegative."))
    0 < q < 1 || throw(ArgumentError("q must lie in (0, 1)."))

    Rq = restmat_cheb(-1.0, 2.0 * q - 1.0, N)

    m_coeffs, m_info = cheb_coeffs_adaptive(
        X -> m_q_fun(X, q);
        tol = tol_cheb,
        maxdeg = 4096,
        M0 = 128,
    )
    Mmq = chebmul_matrix_from_coeffs(m_coeffs, N)

    kernels =
        kernel_coeffs === nothing ?
        build_volterra_kernel_coeffs(q; tol=tol_machine, maxdeg=1024) :
        kernel_coeffs

    V1 = VoltMat_Cheb.voltmat_square_from_coeffmat_fft_convolution(kernels.C1, N; tol=tol_volt)
    V2 = VoltMat_Cheb.voltmat_square_from_coeffmat_fft_convolution(kernels.C2, N; tol=tol_volt)

    I_N = Matrix{Float64}(I, N + 1, N + 1)
    A = I_N - q * Mmq * Rq + Matrix(V1) - q * Matrix(V2) * Rq

    c_g = if rhs_coeffs === nothing
        c_rhs_raw, _ = cheb_coeffs_adaptive(
            X -> rhs_standard(X, q);
            tol = tol_cheb,
            maxdeg = 4096,
            M0 = 256,
        )
        coeffs_to_degree(c_rhs_raw, N)
    else
        coeffs_to_degree(rhs_coeffs, N)
    end

    _run_system_sanity_checks(N, Rq, Mmq, V1, V2, A; q=q, tol_cheb=tol_cheb)

    return (
        A = A,
        rhs = c_g,
        Rq = Rq,
        Mmq = Mmq,
        V1 = V1,
        V2 = V2,
        m_coeffs = m_coeffs,
        m_info = m_info,
        kernel_coeffs = kernels,
    )
end

function solve_for_N(
    N::Integer,
    q;
    tol_volt::Float64 = tol_volt,
    tol_cheb::Float64 = tol_cheb,
    rhs_coeffs = nothing,
    kernel_coeffs = nothing,
)
    sys = assemble_system(
        N,
        q;
        tol_volt = tol_volt,
        tol_cheb = tol_cheb,
        rhs_coeffs = rhs_coeffs,
        kernel_coeffs = kernel_coeffs,
    )

    cu = sys.A \ sys.rhs
    residual_inf = norm(sys.A * cu - sys.rhs, Inf)
    cond_A = cond(sys.A)
    c_exact = exact_solution_coeffs(N; tol=tol_machine)
    exact_discrete_residual_abs = norm(sys.A * c_exact - sys.rhs, Inf)
    exact_discrete_residual_rel =
        exact_discrete_residual_abs / max(norm(sys.rhs, Inf), eps(Float64))
    coeff_err = cu - c_exact

    coeff_error_inf = norm(coeff_err, Inf)
    coeff_error_1 = norm(coeff_err, 1)
    coeff_error_2 = norm(coeff_err, 2)

    coeff_exact_inf = norm(c_exact, Inf)
    coeff_exact_1 = norm(c_exact, 1)
    coeff_exact_2 = norm(c_exact, 2)

    coeff_error_rel_inf = coeff_error_inf / max(coeff_exact_inf, eps(Float64))
    coeff_error_rel_1 = coeff_error_1 / max(coeff_exact_1, eps(Float64))
    coeff_error_rel_2 = coeff_error_2 / max(coeff_exact_2, eps(Float64))
    exact_coeff_error_inf = coeff_error_inf

    Xtest = collect(range(-1.0, 1.0; length=2001))
    U_num = cheb_eval_many(cu, Xtest)
    U_exact = utrue_standard.(Xtest)
    U_exact_big = [utrue_standard_big(X; precision=rhs_precision) for X in Xtest]
    U_exact_trunc = cheb_eval_many(c_exact, Xtest)
    err_inf = maximum(abs.(U_num .- U_exact))
    err_inf_bigtruth = maximum(abs.(U_num .- U_exact_big))
    exact_trunc_error_inf = maximum(abs.(U_exact_trunc .- U_exact))

    point_error_over_coeff_1 = err_inf / max(coeff_error_1, eps(Float64))
    point_error_over_coeff_inf = err_inf / max(coeff_error_inf, eps(Float64))

    return (
        cu = cu,
        c_exact = c_exact,
        coeff_err = coeff_err,
        err_inf = err_inf,
        err_inf_bigtruth = err_inf_bigtruth,
        residual_inf = residual_inf,
        cond_A = cond_A,
        exact_discrete_residual_abs = exact_discrete_residual_abs,
        exact_discrete_residual_rel = exact_discrete_residual_rel,
        exact_coeff_error_inf = exact_coeff_error_inf,
        coeff_error_inf = coeff_error_inf,
        coeff_error_1 = coeff_error_1,
        coeff_error_2 = coeff_error_2,
        coeff_error_rel_inf = coeff_error_rel_inf,
        coeff_error_rel_1 = coeff_error_rel_1,
        coeff_error_rel_2 = coeff_error_rel_2,
        exact_trunc_error_inf = exact_trunc_error_inf,
        point_error_over_coeff_1 = point_error_over_coeff_1,
        point_error_over_coeff_inf = point_error_over_coeff_inf,
        A = sys.A,
        rhs = sys.rhs,
        Rq = sys.Rq,
        Mmq = sys.Mmq,
        V1 = sys.V1,
        V2 = sys.V2,
        m_info = sys.m_info,
        kernel_coeffs = sys.kernel_coeffs,
    )
end

function compute_rhs_coeffs(
    q,
    rhs_mode::Symbol;
    maxN::Integer,
    tol::Float64 = tol_cheb,
    precision::Integer = rhs_precision,
)
    rhs_mode in (:float64, :big) ||
        throw(ArgumentError("rhs_mode must be :float64 or :big."))

    c_raw, info = if rhs_mode == :float64
        cheb_coeffs_adaptive(
            X -> rhs_standard(X, q);
            tol = tol,
            maxdeg = 4096,
            M0 = 256,
        )
    else
        cheb_coeffs_adaptive_big(
            X -> rhs_standard_big_value(X, q; precision=precision);
            tol = tol,
            maxdeg = 4096,
            M0 = 256,
            precision = precision,
        )
    end

    return (
        mode = rhs_mode,
        coeffs_raw = c_raw,
        coeffs = coeffs_to_degree(c_raw, maxN),
        info = info,
    )
end

function run_sweep(;
    q = q_default,
    Ns_to_run = Ns,
    tol_volt::Float64 = tol_volt,
    tol_cheb::Float64 = tol_cheb,
    rhs_mode::Symbol = :float64,
    rhs_coeffs = nothing,
    rhs_info = nothing,
    kernel_coeffs = nothing,
)
    isempty(Ns_to_run) && throw(ArgumentError("Ns_to_run must not be empty."))
    0 < q < 1 || throw(ArgumentError("q must lie in (0, 1)."))

    maxN = maximum(Ns_to_run)

    println("============================================================")
    println("Pantograph first-kind Volterra equation in Chebyshev basis")
    @printf("rhs_mode = %s\n", string(rhs_mode))
    @printf("q = %.16g, N = %d:%d:%d\n", q, first(Ns_to_run), Ns_to_run[2] - Ns_to_run[1], last(Ns_to_run))
    @printf("tol_volt = %.3e, tol_cheb = %.3e\n", tol_volt, tol_cheb)
    println("============================================================")

    rhs_data = if rhs_coeffs === nothing || rhs_info === nothing
        compute_rhs_coeffs(
            q,
            rhs_mode;
            maxN = maxN,
            tol = tol_cheb,
            precision = rhs_precision,
        )
    else
        (
            mode = rhs_mode,
            coeffs_raw = rhs_coeffs,
            coeffs = coeffs_to_degree(rhs_coeffs, maxN),
            info = rhs_info,
        )
    end

    @printf("rhs Chebyshev degree after chop = %d\n", rhs_data.info.degree)
    @printf("rhs sample degree               = %d\n", rhs_data.info.sample_degree)
    @printf("rhs tail                        = %.6e\n", rhs_data.info.tail)
    println("------------------------------------------------------------")

    kernels =
        kernel_coeffs === nothing ?
        build_volterra_kernel_coeffs(q; tol=tol_machine, maxdeg=1024) :
        kernel_coeffs
    println("------------------------------------------------------------")

    errors = Float64[]
    bigtruth_errors = Float64[]
    residuals = Float64[]
    conds = Float64[]
    exact_residual_abs = Float64[]
    exact_residual_rel = Float64[]
    exact_coeff_errors = Float64[]
    coeff_error_infs = Float64[]
    coeff_error_1s = Float64[]
    coeff_error_2s = Float64[]
    coeff_error_rel_infs = Float64[]
    coeff_error_rel_1s = Float64[]
    coeff_error_rel_2s = Float64[]
    exact_trunc_errors = Float64[]
    point_error_over_coeff_1s = Float64[]
    point_error_over_coeff_infs = Float64[]
    A100 = nothing

    for N in Ns_to_run
        result = solve_for_N(
            N,
            q;
            tol_volt = tol_volt,
            tol_cheb = tol_cheb,
            rhs_coeffs = rhs_data.coeffs,
            kernel_coeffs = kernels,
        )

        push!(errors, result.err_inf)
        push!(bigtruth_errors, result.err_inf_bigtruth)
        push!(residuals, result.residual_inf)
        push!(conds, result.cond_A)
        push!(exact_residual_abs, result.exact_discrete_residual_abs)
        push!(exact_residual_rel, result.exact_discrete_residual_rel)
        push!(exact_coeff_errors, result.exact_coeff_error_inf)
        push!(coeff_error_infs, result.coeff_error_inf)
        push!(coeff_error_1s, result.coeff_error_1)
        push!(coeff_error_2s, result.coeff_error_2)
        push!(coeff_error_rel_infs, result.coeff_error_rel_inf)
        push!(coeff_error_rel_1s, result.coeff_error_rel_1)
        push!(coeff_error_rel_2s, result.coeff_error_rel_2)
        push!(exact_trunc_errors, result.exact_trunc_error_inf)
        push!(point_error_over_coeff_1s, result.point_error_over_coeff_1)
        push!(point_error_over_coeff_infs, result.point_error_over_coeff_inf)

        if N == 100
            A100 = result.A
        end

        @printf(
            "N = %3d, err_inf = %.6e, residual_inf = %.6e, cond_A = %.6e\n",
            N,
            result.err_inf,
            result.residual_inf,
            result.cond_A,
        )

        if N in diagnostic_Ns
            @printf(
                "N = %3d | exact discrete residual abs = %.3e, rel = %.3e, coeff error = %.3e\n",
                N,
                result.exact_discrete_residual_abs,
                result.exact_discrete_residual_rel,
                result.exact_coeff_error_inf,
            )
            @printf(
                "N = %3d | coeff err inf = %.3e, l1 = %.3e, l2 = %.3e\n",
                N,
                result.coeff_error_inf,
                result.coeff_error_1,
                result.coeff_error_2,
            )
            @printf(
                "        coeff rel inf = %.3e, rel l1 = %.3e, rel l2 = %.3e\n",
                result.coeff_error_rel_inf,
                result.coeff_error_rel_1,
                result.coeff_error_rel_2,
            )
            @printf(
                "        point err = %.3e, exact trunc err = %.3e\n",
                result.err_inf,
                result.exact_trunc_error_inf,
            )
            @printf(
                "        point/coeff_l1 = %.3e, point/coeff_inf = %.3e\n",
                result.point_error_over_coeff_1,
                result.point_error_over_coeff_inf,
            )
        end

        if N == 100
            idx = sortperm(abs.(result.coeff_err), rev=true)
            topk = min(10, length(idx))

            println("Top coefficient errors for N=100 (rhs_mode=$(rhs_mode)):")

            for r in 1:topk
                j = idx[r] - 1
                @printf(
                    "  j = %3d | cu = %.16e | c_exact = %.16e | err = %.3e\n",
                    j,
                    result.cu[idx[r]],
                    result.c_exact[idx[r]],
                    result.coeff_err[idx[r]],
                )
            end
        end

        if N == sanity_N
            println("------------------------------------------------------------")
            @printf("N=%d sanity residual_inf = %.6e\n", N, result.residual_inf)
            @printf("N=%d sanity err_inf      = %.6e\n", N, result.err_inf)
            @printf("N=%d sanity cond(A)      = %.6e\n", N, result.cond_A)
            println("------------------------------------------------------------")
        end
    end

    return (
        Ns = collect(Ns_to_run),
        errors = errors,
        bigtruth_errors = bigtruth_errors,
        residuals = residuals,
        conds = conds,
        exact_residual_abs = exact_residual_abs,
        exact_residual_rel = exact_residual_rel,
        exact_coeff_errors = exact_coeff_errors,
        coeff_error_infs = coeff_error_infs,
        coeff_error_1s = coeff_error_1s,
        coeff_error_2s = coeff_error_2s,
        coeff_error_rel_infs = coeff_error_rel_infs,
        coeff_error_rel_1s = coeff_error_rel_1s,
        coeff_error_rel_2s = coeff_error_rel_2s,
        exact_trunc_errors = exact_trunc_errors,
        point_error_over_coeff_1s = point_error_over_coeff_1s,
        point_error_over_coeff_infs = point_error_over_coeff_infs,
        rhs_mode = rhs_mode,
        rhs_info = rhs_data.info,
        rhs_coeffs = rhs_data.coeffs,
        kernel_coeffs = kernels,
        A100 = A100,
    )
end


# -----------------------------------------------------------------------------
# Timed run output
# -----------------------------------------------------------------------------

const SOLVE_REPEATS = 3
const NTEST_ERROR_GRID = 2001
const OUTPUT_CSV_NAME = "ex3_error_time.csv"
const OUTPUT_MAT_NAME = "ex3_A_Nmax.mat"

function assemble_system_for_run(
    N::Integer,
    q;
    tol_volt::Float64 = tol_volt,
    tol_cheb::Float64 = tol_cheb,
    rhs_coeffs = nothing,
    kernel_coeffs = nothing,
    m_coeffs = nothing,
)
    N >= 0 || throw(ArgumentError("N must be nonnegative."))
    0 < q < 1 || throw(ArgumentError("q must lie in (0, 1)."))

    Rq = restmat_cheb(-1.0, 2.0 * q - 1.0, N)

    m_coeffs_local = if m_coeffs === nothing
        tmp, _ = cheb_coeffs_adaptive(
            X -> m_q_fun(X, q);
            tol = tol_cheb,
            maxdeg = 4096,
            M0 = 128,
        )
        tmp
    else
        m_coeffs
    end

    Mmq = chebmul_matrix_from_coeffs(m_coeffs_local, N)

    kernels =
        kernel_coeffs === nothing ?
        build_volterra_kernel_coeffs(q; tol=tol_machine, maxdeg=1024) :
        kernel_coeffs

    V1 = VoltMat_Cheb.voltmat_square_from_coeffmat_fft_convolution(kernels.C1, N; tol=tol_volt)
    V2 = VoltMat_Cheb.voltmat_square_from_coeffmat_fft_convolution(kernels.C2, N; tol=tol_volt)

    I_N = Matrix{Float64}(I, N + 1, N + 1)
    A = I_N - q * Mmq * Rq + Matrix(V1) - q * Matrix(V2) * Rq

    c_g = if rhs_coeffs === nothing
        c_rhs_raw, _ = cheb_coeffs_adaptive(
            X -> rhs_standard(X, q);
            tol = tol_cheb,
            maxdeg = 4096,
            M0 = 256,
        )
        coeffs_to_degree(c_rhs_raw, N)
    else
        coeffs_to_degree(rhs_coeffs, N)
    end

    return (
        A = A,
        rhs = c_g,
        Rq = Rq,
        Mmq = Mmq,
        V1 = V1,
        V2 = V2,
        kernel_coeffs = kernels,
    )
end

function solve_repeated(A, rhs; repeats::Integer = SOLVE_REPEATS)
    repeats >= 1 || throw(ArgumentError("repeats must be positive."))

    times = Vector{Float64}(undef, repeats)
    cu = similar(rhs)

    for r in 1:repeats
        local tmp
        times[r] = @elapsed begin
            tmp = A \ rhs
        end
        cu = tmp
    end

    return cu, minimum(times), median(times), times
end

function solve_for_N_timed(
    N::Integer,
    q;
    tol_volt::Float64 = tol_volt,
    tol_cheb::Float64 = tol_cheb,
    rhs_coeffs,
    kernel_coeffs,
    m_coeffs,
    c_exact_all,
    Xtest,
    U_exact,
    U_exact_big,
    solve_repeats::Integer = SOLVE_REPEATS,
)
    local sys
    assembly_time = @elapsed begin
        sys = assemble_system_for_run(
            N,
            q;
            tol_volt = tol_volt,
            tol_cheb = tol_cheb,
            rhs_coeffs = rhs_coeffs,
            kernel_coeffs = kernel_coeffs,
            m_coeffs = m_coeffs,
        )
    end

    cu, solve_time_min, solve_time_median, solve_times = solve_repeated(
        sys.A,
        sys.rhs;
        repeats = solve_repeats,
    )

    c_exact = coeffs_to_degree(c_exact_all, N)
    coeff_err = cu - c_exact
    coeff_error_inf = norm(coeff_err, Inf)
    coeff_error_1 = norm(coeff_err, 1)
    coeff_error_2 = norm(coeff_err, 2)

    U_num = cheb_eval_many(cu, Xtest)
    err_inf = maximum(abs.(U_num .- U_exact))
    err_inf_bigtruth = maximum(abs.(U_num .- U_exact_big))
    residual_inf = norm(sys.A * cu - sys.rhs, Inf)

    return (
        N = N,
        A = sys.A,
        rhs = sys.rhs,
        cu = cu,
        err_inf = err_inf,
        err_inf_bigtruth = err_inf_bigtruth,
        coeff_error_inf = coeff_error_inf,
        coeff_error_1 = coeff_error_1,
        coeff_error_2 = coeff_error_2,
        residual_inf = residual_inf,
        assembly_time = assembly_time,
        solve_time_min = solve_time_min,
        solve_time_median = solve_time_median,
        solve_times = solve_times,
        total_time = assembly_time + solve_time_median,
    )
end

function write_error_time_csv(path::AbstractString, rows; rhs_mode::Symbol, q, kernel_coeffs)
    open(path, "w") do io
        println(
            io,
            "N,q,rhs_mode,err_inf,err_inf_bigtruth,coeff_error_inf,coeff_error_1,coeff_error_2,residual_inf,assembly_time_sec,solve_time_sec,solve_time_min_sec,total_time_sec,solve_repeats,C1_rows,C1_cols,C2_rows,C2_cols,C1_interp_err,C2_interp_err",
        )

        for row in rows
            @printf(
                io,
                "%d,%.17e,%s,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%d,%d,%d,%d,%d,%.17e,%.17e\n",
                row.N,
                q,
                string(rhs_mode),
                row.err_inf,
                row.err_inf_bigtruth,
                row.coeff_error_inf,
                row.coeff_error_1,
                row.coeff_error_2,
                row.residual_inf,
                row.assembly_time,
                row.solve_time_median,
                row.solve_time_min,
                row.total_time,
                length(row.solve_times),
                size(kernel_coeffs.C1, 1),
                size(kernel_coeffs.C1, 2),
                size(kernel_coeffs.C2, 1),
                size(kernel_coeffs.C2, 2),
                kernel_coeffs.err_C1,
                kernel_coeffs.err_C2,
            )
        end
    end

    return path
end

# Minimal MATLAB Level-4 writer for real Float64 matrices.  This avoids an
# additional MAT.jl dependency and can be read by MATLAB/Octave with load(...).
function write_mat_v4_matrix(io, name::AbstractString, A)
    A64 = Matrix{Float64}(A)
    name_bytes = Vector{UInt8}(codeunits(name))

    write(io, Int32(0))                  # MOPT: little-endian IEEE, Float64, full matrix
    write(io, Int32(size(A64, 1)))       # mrows
    write(io, Int32(size(A64, 2)))       # ncols
    write(io, Int32(0))                  # imagf = 0
    write(io, Int32(length(name_bytes) + 1))
    write(io, name_bytes)
    write(io, UInt8(0))
    write(io, vec(A64))                  # Julia is column-major, same as MATLAB

    return nothing
end

function write_mat_v4(path::AbstractString, vars::Pair...)
    open(path, "w") do io
        for var in vars
            write_mat_v4_matrix(io, String(var.first), var.second)
        end
    end

    return path
end

function print_run_header(q, Ns_to_run, rhs_mode)
    println("============================================================")
    println("Pantograph first-kind Volterra equation in Chebyshev basis")
    @printf("q = %.16g\n", q)
    @printf("rhs_mode = %s\n", string(rhs_mode))
    @printf("N count = %d, min N = %d, max N = %d\n", length(Ns_to_run), minimum(Ns_to_run), maximum(Ns_to_run))
    @printf("tol_volt = %.3e, tol_cheb = %.3e\n", tol_volt, tol_cheb)
    @printf("solve repeats per N = %d\n", SOLVE_REPEATS)
    println("Output directory: ", DATA_DIR)
    println("============================================================")
end

function main()
    mkpath(DATA_DIR)

    q = q_default
    Ns_to_run = collect(Ns)
    isempty(Ns_to_run) && throw(ArgumentError("Ns must not be empty."))
    maxN = maximum(Ns_to_run)
    rhs_mode = use_high_precision_rhs ? :big : :float64

    print_run_header(q, Ns_to_run, rhs_mode)

    println("High-precision RHS setup")
    @printf("Complex{BigFloat} erf supported = %s\n", string(HAS_COMPLEX_BIG_ERF))
    if !HAS_COMPLEX_BIG_ERF
        @printf("Using BigFloat Gauss-Legendre fallback with nquad = %d\n", rhs_quad_n)
    end

    dct_norm_diff = check_big_cheb_coeffs_normalization()
    @printf("Big DCT-I normalization check = %.3e\n", dct_norm_diff)

    println("------------------------------------------------------------")
    println("Computing kernel coefficients ...")
    kernel_coeffs = build_volterra_kernel_coeffs(q; tol=tol_machine, maxdeg=1024)

    println("------------------------------------------------------------")
    println("Computing RHS coefficients ...")
    rhs_data = compute_rhs_coeffs(
        q,
        rhs_mode;
        maxN = maxN,
        tol = tol_cheb,
        precision = rhs_precision,
    )
    @printf("rhs Chebyshev degree after chop = %d\n", rhs_data.info.degree)
    @printf("rhs sample degree               = %d\n", rhs_data.info.sample_degree)
    @printf("rhs tail                        = %.6e\n", rhs_data.info.tail)

    println("------------------------------------------------------------")
    println("Computing cached multiplier and exact solution coefficients ...")
    m_coeffs, _ = cheb_coeffs_adaptive(
        X -> m_q_fun(X, q);
        tol = tol_cheb,
        maxdeg = 4096,
        M0 = 128,
    )
    c_exact_all = exact_solution_coeffs(maxN; tol=tol_machine)

    Xtest = collect(range(-1.0, 1.0; length=NTEST_ERROR_GRID))
    U_exact = utrue_standard.(Xtest)
    U_exact_big = [utrue_standard_big(X; precision=rhs_precision) for X in Xtest]

    println("------------------------------------------------------------")
    println("Warmup run, not recorded ...")
    _ = solve_for_N_timed(
        first(Ns_to_run),
        q;
        tol_volt = tol_volt,
        tol_cheb = tol_cheb,
        rhs_coeffs = rhs_data.coeffs,
        kernel_coeffs = kernel_coeffs,
        m_coeffs = m_coeffs,
        c_exact_all = c_exact_all,
        Xtest = Xtest,
        U_exact = U_exact,
        U_exact_big = U_exact_big,
        solve_repeats = 1,
    )

    println("------------------------------------------------------------")
    println("Running sweep ...")
    rows = Vector{Any}()
    max_result = nothing

    for N in Ns_to_run
        result = solve_for_N_timed(
            N,
            q;
            tol_volt = tol_volt,
            tol_cheb = tol_cheb,
            rhs_coeffs = rhs_data.coeffs,
            kernel_coeffs = kernel_coeffs,
            m_coeffs = m_coeffs,
            c_exact_all = c_exact_all,
            Xtest = Xtest,
            U_exact = U_exact,
            U_exact_big = U_exact_big,
            solve_repeats = SOLVE_REPEATS,
        )

        push!(rows, result)

        if N == maxN
            max_result = result
        end

        @printf(
            "N = %4d | err_inf = %.6e | solve = %.6e s | assembly = %.6e s | residual = %.3e\n",
            N,
            result.err_inf,
            result.solve_time_median,
            result.assembly_time,
            result.residual_inf,
        )
    end

    max_result === nothing && error("Largest N result was not recorded.")

    csv_path = joinpath(DATA_DIR, OUTPUT_CSV_NAME)
    mat_path = joinpath(DATA_DIR, OUTPUT_MAT_NAME)

    write_error_time_csv(csv_path, rows; rhs_mode=rhs_mode, q=q, kernel_coeffs=kernel_coeffs)

    write_mat_v4(
        mat_path,
        "A" => max_result.A,
        "rhs" => reshape(max_result.rhs, :, 1),
        "u_coeff" => reshape(max_result.cu, :, 1),
        "Nmax" => reshape([Float64(maxN)], 1, 1),
        "q" => reshape([Float64(q)], 1, 1),
    )

    println("============================================================")
    println("Saved:")
    println("  ", csv_path)
    println("  ", mat_path)
    println("MAT variables: A, rhs, u_coeff, Nmax, q")
    println("============================================================")

    return (
        csv_path = csv_path,
        mat_path = mat_path,
        rows = rows,
        A_Nmax = max_result.A,
    )
end

main()
