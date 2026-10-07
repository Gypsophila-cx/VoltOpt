#!/usr/bin/env julia

# ex6_nonlinear_picard.jl
#
# Purpose:
#   Run the main Example 6 Picard iteration experiment and write the iteration
#   history CSV.
#
# Problem:
#   Nonlinear saturated-memory Volterra--Urysohn equation.
#
# The kernel
#
#     K(x,t,u) = u / (1 + (x - t)u^2)
#
# is a saturated-memory nonlinearity: as the lag x-t or |u| grows, the
# denominator damps the remembered contribution. Picard iteration therefore only
# needs repeated nonlinear Volterra actions with the current iterate.
#
# Method:
#   Builds the fixed manufactured right-hand side independently with adaptive
#   Gauss--Legendre quadrature, then uses adaptive compVolt actions inside a
#   Picard iteration whose working degree is refined without advancing the
#   nonlinear state when a complete update is unresolved.
#
# Output files:
#   data/ex6_nonlinear_picard_results.csv
#
# Run from the repository root:
#   julia --project=. examples/ex6_nonlinear/ex6_nonlinear_picard.jl

using LinearAlgebra
using Printf
using FastGaussQuadrature

const α = 100.0
const λ = 1.0
const oracle_tol = eps(Float64)
const tol_quad = 100eps(Float64)
const tol_chop = 10eps(Float64)
const tol_rhs = 10eps(Float64)
const N0 = 32
const Nmax = 4096
const Mmax = 8192
const max_refinements = 8
const maxit = 100
const M_error_eval = 1000
const rhs_q0 = 16
const rhs_qmax = 4096
const rhs_Nmax = Nmax

function _candidate_compvolt_paths(root::AbstractString)
    return (
        joinpath(root, "src", "compVolt.jl"),
        joinpath(root, "src", "CompVolt", "compVolt.jl"),
    )
end

function _find_compvolt_file(root::AbstractString)
    for path in _candidate_compvolt_paths(root)
        isfile(path) && return path
    end
    return nothing
end

function find_project_root(start_dir::AbstractString=@__DIR__)
    dir = abspath(start_dir)
    while true
        _find_compvolt_file(dir) !== nothing && return dir
        parent = dirname(dir)
        parent == dir && error("Could not find a project root containing canonical src/CompVolt/compVolt.jl.")
        dir = parent
    end
end

const project_root = find_project_root()
const compvolt_file = _find_compvolt_file(project_root)

# The legacy lowercase src/compVolt.jl path remains accepted for older
# checkouts, while src/CompVolt/compVolt.jl is canonical in this tree.
include(compvolt_file)

function lambertw_exp(s::Float64; maxiter::Int=100)
    if !(isfinite(s))
        @warn "lambertw_exp received a non-finite input; returning NaN." s
        return NaN
    end
    if s < -α || s > α
        @warn "lambertw_exp was designed for s in [-100, 100]; continuing anyway." s
    end

    # Solve w + log(w) = s, which is equivalent to w = W(exp(s)).
    # For s <= 1, exp(s) is safe and gives a positive starting point. For large
    # positive s, s - log(s) is the standard asymptotic seed.
    w = s <= 1.0 ? exp(s) : s - log(s)
    w = max(w, floatmin(Float64))

    converged = false
    for _ in 1:maxiter
        f = w + log(w) - s
        if abs(f) <= oracle_tol * max(1.0, abs(s), abs(w))
            converged = true
            break
        end
        fp = 1.0 + inv(w)
        fpp = -inv(w * w)
        denom = 2.0 * fp * fp - f * fpp
        step = isfinite(denom) && denom != 0.0 ? (2.0 * f * fp / denom) : (f / fp)
        w_next = w - step

        if !(isfinite(w_next)) || w_next <= 0.0
            w_next = max(w / 2.0, floatmin(Float64))
        end

        if abs(w_next - w) <= oracle_tol * max(1.0, abs(w_next))
            w = w_next
            converged = true
            break
        end
        w = w_next
    end

    if !converged
        @warn "lambertw_exp did not reach machine precision before maxiter." s w residual=(w + log(w) - s)
    end
    return w
end

# With w = W(exp(αx)) and u = sqrt(w), differentiating w + log(w) = αx gives
# u'(x) = (α/2) * u(x) / (1 + u(x)^2).
uexact(x::Real) = sqrt(lambertw_exp(Float64(α * x)))

Kfun(x, t, u) = u / (1.0 + (x - t) * u^2)

function ascending_values_to_cheb_coeffs(vals::AbstractVector{<:Real})
    coeffs = vals2cheb(vals)
    @inbounds for i in 2:2:length(coeffs)
        coeffs[i] = -coeffs[i]
    end
    return coeffs
end

function cheb_eval(c::AbstractVector, x::Real)
    n = length(c) - 1
    b1 = zero(promote_type(eltype(c), typeof(x)))
    b2 = zero(promote_type(eltype(c), typeof(x)))
    @inbounds for k in n:-1:1
        b0 = 2.0 * x * b1 - b2 + c[k + 1]
        b2 = b1
        b1 = b0
    end
    return x * b1 - b2 + c[1]
end

function cheb_eval_many(c::AbstractVector, xs)
    vals = Vector{Float64}(undef, length(xs))
    @inbounds for i in eachindex(xs)
        vals[i] = Float64(cheb_eval(c, xs[i]))
    end
    return vals
end

function pad_to_degree(c::AbstractVector, N::Int)
    N >= 0 || throw(ArgumentError("N must be nonnegative."))
    length(c) <= N + 1 || throw(ArgumentError(
        "Cannot pad a degree-$(length(c) - 1) polynomial to the smaller degree N=$N.",
    ))
    padded = zeros(eltype(c), N + 1)
    copyto!(padded, 1, c, 1, length(c))
    return padded
end

function strict_tail_chop(
    c_full::AbstractVector,
    N::Int,
    tolerance::Real,
    scale::Real;
    tail_margin::Int=8,
)
    N >= 0 || throw(ArgumentError("N must be nonnegative."))
    length(c_full) >= N + 1 || throw(ArgumentError(
        "The full coefficient vector must contain all degrees through N=$N.",
    ))
    tolerance >= 0 || throw(ArgumentError("The chopping tolerance must be nonnegative."))
    tail_margin >= 1 || throw(ArgumentError("tail_margin must be positive."))

    threshold = Float64(tolerance) * max(1.0, Float64(scale))
    magnitudes = abs.(Float64.(c_full))
    outside_max = length(c_full) > N + 1 ?
        maximum(@view magnitudes[(N + 2):end]) : 0.0
    outside_tail = length(c_full) > N + 1 ?
        sum(@view magnitudes[(N + 2):end]) : 0.0

    if all(iszero, magnitudes)
        return (
            resolved=true,
            cutoff=1,
            plateau=true,
            threshold=threshold,
            outside_max=outside_max,
            outside_tail=outside_tail,
            tail=0.0,
            margin_coefficients=N + 1,
        )
    end

    tail_sum = similar(magnitudes)
    running = 0.0
    @inbounds for i in length(magnitudes):-1:1
        running += magnitudes[i]
        tail_sum[i] = running
    end

    plateau_index = findfirst(x -> x <= threshold, tail_sum)
    cutoff = plateau_index === nothing ? min(length(c_full), N + 1) :
        max(1, plateau_index - 1)
    margin_coefficients = cutoff <= N + 1 ? N + 1 - cutoff : 0
    margin_ok = margin_coefficients >= tail_margin
    outside_ok = outside_tail <= threshold
    resolved = plateau_index !== nothing && margin_ok && outside_ok
    tail = cutoff < length(c_full) ? tail_sum[cutoff + 1] : 0.0

    return (
        resolved=resolved,
        cutoff=cutoff,
        plateau=plateau_index !== nothing,
        threshold=threshold,
        outside_max=outside_max,
        outside_tail=outside_tail,
        tail=tail,
        margin_coefficients=margin_coefficients,
    )
end

function aligned_picard_update(
    c_g::AbstractVector,
    c_volterra::AbstractVector,
    c_old::AbstractVector,
    λ::Real,
)
    ncoeffs = max(length(c_g), length(c_volterra), length(c_old))
    T = promote_type(eltype(c_g), eltype(c_volterra), eltype(c_old), typeof(λ))
    candidate = zeros(T, ncoeffs)
    delta = zeros(T, ncoeffs)

    @inbounds for i in 1:ncoeffs
        g_i = i <= length(c_g) ? c_g[i] : zero(T)
        v_i = i <= length(c_volterra) ? c_volterra[i] : zero(T)
        old_i = i <= length(c_old) ? c_old[i] : zero(T)
        candidate[i] = g_i + λ * v_i
        delta[i] = candidate[i] - old_i
    end
    return candidate, delta
end

function gauss_legendre_rule(q::Int, rule_cache)
    q >= 1 || throw(ArgumentError("The Gauss--Legendre order must be positive."))
    if rule_cache === nothing
        return gausslegendre(q)
    end
    return get!(rule_cache, q) do
        gausslegendre(q)
    end
end

function gauss_legendre_integral(f, a::Real, b::Real, q::Int, rule_cache=nothing)
    a == b && return 0.0
    nodes, weights = gauss_legendre_rule(q, rule_cache)
    half_length = (Float64(b) - Float64(a)) / 2.0
    midpoint = (Float64(a) + Float64(b)) / 2.0
    integral = 0.0
    @inbounds for i in eachindex(nodes)
        value = f(midpoint + half_length * nodes[i])
        isfinite(value) || error("Non-finite integrand value in reference Gauss--Legendre quadrature.")
        integral += weights[i] * value
    end
    integral *= half_length
    isfinite(integral) || error("Non-finite reference Gauss--Legendre integral.")
    return integral
end

function adaptive_gauss_legendre(
    f,
    a::Real,
    b::Real;
    tolerance::Real,
    q0::Int=16,
    qmax::Int=4096,
    rule_cache=nothing,
)
    isfinite(tolerance) && tolerance >= 0 || throw(ArgumentError(
        "The Gauss--Legendre tolerance must be finite and nonnegative.",
    ))
    q0 >= 1 || throw(ArgumentError("q0 must be positive."))
    qmax >= 2q0 || throw(ArgumentError("qmax must permit at least one q-to-2q comparison."))
    a == b && return 0.0, 0.0, 0

    q = q0
    coarse = gauss_legendre_integral(f, a, b, q, rule_cache)
    while true
        qfine = 2q
        qfine <= qmax || error(
            "Reference Gauss--Legendre quadrature did not resolve before qmax=$qmax; " *
            "last completed order was q=$q.",
        )
        fine = gauss_legendre_integral(f, a, b, qfine, rule_cache)
        indicator = abs(fine - coarse) / max(1.0, abs(fine))
        if indicator <= tolerance
            return fine, indicator, qfine
        end
        q = qfine
        coarse = fine
    end
end

function chop_candidate_preserving_old(
    candidate::AbstractVector,
    suggested_cutoff::Int,
    minimum_length::Int,
)
    isempty(candidate) && return copy(candidate)
    1 <= minimum_length <= length(candidate) || throw(ArgumentError(
        "minimum_length must lie between 1 and the candidate length.",
    ))
    cutoff = clamp(max(suggested_cutoff, minimum_length), 1, length(candidate))
    return copy(@view candidate[1:cutoff])
end

function nested_chebyshev_nodes(N::Int, coarse_nodes=nothing)
    N >= 1 || throw(ArgumentError("The Chebyshev degree must be positive."))
    nodes = [-cospi(i / N) for i in 0:N]
    if coarse_nodes !== nothing
        N == 2(length(coarse_nodes) - 1) || throw(ArgumentError(
            "The supplied coarse nodes must have half the requested degree.",
        ))
        @views nodes[1:2:end] .= coarse_nodes
    end
    return nodes
end

function fixed_rhs_value(x::Float64, rule_cache)
    x == -1.0 && return uexact(x), 0.0, 0
    integrand(t) = Kfun(x, t, uexact(t))
    integral, indicator, q_used = adaptive_gauss_legendre(
        integrand,
        -1.0,
        x;
        tolerance=tol_rhs,
        q0=rhs_q0,
        qmax=rhs_qmax,
        rule_cache=rule_cache,
    )
    return uexact(x) - λ * integral, indicator, q_used
end

function build_fixed_rhs()
    build_t0 = time()
    rule_cache = Dict{Int,Tuple{Vector{Float64},Vector{Float64}}}()
    degree = N0
    nodes = nested_chebyshev_nodes(degree)
    values = Vector{Float64}(undef, degree + 1)
    max_quad_indicator = 0.0
    max_q_used = 0
    @inbounds for i in eachindex(nodes)
        values[i], indicator, q_used = fixed_rhs_value(nodes[i], rule_cache)
        max_quad_indicator = max(max_quad_indicator, indicator)
        max_q_used = max(max_q_used, q_used)
    end
    coefficients = ascending_values_to_cheb_coeffs(values)
    last_consistency = Inf
    last_tail = strict_tail_chop(
        coefficients,
        degree,
        tol_chop,
        max(1.0, maximum(abs, values), maximum(abs, coefficients)),
    )

    while true
        fine_degree = 2degree
        fine_degree <= rhs_Nmax || error(
            "Fixed right-hand-side construction failed before rhs_Nmax=$rhs_Nmax: " *
            "last degree=$degree, consistency=$last_consistency, " *
            "tail=$(last_tail.tail), threshold=$(last_tail.threshold), " *
            "max quadrature indicator=$max_quad_indicator.",
        )

        fine_nodes = nested_chebyshev_nodes(fine_degree, nodes)
        fine_values = Vector{Float64}(undef, fine_degree + 1)
        @views fine_values[1:2:end] .= values
        @inbounds for i in 2:2:length(fine_nodes)
            fine_values[i], indicator, q_used = fixed_rhs_value(fine_nodes[i], rule_cache)
            max_quad_indicator = max(max_quad_indicator, indicator)
            max_q_used = max(max_q_used, q_used)
        end

        fine_coefficients = ascending_values_to_cheb_coeffs(fine_values)
        rhs_scale = max(
            1.0,
            maximum(abs, fine_values),
            maximum(abs, fine_coefficients),
        )
        coarse_on_fine = cheb_eval_many(coefficients, fine_nodes)
        consistency = maximum(abs.(coarse_on_fine .- fine_values)) / rhs_scale
        tail = strict_tail_chop(
            fine_coefficients, fine_degree, tol_chop, rhs_scale,
        )

        if consistency <= tol_rhs && tail.resolved
            c_g = copy(@view fine_coefficients[1:tail.cutoff])
            return c_g, (
                rhs_build_time_sec=time() - build_t0,
                degree=fine_degree,
                N_eff=length(c_g) - 1,
                consistency=consistency,
                tail=tail.tail,
                threshold=tail.threshold,
                max_quad_indicator=max_quad_indicator,
                max_q_used=max_q_used,
            )
        end

        degree = fine_degree
        nodes = fine_nodes
        values = fine_values
        coefficients = fine_coefficients
        last_consistency = consistency
        last_tail = tail
    end
end

function fixed_rhs_representation_check(
    c_g::AbstractVector,
    x_error_eval::AbstractVector,
)
    rule_cache = Dict{Int,Tuple{Vector{Float64},Vector{Float64}}}()
    error_inf = 0.0
    max_quad_indicator = 0.0
    max_q_used = 0
    @inbounds for x in x_error_eval
        reference, indicator, q_used = fixed_rhs_value(Float64(x), rule_cache)
        error_inf = max(error_inf, abs(cheb_eval(c_g, x) - reference))
        max_quad_indicator = max(max_quad_indicator, indicator)
        max_q_used = max(max_q_used, q_used)
    end
    return (
        error_inf=error_inf,
        max_quad_indicator=max_quad_indicator,
        max_q_used=max_q_used,
    )
end

function independent_residual_check(
    c_num::AbstractVector,
    c_g::AbstractVector,
    x_residual_check::AbstractVector,
)
    rule_cache = Dict{Int,Tuple{Vector{Float64},Vector{Float64}}}()
    residual_inf = 0.0
    max_quad_indicator = 0.0
    max_q_used = 0
    @inbounds for x in x_residual_check
        if x == -1.0
            integral = 0.0
            indicator = 0.0
            q_used = 0
        else
            integrand(t) = Kfun(x, t, cheb_eval(c_num, t))
            integral, indicator, q_used = adaptive_gauss_legendre(
                integrand,
                -1.0,
                x;
                tolerance=tol_quad,
                q0=rhs_q0,
                qmax=rhs_qmax,
                rule_cache=rule_cache,
            )
        end
        residual = cheb_eval(c_num, x) - cheb_eval(c_g, x) - λ * integral
        residual_inf = max(residual_inf, abs(residual))
        max_quad_indicator = max(max_quad_indicator, indicator)
        max_q_used = max(max_q_used, q_used)
    end
    return (
        residual_inf=residual_inf,
        max_quad_indicator=max_quad_indicator,
        max_q_used=max_q_used,
    )
end

function write_history_csv(path::AbstractString, history)
    headers = [
        "N",
        "iter",
        "residual",
        "residual_relative",
        "err_inf",
        "max_abs_error",
        "solve_time",
        "elapsed_time",
        "status",
        "update_err",
        "node_err",
        "eqgrid_err",
        "action_time_sec",
        "total_elapsed_sec",
        "intermediate_iter",
        "N_eff",
        "M",
        "eta",
        "theta",
        "refinements",
        "plateau",
        "accepted",
        "rhs_build_time_sec",
    ]
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(headers, ","))
        for h in history
            cells = map(headers) do header
                value = getfield(h, Symbol(header))
                value isa AbstractFloat ? @sprintf("%.17e", value) : string(value)
            end
            println(io, join(cells, ","))
        end
    end
end

function relprint(path::AbstractString)
    return replace(relpath(path, project_root), '\\' => '/')
end

function adaptive_picard_solve(
    c_g::AbstractVector,
    x_error_eval::AbstractVector,
    u_exact_error_eval::AbstractVector,
    rhs_build_time_sec::Real,
)
    c_old = zeros(Float64, N0 + 1)
    copyto!(c_old, 1, c_g, 1, min(length(c_g), length(c_old)))
    history = NamedTuple[]
    N = N0
    iter = 0
    intermediate_iter = 0
    ρ0 = nothing
    stopping_resolution_relative = nothing
    converged = false
    failure_reason = ""
    picard_t0 = time()

    while true
        u_old_error_eval = cheb_eval_many(c_old, x_error_eval)
        eqgrid_err = maximum(abs.(u_old_error_eval .- u_exact_error_eval))
        x_nodes = nested_chebyshev_nodes(N)
        node_err = maximum(abs.(cheb_eval_many(c_old, x_nodes) .- uexact.(x_nodes)))

        cu = pad_to_degree(c_old, N)
        action_error = nothing
        info = nothing
        action_t0 = time()
        try
            info = compVolt_nonlinear_fast(
                Kfun,
                cu;
                adaptive=true,
                M0=N,
                tol=tol_quad,
                Mmax=Mmax,
                max_refinements=max_refinements,
                return_info=true,
                truncate_output=false,
            )
        catch err
            action_error = err
        end
        action_time_sec = time() - action_t0

        if action_error !== nothing || !info.converged
            M = info === nothing ? 0 : info.M
            eta = info === nothing || info.eta === nothing ? NaN : Float64(info.eta)
            theta = info === nothing || info.theta === nothing ? NaN : Float64(info.theta)
            refinements = info === nothing ? 0 : info.refinements
            total_elapsed_sec = time() - picard_t0
            push!(history, (
                N=N, iter=iter, residual=NaN, residual_relative=NaN,
                err_inf=eqgrid_err, max_abs_error=node_err,
                solve_time=action_time_sec, elapsed_time=total_elapsed_sec,
                status="quadrature_failed", update_err=NaN, node_err=node_err,
                eqgrid_err=eqgrid_err, action_time_sec=action_time_sec,
                total_elapsed_sec=total_elapsed_sec,
                intermediate_iter=intermediate_iter, N_eff=length(c_old) - 1,
                M=M, eta=eta, theta=theta, refinements=refinements,
                plateau=false, accepted=false,
                rhs_build_time_sec=Float64(rhs_build_time_sec),
            ))
            failure_reason = action_error === nothing ?
                "adaptive quadrature did not converge at N=$N, M=$M, eta=$eta, theta=$theta" :
                sprint(showerror, action_error)
            break
        end

        @assert info.eta !== nothing && info.theta !== nothing
        @assert max(info.eta, info.theta) <= tol_quad
        candidate, delta_full = aligned_picard_update(
            c_g, info.coefficients, c_old, λ,
        )
        update_scale = max(
            1.0,
            maximum(abs, c_old),
            maximum(abs, c_g),
            maximum(abs, candidate),
        )
        tail = strict_tail_chop(delta_full, N, tol_chop, update_scale)
        # The coefficient l1 norm bounds the Chebyshev sup norm because |T_j(x)| <= 1.
        residual = sum(abs, delta_full)
        residual_scale = max(
            1.0,
            sum(abs, c_old),
            sum(abs, candidate),
        )
        residual_relative = residual / residual_scale

        status = "accepted"
        accepted = false
        if !tail.resolved
            status = N >= Nmax ? "Nmax" : "refine_N"
        else
            tail_resolution_relative = tail.threshold / update_scale
            action_resolution_relative = abs(λ) * max(
                Float64(info.eta),
                Float64(info.theta),
            )
            stopping_resolution_relative = max(
                tail_resolution_relative,
                action_resolution_relative,
            )
            if ρ0 === nothing
                ρ0 = residual_relative
            end
            if residual_relative <= stopping_resolution_relative
                status = "converged"
            elseif iter >= maxit
                status = "maxit"
            else
                accepted = true
            end
        end

        total_elapsed_sec = time() - picard_t0
        push!(history, (
            N=N, iter=iter, residual=residual,
            residual_relative=residual_relative,
            err_inf=eqgrid_err, max_abs_error=node_err,
            solve_time=action_time_sec, elapsed_time=total_elapsed_sec,
            status=status, update_err=residual, node_err=node_err,
            eqgrid_err=eqgrid_err, action_time_sec=action_time_sec,
            total_elapsed_sec=total_elapsed_sec,
            intermediate_iter=intermediate_iter, N_eff=length(c_old) - 1,
            M=info.M, eta=Float64(info.eta), theta=Float64(info.theta),
            refinements=info.refinements, plateau=tail.plateau,
            accepted=accepted, rhs_build_time_sec=Float64(rhs_build_time_sec),
        ))

        @printf(
            "intermediate = %3d  iter = %3d  N = %4d  N_eff = %4d  M = %4d  residual = %.16e  eqgrid_err = %.16e  status = %s  action = %.6f sec\n",
            intermediate_iter, iter, N, length(c_old) - 1, info.M,
            residual, eqgrid_err, status, action_time_sec,
        )

        if status == "refine_N"
            old_snapshot = copy(c_old)
            iter_snapshot = iter
            N_next = 2N
            @assert length(pad_to_degree(c_old, N_next)) == 2N + 1
            N = N_next
            @assert c_old == old_snapshot
            @assert iter == iter_snapshot
            intermediate_iter += 1
            continue
        elseif status == "converged"
            converged = true
            break
        elseif status == "Nmax"
            failure_reason = (
                "update unresolved at Nmax=$Nmax: tail=$(tail.tail), " *
                "threshold=$(tail.threshold), outside_max=$(tail.outside_max), " *
                "outside_tail=$(tail.outside_tail), " *
                "margin_coefficients=$(tail.margin_coefficients)"
            )
            break
        elseif status == "maxit"
            failure_reason = "maxit=$maxit accepted Picard updates reached before residual convergence"
            break
        end

        c_new = chop_candidate_preserving_old(
            candidate, tail.cutoff, length(c_old),
        )
        @assert length(c_new) <= N + 1
        c_old = c_new
        iter += 1
        intermediate_iter += 1
    end

    return (
        coefficients=c_old,
        history=history,
        accepted_iterations=iter,
        intermediate_attempts=length(history),
        N=history[end].N,
        N_eff=length(c_old) - 1,
        ρ0=ρ0,
        stopping_resolution_relative=stopping_resolution_relative,
        converged=converged,
        failure_reason=failure_reason,
        total_picard_time_sec=time() - picard_t0,
        total_action_time_sec=sum(h.action_time_sec for h in history),
    )
end

function main()
    script_t0 = time()
    data_dir = joinpath(@__DIR__, "data")
    mkpath(data_dir)
    history_csv = joinpath(data_dir, "ex6_nonlinear_picard_results.csv")
    x_error_eval = collect(range(-1.0, 1.0; length=M_error_eval))
    u_exact_error_eval = uexact.(x_error_eval)

    println("============================================================")
    println("Example 6: adaptive nonlinear Picard history")
    @printf("script file              = %s\n", relprint(@__FILE__))
    @printf("α                        = %.1f\n", α)
    @printf("λ                        = %.1f\n", λ)
    @printf("oracle_tol               = %.16e\n", oracle_tol)
    @printf("tol_quad                 = %.16e\n", tol_quad)
    @printf("tol_chop                 = %.16e\n", tol_chop)
    @printf("tol_rhs                  = %.16e\n", tol_rhs)
    @printf("N0, Nmax                 = %d, %d\n", N0, Nmax)
    @printf("Mmax, max_refinements    = %d, %d\n", Mmax, max_refinements)
    @printf("maxit                    = %d\n", maxit)
    @printf("error observation points = %d\n", M_error_eval)
    @printf("rhs q0, qmax, Nmax       = %d, %d, %d\n", rhs_q0, rhs_qmax, rhs_Nmax)
    println("These tolerances are experiment settings, not error guarantees.")
    println("============================================================")

    c_g, rhs_info = build_fixed_rhs()
    result = adaptive_picard_solve(
        c_g, x_error_eval, u_exact_error_eval, rhs_info.rhs_build_time_sec,
    )
    write_history_csv(history_csv, result.history)

    if !result.converged
        @printf("history csv               = %s\n", relprint(history_csv))
        error("Example 6 failed: $(result.failure_reason)")
    end

    x_residual_check = nested_chebyshev_nodes(result.N)
    independent = independent_residual_check(
        result.coefficients, c_g, x_residual_check,
    )
    rhs_check = fixed_rhs_representation_check(c_g, x_error_eval)
    final_row = result.history[end]
    final_eqgrid_err = maximum(abs.(
        cheb_eval_many(result.coefficients, x_error_eval) .- u_exact_error_eval,
    ))
    total_runtime = time() - script_t0

    println("============================================================")
    println("Example 6 completed")
    @printf("rhs build time            = %.6f sec\n", rhs_info.rhs_build_time_sec)
    @printf("rhs degree / N_eff        = %d / %d\n", rhs_info.degree, rhs_info.N_eff)
    @printf("rhs node consistency      = %.16e\n", rhs_info.consistency)
    @printf("rhs coefficient tail      = %.16e (threshold %.16e)\n",
        rhs_info.tail, rhs_info.threshold)
    @printf("rhs max quad indicator    = %.16e (max q %d)\n",
        rhs_info.max_quad_indicator, rhs_info.max_q_used)
    @printf("rhs eqgrid err            = %.16e\n", rhs_check.error_inf)
    @printf("rhs check quad indicator  = %.16e (max q %d)\n",
        rhs_check.max_quad_indicator, rhs_check.max_q_used)
    @printf("initial relative defect ρ0 = %.16e\n", result.ρ0)
    @printf("final stopping resolution = %.16e\n", result.stopping_resolution_relative)
    @printf("accepted Picard updates   = %d\n", result.accepted_iterations)
    @printf("intermediate attempts     = %d\n", result.intermediate_attempts)
    @printf("final N / N_eff / M       = %d / %d / %d\n",
        result.N, result.N_eff, final_row.M)
    @printf("final coefficient defect  = %.16e\n", final_row.residual)
    @printf("final relative defect     = %.16e\n", final_row.residual_relative)
    @printf("independent residual      = %.16e\n", independent.residual_inf)
    @printf("independent quad indicator= %.16e (max q %d)\n",
        independent.max_quad_indicator, independent.max_q_used)
    @printf("final eqgrid err          = %.16e\n", final_eqgrid_err)
    @printf("cumulative action time    = %.6f sec\n", result.total_action_time_sec)
    @printf("total Picard wall time    = %.6f sec\n", result.total_picard_time_sec)
    @printf("total runtime             = %.6f sec\n", total_runtime)
    @printf("history csv               = %s\n", relprint(history_csv))
    println("============================================================")
    return (
        result=result,
        rhs=rhs_info,
        rhs_check=rhs_check,
        independent=independent,
        final_eqgrid_err=final_eqgrid_err,
        total_runtime_sec=total_runtime,
    )
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
