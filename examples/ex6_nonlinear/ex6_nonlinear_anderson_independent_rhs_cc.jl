#!/usr/bin/env julia

# Example 6: nonlinear saturated-memory Volterra--Urysohn equation.
#
# The kernel
#
#     K(x,t,u) = u / (1 + (x - t)u^2)
#
# is a saturated-memory nonlinearity: as the lag x-t or |u| grows, the
# denominator damps the remembered contribution.  The nonlinear equation is
# solved by Picard iteration accelerated with Anderson extrapolation.
#
# Difference from the previous manufactured test:
#     The right-hand side g_lambda is generated independently of compVolt.
#     For each Chebyshev node x_j, we compute
#
#         g_lambda(x_j) = u_*(x_j)
#             - lambda * int_{-1}^{x_j} K(x_j,t,u_*(t)) dt
#
#     by adaptive high-order Clenshaw--Curtis quadrature in Float64, then convert
#     these nodal values to Chebyshev coefficients.  No BigFloat arithmetic is
#     used, and compVolt is not used when constructing g_lambda.

using LinearAlgebra
using Printf
using Plots

const α = 100.0
const λ = 1.0
const N = 1024
const tol = eps(Float64)
const maxit = 100
const M_eval = 4001

# Independent Clenshaw--Curtis quadrature parameters for constructing g_lambda.
# These can be overridden from the shell, e.g.
#   $env:RHS_CC_NMAX="16384"
#   julia --project=. examples/ex6_nonlinear_anderson_independent_rhs_cc.jl
const RHS_CC_NMIN = parse(Int, get(ENV, "RHS_CC_NMIN", "128"))
const RHS_CC_NMAX = parse(Int, get(ENV, "RHS_CC_NMAX", "8192"))
const RHS_CC_RTOL = parse(Float64, get(ENV, "RHS_CC_RTOL", "1e-15"))
const RHS_CC_ATOL = parse(Float64, get(ENV, "RHS_CC_ATOL", "1e-15"))

# Anderson acceleration parameters.
# ANDERSON_M = m uses at most m residual differences, i.e. at most m+1 recent
# fixed-point evaluations in the constrained least-squares problem.
const ANDERSON_M = 5
const ANDERSON_DAMPING = 1.0
const ANDERSON_MAX_STEP_FACTOR = 100.0

const RUN_DIRECT_BASELINE = get(ENV, "RUN_DIRECT_BASELINE", "0") == "1"
const CHECK_COMPVOLT_RHS = get(ENV, "CHECK_COMPVOLT_RHS", "0") == "1"

function savefig_pdf_png(fig, pdf_path::AbstractString)
    png_path = replace(pdf_path, r"\.pdf$" => ".png")
    savefig(fig, pdf_path)
    savefig(fig, png_path)
    return png_path
end

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
        parent == dir && error("Could not find a project root containing src/compVolt.jl or src/CompVolt/compVolt.jl.")
        dir = parent
    end
end

const project_root = find_project_root()
const compvolt_file = _find_compvolt_file(project_root)

# This checkout exposes the nonlinear API from src/CompVolt/compVolt.jl:
#   compVolt_nonlinear_fast(K, cu; grid=CompVoltGrid(N), truncate_output=true)
# The lowercase src/compVolt.jl path is also accepted above for compatibility
# with the path stated in the example description.
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
        if abs(f) <= tol * max(1.0, abs(s), abs(w))
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

        if abs(w_next - w) <= tol * max(1.0, abs(w_next))
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

function nonlinear_volterra_action_compvolt(cu::AbstractVector, grid, K)
    return compVolt_nonlinear_fast(K, cu; grid=grid, truncate_output=true)
end

# Clenshaw--Curtis rule on [-1,1] in Float64.
# The nodes are y_j = cos(j*pi/n), j=0,...,n, and the weights satisfy
#     int_{-1}^1 f(y) dy ≈ sum_j w_j f(y_j).
function clenshaw_curtis_rule(n::Int)
    n >= 1 || error("Clenshaw--Curtis order n must be positive.")

    θ = collect(range(0.0, π; length=n + 1))
    y = cos.(θ)
    w = zeros(Float64, n + 1)

    if n == 1
        w .= 1.0
        return y, w
    end

    ii = 2:n
    v = ones(Float64, n - 1)

    if iseven(n)
        w[1] = 1.0 / (n^2 - 1.0)
        w[end] = w[1]
        for k in 1:(div(n, 2) - 1)
            v .-= (2.0 / (4.0 * k * k - 1.0)) .* cos.(2.0 * k .* θ[ii])
        end
        v .-= (1.0 / (n^2 - 1.0)) .* cos.(n .* θ[ii])
    else
        w[1] = 1.0 / (n^2)
        w[end] = w[1]
        for k in 1:div(n - 1, 2)
            v .-= (2.0 / (4.0 * k * k - 1.0)) .* cos.(2.0 * k .* θ[ii])
        end
    end

    w[ii] .= (2.0 / n) .* v
    return y, w
end

function make_cc_rules(nmin::Int, nmax::Int)
    nmin >= 1 || error("RHS_CC_NMIN must be positive.")
    nmax >= nmin || error("RHS_CC_NMAX must be no smaller than RHS_CC_NMIN.")

    rules = Tuple{Int,Vector{Float64},Vector{Float64}}[]
    n = nmin
    while n <= nmax
        y, w = clenshaw_curtis_rule(n)
        push!(rules, (n, y, w))
        n *= 2
    end
    return rules
end

function independent_volterra_integral_cc(x::Float64, rules)
    # Integral length is zero at the left endpoint.
    if x <= -1.0
        return 0.0, 0, 0.0, true
    end

    # Map y in [-1,1] to t in [-1,x]:
    #     t = mid + half*y,
    # where mid=(x-1)/2 and half=(x+1)/2.
    mid = 0.5 * (x - 1.0)
    half = 0.5 * (x + 1.0)

    previous = NaN
    last_n = 0
    last_diff = Inf

    for (n, y, w) in rules
        acc = 0.0
        @inbounds for j in eachindex(y)
            t = mid + half * y[j]
            ut = uexact(t)
            acc += w[j] * Kfun(x, t, ut)
        end
        current = half * acc
        last_n = n

        if isfinite(previous)
            last_diff = abs(current - previous)
            if last_diff <= RHS_CC_ATOL + RHS_CC_RTOL * max(1.0, abs(current))
                return current, n, last_diff, true
            end
        end
        previous = current
    end

    return previous, last_n, last_diff, false
end

function construct_rhs_independent_cc(x_nodes::AbstractVector{<:Real}, u_exact_vals::AbstractVector{<:Real})
    rules = make_cc_rules(RHS_CC_NMIN, RHS_CC_NMAX)

    g_vals = Vector{Float64}(undef, length(x_nodes))
    volterra_vals = Vector{Float64}(undef, length(x_nodes))
    quad_orders = Vector{Int}(undef, length(x_nodes))
    quad_diffs = Vector{Float64}(undef, length(x_nodes))
    converged = Vector{Bool}(undef, length(x_nodes))

    @inbounds for j in eachindex(x_nodes)
        x = Float64(x_nodes[j])
        I, qn, qdiff, qconv = independent_volterra_integral_cc(x, rules)
        volterra_vals[j] = I
        quad_orders[j] = qn
        quad_diffs[j] = qdiff
        converged[j] = qconv
        g_vals[j] = Float64(u_exact_vals[j]) - λ * I
    end

    not_converged = count(!, converged)
    max_qdiff = maximum(quad_diffs)
    max_qorder = maximum(quad_orders)

    if not_converged > 0
        @warn "Some independent Clenshaw--Curtis RHS integrals did not meet the requested refinement tolerance." not_converged RHS_CC_NMAX RHS_CC_RTOL RHS_CC_ATOL max_qdiff
    end

    return (
        g_vals=g_vals,
        volterra_vals=volterra_vals,
        quad_orders=quad_orders,
        quad_diffs=quad_diffs,
        converged=converged,
        not_converged=not_converged,
        max_qdiff=max_qdiff,
        max_qorder=max_qorder,
    )
end

function direct_trapezoid_baseline(u_vals::AbstractVector, x_nodes::AbstractVector, K)
    out = zeros(Float64, length(x_nodes))
    @inbounds for j in eachindex(x_nodes)
        xj = x_nodes[j]
        acc = 0.0
        for i in 1:(j - 1)
            a = x_nodes[i]
            b = x_nodes[i + 1]
            acc += 0.5 * (b - a) * (K(xj, a, u_vals[i]) + K(xj, b, u_vals[i + 1]))
        end
        out[j] = acc
    end
    return out
end

function fixed_point_map(cu::AbstractVector, c_g::AbstractVector, grid, K)
    V_u = nonlinear_volterra_action_compvolt(cu, grid, K)
    return c_g .+ λ .* V_u
end

function finite_vector(v::AbstractVector)
    return all(isfinite, v)
end

function anderson_extrapolate(g_hist::Vector{Vector{Float64}}, f_hist::Vector{Vector{Float64}}, m::Int)
    nhist = length(f_hist)
    p = min(m + 1, nhist)
    first = nhist - p + 1

    if p == 1
        return copy(g_hist[end]), [1.0]
    end

    # Constrained form:
    #   min ||sum_i θ_i f_i||_2,  subject to sum_i θ_i = 1,
    # followed by c_{new} = sum_i θ_i G(c_i).
    # Eliminating the last coefficient gives a small unconstrained least-square
    # problem with p-1 unknowns.
    f_ref = f_hist[nhist]
    ΔF = Matrix{Float64}(undef, length(f_ref), p - 1)
    @inbounds for col in 1:(p - 1)
        ΔF[:, col] .= f_hist[first + col - 1] .- f_ref
    end

    γ = ΔF \ (-f_ref)
    θ = Vector{Float64}(undef, p)
    θ[1:(p - 1)] .= γ
    θ[p] = 1.0 - sum(γ)

    c_acc = zeros(Float64, length(g_hist[end]))
    @inbounds for q in 1:p
        c_acc .+= θ[q] .* g_hist[first + q - 1]
    end
    return c_acc, θ
end

function safeguarded_anderson_step(c_old::Vector{Float64}, c_picard::Vector{Float64},
        g_hist::Vector{Vector{Float64}}, f_hist::Vector{Vector{Float64}})
    aa_depth = min(ANDERSON_M, length(f_hist) - 1)
    if aa_depth <= 0
        return copy(c_picard), false, 0, [1.0]
    end

    c_acc_raw, θ = anderson_extrapolate(g_hist, f_hist, ANDERSON_M)
    c_acc = (1.0 - ANDERSON_DAMPING) .* c_picard .+ ANDERSON_DAMPING .* c_acc_raw

    picard_step = norm(c_picard - c_old, Inf)
    accel_step = norm(c_acc - c_old, Inf)
    safe_step = accel_step <= ANDERSON_MAX_STEP_FACTOR * max(picard_step, tol)

    if finite_vector(c_acc) && safe_step
        return c_acc, true, aa_depth, θ
    else
        return copy(c_picard), false, aa_depth, θ
    end
end

function write_history_csv(path::AbstractString, history)
    open(path, "w") do io
        println(io, "iter,residual_err,update_err,node_err,eqgrid_err,aa_depth,used_accel,action_time_sec,total_elapsed_sec")
        for h in history
            @printf(io, "%d,%.17e,%.17e,%.17e,%.17e,%d,%d,%.17e,%.17e\n",
                h.iter, h.residual_err, h.update_err, h.node_err, h.eqgrid_err,
                h.aa_depth, Int(h.used_accel), h.action_time_sec, h.total_elapsed_sec)
        end
    end
end

function relprint(path::AbstractString)
    return replace(relpath(path, project_root), '\\' => '/')
end

function main()
    script_t0 = time()

    figures_dir = joinpath(project_root, "examples", "figures")
    mkpath(figures_dir)

    grid = CompVoltGrid(N)
    x_nodes = grid.x
    x_eval = collect(range(-1.0, 1.0; length=M_eval))

    u_exact_vals = uexact.(x_nodes)
    u_exact_eval = uexact.(x_eval)
    c_exact = ascending_values_to_cheb_coeffs(u_exact_vals)

    # Construct g_lambda independently of compVolt.  This is a continuous
    # manufactured RHS sampled on the Chebyshev grid: the Volterra integral is
    # evaluated by adaptive high-order Clenshaw--Curtis quadrature in Float64.
    rhs_t0 = time()
    rhs_data = construct_rhs_independent_cc(x_nodes, u_exact_vals)
    rhs_build_time = time() - rhs_t0
    c_g = ascending_values_to_cheb_coeffs(rhs_data.g_vals)

    if CHECK_COMPVOLT_RHS
        V_exact_compvolt = nonlinear_volterra_action_compvolt(c_exact, grid, Kfun)
        c_g_compvolt = c_exact .- λ .* V_exact_compvolt
        coeff_rhs_diff = norm(c_g - c_g_compvolt, Inf)
        node_rhs_diff = norm(cheb_eval_many(c_g - c_g_compvolt, x_nodes), Inf)
        @printf("RHS check against compVolt-built RHS: coeff_inf = %.16e, node_inf = %.16e\n",
            coeff_rhs_diff, node_rhs_diff)
    end

    if RUN_DIRECT_BASELINE
        u_g_vals = cheb_eval_many(c_g, x_nodes)
        direct_time = @elapsed direct_vals = direct_trapezoid_baseline(u_g_vals, x_nodes, Kfun)
        fast_time = @elapsed fast_coeffs = nonlinear_volterra_action_compvolt(c_g, grid, Kfun)
        fast_vals = cheb_eval_many(fast_coeffs, x_nodes)
        baseline_diff = norm(direct_vals - fast_vals, Inf)
        @printf("direct baseline action_time_sec = %.6f, compVolt action_time_sec = %.6f, nodal_diff = %.16e\n",
            direct_time, fast_time, baseline_diff)
    end

    # Zero-order approximation: ignore the Volterra memory term and start from g.
    c_old = Vector{Float64}(copy(c_g))
    u_old_nodes = cheb_eval_many(c_old, x_nodes)

    # Store recent G(c_i) and residuals f_i = G(c_i) - c_i for Anderson.
    g_hist = Vector{Vector{Float64}}()
    f_hist = Vector{Vector{Float64}}()

    history = NamedTuple[]
    picard_t0 = time()
    converged = false

    for iter in 1:maxit
        action_t0 = time()
        c_picard = Vector{Float64}(fixed_point_map(c_old, c_g, grid, Kfun))
        action_time_sec = time() - action_t0

        f_old = c_picard .- c_old
        residual_nodes = cheb_eval_many(f_old, x_nodes)
        residual_err = norm(residual_nodes, Inf)

        push!(g_hist, copy(c_picard))
        push!(f_hist, copy(f_old))
        if length(f_hist) > ANDERSON_M + 1
            popfirst!(g_hist)
            popfirst!(f_hist)
        end

        if residual_err <= tol * max(1.0, norm(cheb_eval_many(c_picard, x_nodes), Inf))
            c_new = c_picard
            used_accel = false
            aa_depth = 0
        else
            c_new, used_accel, aa_depth, _ = safeguarded_anderson_step(c_old, c_picard, g_hist, f_hist)
        end

        u_new_nodes = cheb_eval_many(c_new, x_nodes)
        u_num_eval = cheb_eval_many(c_new, x_eval)

        update_err = norm(u_new_nodes - u_old_nodes, Inf)
        node_err = norm(u_new_nodes - u_exact_vals, Inf)
        eqgrid_err = maximum(abs.(u_num_eval .- u_exact_eval))
        total_elapsed_sec = time() - picard_t0

        push!(history, (
            iter=iter,
            residual_err=residual_err,
            update_err=update_err,
            node_err=node_err,
            eqgrid_err=eqgrid_err,
            aa_depth=aa_depth,
            used_accel=used_accel,
            action_time_sec=action_time_sec,
            total_elapsed_sec=total_elapsed_sec,
        ))

        @printf("iter = %3d  residual_err = %.16e  update_err = %.16e  node_err = %.16e  eqgrid_err = %.16e  aa_depth = %d  accelerated = %d  action_time_sec = %.6f\n",
            iter, residual_err, update_err, node_err, eqgrid_err, aa_depth, Int(used_accel), action_time_sec)

        c_old = c_new
        u_old_nodes = u_new_nodes

        if residual_err <= tol * max(1.0, norm(u_old_nodes, Inf))
            converged = true
            break
        end
    end

    if !converged
        @warn "Anderson-accelerated Picard iteration reached maxit before the machine-precision residual criterion." maxit
    end

    total_picard_time = time() - picard_t0
    c_num = c_old
    u_num_eval = cheb_eval_many(c_num, x_eval)
    final_nodes = cheb_eval_many(c_num, x_nodes)

    final_residual_err = history[end].residual_err
    final_update_err = history[end].update_err
    final_node_err = norm(final_nodes - u_exact_vals, Inf)
    final_eqgrid_err = maximum(abs.(u_num_eval .- u_exact_eval))
    final_iter = history[end].iter

    solution_fig = joinpath(figures_dir, "ex6_solution_lambertW_anderson_independent_rhs_N1024.pdf")
    error_fig = joinpath(figures_dir, "ex6_anderson_independent_rhs_error_decay_N1024.pdf")
    history_csv = joinpath(figures_dir, "ex6_anderson_independent_rhs_history_N1024.csv")

    default(label="", linewidth=2)
    sol_plot = plot(
        x_eval, u_num_eval;
        xlabel="x",
        ylabel="u(x)",
        label="Anderson-Picard numerical",
        title="Nonlinear Volterra solution, independent RHS, α=100, λ=1, N=1024",
    )
    plot!(sol_plot, x_eval, u_exact_eval; linestyle=:dash, label="exact")
    solution_png = savefig_pdf_png(sol_plot, solution_fig)

    iters = [h.iter for h in history]
    eqerrs = max.([h.eqgrid_err for h in history], tol)
    residual_errs = max.([h.residual_err for h in history], tol)
    update_errs = max.([h.update_err for h in history], tol)
    ymax = max(maximum(eqerrs), maximum(residual_errs), maximum(update_errs)) * 10.0
    ymax = max(ymax, 10.0 * tol)

    err_plot = plot(
        iters, eqerrs;
        xlabel="Anderson-Picard iteration",
        ylabel="maximum error / residual",
        yscale=:log10,
        ylims=(tol, ymax),
        marker=:circle,
        label="equispaced-grid error",
        title=@sprintf("Anderson-Picard decay, final iter=%d, final err=%.3e", final_iter, final_eqgrid_err),
    )
    plot!(err_plot, iters, residual_errs; linestyle=:dash, marker=:diamond, label="fixed-point residual")
    plot!(err_plot, iters, update_errs; linestyle=:dot, marker=:utriangle, label="accepted update")
    error_png = savefig_pdf_png(err_plot, error_fig)

    write_history_csv(history_csv, history)

    total_runtime = time() - script_t0
    script_rel = relprint(@__FILE__)

    println("============================================================")
    println("Example 6: nonlinear saturated-memory Volterra equation")
    @printf("script file              = %s\n", script_rel)
    @printf("α                        = %.1f\n", α)
    @printf("λ                        = %.1f\n", λ)
    @printf("N                        = %d\n", N)
    println("tol                      = eps(Float64)")
    println("RHS construction          = independent Float64 Clenshaw--Curtis quadrature")
    @printf("RHS_CC_NMIN              = %d\n", RHS_CC_NMIN)
    @printf("RHS_CC_NMAX              = %d\n", RHS_CC_NMAX)
    @printf("RHS_CC_RTOL              = %.3e\n", RHS_CC_RTOL)
    @printf("RHS_CC_ATOL              = %.3e\n", RHS_CC_ATOL)
    @printf("RHS max quadrature order  = %d\n", rhs_data.max_qorder)
    @printf("RHS max refinement diff   = %.16e\n", rhs_data.max_qdiff)
    @printf("RHS unconverged nodes     = %d\n", rhs_data.not_converged)
    @printf("RHS build time            = %.6f sec\n", rhs_build_time)
    @printf("Anderson memory m         = %d\n", ANDERSON_M)
    @printf("Anderson damping          = %.2f\n", ANDERSON_DAMPING)
    @printf("iterations                = %d\n", final_iter)
    @printf("final residual err        = %.16e\n", final_residual_err)
    @printf("final update err          = %.16e\n", final_update_err)
    @printf("final node err            = %.16e\n", final_node_err)
    @printf("final eqgrid err          = %.16e\n", final_eqgrid_err)
    @printf("total iteration time      = %.6f sec\n", total_picard_time)
    @printf("total runtime             = %.6f sec\n", total_runtime)
    @printf("solution figure           = %s\n", relprint(solution_fig))
    @printf("solution png              = %s\n", relprint(solution_png))
    @printf("error figure              = %s\n", relprint(error_fig))
    @printf("error png                 = %s\n", relprint(error_png))
    @printf("history csv               = %s\n", relprint(history_csv))
    println("============================================================")
end

main()
