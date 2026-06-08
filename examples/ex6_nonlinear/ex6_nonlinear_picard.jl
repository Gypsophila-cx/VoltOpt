#!/usr/bin/env julia

# Example 6: nonlinear saturated-memory Volterra--Urysohn equation.
#
# The kernel
#
#     K(x,t,u) = u / (1 + (x - t)u^2)
#
# is a saturated-memory nonlinearity: as the lag x-t or |u| grows, the
# denominator damps the remembered contribution. Picard iteration therefore only
# needs repeated nonlinear Volterra actions with the current iterate.

using LinearAlgebra
using Printf
using Plots

const α = 100.0
const λ = 1.0
const N = 1024
const tol = eps(Float64)
const maxit = 100
const M_eval = 4001

const RUN_DIRECT_BASELINE = get(ENV, "RUN_DIRECT_BASELINE", "0") == "1"

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

function write_history_csv(path::AbstractString, history)
    open(path, "w") do io
        println(io, "iter,update_err,node_err,eqgrid_err,action_time_sec,total_elapsed_sec")
        for h in history
            @printf(io, "%d,%.17e,%.17e,%.17e,%.17e,%.17e\n",
                h.iter, h.update_err, h.node_err, h.eqgrid_err, h.action_time_sec, h.total_elapsed_sec)
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

    # Construct g in the same discrete Chebyshev space. Since
    # g = u_* - λ V_K[u_*], substituting u_* into u = g + λ V_K[u] gives the
    # exact manufactured solution up to the compVolt discretization error.
    V_exact = nonlinear_volterra_action_compvolt(c_exact, grid, Kfun)
    c_g = c_exact .- λ .* V_exact

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
    c_old = copy(c_g)
    u_old_nodes = cheb_eval_many(c_old, x_nodes)

    history = NamedTuple[]
    picard_t0 = time()
    converged = false

    for iter in 1:maxit
        action_t0 = time()
        V_old = nonlinear_volterra_action_compvolt(c_old, grid, Kfun)
        action_time_sec = time() - action_t0

        c_new = c_g .+ λ .* V_old
        u_new_nodes = cheb_eval_many(c_new, x_nodes)
        u_num_eval = cheb_eval_many(c_new, x_eval)

        update_err = norm(u_new_nodes - u_old_nodes, Inf)
        node_err = norm(u_new_nodes - u_exact_vals, Inf)
        eqgrid_err = maximum(abs.(u_num_eval .- u_exact_eval))
        total_elapsed_sec = time() - picard_t0

        push!(history, (
            iter=iter,
            update_err=update_err,
            node_err=node_err,
            eqgrid_err=eqgrid_err,
            action_time_sec=action_time_sec,
            total_elapsed_sec=total_elapsed_sec,
        ))

        @printf("iter = %3d  update_err = %.16e  node_err = %.16e  eqgrid_err = %.16e  action_time_sec = %.6f\n",
            iter, update_err, node_err, eqgrid_err, action_time_sec)

        if update_err <= tol * max(1.0, norm(u_new_nodes, Inf))
            c_old = c_new
            u_old_nodes = u_new_nodes
            converged = true
            break
        end

        c_old = c_new
        u_old_nodes = u_new_nodes
    end

    if !converged
        @warn "Picard iteration reached maxit before the machine-precision update criterion." maxit
    end

    total_picard_time = time() - picard_t0
    c_num = c_old
    u_num_eval = cheb_eval_many(c_num, x_eval)
    final_nodes = cheb_eval_many(c_num, x_nodes)

    final_update_err = history[end].update_err
    final_node_err = norm(final_nodes - u_exact_vals, Inf)
    final_eqgrid_err = maximum(abs.(u_num_eval .- u_exact_eval))
    final_iter = history[end].iter

    solution_fig = joinpath(figures_dir, "ex6_solution_lambertW_picard_N1024.pdf")
    error_fig = joinpath(figures_dir, "ex6_picard_error_decay_N1024.pdf")
    history_csv = joinpath(figures_dir, "ex6_picard_history_N1024.csv")

    default(label="", linewidth=2)
    sol_plot = plot(
        x_eval, u_num_eval;
        xlabel="x",
        ylabel="u(x)",
        label="Picard numerical",
        title="Nonlinear Volterra solution, α=100, λ=1, N=1024",
    )
    plot!(sol_plot, x_eval, u_exact_eval; linestyle=:dash, label="exact")
    solution_png = savefig_pdf_png(sol_plot, solution_fig)

    iters = [h.iter for h in history]
    eqerrs = max.([h.eqgrid_err for h in history], tol)
    update_errs = max.([h.update_err for h in history], tol)
    ymax = max(maximum(eqerrs), maximum(update_errs)) * 10.0
    ymax = max(ymax, 10.0 * tol)

    err_plot = plot(
        iters, eqerrs;
        xlabel="iteration",
        ylabel="error",
        yaxis=:log10,
        ylims=(1e-15, 1),
        marker=false,
        label=nothing,
        color=:black,
        framestyle=:box,
        grid = false,
    )
    #plot!(err_plot, iters, update_errs; linestyle=:dot, marker=:diamond, label="Picard update")
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
    @printf("Picard iterations         = %d\n", final_iter)
    @printf("final update err          = %.16e\n", final_update_err)
    @printf("final node err            = %.16e\n", final_node_err)
    @printf("final eqgrid err          = %.16e\n", final_eqgrid_err)
    @printf("total Picard time         = %.6f sec\n", total_picard_time)
    @printf("total runtime             = %.6f sec\n", total_runtime)
    @printf("solution figure           = %s\n", relprint(solution_fig))
    @printf("solution png              = %s\n", relprint(solution_png))
    @printf("error figure              = %s\n", relprint(error_fig))
    @printf("error png                 = %s\n", relprint(error_png))
    @printf("history csv               = %s\n", relprint(history_csv))
    println("============================================================")
end

main()
