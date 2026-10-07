#!/usr/bin/env julia

# ex6_fixed10_varyN.jl
#
# Purpose:
#   Generate the Example 6 fixed-iteration vary-N comparison CSV.
#
# Problem:
#   Same manufactured nonlinear Volterra--Urysohn equation as
#   ex6_nonlinear_picard.jl.
#
# This script solves the same manufactured nonlinear Volterra--Urysohn equation
# as ex6_anderson_fixed10_varyN.jl, but removes Anderson acceleration. It fixes
# the number of nonlinear iterations to FIXED_ITERS = 10 and varies the
# truncation size N from 10 to 5000. It writes one CSV file into
# examples/ex6_nonlinear/data.
#
# Method:
#   Runs plain Picard/fixed-point iteration for a fixed iteration count and
#   records residual, node-grid, equidistant-grid, and timing diagnostics.
#
# Output files:
#   data/ex6_fixed10_varyN.csv
#
# Run from the repository root:
#   julia --project=. examples/ex6_nonlinear/ex6_fixed10_varyN.jl

using LinearAlgebra
using Printf
using Statistics

const ALPHA = 100.0
const LAMBDA = 1.0
const tol = eps(Float64)
const FIXED_ITERS = 10
const M_error_eval = 1000

# A moderate logarithmic/benchmark-oriented sweep from 10 to 5000.
# For a dense sweep, replace this by collect(10:10:5000), but that will be much slower.
const N_VALUES = [10, 20, 40, 80, 160, 320, 640, 1024, 2048, 4096]

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
# The current tree exposes the nonlinear API from src/CompVolt/compVolt.jl.
# A legacy lowercase src/compVolt.jl path is still accepted for compatibility
# with older example descriptions.
include(compvolt_file)

function lambertw_exp(s::Float64; maxiter::Int=100)
    if !(isfinite(s))
        @warn "lambertw_exp received a non-finite input; returning NaN." s
        return NaN
    end
    if s < -ALPHA || s > ALPHA
        @warn "lambertw_exp was designed for s in [-100, 100]; continuing anyway." s
    end

    # Solve w + log(w) = s, which is equivalent to w = W(exp(s)).
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

uexact(x::Real) = sqrt(lambertw_exp(Float64(ALPHA * x)))
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

function fixed_point_map(cu::AbstractVector, c_g::AbstractVector, grid, K)
    V_u = nonlinear_volterra_action_compvolt(cu, grid, K)
    return c_g .+ LAMBDA .* V_u
end

function relprint(path::AbstractString)
    return replace(relpath(path, project_root), '\\' => '/')
end

function write_varyN_csv(path::AbstractString, results)
    open(path, "w") do io
        println(io, "N,iterations,final_residual_err,final_update_err,final_node_err,final_eqgrid_err,accelerated_steps,rhs_build_time_sec,iteration_time_sec,final_residual_time_sec,total_runtime_sec,avg_action_time_sec")
        for r in results
            @printf(io, "%d,%d,%.17e,%.17e,%.17e,%.17e,%d,%.17e,%.17e,%.17e,%.17e,%.17e\n",
                r.N, r.iterations, r.final_residual_err, r.final_update_err,
                r.final_node_err, r.final_eqgrid_err, r.accelerated_steps,
                r.rhs_build_time_sec, r.iteration_time_sec, r.final_residual_time_sec,
                r.total_runtime_sec, r.avg_action_time_sec)
        end
    end
end

function run_one_N(N::Int, x_error_eval::AbstractVector{Float64})
    total_t0 = time()

    grid = CompVoltGrid(N)
    x_nodes = grid.x

    u_exact_vals = uexact.(x_nodes)
    u_exact_error_eval = uexact.(x_error_eval)

    rhs_t0 = time()
    c_exact = ascending_values_to_cheb_coeffs(u_exact_vals)
    V_exact = nonlinear_volterra_action_compvolt(c_exact, grid, Kfun)
    c_g = Vector{Float64}(c_exact .- LAMBDA .* V_exact)
    rhs_build_time_sec = time() - rhs_t0

    c_old = copy(c_g)
    u_old_nodes = cheb_eval_many(c_old, x_nodes)

    action_times = Float64[]
    update_err = NaN

    iter_t0 = time()
    for _ in 1:FIXED_ITERS
        action_t0 = time()
        c_new = Vector{Float64}(fixed_point_map(c_old, c_g, grid, Kfun))
        push!(action_times, time() - action_t0)

        u_new_nodes = cheb_eval_many(c_new, x_nodes)
        update_err = norm(u_new_nodes - u_old_nodes, Inf)

        c_old = c_new
        u_old_nodes = u_new_nodes
    end
    iteration_time_sec = time() - iter_t0

    c_num = c_old
    final_nodes = u_old_nodes
    final_node_err = norm(final_nodes - u_exact_vals, Inf)

    u_num_error_eval = cheb_eval_many(c_num, x_error_eval)
    final_eqgrid_err = maximum(abs.(u_num_error_eval .- u_exact_error_eval))

    # Compute the true fixed-point residual after exactly FIXED_ITERS Picard steps.
    residual_t0 = time()
    c_final_picard = Vector{Float64}(fixed_point_map(c_num, c_g, grid, Kfun))
    final_residual_time_sec = time() - residual_t0
    final_residual_nodes = cheb_eval_many(c_final_picard .- c_num, x_nodes)
    final_residual_err = norm(final_residual_nodes, Inf)

    total_runtime_sec = time() - total_t0
    avg_action_time_sec = isempty(action_times) ? NaN : mean(action_times)

    return (
        N=N,
        iterations=FIXED_ITERS,
        final_residual_err=final_residual_err,
        final_update_err=update_err,
        final_node_err=final_node_err,
        final_eqgrid_err=final_eqgrid_err,
        accelerated_steps=0,
        rhs_build_time_sec=rhs_build_time_sec,
        iteration_time_sec=iteration_time_sec,
        final_residual_time_sec=final_residual_time_sec,
        total_runtime_sec=total_runtime_sec,
        avg_action_time_sec=avg_action_time_sec,
    )
end

function main()
    script_t0 = time()

    data_dir = joinpath(@__DIR__, "data")
    mkpath(data_dir)

    x_error_eval = collect(range(-1.0, 1.0; length=M_error_eval))
    results = NamedTuple[]

    println("============================================================")
    println("Example 6 benchmark: fixed 10 plain Picard/fixed-point iterations, varying N")
    @printf("alpha                    = %.1f\n", ALPHA)
    @printf("lambda                   = %.1f\n", LAMBDA)
    @printf("fixed iterations          = %d\n", FIXED_ITERS)
    println("method                    = plain Picard/fixed-point")
    @printf("N values                  = %s\n", string(N_VALUES))
    println("------------------------------------------------------------")

    for N in N_VALUES
        GC.gc()
        r = run_one_N(N, x_error_eval)
        push!(results, r)
        @printf("N = %5d  eqgrid_err = %.3e  node_err = %.3e  residual = %.3e  iter_time = %.4f sec  total = %.4f sec  picard_steps = %d\n",
            r.N, r.final_eqgrid_err, r.final_node_err, r.final_residual_err,
            r.iteration_time_sec, r.total_runtime_sec, r.iterations)
    end

    varyN_csv = joinpath(data_dir, "ex6_fixed10_varyN.csv")

    write_varyN_csv(varyN_csv, results)

    total_script_time = time() - script_t0

    println("------------------------------------------------------------")
    @printf("CSV                       = %s\n", relprint(varyN_csv))
    @printf("total script time         = %.6f sec\n", total_script_time)
    println("============================================================")
end

main()
