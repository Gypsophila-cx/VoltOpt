# Cauchy-error comparison between the coefficient-space method and
# Muntz/fractional Jacobi collocation for a non-manufactured weakly
# singular Volterra equation on [-1,1].
#
# Run:
#   julia --project=. test/compare/three_methods/compare_ours_muntz_cauchy.jl
#   julia --project=. test/compare/three_methods/plot_ours_muntz_cauchy.jl
#
# Quick PowerShell run:
#   $env:N_LIST="32,64,128"; julia --project=. test/compare/three_methods/compare_ours_muntz_cauchy.jl

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

const MU = parse(Float64, get(ENV, "MU", "0.5"))
const LAMBDA = parse(Float64, get(ENV, "LAMBDA", "0.5"))
const BETA = parse(Float64, get(ENV, "BETA", "0.5"))
const SIGMA = 0.0
const MUNTZ_LAMBDA = parse(Float64, get(ENV, "MUNTZ_LAMBDA", "0.5"))
const MUNTZ_ALPHA = parse(Float64, get(ENV, "MUNTZ_ALPHA", "-0.5"))
const MUNTZ_BETA = parse(Float64, get(ENV, "MUNTZ_BETA", "-0.5"))

const DEFAULT_N_LIST = [32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
const CHECK_GRID_SIZE = parse(Int, get(ENV, "CHECK_GRID_SIZE", "8001"))
const RHS_SAMPLE_DEGREE = parse(Int, get(ENV, "RHS_SAMPLE_DEGREE", "4096"))
const KERNEL_C_ROWS = parse(Int, get(ENV, "KERNEL_C_ROWS", "61"))
const KERNEL_C_COLS = parse(Int, get(ENV, "KERNEL_C_COLS", "41"))
const KERNEL_I = KERNEL_C_ROWS - 1
const KERNEL_J = KERNEL_C_COLS - 1
const MUNTZ_QUAD_ORDER = parse(Int, get(ENV, "MUNTZ_QUAD_ORDER", "192"))

const RESULTS_CSV = joinpath(DATA_DIR, "ours_muntz_cauchy_results.csv")
const SUMMARY_CSV = joinpath(DATA_DIR, "ours_muntz_cauchy_summary.csv")
const SOLUTIONS_CSV = joinpath(DATA_DIR, "ours_muntz_cauchy_solutions.csv")

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

gx(x) = 1.0 + sin(pi * Float64(x)) / (1000.0 * Float64(x)^2 + 1.0)
gt(t) = gx(2.0 * Float64(t) - 1.0)
Kx(x, y) = inv((Float64(x) + 1.0)^2 + (Float64(y) - 1.0)^2)
Kt(t, s) = Kx(2.0 * Float64(t) - 1.0, 2.0 * Float64(s) - 1.0)

phi_beta_t(t::Real, beta::Real = BETA) = 2.0 * clamp(Float64(t), 0.0, 1.0)^Float64(beta) - 1.0
invphi_t(z::Real, beta::Real = BETA) = ((1.0 + Float64(z)) / 2.0)^(1.0 / Float64(beta))
phi_beta_x(x::Real, beta::Real = BETA) = 2.0 * ((1.0 + Float64(x)) / 2.0)^Float64(beta) - 1.0

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

eval_jfp_x(c, x) = cheb_eval(c, phi_beta_x(x, BETA))

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
    sample_degree = max(RHS_SAMPLE_DEGREE, 4 * maxN)
    stat = @timed begin
        vals = Vector{Float64}(undef, sample_degree + 1)
        @inbounds for j in 0:sample_degree
            z = cos(pi * j / sample_degree)
            x = 2.0 * invphi_t(z, BETA) - 1.0
            vals[j + 1] = gx(x)
        end
        vals
    end
    coeffs = cheb_project_lobatto_from_values(stat.value, maxN)
    check_degree = 2 * sample_degree
    check_vals = Vector{Float64}(undef, check_degree + 1)
    @inbounds for j in 0:check_degree
        z = cos(pi * j / check_degree)
        x = 2.0 * invphi_t(z, BETA) - 1.0
        check_vals[j + 1] = gx(x)
    end
    check_coeffs = cheb_project_lobatto_from_values(check_vals, maxN)
    tail_count = min(20, length(coeffs))
    return coeffs, (
        projection_time = stat.time,
        rhs_projection_tail = maximum(abs, @view coeffs[(end - tail_count + 1):end]),
        rhs_projection_diff = maximum(abs.(coeffs .- check_coeffs)),
        sample_degree = sample_degree,
        check_degree = check_degree,
    )
end

dense_matrix(V) = V isa VoltOpt.PackedLowerHessenberg ? VoltOpt.packed_to_dense(V) : Matrix(V)

function build_kernel_coefficients()
    stat = @timed begin
        values = WeakCheb.sample_duffy_kernel(Kx, KERNEL_I, KERNEL_J, BETA)
        WeakCheb.cheb_coeffs_lobatto_2d(values)
    end
    return stat.value, stat.time
end

struct OursSolution
    N::Int
    dof::Int
    coeffs::Vector{Float64}
    build_time::Float64
    solve_time::Float64
    residual_linf::Float64
end

function solve_ours(N::Int, C, kernel_time, rhs_coeffs)
    rhsN = rhs_coeffs[1:(N + 1)]
    build_stat = @timed begin
        V, _ = VoltOpt.weak_voltmat_cheb(C; C_is_coeffs = true, mu = MU,
            sigma = SIGMA, beta = BETA, N = N, diagnostics = false,
            output_format = :dense, return_info = true)
        A = Matrix{Float64}(LinearAlgebra.I, N + 1, N + 1)
        A .+= LAMBDA .* dense_matrix(V)
        A
    end
    solve_stat = @timed build_stat.value \ rhsN
    coeffs = solve_stat.value
    residual = norm(build_stat.value * coeffs .- rhsN, Inf)
    return OursSolution(N, N + 1, coeffs, kernel_time + build_stat.time,
        solve_stat.time, residual)
end

eval_ours(sol::OursSolution, x) = eval_jfp_x(sol.coeffs, x)

function shifted_gaussjacobi(n::Int, alpha::Real, beta::Real)
    r, w = gaussjacobi(n, Float64(alpha), Float64(beta))
    z = (r .+ 1.0) ./ 2.0
    wshift = w ./ 2.0^(Float64(alpha) + Float64(beta) + 1.0)
    return z, wshift
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
    fill!(basis, 0.0)
    xf = Float64(x)
    tol = 256eps(Float64) * max(1.0, abs(xf))
    @inbounds for j in eachindex(nodes)
        if abs(xf - nodes[j]) <= tol
            basis[j] = 1.0
            return basis
        end
    end
    denom = 0.0
    @inbounds for j in eachindex(nodes)
        v = weights[j] / (xf - nodes[j])
        basis[j] = v
        denom += v
    end
    @inbounds for j in eachindex(nodes)
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

struct MuntzSolution
    N::Int
    dof::Int
    tnodes::Vector{Float64}
    znodes::Vector{Float64}
    bary_weights_z::Vector{Float64}
    values::Vector{Float64}
    build_time::Float64
    solve_time::Float64
    residual_linf::Float64
    qorder::Int
end

function solve_muntz(N::Int)
    dof = N + 1
    lambda_t = LAMBDA * 2.0^(1.0 - MU)
    lambda_frac = MUNTZ_LAMBDA
    qorder = MUNTZ_QUAD_ORDER
    build_stat = @timed begin
        z_nodes, _ = shifted_gaussjacobi(dof, MUNTZ_ALPHA, MUNTZ_BETA)
        tnodes = z_nodes .^ inv(lambda_frac)
        bwz = barycentric_weights(z_nodes)
        theta, wq = shifted_gaussjacobi(qorder, -MU, inv(lambda_frac) - 1.0)
        Aop = zeros(Float64, dof, dof)
        basis = zeros(Float64, dof)
        gnodes = gt.(tnodes)
        @inbounds for i in 1:dof
            ti = tnodes[i]
            factor = ti^(1.0 - MU) / lambda_frac
            for k in eachindex(theta)
                th = theta[k]
                thpow = th^inv(lambda_frac)
                s = ti * thpow
                denom = 1.0 - th
                ratio = abs(denom) <= sqrt(eps(Float64)) ? inv(lambda_frac) :
                        (1.0 - thpow) / denom
                z = s^lambda_frac
                barycentric_basis_values!(basis, z_nodes, bwz, z)
                coeff = wq[k] * factor * ratio^(-MU) * Kt(ti, s)
                for j in 1:dof
                    Aop[i, j] += coeff * basis[j]
                end
            end
        end
        A = Matrix{Float64}(LinearAlgebra.I, dof, dof)
        A .+= lambda_t .* Aop
        (A = A, gnodes = gnodes, tnodes = tnodes, z_nodes = z_nodes, bwz = bwz)
    end
    solve_stat = @timed build_stat.value.A \ build_stat.value.gnodes
    values = solve_stat.value
    residual = norm(build_stat.value.A * values .- build_stat.value.gnodes, Inf)
    return MuntzSolution(N, dof, build_stat.value.tnodes, build_stat.value.z_nodes,
        build_stat.value.bwz, values, build_stat.time, solve_stat.time, residual, qorder)
end

function eval_muntz(sol::MuntzSolution, x)
    t = clamp((Float64(x) + 1.0) / 2.0, 0.0, 1.0)
    z = t^MUNTZ_LAMBDA
    return barycentric_eval(sol.values, sol.znodes, sol.bary_weights_z, z)
end

function linf_l2_difference(f, g, xgrid)
    linf = 0.0
    sumsq = 0.0
    @inbounds for x in xgrid
        d = f(x) - g(x)
        linf = max(linf, abs(d))
        sumsq += d * d
    end
    return linf, sqrt(sumsq / length(xgrid))
end

function compute_cauchy_rows(method, sols, evalfun, xgrid, rhs_info)
    sorted = sort(sols; by = s -> s.N)
    ref = sorted[end]
    rows = NamedTuple[]
    for i in eachindex(sorted)
        sol = sorted[i]
        cref_linf, cref_l2 = linf_l2_difference(x -> evalfun(sol, x), x -> evalfun(ref, x), xgrid)
        succ_linf = NaN
        succ_l2 = NaN
        if i < length(sorted)
            nextsol = sorted[i + 1]
            succ_linf, succ_l2 = linf_l2_difference(x -> evalfun(nextsol, x), x -> evalfun(sol, x), xgrid)
        end
        notes = sol.N == ref.N ? "reference_level" : ""
        push!(rows, (
            method = method,
            N = sol.N,
            dof = sol.dof,
            build_time = sol.build_time,
            solve_time = sol.solve_time,
            total_time = sol.build_time + sol.solve_time,
            cauchy_successive_linf = succ_linf,
            cauchy_successive_l2 = succ_l2,
            cauchy_ref_linf = cref_linf,
            cauchy_ref_l2 = cref_l2,
            residual_linf = sol.residual_linf,
            rhs_projection_tail = rhs_info.rhs_projection_tail,
            rhs_projection_diff = rhs_info.rhs_projection_diff,
            muntz_lambda = method == "muntz_collocation" ? MUNTZ_LAMBDA : NaN,
            muntz_alpha = method == "muntz_collocation" ? MUNTZ_ALPHA : NaN,
            muntz_beta = method == "muntz_collocation" ? MUNTZ_BETA : NaN,
            notes = notes,
        ))
    end
    return rows
end

function csv_cell(x)
    if x isa AbstractFloat
        return isfinite(x) ? @sprintf("%.17e", x) : string(x)
    else
        return replace(string(x), "," => ";")
    end
end

function write_csv(path, headers, rows)
    open(path, "w") do io
        println(io, join(headers, ","))
        for r in rows
            println(io, join((csv_cell(getfield(r, Symbol(h))) for h in headers), ","))
        end
    end
end

const RESULTS_HEADERS = [
    "method", "N", "dof", "build_time", "solve_time", "total_time",
    "cauchy_successive_linf", "cauchy_successive_l2",
    "cauchy_ref_linf", "cauchy_ref_l2", "residual_linf",
    "rhs_projection_tail", "rhs_projection_diff",
    "muntz_lambda", "muntz_alpha", "muntz_beta", "notes",
]

const SUMMARY_HEADERS = [
    "method", "best_N_by_cauchy_ref", "best_cauchy_ref_linf",
    "best_successive_cauchy_linf", "best_total_time", "max_N",
    "final_residual_linf", "cross_method_linf_at_common_maxN",
    "cross_method_l2_at_common_maxN", "notes",
]

function build_summary(rows, cross_linf, cross_l2)
    out = NamedTuple[]
    for method in ("our_coeffspace", "muntz_collocation")
        mrows = [r for r in rows if r.method == method]
        isempty(mrows) && continue
        nonref = [r for r in mrows if r.cauchy_ref_linf > 0 && isfinite(r.cauchy_ref_linf)]
        best = isempty(nonref) ? mrows[end] : nonref[argmin([r.cauchy_ref_linf for r in nonref])]
        succ = [r.cauchy_successive_linf for r in mrows if isfinite(r.cauchy_successive_linf)]
        final = mrows[argmax([r.N for r in mrows])]
        push!(out, (
            method = method,
            best_N_by_cauchy_ref = best.N,
            best_cauchy_ref_linf = best.cauchy_ref_linf,
            best_successive_cauchy_linf = isempty(succ) ? NaN : minimum(succ),
            best_total_time = best.total_time,
            max_N = final.N,
            final_residual_linf = final.residual_linf,
            cross_method_linf_at_common_maxN = cross_linf,
            cross_method_l2_at_common_maxN = cross_l2,
            notes = final.notes,
        ))
    end
    return out
end

function write_solutions_csv(xgrid, ours_ref, muntz_ref)
    open(SOLUTIONS_CSV, "w") do io
        println(io, "x,u_ours,u_muntz,difference")
        for x in xgrid
            uo = eval_ours(ours_ref, x)
            um = eval_muntz(muntz_ref, x)
            @printf(io, "%.17e,%.17e,%.17e,%.17e\n", x, uo, um, uo - um)
        end
    end
end

function main()
    Ns = parse_env_int_list("N_LIST", DEFAULT_N_LIST)
    xgrid = collect(range(-1.0, 1.0; length = CHECK_GRID_SIZE))
    println("============================================================")
    println("Ours vs Muntz Cauchy benchmark without exact solution")
    @printf("mu = %.6f, lambda = %.6f, beta = %.6f\n", MU, LAMBDA, BETA)
    println("g(x) = 1 + sin(pi*x)/(1000*x^2+1)")
    println("N_LIST = ", Ns)
    println("CHECK_GRID_SIZE = ", CHECK_GRID_SIZE)
    println("MUNTZ_QUAD_ORDER = ", MUNTZ_QUAD_ORDER)
    println("============================================================")

    rhs_coeffs, rhs_info = rhs_coefficients(maximum(Ns))
    @printf("RHS projection tail = %.6e, projection diff = %.6e\n",
        rhs_info.rhs_projection_tail, rhs_info.rhs_projection_diff)

    C, kernel_time = build_kernel_coefficients()
    @printf("Kernel coefficient size = %d x %d, build_time = %.6e\n", size(C, 1), size(C, 2), kernel_time)

    ours_sols = OursSolution[]
    muntz_sols = MuntzSolution[]
    try
        solve_ours(first(Ns), C, kernel_time, rhs_coeffs)
        solve_muntz(first(Ns))
    catch
    end

    println()
    println("Running our_coeffspace...")
    for N in Ns
        sol = solve_ours(N, C, kernel_time, rhs_coeffs)
        push!(ours_sols, sol)
        @printf("  N=%5d, dof=%5d, time=%.6e, residual=%.6e\n",
            N, sol.dof, sol.build_time + sol.solve_time, sol.residual_linf)
    end

    println()
    println("Running muntz_collocation...")
    for N in Ns
        sol = solve_muntz(N)
        push!(muntz_sols, sol)
        @printf("  N=%5d, dof=%5d, time=%.6e, residual=%.6e\n",
            N, sol.dof, sol.build_time + sol.solve_time, sol.residual_linf)
    end

    rows = vcat(
        compute_cauchy_rows("our_coeffspace", ours_sols, eval_ours, xgrid, rhs_info),
        compute_cauchy_rows("muntz_collocation", muntz_sols, eval_muntz, xgrid, rhs_info),
    )
    ours_ref = sort(ours_sols; by = s -> s.N)[end]
    muntz_ref = sort(muntz_sols; by = s -> s.N)[end]
    cross_linf, cross_l2 = linf_l2_difference(x -> eval_ours(ours_ref, x), x -> eval_muntz(muntz_ref, x), xgrid)
    summary = build_summary(rows, cross_linf, cross_l2)
    write_csv(RESULTS_CSV, RESULTS_HEADERS, rows)
    write_csv(SUMMARY_CSV, SUMMARY_HEADERS, summary)
    write_solutions_csv(xgrid, ours_ref, muntz_ref)

    println()
    @printf("Cross-method max-N difference: Linf=%.6e, L2=%.6e\n", cross_linf, cross_l2)
    println("Saved CSV:")
    println("  ", RESULTS_CSV)
    println("  ", SUMMARY_CSV)
    println("  ", SOLUTIONS_CSV)

    try
        include(joinpath(SCRIPT_DIR, "plot_ours_muntz_cauchy.jl"))
        Base.invokelatest(plot_ours_muntz_cauchy)
    catch err
        @warn "Plotting failed. Run plot_ours_muntz_cauchy.jl separately." exception = (err, catch_backtrace())
    end
    return (rows = rows, summary = summary)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
