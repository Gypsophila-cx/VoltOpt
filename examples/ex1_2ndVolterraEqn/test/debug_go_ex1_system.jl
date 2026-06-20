using LinearAlgebra
using Printf
using SparseArrays

const EX1_DIR = @__DIR__
const PROJECT_ROOT = abspath(joinpath(EX1_DIR, "..", ".."))
const GO_PROJECT = joinpath(PROJECT_ROOT, "others", "GutlebOlver")

import Pkg
if abspath(Base.active_project()) != joinpath(GO_PROJECT, "Project.toml")
    Pkg.activate(GO_PROJECT)
end

using ApproxFun
using BandedMatrices
using SparseVolterraNew

const x1 = -0.35
const x2 = 0.42
const A1 = 1.0
const A2 = -0.6
const delta1 = 1.0e-3
const delta2 = 1.5e-3

Sstep(x, x0, delta) = 0.5 * (1.0 + tanh((x - x0) / delta))
u_original(x) = 1.0 + A1 * Sstep(x, x1, delta1) + A2 * Sstep(x, x2, delta2)
tau_xy(x, y) = 1.5 + x - 0.5 * y
K_original(x, y) = exp(-(x - y) / tau_xy(x, y)) / tau_xy(x, y)

x_from_s(s) = 2.0 * s - 1.0
K_go(s, r) = 2.0 * K_original(x_from_s(s), x_from_s(r))
u_go(s) = u_original(x_from_s(s))

const GL_N = parse(Int, get(ENV, "GO_EX1_RHS_Q", "120"))
const SPLIT_L = 40.0

function gausslegendre_rule(n::Int)
    beta = [k / sqrt(4.0 * k^2 - 1.0) for k in 1:n-1]
    E = eigen(SymTridiagonal(zeros(n), beta))
    p = sortperm(E.values)
    return E.values[p], (2.0 .* abs2.(E.vectors[1, :]))[p]
end

const GLX, GLW = gausslegendre_rule(GL_N)

function gauss_integral(f, a::Float64, b::Float64)
    b > a || return 0.0
    c = 0.5 * (a + b)
    h = 0.5 * (b - a)
    acc = 0.0
    @inbounds for q in eachindex(GLX)
        acc += GLW[q] * f(c + h * GLX[q])
    end
    return h * acc
end

function rhs_integral_go(s::Real)
    sf = Float64(s)
    sf <= 0.0 && return 0.0
    pts = Float64[0.0, sf]
    for (x0, delta) in ((x1, delta1), (x2, delta2))
        for z in (x0 - SPLIT_L * delta, x0, x0 + SPLIT_L * delta)
            p = 0.5 * (z + 1.0)
            0.0 < p < sf && push!(pts, p)
        end
    end
    sort!(pts)
    unique!(pts)
    total = 0.0
    for j in 1:length(pts)-1
        total += gauss_integral(r -> K_go(sf, r) * u_go(r), pts[j], pts[j + 1])
    end
    return total
end

rhs_go(s) = u_go(s) + rhs_integral_go(s)

const GO_P10 = go_shifted_jacobi_space(1, 0)

function ex1_layer_validation_grid(n::Int)
    pts = collect(range(0.0, 1.0; length=max(401, min(4001, 4n))))
    for (x0, delta) in ((x1, delta1), (x2, delta2))
        for scale in (-40.0, -10.0, -1.0, 0.0, 1.0, 10.0, 40.0)
            s = 0.5 * (x0 + scale * delta + 1.0)
            0.0 <= s <= 1.0 && push!(pts, s)
        end
    end
    return sort!(unique!(pts))
end

function project_go_p10(f, n::Int, validation_grid)
    return build_go_rhs_coefficients(
        f,
        n;
        alpha=1,
        beta=0,
        method=:fixed_gauss_jacobi,
        projection_tol=parse(Float64, get(ENV, "GO_EX1_PROJECTION_TOL", "1e-8")),
        oversampling=parse(Int, get(ENV, "GO_EX1_PROJECTION_OVERSAMPLING", "4")),
        validation_grid=validation_grid,
    )
end

function eval_p10(coeffs, points)
    return Fun(GO_P10, coeffs).(points)
end

function parse_kerneldepths(default::AbstractString)
    vals = Int[]
    for item in split(get(ENV, "GO_EX1_KERNELDEPTH_SCAN", default), ',')
        s = strip(item)
        isempty(s) && continue
        push!(vals, parse(Int, s))
    end
    isempty(vals) && error("GO_EX1_KERNELDEPTH_SCAN produced no kerneldepth values")
    return unique(vals)
end

function main()
    N = parse(Int, get(ENV, "GO_EX1_N", "24"))
    kerneldepths = parse_kerneldepths("8,12,16,24,32")
    max_ok_residual = parse(Float64, get(ENV, "GO_EX1_MAX_OK_RESIDUAL", "1e-6"))
    max_ok_error = parse(Float64, get(ENV, "GO_EX1_MAX_OK_ERROR", "1e-5"))
    max_ok_volterra_error = parse(Float64, get(ENV, "GO_EX1_MAX_OK_VOLTERRA_ERROR", "1e-6"))

    println("ex1 Gutleb--Olver sparse/banded diagnostic")
    println("original equation: u(x) + int_{-1}^x K(x,y)u(y)dy = f(x)")
    println("transformed equation: u_go(s) + int_0^s K_go(s,r)u_go(r)dr = rhs_go(s)")
    println("variable transform: x = 2s - 1, y = 2r - 1")
    println("kernel transform: K_go(s,r) = 2*K_original(2s-1,2r-1)")
    println("solution basis: GO shifted Jacobi P^(1,0) on [0,1]")
    println("raw VK: strict-band Q_y K(1-X,Y) E_y via flip=true; K_go itself is not pre-flipped")
    println("final Volterra term: S10_from_00 * R * L00_from_10 * VK")
    println("final output basis: GO shifted Jacobi P^(1,0)")
    println("tau rows: none; ex1 is a second-kind Volterra integral equation, not a VIDE")
    @printf("N = %d, kerneldepth_scan = %s, GL_N = %d\n", N, string(kerneldepths), GL_N)

    validation_grid = ex1_layer_validation_grid(N)
    local rhs_coeff, rhs_info, exact_coeff, exact_info
    rhs_time = @elapsed begin
        rhs_coeff, rhs_info = project_go_p10(rhs_go, N, validation_grid)
    end
    exact_projection_time = @elapsed begin
        exact_coeff, exact_info = project_go_p10(u_go, N, validation_grid)
    end
    @printf("rhs_time = %.6e\n", rhs_time)
    @printf("exact_projection_time = %.6e\n", exact_projection_time)
    @printf("rhs validation_error = %.6e, tail_norm = %.6e, status = %s\n",
        rhs_info.validation_error, rhs_info.tail_norm, string(rhs_info.status))
    @printf("exact validation_error = %.6e, tail_norm = %.6e, status = %s\n",
        exact_info.validation_error, exact_info.tail_norm, string(exact_info.status))
    println("rhs notes = ", go_rhs_info_notes(rhs_info))
    println("exact projection notes = ", go_rhs_info_notes(exact_info))

    grid = collect(range(0.0, 1.0; length=max(1001, 4N)))
    integral_truth = rhs_integral_go.(grid)

    println()
    println("kerneldepth scan")
    @printf("%-12s %-14s %-14s %-14s %-14s %-14s %-12s %-12s %-12s %-12s\n",
        "kerneldepth", "residual_max", "residual_rel", "volterra_err", "solve_err",
        "matrix_time", "VK_nnz", "A_nnz", "A_lower", "A_upper")

    results = NamedTuple[]
    for kerneldepth in kerneldepths
        local coeffs, VK, A, V_phys, system
        kernel_approx_time = @elapsed begin
            coeffs = build_gutleb_kernel_coefficients(K_go, kerneldepth; flip=true)
        end
        matrix_construct_time = @elapsed begin
            VK = gutleb_olver_banded_construct(coeffs, N; flip=true)
        end
        system_assembly_time = @elapsed begin
            system = assemble_go_second_kind_vie_operator_from_raw(
                VK,
                N;
                sign=1.0,
                flip_policy="strict-band constructor flip=true; K_go is not pre-flipped",
            )
            A = system.A
            V_phys = system.V_phys
        end

        residual = A * exact_coeff - rhs_coeff
        residual_max = norm(residual, Inf)
        residual_rel = residual_max / max(norm(rhs_coeff, Inf), eps(Float64))
        volterra_vals = eval_p10(V_phys * exact_coeff, grid)
        volterra_error = maximum(abs.(volterra_vals .- integral_truth))

        local sol
        solve_time = @elapsed begin
            sol = A \ rhs_coeff
        end
        solve_error = maximum(abs.(eval_p10(sol, grid) .- u_go.(grid)))
        lA, uA = bandwidths(A)
        @printf("%-12d %-14.6e %-14.6e %-14.6e %-14.6e %-14.6e %-12d %-12d %-12d %-12d\n",
            kerneldepth,
            residual_max,
            residual_rel,
            volterra_error,
            solve_error,
            matrix_construct_time + system_assembly_time,
            nnz(sparse(VK)),
            nnz(sparse(A)),
            Int(lA),
            Int(uA))

        push!(results, (
            kerneldepth=kerneldepth,
            coeffs=coeffs,
            VK=VK,
            A=A,
            V_phys=V_phys,
            info=system.info,
            L00_from_10=system.L00_from_10,
            R=system.R,
            S10_from_00=system.S10_from_00,
            kernel_approx_time=kernel_approx_time,
            matrix_construct_time=matrix_construct_time,
            system_assembly_time=system_assembly_time,
            solve_time=solve_time,
            residual=residual,
            residual_max=residual_max,
            residual_rel=residual_rel,
            volterra_error=volterra_error,
            solve_error=solve_error,
        ))
    end

    best = results[argmin([r.residual_rel for r in results])]
    println()
    println("selected result: minimum residual_rel in scan")
    println("VK storage = ", typeof(best.VK))
    println("A storage = ", typeof(best.A))
    println("V_phys storage = ", typeof(best.V_phys))
    println("VK bandwidth = ", bandwidths(best.VK))
    println("A bandwidth = ", bandwidths(best.A))
    println("operator solution_basis = ", best.info.solution_basis)
    println("operator rhs_basis = ", best.info.rhs_basis)
    println("operator raw_operator = ", best.info.raw_operator)
    println("operator final_chain = ", best.info.final_chain)
    println("operator final_system = ", best.info.final_system)
    println("operator flip_policy = ", best.info.flip_policy)
    println("operator dense_fallback = ", best.info.dense_fallback)
    println("operator raw_vk_added_to_identity = ", best.info.raw_vk_added_to_identity)
    println("operator tau_rows = ", best.info.tau_rows)
    @printf("kernel_approx_time = %.6e\n", best.kernel_approx_time)
    @printf("matrix_construct_time = %.6e\n", best.matrix_construct_time)
    @printf("system_assembly_time = %.6e\n", best.system_assembly_time)
    @printf("solve_time = %.6e\n", best.solve_time)
    @printf("tau residual = %.6e (no tau rows)\n", 0.0)
    @printf("body residual max = %.6e\n", best.residual_max)
    @printf("body residual relative = %.6e\n", best.residual_rel)
    @printf("V_phys physical integral max_abs error = %.6e\n", best.volterra_error)
    @printf("physical-grid max_abs error = %.6e\n", best.solve_error)
    println("first body residuals = ", best.residual[1:min(10, end)])
    println("last body residuals = ", best.residual[max(1, end - 9):end])

    if rhs_info.status == :success &&
       exact_info.status == :success &&
       best.residual_rel <= max_ok_residual &&
       best.volterra_error <= max_ok_volterra_error &&
       best.solve_error <= max_ok_error
        println("status = success")
    else
        println("status = failed")
        println("notes = ex1 GO success is not certified; require RHS/exact projection validation, small exact-coefficient residual, small V_phys physical check, and small solve error")
        exit(2)
    end
end

main()
