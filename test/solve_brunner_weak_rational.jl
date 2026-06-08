using LinearAlgebra
using Printf

const SCRIPT_DIR = @__DIR__
const PROJECT_ROOT = dirname(SCRIPT_DIR)
const DATA_DIR = joinpath(PROJECT_ROOT, "test", "data")
const FIGURES_DIR = joinpath(PROJECT_ROOT, "test", "figures")

const DEFAULT_MS = [32, 64, 128, 256, 512, 1024]
const DEFAULT_M = 2
const DEFAULT_Q = 2.0
const DEFAULT_MU = 0.5
const DEFAULT_LAMBDA = 0.5
const DEFAULT_NQ = 64
const DEFAULT_REF_FACTOR = 4
const DEFAULT_SAMPLE_COUNT = 2001

const _GL01_CACHE = Dict{Int,Tuple{Vector{Float64},Vector{Float64}}}()

struct BrunnerSolution
    M::Int
    m::Int
    q::Float64
    mu::Float64
    lambda::Float64
    lambda_hat::Float64
    nq::Int
    mesh::Vector{Float64}
    c::Vector{Float64}
    nodes::Matrix{Float64}
    Uvals::Matrix{Float64}
    solve_time::Float64
end

dof(sol::BrunnerSolution) = sol.M * sol.m

# Transformed kernel and right-hand side on [0, 1].
Khat(s, t) = inv(4.0 * s^2 + 4.0 * (t - 1.0)^2)
G(s) = 1.0 + sin(pi * (2.0 * s - 1.0)) / (1.0e3 * (2.0 * s - 1.0)^2 + 1.0)

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

function gausslegendre01(nq)
    nq > 0 || throw(ArgumentError("nq must be positive."))

    cached = get(_GL01_CACHE, nq, nothing)
    cached !== nothing && return cached

    x = Vector{Float64}(undef, nq)
    w = Vector{Float64}(undef, nq)

    # Prefer the project's FastGaussQuadrature dependency when available, but
    # keep a standard-library Golub-Welsch fallback so this script remains
    # robust as an independent benchmark.
    used_fastgauss = false
    try
        @eval import FastGaussQuadrature
        xi, wi = FastGaussQuadrature.gausslegendre(nq)
        @inbounds for k in 1:nq
            x[k] = 0.5 * (xi[k] + 1.0)
            w[k] = 0.5 * wi[k]
        end
        used_fastgauss = true
    catch
        used_fastgauss = false
    end

    if !used_fastgauss
        if nq == 1
            x[1] = 0.5
            w[1] = 1.0
        else
            beta = [i / sqrt(4.0 * i^2 - 1.0) for i in 1:(nq - 1)]
            T = SymTridiagonal(zeros(nq), beta)
            E = eigen(T)
            xi = E.values
            wi = 2.0 .* abs2.(E.vectors[1, :])
            @inbounds for k in 1:nq
                x[k] = 0.5 * (xi[k] + 1.0)
                w[k] = 0.5 * wi[k]
            end
        end
    end

    _GL01_CACHE[nq] = (x, w)
    return x, w
end

function product_weight(delta, theta_max, j, c, mu; nq = DEFAULT_NQ)
    0.0 < mu < 1.0 || throw(ArgumentError("mu must satisfy 0 < mu < 1."))
    theta_max <= 0.0 && return 0.0
    delta + 1e-14 >= theta_max ||
        throw(ArgumentError("theta_max must not exceed delta."))

    xq, wq = gausslegendre01(nq)
    singular_current_panel =
        delta <= 1.0 + 1e-14 &&
        abs(theta_max - delta) <= 1e-12 * max(1.0, abs(delta))

    if singular_current_panel
        delta <= 0.0 && return 0.0
        exponent = inv(1.0 - mu)
        acc = 0.0
        @inbounds for k in eachindex(xq)
            theta = delta * (1.0 - xq[k]^exponent)
            acc += wq[k] * lagrange_basis(theta, j, c)
        end
        return delta^(1.0 - mu) * acc / (1.0 - mu)
    else
        acc = 0.0
        @inbounds for k in eachindex(xq)
            theta = theta_max * xq[k]
            acc += wq[k] * (delta - theta)^(-mu) * lagrange_basis(theta, j, c)
        end
        return theta_max * acc
    end
end

function default_collocation_nodes(m::Int)
    m > 0 || throw(ArgumentError("m must be positive."))
    m == 1 && return [1.0]
    m == 2 && return [1.0 / 3.0, 1.0]

    # Fallback for experimentation; the benchmark default is the requested
    # two-point right-Radau-like choice [1/3, 1].
    return collect(range(1.0 / m, 1.0; length = m))
end

function validate_collocation_nodes(c)
    all(isfinite, c) || throw(ArgumentError("collocation nodes must be finite."))
    all(x -> 0.0 < x <= 1.0, c) ||
        throw(ArgumentError("collocation nodes must lie in (0, 1]."))
    all(diff(c) .> 0.0) ||
        throw(ArgumentError("collocation nodes must be strictly increasing."))
    return Float64.(c)
end

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
    @inbounds for k in eachindex(xq)
        theta = xq[k]
        factor = wq[k] * (delta - theta)^(-mu)
        for j in eachindex(c)
            weights[j] += factor * lagrange_basis(theta, j, c)
        end
    end
    return weights
end

function solve_brunner(
    M;
    m = DEFAULT_M,
    q = DEFAULT_Q,
    mu = DEFAULT_MU,
    lambda = DEFAULT_LAMBDA,
    nq = DEFAULT_NQ,
    c = nothing,
)
    cvec = validate_collocation_nodes(c === nothing ? default_collocation_nodes(m) : c)
    length(cvec) == m ||
        throw(ArgumentError("length(c) must match m."))

    lambda_hat = Float64(lambda) * 2.0^(1.0 - Float64(mu))
    mesh = graded_mesh(M, q)
    h = diff(mesh)
    hpow = h .^ (1.0 - Float64(mu))
    nodes = collocation_points(mesh, cvec)
    Uvals = zeros(Float64, M, m)

    current_weights = Matrix{Float64}(undef, m, m)
    for i in 1:m
        for j in 1:m
            current_weights[i, j] =
                product_weight(cvec[i], cvec[i], j, cvec, mu; nq = nq)
        end
    end

    xq, wq = gausslegendre01(nq)
    smooth_weights = zeros(Float64, m)
    A = Matrix{Float64}(undef, m, m)
    b = Vector{Float64}(undef, m)

    solve_time = @elapsed begin
        @inbounds for n in 1:M
            fill!(A, 0.0)
            fill!(b, 0.0)

            for i in 1:m
                sni = nodes[n, i]
                history = 0.0

                for l in 1:(n - 1)
                    delta = (sni - mesh[l]) / h[l]
                    local_smooth_weights!(smooth_weights, delta, cvec, mu, xq, wq)
                    for j in 1:m
                        history += hpow[l] *
                                   smooth_weights[j] *
                                   Khat(sni, nodes[l, j]) *
                                   Uvals[l, j]
                    end
                end

                b[i] = G(sni) - lambda_hat * history

                for j in 1:m
                    A[i, j] = (i == j ? 1.0 : 0.0) +
                              lambda_hat *
                              hpow[n] *
                              current_weights[i, j] *
                              Khat(sni, nodes[n, j])
                end
            end

            Uvals[n, :] .= A \ b
        end
    end

    return BrunnerSolution(
        Int(M),
        Int(m),
        Float64(q),
        Float64(mu),
        Float64(lambda),
        lambda_hat,
        Int(nq),
        mesh,
        cvec,
        nodes,
        Uvals,
        solve_time,
    )
end

function eval_brunner_solution(s_eval, mesh, c, Uvals)
    s = clamp(Float64(s_eval), 0.0, 1.0)
    M = length(mesh) - 1
    panel = clamp(searchsortedfirst(mesh, s) - 1, 1, M)
    h = mesh[panel + 1] - mesh[panel]
    theta = (s - mesh[panel]) / h

    u = 0.0
    @inbounds for j in eachindex(c)
        u += Uvals[panel, j] * lagrange_basis(theta, j, c)
    end
    return u
end

eval_brunner_solution(s, sol::BrunnerSolution) =
    eval_brunner_solution(s, sol.mesh, sol.c, sol.Uvals)

function cauchy_error_from_solutions(
    sol::BrunnerSolution,
    refsol::BrunnerSolution;
    sample_count = DEFAULT_SAMPLE_COUNT,
)
    sample_count >= 2 ||
        throw(ArgumentError("sample_count must be at least 2."))

    err = 0.0
    @inbounds for k in 0:(sample_count - 1)
        s = k / (sample_count - 1)
        err = max(
            err,
            abs(eval_brunner_solution(s, sol) - eval_brunner_solution(s, refsol)),
        )
    end
    return err
end

function cauchy_error(M, Mref; sample_count = DEFAULT_SAMPLE_COUNT, kwargs...)
    sol = solve_brunner(M; kwargs...)
    refsol = solve_brunner(Mref; kwargs...)
    return cauchy_error_from_solutions(sol, refsol; sample_count = sample_count)
end

function csv_cell(x)
    x === missing && return "missing"
    x isa Integer && return string(x)
    x isa AbstractFloat && return @sprintf("%.17e", x)
    return string(x)
end

function write_results_csv(path::AbstractString, rows)
    mkpath(dirname(path))
    headers = ["M", "dof", "m", "q", "mu", "lambda", "solve_time", "cauchy_err"]
    open(path, "w") do io
        println(io, join(headers, ","))
        for r in rows
            println(io, join((csv_cell(getfield(r, Symbol(h))) for h in headers), ","))
        end
    end
end

function maybe_plot_results(rows; fig_dir = FIGURES_DIR)
    mkpath(fig_dir)

    dofs = [Float64(r.dof) for r in rows]
    errs = [Float64(r.cauchy_err) for r in rows]
    times = [Float64(r.solve_time) for r in rows]

    try
        @eval import Plots
        Base.invokelatest(Plots.gr)

        valid_err = findall(i -> dofs[i] > 0.0 && errs[i] > 0.0 && isfinite(errs[i]), eachindex(dofs))
        if !isempty(valid_err)
            plt_err = Base.invokelatest(
                Plots.plot,
                dofs[valid_err],
                errs[valid_err];
                xscale = :log10,
                yscale = :log10,
                linewidth = 2.2,
                marker = :circle,
                markersize = 4,
                label = nothing,
                xlabel = "dof",
                ylabel = "Cauchy error",
                framestyle = :box,
                grid = false,
                size = (760, 460),
            )
            err_pdf = joinpath(fig_dir, "brunner_weak_rational_error_vs_dof.pdf")
            Base.invokelatest(Plots.savefig, plt_err, err_pdf)
            println("PDF saved to: ", err_pdf)
        end

        valid_time = findall(i -> dofs[i] > 0.0 && times[i] > 0.0 && isfinite(times[i]), eachindex(dofs))
        if !isempty(valid_time)
            plt_time = Base.invokelatest(
                Plots.plot,
                dofs[valid_time],
                times[valid_time];
                xscale = :log10,
                yscale = :log10,
                linewidth = 2.2,
                marker = :circle,
                markersize = 4,
                label = nothing,
                xlabel = "dof",
                ylabel = "solve time (sec)",
                framestyle = :box,
                grid = false,
                size = (760, 460),
            )
            time_pdf = joinpath(fig_dir, "brunner_weak_rational_time_vs_dof.pdf")
            Base.invokelatest(Plots.savefig, plt_time, time_pdf)
            println("PDF saved to: ", time_pdf)
        end

        return true
    catch err
        @warn "Plotting skipped; CSV output is still available." exception = (err, catch_backtrace())
        return false
    end
end

function parse_int_list(text)
    vals = Int[]
    for part in split(text, ",")
        v = tryparse(Int, strip(part))
        v === nothing && throw(ArgumentError("invalid integer list: $(text)"))
        push!(vals, v)
    end
    return vals
end

function parse_cli_args(args)
    opts = Dict{Symbol,Any}(
        :Ms => copy(DEFAULT_MS),
        :m => DEFAULT_M,
        :q => DEFAULT_Q,
        :mu => DEFAULT_MU,
        :lambda => DEFAULT_LAMBDA,
        :nq => DEFAULT_NQ,
        :ref_factor => DEFAULT_REF_FACTOR,
        :Mref => nothing,
        :sample_count => DEFAULT_SAMPLE_COUNT,
        :csv => joinpath(DATA_DIR, "brunner_weak_rational_cauchy.csv"),
    )

    for arg in args
        if startswith(arg, "--Ms=")
            opts[:Ms] = parse_int_list(arg[6:end])
        elseif startswith(arg, "--Mref=")
            opts[:Mref] = parse(Int, arg[8:end])
        elseif startswith(arg, "--m=")
            opts[:m] = parse(Int, arg[5:end])
        elseif startswith(arg, "--q=")
            opts[:q] = parse(Float64, arg[5:end])
        elseif startswith(arg, "--mu=")
            opts[:mu] = parse(Float64, arg[6:end])
        elseif startswith(arg, "--lambda=")
            opts[:lambda] = parse(Float64, arg[10:end])
        elseif startswith(arg, "--nq=")
            opts[:nq] = parse(Int, arg[6:end])
        elseif startswith(arg, "--ref-factor=")
            opts[:ref_factor] = parse(Int, arg[14:end])
        elseif startswith(arg, "--sample-count=")
            opts[:sample_count] = parse(Int, arg[16:end])
        elseif startswith(arg, "--csv=")
            opts[:csv] = arg[7:end]
        elseif arg in ("-h", "--help")
            println("Usage: julia --project=. test/solve_brunner_weak_rational.jl [options]")
            println("Options:")
            println("  --Ms=32,64,128      comma-separated tested M values")
            println("  --Mref=4096         fixed reference M for all Cauchy errors")
            println("  --ref-factor=4      reference M is ref-factor*M when --Mref is absent")
            println("  --m=2               local collocation count")
            println("  --q=2.0             graded mesh exponent")
            println("  --mu=0.5            weak singular exponent")
            println("  --lambda=0.5        Volterra coupling")
            println("  --nq=64             Gauss-Legendre quadrature order")
            println("  --sample-count=2001 Cauchy sample grid size")
            println("  --csv=PATH          output CSV path")
            exit(0)
        else
            throw(ArgumentError("unknown argument: $(arg)"))
        end
    end

    return opts
end

function run_brunner_benchmark(;
    Ms = DEFAULT_MS,
    m = DEFAULT_M,
    q = DEFAULT_Q,
    mu = DEFAULT_MU,
    lambda = DEFAULT_LAMBDA,
    nq = DEFAULT_NQ,
    ref_factor = DEFAULT_REF_FACTOR,
    Mref = nothing,
    sample_count = DEFAULT_SAMPLE_COUNT,
    csv_path = joinpath(DATA_DIR, "brunner_weak_rational_cauchy.csv"),
)
    isempty(Ms) && throw(ArgumentError("Ms must be nonempty."))
    any(<=(0), Ms) && throw(ArgumentError("all M values must be positive."))
    ref_factor > 1 || throw(ArgumentError("ref_factor must be greater than 1."))

    mkpath(DATA_DIR)
    mkpath(FIGURES_DIR)

    Mrefs = Mref === nothing ? [ref_factor * M for M in Ms] : fill(Int(Mref), length(Ms))
    any(Mrefs .<= Ms) &&
        throw(ArgumentError("each reference M must be larger than the tested M."))

    needed_Ms = sort(unique(vcat(Int.(Ms), Int.(Mrefs))))
    solutions = Dict{Int,BrunnerSolution}()

    println("============================================================")
    println("Brunner-type graded mesh collocation / product integration")
    println("Problem: U(s) + lambda_hat * int_0^s (s-t)^(-mu) Khat(s,t) U(t) dt = G(s)")
    @printf("mu = %.16g, lambda = %.16g, lambda_hat = %.16g\n", mu, lambda, lambda * 2.0^(1.0 - mu))
    @printf("m = %d, c = %s, nq = %d\n", m, string(default_collocation_nodes(m)), nq)
    @printf("graded mesh: s_n = (n/M)^q, q = %.16g\n", q)
    if abs(q - 2.0) <= 100eps(Float64)
        println("q = 2 is used as the natural baseline for the typical s^(1-mu)=sqrt(s) left-endpoint behavior when mu = 0.5.")
    end
    @printf("sample_count = %d\n", sample_count)
    println("tested M values     = ", collect(Ms))
    println("reference M values  = ", Mrefs)
    println("solves to compute   = ", needed_Ms)
    println("============================================================")

    # Compile a tiny solve path outside the reported M rows.
    solve_brunner(2; m = m, q = q, mu = mu, lambda = lambda, nq = min(nq, 16))

    for Msolve in needed_Ms
        @printf("Solving M = %d (dof = %d) ...\n", Msolve, Msolve * m)
        sol = solve_brunner(Msolve; m = m, q = q, mu = mu, lambda = lambda, nq = nq)
        solutions[Msolve] = sol
        @printf("  done in %.6f s\n", sol.solve_time)
    end

    rows = NamedTuple[]
    println()
    println("Results:")
    for (M, Mr) in zip(Ms, Mrefs)
        sol = solutions[Int(M)]
        refsol = solutions[Int(Mr)]
        err = cauchy_error_from_solutions(sol, refsol; sample_count = sample_count)
        row = (
            M = Int(M),
            dof = dof(sol),
            m = sol.m,
            q = sol.q,
            mu = sol.mu,
            lambda = sol.lambda,
            solve_time = sol.solve_time,
            cauchy_err = err,
        )
        push!(rows, row)

        @printf(
            "M = %4d, dof = %5d, time = %.6e, cauchy_err = %.6e\n",
            row.M,
            row.dof,
            row.solve_time,
            row.cauchy_err,
        )
    end

    write_results_csv(csv_path, rows)
    println()
    println("CSV saved to: ", csv_path)
    maybe_plot_results(rows; fig_dir = FIGURES_DIR)
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    opts = parse_cli_args(ARGS)
    run_brunner_benchmark(
        Ms = opts[:Ms],
        m = opts[:m],
        q = opts[:q],
        mu = opts[:mu],
        lambda = opts[:lambda],
        nq = opts[:nq],
        ref_factor = opts[:ref_factor],
        Mref = opts[:Mref],
        sample_count = opts[:sample_count],
        csv_path = opts[:csv],
    )
end
