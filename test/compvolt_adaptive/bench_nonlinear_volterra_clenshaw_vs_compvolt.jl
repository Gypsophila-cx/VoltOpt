# bench_nonlinear_volterra_clenshaw_vs_compvolt.jl
#
# Purpose:
#   Benchmark two complete adaptive nonlinear Volterra action implementations
#   on the same degree-N polynomial input.
#
# Benchmark:
#   Compares a Clenshaw value-matrix quadrature path with the VoltOpt
#   compVolt_nonlinear_fast path for
#   K(x,t,u) = u / (1 + (x - t)u^2).
#
# Method:
#   Both methods start at M0=N, double the Chebyshev grid degree, apply the
#   same eta/theta acceptance test, and return the full resolved coefficient
#   vector. Timings include every adaptive level and all per-call preparation.
#
# Output files:
#   data/bench_nonlinear_volterra_clenshaw_vs_compvolt.csv
#
# Run from the repository root:
#   julia --project=. test/compvolt_adaptive/bench_nonlinear_volterra_clenshaw_vs_compvolt.jl

using LinearAlgebra
using Printf
using Statistics

# ============================================================
# Paths
# ============================================================

function find_project_root(start_dir::AbstractString)
    dir = abspath(start_dir)
    while true
        if isfile(joinpath(dir, "Project.toml")) && isdir(joinpath(dir, "src"))
            return dir
        end
        parent = dirname(dir)
        parent == dir && error("Could not locate VoltOpt project root from $start_dir")
        dir = parent
    end
end

script_dir = @__DIR__
project_root = find_project_root(script_dir)
data_dir = joinpath(@__DIR__, "data")
fig_dir = joinpath(@__DIR__, "figures")
mkpath(data_dir)
mkpath(fig_dir)

include(joinpath(project_root, "src", "VoltOpt.jl"))
using .VoltOpt

include(joinpath(project_root, "src", "CompVolt", "clenshaw_valuemat.jl"))

# ============================================================
## Test problem
# ============================================================

const Ns_clenshaw = [16, 32, 64, 128, 256, 512, 1024]
const Ns_compvolt = copy(Ns_clenshaw)
const tol_quad = 100eps(Float64)
const Mmax = 8192
const max_refinements = 8
const consistency_tol = 100tol_quad

Kfun(x, t, u) = u / (1.0 + (x - t) * u^2)

function smooth_test_coefficients(N::Int)
    N >= 1 || throw(ArgumentError("N must be at least 1."))
    tail_weights = [exp(-12j / N) for j in 1:N]
    normalization = sum(tail_weights)
    cu = Vector{Float64}(undef, N + 1)
    cu[1] = 1.0
    @inbounds for j in 1:N
        cu[j + 1] = 0.2 * tail_weights[j] / normalization
    end
    return cu
end

# ============================================================
# Nonlinear Volterra actions
# ============================================================

function clenshaw_level_values(
    cu::Vector{Float64},
    grid::VoltOpt.CompVoltGrid,
    level::Symbol,
)
    M = grid.N
    x_nodes, eta_nodes, U_grid = clenshaw_valuemat(cu, grid.x, grid.x)
    VoltOpt._cv_check_finite(
        U_grid, "Clenshaw value matrix", M, :adaptive, level,
    )

    Fval = zeros(Float64, M + 1)
    @inbounds for j in 2:(M + 1)
        xj = x_nodes[j]
        scale = grid.scale[j]
        shift = grid.shift[j]
        s = 0.0
        @simd for i in 1:(M + 1)
            t = scale * eta_nodes[i] + shift
            value = Kfun(xj, t, U_grid[j, i])
            VoltOpt._cv_check_finite(
                value, "kernel function value", M, :adaptive, level,
            )
            s += grid.w[i] * value
        end
        VoltOpt._cv_check_finite(
            s, "Clenshaw--Curtis row sum", M, :adaptive, level,
        )
        Fval[j] = scale * s
        VoltOpt._cv_check_finite(
            Fval[j], "Clenshaw--Curtis integral value", M, :adaptive, level,
        )
    end
    VoltOpt._cv_check_finite(
        Fval, "Clenshaw--Curtis integral values", M, :adaptive, level,
    )
    return Fval
end

function nonlinear_volterra_action_clenshaw(
    cu::Vector{Float64};
    M0::Int=length(cu) - 1,
    tol::Real=tol_quad,
    Mmax::Int=Mmax,
    max_refinements::Int=max_refinements,
)
    isempty(cu) && return (
        coefficients=Float64[], M=0, eta=nothing, theta=nothing,
        refinements=0, converged=true,
    )
    max_refinements >= 0 || throw(ArgumentError("max_refinements must be nonnegative."))
    Mmax >= 1 || throw(ArgumentError("Mmax must be at least 1."))
    isfinite(tol) && tol >= 0 || throw(ArgumentError("tol must be finite and nonnegative."))

    input_degree = length(cu) - 1
    Mcoarse = max(input_degree, M0, 1)
    Mcoarse <= Mmax || throw(ArgumentError(
        "Mmax=$Mmax is smaller than the initial degree M=$Mcoarse.",
    ))

    coarse_grid = VoltOpt._cv_exact_grid(Mcoarse, Float64)
    coarse_values = clenshaw_level_values(cu, coarse_grid, :coarse)
    coarse_coefficients = nothing
    eta = nothing
    theta = nothing
    refinements = 0

    while true
        if refinements >= max_refinements
            coefficients = coarse_coefficients === nothing ?
                VoltOpt._cv_output_coefficients(
                    coarse_values, Mcoarse, :adaptive, :coarse,
                ) : coarse_coefficients
            return (
                coefficients=coefficients,
                M=Mcoarse,
                eta=eta,
                theta=theta,
                refinements=refinements,
                converged=false,
            )
        end

        Mfine = 2Mcoarse
        if Mfine > Mmax
            coefficients = coarse_coefficients === nothing ?
                VoltOpt._cv_output_coefficients(
                    coarse_values, Mcoarse, :adaptive, :coarse,
                ) : coarse_coefficients
            return (
                coefficients=coefficients,
                M=Mcoarse,
                eta=eta,
                theta=theta,
                refinements=refinements,
                converged=false,
            )
        end

        fine_grid = VoltOpt._cv_exact_grid(Mfine, Float64; coarse=coarse_grid)
        fine_values = clenshaw_level_values(cu, fine_grid, :fine)
        fine_coefficients = VoltOpt._cv_output_coefficients(
            fine_values, Mfine, :adaptive, :fine,
        )
        refinements += 1
        eta, theta = VoltOpt._cv_refinement_indicators(
            coarse_values, fine_values, fine_coefficients, Mcoarse,
        )

        if max(eta, theta) <= tol
            return (
                coefficients=fine_coefficients,
                M=Mfine,
                eta=eta,
                theta=theta,
                refinements=refinements,
                converged=true,
            )
        end

        Mcoarse = Mfine
        coarse_grid = fine_grid
        coarse_values = fine_values
        coarse_coefficients = fine_coefficients
    end
end

function nonlinear_volterra_action_compvolt(
    cu::Vector{Float64};
    M0::Int=length(cu) - 1,
    tol::Real=tol_quad,
    Mmax::Int=Mmax,
    max_refinements::Int=max_refinements,
)
    return VoltOpt.compVolt_nonlinear_fast(
        Kfun,
        cu;
        adaptive=true,
        M0=M0,
        tol=tol,
        Mmax=Mmax,
        max_refinements=max_refinements,
        return_info=true,
        truncate_output=false,
    )
end

function padded_coefficient_difference(a::AbstractVector, b::AbstractVector)
    ncoeffs = max(length(a), length(b))
    difference = zeros(Float64, ncoeffs)
    difference[1:length(a)] .+= a
    difference[1:length(b)] .-= b
    return norm(difference, Inf)
end

function validate_adaptive_result(method::AbstractString, N::Int, info, tol::Real)
    info.converged || error(
        "$method did not resolve N=$N: final M=$(info.M), eta=$(info.eta), " *
        "theta=$(info.theta), refinements=$(info.refinements).",
    )
    info.eta !== nothing && info.theta !== nothing || error(
        "$method returned missing adaptive indicators for N=$N, M=$(info.M).",
    )
    max(info.eta, info.theta) <= tol || error(
        "$method accepted N=$N, M=$(info.M) with eta=$(info.eta), " *
        "theta=$(info.theta), exceeding tol=$tol.",
    )
    length(info.coefficients) == info.M + 1 || error(
        "$method returned $(length(info.coefficients)) coefficients at M=$(info.M).",
    )
    all(isfinite, info.coefficients) || error(
        "$method returned non-finite coefficients for N=$N, M=$(info.M).",
    )
    return info
end

# ============================================================
# Benchmark helpers
# ============================================================

function allocation_count(gcstats)
    total = 0
    for name in (:poolalloc, :bigalloc, :malloc, :realloc)
        if hasproperty(gcstats, name)
            total += Int(getproperty(gcstats, name))
        end
    end
    return total
end

function bench_call(f; samples::Int = 3, seconds::Float64 = 2.0)
    GC.gc()

    times_sec = Float64[]
    memory_bytes = Int[]
    allocs = Int[]

    t_start = time()
    for sample_id in 1:samples
        GC.gc()
        stat = @timed f()
        push!(times_sec, stat.time)
        push!(memory_bytes, stat.bytes)
        push!(allocs, allocation_count(stat.gcstats))

        if sample_id >= 1 && (time() - t_start) >= seconds
            break
        end
    end

    return (
        time_min_sec = minimum(times_sec),
        time_median_sec = median(times_sec),
        memory_bytes = minimum(memory_bytes),
        allocs = minimum(allocs),
        samples_run = length(times_sec),
    )
end

function csv_cell(x)
    x === nothing && return ""
    x === missing && return ""
    x isa AbstractFloat && !isfinite(x) && return ""
    return string(x)
end

function write_csv(csv_file::AbstractString, rows::Vector{<:NamedTuple})
    mkpath(dirname(csv_file))
    headers = [
        "method",
        "N",
        "time",
        "allocs",
        "memory",
        "check_norm",
        "time_median",
        "samples_run",
        "M",
        "eta",
        "theta",
        "refinements",
        "converged",
    ]

    open(csv_file, "w") do io
        println(io, join(headers, ","))
        for r in rows
            println(io, join((csv_cell(getfield(r, Symbol(h))) for h in headers), ","))
        end
    end
end

# ============================================================
# Main benchmark
# ============================================================

function bench_nonlinear_volterra_clenshaw_vs_compvolt(;
    samples::Int = 3,
    seconds::Float64 = 2.0,
    output_csv::String = joinpath(data_dir, "bench_nonlinear_volterra_clenshaw_vs_compvolt.csv"),
)
    results = NamedTuple[]
    Ns_clenshaw == Ns_compvolt || error(
        "This benchmark requires identical N lists for both methods.",
    )

    println("============================================================")
    println("Benchmark: complete adaptive nonlinear Volterra action")
    println("Kernel                  = u / (1 + (x - t) * u^2)")
    println("input coefficients      = cu[1]=1, cu[j+1]=0.2exp(-12j/N)/sum(exp(-12l/N), l=1:N)")
    println("clenshaw N values       = $(Ns_clenshaw)")
    println("compVolt N values       = $(Ns_compvolt)")
    @printf("tol_quad                = %.16e\n", tol_quad)
    @printf("consistency tolerance   = %.16e times output scale\n", consistency_tol)
    println("M0                      = N")
    println("Mmax                    = $(Mmax)")
    println("max_refinements         = $(max_refinements)")
    println("Julia threads           = $(Threads.nthreads())")
    println("samples                 = $(samples)")
    println("seconds                 = $(seconds)")
    println("script_dir              = $(script_dir)")
    println("project_root            = $(project_root)")
    println("data_dir                = $(data_dir)")
    println("output_csv              = $(output_csv)")
    println("============================================================")

    for N in Ns_clenshaw
        println()
        println("------------------------------------------------------------")
        @printf("shared input degree N = %d\n", N)

        cu = smooth_test_coefficients(N)
        kwargs = (;
            M0=N,
            tol=tol_quad,
            Mmax=Mmax,
            max_refinements=max_refinements,
        )

        # JIT warm-up and consistency checking are deliberately outside timing.
        clenshaw_warm = validate_adaptive_result(
            "clenshaw_valuemat",
            N,
            nonlinear_volterra_action_clenshaw(cu; kwargs...),
            tol_quad,
        )
        compvolt_warm = validate_adaptive_result(
            "compVolt",
            N,
            nonlinear_volterra_action_compvolt(cu; kwargs...),
            tol_quad,
        )
        check_norm = padded_coefficient_difference(
            clenshaw_warm.coefficients, compvolt_warm.coefficients,
        )
        output_scale = max(
            1.0,
            maximum(abs, clenshaw_warm.coefficients),
            maximum(abs, compvolt_warm.coefficients),
        )
        check_norm <= consistency_tol * output_scale || error(
            "Implementation mismatch at N=$N: check_norm=$check_norm, " *
            "allowed=$(consistency_tol * output_scale), " *
            "Clenshaw M=$(clenshaw_warm.M), compVolt M=$(compvolt_warm.M).",
        )

        clenshaw_stat = bench_call(
            () -> nonlinear_volterra_action_clenshaw(cu; kwargs...);
            samples = samples,
            seconds = seconds,
        )
        push!(
            results,
            (
                method = "clenshaw_valuemat",
                N = N,
                time = clenshaw_stat.time_min_sec,
                allocs = clenshaw_stat.allocs,
                memory = clenshaw_stat.memory_bytes,
                check_norm = check_norm,
                time_median = clenshaw_stat.time_median_sec,
                samples_run = clenshaw_stat.samples_run,
                M = clenshaw_warm.M,
                eta = clenshaw_warm.eta,
                theta = clenshaw_warm.theta,
                refinements = clenshaw_warm.refinements,
                converged = clenshaw_warm.converged,
            ),
        )
        @printf(
            "clenshaw_valuemat | median = %.6e s | M = %d | eta = %.3e | theta = %.3e | refinements = %d | check_norm = %.3e\n",
            clenshaw_stat.time_median_sec,
            clenshaw_warm.M,
            clenshaw_warm.eta,
            clenshaw_warm.theta,
            clenshaw_warm.refinements,
            check_norm,
        )

        compvolt_stat = bench_call(
            () -> nonlinear_volterra_action_compvolt(cu; kwargs...);
            samples = samples,
            seconds = seconds,
        )
        push!(
            results,
            (
                method = "compVolt",
                N = N,
                time = compvolt_stat.time_min_sec,
                allocs = compvolt_stat.allocs,
                memory = compvolt_stat.memory_bytes,
                check_norm = check_norm,
                time_median = compvolt_stat.time_median_sec,
                samples_run = compvolt_stat.samples_run,
                M = compvolt_warm.M,
                eta = compvolt_warm.eta,
                theta = compvolt_warm.theta,
                refinements = compvolt_warm.refinements,
                converged = compvolt_warm.converged,
            ),
        )
        @printf(
            "compVolt          | median = %.6e s | M = %d | eta = %.3e | theta = %.3e | refinements = %d | check_norm = %.3e\n",
            compvolt_stat.time_median_sec,
            compvolt_warm.M,
            compvolt_warm.eta,
            compvolt_warm.theta,
            compvolt_warm.refinements,
            check_norm,
        )
    end

    write_csv(output_csv, results)

    println()
    println("CSV saved to: ", output_csv)
    return results
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    bench_nonlinear_volterra_clenshaw_vs_compvolt(
        samples = 5,
        seconds = 60.0,
        output_csv = joinpath(data_dir, "bench_nonlinear_volterra_clenshaw_vs_compvolt.csv"),
    )
end
