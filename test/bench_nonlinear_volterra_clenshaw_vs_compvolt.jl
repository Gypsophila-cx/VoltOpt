using LinearAlgebra
using Printf
using Statistics

# ============================================================
# Paths
# ============================================================

script_dir = @__DIR__
project_root = dirname(script_dir)
data_dir = joinpath(project_root, "test", "data")
fig_dir = joinpath(project_root, "test", "figures")
mkpath(data_dir)
mkpath(fig_dir)

include(joinpath(project_root, "src", "VoltOpt.jl"))
using .VoltOpt

include(joinpath(project_root, "src", "CompVolt", "clenshaw_valuemat.jl"))

# ============================================================
## Test problem
# ============================================================

const Ns_clenshaw = [16, 32, 64, 128, 256, 512, 1024]
const Ns_compvolt = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

Kfun(x, t, u) = u / (1.0 + (x - t) * u^2)
utest(x) = 1.0 + 0.2 * cospi(x) + 0.1 * sinpi(2.0 * x)

function cheb_nodes_ascending(N::Int)
    return [-cospi(i / N) for i in 0:N]
end

function ascending_values_to_cheb_coeffs(vals::AbstractVector{<:Real})
    coeffs = VoltOpt.vals2cheb(vals)
    @inbounds for i in 2:2:length(coeffs)
        coeffs[i] = -coeffs[i]
    end
    return Vector{Float64}(coeffs)
end

function smooth_test_coefficients(N::Int)
    x = cheb_nodes_ascending(N)
    return ascending_values_to_cheb_coeffs(utest.(x))
end

# ============================================================
# Nonlinear Volterra actions
# ============================================================

function nonlinear_volterra_action_clenshaw(
    cu::Vector{Float64},
    N::Int,
    weights::AbstractVector{Float64},
)
    x_nodes, eta_nodes, U_grid = clenshaw_valuemat(cu, N + 1, N + 1)

    Fval = zeros(Float64, N + 1)
    @inbounds for j in 1:(N + 1)
        xj = x_nodes[j]
        scale = (xj + 1.0) / 2.0

        if j == 1
            Fval[j] = 0.0
            continue
        end

        shift = (xj - 1.0) / 2.0
        s = 0.0
        @simd for i in 1:(N + 1)
            t = scale * eta_nodes[i] + shift
            s += weights[i] * Kfun(xj, t, U_grid[j, i])
        end
        Fval[j] = scale * s
    end

    return ascending_values_to_cheb_coeffs(Fval)
end

function nonlinear_volterra_action_compvolt(
    cu::Vector{Float64},
    grid::VoltOpt.CompVoltGrid,
)
    return Vector{Float64}(
        VoltOpt.compVolt_nonlinear_fast(Kfun, cu; grid = grid, truncate_output = true),
    )
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
    headers = [
        "method",
        "N",
        "time",
        "allocs",
        "memory",
        "check_norm",
        "time_median",
        "samples_run",
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
    check_norms = Dict{Int,Float64}()

    println("============================================================")
    println("Benchmark: nonlinear Volterra action")
    println("Kernel                  = u / (1 + (x - t) * u^2)")
    println("u(x)                    = 1 + 0.2cos(pi*x) + 0.1sin(2pi*x)")
    println("clenshaw N values       = $(Ns_clenshaw)")
    println("compVolt N values       = $(Ns_compvolt)")
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
        @printf("method = clenshaw_valuemat, N = %d\n", N)

        cu = smooth_test_coefficients(N)
        weights = VoltOpt.clenshaw_curtis_analytical_weights(N)
        grid = VoltOpt.CompVoltGrid(N)

        # Warm-up both paths on the shared input and retain a small consistency
        # check for the overlapping N values.
        clenshaw_warm = nonlinear_volterra_action_clenshaw(cu, N, weights)
        compvolt_warm = nonlinear_volterra_action_compvolt(cu, grid)
        check_norm = norm(clenshaw_warm - compvolt_warm, Inf)
        check_norms[N] = check_norm

        stat = bench_call(
            () -> nonlinear_volterra_action_clenshaw(cu, N, weights);
            samples = samples,
            seconds = seconds,
        )

        push!(
            results,
            (
                method = "clenshaw_valuemat",
                N = N,
                time = stat.time_min_sec,
                allocs = stat.allocs,
                memory = stat.memory_bytes,
                check_norm = check_norm,
                time_median = stat.time_median_sec,
                samples_run = stat.samples_run,
            ),
        )

        @printf(
            "clenshaw_valuemat | best = %.6e s | median = %.6e s | mem = %.3f MiB | allocs = %d | check_norm = %.3e\n",
            stat.time_min_sec,
            stat.time_median_sec,
            stat.memory_bytes / 1024.0^2,
            stat.allocs,
            check_norm,
        )
    end

    for N in Ns_compvolt
        println()
        println("------------------------------------------------------------")
        @printf("method = compVolt, N = %d\n", N)

        cu = smooth_test_coefficients(N)
        grid = VoltOpt.CompVoltGrid(N)

        # Warm-up the measured path before timing.
        nonlinear_volterra_action_compvolt(cu, grid)
        check_norm = get(check_norms, N, NaN)

        stat = bench_call(
            () -> nonlinear_volterra_action_compvolt(cu, grid);
            samples = samples,
            seconds = seconds,
        )

        push!(
            results,
            (
                method = "compVolt",
                N = N,
                time = stat.time_min_sec,
                allocs = stat.allocs,
                memory = stat.memory_bytes,
                check_norm = check_norm,
                time_median = stat.time_median_sec,
                samples_run = stat.samples_run,
            ),
        )

        @printf(
            "compVolt          | best = %.6e s | median = %.6e s | mem = %.3f MiB | allocs = %d | check_norm = %s\n",
            stat.time_min_sec,
            stat.time_median_sec,
            stat.memory_bytes / 1024.0^2,
            stat.allocs,
            isfinite(check_norm) ? @sprintf("%.3e", check_norm) : "",
        )
    end

    write_csv(output_csv, results)

    println()
    println("CSV saved to: ", output_csv)
    return results
end

bench_nonlinear_volterra_clenshaw_vs_compvolt(
    samples = 5,
    seconds = 60.0,
    output_csv = joinpath(data_dir, "bench_nonlinear_volterra_clenshaw_vs_compvolt.csv"),
)
