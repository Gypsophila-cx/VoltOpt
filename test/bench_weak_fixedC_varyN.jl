using Printf
using Statistics

# ============================================================
# Paths
# ============================================================

script_dir = @__DIR__
project_root = dirname(script_dir)
data_dir = joinpath(project_root, "test", "data")
mkpath(data_dir)

include(joinpath(project_root, "src", "VoltOpt.jl"))
using .VoltOpt

const WeakCheb = VoltOpt.Weak_VoltMat_Cheb

# ============================================================
# Test problem
# ============================================================

Kexp(x, y) = exp(-x * y)

# ============================================================
# Helpers
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
    memory_MiB = Float64[]
    allocs = Int[]

    t_start = time()
    for sample_id in 1:samples
        GC.gc()
        stat = @timed f()
        push!(times_sec, stat.time)
        push!(memory_MiB, stat.bytes / 1024.0^2)
        push!(allocs, allocation_count(stat.gcstats))

        if sample_id >= 1 && (time() - t_start) >= seconds
            break
        end
    end

    return (
        time_min_sec = minimum(times_sec),
        time_median_sec = median(times_sec),
        memory_MiB = minimum(memory_MiB),
        allocs = minimum(allocs),
        samples_run = length(times_sec),
    )
end

function build_duffy_coefficients(K, C_rows::Int, C_cols::Int, beta::Real)
    C_rows > 1 || throw(ArgumentError("C_rows must be at least 2"))
    C_cols > 1 || throw(ArgumentError("C_cols must be at least 2"))

    stat = @timed begin
        F = WeakCheb.sample_duffy_kernel(K, C_rows - 1, C_cols - 1, beta)
        WeakCheb.cheb_coeffs_lobatto_2d(F)
    end

    return stat.value, stat.time
end

function weak_build(C, N::Int; mu::Real, sigma::Real, beta::Real)
    return VoltOpt.weak_voltmat_cheb(
        C;
        C_is_coeffs = true,
        mu = mu,
        sigma = sigma,
        beta = beta,
        N = N,
        output_format = :packed_lower_hessenberg,
        diagnostics = false,
        H_recurrence_mode = :block_i,
        return_info = true,
    )
end

function structural_nnz(V)
    return V isa VoltOpt.PackedLowerHessenberg ? length(V.values) : count(!iszero, V)
end

function upper_bandwidth(V)
    return size(V, 2) - 1
end

function csv_cell(x)
    x === nothing && return "missing"
    x === missing && return "missing"
    return string(x)
end

function write_csv(csv_file::AbstractString, rows::Vector{<:NamedTuple})
    headers = [
        "N",
        "C_rows",
        "C_cols",
        "C_size",
        "mu",
        "sigma",
        "beta",
        "eta",
        "bandwidth",
        "lower_bw",
        "upper_bw",
        "top_rows",
        "nnz",
        "stored_entries",
        "output_format",
        "assembly_mode",
        "H_recurrence_mode",
        "t_build",
        "t_build_median",
        "t_coeff",
        "t_total",
        "memory_MiB",
        "allocs",
        "samples_run",
        "timing_initial_moments",
        "timing_precombine",
        "timing_assembly",
        "timing_multiplier",
        "timing_diagnostics",
        "timing_total_reported",
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

function bench_weak_fixedC_varyN(;
    C_size::Tuple{Int,Int} = (32, 32),
    N_values::Vector{Int} = [100, 200, 400, 800, 1600, 3200],
    mu::Float64 = 0.5,
    sigma::Float64 = 0.0,
    beta::Float64 = 0.5,
    samples::Int = 3,
    seconds::Float64 = 2.0,
    output_csv::String = joinpath(data_dir, "bench_weak_fixedC_varyN.csv"),
)
    C_rows, C_cols = C_size
    C, t_coeff = build_duffy_coefficients(Kexp, C_rows, C_cols, beta)
    results = NamedTuple[]

    println("============================================================")
    println("Benchmark: weakly singular Chebyshev/JFP Volterra matrix")
    println("case                    = fixed C/bandwidth, varying N")
    println("kernel                  = exp(-x*y)")
    println("weak kernel factor      = (x-y)^(-mu)")
    println("C size                  = $(C_rows) x $(C_cols)")
    println("N values                = $N_values")
    println("mu, sigma, beta         = ($mu, $sigma, $beta)")
    println("coefficient time        = $t_coeff")
    println("output_format           = packed_lower_hessenberg")
    println("samples                 = $samples")
    println("seconds                 = $seconds")
    println("script_dir              = $script_dir")
    println("project_root            = $project_root")
    println("data_dir                = $data_dir")
    println("output_csv              = $output_csv")
    println("============================================================")

    # Compile the main path outside the measured loops.
    weak_build(C, min(first(N_values), 64); mu = mu, sigma = sigma, beta = beta)

    for N in N_values
        println()
        println("------------------------------------------------------------")
        @printf("N = %d, C = %d x %d\n", N, C_rows, C_cols)

        Vwarm, info = weak_build(C, N; mu = mu, sigma = sigma, beta = beta)
        lower_bw = info.lower_bandwidth_theory
        upper_bw = upper_bandwidth(Vwarm)
        nnz_val = structural_nnz(Vwarm)

        build_stat = bench_call(
            () -> weak_build(C, N; mu = mu, sigma = sigma, beta = beta);
            samples = samples,
            seconds = seconds,
        )

        r = (
            N = N,
            C_rows = C_rows,
            C_cols = C_cols,
            C_size = C_rows * C_cols,
            mu = mu,
            sigma = sigma,
            beta = beta,
            eta = info.eta,
            bandwidth = lower_bw,
            lower_bw = lower_bw,
            upper_bw = upper_bw,
            top_rows = missing,
            nnz = nnz_val,
            stored_entries = info.stored_entries,
            output_format = info.output_format,
            assembly_mode = info.assembly_mode,
            H_recurrence_mode = info.H_recurrence_mode,
            t_build = build_stat.time_min_sec,
            t_build_median = build_stat.time_median_sec,
            t_coeff = t_coeff,
            t_total = t_coeff + build_stat.time_min_sec,
            memory_MiB = build_stat.memory_MiB,
            allocs = build_stat.allocs,
            samples_run = build_stat.samples_run,
            timing_initial_moments = info.timings.initial_moments,
            timing_precombine = info.timings.precombine,
            timing_assembly = info.timings.assembly,
            timing_multiplier = info.timings.multiplier,
            timing_diagnostics = info.timings.diagnostics,
            timing_total_reported = info.timings.total,
        )

        push!(results, r)

        @printf(
            "Weak | build = %10.6f s | coeff = %10.6f s | nnz = %10d | lower_bw = %4d | upper_bw = %4d | mem = %8.2f MiB\n",
            r.t_build,
            r.t_coeff,
            r.nnz,
            r.lower_bw,
            r.upper_bw,
            r.memory_MiB,
        )
    end

    write_csv(output_csv, results)
    println()
    println("CSV saved to: ", output_csv)

    return results
end

# ============================================================
# Run
# ============================================================

results = bench_weak_fixedC_varyN(
    C_size = (32, 32),
    N_values = unique(round.(Int, exp.(range(log(100), log(10000), length=20)))),
    mu = 0.5,
    sigma = 0.0,
    beta = 0.5,
    samples = 3,
    seconds = 2.0,
    output_csv = joinpath(data_dir, "bench_weak_fixedC_varyN.csv"),
)
