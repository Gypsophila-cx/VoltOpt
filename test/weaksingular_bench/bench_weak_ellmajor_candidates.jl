using BenchmarkTools
using Printf
using Random

include(joinpath(@__DIR__, "..", "..", "src", "VoltOpt.jl"))
const W = VoltOpt.Weak_VoltMat_Cheb

function parse_int_list(name, default)
    raw = get(ENV, name, default)
    return parse.(Int, split(raw, ','))
end

function stats(f; samples::Int, seconds::Float64)
    GC.gc()
    bench = @benchmarkable ($f)() evals = 1
    trial = run(bench; samples = samples, seconds = seconds)
    med = median(trial)
    return (
        minimum_seconds = minimum(trial).time / 1e9,
        median_seconds = med.time / 1e9,
        memory_bytes = med.memory,
        allocations = med.allocs,
    )
end

function deterministic_coefficients(I::Int, J::Int)
    rng = MersenneTwister(0x454c4c + 257I + J)
    C = 0.04 .* randn(rng, I + 1, J + 1)
    C[1, 1] += 1.0
    return C
end

function run_case!(rows, I::Int, J::Int, N::Int, case_group::String;
                   samples::Int, seconds::Float64)
    C = deterministic_coefficients(I, J)
    Minit = W.initial_moments_fast(0.5, 0.0, 0.5, J, N;
                                   de_qfactor = 12,
                                   accurate_sum = false)
    Mcomb = W._precombine_kernel_moments(C, Minit, N, J; mode = :blas)
    lower_bw = I + 1

    current() = W._assemble_matrix_precombined_eta1_kmajor_from_Mcomb(
        copy(Mcomb), N, 0.0;
        output_format = :packed_lower_hessenberg,
        lower_bw = lower_bw,
    )
    scatter() = W._assemble_eta1_ellmajor_scatter_from_Mcomb(
        copy(Mcomb), N, 0.0;
        output_format = :packed_lower_hessenberg,
        lower_bw = lower_bw,
    )
    scatter_division() = W._assemble_eta1_ellmajor_scatter_from_Mcomb(
        copy(Mcomb), N, 0.0;
        output_format = :packed_lower_hessenberg,
        lower_bw = lower_bw,
        use_reciprocal_table = false,
    )
    gather() = W._assemble_eta1_ellmajor_gather_from_Mcomb(
        Mcomb, N, 0.0;
        output_format = :packed_lower_hessenberg,
        lower_bw = lower_bw,
    )

    # Compile every measured path before timing.  Compare each candidate to a
    # fresh current-path result because the current implementation reuses and
    # overwrites its Mcomb argument as one recurrence buffer.
    reference = current()
    scatter_result = scatter()
    scatter_division_result = scatter_division()
    gather_result = gather()
    scatter_error = maximum(abs, reference.values .- scatter_result.values)
    scatter_division_error =
        maximum(abs, reference.values .- scatter_division_result.values)
    gather_error = maximum(abs, reference.values .- gather_result.values)

    @printf("\n%s I=%d J=%d N=%d scatter_err=%.3e gather_err=%.3e\n",
            case_group, I, J, N, scatter_error, gather_error)
    for (candidate, f, err) in (
        ("current_k_major", current, 0.0),
        ("ell_major_local_scatter", scatter, scatter_error),
        ("ell_major_scatter_division", scatter_division, scatter_division_error),
        ("ell_major_direct_gather", gather, gather_error),
    )
        s = stats(f; samples = samples, seconds = seconds)
        @printf("%-28s min=%9.6f s median=%9.6f s memory=%9.3f MiB\n",
                candidate, s.minimum_seconds, s.median_seconds,
                s.memory_bytes / 2.0^20)
        push!(rows, merge((;
            case_group,
            I,
            J,
            N,
            candidate,
            max_abs_error = err,
        ), s))
    end
    return nothing
end

function write_csv(path, rows)
    mkpath(dirname(path))
    fields = propertynames(first(rows))
    open(path, "w") do io
        println(io, join(fields, ','))
        for row in rows
            println(io, join((getproperty(row, f) for f in fields), ','))
        end
    end
end

function main()
    Ns = parse_int_list("WEAK_ELLMAJOR_NS", "512,1000,2000,4000")
    Is = parse_int_list("WEAK_ELLMAJOR_IS", "31,127")
    fixed_N = parse(Int, get(ENV, "WEAK_ELLMAJOR_FIXED_N", "1000"))
    samples = parse(Int, get(ENV, "WEAK_ELLMAJOR_SAMPLES", "2"))
    seconds = parse(Float64, get(ENV, "WEAK_ELLMAJOR_SECONDS", "30"))
    output = get(ENV, "WEAK_ELLMAJOR_OUTPUT",
                 joinpath(@__DIR__, "data", "bench_weak_ellmajor_candidates.csv"))

    rows = NamedTuple[]
    for N in Ns
        run_case!(rows, 31, 31, N, "vary_N";
                  samples = samples, seconds = seconds)
    end
    for I in Is
        run_case!(rows, I, I, fixed_N, "vary_I";
                  samples = samples, seconds = seconds)
    end
    write_csv(output, rows)
    println("wrote $output")
end

main()
