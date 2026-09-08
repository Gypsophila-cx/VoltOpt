# Internal component benchmark for the current weak-singular construction.
# This is deliberately separate from the formal end-to-end benchmark CSVs.

using BenchmarkTools
using LinearAlgebra
using Printf
using Random

project_root = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(project_root, "src", "VoltOpt.jl"))
using .VoltOpt

const W = VoltOpt.Weak_VoltMat_Cheb

function parse_int_list(name::String, default::String)
    return parse.(Int, split(get(ENV, name, default), ','))
end

function trial_stats(f; samples::Int, seconds::Float64)
    GC.gc()
    bench = @benchmarkable ($f)() evals=1
    trial = run(bench; samples=samples, seconds=seconds)
    isempty(trial.times) && error("BenchmarkTools returned no samples")
    ts = sort(Float64.(trial.times))
    mid = length(ts) >> 1
    med = isodd(length(ts)) ? ts[mid + 1] : 0.5 * (ts[mid] + ts[mid + 1])
    return (
        minimum = first(ts) * 1.0e-9,
        median = med * 1.0e-9,
        memory_mib = trial.memory / 1024.0^2,
        allocs = trial.allocs,
        samples = length(ts),
    )
end

function recurrence_only!(source, current, next, N::Int)
    copyto!(current, source)
    I1 = size(source, 1)
    for k in 0:(N - 1)
        W._blocki_next_layer!(next, current, k, N, I1)
        current, next = next, current
    end
    return current
end

function _next_layer_reciprocal!(next, current, invint, k::Int, N::Int, I1::Int)
    @inbounds @simd for ii in 1:I1
        next[ii, k + 1] = 0.0
    end
    @inbounds for ell in (k + 1):N
        invden = invint[ell + k]
        a = (ell - k) * invden
        c = -(ell - k - 2) * invden
        @simd for ii in 1:I1
            next[ii, ell + 1] =
                a * current[ii, ell + 1] +
                current[ii, ell] +
                c * next[ii, ell]
        end
    end
    return next
end

function recurrence_only_reciprocal!(source, current, next, invint, N::Int)
    copyto!(current, source)
    I1 = size(source, 1)
    for k in 0:(N - 1)
        _next_layer_reciprocal!(next, current, invint, k, N, I1)
        current, next = next, current
    end
    return current
end

function recurrence_inner_scatter!(S, source, current, next, row_scale,
                                   I::Int, N::Int)
    fill!(S.values, 0.0)
    copyto!(current, source)
    I1 = I + 1
    for k in 0:N
        W._blocki_scatter_S_layer!(S, current, row_scale, k, I, N)
        k == N && break
        W._blocki_next_layer!(next, current, k, N, I1)
        current, next = next, current
    end
    return S
end

function recurrence_fused_eta1_scatter!(V, source, current, next, row_scale,
                                        I::Int, N::Int)
    fill!(V.values, 0.0)
    copyto!(current, source)
    I1 = I + 1
    for k in 0:N
        W._blocki_scatter_eta1_layer!(V, current, row_scale, k, I, N)
        k == N && break
        W._blocki_next_layer!(next, current, k, N, I1)
        current, next = next, current
    end
    return V
end

function full_build(C, N::Int)
    return VoltOpt.weak_voltmat_cheb(
        C;
        C_is_coeffs=true,
        mu=0.5,
        sigma=0.0,
        beta=0.5,
        N=N,
        output_format=:packed_lower_hessenberg,
        diagnostics=false,
        H_recurrence_mode=:block_i,
        return_info=true,
    )
end

function write_row(io, label, stage, N, I, J, stat)
    println(io, join((
        label, stage, N, I, J, stat.minimum, stat.median,
        stat.memory_mib, stat.allocs, stat.samples,
    ), ','))
    flush(io)
end

function run_case!(io, label::String, N::Int, I::Int, J::Int,
                   samples::Int, seconds::Float64)
    @printf("\ncase label=%s N=%d I=%d J=%d\n", label, N, I, J)
    rng = MersenneTwister(10_000N + 100I + J)
    C = 0.02 .* randn(rng, I + 1, J + 1)
    C[1, 1] += 1.0

    # Construct immutable inputs once for recurrence/assembly measurements.
    Minit = W.initial_moments_fast(0.5, 0.0, 0.5, J, N)
    Mcomb = W._precombine_kernel_moments(C, Minit, N, J)
    current = similar(Mcomb)
    next = similar(Mcomb)
    invint = W._ellmajor_inverse_integers(N)
    row_scale = W._four_point_row_scale(N, 0.0)

    # Warm every measured path before BenchmarkTools samples.
    W.initial_moments_fast(0.5, 0.0, 0.5, J, min(N, 64))
    W._precombine_kernel_moments(C, Minit, N, J)
    recurrence_only!(Mcomb, current, next, N)
    recurrence_only_reciprocal!(Mcomb, current, next, invint, N)

    stat_A = trial_stats(
        () -> W.initial_moments_fast(0.5, 0.0, 0.5, J, N);
        samples=samples,
        seconds=seconds,
    )
    stat_B = trial_stats(
        () -> W._precombine_kernel_moments(C, Minit, N, J);
        samples=samples,
        seconds=seconds,
    )
    stat_C = trial_stats(
        () -> recurrence_only!(Mcomb, current, next, N);
        samples=samples,
        seconds=seconds,
    )
    stat_Crecip = trial_stats(
        () -> recurrence_only_reciprocal!(Mcomb, current, next, invint, N);
        samples=samples,
        seconds=seconds,
    )

    S = W.packed_lower_hessenberg_zeros(Float64, N + 1, I)
    recurrence_inner_scatter!(S, Mcomb, current, next, row_scale, I, N)
    stat_D = trial_stats(
        () -> recurrence_inner_scatter!(S, Mcomb, current, next, row_scale, I, N);
        samples=samples,
        seconds=seconds,
    )

    g = [0.5, 0.5]
    W.apply_integer_cheb_multiplier_columns(
        S, g, N, I;
        output_format=:packed_lower_hessenberg,
        lower_bw_V=I + 1,
    )
    stat_E = trial_stats(
        () -> W.apply_integer_cheb_multiplier_columns(
            S, g, N, I;
            output_format=:packed_lower_hessenberg,
            lower_bw_V=I + 1,
        );
        samples=samples,
        seconds=seconds,
    )

    stat_F = trial_stats(
        () -> W.packed_lower_hessenberg_zeros(Float64, N + 1, I + 1);
        samples=samples,
        seconds=seconds,
    )
    stat_Fundef = trial_stats(
        () -> W._packed_lower_hessenberg_uninitialized(Float64, N + 1, I + 1);
        samples=samples,
        seconds=seconds,
    )

    V = W.packed_lower_hessenberg_zeros(Float64, N + 1, I + 1)
    recurrence_fused_eta1_scatter!(V, Mcomb, current, next, row_scale, I, N)
    stat_Dfused = trial_stats(
        () -> recurrence_fused_eta1_scatter!(
            V, Mcomb, current, next, row_scale, I, N,
        );
        samples=samples,
        seconds=seconds,
    )

    full_build(C, min(N, 64))
    full_build(C, N)
    stat_G = trial_stats(
        () -> full_build(C, N);
        samples=samples,
        seconds=seconds,
    )

    for (stage, stat) in (
        ("A_initial_moments", stat_A),
        ("B_precombine", stat_B),
        ("C_recurrence_only", stat_C),
        ("C2_recurrence_reciprocal", stat_Crecip),
        ("D_recurrence_inner_scatter", stat_D),
        ("D2_recurrence_fused_eta1_scatter", stat_Dfused),
        ("E_eta1_multiplier", stat_E),
        ("F_packed_zero_allocation", stat_F),
        ("F2_packed_undef_allocation", stat_Fundef),
        ("G_complete_build", stat_G),
    )
        write_row(io, label, stage, N, I, J, stat)
        @printf("%-38s min=%10.6f median=%10.6f mem=%8.2f MiB allocs=%d\n",
            stage, stat.minimum, stat.median, stat.memory_mib, stat.allocs)
    end

    return nothing
end

function main()
    Ns = parse_int_list("WEAK_INTERNAL_NS", "512,1000,2000,4000,8000,10000")
    Is = parse_int_list("WEAK_INTERNAL_IS", "7,31,127")
    fixed_N = parse(Int, get(ENV, "WEAK_INTERNAL_FIXED_N", "2000"))
    samples = parse(Int, get(ENV, "WEAK_INTERNAL_SAMPLES", "2"))
    seconds = parse(Float64, get(ENV, "WEAK_INTERNAL_SECONDS", "30"))
    output = get(
        ENV,
        "WEAK_INTERNAL_OUTPUT",
        joinpath(@__DIR__, "data", "bench_weak_internal_breakdown_before.csv"),
    )
    mkpath(dirname(output))
    open(output, "w") do io
        println(io, "label,stage,N,I,J,t_min,t_median,memory_MiB,allocs,samples")
        for N in Ns
            run_case!(io, "varyN", N, 31, 31, samples, seconds)
        end
        for I in Is
            I == 31 && continue
            run_case!(io, "varyI", fixed_N, I, I, samples, seconds)
        end
    end
    println("wrote ", output)
end

main()
