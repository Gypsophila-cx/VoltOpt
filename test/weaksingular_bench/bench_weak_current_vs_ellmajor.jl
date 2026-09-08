using BenchmarkTools
using LinearAlgebra
using Printf
using Random

include(joinpath(@__DIR__, "..", "..", "src", "VoltOpt.jl"))
const W = VoltOpt.Weak_VoltMat_Cheb

parse_int_list(name, default) = parse.(Int, split(get(ENV, name, default), ','))

function trial_stats(f; samples::Int, seconds::Float64)
    GC.gc()
    trial = run(@benchmarkable(($f)(), evals=1); samples=samples, seconds=seconds)
    isempty(trial.times) && error("BenchmarkTools returned no samples")
    ts = sort(Float64.(trial.times))
    mid = length(ts) >> 1
    med = isodd(length(ts)) ? ts[mid + 1] : 0.5 * (ts[mid] + ts[mid + 1])
    return (;
        t_min = first(ts) * 1e-9,
        t_median = med * 1e-9,
        memory_MiB = trial.memory / 2.0^20,
        allocs = trial.allocs,
        samples = length(ts),
    )
end

function deterministic_coefficients(I::Int, J::Int)
    rng = MersenneTwister(0x4142 + 257I + J)
    C = 0.04 .* randn(rng, I + 1, J + 1)
    C[1, 1] += 1.0
    return C
end

function recurrence_kmajor!(source, current, next, N::Int)
    copyto!(current, source)
    I1 = size(source, 1)
    for k in 0:(N - 1)
        W._blocki_next_layer!(next, current, k, N, I1)
        current, next = next, current
    end
    return current
end

function recurrence_ellmajor!(source, prev, curr, invint, N::Int)
    copyto!(prev, source)
    I1 = size(source, 1)
    @inbounds for ell in 1:N
        @simd for ii in 1:I1
            curr[ii, 1] = source[ii, ell + 1]
            prev[ii, ell + 1] = 0.0
        end
        ellf = Float64(ell)
        for k in 0:(ell - 1)
            invden = invint[ell + k]
            kf = Float64(k)
            a = (ellf - kf) * invden
            c = -(ellf - kf - 2.0) * invden
            @simd for ii in 1:I1
                curr[ii, k + 2] =
                    a * curr[ii, k + 1] +
                    prev[ii, k + 1] +
                    c * prev[ii, k + 2]
            end
        end
        prev, curr = curr, prev
    end
    return prev
end

# Exact pre-optimization accurate_sum=false DE loop, retained only here so the
# fused recurrence/reduction can be measured in the same Julia process.
function compute_H00k_DE_split(alpha::Real, mu::Real, beta_map::Real, Kmax::Int;
                               qfactor::Int=12,
                               epsq::Float64=eps(Float64))
    alphaf = Float64(alpha)
    mu_sing = Float64(mu)
    betaf = Float64(beta_map)
    H = zeros(Float64, Kmax + 1)
    H[1] = 2.0^(1.0 - mu_sing) *
           exp(W.loggamma(alphaf + 1.0) + W.loggamma(1.0 - mu_sing) -
               W.loggamma(alphaf + 2.0 - mu_sing))
    Kmax == 0 && return H

    A = (alphaf + 1.0) / betaf - 1.0
    B = 1.0 - mu_sing
    S = W._tanh_sinh_S_for_qeps(epsq)
    Q = max(qfactor * Kmax, 80)
    h = 2.0 * S / (Q - 1)
    pref = -2.0^(1.0 - mu_sing) / betaf
    xnodes = Float64[]
    weights = Float64[]
    sizehint!(xnodes, Q)
    sizehint!(weights, Q)
    @inbounds for nu in 1:Q
        s = -S + (nu - 1) * h
        q, dq = W._tanh_sinh_q_and_dq(s)
        if !(0.0 < q < 1.0) || dq == 0.0 || !isfinite(dq)
            continue
        end
        Sb = W._S_beta_stable(q, betaf, mu_sing)
        w = h * dq * q^A * (1.0 - q)^B * Sb
        if isfinite(w)
            push!(xnodes, 2.0 * q - 1.0)
            push!(weights, w)
        end
    end
    H[2] = pref * sum(weights)
    Kmax == 1 && return H

    nnode = length(weights)
    R_nm1 = zeros(Float64, nnode)
    R_n = ones(Float64, nnode)
    R_np1 = similar(R_n)
    @inbounds for n in 0:(Kmax - 2)
        nf = Float64(n)
        a = (nf + 2.0) / (2.0 * nf + 3.0)
        b = -1.0 / ((2.0 * nf + 1.0) * (2.0 * nf + 3.0))
        c = nf / (2.0 * nf + 1.0)
        @simd for i in 1:nnode
            R_np1[i] = ((xnodes[i] - b) * R_n[i] - c * R_nm1[i]) / a
        end
        H[n + 3] = pref * dot(weights, R_np1)
        R_nm1, R_n, R_np1 = R_n, R_np1, R_nm1
    end
    return H
end

function build_kmajor(C, N::Int)
    J = size(C, 2) - 1
    Minit = W.initial_moments_fast(0.5, 0.0, 0.5, J, N;
                                   de_qfactor=12, accurate_sum=false)
    Mcomb = W._precombine_kernel_moments(C, Minit, N, J; mode=:blas)
    return W._assemble_matrix_precombined_eta1_kmajor_from_Mcomb(
        Mcomb, N, 0.0;
        output_format=:packed_lower_hessenberg,
        lower_bw=size(C, 1),
    )
end

function build_ellmajor(C, N::Int)
    J = size(C, 2) - 1
    Minit = W.initial_moments_fast(0.5, 0.0, 0.5, J, N;
                                   de_qfactor=12, accurate_sum=false)
    Mcomb = W._precombine_kernel_moments(C, Minit, N, J; mode=:blas)
    return W._assemble_eta1_ellmajor_scatter_from_Mcomb(
        Mcomb, N, 0.0;
        output_format=:packed_lower_hessenberg,
        lower_bw=size(C, 1),
    )
end

function public_build(C, N::Int)
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

function write_row(io, group, stage, N, I, J, stat, maxerr)
    println(io, join((group, stage, N, I, J, stat.t_min, stat.t_median,
                      stat.memory_MiB, stat.allocs, stat.samples, maxerr), ','))
    flush(io)
end

function run_case!(io, group::String, N::Int, I::Int, J::Int,
                   samples::Int, seconds::Float64)
    @printf("\n%s N=%d I=%d J=%d\n", group, N, I, J)
    C = deterministic_coefficients(I, J)
    Minit = W.initial_moments_fast(0.5, 0.0, 0.5, J, N)
    Mcomb = W._precombine_kernel_moments(C, Minit, N, J; mode=:blas)
    b1 = similar(Mcomb)
    b2 = similar(Mcomb)
    invint = W._ellmajor_inverse_integers(N)

    recurrence_kmajor!(Mcomb, b1, b2, N)
    recurrence_ellmajor!(Mcomb, b1, b2, invint, N)
    ref = build_kmajor(C, N)
    opt = build_ellmajor(C, N)
    maxerr = maximum(abs, ref.values .- opt.values)
    ref = nothing
    public_result, info = public_build(C, N)
    info.assembly_mode === :precombined_ellmajor_serial ||
        error("public path did not select ell-major assembly")
    info.multiplier_mode === :deferred_eta1_column ||
        error("public path did not select the deferred eta=1 multiplier")
    maximum(abs, opt.values .- public_result.values) == 0.0 ||
        error("public and internal ell-major builds disagree")
    opt = nothing
    public_result = nothing
    GC.gc()

    stages = (
        ("C_recurrence_kmajor", () -> recurrence_kmajor!(Mcomb, b1, b2, N), 0.0),
        ("C_recurrence_ellmajor", () -> recurrence_ellmajor!(Mcomb, b1, b2, invint, N), 0.0),
        ("D_assembly_kmajor", () -> W._assemble_matrix_precombined_eta1_kmajor_from_Mcomb(
            copy(Mcomb), N, 0.0;
            output_format=:packed_lower_hessenberg, lower_bw=I + 1), 0.0),
        ("D_assembly_ellmajor", () -> W._assemble_eta1_ellmajor_scatter_from_Mcomb(
            copy(Mcomb), N, 0.0;
            output_format=:packed_lower_hessenberg, lower_bw=I + 1), maxerr),
        ("G_complete_kmajor", () -> build_kmajor(C, N), 0.0),
        ("G_complete_ellmajor", () -> build_ellmajor(C, N), maxerr),
        ("G_public_ellmajor", () -> public_build(C, N), maxerr),
    )
    for (stage, f, err) in stages
        stat = trial_stats(f; samples=samples, seconds=seconds)
        write_row(io, group, stage, N, I, J, stat, err)
        @printf("%-27s min=%9.6f median=%9.6f mem=%8.2f MiB\n",
                stage, stat.t_min, stat.t_median, stat.memory_MiB)
    end
end

function run_de!(io, N::Int, J::Int, samples::Int, seconds::Float64)
    Kmax = N + J
    split() = compute_H00k_DE_split(0.0, 0.5, 0.5, Kmax; qfactor=12)
    fused() = W.compute_H00k_DE(0.0, 0.5, 0.5, Kmax;
                                qfactor=12, accurate_sum=false)
    old = split()
    new = fused()
    err = maximum(abs, old .- new)
    for (stage, f) in (("A_DE_split", split), ("A_DE_fused", fused))
        stat = trial_stats(f; samples=samples, seconds=seconds)
        write_row(io, "DE", stage, N, 0, J, stat, err)
        @printf("%-27s N=%d min=%9.6f median=%9.6f err=%.3e\n",
                stage, N, stat.t_min, stat.t_median, err)
    end
end

function main()
    Ns = parse_int_list("WEAK_AB_NS", "512,1000,2000,4000,8000,10000")
    Is = parse_int_list("WEAK_AB_IS", "7,31,127")
    fixed_N = parse(Int, get(ENV, "WEAK_AB_FIXED_N", "2000"))
    samples = parse(Int, get(ENV, "WEAK_AB_SAMPLES", "2"))
    seconds = parse(Float64, get(ENV, "WEAK_AB_SECONDS", "60"))
    output = get(ENV, "WEAK_AB_OUTPUT",
                 joinpath(@__DIR__, "data", "bench_weak_current_vs_ellmajor.csv"))
    mkpath(dirname(output))
    open(output, "w") do io
        println(io, "group,stage,N,I,J,t_min,t_median,memory_MiB,allocs,samples,max_abs_error")
        for N in Ns
            run_case!(io, "varyN", N, 31, 31, samples, seconds)
        end
        for I in Is
            I == 31 && continue
            run_case!(io, "varyI", fixed_N, I, I, samples, seconds)
        end
        for N in unique(vcat(Ns, fixed_N))
            run_de!(io, N, 31, samples, seconds)
        end
    end
    println("wrote ", output)
end

main()
