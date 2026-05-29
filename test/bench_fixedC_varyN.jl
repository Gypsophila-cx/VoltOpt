using SparseArrays
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

# ============================================================
# Kernel and coefficient matrices
# ============================================================

Kexp(x, y) = exp(-x * y)
Khat_cheb(x, y) = Kexp(x, (1 + x) * (1 + y) / 2 - 1)

function fixed_size_coeffmat(C::AbstractMatrix{<:Real}, nrow::Int, ncol::Int)
    nrow > 0 || throw(ArgumentError("nrow must be positive"))
    ncol > 0 || throw(ArgumentError("ncol must be positive"))

    Cout = zeros(Float64, nrow, ncol)
    nr = min(nrow, size(C, 1))
    nc = min(ncol, size(C, 2))
    Cout[1:nr, 1:nc] .= @view C[1:nr, 1:nc]
    return Cout
end

function build_fixed_coefficients(;
    C_rows::Int,
    C_cols::Int,
    alpha::Float64,
    beta::Float64,
    cheb_coeff_tol::Float64,
    jacobi_interp_tol::Float64,
)
    Ccheb_raw = VoltOpt.adaptive_cheb2_coeffmat(
        Khat_cheb;
        tol = cheb_coeff_tol,
    )
    Ccheb = fixed_size_coeffmat(Ccheb_raw, C_rows, C_cols)

    Cjac_raw, jac_interp_info = VoltOpt.adaptive_jacobi_interp_coeffs_Khat(
        Kexp,
        alpha,
        beta;
        tol = jacobi_interp_tol,
        nmin = min(C_rows, 8),
        mmin = min(C_cols, 8),
        nmax = C_rows,
        mmax = C_cols,
    )
    Cjac = fixed_size_coeffmat(Cjac_raw, C_rows, C_cols)

    return Ccheb, Cjac, Ccheb_raw, Cjac_raw, jac_interp_info
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

function bench_call(f; samples::Int = 5, seconds::Float64 = 2.0)
    GC.gc()

    times_ms = Float64[]
    memory_MiB = Float64[]
    allocs = Int[]

    t_start = time()
    for sample_id in 1:samples
        GC.gc()
        stat = @timed f()
        push!(times_ms, stat.time * 1000.0)
        push!(memory_MiB, stat.bytes / 1024.0^2)
        push!(allocs, allocation_count(stat.gcstats))

        if sample_id >= 1 && (time() - t_start) >= seconds
            break
        end
    end

    return (
        time_min_ms = minimum(times_ms),
        time_median_ms = median(times_ms),
        time_sec = minimum(times_ms) / 1000.0,
        memory_MiB = minimum(memory_MiB),
        allocs = minimum(allocs),
        samples_run = length(times_ms),
    )
end

function sparse_bandwidths(A::SparseMatrixCSC)
    lower = 0
    upper = 0

    @inbounds for col in 1:size(A, 2)
        for p in A.colptr[col]:(A.colptr[col + 1] - 1)
            row = A.rowval[p]
            if row >= col
                lower = max(lower, row - col)
            else
                upper = max(upper, col - row)
            end
        end
    end

    return lower, upper
end

function ntget(nt::NamedTuple, key::Symbol, default)
    return key in keys(nt) ? getfield(nt, key) : default
end

function csv_cell(x)
    x === nothing && return ""
    return string(x)
end

function write_csv(csv_file::AbstractString, rows::Vector{<:NamedTuple})
    headers = [
        "N",
        "C_rows",
        "C_cols",
        "alpha",
        "beta",
        "cheb_tail_half_bandwidth",
        "jacobi_n_top",
        "jacobi_bandwidth",
        "cheb_raw_C_rows",
        "cheb_raw_C_cols",
        "jacobi_raw_C_rows",
        "jacobi_raw_C_cols",
        "cheb_min_ms",
        "cheb_median_ms",
        "cheb_time_sec",
        "cheb_memory_MiB",
        "cheb_allocs",
        "cheb_samples_run",
        "cheb_nnz",
        "cheb_rows",
        "cheb_cols",
        "cheb_bw_lower",
        "cheb_bw_upper",
        "jacobi_min_ms",
        "jacobi_median_ms",
        "jacobi_time_sec",
        "jacobi_memory_MiB",
        "jacobi_allocs",
        "jacobi_samples_run",
        "jacobi_nnz",
        "jacobi_top_nnz",
        "jacobi_tail_nnz",
        "jacobi_final_nnz",
        "jacobi_rows",
        "jacobi_cols",
        "jacobi_bw_lower",
        "jacobi_bw_upper",
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

function bench_cheb_vs_jacobi_fixedC_varyN(;
    Csize::Int = 20,
    N_values::Vector{Int} = [100, 200, 400, 800, 1200, 1600, 2400, 3200, 4800, 6400, 9600, 12800],

    # Chebyshev options
    cheb_coeff_tol::Float64 = 64eps(Float64),
    cheb_tol::Float64 = 64eps(Float64),
    cheb_use_fft_top::Bool = true,
    cheb_tail_half_bandwidth::Union{Nothing,Int} = nothing,

    # Jacobi options
    alpha::Float64 = 0.0,
    beta::Float64 = 0.0,
    jacobi_n_top::Union{Nothing,Int} = nothing,
    jacobi_bandwidth::Union{Nothing,Int} = nothing,
    jacobi_G_backend::Symbol = :recurrence,
    jacobi_dropzeros_tol::Float64 = 64eps(Float64),
    jacobi_lin_droptol::Float64 = 0.0,
    jacobi_init_precision::Int = 256,
    jacobi_interp_tol::Float64 = 64eps(Float64),

    # Benchmark options
    samples::Int = 5,
    seconds::Float64 = 2.0,
    seed::Int = 1234,
    output_csv::String = joinpath(data_dir, "bench_fixedC_varyN.csv"),
)
    C_rows = Csize
    C_cols = Csize

    Ccheb, Cjac, Ccheb_raw, Cjac_raw, jac_interp_info = build_fixed_coefficients(
        C_rows = C_rows,
        C_cols = C_cols,
        alpha = alpha,
        beta = beta,
        cheb_coeff_tol = cheb_coeff_tol,
        jacobi_interp_tol = jacobi_interp_tol,
    )

    results = NamedTuple[]

    println("============================================================")
    println("Benchmark: K(x,y)=exp(-x*y), fixed C and varying N")
    println("Kernel                  = exp(-x*y)")
    println("Fixed C size            = $(C_rows) x $(C_cols)")
    println("Cheb raw C size         = $(size(Ccheb_raw, 1)) x $(size(Ccheb_raw, 2))")
    println("Jacobi raw C size       = $(size(Cjac_raw, 1)) x $(size(Cjac_raw, 2))")
    println("Jacobi interp converged = ", ntget(jac_interp_info, :converged, ""))
    println("N values                = $N_values")
    println("Cheb tol                = $cheb_tol")
    println("Cheb use_fft_top        = $cheb_use_fft_top")
    println("Cheb tail bandwidth     = $cheb_tail_half_bandwidth")
    println("Jacobi alpha, beta      = ($alpha, $beta)")
    println("Jacobi interp tol       = $jacobi_interp_tol")
    println("Jacobi n_top            = $jacobi_n_top")
    println("Jacobi bandwidth        = $jacobi_bandwidth")
    println("Jacobi G_backend        = $jacobi_G_backend")
    println("Jacobi dropzeros_tol    = $jacobi_dropzeros_tol")
    println("samples                 = $samples")
    println("seconds                 = $seconds")
    println("seed                    = $seed")
    println("script_dir              = $script_dir")
    println("project_root            = $project_root")
    println("data_dir                = $data_dir")
    println("output_csv              = $output_csv")
    println("============================================================")

    for N in N_values
        println()
        println("------------------------------------------------------------")
        @printf("N = %d, C = %d x %d\n", N, C_rows, C_cols)

        Vcheb_warm = VoltOpt.voltmat_square_from_coeffmat_fft_convolution(
            Ccheb,
            N;
            tol = cheb_tol,
            use_fft_top = cheb_use_fft_top,
            tail_half_bandwidth = cheb_tail_half_bandwidth,
        )

        cheb_nnz = nnz(Vcheb_warm)
        cheb_size = size(Vcheb_warm)
        cheb_bw_lower, cheb_bw_upper = sparse_bandwidths(Vcheb_warm)

        cheb_stat = bench_call(
            () -> VoltOpt.voltmat_square_from_coeffmat_fft_convolution(
                Ccheb,
                N;
                tol = cheb_tol,
                use_fft_top = cheb_use_fft_top,
                tail_half_bandwidth = cheb_tail_half_bandwidth,
            );
            samples = samples,
            seconds = seconds,
        )

        Vjac_warm, jac_info = VoltOpt.voltmat_jacobi_full(
            Cjac,
            N,
            alpha,
            beta;
            n_top = jacobi_n_top,
            bandwidth = jacobi_bandwidth,
            G_backend = jacobi_G_backend,
            init_precision = jacobi_init_precision,
            dropzeros_tol = jacobi_dropzeros_tol,
            lin_droptol = jacobi_lin_droptol,
            return_info = true,
        )

        jacobi_nnz = nnz(Vjac_warm)
        jacobi_size = size(Vjac_warm)
        jacobi_bw_lower, jacobi_bw_upper = sparse_bandwidths(Vjac_warm)

        jacobi_stat = bench_call(
            () -> VoltOpt.voltmat_jacobi_full(
                Cjac,
                N,
                alpha,
                beta;
                n_top = jacobi_n_top,
                bandwidth = jacobi_bandwidth,
                G_backend = jacobi_G_backend,
                init_precision = jacobi_init_precision,
                dropzeros_tol = jacobi_dropzeros_tol,
                lin_droptol = jacobi_lin_droptol,
                return_info = false,
            );
            samples = samples,
            seconds = seconds,
        )

        r = (
            N = N,
            C_rows = C_rows,
            C_cols = C_cols,
            alpha = alpha,
            beta = beta,
            cheb_tail_half_bandwidth = cheb_tail_half_bandwidth,
            jacobi_n_top = jacobi_n_top,
            jacobi_bandwidth = ntget(jac_info, :bandwidth, jacobi_bandwidth),
            cheb_raw_C_rows = size(Ccheb_raw, 1),
            cheb_raw_C_cols = size(Ccheb_raw, 2),
            jacobi_raw_C_rows = size(Cjac_raw, 1),
            jacobi_raw_C_cols = size(Cjac_raw, 2),

            cheb_min_ms = cheb_stat.time_min_ms,
            cheb_median_ms = cheb_stat.time_median_ms,
            cheb_time_sec = cheb_stat.time_sec,
            cheb_memory_MiB = cheb_stat.memory_MiB,
            cheb_allocs = cheb_stat.allocs,
            cheb_samples_run = cheb_stat.samples_run,
            cheb_nnz = cheb_nnz,
            cheb_rows = cheb_size[1],
            cheb_cols = cheb_size[2],
            cheb_bw_lower = cheb_bw_lower,
            cheb_bw_upper = cheb_bw_upper,

            jacobi_min_ms = jacobi_stat.time_min_ms,
            jacobi_median_ms = jacobi_stat.time_median_ms,
            jacobi_time_sec = jacobi_stat.time_sec,
            jacobi_memory_MiB = jacobi_stat.memory_MiB,
            jacobi_allocs = jacobi_stat.allocs,
            jacobi_samples_run = jacobi_stat.samples_run,
            jacobi_nnz = jacobi_nnz,
            jacobi_top_nnz = ntget(jac_info, :top_nnz, ""),
            jacobi_tail_nnz = ntget(jac_info, :tail_nnz, ""),
            jacobi_final_nnz = ntget(jac_info, :final_nnz, jacobi_nnz),
            jacobi_rows = jacobi_size[1],
            jacobi_cols = jacobi_size[2],
            jacobi_bw_lower = jacobi_bw_lower,
            jacobi_bw_upper = jacobi_bw_upper,
        )

        push!(results, r)

        @printf(
            "Cheb   | min = %10.4f ms | med = %10.4f ms | mem = %9.2f MiB | allocs = %8d | nnz = %10d | bw = (%d,%d)\n",
            r.cheb_min_ms,
            r.cheb_median_ms,
            r.cheb_memory_MiB,
            r.cheb_allocs,
            r.cheb_nnz,
            r.cheb_bw_lower,
            r.cheb_bw_upper,
        )

        @printf(
            "Jacobi | min = %10.4f ms | med = %10.4f ms | mem = %9.2f MiB | allocs = %8d | nnz = %10d | bw = (%d,%d)\n",
            r.jacobi_min_ms,
            r.jacobi_median_ms,
            r.jacobi_memory_MiB,
            r.jacobi_allocs,
            r.jacobi_nnz,
            r.jacobi_bw_lower,
            r.jacobi_bw_upper,
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

results = bench_cheb_vs_jacobi_fixedC_varyN(
    Csize = 20,
    N_values = [100, 200, 400, 800, 1200, 1600, 2400, 3200, 4800, 6400, 9600, 12800],

    cheb_coeff_tol = 64eps(Float64),
    cheb_tol = 64eps(Float64),
    cheb_use_fft_top = true,

    alpha = 0.0,
    beta = 0.0,
    jacobi_G_backend = :recurrence,
    jacobi_dropzeros_tol = 64eps(Float64),
    jacobi_interp_tol = 64eps(Float64),

    samples = 5,
    seconds = 2.0,
    seed = 1234,
    output_csv = joinpath(data_dir, "bench_fixedC_varyN.csv"),
)
