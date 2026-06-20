############################################################
# ex4_VoltMat.jl
#
# Solve a singularly perturbed Volterra integro-differential
# equation with a non-convolution oscillatory kernel.
#
# Physical problem on t in [0,1]:
#
#   eps*v'(t) + (1+t)*v(t)
#     + ∫_0^t cos(omega*t*s) v(s) ds
#     = 2 + sin(pi*t),
#
#   v(0) = 1.
#
# The exact solution is not used.  Since the reduced problem
# with eps = 0 requires v_0(0) = f(0) = 2, while the imposed
# condition is v(0)=1, the solution has an initial boundary
# layer for small eps.
#
# Chebyshev mapping:
#
#   t = (x+1)/2,   u(x) = v((x+1)/2),   x in [-1,1].
#
# Mapped problem:
#
#   2eps*u'(x) + (x+3)/2*u(x)
#     + 1/2 ∫_{-1}^x cos(omega*(x+1)*(y+1)/4) u(y) dy
#     = 2 + sin(pi*(x+1)/2),
#
#   u(-1) = 1.
#
# RHS coefficients are precomputed once in the Chebyshev basis and then only
# truncated or zero-padded for each N.
#
# Diagnostics:
#   No true error is computed.  The script reports Cauchy
#   errors between the degree-N solution and a degree-Np
#   solution, where Np = max(N+1, round(Int, 1.01*N)).
############################################################

using LinearAlgebra
using SparseArrays
using Printf
using FFTW

script_dir = @__DIR__
include(joinpath(script_dir, "ex4_common.jl"))

function find_project_root(start_dir::AbstractString=@__DIR__)
    dir = abspath(start_dir)
    while true
        if isfile(joinpath(dir, "Project.toml")) && isdir(joinpath(dir, "src"))
            return dir
        end
        parent = dirname(dir)
        parent == dir && error("Could not find the VoltOpt project root from $(start_dir).")
        dir = parent
    end
end

VOLTERRA_ROOT = find_project_root()
SRC_DIR = joinpath(VOLTERRA_ROOT, "src")
DATA_DIR = joinpath(script_dir, "data")

include(joinpath(SRC_DIR, "VoltMat", "voltmat_cheb.jl"))
using .VoltMat_Cheb


# ============================================================
# 0. Parameters
# ============================================================

# Use EPS = 1e-3 and smaller N_FINAL for debugging.
# Then decrease EPS and increase N_FINAL.
EPS = ex4_env_float("EX4_EPS", 1.0e-7)
# const EPS = 1.0e-3

OMEGA = ex4_env_float("EX4_OMEGA", 1.0)

N_FINAL = ex4_env_int("EX4_N_FINAL", 32_000)
N_COUNT = ex4_env_int("EX4_N_COUNT", 100)
# For quick local checks:
#   $env:EX4_N_FINAL = "32"; $env:EX4_N_COUNT = "4"

N_LIST = ex4_default_n_list(N_FINAL, N_COUNT)

DROP_TOL = eps()
RHS_COEFF_METHOD = :dct
CAUCHY_REF_FACTOR = 1.01
GRID_M = ex4_env_int("EX4_GRID_M", 4096)
SHAPE_POINTS = ex4_env_int("EX4_SHAPE_POINTS", 1001)
WRITE_SHAPE = ex4_env_bool("EX4_WRITE_SHAPE", false)
SUMMARY_CSV = get(ENV, "EX4_SUMMARY_CSV", "")
SHAPE_CSV = get(ENV, "EX4_SHAPE_CSV", joinpath(DATA_DIR, "ex4_solution_shape.csv"))


# ============================================================
# 1. Rectangular ultraspherical matrices
# ============================================================

function diffmat_cheb_to_ultra1_rect(N::Int)
    I = Int[]
    J = Int[]
    V = Float64[]

    sizehint!(I, N)
    sizehint!(J, N)
    sizehint!(V, N)

    for k in 1:N
        push!(I, k)
        push!(J, k + 1)
        push!(V, Float64(k))
    end

    return sparse(I, J, V, N, N + 1)
end


function convertmat_cheb_to_ultra1_rect(nrows::Int, ncols::Int)
    I = Int[]
    J = Int[]
    V = Float64[]

    sizehint!(I, 2ncols)
    sizehint!(J, 2ncols)
    sizehint!(V, 2ncols)

    # T_0 = C_0^(1)
    if nrows >= 1 && ncols >= 1
        push!(I, 1)
        push!(J, 1)
        push!(V, 1.0)
    end

    # T_1 = 1/2 C_1^(1)
    if nrows >= 2 && ncols >= 2
        push!(I, 2)
        push!(J, 2)
        push!(V, 0.5)
    end

    # T_k = 1/2 C_k^(1) - 1/2 C_{k-2}^(1), k >= 2.
    for k in 2:(ncols - 1)
        if k <= nrows - 1
            push!(I, k + 1)
            push!(J, k + 1)
            push!(V, 0.5)
        end

        if k - 2 <= nrows - 1
            push!(I, k - 1)
            push!(J, k + 1)
            push!(V, -0.5)
        end
    end

    return sparse(I, J, V, nrows, ncols)
end


function cheb_mult_affine_a_rect(N::Int)
    # a(x) = (x+3)/2 = 3/2 T_0 + 1/2 T_1.
    # Input modes:  0,...,N.
    # Output modes: 0,...,N+1.

    I = Int[]
    J = Int[]
    V = Float64[]

    sizehint!(I, 3N + 4)
    sizehint!(J, 3N + 4)
    sizehint!(V, 3N + 4)

    # 3/2 T_0 * u
    for j in 0:N
        push!(I, j + 1)
        push!(J, j + 1)
        push!(V, 1.5)
    end

    # 1/2 T_1 * T_0 = 1/2 T_1
    push!(I, 2)
    push!(J, 1)
    push!(V, 0.5)

    # 1/2 T_1 * T_j = 1/4 T_{j-1} + 1/4 T_{j+1}
    for j in 1:N
        push!(I, j)
        push!(J, j + 1)
        push!(V, 0.25)

        push!(I, j + 2)
        push!(J, j + 1)
        push!(V, 0.25)
    end

    return sparse(I, J, V, N + 2, N + 1)
end


# ============================================================
# 2. Chebyshev transforms and evaluation
# ============================================================

function cheb_lobatto_nodes(M::Int)
    M >= 1 || error("M must be at least 1.")
    return cos.(pi .* (0:M) ./ M)
end


function cheb_coeffs_dct_values(vals::AbstractVector{<:Real})
    N = length(vals) - 1
    N >= 1 || error("Need at least two values for REDFT00.")

    y = FFTW.r2r(Float64.(vals), FFTW.REDFT00)

    c = y ./ N
    c[1]   *= 0.5
    c[end] *= 0.5

    return c
end


function chebcoeffs_fixedN(f, N::Int; method=:dct)
    if method == :dct
        x = cheb_lobatto_nodes(N)
        vals = f.(x)
        return cheb_coeffs_dct_values(vals)
    else
        error("Only method=:dct is used in this script.")
    end
end



# ============================================================
# 2b. Fixed RHS coefficient cache
# ============================================================

struct FixedChebRHSCache
    coeffs::Vector{Float64}
    degree::Int
    tail_norm::Float64
    refine_diff::Float64
    tail_len::Int
end

function nextpow2_int(n::Integer)
    n <= 1 && return 1
    p = 1
    while p < n
        p <<= 1
    end
    return p
end

function fixed_coeff_slice(cfixed::AbstractVector{<:Real}, len::Int)
    out = zeros(Float64, len)
    m = min(len, length(cfixed))
    out[1:m] .= cfixed[1:m]
    return out
end

function build_fixed_cheb_rhs_cache(
    f,
    min_degree::Int;
    tol::Float64=100.0 * eps(Float64),
    max_degree::Int=max(nextpow2_int(min_degree), 1 << 22),
    tail_len::Int=64,
)
    min_degree >= 1 || error("min_degree must be at least 1.")
    M = nextpow2_int(max(128, min_degree))
    M <= max_degree || error("min_degree=$(min_degree) exceeds max_degree=$(max_degree).")

    prev_c = nothing
    last_tail = Inf
    last_refine = Inf

    while true
        c = chebcoeffs_fixedN(f, M; method=:dct)
        lt = min(tail_len, length(c))
        last_tail = norm(c[(end - lt + 1):end], Inf)

        if prev_c !== nothing
            L = min(length(prev_c), length(c))
            last_refine = norm(prev_c[1:L] .- c[1:L], Inf)
        end

        @printf("  RHS Chebyshev cache trial degree = %d, tail_inf = %.6e, refine_inf = %.6e\n",
                M, last_tail, last_refine)

        if M >= min_degree && prev_c !== nothing && last_tail <= tol && last_refine <= 10.0 * tol
            return FixedChebRHSCache(Vector{Float64}(c), M, last_tail, last_refine, lt)
        end

        if M >= max_degree
            @warn "RHS Chebyshev cache reached max_degree before the requested tolerance was certified." degree=M tail_norm=last_tail refine_diff=last_refine tol=tol
            return FixedChebRHSCache(Vector{Float64}(c), M, last_tail, last_refine, lt)
        end

        prev_c = c
        M *= 2
    end
end

function rhs_cheb_coeffs_from_cache(cache::FixedChebRHSCache, len::Int)
    return fixed_coeff_slice(cache.coeffs, len)
end

function cheb_values_on_lobatto_grid(c::AbstractVector{<:Real}, M::Int)
    N = length(c) - 1
    @assert M >= N

    cp = zeros(Float64, M + 1)
    cp[1:N + 1] .= c

    tmp = copy(cp)
    tmp[1]   *= 2.0
    tmp[end] *= 2.0

    vals = FFTW.r2r(tmp, FFTW.REDFT00) ./ 2.0
    return vals
end


# ============================================================
# 3. RHS and non-convolution oscillatory kernel
# ============================================================

@inline t_of_x(x) = 0.5 * (x + 1.0)

@inline function f_phys(t)
    return 2.0 + sin(pi * t)
end

@inline f_mapped_x(x) = f_phys(t_of_x(x))

@inline function K_mapped(x, y)
    # This is the mapped kernel for K(t,s)=cos(omega*t*s).
    # The factor 1/2 is the Jacobian ds = dy/2.
    return 0.5 * cos(OMEGA * (x + 1.0) * (y + 1.0) / 4.0)
end


# ============================================================
# 4. Volterra matrix construction
# ============================================================

function build_volterra_matrix_square_default(N::Int; tol=DROP_TOL)
    # Important:
    #   Do not pass tail_half_bandwidth.
    #   voltmat_cheb_square will use its internal default tail bandwidth.
    VK = VoltMat_Cheb.voltmat_cheb_square(
        K_mapped,
        N;
        tol=tol,
        use_fft_top=true,
    )

    VKs = sparse(Float64.(VK))

    @printf("  square VK: size = %d x %d, nnz = %d\n",
            size(VKs, 1), size(VKs, 2), nnz(VKs))

    return VKs
end


function build_volterra_matrix_rect_default(N::Int; tol=DROP_TOL)
    # Unknown u_N has input modes 0,...,N.
    # Non-derivative terms must be kept to output modes 0,...,N+1.
    # voltmat_cheb_square returns square matrices, so build degree N+1
    # and keep columns 0,...,N.

    VK_big = build_volterra_matrix_square_default(N + 1; tol=tol)

    VK = sparse(VK_big[:, 1:(N + 1)])

    @printf("  rectangular VK: size = %d x %d, nnz = %d\n",
            size(VK, 1), size(VK, 2), nnz(VK))

    return VK
end


# ============================================================
# 5. Assemble top-tau ultraspherical system
# ============================================================

function assemble_tau_system_rect_default(
    N::Int;
    rhs_cache::FixedChebRHSCache,
    tol=DROP_TOL,
)
    local S0, VK, A_tau

    build_operator_sec = @elapsed begin
        @printf("  Building rectangular D0, S0, Ma ...\n")

        # Residual rows: C_0^(1),...,C_{N-1}^(1)
        D0 = diffmat_cheb_to_ultra1_rect(N)

        # Convert Chebyshev output modes 0,...,N+1 to C^(1) rows 0,...,N-1.
        S0 = convertmat_cheb_to_ultra1_rect(N, N + 2)

        # Multiplication by a(x), input 0:N -> output 0:N+1.
        Ma = cheb_mult_affine_a_rect(N)

        @printf("  Building Volterra matrix with voltmat_cheb_square default bandwidth ...\n")
        VK = build_volterra_matrix_rect_default(N; tol=tol)

        @printf("  Assembling residual operator ...\n")
        A_res = sparse(2.0 * EPS * D0 + S0 * (Ma + VK))

        # Boundary row u(-1)=1:
        #   sum_k (-1)^k c_k = 1.
        bc_vals = [isodd(k) ? -1.0 : 1.0 for k in 0:N]

        bc_row = sparse(
            ones(Int, N + 1),
            collect(1:N + 1),
            bc_vals,
            1,
            N + 1,
        )

        A_tau = vcat(bc_row, A_res)
    end

    local rhs_tau
    build_rhs_sec = @elapsed begin
        @printf("  Taking RHS coefficients 0:%d from fixed Chebyshev cache of degree %d ...\n", N + 1, rhs_cache.degree)
        f_cheb = rhs_cheb_coeffs_from_cache(rhs_cache, N + 2)

        rhs_res = Vector{Float64}(S0 * f_cheb)

        rhs_tau = Vector{Float64}(undef, N + 1)
        rhs_tau[1] = 1.0
        rhs_tau[2:end] .= rhs_res
    end

    return A_tau, rhs_tau, VK, build_operator_sec, build_rhs_sec
end


# ============================================================
# 6. Solve one N
# ============================================================

function solve_one_N_default(
    N::Int;
    rhs_cache::FixedChebRHSCache,
    tol=DROP_TOL,
)
    @printf("\n============================================================\n")
    @printf("Solving N = %d\n", N)
    @printf("eps = %.3e, omega = %.3e\n", EPS, OMEGA)

    status = "ok"

    try
        A_tau, rhs_tau, VK, build_operator_sec, build_rhs_sec = assemble_tau_system_rect_default(
            N;
            rhs_cache=rhs_cache,
            tol=tol,
        )

        @printf("  size(A_tau) = %d x %d\n", size(A_tau, 1), size(A_tau, 2))
        @printf("  nnz(A_tau)  = %d\n", nnz(A_tau))

        local c
        solve_sec = @elapsed begin
            @printf("  Sparse LU factorization and solve ...\n")
            F = lu(A_tau)
            c = F \ rhs_tau
        end

        res_inf = norm(A_tau * c - rhs_tau, Inf)
        if !all(isfinite, c) || !isfinite(res_inf)
            status = "failed"
        end
        @printf("  residual_inf = %.6e, status = %s\n", res_inf, status)

        total_sec = build_operator_sec + build_rhs_sec + solve_sec
        return (; N, coeffs=c, res_inf, nnzA=nnz(A_tau), nnzVK=nnz(VK),
                  build_operator_sec, build_rhs_sec, solve_sec, total_sec, status)
    catch err
        @warn "VoltMat solve failed" N err
        c = fill(NaN, N + 1)
        return (; N, coeffs=c, res_inf=NaN, nnzA=0, nnzVK=0,
                  build_operator_sec=NaN, build_rhs_sec=NaN,
                  solve_sec=NaN, total_sec=NaN, status=ex4_status_from_error(err))
    end
end


# ============================================================
# 7. Cauchy diagnostics
# ============================================================

function reference_degree(N::Int; factor=CAUCHY_REF_FACTOR)
    return ex4_reference_degree(N; factor=factor)
end


function cauchy_coeff_error_inf(cN::AbstractVector{<:Real},
                                cP::AbstractVector{<:Real})
    N1 = length(cN)
    N2 = length(cP)
    L = max(N1, N2)

    cN_pad = zeros(Float64, L)
    cP_pad = zeros(Float64, L)

    cN_pad[1:N1] .= cN
    cP_pad[1:N2] .= cP

    return norm(cN_pad .- cP_pad, Inf)
end

function cheb_eval_clenshaw(c::AbstractVector{<:Real}, x::Real)
    b1 = 0.0
    b2 = 0.0
    for j in length(c):-1:2
        b0 = 2.0 * x * b1 - b2 + c[j]
        b2 = b1
        b1 = b0
    end
    return x * b1 - b2 + c[1]
end

function cheb_values_on_x_grid(c::AbstractVector{<:Real}, xgrid::AbstractVector{<:Real})
    vals = Vector{Float64}(undef, length(xgrid))
    for i in eachindex(xgrid)
        vals[i] = cheb_eval_clenshaw(c, xgrid[i])
    end
    return vals
end

function cauchy_grid_error_inf(cN::AbstractVector{<:Real},
                               cP::AbstractVector{<:Real};
                               Ndiag::Int)
    tgrid = ex4_diagnostic_physical_grid(GRID_M)
    xgrid = 2.0 .* tgrid .- 1.0
    uN = cheb_values_on_x_grid(cN, xgrid)
    uP = cheb_values_on_x_grid(cP, xgrid)

    return maximum(abs.(uN .- uP))
end


function tail_coeff_norm_inf(c::AbstractVector{<:Real}; fraction=0.10)
    L = length(c)
    tail_len = max(1, ceil(Int, fraction * L))
    first_tail = max(1, L - tail_len + 1)
    return norm(c[first_tail:end], Inf)
end


# ============================================================
# 8. Run sequence
# ============================================================

function append_shape_rows_voltmat(filename::AbstractString, N::Int, N_ref::Int,
                                   cN::AbstractVector{<:Real}, cP::AbstractVector{<:Real})
    tgrid = ex4_shape_x_grid(SHAPE_POINTS)
    xgrid = 2.0 .* tgrid .- 1.0
    uN = cheb_values_on_x_grid(cN, xgrid)
    uP = cheb_values_on_x_grid(cP, xgrid)

    mkpath(dirname(filename))
    open(filename, "a") do io
        for i in eachindex(tgrid)
            row = ("VoltMat", N, tgrid[i], uN[i], uP[i], abs(uN[i] - uP[i]))
            println(io, join(ex4_csv_cell.(row), ","))
        end
        flush(io)
    end

    @printf("  shape rows appended to %s for N = %d, N_ref = %d\n", filename, N, N_ref)
end

function run_error_sequence_default(;
    N_list=N_LIST,
    tol=DROP_TOL,
    rhs_tol::Float64= eps(Float64),
    result_csv::AbstractString=joinpath(DATA_DIR, "ex4_VoltMat_results.csv"),
    summary_csv::AbstractString=SUMMARY_CSV,
    shape_csv::AbstractString=SHAPE_CSV,
    write_shape::Bool=WRITE_SHAPE,
)
    needed_N_sorted = ex4_collect_needed_degrees(N_list; factor=CAUCHY_REF_FACTOR)
    shape_N = ex4_env_int("EX4_SHAPE_N", maximum(N_list))

    @printf("\nDegrees to solve:\n")
    @printf("%s\n", string(needed_N_sorted))
    @printf("Diagnostic physical grid points = %d\n", GRID_M + 1)

    # The residual equation needs Chebyshev RHS modes 0:(N+1).
    # Precompute one sufficiently long fixed RHS coefficient vector once;
    # all solves below only truncate or zero-pad this vector.
    min_rhs_degree = maximum(needed_N_sorted) + 1
    @printf("\nPrecomputing fixed RHS Chebyshev coefficients, min degree = %d ...\n", min_rhs_degree)
    rhs_timed = @timed build_fixed_cheb_rhs_cache(f_mapped_x, min_rhs_degree; tol=rhs_tol)
    rhs_cache = rhs_timed.value
    @printf("\nRHS fixed basis      = Chebyshev\n")
    @printf("RHS fixed length     = %d\n", length(rhs_cache.coeffs))
    @printf("RHS tail inf norm    = %.6e\n", rhs_cache.tail_norm)
    @printf("RHS refine inf norm  = %.6e\n", rhs_cache.refine_diff)
    @printf("RHS build time       = %.6e\n", rhs_timed.time)

    mkpath(DATA_DIR)
    ex4_write_csv_header(result_csv, ex4_result_columns)

    last_coeffs = Float64[]
    last_N = 0
    final_rows = NamedTuple[]

    for N in N_list
        N_ref = reference_degree(N)
        local solN, solP, e_coeff, e_grid, e_tail, pair_status

        pair_elapsed = @elapsed begin
            solN = solve_one_N_default(N; rhs_cache=rhs_cache, tol=tol)
            solP = solve_one_N_default(N_ref; rhs_cache=rhs_cache, tol=tol)

            pair_status = ex4_pair_status(solN.status, solP.status)
            e_coeff = cauchy_coeff_error_inf(solN.coeffs, solP.coeffs)
            e_grid = cauchy_grid_error_inf(solN.coeffs, solP.coeffs; Ndiag=N_ref)
            e_tail = tail_coeff_norm_inf(solN.coeffs)
        end

        build_operator_sec = solN.build_operator_sec + solP.build_operator_sec
        build_rhs_sec = solN.build_rhs_sec + solP.build_rhs_sec
        solve_sec = solN.solve_sec + solP.solve_sec
        nnzA = solN.nnzA
        nnzVK = solN.nnzVK

        row = (
            "VoltMat", N, N_ref, OMEGA, e_coeff, e_grid, e_tail,
            build_operator_sec, build_rhs_sec, solve_sec, pair_elapsed,
            nnzA, nnzVK, solN.res_inf, pair_status,
        )
        ex4_append_csv_row(result_csv, row)

        if !isempty(summary_csv)
            ex4_append_csv_row(summary_csv, (
                "VoltMat", N, N_ref, e_coeff, e_grid, pair_elapsed, pair_status,
            ))
        end

        if write_shape && N == shape_N
            append_shape_rows_voltmat(shape_csv, N, N_ref, solN.coeffs, solP.coeffs)
        end

        @printf(
            "\n[VoltMat] N = %d, N_ref = %d, cauchy_coeff = %.6e, cauchy_grid = %.6e, total_time_sec = %.6e, status = %s\n",
            N, N_ref, e_coeff, e_grid, pair_elapsed, pair_status,
        )
        @printf("[VoltMat] appended row to %s\n", result_csv)
        flush(stdout)

        last_coeffs = Vector{Float64}(solN.coeffs)
        last_N = N
        push!(final_rows, (; method="VoltMat", N, N_ref, cauchy_coeff_error=e_coeff,
            cauchy_grid_error=e_grid, total_time_sec=pair_elapsed, status=pair_status))
        GC.gc()
    end

    return (; result_csv, summary_csv, shape_csv, last_coeffs, last_N, final_rows,
              rhs_build_time_sec=rhs_timed.time)
end


function write_coefficients_csv(filename, c)
    mkpath(dirname(filename))
    open(filename, "w") do io
        println(io, "n,c_n,abs_c_n")
        for (j, cj) in enumerate(c)
            @printf(io, "%d,%.17e,%.17e\n", j - 1, cj, abs(cj))
        end
        flush(io)
    end
    return filename
end


# ============================================================
# 9. Main
# ============================================================

if abspath(PROGRAM_FILE) == abspath(@__FILE__)

out = run_error_sequence_default(
    N_list=N_LIST;
    tol=DROP_TOL,
)

tag = ex4_spvide_output_tag(EPS, OMEGA)
coeff_csv = joinpath(DATA_DIR, "ex4_VoltMat_coefficients_N$(out.last_N)_$(tag).csv")
write_coefficients_csv(coeff_csv, out.last_coeffs)

@printf("\nDone.\n")
@printf("Saved CSV outputs:\n")
@printf("  %s\n", out.result_csv)
@printf("  %s\n", coeff_csv)
if !isempty(out.summary_csv)
    @printf("  %s\n", out.summary_csv)
end
if WRITE_SHAPE
    @printf("  %s\n", out.shape_csv)
end

end
