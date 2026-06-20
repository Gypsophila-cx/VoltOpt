# ex4_VoltMat.jl
#
# Manufactured oscillatory Volterra integro-differential equation:
#
#   kappa^{-1} u'(x) + u(x)
#   + ∫_0^x cos(omega*x*y) u(y) dy = f(x),
#   x ∈ [0,1],  u(0)=1.
#
# Exact solution:
#
#   u(x) = cos(kappa*x),   kappa = 1000.
#
# Integral term:
#
#   I(x) = ∫_0^x cos(omega*x*y) cos(kappa*y) dy
#
#        = 1/2 [
#             sin((omega*x-kappa)*x)/(omega*x-kappa)
#           + sin((omega*x+kappa)*x)/(omega*x+kappa)
#          ].
#
# Stable form:
#
#   sin(a*x)/a = x*sinc(a*x),  sinc(z)=sin(z)/z.
#
# Therefore:
#
#   f(x) = -sin(kappa*x) + cos(kappa*x) + I(x).
#
# Mapping:
#
#   x = (t+1)/2,  y = (s+1)/2,  v(t)=u((t+1)/2).
#
# Since u'(x)=2v'(t), the mapped equation is
#
#   (2/kappa) v'(t) + v(t)
#   + 1/2 ∫_{-1}^t cos(omega*((t+1)/2)*((s+1)/2)) v(s) ds
#   = f((t+1)/2),
#
# with v(-1)=1.
#
# Output:
#
#   data/ex4_VoltMat_error.csv
#   data/ex4_VoltMat_time.csv

using LinearAlgebra
using SparseArrays
using Printf
using FFTW

# ------------------------------------------------------------
# Locate and include VoltOpt
# ------------------------------------------------------------

function find_project_root(start_dir::AbstractString)
    dir = abspath(start_dir)
    while true
        if isfile(joinpath(dir, "src", "VoltOpt.jl"))
            return dir
        end

        parent = dirname(dir)
        if parent == dir
            error("Cannot find project root containing src/VoltOpt.jl.")
        end

        dir = parent
    end
end

project_root = find_project_root(@__DIR__)
include(joinpath(project_root, "src", "VoltOpt.jl"))
using .VoltOpt

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------

omega = 50.0
kappa = 1000.0

N_min = 10
N_max = 2000

# 100 equally spaced N values in [10, 2000].
# If you want N = 10,20,...,2000, replace this by:
# Ns = collect(10:10:2000)
num_N = 100
Ns = unique(round.(Int, range(N_min, N_max, length=num_N)))

# Grid for max-norm error evaluation.
grid_M = 20000

# RHS oversampling degree.
# This does not affect VoltMat kernel interpolation.
# It only avoids aliasing in the highly oscillatory manufactured RHS.
N_rhs_fixed = 8192

# Let VoltMat automatically choose the kernel interpolation scale.
assembly = :fft
tol = eps()

data_dir = joinpath(@__DIR__, "data")
mkpath(data_dir)

error_csv = joinpath(data_dir, "ex4_VoltMat_error.csv")
time_csv  = joinpath(data_dir, "ex4_VoltMat_time.csv")

# ------------------------------------------------------------
# Stable manufactured RHS
# ------------------------------------------------------------

function sinc_stable(z::Float64)
    az = abs(z)

    if az < 1.0e-4
        z2 = z*z
        return 1.0 - z2/6.0 + z2*z2/120.0 - z2*z2*z2/5040.0
    else
        return sin(z)/z
    end
end

function exact_integral_original(x::Float64; omega=omega, kappa=kappa)
    z_minus = (omega*x - kappa)*x
    z_plus  = (omega*x + kappa)*x

    return 0.5*x*(sinc_stable(z_minus) + sinc_stable(z_plus))
end

function rhs_original(x::Float64; omega=omega, kappa=kappa)
    return -sin(kappa*x) +
           cos(kappa*x) +
           exact_integral_original(x; omega=omega, kappa=kappa)
end

exact_original(x::Float64; kappa=kappa) = cos(kappa*x)

function rhs_mapped(t::Float64)
    x = 0.5*(t + 1.0)
    return rhs_original(x)
end

function exact_mapped(t::Float64)
    x = 0.5*(t + 1.0)
    return exact_original(x)
end

function kernel_mapped(t::Float64, s::Float64)
    x = 0.5*(t + 1.0)
    y = 0.5*(s + 1.0)

    return cos(omega*x*y)
end

# ------------------------------------------------------------
# Chebyshev utilities
# ------------------------------------------------------------

function cheb_lobatto_nodes(N::Int)
    return [cospi(j/N) for j in 0:N]
end

function cheb_coeffs_lobatto(f, N::Int)
    vals = [f(x) for x in cheb_lobatto_nodes(N)]

    c = FFTW.r2r(vals, FFTW.REDFT00)
    a = c ./ N

    a[1]   *= 0.5
    a[end] *= 0.5

    return a
end

function cheb_coeffs_oversampled_trunc(f, N::Int, Nsample::Int)
    if Nsample < N
        error("Nsample must be >= N.")
    end

    a_big = cheb_coeffs_lobatto(f, Nsample)
    return a_big[1:N+1]
end

function cheb_eval(coeffs::AbstractVector{T}, x::Float64) where {T}
    n = length(coeffs) - 1

    if n == 0
        return coeffs[1]
    end

    b1 = zero(T)
    b2 = zero(T)

    for j in n:-1:1
        b0 = 2.0*x*b1 - b2 + coeffs[j+1]
        b2 = b1
        b1 = b0
    end

    return x*b1 - b2 + coeffs[1]
end

# ------------------------------------------------------------
# Ultraspherical first-derivative and conversion matrices
# ------------------------------------------------------------

function cheb_derivative_to_ultra1(N::Int)
    # Unknown coefficients:
    #
    #   v(t) = sum_{n=0}^N a_n T_n(t).
    #
    # Derivative:
    #
    #   d/dt T_n(t) = n U_{n-1}(t),
    #
    # where U_j = C_j^{(1)}.
    #
    # Output size:
    #
    #   N × (N+1),
    #
    # mapping Chebyshev coefficients to U_0,...,U_{N-1}.
    rows = Int[]
    cols = Int[]
    vals = Float64[]

    for n in 1:N
        push!(rows, n)
        push!(cols, n+1)
        push!(vals, float(n))
    end

    return sparse(rows, cols, vals, N, N+1)
end

function cheb_to_ultra1_trunc(N::Int)
    # Convert Chebyshev T coefficients a_0,...,a_N
    # to ultraspherical C^(1), equivalently U coefficients,
    # truncated to U_0,...,U_{N-1}.
    #
    # Relations:
    #
    #   T_0 = U_0,
    #   T_1 = 1/2 U_1,
    #   T_n = 1/2(U_n - U_{n-2}), n >= 2.
    rows = Int[]
    cols = Int[]
    vals = Float64[]

    # T_0 -> U_0
    push!(rows, 1)
    push!(cols, 1)
    push!(vals, 1.0)

    # T_1 -> 1/2 U_1, retained only if N >= 2.
    if N >= 2
        push!(rows, 2)
        push!(cols, 2)
        push!(vals, 0.5)
    end

    for n in 2:N
        col = n + 1

        # +1/2 U_n, retained only when n <= N-1.
        if n <= N-1
            push!(rows, n+1)
            push!(cols, col)
            push!(vals, 0.5)
        end

        # -1/2 U_{n-2}, always retained.
        push!(rows, n-1)
        push!(cols, col)
        push!(vals, -0.5)
    end

    return sparse(rows, cols, vals, N, N+1)
end

function left_boundary_row(N::Int)
    # v(-1) = sum_{n=0}^N a_n (-1)^n = 1.
    ncoeff = N + 1

    rows = ones(Int, ncoeff)
    cols = collect(1:ncoeff)
    vals = [isodd(j-1) ? -1.0 : 1.0 for j in 1:ncoeff]

    return sparse(rows, cols, vals, 1, ncoeff)
end

# ------------------------------------------------------------
# Robust wrapper for VoltMat_Cheb.voltmat_cheb
# ------------------------------------------------------------

function add_voltmat_candidates!(candidates, mod, prefix::String)
    for fname in (:voltmat_cheb_square, :voltmat_cheb)
        if isdefined(mod, fname)
            f = getfield(mod, fname)
            push!(candidates, (name = prefix * "." * String(fname), f = f))
        end
    end

    for subname in (:VoltMat_Cheb, :VoltMat)
        if isdefined(mod, subname)
            submod = getfield(mod, subname)

            for fname in (:voltmat_cheb_square, :voltmat_cheb)
                if isdefined(submod, fname)
                    f = getfield(submod, fname)
                    push!(candidates, (
                        name = prefix * "." * String(subname) * "." * String(fname),
                        f = f
                    ))
                end
            end
        end
    end

    return candidates
end

function voltmat_candidates()
    candidates = []

    if isdefined(Main, :VoltOpt)
        add_voltmat_candidates!(candidates, getfield(Main, :VoltOpt), "VoltOpt")
    end

    if isdefined(Main, :VoltMat_Cheb)
        add_voltmat_candidates!(candidates, getfield(Main, :VoltMat_Cheb), "VoltMat_Cheb")
    end

    if isdefined(Main, :VoltMat)
        add_voltmat_candidates!(candidates, getfield(Main, :VoltMat), "VoltMat")
    end

    if isempty(candidates)
        error("Cannot find voltmat_cheb or voltmat_cheb_square.")
    end

    return candidates
end

function unwrap_matrix(out)
    if out isa AbstractMatrix
        return out
    elseif out isa Tuple
        for item in out
            if item isa AbstractMatrix
                return item
            end
        end
    end

    error("VoltMat constructor did not return an AbstractMatrix or a tuple containing one.")
end

function sparse_square(A, n::Int)
    if size(A, 1) < n || size(A, 2) < n
        error("VoltMat matrix size $(size(A)) is smaller than requested square size $n.")
    end

    try
        return sparse(A[1:n, 1:n])
    catch
        return sparse(Matrix(A)[1:n, 1:n])
    end
end

function build_voltmat_square_autoK(K, ncoeff::Int;
                                    assembly=:fft,
                                    tol=eps())

    candidates = voltmat_candidates()
    tried_errors = String[]

    # Some implementations use ncoeff as size;
    # others use degree ncoeff-1.
    trial_args = unique([ncoeff, ncoeff-1, ncoeff+1])

    for cand in candidates
        for narg in trial_args
            narg <= 0 && continue

            calls = [
                (
                    desc = "$(cand.name)(K, $narg; assembly, tol)",
                    run = () -> cand.f(K, narg;
                        assembly=assembly,
                        tol=tol)
                ),
                (
                    desc = "$(cand.name)(K, $narg; assembly)",
                    run = () -> cand.f(K, narg;
                        assembly=assembly)
                )
            ]

            for call in calls
                try
                    out = call.run()
                    A = unwrap_matrix(out)

                    if size(A, 1) >= ncoeff && size(A, 2) >= ncoeff
                        return sparse_square(A, ncoeff), call.desc, narg
                    else
                        push!(tried_errors,
                            "$(call.desc) returned size $(size(A)), need at least $ncoeff × $ncoeff")
                    end
                catch err
                    push!(tried_errors,
                        "$(call.desc) failed: " * sprint(showerror, err))
                end
            end
        end
    end

    error("""
    Failed to build VoltMat matrix with automatic kernel interpolation.

    Tried calls:
    $(join(tried_errors, "\n"))
    """)
end

# ------------------------------------------------------------
# Solve one N
# ------------------------------------------------------------

function solve_one_N(N::Int)
    ncoeff = N + 1

    VK = nothing
    used_constructor = ""
    used_arg = 0

    op_sec = @elapsed begin
        VK, used_constructor, used_arg =
    build_voltmat_square_autoK(kernel_mapped, ncoeff;
        assembly=assembly,
        tol=tol)
    end

    A = nothing

    sys_sec = @elapsed begin
        D = cheb_derivative_to_ultra1(N)
        S = cheb_to_ultra1_trunc(N)
        Icheb = spdiagm(0 => ones(ncoeff))

        # Mapped equation:
        #
        #   (2/kappa) v'(t) + v(t) + 1/2 V_K v(t) = f((t+1)/2).
        #
        Aeq = (2.0/kappa)*D + S*(Icheb + 0.5*VK)

        bc = left_boundary_row(N)
        A = vcat(bc, Aeq)
    end

    rhs = nothing

    rhs_sec = @elapsed begin
        # Use oversampled RHS coefficients to avoid interpolation aliasing.
        # This does not manually control VoltMat kernel interpolation.
        fcheb = cheb_coeffs_oversampled_trunc(rhs_mapped, N, N_rhs_fixed)

        S = cheb_to_ultra1_trunc(N)
        rhs_u = S*fcheb

        # v(-1)=u(0)=1.
        rhs = vcat([1.0], rhs_u)
    end

    coeffs = nothing

    solve_sec = @elapsed begin
        coeffs = A \ rhs
    end

    err_inf = 0.0

    eval_sec = @elapsed begin
        for j in 0:grid_M
            t = -1.0 + 2.0*j/grid_M
            err_inf = max(err_inf, abs(cheb_eval(coeffs, t) - exact_mapped(t)))
        end
    end

    total_sec = op_sec + sys_sec + rhs_sec + solve_sec

    return (
        N = N,
        ncoeff = ncoeff,
        err_inf = err_inf,
        op_sec = op_sec,
        sys_sec = sys_sec,
        rhs_sec = rhs_sec,
        solve_sec = solve_sec,
        eval_sec = eval_sec,
        total_sec = total_sec,
        nnz_VK = nnz(VK),
        nnz_A = nnz(A),
        constructor = used_constructor,
        constructor_arg = used_arg
    )
end

# ------------------------------------------------------------
# CSV writing
# ------------------------------------------------------------

function write_error_csv(path, rows)
    open(path, "w") do io
        println(io, "N,ncoeff,err_inf,omega,kappa,N_rhs_fixed")
        for r in rows
            @printf(io, "%d,%d,%.17e,%.17e,%.17e,%d\n",
                r.N, r.ncoeff, r.err_inf, omega, kappa, N_rhs_fixed)
        end
    end
end

function write_time_csv(path, rows)
    open(path, "w") do io
        println(io, "N,ncoeff,op_sec,sys_sec,rhs_sec,solve_sec,eval_sec,total_sec,nnz_VK,nnz_A,omega,kappa,N_rhs_fixed,assembly,tol,tail_half_bandwidth,constructor,constructor_arg")

        for r in rows
            @printf(io,
                "%d,%d,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%d,%d,%.17e,%.17e,%d,%s,%.17e,%d,%s,%d\n",
                r.N, r.ncoeff,
                r.op_sec, r.sys_sec, r.rhs_sec, r.solve_sec, r.eval_sec, r.total_sec,
                r.nnz_VK, r.nnz_A,
                omega, kappa, N_rhs_fixed,
                string(assembly), tol, tail_half_bandwidth,
                r.constructor, r.constructor_arg
            )
        end
    end
end

# ------------------------------------------------------------
# Main run
# ------------------------------------------------------------

println("============================================================")
println("ex4 VoltMat + ultraspherical spectral method")
println("Equation: kappa^{-1} u' + u + V_K u = f")
println("VoltMat kernel interpolation: automatic")
println("project_root        = ", project_root)
println("omega               = ", omega)
println("kappa               = ", kappa)
println("N range             = ", first(Ns), " : ", last(Ns), ", count = ", length(Ns))
println("grid_M              = ", grid_M)
println("N_rhs_fixed         = ", N_rhs_fixed)
println("assembly            = ", assembly)
println("tol                 = ", tol)
println("tail_half_bandwidth = ", tail_half_bandwidth)
println("data_dir            = ", data_dir)
println("============================================================")

rows = []

for (idx, N) in enumerate(Ns)
    println()
    @printf("Solving %3d / %3d: N = %d\n", idx, length(Ns), N)

    r = solve_one_N(N)
    push!(rows, r)

    @printf("  err_inf     = %.6e\n", r.err_inf)
    @printf("  op_sec      = %.6e\n", r.op_sec)
    @printf("  sys_sec     = %.6e\n", r.sys_sec)
    @printf("  rhs_sec     = %.6e\n", r.rhs_sec)
    @printf("  solve_sec   = %.6e\n", r.solve_sec)
    @printf("  total_sec   = %.6e\n", r.total_sec)
    @printf("  nnz(VK)     = %d, nnz(A) = %d\n", r.nnz_VK, r.nnz_A)
    println("  constructor = ", r.constructor)

    write_error_csv(error_csv, rows)
    write_time_csv(time_csv, rows)
end

println()
println("============================================================")
println("Finished.")
println("Wrote:")
println("  ", error_csv)
println("  ", time_csv)
println("============================================================")