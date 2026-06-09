############################################################
# Diagnostic-only script for the ex4 Gutleb--Olver SPVIDE path.
#
# Purpose:
#   Check the full sparse/semi-banded SPVIDE system at coefficient level:
#   RHS projection, exact-solution projection, tau/body row selection,
#   exact-coefficient residuals, discrete-RHS recovery, and eps scans.
#
# Current conclusion:
#   - RHS projection to P^(1,2) is correct.
#   - Solution projection is accurate for moderate eps.
#   - Scheme A, [tau row; body rows 1:N-1], is the most consistent square
#     truncation.
#   - For the original eps=1e-7, the boundary layer is unresolved at the
#     currently tested N; resolving it may require O(10^4)-O(3e4) modes.
#
# This script is independent diagnostics only.  It does not write benchmark
# CSV files and does not affect the formal benchmark output.  Large-N
# projection experiments are intentionally not run here by default.
############################################################

using Pkg

function find_project_root(start_dir::AbstractString = @__DIR__)
    dir = abspath(start_dir)
    while true
        if isfile(joinpath(dir, "Project.toml")) && isdir(joinpath(dir, "src"))
            return dir
        end
        parent = dirname(dir)
        parent == dir && error("Could not find project root from $(start_dir).")
        dir = parent
    end
end

const PROJECT_ROOT = find_project_root()
const GO_PROJECT = joinpath(PROJECT_ROOT, "others", "GutlebOlver")

Pkg.activate(GO_PROJECT; io=devnull)

using ApproxFun
using BandedMatrices
using LinearAlgebra
using Printf
using SparseArrays
using SparseVolterraNew

const MU = -1.0
const P01 = Jacobi(0, 1, 0..1)
const P00 = Jacobi(0, 0, 0..1)
const P12 = Jacobi(1, 2, 0..1)

phi1(z) = abs(z) < 1.0e-8 ? 1.0 + z / 2.0 + z^2 / 6.0 + z^3 / 24.0 + z^4 / 120.0 : expm1(z) / z
u_exact(s, eps) = sin(s) + exp(-s / eps)
du_exact(s, eps) = cos(s) - exp(-s / eps) / eps
K_go(s, r) = exp(MU * s * r)

function rhs_exact(s, eps)
    layer = exp(-s / eps)
    v = sin(s) + layer
    diff_part = eps * cos(s) - layer
    local_part = (1.0 + s) * v
    a = MU * s
    I_sin = (exp(a * s) * (a * sin(s) - cos(s)) + 1.0) / (a^2 + 1.0)
    I_layer = s * phi1(MU * s^2 - s / eps)
    return diff_part + local_part + I_sin + I_layer
end

function coeffs_in_basis(f, basis, n::Int)
    return pad(Fun(f, basis).coefficients, n)
end

function eval_in_basis(c, basis, grid)
    return Fun(basis, c).(grid)
end

function ex4_kerneldepth(N::Int)
    return min(8, max(2, ceil(Int, sqrt(N))))
end

function structural_nnz(n::Int, l::Int, u::Int)
    total = 0
    for j in 1:n
        total += max(0, min(n, j + l) - max(1, j - u) + 1)
    end
    return total
end

function storage_stats(A)
    if A isa SparseMatrixCSC
        rows, cols, vals = findnz(A)
        lb = 0
        ub = 0
        actual = 0
        for (i, j, v) in zip(rows, cols, vals)
            iszero(v) && continue
            actual += 1
            lb = max(lb, i - j)
            ub = max(ub, j - i)
        end
        return (storage=string(typeof(A)), nnz=actual, lower=lb, upper=ub,
                density=actual / prod(size(A)))
    elseif A isa BandedMatrices.AbstractBandedMatrix
        l, u = bandwidths(A)
        actual = structural_nnz(size(A, 1), Int(l), Int(u))
        return (storage=string(typeof(A)), nnz=actual, lower=Int(l), upper=Int(u),
                density=actual / prod(size(A)))
    else
        actual = count(!iszero, A)
        return (storage=string(typeof(A)), nnz=actual, lower="dense_or_unknown",
                upper="dense_or_unknown", density=actual / prod(size(A)))
    end
end

function tau_row(n::Int)
    vals = zeros(Float64, n)
    for j in 1:n
        ej = zeros(Float64, n)
        ej[j] = 1.0
        vals[j] = Fun(P01, ej)(0.0)
    end
    return sparse(ones(Int, n), collect(1:n), vals, 1, n)
end

function replace_first_row_with_tau!(A, tau)
    n = size(A, 2)
    for j in 1:n
        A[1, j] = tau[1, j]
    end
    return A
end

function build_parts(N::Int, eps::Float64)
    n = N + 1
    kerneldepth = ex4_kerneldepth(N)
    VK = build_gutleb_operator(K_go, GutlebOlverConfig(N=n, kerneldepth=kerneldepth, flip=true))
    Vrl = apply_reflect_lowering_strictband(VK, n)
    Vconv = apply_conversion_00_to_12_strictband(Vrl, n)

    D = Derivative(P01, 1)[1:n, 1:n]
    C01to12 = Conversion(P01, P12)[1:n, 1:n]
    Mlocal = Multiplication(Fun(s -> 1.0 + s, P01), P01)[1:n, 1:n]

    derivative = eps * D
    multiplication = C01to12 * Mlocal
    volterra = Vconv
    body = derivative + multiplication + volterra

    uc = coeffs_in_basis(s -> u_exact(s, eps), P01, n)
    rhs_coeff = coeffs_in_basis(s -> rhs_exact(s, eps), P12, n)
    return (; n, kerneldepth, VK, derivative, multiplication, volterra, body, uc, rhs_coeff)
end

function assemble_scheme(parts, scheme::Symbol)
    n = parts.n
    tau = tau_row(n)
    if scheme == :A
        rows = 1:(n - 1)
        A = vcat(tau, sparse(parts.body[rows, :]))
        b = vcat([1.0], parts.rhs_coeff[rows])
    elseif scheme == :B
        rows = 2:n
        A = vcat(tau, sparse(parts.body[rows, :]))
        b = vcat([1.0], parts.rhs_coeff[rows])
    elseif scheme == :D
        A = sparse(parts.body[1:n, :])
        replace_first_row_with_tau!(A, tau)
        b = copy(parts.rhs_coeff[1:n])
        b[1] = 1.0
    elseif scheme == :C
        A = vcat(tau, sparse(parts.body[1:n, :]))
        b = vcat([1.0], parts.rhs_coeff[1:n])
    else
        error("unknown scheme $scheme")
    end
    return A, b
end

function body_residual(parts)
    lhs_d = parts.derivative * parts.uc
    lhs_m = parts.multiplication * parts.uc
    lhs_v = parts.volterra * parts.uc
    lhs = lhs_d + lhs_m + lhs_v
    r = lhs - parts.rhs_coeff
    rel = norm(r, Inf) / max(norm(parts.rhs_coeff, Inf), Base.eps(Float64))
    return (; lhs_d, lhs_m, lhs_v, lhs, r, rel)
end

function solve_error(A, b, eps::Float64)
    c = A \ b
    grid = collect(range(0.0, 1.0; length=1001))
    vals = eval_in_basis(c, P01, grid)
    exact = u_exact.(grid, eps)
    return maximum(abs.(vals .- exact))
end

function projection_checks()
    grid = collect(range(0.0, 1.0; length=1001))
    println("RHS projection checks to P^(1,2)")
    @printf("%-18s %-6s %-12s\n", "function", "N", "max_abs_err")
    for (name, f) in [
        ("1", s -> 1.0),
        ("s", s -> s),
        ("s^2", s -> s^2),
        ("1+s+s^2", s -> 1.0 + s + s^2),
    ]
        for N in (32, 64, 128)
            n = N + 1
            c = coeffs_in_basis(f, P12, n)
            err = maximum(abs.(eval_in_basis(c, P12, grid) .- f.(grid)))
            @printf("%-18s %-6d %-12.4e\n", name, N, err)
        end
    end
    for eps in (1.0e-1, 1.0e-2, 1.0e-4, 1.0e-7)
        for N in (32, 64, 128, 256)
            n = N + 1
            c = coeffs_in_basis(s -> rhs_exact(s, eps), P12, n)
            err = maximum(abs.(eval_in_basis(c, P12, grid) .- rhs_exact.(grid, eps)))
            @printf("%-18s %-6d %-12.4e\n", "rhs eps=$(eps)", N, err)
        end
    end

    println()
    println("Exact u projection checks to P^(0,1)")
    @printf("%-10s %-6s %-12s\n", "eps", "N", "max_abs_err")
    for eps in (1.0e-1, 1.0e-2, 1.0e-4, 1.0e-7)
        for N in (32, 64, 128, 256)
            n = N + 1
            c = coeffs_in_basis(s -> u_exact(s, eps), P01, n)
            err = maximum(abs.(eval_in_basis(c, P01, grid) .- u_exact.(grid, eps)))
            @printf("%-10.1e %-6d %-12.4e\n", eps, N, err)
        end
    end
end

function residual_diagnostics()
    println()
    println("System residual diagnostics")
    @printf("%-8s %-6s %-6s %-12s %-12s %-12s %-12s %-12s %-10s %-10s %-10s\n",
        "eps", "N", "scheme", "tau_res", "body_max", "body_rel", "solve_err",
        "cond_est", "nnz", "lower", "upper")
    for eps in (1.0e-1, 1.0e-2, 1.0e-4, 1.0e-7)
        for N in (32, 64, 128)
            parts = build_parts(N, eps)
            br = body_residual(parts)
            for scheme in (:A, :B, :D, :C)
                A, b = assemble_scheme(parts, scheme)
                rex = A * parts.uc - b
                tau_res = abs(rex[1])
                body_max = length(rex) > 1 ? norm(rex[2:end], Inf) : 0.0
                body_rel = body_max / max(norm(b[2:end], Inf), Base.eps(Float64))
                serr = solve_error(A, b, eps)
                cond_est = "skipped"
                if N <= 32
                    # Debug-only dense copy for a small condition estimate.
                    cond_est = @sprintf("%.3e", cond(Matrix(A)))
                end
                st = storage_stats(A)
                @printf("%-8.1e %-6d %-6s %-12.4e %-12.4e %-12.4e %-12.4e %-12s %-10s %-10s %-10s\n",
                    eps, N, string(scheme), tau_res, body_max, body_rel, serr,
                    cond_est, string(st.nnz), string(st.lower), string(st.upper))
            end
            if N == 32
                println("  coefficient norms for eps=$(eps), N=$(N):")
                @printf("    derivative %.4e multiplication %.4e volterra %.4e rhs %.4e lhs-rhs %.4e\n",
                    norm(br.lhs_d, Inf), norm(br.lhs_m, Inf), norm(br.lhs_v, Inf),
                    norm(parts.rhs_coeff, Inf), norm(br.r, Inf))
                println("    first body residuals: ", br.r[1:min(10, length(br.r))])
                println("    last body residuals: ", br.r[max(1, end-9):end])
            end
        end
    end
end

function discrete_rhs_test()
    println()
    println("Discrete RHS recovery test: b_body = body * u_exact_coeff")
    @printf("%-8s %-6s %-6s %-12s\n", "eps", "N", "scheme", "solve_err")
    for eps in (1.0e-1, 1.0e-2, 1.0e-4, 1.0e-7)
        for N in (32, 64, 128)
            parts = build_parts(N, eps)
            discrete_rhs = parts.body * parts.uc
            for scheme in (:A, :B, :D)
                local A, b
                n = parts.n
                tau = tau_row(n)
                if scheme == :A
                    rows = 1:(n - 1)
                    A = vcat(tau, sparse(parts.body[rows, :]))
                    b = vcat([1.0], discrete_rhs[rows])
                elseif scheme == :B
                    rows = 2:n
                    A = vcat(tau, sparse(parts.body[rows, :]))
                    b = vcat([1.0], discrete_rhs[rows])
                else
                    A = sparse(parts.body[1:n, :])
                    replace_first_row_with_tau!(A, tau)
                    b = copy(discrete_rhs[1:n])
                    b[1] = 1.0
                end
                serr = solve_error(A, b, eps)
                @printf("%-8.1e %-6d %-6s %-12.4e\n", eps, N, string(scheme), serr)
            end
        end
    end
end

projection_checks()
residual_diagnostics()
discrete_rhs_test()
