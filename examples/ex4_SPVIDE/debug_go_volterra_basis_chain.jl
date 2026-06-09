############################################################
# Diagnostic-only script for the ex4 Gutleb--Olver SPVIDE path.
#
# Purpose:
#   Verify the basis/output/reflect/weight chain from the raw strict-band
#   Gutleb--Olver Volterra operator VK to the physical int_0^s operator.
#
# Current conclusion:
#   For the ex4 convention, use flip=true and
#       Reflect * WLowering * VK
#   for the physical int_0^s operator.  To enter the final P^(1,2) output
#   basis, use
#       Conversion(P00 -> P12) * Reflect * WLowering * VK.
#
# This script is independent diagnostics only.  It does not write benchmark
# CSV files and does not affect the formal benchmark output.
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

const P01 = Jacobi(0, 1, 0..1)
const P10 = Jacobi(1, 0, 0..1)
const P00 = Jacobi(0, 0, 0..1)
const P12 = Jacobi(1, 2, 0..1)
const P21 = Jacobi(2, 1, 0..1)

function coeffs_in_basis(f, basis, n::Int)
    return pad(Fun(f, basis).coefficients, n)
end

function eval_in_basis(c, basis, grid)
    return Fun(basis, c).(grid)
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
        return (storage=string(typeof(A)), nnz=actual, lower=lb, upper=ub)
    elseif A isa BandedMatrices.AbstractBandedMatrix
        l, u = bandwidths(A)
        return (storage=string(typeof(A)), nnz=structural_nnz(size(A, 1), Int(l), Int(u)),
                lower=Int(l), upper=Int(u))
    else
        actual = 0
        lb = 0
        ub = 0
        for j in axes(A, 2), i in axes(A, 1)
            if !iszero(A[i, j])
                actual += 1
                lb = max(lb, i - j)
                ub = max(ub, j - i)
            end
        end
        return (storage=string(typeof(A)), nnz=actual, lower=lb, upper=ub)
    end
end

function chain_matrices(n::Int, VK)
    S = Fun(identity, P01)
    J = Multiplication(S, P01)[1:n, 1:n]
    ImJ = BandedMatrix{Float64}(I, (n, n)) - J
    W = WLoweringP01P00(n)
    R = reflectPabtoPba(n)
    C00to12 = Conversion(P00, P12)[1:n, 1:n]
    C10to12 = Conversion(P10, P12)[1:n, 1:n]
    C00to21 = Conversion(P00, P21)[1:n, 1:n]

    return [
        (name="raw_VK_as_P01", A=VK, outbasis=P01,
            raw="P01?", final="P01", notes="raw strict-band operator"),
        (name="W_VK_as_P00", A=W * VK, outbasis=P00,
            raw="P01", final="P00", notes="weighted lowering after raw VK"),
        (name="R_W_VK_as_P00", A=R * W * VK, outbasis=P00,
            raw="P01", final="P00", notes="reflect after WLowering"),
        (name="W_R_VK_as_P00", A=W * R * VK, outbasis=P00,
            raw="P01", final="P00", notes="WLowering after reflect; basis compatibility suspect"),
        (name="C12_W_VK", A=C00to12 * W * VK, outbasis=P12,
            raw="P00", final="P12", notes="lowering then convert P00->P12"),
        (name="C12_R_W_VK_current", A=C00to12 * R * W * VK, outbasis=P12,
            raw="P00", final="P12", notes="current ex4 benchmark Volterra chain"),
        (name="C21_R_W_VK", A=C00to21 * R * W * VK, outbasis=P21,
            raw="P00", final="P21", notes="P10-style final output convention"),
        (name="one_minus_s_VK_as_P01", A=ImJ * VK, outbasis=P01,
            raw="P01", final="P01", notes="(1-s) times raw VK, from triVolterraFullKernelFunP01"),
        (name="R_one_minus_s_VK_as_P10", A=R * ImJ * VK, outbasis=P10,
            raw="P01", final="P10", notes="reflected (1-s) raw VK, from flip=true function path"),
        (name="C12_R_one_minus_s_VK", A=C10to12 * R * ImJ * VK, outbasis=P12,
            raw="P10", final="P12", notes="reflected physical output converted to ex4 P12 output"),
    ]
end

const CASES = [
    (name="K=1,u=1", K=(s, r) -> 1.0, u=s -> 1.0, exact=s -> s, kerneldepth=1),
    (name="K=1,u=r", K=(s, r) -> 1.0, u=s -> s, exact=s -> 0.5 * s^2, kerneldepth=1),
    (name="K=1,u=r^2", K=(s, r) -> 1.0, u=s -> s^2, exact=s -> s^3 / 3.0, kerneldepth=1),
    (name="K=s,u=1", K=(s, r) -> s, u=s -> 1.0, exact=s -> s^2, kerneldepth=2),
    (name="K=r,u=1", K=(s, r) -> r, u=s -> 1.0, exact=s -> 0.5 * s^2, kerneldepth=2),
    (name="K=1+s+r,u=1+r", K=(s, r) -> 1.0 + s + r, u=s -> 1.0 + s,
        exact=s -> s + 2.0 * s^2 + (5.0 / 6.0) * s^3, kerneldepth=3),
]

function run_volterra_diagnostics(; Ns=(8, 16, 32), flips=(false, true))
    grid = collect(range(0.0, 1.0; length=401))
    println("GO Volterra basis-chain diagnostics")
    println("Target physical operator: int_0^s K(s,r)u(r)dr")
    println("solution/input convention A: coefficients in shifted Jacobi P^(0,1)")
    println()
    @printf("%-28s %-5s %-5s %-28s %-10s %-10s %-12s %-10s %-9s %-9s %s\n",
        "chain_name", "N", "flip", "kernel_case", "input", "out_basis",
        "max_abs_err", "nnz", "lower", "upper", "notes")

    best = Dict{Tuple{String, Int, Bool}, Tuple{String, Float64}}()
    for case in CASES
        for N in Ns
            n = N + 1
            ucoeff = coeffs_in_basis(case.u, P01, n)
            exact_vals = case.exact.(grid)
            for flip in flips
                VK = build_gutleb_operator(case.K, GutlebOlverConfig(N=n, kerneldepth=case.kerneldepth, flip=flip))
                for chain in chain_matrices(n, VK)
                    y = chain.A * ucoeff
                    vals = eval_in_basis(y, chain.outbasis, grid)
                    err = maximum(abs.(vals .- exact_vals))
                    stats = storage_stats(chain.A)
                    key = (case.name, N, flip)
                    if !haskey(best, key) || err < best[key][2]
                        best[key] = (chain.name, err)
                    end
                    @printf("%-28s %-5d %-5s %-28s %-10s %-10s %-12.4e %-10s %-9s %-9s %s\n",
                        chain.name, N, string(flip), case.name, "P01", string(chain.final),
                        err, string(stats.nnz), string(stats.lower), string(stats.upper), chain.notes)
                end
            end
        end
    end

    println()
    println("Best chain per polynomial case/N/flip")
    @printf("%-28s %-5s %-5s %-28s %-12s\n", "kernel_case", "N", "flip", "best_chain", "max_abs_err")
    for key in sort(collect(keys(best)); by=x -> (x[1], x[2], x[3]))
        chain, err = best[key]
        @printf("%-28s %-5d %-5s %-28s %-12.4e\n", key[1], key[2], string(key[3]), chain, err)
    end
end

function run_derivative_multiplication_checks(; n=33)
    grid = collect(range(0.0, 1.0; length=401))
    u(s) = 1.0 + s + s^2
    du(s) = 1.0 + 2.0s
    eps = 1.0
    c = coeffs_in_basis(u, P01, n)

    D = Derivative(P01, 1)[1:n, 1:n]
    C01to12 = Conversion(P01, P12)[1:n, 1:n]
    Mlocal = Multiplication(Fun(s -> 1.0 + s, P01), P01)[1:n, 1:n]

    dvals = eval_in_basis(eps * D * c, P12, grid)
    mvals = eval_in_basis(C01to12 * Mlocal * c, P12, grid)
    dmvals = eval_in_basis((eps * D + C01to12 * Mlocal) * c, P12, grid)

    derr = maximum(abs.(dvals .- eps .* du.(grid)))
    merr = maximum(abs.(mvals .- ((1.0 .+ grid) .* u.(grid))))
    dmerr = maximum(abs.(dmvals .- (eps .* du.(grid) .+ (1.0 .+ grid) .* u.(grid))))

    println()
    println("Derivative/multiplication diagnostics in ex4 current convention")
    println("solution basis: P^(0,1), output basis: P^(1,2)")
    @printf("derivative_error       %.6e\n", derr)
    @printf("multiplication_error   %.6e\n", merr)
    @printf("deriv_plus_mult_error  %.6e\n", dmerr)
end

run_volterra_diagnostics()
run_derivative_multiplication_checks()
