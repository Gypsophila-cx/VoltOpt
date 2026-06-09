############################################################
# Fixed-N shifted Jacobi P^(0,1) projection diagnostics.
#
# This script checks a true fixed-N quadrature projection path.  It does not
# use an adaptive ApproxFun representation padded to length N.  The projection
# coefficients are obtained from Gauss-Jacobi quadrature for the P^(0,1)
# shifted Jacobi weight on [0,1].
#
# The implementation is intended as a debug/projection diagnostic.  For very
# large N, a fast Jacobi transform would be preferable.
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

Pkg.activate(PROJECT_ROOT; io=devnull)

using FastGaussQuadrature
using LinearAlgebra
using Printf

function jacobi_p01_values(s::AbstractVector, n::Int)
    m = length(s)
    V = Matrix{Float64}(undef, m, n)
    n == 0 && return V
    x = @. 2.0 * s - 1.0
    V[:, 1] .= 1.0
    n == 1 && return V

    # P_1^(0,1)(x) = ((alpha-beta) + (alpha+beta+2)x)/2 = -1/2 + 3x/2.
    V[:, 2] .= -0.5 .+ 1.5 .* x
    for k in 1:(n - 2)
        # Recurrence for Jacobi P_k^(alpha,beta), alpha=0, beta=1.
        alpha = 0.0
        beta = 1.0
        kk = Float64(k)
        a1 = 2.0 * (kk + 1.0) * (kk + alpha + beta + 1.0) *
            (2.0 * kk + alpha + beta)
        a2 = (2.0 * kk + alpha + beta + 1.0) *
            (alpha^2 - beta^2)
        a3 = (2.0 * kk + alpha + beta) *
            (2.0 * kk + alpha + beta + 1.0) *
            (2.0 * kk + alpha + beta + 2.0)
        a4 = 2.0 * (kk + alpha) * (kk + beta) *
            (2.0 * kk + alpha + beta + 2.0)
        @views V[:, k + 2] .= ((a2 .+ a3 .* x) .* V[:, k + 1] .- a4 .* V[:, k]) ./ a1
    end
    return V
end

function p01_norms(n::Int)
    # Standard Jacobi norm on [-1,1] for alpha=0,beta=1 is 2/(n+1).
    # Mapping to [0,1] with shifted weight s gives an extra factor 1/4.
    return [1.0 / (2.0 * (k + 1.0)) for k in 0:(n - 1)]
end

function fixedN_p01_projection(f, n::Int; qfactor::Int=2)
    qn = max(n + 8, qfactor * n)
    x, w = gaussjacobi(qn, 0.0, 1.0)
    s = @. 0.5 * (x + 1.0)
    # Standard weights integrate (1+x)g(x) dx.  Shifted weight s ds is 1/4
    # of that measure; this factor cancels with the shifted norm scale.
    vals = f.(s)
    V = jacobi_p01_values(s, n)
    h = p01_norms(n)
    c = Vector{Float64}(undef, n)
    for j in 1:n
        c[j] = 0.25 * dot(w, vals .* V[:, j]) / h[j]
    end
    return c
end

function eval_p01(c::AbstractVector, grid)
    V = jacobi_p01_values(collect(grid), length(c))
    return V * c
end

function projection_error(f, n::Int, grid)
    t = @elapsed c = fixedN_p01_projection(f, n)
    vals = eval_p01(c, grid)
    err = maximum(abs.(vals .- f.(grid)))
    tail = norm(c[end-min(50, length(c))+1:end], Inf)
    return err, t, norm(c, Inf), tail
end

const GRID = unique(sort(vcat(
    collect(range(0.0, 1.0e-7; length=800)),
    collect(range(1.0e-7, 1.0e-6; length=800)),
    collect(range(1.0e-6, 1.0e-5; length=800)),
    collect(range(1.0e-5, 1.0; length=1200)),
)))

const CASES = [
    ("1", s -> 1.0),
    ("s", s -> s),
    ("s^2", s -> s^2),
    ("sin(s)", s -> sin(s)),
    ("exp eps=1e-2", s -> exp(-s / 1.0e-2)),
    ("exp eps=1e-4", s -> exp(-s / 1.0e-4)),
    ("exp eps=1e-7", s -> exp(-s / 1.0e-7)),
]

println("Fixed-N shifted Jacobi P^(0,1) projection diagnostics")
println("projection_method=fixed_N_jacobi_gauss_projection")
@printf("%-16s %-6s %-12s %-12s %-12s %-12s\n",
    "case", "N", "max_error", "proj_time", "coeff_norm", "tail_norm")
for (name, f) in CASES
    for N in (16, 32, 64, 128, 256, 512)
        err, t, cn, tn = projection_error(f, N + 1, GRID)
        @printf("%-16s %-6d %-12.4e %-12.4e %-12.4e %-12.4e\n",
            name, N, err, t, cn, tn)
    end
end
