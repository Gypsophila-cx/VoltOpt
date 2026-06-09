using LinearAlgebra
using SparseArrays
using Printf
using FFTW
using SpecialFunctions

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
const SRC_DIR = joinpath(PROJECT_ROOT, "src")
const DATA_DIR = joinpath(PROJECT_ROOT, "examples", "data")
const GUTLEB_ROOT = joinpath(PROJECT_ROOT, "others", "GutlebOlver")

mkpath(DATA_DIR)

include(joinpath(SRC_DIR, "VoltMat", "voltmat_cheb.jl"))
using .VoltMat_Cheb

if !(GUTLEB_ROOT in LOAD_PATH)
    pushfirst!(LOAD_PATH, GUTLEB_ROOT)
end

using SparseVolterraNew
using ApproxFun
using BandedMatrices

const QUICK = get(ENV, "VOLT_BENCH_QUICK", "0") == "1"
const MAX_N = parse(Int, get(ENV, "VOLT_BENCH_MAX_N", QUICK ? "64" : string(typemax(Int))))
const MAX_POINTS = parse(Int, get(ENV, "VOLT_BENCH_MAX_POINTS", QUICK ? "4" : "0"))
const GO_KERNELDEPTH = parse(Int, get(ENV, "VOLT_BENCH_GO_KERNELDEPTH", "24"))
const TOL = eps(Float64)

struct VolterraTerm
    sign::Float64
    kernel::Function
    input_scale::Float64
end

Base.@kwdef struct ProblemConfig
    name::String
    Ns::Vector{Int}
    ncoeff_from_N::Function
    rhs::Function
    exact::Function
    local_coeff::Function = s -> 0.0
    deriv_coeff::Function = s -> 0.0
    delay_coeff::Function = s -> 0.0
    delay_scale::Float64 = 1.0
    volterra_terms::Vector{VolterraTerm} = VolterraTerm[]
    boundary_left::Union{Nothing,Float64} = nothing
    error_type::String = "max_abs"
    notes::String = ""
end

evenly_spaced_integer_nodes(lo::Int, hi::Int, count::Int) =
    round.(Int, range(lo, hi; length = count))

filter_Ns(Ns) = begin
    out = [N for N in Ns if N <= MAX_N]
    if MAX_POINTS > 0 && length(out) > MAX_POINTS
        idx = MAX_POINTS == 1 ? [length(out)] : round.(Int, range(1, length(out); length = MAX_POINTS))
        out = out[idx]
    end
    isempty(out) && (out = [minimum(Ns)])
    unique(out)
end

chebX(s) = 2.0 * s - 1.0
fromX(X) = 0.5 * (X + 1.0)

function cheb_basis_matrix(n::Int, s)
    X = chebX.(collect(s))
    B = Matrix{Float64}(undef, length(X), n)
    B[:, 1] .= 1.0
    n >= 2 && (B[:, 2] .= X)
    for j in 3:n
        @views B[:, j] .= 2.0 .* X .* B[:, j - 1] .- B[:, j - 2]
    end
    return B
end

function cheb_deriv_matrix(n::Int, s)
    X = chebX.(collect(s))
    D = zeros(Float64, length(X), n)
    n == 1 && return D
    Ujm2 = ones(Float64, length(X))
    D[:, 2] .= 2.0 .* Ujm2
    n == 2 && return D
    Ujm1 = 2.0 .* X
    D[:, 3] .= 2.0 * 2.0 .* Ujm1
    for j in 4:n
        Uj = 2.0 .* X .* Ujm1 .- Ujm2
        D[:, j] .= 2.0 * (j - 1) .* Uj
        Ujm2, Ujm1 = Ujm1, Uj
    end
    return D
end

function jacobi_basis_matrix(space, n::Int, s; derivative::Bool = false)
    xs = collect(s)
    B = Matrix{Float64}(undef, length(xs), n)
    for j in 1:n
        c = zeros(Float64, n)
        c[j] = 1.0
        f = Fun(space, c)
        if derivative
            f = Derivative(space, 1) * f
        end
        for i in eachindex(xs)
            B[i, j] = f(xs[i])
        end
    end
    return B
end

function composition_matrix(basis_nodes, basis_scaled)
    return basis_nodes \ basis_scaled
end

function voltmat_integral_coeff_matrix(term::VolterraTerm, n::Int)
    KX(X, Y) = 0.5 * term.kernel(fromX(X), fromX(Y))
    V = sparse(Float64.(VoltMat_Cheb.voltmat_cheb_square(KX, n - 1; tol = TOL)))
    return Matrix(V[1:n, 1:n])
end

function gutleb_integral_coeff_matrix(term::VolterraTerm, n::Int)
    kdepth = max(1, min(GO_KERNELDEPTH, n))
    coeffs = build_gutleb_kernel_coefficients(term.kernel, kdepth; flip = false)
    G = build_gutleb_operator_from_coefficients(coeffs, n; flip = false)
    S = Jacobi(0, 1, 0..1)
    x = Fun(identity, S)
    Jx = Multiplication(x, S)[1:n, 1:n]
    return Matrix(Jx * G), coeffs
end

function collocation_nodes(n::Int, has_bc::Bool)
    n == 1 && return [0.0]
    return [0.5 * (1.0 - cospi(j / (n - 1))) for j in 0:(n - 1)]
end

function solve_problem_method(prob::ProblemConfig, N::Int, method::Symbol)
    n = prob.ncoeff_from_N(N)
    has_bc = prob.boundary_left !== nothing
    has_deriv = any(abs(prob.deriv_coeff(s)) > 0 for s in (0.0, 0.37, 1.0))
    has_delay = any(abs(prob.delay_coeff(s)) > 0 for s in (0.0, 0.37, 1.0))
    srows = collocation_nodes(n, has_bc)

    local B
    rhs_time = @elapsed begin
        rhs = prob.rhs.(srows)
        if has_bc
            rhs[1] = prob.boundary_left
        end
    end

    kernel_approx_time = 0.0
    matrix_construct_time = 0.0
    solve_time = 0.0
    bandwidth = -1
    kernel_size_y = -1
    kernel_size_t = -1
    rank_value = -1
    notes = prob.notes

    A = zeros(Float64, n, n)
    c = zeros(Float64, n)

    if method === :voltmat
        B = cheb_basis_matrix(n, srows)
        basis_nodes = collocation_nodes(n, false)
        Bnodes = cheb_basis_matrix(n, basis_nodes)

        matrix_construct_time += @elapsed begin
            A .+= Diagonal(prob.local_coeff.(srows)) * B
            if has_deriv
                D = cheb_deriv_matrix(n, srows)
                A .+= Diagonal(prob.deriv_coeff.(srows)) * D
            end
            if has_delay
                Bdelay = cheb_basis_matrix(n, prob.delay_scale .* srows)
                A .+= Diagonal(prob.delay_coeff.(srows)) * Bdelay
            end
            if has_bc
                A[1, :] .= cheb_basis_matrix(n, [0.0])[1, :]
            end
        end

        for term in prob.volterra_terms
            local V
            kernel_approx_time += @elapsed begin
                V = voltmat_integral_coeff_matrix(term, n)
            end
            matrix_construct_time += @elapsed begin
                C = if term.input_scale == 1.0
                    Matrix{Float64}(I, n, n)
                else
                    Bscaled = cheb_basis_matrix(n, term.input_scale .* basis_nodes)
                    composition_matrix(Bnodes, Bscaled)
                end
                A .+= term.sign .* (B * V * C)
            end
            nzidx = findall(!iszero, sparse(V))
            bw = isempty(nzidx) ? 0 : maximum(abs(I[1] - I[2]) for I in nzidx)
            bandwidth = max(bandwidth, bw)
        end

        solve_time = @elapsed begin
            c = A \ rhs
        end

        eval_solution = xs -> cheb_basis_matrix(n, xs) * c
    elseif method === :gutleb_olver
        S = Jacobi(0, 1, 0..1)
        B = jacobi_basis_matrix(S, n, srows)
        basis_nodes = collocation_nodes(n, false)
        Bnodes = jacobi_basis_matrix(S, n, basis_nodes)

        matrix_construct_time += @elapsed begin
            A .+= Diagonal(prob.local_coeff.(srows)) * B
            if has_deriv
                D = jacobi_basis_matrix(S, n, srows; derivative = true)
                A .+= Diagonal(prob.deriv_coeff.(srows)) * D
            end
            if has_delay
                Bdelay = jacobi_basis_matrix(S, n, prob.delay_scale .* srows)
                A .+= Diagonal(prob.delay_coeff.(srows)) * Bdelay
            end
            if has_bc
                A[1, :] .= jacobi_basis_matrix(S, n, [0.0])[1, :]
            end
        end

        for term in prob.volterra_terms
            local V, coeffs
            kernel_approx_time += @elapsed begin
                V, coeffs = gutleb_integral_coeff_matrix(term, n)
            end
            matrix_construct_time += @elapsed begin
                C = if term.input_scale == 1.0
                    Matrix{Float64}(I, n, n)
                else
                    Bscaled = jacobi_basis_matrix(S, n, term.input_scale .* basis_nodes)
                    composition_matrix(Bnodes, Bscaled)
                end
                A .+= term.sign .* (B * V * C)
            end
            bw = BandedMatrices.bandwidths(BandedMatrix(V))
            bandwidth = max(bandwidth, max(Int(bw[1]), Int(bw[2])))
            kernel_size_y = max(kernel_size_y, coeffs.kerneldepth)
            kernel_size_t = max(kernel_size_t, coeffs.kerneldepth)
        end
        notes *= "; GO integral = Multiplication(x, Jacobi(0,1)) * strict-banded operator"

        solve_time = @elapsed begin
            c = A \ rhs
        end

        eval_solution = xs -> jacobi_basis_matrix(S, n, xs) * c
    else
        error("unknown method $(method)")
    end

    xerr = collect(range(0.0, 1.0; length = max(1001, 4n)))
    vals = eval_solution(xerr)
    exact_vals = prob.exact.(xerr)
    err = maximum(abs.(vals .- exact_vals))
    condA = n <= 512 ? cond(A) : NaN
    total = rhs_time + kernel_approx_time + matrix_construct_time + solve_time
    if !isfinite(err)
        error("non-finite error for $(prob.name), $(method), N=$(N)")
    end
    if any(t -> t < 0, (rhs_time, kernel_approx_time, matrix_construct_time, solve_time))
        error("negative timing for $(prob.name), $(method), N=$(N)")
    end

    return (
        problem = prob.name,
        method = String(method),
        N = N,
        error_type = prob.error_type,
        error = err,
        rhs_time = rhs_time,
        kernel_approx_time = kernel_approx_time,
        matrix_construct_time = matrix_construct_time,
        solve_time = solve_time,
        total_time = total,
        bandwidth = bandwidth,
        kernel_size_y = kernel_size_y,
        kernel_size_t = kernel_size_t,
        rank = rank_value,
        condition_number = condA,
        reference_N = "",
        notes = notes,
    )
end

function gausslegendre_rule(n::Int)
    beta = [k / sqrt(4.0 * k^2 - 1.0) for k in 1:n-1]
    E = eigen(SymTridiagonal(zeros(n), beta))
    p = sortperm(E.values)
    return E.values[p], (2 .* abs2.(E.vectors[1, :]))[p]
end

const GLX, GLW = gausslegendre_rule(300)

function int_0_s(f, s)
    s <= 0 && return 0.0
    h = 0.5 * s
    c = h
    acc = 0.0
    @inbounds for i in eachindex(GLX)
        acc += GLW[i] * f(c + h * GLX[i])
    end
    return h * acc
end

warm_KX(X, Y) = ((X + 1) * (X + 2) / 4) * exp(((Y + 1) * (X - Y)) / 4)
warm_K(s, r) = 2.0 * warm_KX(chebX(s), chebX(r))
warm_u(s) = exp(s^2)

ex1_x1, ex1_x2 = -0.35, 0.42
ex1_A1, ex1_A2 = 1.0, -0.6
ex1_e1, ex1_e2 = 1.0e-3, 1.5e-3
Sstep(x, x0, e) = 0.5 * (1.0 + tanh((x - x0) / e))
ex1_uX(X) = 1.0 + ex1_A1 * Sstep(X, ex1_x1, ex1_e1) + ex1_A2 * Sstep(X, ex1_x2, ex1_e2)
ex1_tau(X, Y) = 1.5 + X - 0.5 * Y
ex1_KX(X, Y) = exp(-(X - Y) / ex1_tau(X, Y)) / ex1_tau(X, Y)
ex1_K(s, r) = 2.0 * ex1_KX(chebX(s), chebX(r))
ex1_u(s) = ex1_uX(chebX(s))
ex1_rhs(s) = ex1_u(s) + int_0_s(r -> ex1_K(s, r) * ex1_u(r), s)

ex2_u(s) = 1.0 / (25 * chebX(s)^2 + 1)
ex2_K1x_X(X, Y) = begin d = X - Y; -40d * exp(-20d^2) end
ex2_K2x_X(X, Y) = begin d = X - Y; (-40d - (2X - Y)) * exp(-20d^2 - X * d) end
ex2_K1(s, r) = 2.0 * ex2_K1x_X(chebX(s), chebX(r))
ex2_K2(s, r) = 2.0 * ex2_K2x_X(chebX(s), chebX(r))
ex2_rhs_from(K, s) = ex2_u(s) + int_0_s(r -> K(s, r) * ex2_u(r), s)

const ex3_q = 0.5
ex3_u(s) = exp(s) * cos(20s)
ex3_rhs(s) = begin
    lambda = 1.0 + 20.0im
    z = sqrt(pi) / 2 * lambda * exp(lambda * s + lambda^2 / 4) *
        (erf((1 - ex3_q) * s + lambda / 2) - erf(lambda / 2))
    endpoint = (1 - ex3_q) * exp(ex3_q * s - (1 - ex3_q)^2 * s^2) * cos(20 * ex3_q * s)
    real(z) + endpoint
end
ex3_K1(s, r) = -2.0 * (s - r) * exp(-(s - r)^2)
ex3_K2(s, r) = -2.0 * (s - ex3_q * r) * exp(-(s - ex3_q * r)^2)
ex3_delay_coeff(s) = -ex3_q * exp(-((1 - ex3_q) * s)^2)

const ex4_EPS = 1.0e-7
const ex4_MU = -1.0
ex4_u(s) = sin(s) + exp(-s / ex4_EPS)
phi1(z) = abs(z) < 1e-8 ? 1 + z / 2 + z^2 / 6 + z^3 / 24 + z^4 / 120 : expm1(z) / z
ex4_rhs(s) = begin
    layer = exp(-s / ex4_EPS)
    v = sin(s) + layer
    diff_part = ex4_EPS * cos(s) - layer
    local_part = (1 + s) * v
    a = ex4_MU * s
    I_sin = (exp(a * s) * (a * sin(s) - cos(s)) + 1.0) / (a^2 + 1.0)
    I_layer = s * phi1(ex4_MU * s^2 - s / ex4_EPS)
    diff_part + local_part + I_sin + I_layer
end
ex4_K(s, r) = exp(ex4_MU * s * r)

function problems()
    [
        ProblemConfig(
            name = "warmup",
            Ns = collect(1:100),
            ncoeff_from_N = N -> max(N, 2),
            rhs = s -> 2s + 1,
            exact = warm_u,
            local_coeff = s -> 1.0,
            deriv_coeff = s -> 1.0,
            volterra_terms = [VolterraTerm(-1.0, warm_K, 1.0)],
            boundary_left = 1.0,
            notes = "Ns from examples/warmup/solve_vide_low_order_cheb_leg.jl DEFAULT_NS",
        ),
        ProblemConfig(
            name = "ex1",
            Ns = evenly_spaced_integer_nodes(2, 40000, 100),
            ncoeff_from_N = N -> N,
            rhs = ex1_rhs,
            exact = ex1_u,
            local_coeff = s -> 1.0,
            volterra_terms = [VolterraTerm(1.0, ex1_K, 1.0)],
            notes = "Ns from examples/ex1_2ndVolterraEqn/ex1_aging_maxwell.jl evenly_spaced_integer_nodes(2,40000,100)",
        ),
        ProblemConfig(
            name = "ex2",
            Ns = [16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024],
            ncoeff_from_N = N -> N + 1,
            rhs = s -> ex2_rhs_from(ex2_K1, s),
            exact = ex2_u,
            local_coeff = s -> 1.0,
            volterra_terms = [VolterraTerm(1.0, ex2_K1, 1.0)],
            notes = "Ns from examples/ex2_1stVolterraEqn; differentiated second-kind K1 case",
        ),
        ProblemConfig(
            name = "ex3",
            Ns = collect(2:100),
            ncoeff_from_N = N -> N + 1,
            rhs = ex3_rhs,
            exact = ex3_u,
            local_coeff = s -> 1.0,
            delay_coeff = ex3_delay_coeff,
            delay_scale = ex3_q,
            volterra_terms = [VolterraTerm(1.0, ex3_K1, 1.0), VolterraTerm(-ex3_q, ex3_K2, ex3_q)],
            notes = "Ns from examples/ex3_pantograph/ex3_pantograph_firstkind.jl const Ns = collect(2:1:100)",
        ),
        ProblemConfig(
            name = "ex4",
            Ns = evenly_spaced_integer_nodes(2, 32000, 100),
            ncoeff_from_N = N -> N + 1,
            rhs = ex4_rhs,
            exact = ex4_u,
            local_coeff = s -> 1.0 + s,
            deriv_coeff = s -> ex4_EPS,
            volterra_terms = [VolterraTerm(1.0, ex4_K, 1.0)],
            boundary_left = 1.0,
            notes = "Ns from examples/ex4_SPVIDE/ex4_SPVIDE.jl N_LIST; script records true max_abs error matching available exact solution",
        ),
    ]
end

const CSV_HEADERS = [
    :problem, :method, :N, :error_type, :error, :rhs_time,
    :kernel_approx_time, :matrix_construct_time, :solve_time, :total_time,
    :bandwidth, :kernel_size_y, :kernel_size_t, :rank, :condition_number,
    :reference_N, :notes,
]

csv_escape(x) = begin
    s = string(x)
    if occursin(",", s) || occursin("\"", s) || occursin("\n", s)
        "\"" * replace(s, "\"" => "\"\"") * "\""
    else
        s
    end
end

function write_csv(path, rows)
    open(path, "w") do io
        println(io, join(string.(CSV_HEADERS), ","))
        for r in rows
            println(io, join((csv_escape(getproperty(r, h)) for h in CSV_HEADERS), ","))
        end
    end
    return path
end

function main(selected = String[])
    println("Gutleb--Olver vs voltmat benchmark")
    println("MAX_N = $(MAX_N), MAX_POINTS = $(MAX_POINTS), GO_KERNELDEPTH = $(GO_KERNELDEPTH)")
    QUICK && println("VOLT_BENCH_QUICK=1: running a short verification subset.")

    for prob in problems()
        if !isempty(selected) && !(prob.name in selected)
            continue
        end
        Ns = filter_Ns(prob.Ns)
        rows = NamedTuple[]
        println("============================================================")
        println("problem = $(prob.name), Ns = $(Ns)")
        for N in Ns
            for method in (:voltmat, :gutleb_olver)
                try
                    row = solve_problem_method(prob, N, method)
                    push!(rows, row)
                    @printf(
                        "problem = %s, method = %s, N = %d, error = %.6e, total_time = %.6e\n",
                        row.problem, row.method, row.N, row.error, row.total_time,
                    )
                catch err
                    msg = sprint(showerror, err)
                    @warn "run failed" problem=prob.name method=method N=N exception=msg
                    push!(rows, (
                        problem = prob.name, method = String(method), N = N,
                        error_type = prob.error_type, error = NaN,
                        rhs_time = 0.0, kernel_approx_time = 0.0,
                        matrix_construct_time = 0.0, solve_time = 0.0,
                        total_time = 0.0, bandwidth = -1, kernel_size_y = -1,
                        kernel_size_t = -1, rank = -1, condition_number = NaN,
                        reference_N = "", notes = prob.notes * "; failed: " * msg,
                    ))
                end
                GC.gc()
            end
        end
        path = joinpath(DATA_DIR, "$(prob.name)_gutleb_olver_vs_voltmat.csv")
        write_csv(path, rows)
        println("Saved CSV: ", path)
    end
end

main(ARGS)
