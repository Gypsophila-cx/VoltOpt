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

Pkg.activate(GO_PROJECT)

using BandedMatrices
using LinearAlgebra
using SparseArrays
using SparseVolterraNew

K_constant(x, y) = 1.0

function banded_structural_nnz(n::Integer, l::Integer, u::Integer)
    total = 0
    for j in 1:n
        total += max(0, min(n, j + l) - max(1, j - u) + 1)
    end
    return total
end

function smoke_stats(A)
    if A isa SparseMatrixCSC
        rows, cols, vals = findnz(A)
        lower = 0
        upper = 0
        actual_nnz = 0
        for (i, j, v) in zip(rows, cols, vals)
            iszero(v) && continue
            actual_nnz += 1
            lower = max(lower, i - j)
            upper = max(upper, j - i)
        end
        storage_nnz = nnz(A)
    elseif A isa BandedMatrices.AbstractBandedMatrix
        lower, upper = bandwidths(A)
        lower = Int(lower)
        upper = Int(upper)
        actual_nnz = banded_structural_nnz(size(A, 1), lower, upper)
        storage_nnz = actual_nnz
    else
        lower = 0
        upper = 0
        actual_nnz = 0
        for j in axes(A, 2), i in axes(A, 1)
            if !iszero(A[i, j])
                actual_nnz += 1
                lower = max(lower, i - j)
                upper = max(upper, j - i)
            end
        end
        storage_nnz = actual_nnz
    end

    return (
        storage_type = string(typeof(A)),
        is_dense = A isa Matrix,
        is_sparse = A isa SparseMatrixCSC,
        is_banded = A isa BandedMatrices.AbstractBandedMatrix,
        nnz = actual_nnz,
        storage_nnz = storage_nnz,
        lower_bandwidth = lower,
        upper_bandwidth = upper,
        density = actual_nnz / prod(size(A)),
    )
end

function run_smoke_case(N::Int)
    kerneldepth = 1

    local coeffs
    kernel_approx_time = @elapsed begin
        coeffs = build_gutleb_kernel_coefficients(K_constant, kerneldepth; flip = true)
    end

    local V
    matrix_construct_time = @elapsed begin
        V = build_gutleb_operator_from_coefficients(coeffs, N; flip = true)
    end

    reject_dense = N > 32 && V isa Matrix
    reject_dense && error("GO Volterra operator is dense Matrix for N=$N")

    # First-kind reference from the GO overview:
    # int_0^s u(r) dr = s with u(r)=1.
    # After the GO reflection q(x)=g(1-x)/(1-x), q is the constant 1.
    rhs_time = @elapsed begin
        rhs = zeros(Float64, N)
        rhs[1] = 1.0
    end

    local A
    system_assembly_time = @elapsed begin
        A = V
    end
    N > 32 && A isa Matrix && error("GO final system is dense Matrix for N=$N")

    local sol
    solve_time = @elapsed begin
        sol = A \ rhs
    end

    exact_coeffs = zeros(Float64, N)
    exact_coeffs[1] = 1.0
    err = norm(sol - exact_coeffs, Inf)
    isfinite(err) || error("non-finite GO smoke-test error for N=$N")

    op = smoke_stats(V)
    sys = smoke_stats(A)

    return (
        N = N,
        error = err,
        operator_storage_type = op.storage_type,
        operator_nnz = op.nnz,
        system_storage_type = sys.storage_type,
        system_nnz = sys.nnz,
        lower_bandwidth = sys.lower_bandwidth,
        upper_bandwidth = sys.upper_bandwidth,
        density = sys.density,
        rhs_time = rhs_time,
        kernel_approx_time = kernel_approx_time,
        matrix_construct_time = matrix_construct_time,
        system_assembly_time = system_assembly_time,
        solve_time = solve_time,
        total_time = rhs_time + kernel_approx_time + matrix_construct_time +
            system_assembly_time + solve_time,
    )
end

results = [run_smoke_case(N) for N in (16, 32, 64)]

for r in results
    println(r)
end

for i in 2:length(results)
    prev = results[i - 1]
    cur = results[i]
    cur.system_nnz <= 3 * prev.system_nnz ||
        error("GO system nnz grows too quickly from N=$(prev.N) to N=$(cur.N)")
    cur.lower_bandwidth <= prev.lower_bandwidth + 1 ||
        error("GO lower bandwidth is not stably bounded")
    cur.upper_bandwidth <= prev.upper_bandwidth + 1 ||
        error("GO upper bandwidth is not stably bounded")
    cur.error <= max(10 * prev.error, 1.0e-12) ||
        error("GO smoke-test error does not decrease or plateau reasonably")
end

println("Gutleb--Olver sparse-path smoke test passed.")
