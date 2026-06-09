using LinearAlgebra
using SparseArrays
using Printf

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

const BENCH_PROJECT_ROOT = find_project_root()
const BENCH_DATA_DIR = joinpath(BENCH_PROJECT_ROOT, "examples", "data")
mkpath(BENCH_DATA_DIR)

include(joinpath(BENCH_PROJECT_ROOT, "src", "VoltMat", "voltmat_cheb.jl"))
using .VoltMat_Cheb

const CSV_HEADERS = [
    :problem,
    :method,
    :N,
    :status,
    :error_type,
    :error,
    :exact_error,
    :cauchy_error,
    :coefficient_error,
    :rhs_time,
    :kernel_approx_time,
    :matrix_construct_time,
    :system_assembly_time,
    :solve_time,
    :factor_time,
    :backsolve_time,
    :total_time,
    :bandwidth,
    :kernel_size_y,
    :kernel_size_t,
    :rank,
    :condition_number,
    :reference_N,
    :basis_family,
    :solution_representation,
    :evaluation_grid,
    :integral_form,
    :variable_transform,
    :go_storage_type,
    :go_operator_nnz,
    :go_system_nnz,
    :go_lower_bandwidth,
    :go_upper_bandwidth,
    :go_density,
    :rhs_projection_method,
    :rhs_basis_family,
    :rhs_basis_parameters,
    :rhs_coefficient_length,
    :rhs_tail_norm,
    :rhs_tail_window,
    :rhs_validation_error,
    :rhs_status,
    :notes,
]

Base.@kwdef struct BenchProblem
    name::String
    Ns::Vector{Int}
    ncoeff_from_N::Function
    rhs::Function
    exact::Function
    local_coeff::Function = s -> 0.0
    deriv_coeff::Function = s -> 0.0
    delay_coeff::Function = s -> 0.0
    delay_scale::Float64 = 1.0
    volterra_terms::Vector{NamedTuple} = NamedTuple[]
    boundary_left::Union{Nothing,Float64} = nothing
    error_type::String = "max_abs"
    primary_error::Symbol = :exact
    reference_N_from::Function = N -> 0
    original_script::String
    n_source::String
    error_source::String
    integral_form::String
    go_integral_form::String
    variable_transform::String
    go_output_basis::String
    voltmat_basis::String = "Chebyshev"
    evaluation_grid::String = "physical [0,1] grid mapped from original [-1,1] if needed"
    notes::String = ""
    go_solver::Union{Nothing,Function} = nothing
end

evenly_spaced_integer_nodes(lo::Int, hi::Int, count::Int) =
    round.(Int, range(lo, hi; length = count))

function bench_sanity_mode()
    return get(ENV, "VOLT_BENCH_SANITY", "0") == "1"
end

function filter_bench_Ns(Ns::Vector{Int})
    out = collect(Ns)
    max_n = get(ENV, "VOLT_BENCH_MAX_N", "")
    max_n_limit = isempty(max_n) ? nothing : parse(Int, max_n)
    if !isempty(max_n)
        lim = max_n_limit
        out = [N for N in out if N <= lim]
    end
    if bench_sanity_mode()
        sanity_cap = max_n_limit === nothing ? 64 : max_n_limit
        candidates = [N for N in out if N <= sanity_cap]
        if length(candidates) < min(3, length(Ns)) && max_n_limit !== nothing
            lo = first(Ns)
            hi = min(max_n_limit, maximum(Ns))
            if hi >= lo
                synthetic = unique(round.(Int, range(lo, hi; length = min(3, max(1, hi - lo + 1)))))
                candidates = unique(vcat(candidates, synthetic))
                sort!(candidates)
            end
        end
        if length(candidates) < min(3, length(out))
            candidates = out[1:min(3, length(out))]
        end
        idx = unique(round.(Int, range(1, length(candidates); length = min(4, length(candidates)))))
        out = candidates[idx]
    end
    max_points = parse(Int, get(ENV, "VOLT_BENCH_MAX_POINTS", "0"))
    if !bench_sanity_mode() && max_points > 0 && max_n_limit !== nothing && !isempty(Ns)
        lo = first(Ns)
        hi = min(max_n_limit, maximum(Ns))
        if hi >= lo
            synthetic_count = min(max_points, max(1, hi - lo + 1))
            synthetic = unique(round.(Int, range(lo, hi; length = synthetic_count)))
            out = unique(vcat(out, synthetic))
            sort!(out)
        end
    end
    if max_points > 0 && length(out) > max_points
        idx = unique(round.(Int, range(1, length(out); length = max_points)))
        out = out[idx]
    end
    isempty(out) && error("No N values left after filtering.")
    return unique(out)
end

chebX(s) = 2.0 * s - 1.0
fromX(X) = 0.5 * (X + 1.0)

function cheb_basis_matrix(n::Int, s)
    xs = collect(s)
    X = chebX.(xs)
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
    D[:, 3] .= 4.0 .* Ujm1
    for j in 4:n
        Uj = 2.0 .* X .* Ujm1 .- Ujm2
        D[:, j] .= 2.0 * (j - 1) .* Uj
        Ujm2, Ujm1 = Ujm1, Uj
    end
    return D
end

function collocation_nodes(n::Int)
    n == 1 && return [0.0]
    return [0.5 * (1.0 - cospi(j / (n - 1))) for j in 0:(n - 1)]
end

function voltmat_integral_coeff_matrix(term, n::Int)
    KX(X, Y) = 0.5 * term.kernel(fromX(X), fromX(Y))
    V = sparse(Float64.(VoltMat_Cheb.voltmat_cheb_square(KX, n - 1; tol = eps(Float64))))
    return sparse(V[1:n, 1:n])
end

function matrix_storage_type(A)
    T = typeof(A)
    mod = parentmodule(T)
    return string(mod, ".", nameof(T))
end

is_dense_matrix(A) = A isa Matrix

function is_banded_matrix(A)
    s = string(typeof(A))
    return occursin("BandedMatrices.", s) ||
        occursin("BlockBandedMatrices.", s) ||
        occursin("BandedMatrix", s) ||
        occursin("Banded", s)
end

function is_lazy_operator(A)
    s = string(typeof(A))
    return occursin("LazyArrays.", s) ||
        occursin("LinearMaps.", s) ||
        occursin("Operator", s) ||
        occursin("Lazy", s)
end

function safe_banded_bandwidths(A)
    mod = parentmodule(typeof(A))
    if isdefined(mod, :bandwidths)
        l, u = getfield(mod, :bandwidths)(A)
        return Int(l), Int(u)
    elseif isdefined(Main, :BandedMatrices) && isdefined(getfield(Main, :BandedMatrices), :bandwidths)
        l, u = getfield(getfield(Main, :BandedMatrices), :bandwidths)(A)
        return Int(l), Int(u)
    end
    return missing, missing
end

function banded_structural_nnz(n::Integer, l::Integer, u::Integer)
    total = 0
    for j in 1:n
        total += max(0, min(n, j + l) - max(1, j - u) + 1)
    end
    return total
end

function safe_nnz(A)
    if A isa SparseMatrixCSC
        return nnz(A)
    elseif is_banded_matrix(A)
        l, u = safe_banded_bandwidths(A)
        if l !== missing && u !== missing
            return banded_structural_nnz(size(A, 1), l, u)
        end
        return missing
    elseif is_lazy_operator(A)
        return missing
    else
        return count(!iszero, A)
    end
end

function safe_density(A)
    n = safe_nnz(A)
    n === missing && return missing
    return n / (size(A, 1) * size(A, 2))
end

function sparse_bandwidths(A)
    return safe_lower_bandwidth(A), safe_upper_bandwidth(A)
end

function safe_lower_bandwidth(A)
    if is_banded_matrix(A)
        l, _ = safe_banded_bandwidths(A)
        return l
    elseif A isa SparseMatrixCSC
        rows, cols, vals = findnz(A)
        lower = 0
        for (i, j, v) in zip(rows, cols, vals)
            abs(v) > 0 && (lower = max(lower, i - j))
        end
        return lower
    elseif is_lazy_operator(A)
        return missing
    else
        lower = 0
        @inbounds for j in axes(A, 2), i in axes(A, 1)
            !iszero(A[i, j]) && (lower = max(lower, i - j))
        end
        return lower
    end
end

function safe_upper_bandwidth(A)
    if is_banded_matrix(A)
        _, u = safe_banded_bandwidths(A)
        return u
    elseif A isa SparseMatrixCSC
        rows, cols, vals = findnz(A)
        upper = 0
        for (i, j, v) in zip(rows, cols, vals)
            abs(v) > 0 && (upper = max(upper, j - i))
        end
        return upper
    elseif is_lazy_operator(A)
        return missing
    else
        upper = 0
        @inbounds for j in axes(A, 2), i in axes(A, 1)
            !iszero(A[i, j]) && (upper = max(upper, j - i))
        end
        return upper
    end
end

function matrix_density(A)
    return safe_density(A)
end

function reject_dense_go_matrix(A, N; name::AbstractString="Gutleb--Olver matrix")
    if N > 32 && is_dense_matrix(A)
        error("$name rejected for N=$N: dense Matrix storage is not an acceptable Gutleb--Olver sparse/banded path")
    end
    return false
end

function fail_row(prob::BenchProblem, method::String, N::Int, status::String, msg::AbstractString)
    return (
        problem = prob.name,
        method = method,
        N = N,
        status = status,
        error_type = prob.error_type,
        error = NaN,
        exact_error = NaN,
        cauchy_error = NaN,
        coefficient_error = NaN,
        rhs_time = NaN,
        kernel_approx_time = NaN,
        matrix_construct_time = NaN,
        system_assembly_time = NaN,
        solve_time = NaN,
        factor_time = NaN,
        backsolve_time = NaN,
        total_time = NaN,
        bandwidth = "missing",
        kernel_size_y = "missing",
        kernel_size_t = "missing",
        rank = "missing",
        condition_number = NaN,
        reference_N = "missing",
        basis_family = method == "gutleb_olver" ? "shifted Jacobi/Proriol" : prob.voltmat_basis,
        solution_representation = "missing",
        evaluation_grid = prob.evaluation_grid,
        integral_form = method == "gutleb_olver" ? prob.go_integral_form : prob.integral_form,
        variable_transform = prob.variable_transform,
        go_storage_type = method == "gutleb_olver" ? "missing" : "not_applicable",
        go_operator_nnz = method == "gutleb_olver" ? "missing" : "not_applicable",
        go_system_nnz = method == "gutleb_olver" ? "missing" : "not_applicable",
        go_lower_bandwidth = method == "gutleb_olver" ? "missing" : "not_applicable",
        go_upper_bandwidth = method == "gutleb_olver" ? "missing" : "not_applicable",
        go_density = method == "gutleb_olver" ? "missing" : "not_applicable",
        rhs_projection_method = method == "gutleb_olver" ? "missing" : "not_applicable",
        rhs_basis_family = method == "gutleb_olver" ? "missing" : "not_applicable",
        rhs_basis_parameters = method == "gutleb_olver" ? "missing" : "not_applicable",
        rhs_coefficient_length = method == "gutleb_olver" ? "missing" : "not_applicable",
        rhs_tail_norm = method == "gutleb_olver" ? NaN : "not_applicable",
        rhs_tail_window = method == "gutleb_olver" ? "missing" : "not_applicable",
        rhs_validation_error = method == "gutleb_olver" ? NaN : "not_applicable",
        rhs_status = method == "gutleb_olver" ? "missing" : "not_applicable",
        notes = msg,
    )
end

function solve_voltmat_coefficients(prob::BenchProblem, N::Int)
    n = prob.ncoeff_from_N(N)
    rows = collocation_nodes(n)
    has_bc = prob.boundary_left !== nothing
    has_deriv = any(abs(prob.deriv_coeff(s)) > 0 for s in (0.0, 0.37, 1.0))
    has_delay = any(abs(prob.delay_coeff(s)) > 0 for s in (0.0, 0.37, 1.0))
    rhs = zeros(Float64, n)
    A = zeros(Float64, n, n)
    coeffs = zeros(Float64, n)
    bandwidth = 0
    kernel_size_y = 0
    kernel_size_t = 0

    rhs_time = @elapsed begin
        rhs .= prob.rhs.(rows)
        has_bc && (rhs[1] = prob.boundary_left)
    end

    kernel_approx_time = 0.0
    matrix_construct_time = @elapsed begin
        B = cheb_basis_matrix(n, rows)
        A .+= Diagonal(prob.local_coeff.(rows)) * B
        if has_deriv
            A .+= Diagonal(prob.deriv_coeff.(rows)) * cheb_deriv_matrix(n, rows)
        end
        if has_delay
            A .+= Diagonal(prob.delay_coeff.(rows)) * cheb_basis_matrix(n, prob.delay_scale .* rows)
        end
        if has_bc
            A[1, :] .= cheb_basis_matrix(n, [0.0])[1, :]
        end
    end

    basis_nodes = collocation_nodes(n)
    B = cheb_basis_matrix(n, rows)
    Bnodes = cheb_basis_matrix(n, basis_nodes)
    for term in prob.volterra_terms
        local V
        kernel_approx_time += @elapsed begin
            V = voltmat_integral_coeff_matrix(term, n)
        end
        kernel_size_y = max(kernel_size_y, size(V, 1))
        kernel_size_t = max(kernel_size_t, size(V, 2))
        lb, ub = sparse_bandwidths(V)
        bandwidth = max(bandwidth, lb, ub)
        matrix_construct_time += @elapsed begin
            C = if term.input_scale == 1.0
                Matrix{Float64}(I, n, n)
            else
                Bscaled = cheb_basis_matrix(n, term.input_scale .* basis_nodes)
                Bnodes \ Bscaled
            end
            A .+= term.sign .* (B * Matrix(V) * C)
        end
    end

    system_assembly_time = 0.0
    factor_time = NaN
    backsolve_time = NaN
    solve_time = @elapsed begin
        coeffs = A \ rhs
    end

    total_time = rhs_time + kernel_approx_time + matrix_construct_time + system_assembly_time + solve_time

    return (
        coeffs = coeffs,
        A = A,
        n = n,
        rhs_time = rhs_time,
        kernel_approx_time = kernel_approx_time,
        matrix_construct_time = matrix_construct_time,
        system_assembly_time = system_assembly_time,
        solve_time = solve_time,
        factor_time = factor_time,
        backsolve_time = backsolve_time,
        total_time = total_time,
        bandwidth = bandwidth,
        kernel_size_y = kernel_size_y,
        kernel_size_t = kernel_size_t,
    )
end

function cheb_coefficient_cauchy(cN::AbstractVector, cP::AbstractVector)
    L = max(length(cN), length(cP))
    a = zeros(Float64, L)
    b = zeros(Float64, L)
    a[1:length(cN)] .= cN
    b[1:length(cP)] .= cP
    return maximum(abs.(a .- b))
end

function solve_voltmat_row(prob::BenchProblem, N::Int)
    sol = solve_voltmat_coefficients(prob, N)
    n = sol.n
    xerr = collect(range(0.0, 1.0; length = max(1001, 4n)))
    vals = cheb_basis_matrix(n, xerr) * sol.coeffs
    exact_vals = prob.exact.(xerr)
    exact_error = maximum(abs.(vals .- exact_vals))
    coefficient_error = NaN
    cauchy_error = NaN
    reference_N = "missing"

    if prob.primary_error == :cauchy
        refN = Int(prob.reference_N_from(N))
        if refN <= N
            error("Cauchy error requested but reference_N_from(N)=$(refN) is not larger than N=$(N)")
        end
        ref = solve_voltmat_coefficients(prob, refN)
        cauchy_error = cheb_coefficient_cauchy(sol.coeffs, ref.coeffs)
        coefficient_error = cauchy_error
        reference_N = refN
    end

    error_value = prob.primary_error == :cauchy ? cauchy_error : exact_error

    isfinite(exact_error) || error("non-finite exact error")
    isfinite(error_value) || error("non-finite primary error")
    condA = n <= 512 ? cond(sol.A) : NaN

    return (
        problem = prob.name,
        method = "voltmat",
        N = N,
        status = "success",
        error_type = prob.error_type,
        error = error_value,
        exact_error = exact_error,
        cauchy_error = cauchy_error,
        coefficient_error = coefficient_error,
        rhs_time = sol.rhs_time,
        kernel_approx_time = sol.kernel_approx_time,
        matrix_construct_time = sol.matrix_construct_time,
        system_assembly_time = sol.system_assembly_time,
        solve_time = sol.solve_time,
        factor_time = sol.factor_time,
        backsolve_time = sol.backsolve_time,
        total_time = sol.total_time,
        bandwidth = sol.bandwidth,
        kernel_size_y = sol.kernel_size_y,
        kernel_size_t = sol.kernel_size_t,
        rank = "missing",
        condition_number = condA,
        reference_N = reference_N,
        basis_family = prob.voltmat_basis,
        solution_representation = "Chebyshev coefficients from voltmat collocation wrapper",
        evaluation_grid = prob.evaluation_grid,
        integral_form = prob.integral_form,
        variable_transform = prob.variable_transform,
        go_storage_type = "not_applicable",
        go_operator_nnz = "not_applicable",
        go_system_nnz = "not_applicable",
        go_lower_bandwidth = "not_applicable",
        go_upper_bandwidth = "not_applicable",
        go_density = "not_applicable",
        rhs_projection_method = "not_applicable",
        rhs_basis_family = "not_applicable",
        rhs_basis_parameters = "not_applicable",
        rhs_coefficient_length = "not_applicable",
        rhs_tail_norm = "not_applicable",
        rhs_tail_window = "not_applicable",
        rhs_validation_error = "not_applicable",
        rhs_status = "not_applicable",
        notes = prob.notes * "; voltmat kernel approximation and operator construction are not fully separable in this wrapper",
    )
end

function go_skipped_row(prob::BenchProblem, N::Int)
    msg = "skipped: certified GO sparse/banded Volterra operator exists, but this example is not marked success because RHS projection/output-basis/system assembly is not yet validated for $(prob.go_integral_form). See the example diagnostics when present; dense collocation GO fallback is rejected."
    return fail_row(prob, "gutleb_olver", N, "skipped", msg)
end

function rhs_projection_row_fields(info)
    params = info.basis_parameters
    return (
        rhs_projection_method = info.method,
        rhs_basis_family = info.basis_family,
        rhs_basis_parameters = "(alpha=$(params.alpha), beta=$(params.beta), domain=$(params.domain))",
        rhs_coefficient_length = info.coefficient_length,
        rhs_tail_norm = info.tail_norm,
        rhs_tail_window = info.tail_window,
        rhs_validation_error = info.validation_error,
        rhs_status = info.status,
    )
end

function solve_gutleb_olver_row(prob::BenchProblem, N::Int)
    prob.go_solver === nothing && return go_skipped_row(prob, N)
    return prob.go_solver(prob, N)
end

function csv_escape(x)
    if x isa AbstractFloat && isnan(x)
        return "NaN"
    end
    s = string(x)
    if occursin(",", s) || occursin("\"", s) || occursin("\n", s)
        return "\"" * replace(s, "\"" => "\"\"") * "\""
    end
    return s
end

function write_bench_csv(path::AbstractString, rows)
    open(path, "w") do io
        println(io, join(string.(CSV_HEADERS), ","))
        for row in rows
            println(io, join((csv_escape(getproperty(row, h)) for h in CSV_HEADERS), ","))
        end
    end
    return path
end

function output_csv_path(problem::AbstractString)
    suffix = bench_sanity_mode() ? "_gutleb_olver_vs_voltmat_sanity.csv" : "_gutleb_olver_vs_voltmat.csv"
    return joinpath(BENCH_DATA_DIR, problem * suffix)
end

function csv_split_simple(line::AbstractString)
    fields = String[]
    buf = IOBuffer()
    inquote = false
    i = firstindex(line)
    while i <= lastindex(line)
        c = line[i]
        if c == '"'
            if inquote && i < lastindex(line) && line[nextind(line, i)] == '"'
                write(buf, '"')
                i = nextind(line, i)
            else
                inquote = !inquote
            end
        elseif c == ',' && !inquote
            push!(fields, String(take!(buf)))
        else
            write(buf, c)
        end
        i = nextind(line, i)
    end
    push!(fields, String(take!(buf)))
    return fields
end

function existing_benchmark_keys(path::AbstractString)
    isfile(path) || return Set{Tuple{String,String,Int,String}}()
    lines = readlines(path)
    isempty(lines) && return Set{Tuple{String,String,Int,String}}()
    headers = Symbol.(csv_split_simple(lines[1]))
    index = Dict(h => i for (i, h) in pairs(headers))
    needed = (:problem, :method, :N, :status)
    all(haskey(index, h) for h in needed) || return Set{Tuple{String,String,Int,String}}()
    keys = Set{Tuple{String,String,Int,String}}()
    for line in lines[2:end]
        isempty(strip(line)) && continue
        fields = csv_split_simple(line)
        try
            push!(keys, (
                fields[index[:problem]],
                fields[index[:method]],
                parse(Int, fields[index[:N]]),
                fields[index[:status]],
            ))
        catch
        end
    end
    return keys
end

function append_bench_row(path::AbstractString, row)
    newfile = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        newfile && println(io, join(string.(CSV_HEADERS), ","))
        println(io, join((csv_escape(getproperty(row, h)) for h in CSV_HEADERS), ","))
        flush(io)
    end
    return path
end

function run_benchmark(prob::BenchProblem)
    Ns = filter_bench_Ns(prob.Ns)
    rows = NamedTuple[]
    path = output_csv_path(prob.name)
    force = get(ENV, "VOLT_BENCH_FORCE_RERUN", "0") == "1"
    if force && isfile(path)
        rm(path)
    end
    existing = force ? Set{Tuple{String,String,Int,String}}() : existing_benchmark_keys(path)
    println("problem = $(prob.name), Ns = $(Ns)")
    for N in Ns
        try
            if any(k -> k[1] == prob.name && k[2] == "voltmat" && k[3] == N, existing)
                @printf("  voltmat N=%d skipped existing CSV row\n", N)
            else
                row = solve_voltmat_row(prob, N)
                push!(rows, row)
                append_bench_row(path, row)
                @printf("  voltmat N=%d error=%.6e total=%.6e\n", N, row.error, row.total_time)
            end
        catch err
            msg = sprint(showerror, err)
            @warn "voltmat failed" problem=prob.name N=N exception=msg
            row = fail_row(prob, "voltmat", N, "failed", "failed: " * msg)
            push!(rows, row)
            append_bench_row(path, row)
        end
        try
            if any(k -> k[1] == prob.name && k[2] == "gutleb_olver" && k[3] == N, existing)
                @printf("  gutleb_olver N=%d skipped existing CSV row\n", N)
            else
                row = solve_gutleb_olver_row(prob, N)
                push!(rows, row)
                append_bench_row(path, row)
            end
        catch err
            msg = sprint(showerror, err)
            @warn "gutleb_olver failed" problem=prob.name N=N exception=msg
            row = fail_row(prob, "gutleb_olver", N, "failed", "failed: " * msg)
            push!(rows, row)
            append_bench_row(path, row)
        end
        GC.gc()
    end
    println("Saved CSV: ", path)
    return rows, path
end
