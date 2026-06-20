const SIMPLE_SOLUTION_SPACE = Jacobi(0, 1, 0..1)
const SECTION522_SOLUTION_SPACE = Jacobi(0, 1, 0..1)
const SECTION522_OUTPUT_SPACE = Jacobi(1, 2, 0..1)
const SECTION522_DERIVATIVE_ORDER = 1
const SECTION522_FLIP = true
const SECTION522_K = 100.0
const SECTION522_ACCEPTANCE_N = 128
const SECTION522_ACCEPTANCE_KERNELDEPTH = 12
const DEFAULT_GRID = range(0.0, 1.0; length=1000)

_resolve_N(N::Integer, n) = isnothing(n) ? Int(N) : Int(n)

pad_to_length(c, n::Integer) = begin
    n < 0 && throw(ArgumentError("target length must be non-negative"))
    v = zeros(Float64, Int(n))
    m = min(length(c), Int(n))
    if m > 0
        v[1:m] .= c[1:m]
    end
    v
end

storage_label(A) =
    A isa BandedMatrix ? "BandedMatrix" :
    A isa SparseMatrixCSC ? "SparseMatrixCSC" :
    string(typeof(A))

"""
    safe_bandwidth(A) -> (lower, upper)

Return the actual nonzero lower/upper bandwidth of a `BandedMatrix` or
`SparseMatrixCSC` by scanning only the sparse/banded storage pattern. This
diagnostic does not materialize a dense matrix.
"""
function safe_bandwidth(A::BandedMatrix)
    rows, cols = size(A)
    structural_l, structural_u = BandedMatrices.bandwidths(A)
    structural_l = Int(structural_l)
    structural_u = Int(structural_u)

    actual_l = 0
    actual_u = 0
    nz = 0
    @inbounds for j = 1:cols
        ilo = max(1, j - structural_u)
        ihi = min(rows, j + structural_l)
        for i = ilo:ihi
            if !iszero(A[i, j])
                nz += 1
                actual_l = max(actual_l, i - j)
                actual_u = max(actual_u, j - i)
            end
        end
    end

    return (actual_l, actual_u)
end

function safe_bandwidth(A::SparseMatrixCSC)
    rowidx, colidx, vals = findnz(A)
    lower_bw = 0
    upper_bw = 0
    @inbounds for p in eachindex(vals)
        if !iszero(vals[p])
            lower_bw = max(lower_bw, rowidx[p] - colidx[p])
            upper_bw = max(upper_bw, colidx[p] - rowidx[p])
        end
    end
    return (lower_bw, upper_bw)
end

safe_bandwidth(A) =
    throw(ArgumentError("safe_bandwidth is only defined for BandedMatrix and SparseMatrixCSC inputs"))

"""
    safe_nnz(A) -> Integer

Count structurally stored nonzero values in a `BandedMatrix` or
`SparseMatrixCSC` without converting the operator to dense storage.
"""
function safe_nnz(A::BandedMatrix)
    rows, cols = size(A)
    structural_l, structural_u = BandedMatrices.bandwidths(A)
    structural_l = Int(structural_l)
    structural_u = Int(structural_u)

    nz = 0
    @inbounds for j = 1:cols
        ilo = max(1, j - structural_u)
        ihi = min(rows, j + structural_l)
        for i = ilo:ihi
            if !iszero(A[i, j])
                nz += 1
            end
        end
    end

    return nz
end

safe_nnz(A::SparseMatrixCSC) = count(!iszero, nonzeros(A))

safe_nnz(A) =
    throw(ArgumentError("safe_nnz is only defined for BandedMatrix and SparseMatrixCSC inputs"))

"""
    safe_density(A) -> Float64

Return `safe_nnz(A) / prod(size(A))` for sparse or banded operators without any
dense fallback.
"""
function safe_density(A)
    rows, cols = size(A)
    return safe_nnz(A) / (rows * cols)
end

function operator_stats(A)
    rows, cols = size(A)
    lower_bw, upper_bw = safe_bandwidth(A)
    nz = safe_nnz(A)
    return (
        type=storage_label(A),
        rows=rows,
        cols=cols,
        nnz=nz,
        lower_bandwidth=lower_bw,
        upper_bandwidth=upper_bw,
        bandwidth=(lower_bw, upper_bw),
        density=nz / (rows * cols),
    )
end

banded_stats(A::BandedMatrix) = operator_stats(A)
sparse_stats(A::SparseMatrixCSC) = operator_stats(A)

simple_kernel(x, y) = 1.0
simple_exact_solution(x) = 1.0 + x
simple_rhs_function(x) = 1.0 + 2.0 * x + 0.5 * x^2

function simple_solution_fun()
    x = Fun(identity, SIMPLE_SOLUTION_SPACE)
    return 1.0 + x
end

function simple_rhs_fun()
    x = Fun(identity, SIMPLE_SOLUTION_SPACE)
    return 1.0 + 2.0 * x + 0.5 * x^2
end

"""
    build_simple_secondkind_system(; N=32)

Build the manufactured `K=1` second-kind Volterra system in the shifted
`P^(1,0)` solution/output basis. The returned `NamedTuple` contains the raw
strict-band Volterra operator, the physical Volterra operator, the banded
system matrix, RHS coefficients, and exact coefficients.

The construction keeps the sparse/strict-band path and does not form a dense
matrix.
"""
function build_simple_secondkind_system(; N::Integer=32, n=nothing)
    Nval = _resolve_N(N, n)
    plan = precompute_triangular_kernel_plan(1, true)
    coeffs = triangular_kernel_coeffs_from_plan(plan, simple_kernel)
    Vraw = triVolterraFullKernelOpP01_strictband_from_coeffs(coeffs, Nval, true)
    Vphys = build_secondkind_volterra_strictband(Vraw, Nval)
    A = build_secondkind_system_strictband(Vraw, Nval; sign=1.0)

    rhs_base = simple_rhs_fun()
    exact_base = simple_solution_fun()
    rhs_coeff = pad_to_length(rhs_base.coefficients, Nval)
    exact_coeff = pad_to_length(exact_base.coefficients, Nval)

    return (
        N=Nval,
        kerneldepth=1,
        coefficients=coeffs,
        raw_operator=Vraw,
        final_volterra_operator=Vphys,
        system=A,
        rhs_coefficients=rhs_coeff,
        exact_coefficients=exact_coeff,
    )
end

"""
    solve_simple_secondkind_sparse(; N=32)

Solve the manufactured `K=1` second-kind Volterra problem using the sparse
Gutleb--Olver workflow. Returns a `NamedTuple` with validation errors,
operator storage diagnostics, timings, `status`, and `notes`.

The solve uses the banded system directly and does not use dense collocation or
dense-to-sparse conversion.
"""
function solve_simple_secondkind_sparse(; N::Integer=32, n=nothing)
    Nval = _resolve_N(N, n)
    local built
    local residual
    local u_coeff
    local max_abs_error
    local rhs_validation_error
    local build_time
    local solve_time
    local evaluation_time

    total_time = @elapsed begin
        build_time = @elapsed begin
            built = build_simple_secondkind_system(; N=Nval)
        end

        residual = built.system * built.exact_coefficients - built.rhs_coefficients
        solve_time = @elapsed begin
            u_coeff = built.system \ built.rhs_coefficients
        end
        evaluation_time = @elapsed begin
            u_num = Fun(SIMPLE_SOLUTION_SPACE, u_coeff)
            max_abs_error = maximum(abs(u_num(x) - simple_exact_solution(x)) for x in DEFAULT_GRID)
            rhs_validation_error = maximum(
                abs(Fun(SIMPLE_SOLUTION_SPACE, built.rhs_coefficients)(x) - simple_rhs_function(x))
                for x in DEFAULT_GRID
            )
        end
    end

    raw_stats = banded_stats(built.raw_operator)
    final_volterra_stats = banded_stats(built.final_volterra_operator)
    system_stats = banded_stats(built.system)
    residual_relative = norm(residual) / norm(built.rhs_coefficients)
    timings = (
        build=build_time,
        rhs=0.0,
        kernel_coefficients=build_time,
        volterra_construction=0.0,
        system_assembly=0.0,
        solve=solve_time,
        evaluation=evaluation_time,
        exact_projection=0.0,
        total=total_time,
    )
    status = rhs_validation_error < 1e-13 &&
             residual_relative < 1e-13 &&
             max_abs_error < 1e-13 ? "pass" : "check"

    return (
        k=missing,
        N=Nval,
        kerneldepth=1,
        realized_kerneldepth=built.coefficients.kerneldepth,
        rhs_validation_error=rhs_validation_error,
        tau_residual=0.0,
        body_residual_relative=residual_relative,
        residual_relative=residual_relative,
        max_abs_error=max_abs_error,
        raw_operator_type=raw_stats.type,
        raw_operator_nnz=raw_stats.nnz,
        raw_operator_bandwidth=raw_stats.bandwidth,
        body_operator_type=final_volterra_stats.type,
        body_operator_nnz=final_volterra_stats.nnz,
        body_operator_bandwidth=final_volterra_stats.bandwidth,
        final_system_type=system_stats.type,
        final_system_nnz=system_stats.nnz,
        full_bandwidth=system_stats.bandwidth,
        body_bandwidth=final_volterra_stats.bandwidth,
        density=system_stats.density,
        timings=timings,
        status=status,
        notes=("Manufactured K=1 second-kind Volterra equation.",),
        raw_stats=raw_stats,
        final_volterra_stats=final_volterra_stats,
        body_stats=final_volterra_stats,
        system_stats=system_stats,
    )
end

simple_secondkind_sparse_case(; n=nothing, N::Integer=32) =
    solve_simple_secondkind_sparse(; N=N, n=n)

section522_exact_solution(x, k) = atan(k * x)

section522_kernel(x, y) = y * exp(x^2)

section522_rhs_function(x, k) =
    k / (k^2 * x^2 + 1.0) -
    (exp(x^2) * atan(k * x)) / (2.0 * k^2) +
    (exp(x^2) * x) / (2.0 * k) -
    0.5 * exp(x^2) * x^2 * atan(k * x)

initial_row_value(j::Integer) = isodd(j) ? 1.0 : -1.0

function section522_rhs_projection(k::Real, n::Integer)
    gF = Fun(x -> section522_rhs_function(x, k), SECTION522_OUTPUT_SPACE, n)
    c = pad_to_length(gF.coefficients, n)
    tail_start = max(1, length(c) - 9)
    validation_error = maximum(
        abs(gF(x) - section522_rhs_function(x, k))
        for x in DEFAULT_GRID
    )
    return (
        coefficients=c,
        coefficient_length=length(gF.coefficients),
        tail_norm=norm(view(c, tail_start:length(c))),
        validation_error=validation_error,
        method="Fun(rhs, Jacobi(1,2,0..1), N)",
        basis="Jacobi(1,2,0..1)",
    )
end

"""
    build_section522_rhs(; k=100.0, N=128)

Project the Section 5.2.2 right-hand side for
`u'(x) - int_0^x y*exp(x^2)*u(y)dy = g(x,k)` into the final output basis
`Jacobi(1,2,0..1)`. Returns a `NamedTuple` with coefficients, projection
metadata, tail norm, and physical-grid validation error.
"""
function build_section522_rhs(;
    k::Real=SECTION522_K,
    N::Integer=SECTION522_ACCEPTANCE_N,
    n=nothing,
)
    Nval = _resolve_N(N, n)
    return section522_rhs_projection(k, Nval)
end

function section522_exact_coefficients(k::Real, n::Integer)
    uF = Fun(x -> section522_exact_solution(x, k), SECTION522_SOLUTION_SPACE, n)
    return pad_to_length(uF.coefficients, n)
end

function assemble_section522_semibanded_system(body_operator, n::Integer)
    n < 2 && throw(ArgumentError("n must be at least 2"))
    size(body_operator, 1) == n && size(body_operator, 2) == n ||
        throw(DimensionMismatch("body operator must be n by n"))

    rowidx = Int[]
    colidx = Int[]
    vals = Float64[]

    for j = 1:n
        push!(rowidx, 1)
        push!(colidx, j)
        push!(vals, initial_row_value(j))
    end

    lower_bw, upper_bw = BandedMatrices.bandwidths(body_operator)
    lower_bw = Int(lower_bw)
    upper_bw = Int(upper_bw)

    @inbounds for source_row = 1:(n - 1)
        jlo = max(1, source_row - lower_bw)
        jhi = min(n, source_row + upper_bw)
        for j = jlo:jhi
            val = body_operator[source_row, j]
            if !iszero(val)
                push!(rowidx, source_row + 1)
                push!(colidx, j)
                push!(vals, val)
            end
        end
    end

    return sparse(rowidx, colidx, vals, n, n)
end

function section522_kernel_coefficients(kerneldepth::Integer)
    plan = precompute_triangular_kernel_plan(kerneldepth, SECTION522_FLIP)
    return triangular_kernel_coeffs_from_plan(plan, section522_kernel)
end

function section522_kernel_coefficients_timed(kerneldepth::Integer)
    local coeffs
    elapsed = @elapsed begin
        coeffs = section522_kernel_coefficients(kerneldepth)
    end
    return coeffs, elapsed
end

function section522_body_operators(n::Integer, coeffs)
    Vraw = triVolterraFullKernelOpP01_strictband_from_coeffs(
        coeffs,
        n,
        SECTION522_FLIP,
    )
    Vreflect_lower = apply_reflect_lowering_strictband(Vraw, n)
    Vfinal = apply_conversion_00_to_12_strictband(Vreflect_lower, n)
    D = Derivative(SECTION522_SOLUTION_SPACE, SECTION522_DERIVATIVE_ORDER)[1:n, 1:n]
    body = build_VIDE_LHS_sec522_strictband(Vraw, n)
    return (
        Vraw=Vraw,
        Vreflect_lower=Vreflect_lower,
        Vfinal=Vfinal,
        derivative=D,
        body=body,
    )
end

"""
    build_section522_operator(; N=128, kerneldepth=12)

Build the Section 5.2.2 strict-band/semi-banded operator package. The returned
`NamedTuple` contains the raw `Q_y K(1-X,Y) E_y` Volterra operator, reflected
lowered operator, final Volterra operator, derivative operator, body operator,
semi-banded sparse system, storage diagnostics, and construction timings.

The Volterra path uses `flip=true`, weighted lowering, reflection, and
`00 -> 12` conversion. No dense matrix is formed.
"""
function build_section522_operator(;
    N::Integer=SECTION522_ACCEPTANCE_N,
    n=nothing,
    kerneldepth::Integer=SECTION522_ACCEPTANCE_KERNELDEPTH,
)
    Nval = _resolve_N(N, n)
    local coeffs
    local ops
    local system
    local kernel_time
    local volterra_construction_time
    local system_assembly_time

    kernel_time = @elapsed begin
        coeffs = section522_kernel_coefficients(kerneldepth)
    end
    volterra_construction_time = @elapsed begin
        ops = section522_body_operators(Nval, coeffs)
    end
    system_assembly_time = @elapsed begin
        system = assemble_section522_semibanded_system(ops.body, Nval)
    end

    return (
        N=Nval,
        kerneldepth=Int(kerneldepth),
        realized_kerneldepth=coeffs.kerneldepth,
        coefficients=coeffs,
        raw_operator=ops.Vraw,
        reflected_lowered_operator=ops.Vreflect_lower,
        final_volterra_operator=ops.Vfinal,
        derivative_operator=ops.derivative,
        body_operator=ops.body,
        system=system,
        raw_stats=banded_stats(ops.Vraw),
        reflect_lower_stats=banded_stats(ops.Vreflect_lower),
        final_volterra_stats=banded_stats(ops.Vfinal),
        derivative_stats=banded_stats(ops.derivative),
        body_stats=banded_stats(ops.body),
        system_stats=sparse_stats(system),
        timings=(
            kernel_coefficients=kernel_time,
            volterra_construction=volterra_construction_time,
            system_assembly=system_assembly_time,
        ),
    )
end

"""
    solve_section522_k100_sparse(; N=128, kerneldepth=12, k=100.0)

Solve the Section 5.2.2 VIDE with exact solution `atan(k*x)` using the
strict-band/semi-banded sparse Gutleb--Olver workflow. Returns a structured
`NamedTuple` with validation errors, residuals, storage diagnostics, timings,
`status`, and `notes`.

The final system is assembled directly as `SparseMatrixCSC` from one
evaluation row plus banded body rows, and is solved without dense collocation or
dense-to-sparse conversion.
"""
function solve_section522_k100_sparse(;
    N::Integer=SECTION522_ACCEPTANCE_N,
    n=nothing,
    kerneldepth::Integer=SECTION522_ACCEPTANCE_KERNELDEPTH,
    k::Real=SECTION522_K,
)
    Nval = _resolve_N(N, n)
    local rhs
    local exact_coeff
    local built
    local b
    local residual
    local solution_coeff
    local max_abs_error
    local rhs_time
    local exact_projection_time
    local solve_time
    local evaluation_time

    workflow_time = @elapsed begin
        built = build_section522_operator(; N=Nval, kerneldepth=kerneldepth)
        rhs_time = @elapsed begin
            rhs = build_section522_rhs(; k=k, N=Nval)
        end
        exact_projection_time = @elapsed begin
            exact_coeff = section522_exact_coefficients(k, Nval)
        end

        b = zeros(Float64, Nval)
        b[2:Nval] .= rhs.coefficients[1:(Nval - 1)]

        residual = built.system * exact_coeff - b
        solve_time = @elapsed begin
            solution_coeff = built.system \ b
        end
        evaluation_time = @elapsed begin
            u_num = Fun(SECTION522_SOLUTION_SPACE, solution_coeff)
            max_abs_error = maximum(
                abs(u_num(x) - section522_exact_solution(x, k))
                for x in DEFAULT_GRID
            )
        end
    end

    body_residual = residual[2:end]
    body_rhs = b[2:end]
    body_residual_relative = norm(body_residual) / norm(body_rhs)
    raw_stats = built.raw_stats
    body_stats = built.body_stats
    system_stats = built.system_stats
    timings = (
        build=sum(values(built.timings)),
        rhs=rhs_time,
        kernel_coefficients=built.timings.kernel_coefficients,
        volterra_construction=built.timings.volterra_construction,
        system_assembly=built.timings.system_assembly,
        solve=solve_time,
        evaluation=evaluation_time,
        exact_projection=exact_projection_time,
        total=workflow_time,
    )
    status = rhs.validation_error <= 1e-5 &&
             body_residual_relative <= 1e-6 &&
             max_abs_error <= 1e-6 ? "pass" : "check"

    return (
        k=Float64(k),
        N=Nval,
        kerneldepth=Int(kerneldepth),
        realized_kerneldepth=built.realized_kerneldepth,
        rhs_projection_method=rhs.method,
        rhs_basis=rhs.basis,
        rhs_coefficient_length=rhs.coefficient_length,
        rhs_tail_norm=rhs.tail_norm,
        rhs_validation_error=rhs.validation_error,
        tau_residual=abs(residual[1]),
        body_residual_norm=norm(body_residual),
        body_residual_relative=body_residual_relative,
        residual_first=residual[1:min(10, length(residual))],
        residual_last=residual[max(1, length(residual) - 9):length(residual)],
        max_abs_error=max_abs_error,
        raw_operator_type=raw_stats.type,
        raw_operator_nnz=raw_stats.nnz,
        raw_operator_bandwidth=raw_stats.bandwidth,
        body_operator_type=body_stats.type,
        body_operator_nnz=body_stats.nnz,
        body_operator_bandwidth=body_stats.bandwidth,
        final_system_type=system_stats.type,
        final_system_nnz=system_stats.nnz,
        full_bandwidth=system_stats.bandwidth,
        body_bandwidth=body_stats.bandwidth,
        density=system_stats.density,
        timings=timings,
        status=status,
        notes=(
            "Section 5.2.2 k=100 strict-band/semi-banded sparse VIDE workflow.",
            "No dense collocation or dense-to-sparse matrix conversion is used.",
        ),
        raw_stats=raw_stats,
        reflect_lower_stats=built.reflect_lower_stats,
        final_volterra_stats=built.final_volterra_stats,
        derivative_stats=built.derivative_stats,
        body_stats=body_stats,
        system_stats=system_stats,
        rhs_time=rhs_time,
        kernel_coefficient_time=built.timings.kernel_coefficients,
        volterra_construction_time=built.timings.volterra_construction,
        system_assembly_time=built.timings.system_assembly,
        solve_time=solve_time,
        evaluation_time=evaluation_time,
        exact_projection_time=exact_projection_time,
        total_time=workflow_time,
    )
end

"""
    print_section522_result(result)

Print the detailed Section 5.2.2 workflow result returned by
`solve_section522_k100_sparse`, including residuals, storage diagnostics, and
timings. Returns the input result unchanged.
"""
function print_section522_result(row)
    @printf(
        "k=%.6g N=%d kerneldepth=%d rhs_validation=%.17e tau_residual=%.17e body_residual_relative=%.17e max_abs_error=%.17e total_time=%.17e\n",
        row.k,
        row.N,
        row.kerneldepth,
        row.rhs_validation_error,
        row.tau_residual,
        row.body_residual_relative,
        row.max_abs_error,
        row.total_time,
    )
    println("  raw V      : ", row.raw_stats.type,
            " nnz=", row.raw_stats.nnz,
            " bw=(", row.raw_stats.lower_bandwidth, ",", row.raw_stats.upper_bandwidth, ")",
            " density=", @sprintf("%.6e", row.raw_stats.density))
    println("  body       : ", row.body_stats.type,
            " nnz=", row.body_stats.nnz,
            " bw=(", row.body_stats.lower_bandwidth, ",", row.body_stats.upper_bandwidth, ")",
            " density=", @sprintf("%.6e", row.body_stats.density))
    println("  system     : ", row.system_stats.type,
            " nnz=", row.system_stats.nnz,
            " bw=(", row.system_stats.lower_bandwidth, ",", row.system_stats.upper_bandwidth, ")",
            " density=", @sprintf("%.6e", row.system_stats.density))
    @printf(
        "  timings: rhs=%.17e kernel_coeff=%.17e volterra=%.17e system_assembly=%.17e solve=%.17e evaluation=%.17e exact_projection=%.17e total=%.17e\n",
        row.rhs_time,
        row.kernel_coefficient_time,
        row.volterra_construction_time,
        row.system_assembly_time,
        row.solve_time,
        row.evaluation_time,
        row.exact_projection_time,
        row.total_time,
    )
    println("  residual first entries = ", row.residual_first)
    println("  residual last entries  = ", row.residual_last)
    return row
end

"""
    print_section522_summary(result)

Print a single CSV-style summary row for a Section 5.2.2 workflow result.
Returns the input result unchanged.
"""
function print_section522_summary(row)
    println("k,N,kerneldepth,rhs_validation_error,raw_operator_type,raw_lower_bandwidth,raw_upper_bandwidth,body_operator_type,body_lower_bandwidth,body_upper_bandwidth,final_system_type,system_nnz,lower_bandwidth,upper_bandwidth,density,residual_relative,max_abs_error,rhs_time,kernel_coefficient_time,volterra_construction_time,system_assembly_time,solve_time,evaluation_time,total_time")
    @printf(
        "%.17e,%d,%d,%.17e,%s,%d,%d,%s,%d,%d,%s,%d,%d,%d,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e\n",
        row.k,
        row.N,
        row.kerneldepth,
        row.rhs_validation_error,
        row.raw_stats.type,
        row.raw_stats.lower_bandwidth,
        row.raw_stats.upper_bandwidth,
        row.body_stats.type,
        row.body_stats.lower_bandwidth,
        row.body_stats.upper_bandwidth,
        row.system_stats.type,
        row.system_stats.nnz,
        row.system_stats.lower_bandwidth,
        row.system_stats.upper_bandwidth,
        row.system_stats.density,
        row.body_residual_relative,
        row.max_abs_error,
        row.rhs_time,
        row.kernel_coefficient_time,
        row.volterra_construction_time,
        row.system_assembly_time,
        row.solve_time,
        row.evaluation_time,
        row.total_time,
    )
    return row
end

const RESULT_CSV_COLUMNS = (
    :k,
    :N,
    :kerneldepth,
    :rhs_validation_error,
    :tau_residual,
    :body_residual_relative,
    :max_abs_error,
    :raw_operator_type,
    :raw_operator_nnz,
    :raw_operator_bandwidth,
    :body_operator_type,
    :body_operator_nnz,
    :body_operator_bandwidth,
    :final_system_type,
    :final_system_nnz,
    :full_bandwidth,
    :body_bandwidth,
    :density,
    :rhs_time,
    :kernel_coefficient_time,
    :volterra_construction_time,
    :system_assembly_time,
    :solve_time,
    :evaluation_time,
    :exact_projection_time,
    :total_time,
    :status,
    :notes,
)

_csv_results(result::NamedTuple) = (result,)
_csv_results(results) = results

function _result_value(result::NamedTuple, key::Symbol)
    if key in keys(result)
        return getproperty(result, key)
    elseif key == :rhs_time && :timings in keys(result)
        return result.timings.rhs
    elseif key == :kernel_coefficient_time && :timings in keys(result)
        return result.timings.kernel_coefficients
    elseif key == :volterra_construction_time && :timings in keys(result)
        return result.timings.volterra_construction
    elseif key == :system_assembly_time && :timings in keys(result)
        return result.timings.system_assembly
    elseif key == :solve_time && :timings in keys(result)
        return result.timings.solve
    elseif key == :evaluation_time && :timings in keys(result)
        return result.timings.evaluation
    elseif key == :exact_projection_time && :timings in keys(result)
        return result.timings.exact_projection
    elseif key == :total_time && :timings in keys(result)
        return result.timings.total
    else
        return missing
    end
end

function _csv_escape(value)
    text = value === missing ? "" : string(value)
    if occursin(',', text) || occursin('"', text) || occursin('\n', text) || occursin('\r', text)
        return "\"" * replace(text, "\"" => "\"\"") * "\""
    end
    return text
end

function _write_result_csv_header(io)
    println(io, join(string.(RESULT_CSV_COLUMNS), ","))
end

function _write_result_csv_row(io, result::NamedTuple)
    values = (_csv_escape(_result_value(result, col)) for col in RESULT_CSV_COLUMNS)
    println(io, join(values, ","))
end

"""
    write_results_csv(path, results)

Write one or more workflow result `NamedTuple`s to `path` as CSV using the
stable public result fields. This utility uses only Base I/O and keeps all
operator diagnostics on the sparse/banded path.
"""
function write_results_csv(path::AbstractString, results)
    open(path, "w") do io
        _write_result_csv_header(io)
        for result in _csv_results(results)
            _write_result_csv_row(io, result)
        end
    end
    return path
end

"""
    append_result_csv(path, result)

Append one workflow result `NamedTuple` to a CSV file, writing the header first
when the file is new or empty.
"""
function append_result_csv(path::AbstractString, result::NamedTuple)
    needs_header = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        needs_header && _write_result_csv_header(io)
        _write_result_csv_row(io, result)
    end
    return path
end
