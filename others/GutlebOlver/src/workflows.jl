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
`P^(1,0)` solution/output basis.  The construction keeps the strict-band path
and does not form a dense operator.
"""
function build_simple_secondkind_system(; N::Integer=32, n=nothing)
    Nval = _resolve_N(N, n)
    plan = precompute_triangular_kernel_plan(1, true)
    coeffs = triangular_kernel_coeffs_from_plan(plan, simple_kernel)
    Vraw = triVolterraFullKernelOpP01_strictband_from_coeffs(coeffs, Nval, true)
    Vphys = build_secondkind_volterra_strictband(Vraw, Nval)
    A = build_secondkind_equation_system(Vraw, Nval; convention=:plus)

    rhs_coeff, rhs_info = build_go_rhs_coefficients(
        simple_rhs_function,
        Nval;
        alpha=0,
        beta=1,
        method=:exact_polynomial,
    )
    exact_base = simple_solution_fun()
    exact_coeff = pad_to_length(exact_base.coefficients, Nval)

    return (
        N=Nval,
        kerneldepth=1,
        coefficients=coeffs,
        raw_operator=Vraw,
        final_volterra_operator=Vphys,
        system=A,
        rhs_coefficients=rhs_coeff,
        rhs_info=rhs_info,
        exact_coefficients=exact_coeff,
    )
end

"""
    solve_simple_secondkind_sparse(; N=32)

Solve the manufactured `K=1` second-kind Volterra problem using the sparse
Gutleb--Olver workflow.  Returns validation errors, diagnostics, timings,
`status`, and `notes`.
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
        final_system_bandwidth=system_stats.bandwidth,
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

function section522_rhs_projection(k::Real, n::Integer)
    coefficients, info = build_go_rhs_coefficients(
        x -> section522_rhs_function(x, k),
        n;
        alpha=1,
        beta=2,
        method=:fixed_gauss_jacobi,
    )
    return (
        coefficients=coefficients,
        rhs_coeff=coefficients,
        coefficient_length=info.coefficient_length,
        tail_norm=info.tail_norm,
        validation_error=info.validation_error,
        method=info.method,
        basis=info.basis,
        status=info.status,
        notes=info.notes,
    )
end

"""
    build_section522_rhs(; k=100.0, N=128)

Project the Section 5.2.2 right-hand side into `Jacobi(1,2,0..1)`.
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
    D = build_derivative_P01_to_P12(n)
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

Build the Section 5.2.2 strict-band/semi-banded operator package.  The
Volterra path uses `flip=true`, weighted lowering, reflection, and `00 -> 12`
conversion.  No dense matrix is formed.
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
strict-band/semi-banded sparse Gutleb--Olver workflow.
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
    build_total = built.timings.kernel_coefficients +
                  built.timings.volterra_construction +
                  built.timings.system_assembly
    timings = (
        build=build_total,
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
        final_system_bandwidth=system_stats.bandwidth,
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
`solve_section522_k100_sparse`.
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
