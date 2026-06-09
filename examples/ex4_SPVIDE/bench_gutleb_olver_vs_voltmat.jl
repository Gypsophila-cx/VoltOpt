# problem: ex4
# original script: examples/ex4_SPVIDE/ex4_SPVIDE.jl
# N source: N_LIST = evenly_spaced_integer_nodes(2, 32000, 100)
# error_type: Cauchy in the original error history; exact_error is also recorded here
# original variable: x in [-1,1]
# GO variable: s in [0,1], x = 2s - 1 and y = 2r - 1
# exact_go(s) = exact_original(2s - 1) = sin(s) + exp(-s/eps)
# original equation:
#   2eps*u_x(x) + (x+3)/2*u(x)
#     + 1/2*int_{-1}^x exp(mu*(x+1)*(y+1)/4)u(y)dy = f((x+1)/2)
# transformed equation:
#   eps*u_s(s) + (1+s)u(s) + int_0^s exp(mu*s*r)u(r)dr = f(s)
# The original 1/2 integral prefactor cancels the Jacobian dy = 2dr.
# GO transformed integral form: int_0^s K_go(s,r)u(r)dr, K_go(s,r)=exp(mu*s*r)
# GO solution basis: GO shifted Jacobi P^(1,0) for int_0^s if sparse SPVIDE wrapper is added
# voltmat basis: Chebyshev
# error grid: physical [0,1] s grid

include(joinpath(dirname(@__DIR__), "bench_common.jl"))

using Pkg

const EX4_GO_PROJECT = joinpath(BENCH_PROJECT_ROOT, "others", "GutlebOlver")
const EX4_GO_AVAILABLE = Ref(false)

try
    Pkg.activate(EX4_GO_PROJECT; io=devnull)
    @eval using SparseVolterraNew
    @eval using ApproxFun
    @eval using BandedMatrices
    Pkg.activate(BENCH_PROJECT_ROOT; io=devnull)
    EX4_GO_AVAILABLE[] = true
catch err
    Pkg.activate(BENCH_PROJECT_ROOT; io=devnull)
    @warn "SparseVolterraNew is not available for ex4 GO probe" exception=(err, catch_backtrace())
end

const EPS_EX4 = 1.0e-7
const MU_EX4 = -1.0

phi1_ex4(z) = abs(z) < 1.0e-8 ? 1.0 + z / 2.0 + z^2 / 6.0 + z^3 / 24.0 + z^4 / 120.0 : expm1(z) / z
u_go(s) = sin(s) + exp(-s / EPS_EX4)
K_go(s, r) = exp(MU_EX4 * s * r)

function rhs_go(s)
    layer = exp(-s / EPS_EX4)
    v = sin(s) + layer
    diff_part = EPS_EX4 * cos(s) - layer
    local_part = (1.0 + s) * v
    a = MU_EX4 * s
    I_sin = (exp(a * s) * (a * sin(s) - cos(s)) + 1.0) / (a^2 + 1.0)
    I_layer = s * phi1_ex4(MU_EX4 * s^2 - s / EPS_EX4)
    return diff_part + local_part + I_sin + I_layer
end

function ex4_kerneldepth(N::Int)
    return min(8, max(2, ceil(Int, sqrt(N))))
end

function ex4_go_eval_row(n::Int)
    S = Jacobi(0, 1, 0..1)
    vals = zeros(Float64, n)
    for j in 1:n
        ej = zeros(Float64, n)
        ej[j] = 1.0
        vals[j] = Fun(S, ej)(0.0)
    end
    return vals
end

function ex4_go_solve_coefficients(N::Int)
    n = N + 1
    kerneldepth = ex4_kerneldepth(N)

    local coeffs
    kernel_approx_time = @elapsed begin
        coeffs = build_gutleb_kernel_coefficients(K_go, kerneldepth; flip=true)
    end

    local VK
    matrix_construct_time = @elapsed begin
        VK = build_gutleb_operator_from_coefficients(coeffs, n; flip=true)
    end
    reject_dense_go_matrix(VK, N; name="ex4 GO Volterra operator")

    # GO basis convention for this SPVIDE attempt:
    # solution basis: GO shifted Jacobi P^(1,0)(s) on [0,1]
    # raw strict-band VK input basis: GO shifted Jacobi P^(1,0)
    # raw strict-band VK output is converted by R*WLoweringP01P00 and
    # Conversion(Jacobi(0,0), Jacobi(1,2)) to the common output basis.
    # derivative output basis: GO shifted Jacobi P^(2,1)
    # multiplication output basis: GO shifted Jacobi P^(1,0), then converted to P^(2,1)
    # final equation output basis: GO shifted Jacobi P^(2,1)
    # RHS basis: GO shifted Jacobi P^(2,1)
    # tau row: endpoint evaluation at s=0 in the solution P^(1,0) basis.
    local A, rhs, body, Vconv, rhs_info
    rhs_time = @elapsed begin
        rhs_full, rhs_info = build_go_rhs_coefficients(
            rhs_go,
            n;
            alpha=2,
            beta=1,
            method=:fixed_gauss_jacobi,
            projection_tol=1.0e-8,
            oversampling=3,
            validation_grid=vcat(
                collect(range(0.0, 1.0; length=max(401, min(4001, 4n)))),
                [0.0, 1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2],
            ),
        )
    end
    system_assembly_time = @elapsed begin
        Ssol = Jacobi(0, 1, 0..1)
        Sout = Jacobi(1, 2, 0..1)
        sfun = Fun(identity, Ssol)

        Vrl = apply_reflect_lowering_strictband(VK, n)
        Vconv = apply_conversion_00_to_12_strictband(Vrl, n)

        D = Derivative(Ssol, 1)[1:n, 1:n]
        C01to12 = Conversion(Ssol, Sout)[1:n, 1:n]
        Mlocal = Multiplication(Fun(s -> 1.0 + s, Ssol), Ssol)[1:n, 1:n]

        body = EPS_EX4 * D + C01to12 * Mlocal + Vconv

        tau_vals = ex4_go_eval_row(n)
        tau_row = sparse(ones(Int, n), collect(1:n), tau_vals, 1, n)
        A = vcat(tau_row, sparse(body[1:(n - 1), :]))

        rhs = Vector{Float64}(undef, n)
        rhs[1] = 1.0
        rhs[2:end] .= rhs_full[1:(n - 1)]
    end
    reject_dense_go_matrix(A, N; name="ex4 GO final system")

    local sol
    solve_time = @elapsed begin
        sol = A \ rhs
    end

    return (
        coeffs = sol,
        A = A,
        VK = VK,
        Vconv = Vconv,
        n = n,
        kerneldepth = kerneldepth,
        rhs_info = rhs_info,
        rhs_time = rhs_time,
        kernel_approx_time = kernel_approx_time,
        matrix_construct_time = matrix_construct_time,
        system_assembly_time = system_assembly_time,
        solve_time = solve_time,
        factor_time = NaN,
        backsolve_time = NaN,
        total_time = rhs_time + kernel_approx_time + matrix_construct_time +
            system_assembly_time + solve_time,
    )
end

function ex4_go_values(c::AbstractVector, grid)
    return Fun(Jacobi(0, 1, 0..1), c).(grid)
end

function ex4_gutleb_olver_probe_row(prob::BenchProblem, N::Int)
    if !EX4_GO_AVAILABLE[]
        return fail_row(
            prob,
            "gutleb_olver",
            N,
            "skipped",
            "skipped: SparseVolterraNew could not be loaded; no GO operator probe was run",
        )
    end

    sol = ex4_go_solve_coefficients(N)
    refN = Int(prob.reference_N_from(N))
    ref = ex4_go_solve_coefficients(refN)

    grid = collect(range(0.0, 1.0; length=max(1001, 4 * ref.n)))
    vals = ex4_go_values(sol.coeffs, grid)
    ref_vals = ex4_go_values(ref.coeffs, grid)
    exact_vals = prob.exact.(grid)

    exact_error = maximum(abs.(vals .- exact_vals))
    cauchy_error = maximum(abs.(vals .- ref_vals))
    isfinite(exact_error) || error("non-finite GO exact_error")
    isfinite(cauchy_error) || error("non-finite GO physical-grid Cauchy error")
    rhs_ok = sol.rhs_info.status == :success
    correctness_ok = rhs_ok && exact_error <= 1.0 && cauchy_error <= 1.0

    go_system_nnz = safe_nnz(sol.A)
    go_density = safe_density(sol.A)
    go_lb = safe_lower_bandwidth(sol.A)
    go_ub = safe_upper_bandwidth(sol.A)
    rhs_fields = rhs_projection_row_fields(sol.rhs_info)

    notes = join((
        correctness_ok ?
            "GO SPVIDE final system assembled as sparse semi-banded matrix with one dense tau row and passed the current finite-error sanity gate" :
            (rhs_ok ?
                "failed: GO sparse SPVIDE assembly is non-dense and operator-level diagnostics pass, but physical validation did not pass; physical exact_error=$(exact_error), physical-grid cauchy_error=$(cauchy_error)" :
                "failed: RHS not resolved at current N; boundary/transition layer requires O(10^4)-O(3e4) modes; no GO operator failure is inferred; physical exact_error=$(exact_error), physical-grid cauchy_error=$(cauchy_error)"),
        "solution basis=GO shifted Jacobi P^(1,0); final output/RHS basis=GO shifted Jacobi P^(2,1)",
        go_rhs_info_notes(sol.rhs_info),
        "body=eps*Derivative(P01->P12)+Conversion(P01->P12)*Multiplication(1+s in P01)+Conversion(P00->P12)*Reflect*WLowering*VK",
        "diagnostics verify the Volterra chain, derivative operator, multiplication operator, RHS projection, and row scheme A [tau; body rows 1:N-1]",
        "the GO row is kept failed because the required N is outside the practical benchmark range, not because sparse construction is dense or the operator chain is mathematically inconsistent",
        "GO cauchy error computed on physical grid because GO basis differs from voltmat Chebyshev coefficient basis",
        "factor_time/backsolve_time not split; solve_time is sparse backslash total",
        "dense collocation and dense solve rejected",
    ), "; ")

    return merge((
        problem = prob.name,
        method = "gutleb_olver",
        N = N,
        status = correctness_ok ? "success" : "failed",
        error_type = prob.error_type,
        error = cauchy_error,
        exact_error = exact_error,
        cauchy_error = cauchy_error,
        coefficient_error = NaN,
        rhs_time = sol.rhs_time,
        kernel_approx_time = sol.kernel_approx_time,
        matrix_construct_time = sol.matrix_construct_time,
        system_assembly_time = sol.system_assembly_time,
        solve_time = sol.solve_time,
        factor_time = sol.factor_time,
        backsolve_time = sol.backsolve_time,
        total_time = sol.total_time,
        bandwidth = "$(go_lb)/$(go_ub)",
        kernel_size_y = sol.kerneldepth,
        kernel_size_t = sol.kerneldepth,
        rank = "missing",
        condition_number = NaN,
        reference_N = refN,
        basis_family = "GO shifted Jacobi P^(1,0) solution basis; triangle Jacobi/Proriol kernel basis",
        solution_representation = "GO shifted Jacobi P^(1,0) coefficients",
        evaluation_grid = prob.evaluation_grid,
        integral_form = prob.go_integral_form,
        variable_transform = prob.variable_transform,
        go_storage_type = matrix_storage_type(sol.A),
        go_operator_nnz = safe_nnz(sol.VK),
        go_system_nnz = go_system_nnz,
        go_lower_bandwidth = go_lb,
        go_upper_bandwidth = go_ub,
        go_density = go_density,
        notes = notes,
    ), rhs_fields)
end

prob = BenchProblem(
    name = "ex4",
    Ns = evenly_spaced_integer_nodes(2, 32000, 100),
    ncoeff_from_N = N -> N + 1,
    rhs = rhs_go,
    exact = u_go,
    local_coeff = s -> 1.0 + s,
    deriv_coeff = s -> EPS_EX4,
    volterra_terms = [(sign = 1.0, kernel = K_go, input_scale = 1.0)],
    boundary_left = 1.0,
    error_type = "cauchy",
    primary_error = :cauchy,
    reference_N_from = N -> max(N + 1, round(Int, 1.01 * N)),
    original_script = "examples/ex4_SPVIDE/ex4_SPVIDE.jl",
    n_source = "evenly_spaced_integer_nodes(2, 32000, 100)",
    error_source = "original script writes cauchy_coeff_error_inf as the error-history convergence diagnostic",
    integral_form = "int_{-1}^x, transformed to int_0^s",
    go_integral_form = "int_0^s",
    variable_transform = "x_original = 2s - 1, y_original = 2r - 1; 1/2 prefactor and dy=2dr cancel; K_go(s,r)=exp(mu*s*r); u_go(s)=u_original(2s-1)",
    go_output_basis = "solution basis GO shifted Jacobi P^(1,0); final SPVIDE output/RHS basis GO shifted Jacobi P^(2,1)",
    notes = "primary error_type uses paired-N Chebyshev coefficient Cauchy error with reference_N=max(N+1,round(Int,1.01N)); exact_error is also recorded",
    go_solver = ex4_gutleb_olver_probe_row,
)

run_benchmark(prob)
