# problem: warmup
# original script: examples/warmup/solve_vide_low_order_cheb_leg.jl
# N source: DEFAULT_NS = collect(1:100)
# error_type: max_abs, from max_error_cheb/max_error_leg on the physical [-1,1] grid
# original integral form: int_{-1}^x K(x,t)u(t)dt
# GO integral form after transform: int_0^s with x_original = 2s - 1
# kernel transform: K_go(s,r) = 2*K_original(2s-1,2r-1)
# GO solution basis: GO shifted Jacobi P^(1,0) if a certified sparse VIDE wrapper is added
# voltmat basis: Chebyshev, matching the Chebyshev branch of the original script
# error grid: physical [0,1] s grid, equivalent to original x=2s-1 grid

include(joinpath(dirname(@__DIR__), "bench_common.jl"))

using Pkg

const WARMUP_GO_PROJECT = joinpath(BENCH_PROJECT_ROOT, "others", "GutlebOlver")
const WARMUP_GO_AVAILABLE = Ref(false)

try
    Pkg.activate(WARMUP_GO_PROJECT; io=devnull)
    @eval using SparseVolterraNew
    @eval using ApproxFun
    @eval using SparseArrays
    Pkg.activate(BENCH_PROJECT_ROOT; io=devnull)
    WARMUP_GO_AVAILABLE[] = true
catch err
    Pkg.activate(BENCH_PROJECT_ROOT; io=devnull)
    @warn "SparseVolterraNew is not available for warmup GO solve" exception=(err, catch_backtrace())
end

K_original(x, t) = ((x + 1) * (x + 2) / 4) * exp(((t + 1) * (x - t)) / 4)
K_go(s, r) = 2.0 * K_original(chebX(s), chebX(r))
u_go(s) = exp(s^2)

function warmup_kerneldepth(N::Int)
    return min(10, max(3, ceil(Int, sqrt(max(N, 2)))))
end

function warmup_go_eval_row(n::Int)
    S = Jacobi(0, 1, 0..1)
    vals = zeros(Float64, n)
    for j in 1:n
        ej = zeros(Float64, n)
        ej[j] = 1.0
        vals[j] = Fun(S, ej)(0.0)
    end
    return vals
end

function warmup_go_solve_coefficients(N::Int)
    n = max(N, 2)
    kerneldepth = warmup_kerneldepth(N)

    local kcoeffs
    kernel_approx_time = @elapsed begin
        kcoeffs = build_gutleb_kernel_coefficients(K_go, kerneldepth; flip=true)
    end

    local VK
    matrix_construct_time = @elapsed begin
        VK = build_gutleb_operator_from_coefficients(kcoeffs, n; flip=true)
    end
    reject_dense_go_matrix(VK, N; name="warmup GO Volterra operator")

    local rhs_body, rhs_info
    rhs_time = @elapsed begin
        rhs_body, rhs_info = build_go_rhs_coefficients(
            s -> 2s + 1,
            n;
            alpha=2,
            beta=1,
            method=:fixed_gauss_jacobi,
            projection_tol=1.0e-10,
            oversampling=3,
        )
    end

    rhs_info.status == :success ||
        error("warmup GO RHS projection failed: $(go_rhs_info_notes(rhs_info))")

    local A, rhs, body
    system_assembly_time = @elapsed begin
        Ssol = Jacobi(0, 1, 0..1)
        Sout = Jacobi(1, 2, 0..1)

        Vrl = apply_reflect_lowering_strictband(VK, n)
        Vconv = apply_conversion_00_to_12_strictband(Vrl, n)
        D = Derivative(Ssol, 1)[1:n, 1:n]
        C01to12 = Conversion(Ssol, Sout)[1:n, 1:n]
        body = D + C01to12 - Vconv

        tau_vals = warmup_go_eval_row(n)
        tau_row = sparse(ones(Int, n), collect(1:n), tau_vals, 1, n)
        A = vcat(tau_row, sparse(body[1:(n - 1), :]))

        rhs = Vector{Float64}(undef, n)
        rhs[1] = 1.0
        rhs[2:end] .= rhs_body[1:(n - 1)]
    end
    reject_dense_go_matrix(A, N; name="warmup GO final system")

    local sol
    solve_time = @elapsed begin
        sol = A \ rhs
    end

    return (
        coeffs = sol,
        A = A,
        VK = VK,
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

warmup_go_values(c::AbstractVector, grid) = Fun(Jacobi(0, 1, 0..1), c).(grid)

function warmup_gutleb_olver_row(prob::BenchProblem, N::Int)
    if !WARMUP_GO_AVAILABLE[]
        return fail_row(prob, "gutleb_olver", N, "skipped",
                        "skipped: SparseVolterraNew could not be loaded; no GO solve was run")
    end

    sol = warmup_go_solve_coefficients(N)
    grid = collect(range(0.0, 1.0; length=max(1001, 4 * sol.n)))
    vals = warmup_go_values(sol.coeffs, grid)
    exact_vals = prob.exact.(grid)
    exact_error = maximum(abs.(vals .- exact_vals))
    isfinite(exact_error) || error("non-finite warmup GO exact_error")
    correctness_ok = exact_error <= 1.0e-5

    go_system_nnz = safe_nnz(sol.A)
    go_density = safe_density(sol.A)
    go_lb = safe_lower_bandwidth(sol.A)
    go_ub = safe_upper_bandwidth(sol.A)
    rhs_fields = rhs_projection_row_fields(sol.rhs_info)

    notes = join((
        correctness_ok ?
            "GO warmup sparse/banded VIDE solve passed physical-grid exact validation" :
            "failed: GO warmup sparse/banded VIDE solve did not pass physical-grid exact validation; exact_error=$(exact_error)",
        "solution basis=GO shifted Jacobi P^(1,0); final output/RHS basis=GO shifted Jacobi P^(2,1)",
        go_rhs_info_notes(sol.rhs_info),
        "dense collocation and dense final Matrix rejected",
    ), "; ")

    return merge((
        problem = prob.name,
        method = "gutleb_olver",
        N = N,
        status = correctness_ok ? "success" : "failed",
        error_type = prob.error_type,
        error = exact_error,
        exact_error = exact_error,
        cauchy_error = NaN,
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
        reference_N = "missing",
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
    name = "warmup",
    Ns = collect(1:100),
    ncoeff_from_N = N -> max(N, 2),
    rhs = s -> 2s + 1,
    exact = u_go,
    local_coeff = s -> 1.0,
    deriv_coeff = s -> 1.0,
    volterra_terms = [(sign = -1.0, kernel = K_go, input_scale = 1.0)],
    boundary_left = 1.0,
    error_type = "max_abs",
    primary_error = :exact,
    original_script = "examples/warmup/solve_vide_low_order_cheb_leg.jl",
    n_source = "DEFAULT_NS = collect(1:100)",
    error_source = "max_error_cheb/max_error_leg",
    integral_form = "int_{-1}^x, transformed to int_0^s",
    go_integral_form = "int_0^s",
    variable_transform = "x_original = 2s - 1; u_go(s)=u_original(2s-1); K_go(s,r)=2*K_original(2s-1,2r-1)",
    go_output_basis = "solution basis GO shifted Jacobi P^(1,0); final VIDE output/RHS basis GO shifted Jacobi P^(2,1)",
        notes = "warmup VIDE: u_s'(s)+u(s)-int_0^s K_go(s,r)u(r)dr=2s+1",
    go_solver = warmup_gutleb_olver_row,
)

run_benchmark(prob)
