# problem: ex1
# original script: examples/ex1_2ndVolterraEqn/ex1_aging_maxwell.jl
# N source: evenly_spaced_integer_nodes(2, 40000, 100)
# error_type: max_abs, from err_inf in the original script
# original equation: u(x) + int_{-1}^x K_original(x,y)u(y)dy = f(x), x in [-1,1]
# transformed equation: u_go(s) + int_0^s K_go(s,r)u_go(r)dr = rhs_go(s), s in [0,1]
# original integral form: int_{-1}^x K(x,y)u(y)dy
# GO integral form after transform: int_0^s with x_original = 2s - 1
# variable transform: x = 2s - 1, y = 2r - 1
# derivative scaling: none; ex1 is a second-kind Volterra integral equation, not a VIDE
# kernel transform: K_go(s,r) = 2*K_original(2s-1,2r-1)
# RHS transform: rhs_go(s) = f_original(2s-1), built independently by split Gauss quadrature
# exact solution transform: u_go(s) = u_original(2s-1)
# GO output basis: GO shifted Jacobi P^(1,0) for the second-kind int_0^s system
# voltmat basis: Chebyshev
# error grid: physical [0,1] s grid, mapped from original x=2s-1

include(joinpath(dirname(@__DIR__), "bench_common.jl"))

using Pkg

const EX1_GO_PROJECT = joinpath(BENCH_PROJECT_ROOT, "others", "GutlebOlver")
const EX1_GO_AVAILABLE = Ref(false)

try
    Pkg.activate(EX1_GO_PROJECT; io=devnull)
    @eval using SparseVolterraNew
    @eval using ApproxFun
    Pkg.activate(BENCH_PROJECT_ROOT; io=devnull)
    EX1_GO_AVAILABLE[] = true
catch err
    Pkg.activate(BENCH_PROJECT_ROOT; io=devnull)
    @warn "SparseVolterraNew is not available for ex1 GO solve" exception=(err, catch_backtrace())
end

const x1 = -0.35
const x2 = 0.42
const A1 = 1.0
const A2 = -0.6
const delta1 = 1.0e-3
const delta2 = 1.5e-3

Sstep(x, x0, delta) = 0.5 * (1.0 + tanh((x - x0) / delta))
u_original(x) = 1.0 + A1 * Sstep(x, x1, delta1) + A2 * Sstep(x, x2, delta2)
tau_xy(x, y) = 1.5 + x - 0.5 * y
K_original(x, y) = exp(-(x - y) / tau_xy(x, y)) / tau_xy(x, y)
K_go(s, r) = 2.0 * K_original(chebX(s), chebX(r))
u_go(s) = u_original(chebX(s))

function gausslegendre_rule(n::Int)
    beta = [k / sqrt(4.0 * k^2 - 1.0) for k in 1:n-1]
    E = eigen(SymTridiagonal(zeros(n), beta))
    p = sortperm(E.values)
    return E.values[p], (2 .* abs2.(E.vectors[1, :]))[p]
end

const GLX_EX1, GLW_EX1 = gausslegendre_rule(160)
const RHS_SPLIT_L_EX1 = 40.0
const EX1_GO_MAX_RESIDUAL_REL = parse(Float64, get(ENV, "GO_EX1_MAX_OK_RESIDUAL", "1e-6"))
const EX1_GO_MAX_ERROR = parse(Float64, get(ENV, "GO_EX1_MAX_OK_ERROR", "1e-5"))
const EX1_GO_MAX_VOLTERRA_ERROR = parse(Float64, get(ENV, "GO_EX1_MAX_OK_VOLTERRA_ERROR", "1e-6"))

function gauss_integral_on_interval_ex1(f, a::Float64, b::Float64)
    b > a || return 0.0
    c = 0.5 * (a + b)
    h = 0.5 * (b - a)
    acc = 0.0
    @inbounds for i in eachindex(GLX_EX1)
        r = c + h * GLX_EX1[i]
        acc += GLW_EX1[i] * f(r)
    end
    return h * acc
end

function rhs_integral_direct_go_ex1(s::Real)
    sf = Float64(s)
    sf <= 0.0 && return 0.0

    pts = Float64[0.0, sf]
    for (x0, delta) in ((x1, delta1), (x2, delta2))
        a = fromX(x0 - RHS_SPLIT_L_EX1 * delta)
        c = fromX(x0)
        b = fromX(x0 + RHS_SPLIT_L_EX1 * delta)
        0.0 < a < sf && push!(pts, a)
        0.0 < c < sf && push!(pts, c)
        0.0 < b < sf && push!(pts, b)
    end

    sort!(pts)
    unique_pts = Float64[]
    lastp = NaN
    for p in pts
        if isempty(unique_pts) || abs(p - lastp) > 10eps(Float64) * max(1.0, abs(p))
            push!(unique_pts, p)
            lastp = p
        end
    end

    total = 0.0
    @inbounds for j in 1:(length(unique_pts) - 1)
        a = unique_pts[j]
        b = unique_pts[j + 1]
        total += gauss_integral_on_interval_ex1(r -> K_go(sf, r) * u_go(r), a, b)
    end
    return total
end

rhs_go(s) = u_go(s) + rhs_integral_direct_go_ex1(s)

function ex1_layer_validation_grid(n::Int)
    pts = collect(range(0.0, 1.0; length=max(401, min(4001, 4n))))
    for (x0, delta) in ((x1, delta1), (x2, delta2))
        c = fromX(x0)
        for scale in (-40.0, -10.0, -1.0, 0.0, 1.0, 10.0, 40.0)
            s = fromX(x0 + scale * delta)
            0.0 <= s <= 1.0 && push!(pts, s)
        end
        0.0 <= c <= 1.0 && push!(pts, c)
    end
    return sort!(unique!(pts))
end

function ex1_go_kerneldepth(N::Int)
    return parse(Int, get(ENV, "GO_EX1_KERNELDEPTH", "32"))
end

function ex1_go_project_p10(f, n::Int)
    return build_go_rhs_coefficients(
        f,
        n;
        alpha=1,
        beta=0,
        method=:fixed_gauss_jacobi,
        projection_tol=parse(Float64, get(ENV, "GO_EX1_PROJECTION_TOL", "1e-8")),
        oversampling=parse(Int, get(ENV, "GO_EX1_PROJECTION_OVERSAMPLING", "4")),
        validation_grid=ex1_layer_validation_grid(n),
    )
end

function ex1_go_values(c::AbstractVector, grid)
    return Fun(go_shifted_jacobi_space(1, 0), c).(grid)
end

function ex1_gutleb_olver_row(prob::BenchProblem, N::Int)
    if !EX1_GO_AVAILABLE[]
        return fail_row(prob, "gutleb_olver", N, "skipped",
                        "skipped: SparseVolterraNew could not be loaded; no GO solve was run")
    end

    n = prob.ncoeff_from_N(N)
    kerneldepth = ex1_go_kerneldepth(N)
    local rhs_coeff, rhs_info, exact_coeff, exact_info, coeffs, VK, A, V_phys, system, sol

    rhs_time = @elapsed begin
        rhs_coeff, rhs_info = ex1_go_project_p10(rhs_go, n)
    end
    exact_projection_time = @elapsed begin
        exact_coeff, exact_info = ex1_go_project_p10(u_go, n)
    end
    kernel_approx_time = @elapsed begin
        coeffs = build_gutleb_kernel_coefficients(K_go, kerneldepth; flip=true)
    end
    matrix_construct_time = @elapsed begin
        VK = build_gutleb_operator_from_coefficients(coeffs, n; flip=true)
    end
    system_assembly_time = @elapsed begin
        system = assemble_go_second_kind_vie_operator_from_raw(
            VK,
            n;
            sign=1.0,
            flip_policy="strict-band constructor flip=true; K_go is not pre-flipped",
        )
        A = system.A
        V_phys = system.V_phys
    end

    residual = A * exact_coeff - rhs_coeff
    residual_max = norm(residual, Inf)
    residual_rel = residual_max / max(norm(rhs_coeff, Inf), eps(Float64))

    grid = collect(range(0.0, 1.0; length=max(1001, 4n)))
    volterra_values = ex1_go_values(V_phys * exact_coeff, grid)
    volterra_truth = rhs_integral_direct_go_ex1.(grid)
    volterra_error = maximum(abs.(volterra_values .- volterra_truth))

    solve_time = @elapsed begin
        sol = A \ rhs_coeff
    end
    exact_error = maximum(abs.(ex1_go_values(sol, grid) .- u_go.(grid)))

    rhs_fields = rhs_projection_row_fields(rhs_info)
    rhs_ok = rhs_info.status == :success && exact_info.status == :success
    residual_ok = residual_rel <= EX1_GO_MAX_RESIDUAL_REL
    volterra_ok = volterra_error <= EX1_GO_MAX_VOLTERRA_ERROR
    error_ok = exact_error <= EX1_GO_MAX_ERROR
    correctness_ok = rhs_ok && residual_ok && volterra_ok && error_ok

    go_system_nnz = safe_nnz(A)
    go_density = safe_density(A)
    go_lb = safe_lower_bandwidth(A)
    go_ub = safe_upper_bandwidth(A)

    msg = correctness_ok ?
        "GO ex1 sparse/banded second-kind VIE solve passed residual and physical-grid validation" :
        "failed: GO ex1 second-kind validation did not pass; rhs_status=$(rhs_info.status), exact_projection_status=$(exact_info.status), residual_rel=$(residual_rel), volterra_error=$(volterra_error), exact_error=$(exact_error)"
    notes = join((
        msg,
        go_rhs_info_notes(rhs_info),
        "exact_projection_validation_error=$(exact_info.validation_error); exact_projection_tail_norm=$(exact_info.tail_norm); exact_projection_status=$(exact_info.status)",
        "raw_operator=$(system.info.raw_operator)",
        "flip_policy=$(system.info.flip_policy)",
        "final_chain=$(system.info.final_chain)",
        "final_system=$(system.info.final_system); raw_vk_added_to_identity=$(system.info.raw_vk_added_to_identity)",
        "solution_basis=$(system.info.solution_basis); rhs_basis=$(system.info.rhs_basis)",
        "dense_fallback=$(system.info.dense_fallback); tau_rows=$(system.info.tau_rows)",
        "exact-coefficient residual_max=$(residual_max); residual_rel=$(residual_rel); V_phys_physical_error=$(volterra_error)",
        "fast fixed-N Jacobi transform unavailable; direct Gauss-Jacobi projection capped at N <= $(GO_RHS_DIRECT_MAX_N)",
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
        rhs_time = rhs_time,
        kernel_approx_time = kernel_approx_time,
        matrix_construct_time = matrix_construct_time,
        system_assembly_time = system_assembly_time,
        solve_time = solve_time,
        factor_time = NaN,
        backsolve_time = NaN,
        total_time = rhs_time + exact_projection_time + kernel_approx_time +
            matrix_construct_time + system_assembly_time + solve_time,
        bandwidth = "$(go_lb)/$(go_ub)",
        kernel_size_y = kerneldepth,
        kernel_size_t = kerneldepth,
        rank = "missing",
        condition_number = NaN,
        reference_N = "missing",
        basis_family = "GO shifted Jacobi P^(1,0) solution/RHS/output basis; triangle Jacobi/Proriol kernel basis",
        solution_representation = "GO shifted Jacobi P^(1,0) coefficients",
        evaluation_grid = prob.evaluation_grid,
        integral_form = prob.go_integral_form,
        variable_transform = prob.variable_transform,
        go_storage_type = matrix_storage_type(A),
        go_operator_nnz = safe_nnz(VK),
        go_system_nnz = go_system_nnz,
        go_lower_bandwidth = go_lb,
        go_upper_bandwidth = go_ub,
        go_density = go_density,
        notes = notes,
    ), rhs_fields)
end

prob = BenchProblem(
    name = "ex1",
    Ns = evenly_spaced_integer_nodes(2, 40000, 100),
    ncoeff_from_N = N -> N,
    rhs = rhs_go,
    exact = u_go,
    local_coeff = s -> 1.0,
    volterra_terms = [(sign = 1.0, kernel = K_go, input_scale = 1.0)],
    error_type = "max_abs",
    primary_error = :exact,
    original_script = "examples/ex1_2ndVolterraEqn/ex1_aging_maxwell.jl",
    n_source = "evenly_spaced_integer_nodes(2, 40000, 100)",
    error_source = "err_inf physical max absolute error",
    integral_form = "int_{-1}^x, transformed to int_0^s",
    go_integral_form = "int_0^s",
    variable_transform = "x_original = 2s - 1; K_go(s,r)=2*K_original(2s-1,2r-1)",
    go_output_basis = "GO shifted Jacobi P^(1,0); RHS projection uses build_go_rhs_coefficients(alpha=1,beta=0)",
    notes = "aging Maxwell second-kind Volterra integral equation; RHS uses split Gauss quadrature matching the original narrow-layer construction; GO path uses A=I+S10_from_00*R*L00_from_10*VK with no tau rows",
    go_solver = ex1_gutleb_olver_row,
)

run_benchmark(prob)
