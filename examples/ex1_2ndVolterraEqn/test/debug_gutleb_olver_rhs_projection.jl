include(joinpath(@__DIR__, "ex1_2nd_volterra_common.jl"))

const DEBUG_GO_NS = ex1_parse_int_list_env("EX1_GO_DEBUG_NS", [8000, 16000, 30000])
const DEBUG_GO_OUTPUT_CSV = joinpath(@__DIR__, "debug_gutleb_olver_rhs_projection.csv")
const DEBUG_GO_KERNELDEPTH = parse(Int, get(ENV, "EX1_GO_KERNELDEPTH", "32"))

const DEBUG_COLUMNS = [
    :N,
    :rhs_method,
    :u_method,
    :max_abs_rhs_projection_error,
    :rel_rhs_projection_error,
    :max_abs_u_projection_error,
    :rel_u_projection_error,
    :max_abs_error_consistent_rhs,
    :max_abs_error_physical_rhs,
    :consistent_coeff_inferr,
    :consistent_eval_delta_inferr,
    :physical_coeff_delta_inferr,
    :physical_eval_delta_inferr,
    :discrete_rhs_mismatch_rel,
    :operator_residual_consistent_rel,
    :operator_residual_physical_rel,
    :build_time,
    :diagnostic_status,
    :notes,
]

function debug_write_csv(path::AbstractString, rows)
    open(path, "w") do io
        println(io, join(string.(DEBUG_COLUMNS), ","))
        for row in rows
            vals = [ex1_csv_escape(getproperty(row, col)) for col in DEBUG_COLUMNS]
            println(io, join(vals, ","))
        end
    end
    return path
end

function debug_failure_row(N::Int, err)
    return (
        N=N,
        rhs_method="",
        u_method="",
        max_abs_rhs_projection_error=NaN,
        rel_rhs_projection_error=NaN,
        max_abs_u_projection_error=NaN,
        rel_u_projection_error=NaN,
        max_abs_error_consistent_rhs=NaN,
        max_abs_error_physical_rhs=NaN,
        consistent_coeff_inferr=NaN,
        consistent_eval_delta_inferr=NaN,
        physical_coeff_delta_inferr=NaN,
        physical_eval_delta_inferr=NaN,
        discrete_rhs_mismatch_rel=NaN,
        operator_residual_consistent_rel=NaN,
        operator_residual_physical_rel=NaN,
        build_time=NaN,
        diagnostic_status="failed",
        notes=sprint(showerror, err),
    )
end

function debug_build_go_system(N::Int; kerneldepth::Int=DEBUG_GO_KERNELDEPTH)
    local plan, coeffs, Vraw, A
    t = @elapsed begin
        plan = precompute_triangular_kernel_plan(kerneldepth, true)
        coeffs = triangular_kernel_coeffs_from_plan(plan, ex1_kernel_go)
        Vraw = triVolterraFullKernelOpP01_strictband_from_coeffs(coeffs, N, true)
        A = build_secondkind_system_strictband(Vraw, N; sign=1.0)
    end
    return A, coeffs, t
end

function debug_projection_values(coeffs, f, truth_values, scale, s_test)
    vals = ex1_go_values(coeffs, s_test)
    max_abs = maximum(abs.(vals .- truth_values))
    rel = max_abs / scale
    return vals, max_abs, rel
end

function debug_check_cheb2jac_scaling(; N::Int=32)
    grid = collect(range(0.0, 1.0; length=1001))
    tests = (
        ("1", s -> 1.0),
        ("s", s -> s),
        ("s^2", s -> s * s),
    )
    parts = String[]
    maxerr = 0.0
    for (name, f) in tests
        coeffs, info = ex1_project_go_rhs_cheb2jac_normalized(f, N)
        vals = ex1_go_values(coeffs, grid)
        truth = f.(grid)
        err = maximum(abs.(vals .- truth))
        maxerr = max(maxerr, err)
        push!(parts, "$name=$( @sprintf("%.3e", err) )")
        println("  polynomial ", name, " cheb2jac scaling error = ", @sprintf("%.3e", err),
            " method=", info.method)
    end
    return maxerr, join(parts, "; ")
end

function debug_one_N(N::Int, rhs_truth, rhs_scale, u_truth, u_scale, s_test)
    local b_go, rhs_info, c_exact_go, u_info, A, coeffs
    local rhs_vals, u_proj_vals
    t_build = @elapsed begin
        b_go, rhs_info = ex1_project_go_rhs(ex1_rhs_go, N)
        c_exact_go, u_info = ex1_project_go_rhs(ex1_u_go, N)
        A, coeffs, t_A = debug_build_go_system(N)
    end

    rhs_vals, rhs_abs_err, rhs_rel_err =
        debug_projection_values(b_go, ex1_rhs_go, rhs_truth, rhs_scale, s_test)
    u_proj_vals, u_abs_err, u_rel_err =
        debug_projection_values(c_exact_go, ex1_u_go, u_truth, u_scale, s_test)

    b_consistent = A * c_exact_go
    c_num_consistent = A \ b_consistent
    c_num_rhs = A \ b_go

    consistent_vals = ex1_go_values(c_num_consistent, s_test)
    physical_vals = ex1_go_values(c_num_rhs, s_test)
    consistent_delta_vals = ex1_go_values(c_num_consistent .- c_exact_go, s_test)
    physical_delta_vals = ex1_go_values(c_num_rhs .- c_exact_go, s_test)

    max_abs_error_consistent_rhs = maximum(abs.(consistent_vals .- u_truth))
    max_abs_error_physical_rhs = maximum(abs.(physical_vals .- u_truth))
    consistent_coeff_inferr = norm(c_num_consistent .- c_exact_go, Inf) /
        max(norm(c_exact_go, Inf), eps(Float64))
    physical_coeff_delta_inferr = norm(c_num_rhs .- c_exact_go, Inf) /
        max(norm(c_exact_go, Inf), eps(Float64))
    consistent_eval_delta_inferr = maximum(abs.(consistent_delta_vals))
    physical_eval_delta_inferr = maximum(abs.(physical_delta_vals))
    discrete_rhs_mismatch_rel = norm(b_consistent .- b_go, Inf) /
        max(norm(b_go, Inf), eps(Float64))
    operator_residual_consistent_rel = norm(A * c_num_consistent .- b_consistent, Inf) /
        max(norm(b_consistent, Inf), eps(Float64))
    operator_residual_physical_rel = norm(A * c_num_rhs .- b_go, Inf) /
        max(norm(b_go, Inf), eps(Float64))

    status =
        !all(isfinite, b_go) || !all(isfinite, c_exact_go) ? "nonfinite_projection" :
        consistent_coeff_inferr <= 1e-12 &&
            consistent_eval_delta_inferr <= 1e-12 &&
            max_abs_error_consistent_rhs > 1e-12 ? "projection_limited_operator_solver_ok" :
        consistent_coeff_inferr <= 1e-12 &&
            consistent_eval_delta_inferr <= 1e-12 ? "operator_solver_ok" :
        "operator_or_evaluator_limited"

    notes = join((
        "kerneldepth=$DEBUG_GO_KERNELDEPTH",
        "kernel_coeff_size=$(coeffs.kerneldepth)",
        "rhs_status=$(rhs_info.status)",
        "u_status=$(u_info.status)",
        "rhs_validation_error=$(rhs_info.validation_error)",
        "u_validation_error=$(u_info.validation_error)",
        "consistent_rhs_error is bounded below by the projected exact-solution evaluation error",
    ), "; ")

    return (
        N=N,
        rhs_method=string(rhs_info.method),
        u_method=string(u_info.method),
        max_abs_rhs_projection_error=rhs_abs_err,
        rel_rhs_projection_error=rhs_rel_err,
        max_abs_u_projection_error=u_abs_err,
        rel_u_projection_error=u_rel_err,
        max_abs_error_consistent_rhs=max_abs_error_consistent_rhs,
        max_abs_error_physical_rhs=max_abs_error_physical_rhs,
        consistent_coeff_inferr=consistent_coeff_inferr,
        consistent_eval_delta_inferr=consistent_eval_delta_inferr,
        physical_coeff_delta_inferr=physical_coeff_delta_inferr,
        physical_eval_delta_inferr=physical_eval_delta_inferr,
        discrete_rhs_mismatch_rel=discrete_rhs_mismatch_rel,
        operator_residual_consistent_rel=operator_residual_consistent_rel,
        operator_residual_physical_rel=operator_residual_physical_rel,
        build_time=t_build,
        diagnostic_status=status,
        notes=notes,
    )
end

function run_debug_gutleb_olver_rhs_projection()
    EX1_HAS_GO[] || error("GutlebOlver is unavailable")
    EX1_HAS_FASTTRANSFORMS[] || error("FastTransforms is unavailable")

    println("Gutleb--Olver RHS projection diagnostics for ex1 second-kind Volterra")
    println("N values: ", DEBUG_GO_NS)
    println("output: ", DEBUG_GO_OUTPUT_CSV)
    println("Polynomial scaling checks for normalized cheb2jac -> GO Jacobi(0,1,0..1):")
    poly_maxerr, poly_notes = debug_check_cheb2jac_scaling()

    println("Precomputing high-density RHS truth on the ex1 test grid...")
    local rhs_truth
    t_rhs_truth = @elapsed begin
        rhs_truth = ex1_rhs_go.(EX1_STEST)
    end
    rhs_scale = maximum(abs.(rhs_truth))
    println("  RHS truth points = ", length(EX1_STEST), ", time = ", @sprintf("%.3f", t_rhs_truth), " s")

    rows = NamedTuple[]
    for N in DEBUG_GO_NS
        println()
        @printf("N = %d\n", N)
        row = try
            GC.gc()
            r = debug_one_N(N, rhs_truth, rhs_scale, EX1_UTEST_EXACT, EX1_UTEST_SCALE, EX1_STEST)
            r = merge(r, (
                notes=r.notes * "; poly_scaling_max_error=$( @sprintf("%.3e", poly_maxerr) ); " * poly_notes,
            ))
            @printf("  rhs_proj=%.3e, u_proj=%.3e, consistent=%.3e, physical=%.3e, coeff_delta=%.3e, eval_delta=%.3e, status=%s\n",
                r.max_abs_rhs_projection_error,
                r.max_abs_u_projection_error,
                r.max_abs_error_consistent_rhs,
                r.max_abs_error_physical_rhs,
                r.consistent_coeff_inferr,
                r.consistent_eval_delta_inferr,
                r.diagnostic_status)
            r
        catch err
            @warn "Diagnostic failed for N=$N" exception=(err, catch_backtrace())
            debug_failure_row(N, err)
        end
        push!(rows, row)
        debug_write_csv(DEBUG_GO_OUTPUT_CSV, rows)
    end

    println()
    println("Wrote ", DEBUG_GO_OUTPUT_CSV)
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_debug_gutleb_olver_rhs_projection()
end
