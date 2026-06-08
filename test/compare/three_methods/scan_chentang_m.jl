# Scan the Chen--Tang regularization order m for the transformed
# Jacobi spectral-collocation baseline.
#
# Expected behavior:
# - Larger m usually makes the transformed unknown smoother, so the
#   transformed Linf error and discrete consistency can improve.
# - Recovering y(t) = y0 + t^(-a) U(2t-1), a = mu + m - 1, can strongly
#   amplify small transformed errors near t=0.
# - Therefore larger m is not automatically better; this scan searches for
#   a useful m range for the current manufactured solution.
#
# Run from the project root:
#   julia --project=. test/compare/three_methods/scan_chentang_m.jl
#   julia --project=. test/compare/three_methods/plot_scan_chentang_m.jl
#
# Quick PowerShell run:
#   $env:M_LIST="2,3,4"; $env:N_LIST="16,32,64"; julia --project=. test/compare/three_methods/scan_chentang_m.jl

using Printf

const THIS_DIR = @__DIR__
include(joinpath(THIS_DIR, "compare_three_weak_methods.jl"))

const M_VALUES = parse_env_int_list("M_LIST", collect(2:10))
const SCAN_N_VALUES = parse_env_int_list("N_LIST", DEFAULT_N_LIST)
const ENDPOINT_CUTOFF = parse(Float64, get(ENV, "ENDPOINT_CUTOFF", "1e-8"))

const SCAN_RESULTS_CSV = joinpath(DATA_DIR, "chentang_m_scan_results.csv")
const SCAN_SUMMARY_CSV = joinpath(DATA_DIR, "chentang_m_scan_summary.csv")

const SCAN_HEADERS = [
    "m", "a", "N", "dof", "build_time", "solve_time", "total_time",
    "linf_error", "linf_error_full_grid", "linf_error_excluding_endpoint_layer",
    "l2_grid_error", "transformed_linf_error", "residual_linf", "consistency",
    "max_abs_U_exact_nodes", "amplification_indicator", "rhs_stability", "notes",
]

const SUMMARY_HEADERS = [
    "m", "a", "best_N_by_original_error", "best_original_error",
    "best_original_error_excluding_endpoint_layer", "best_transformed_error",
    "best_total_time", "best_consistency", "best_amplification_indicator", "notes",
]

function scan_failure_row(m, N, note, rhs_stability)
    a = MU + Float64(m) - 1.0
    return (
        m = m,
        a = a,
        N = N,
        dof = N + 1,
        build_time = NaN,
        solve_time = NaN,
        total_time = NaN,
        linf_error = NaN,
        linf_error_full_grid = NaN,
        linf_error_excluding_endpoint_layer = NaN,
        l2_grid_error = NaN,
        transformed_linf_error = NaN,
        residual_linf = NaN,
        consistency = NaN,
        max_abs_U_exact_nodes = NaN,
        amplification_indicator = NaN,
        rhs_stability = rhs_stability,
        notes = note,
    )
end

function scan_row_from_result(r)
    return (
        m = r.m,
        a = r.a,
        N = r.N,
        dof = r.dof,
        build_time = r.build_time,
        solve_time = r.solve_time,
        total_time = r.total_time,
        linf_error = r.linf_error,
        linf_error_full_grid = r.linf_error_full_grid,
        linf_error_excluding_endpoint_layer = r.linf_error_excluding_endpoint_layer,
        l2_grid_error = r.l2_grid_error,
        transformed_linf_error = r.transformed_linf_error,
        residual_linf = r.residual_linf,
        consistency = r.consistency,
        max_abs_U_exact_nodes = r.max_abs_U_exact_nodes,
        amplification_indicator = r.amplification_indicator,
        rhs_stability = r.rhs_stability,
        notes = r.notes,
    )
end

function build_summary(rows)
    summary = NamedTuple[]
    for m in sort(unique(r.m for r in rows))
        mrows = [r for r in rows if r.m == m && isfinite(r.linf_error_full_grid)]
        if isempty(mrows)
            push!(summary, (
                m = m,
                a = MU + Float64(m) - 1.0,
                best_N_by_original_error = -1,
                best_original_error = NaN,
                best_original_error_excluding_endpoint_layer = NaN,
                best_transformed_error = NaN,
                best_total_time = NaN,
                best_consistency = NaN,
                best_amplification_indicator = NaN,
                notes = "no finite rows",
            ))
            continue
        end
        best_original = mrows[argmin([r.linf_error_full_grid for r in mrows])]
        best_transformed = mrows[argmin([r.transformed_linf_error for r in mrows])]
        push!(summary, (
            m = m,
            a = best_original.a,
            best_N_by_original_error = best_original.N,
            best_original_error = best_original.linf_error_full_grid,
            best_original_error_excluding_endpoint_layer = best_original.linf_error_excluding_endpoint_layer,
            best_transformed_error = best_transformed.transformed_linf_error,
            best_total_time = best_original.total_time,
            best_consistency = best_original.consistency,
            best_amplification_indicator = best_original.amplification_indicator,
            notes = @sprintf("best transformed N=%d; original-row transformed=%.3e",
                best_transformed.N, best_original.transformed_linf_error),
        ))
    end
    return summary
end

function main()
    println("============================================================")
    println("Chen--Tang transformed collocation regularization-order scan")
    @printf("mu = %.6f, lambda = %.6f\n", MU, LAMBDA)
    println("M_LIST = ", M_VALUES)
    println("N_LIST = ", SCAN_N_VALUES)
    @printf("RHS_QUAD_ORDER = %d, CHECK_GRID_SIZE = %d, endpoint cutoff = %.1e\n",
        RHS_QUAD_ORDER, CHECK_GRID_SIZE, ENDPOINT_CUTOFF)
    println("Expected: transformed errors may improve with m, but original errors can be amplified near t=0.")
    println("============================================================")

    rhs_stab = check_rhs_stability()
    maxN = maximum(SCAN_N_VALUES)
    rhs_coeffs, rhs_info = rhs_coefficients(maxN)
    @printf("Common RHS generation: point_time=%.6e, projection_time=%.6e\n",
        rhs_info.rhs_time, rhs_info.projection_time)

    tgrid = collect(range(0.0, 1.0; length = CHECK_GRID_SIZE))
    exact_vals = u_exact_t.(tgrid)
    rgrid = collect(range(0.0, 1.0; length = RESIDUAL_GRID_SIZE))

    rows = NamedTuple[]
    try
        run_chentang_transformed(first(SCAN_N_VALUES);
            m = first(M_VALUES), rhs_coeffs = rhs_coeffs, tgrid = [0.0, 0.5, 1.0],
            exact_vals = u_exact_t.([0.0, 0.5, 1.0]), rgrid = [0.0, 0.5, 1.0],
            rhs_stability = rhs_stab.diff, endpoint_cutoff = ENDPOINT_CUTOFF)
    catch
    end

    for m in M_VALUES
        for N in SCAN_N_VALUES
            try
                r = run_chentang_transformed(N;
                    m = m,
                    rhs_coeffs = rhs_coeffs,
                    tgrid = tgrid,
                    exact_vals = exact_vals,
                    rgrid = rgrid,
                    rhs_stability = rhs_stab.diff,
                    endpoint_cutoff = ENDPOINT_CUTOFF,
                )
                push!(rows, scan_row_from_result(r))
                @printf("[m=%d, N=%d] original=%.6e, original_excl=%.6e, transformed=%.6e, consistency=%.6e, residual=%.6e, amplification=%.6e, time=%.6e\n",
                    m, N, r.linf_error_full_grid, r.linf_error_excluding_endpoint_layer,
                    r.transformed_linf_error, r.consistency, r.residual_linf,
                    r.amplification_indicator, r.total_time)
            catch err
                @warn "Chen--Tang m scan failed." m N exception = (err, catch_backtrace())
                push!(rows, scan_failure_row(m, N, "failed: $(typeof(err))", rhs_stab.diff))
            end
        end
    end

    summary = build_summary(rows)
    write_csv(SCAN_RESULTS_CSV, SCAN_HEADERS, rows)
    write_csv(SCAN_SUMMARY_CSV, SUMMARY_HEADERS, summary)

    println()
    println("Summary by m:")
    for s in summary
        @printf("m=%d: best original error = %.6e, best transformed error = %.6e, best N = %d, amplification = %.6e\n",
            s.m, s.best_original_error, s.best_transformed_error,
            s.best_N_by_original_error, s.best_amplification_indicator)
    end
    println("Saved CSV:")
    println("  ", SCAN_RESULTS_CSV)
    println("  ", SCAN_SUMMARY_CSV)

    try
        include(joinpath(THIS_DIR, "plot_scan_chentang_m.jl"))
        Base.invokelatest(plot_scan_chentang_m)
    catch err
        @warn "Scan plotting failed. Run plot_scan_chentang_m.jl separately." exception = (err, catch_backtrace())
    end

    return (rows = rows, summary = summary)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
