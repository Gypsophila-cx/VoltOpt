# Optional scan of the fractional power lambda_frac in the Muntz
# collocation baseline. For EXAMPLE=oscillatory_sqrt, lambda_frac=0.5
# matches the endpoint sqrt(t) structure but no longer exactly represents
# the oscillatory factor.
#
# Run:
#   julia --project=. test/compare/three_methods/scan_muntz_lambda.jl
#   julia --project=. test/compare/three_methods/plot_scan_muntz_lambda.jl
#
# PowerShell quick run:
#   $env:MUNTZ_LAMBDA_LIST="0.333333333333,0.5,1.0"; $env:N_LIST="16,32,64"; julia --project=. test/compare/three_methods/scan_muntz_lambda.jl

using Printf

const THIS_DIR = @__DIR__
include(joinpath(THIS_DIR, "compare_three_weak_methods.jl"))

function parse_env_float_list(env_name::AbstractString, default_values)
    raw = get(ENV, env_name, nothing)
    raw === nothing && return collect(Float64, default_values)
    vals = Float64[]
    for piece in split(raw, ",")
        s = strip(piece)
        isempty(s) && continue
        v = tryparse(Float64, s)
        v === nothing && error("Invalid floating-point value in $(env_name): $(s)")
        push!(vals, v)
    end
    isempty(vals) && error("$(env_name) did not contain any values.")
    return vals
end

const LAMBDA_VALUES = parse_env_float_list("MUNTZ_LAMBDA_LIST",
    [0.25, 0.333333333333, 0.4, 0.5, 0.666666666667, 0.75, 1.0])
const SCAN_N_VALUES = parse_env_int_list("N_LIST", DEFAULT_N_LIST)
lambda_scan_stem(base::AbstractString) = EXAMPLE == "low_order_sqrt_poly" ? base : base * "_" * EXAMPLE
const LAMBDA_RESULTS_CSV = joinpath(DATA_DIR, lambda_scan_stem("muntz_lambda_scan") * "_results.csv")
const LAMBDA_SUMMARY_CSV = joinpath(DATA_DIR, lambda_scan_stem("muntz_lambda_scan") * "_summary.csv")

const LAMBDA_SCAN_HEADERS = [
    "muntz_lambda", "N", "dof", "build_time", "solve_time", "total_time",
    "linf_error", "l2_grid_error", "residual_linf", "consistency_linf",
    "rhs_stability", "muntz_alpha", "muntz_beta", "notes",
]

const LAMBDA_SUMMARY_HEADERS = [
    "muntz_lambda", "best_N", "best_linf_error", "best_total_time",
    "best_consistency_linf", "notes",
]

function lambda_scan_row(r)
    return (
        muntz_lambda = r.muntz_lambda,
        N = r.N,
        dof = r.dof,
        build_time = r.build_time,
        solve_time = r.solve_time,
        total_time = r.total_time,
        linf_error = r.linf_error,
        l2_grid_error = r.l2_grid_error,
        residual_linf = r.residual_linf,
        consistency_linf = r.consistency_linf,
        rhs_stability = r.rhs_stability,
        muntz_alpha = r.muntz_alpha,
        muntz_beta = r.muntz_beta,
        notes = r.notes,
    )
end

function lambda_failure_row(lambda_frac, N, rhs_stability, note)
    return (
        muntz_lambda = lambda_frac,
        N = N,
        dof = N + 1,
        build_time = NaN,
        solve_time = NaN,
        total_time = NaN,
        linf_error = NaN,
        l2_grid_error = NaN,
        residual_linf = NaN,
        consistency_linf = NaN,
        rhs_stability = rhs_stability,
        muntz_alpha = MUNTZ_ALPHA,
        muntz_beta = MUNTZ_BETA,
        notes = note,
    )
end

function build_lambda_summary(rows)
    out = NamedTuple[]
    for lam in sort(unique(r.muntz_lambda for r in rows))
        lrows = [r for r in rows if r.muntz_lambda == lam && isfinite(r.linf_error)]
        if isempty(lrows)
            push!(out, (
                muntz_lambda = lam,
                best_N = -1,
                best_linf_error = NaN,
                best_total_time = NaN,
                best_consistency_linf = NaN,
                notes = "no finite rows",
            ))
            continue
        end
        best = lrows[argmin([r.linf_error for r in lrows])]
        push!(out, (
            muntz_lambda = lam,
            best_N = best.N,
            best_linf_error = best.linf_error,
            best_total_time = best.total_time,
            best_consistency_linf = best.consistency_linf,
            notes = best.notes,
        ))
    end
    return out
end

function main()
    println("============================================================")
    println("Muntz collocation lambda_frac scan")
    println("EXAMPLE = ", EXAMPLE)
    @printf("mu = %.6f, lambda_coup = %.6f, alpha = %.3f, beta = %.3f\n",
        MU, LAMBDA, MUNTZ_ALPHA, MUNTZ_BETA)
    println("MUNTZ_LAMBDA_LIST = ", LAMBDA_VALUES)
    println("N_LIST = ", SCAN_N_VALUES)
    println("Expected: lambda_frac=0.5 should match the sqrt(t) endpoint term; lambda=1 should be worse.")
    println("============================================================")

    rhs_stab = check_rhs_stability()
    rhs_coeffs, rhs_info = rhs_coefficients(maximum(SCAN_N_VALUES))
    @printf("Common RHS generation: point_time=%.6e, projection_time=%.6e\n",
        rhs_info.rhs_time, rhs_info.projection_time)

    tgrid = collect(range(0.0, 1.0; length = CHECK_GRID_SIZE))
    exact_vals = u_exact_t.(tgrid)

    rows = NamedTuple[]
    try
        run_muntz_collocation(first(SCAN_N_VALUES);
            lambda_frac = first(LAMBDA_VALUES), rhs_coeffs = rhs_coeffs,
            tgrid = [0.0, 0.5, 1.0], exact_vals = u_exact_t.([0.0, 0.5, 1.0]),
            rhs_stability = rhs_stab.diff)
    catch
    end

    for lam in LAMBDA_VALUES
        for N in SCAN_N_VALUES
            try
                r = run_muntz_collocation(N;
                    lambda_frac = lam,
                    rhs_coeffs = rhs_coeffs,
                    tgrid = tgrid,
                    exact_vals = exact_vals,
                    rhs_stability = rhs_stab.diff)
                push!(rows, lambda_scan_row(r))
                @printf("[lambda=%.12g, N=%d] error=%.6e, consistency=%.6e, residual=%.6e, time=%.6e\n",
                    lam, N, r.linf_error, r.consistency_linf, r.residual_linf, r.total_time)
            catch err
                @warn "Muntz lambda scan failed." lam N exception = (err, catch_backtrace())
                push!(rows, lambda_failure_row(lam, N, rhs_stab.diff, "failed: $(typeof(err))"))
            end
        end
    end

    summary = build_lambda_summary(rows)
    write_csv(LAMBDA_RESULTS_CSV, LAMBDA_SCAN_HEADERS, rows)
    write_csv(LAMBDA_SUMMARY_CSV, LAMBDA_SUMMARY_HEADERS, summary)

    println()
    println("Summary by lambda_frac:")
    for s in summary
        @printf("lambda=%.12g: best error = %.6e, best N = %d, consistency = %.6e\n",
            s.muntz_lambda, s.best_linf_error, s.best_N, s.best_consistency_linf)
    end

    try
        include(joinpath(THIS_DIR, "plot_scan_muntz_lambda.jl"))
        Base.invokelatest(plot_scan_muntz_lambda)
    catch err
        @warn "Muntz lambda scan plotting failed. Run plot_scan_muntz_lambda.jl separately." exception = (err, catch_backtrace())
    end
    return (rows = rows, summary = summary)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
