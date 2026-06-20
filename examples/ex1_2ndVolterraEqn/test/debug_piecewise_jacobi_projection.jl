include(joinpath(@__DIR__, "ex1_2nd_volterra_common.jl"))

const PIECEWISE_DEBUG_NS = ex1_parse_int_list_env("EX1_PIECEWISE_DEBUG_NS", [8000, 16000, 30000])
const PIECEWISE_DEBUG_CSV = joinpath(@__DIR__, "debug_piecewise_jacobi_projection.csv")
const PIECEWISE_DEBUG_KERNELDEPTH = parse(Int, get(ENV, "EX1_GO_KERNELDEPTH", "32"))
const PIECEWISE_BIGFLOAT_PROBE_CACHE = Ref{Union{Nothing,String}}(nothing)

const PIECEWISE_COLUMNS = [
    :N,
    :quad_order,
    :endpoint_correction,
    :cheb2jac_u_projection_error,
    :piecewise_u_projection_error,
    :cheb2jac_rhs_projection_error,
    :piecewise_rhs_projection_error,
    :go_error_piecewise_rhs,
    :go_error_cheb2jac_rhs,
    :piecewise_projection_time,
    :cheb2jac_projection_time,
    :solve_time,
    :total_time,
    :quadrature_order_sweep,
    :bigfloat_probe,
    :exact_projection_vs_physical_note,
    :status,
    :notes,
]

function piecewise_debug_quad_order(N::Int)
    raw = get(ENV, "EX1_PIECEWISE_DEBUG_QUAD_ORDER", "")
    !isempty(strip(raw)) && return parse(Int, raw)
    return N <= 8000 ? 4096 : N <= 16000 ? 16384 : 32768
end

function piecewise_write_csv(path::AbstractString, rows)
    open(path, "w") do io
        println(io, join(string.(PIECEWISE_COLUMNS), ","))
        for row in rows
            println(io, join((ex1_csv_escape(getproperty(row, c)) for c in PIECEWISE_COLUMNS), ","))
        end
    end
    return path
end

function piecewise_failure_row(N::Int, err)
    return (
        N=N,
        quad_order=missing,
        endpoint_correction="",
        cheb2jac_u_projection_error=NaN,
        piecewise_u_projection_error=NaN,
        cheb2jac_rhs_projection_error=NaN,
        piecewise_rhs_projection_error=NaN,
        go_error_piecewise_rhs=NaN,
        go_error_cheb2jac_rhs=NaN,
        piecewise_projection_time=NaN,
        cheb2jac_projection_time=NaN,
        solve_time=NaN,
        total_time=NaN,
        quadrature_order_sweep="",
        bigfloat_probe="",
        exact_projection_vs_physical_note="",
        status="failed",
        notes=sprint(showerror, err),
    )
end

function piecewise_build_go_system(N::Int)
    local plan, coeffs, Vraw, A
    t = @elapsed begin
        plan = precompute_triangular_kernel_plan(PIECEWISE_DEBUG_KERNELDEPTH, true)
        coeffs = triangular_kernel_coeffs_from_plan(plan, ex1_kernel_go)
        Vraw = triVolterraFullKernelOpP01_strictband_from_coeffs(coeffs, N, true)
        A = build_secondkind_system_strictband(Vraw, N; sign=1.0)
    end
    return A, t
end

function piecewise_solve_error(A, b, s_test, u_truth)
    local c
    t = @elapsed begin
        c = A \ b
    end
    vals = ex1_go_values(c, s_test)
    return maximum(abs.(vals .- u_truth)), t
end

function piecewise_order_sweep_u(N::Int)
    if lowercase(get(ENV, "EX1_PIECEWISE_RUN_SWEEP", "false")) != "true"
        return "not_run_by_default; set EX1_PIECEWISE_RUN_SWEEP=true to recompute per-N sweep"
    end
    orders = N <= 8000 ? (1024, 2048, 4096) :
             N <= 16000 ? (4096, 8192, 16384) :
             (8192, 16384, 32768)
    parts = String[]
    for q in orders
        try
            c, info = ex1_project_go_rhs_piecewise_direct(
                ex1_u_go,
                N;
                quad_order=q,
                endpoint_correction=:both,
            )
            push!(parts, "$q:$( @sprintf("%.3e", info.validation_error) )")
        catch err
            push!(parts, "$q:failed($(sprint(showerror, err)))")
        end
    end
    return join(parts, "; ")
end

function piecewise_bigfloat_probe()
    PIECEWISE_BIGFLOAT_PROBE_CACHE[] !== nothing && return PIECEWISE_BIGFLOAT_PROBE_CACHE[]
    probe_N = parse(Int, get(ENV, "EX1_PIECEWISE_BIGFLOAT_PROBE_N", "128"))
    probe_q = parse(Int, get(ENV, "EX1_PIECEWISE_BIGFLOAT_PROBE_Q", "128"))
    result = try
        c64, i64 = ex1_project_go_rhs_piecewise_direct(
            ex1_u_go,
            probe_N;
            quad_order=probe_q,
            precision=:float64,
            endpoint_correction=:both,
        )
        cbig, ibig = ex1_project_go_rhs_piecewise_direct(
            ex1_u_go,
            probe_N;
            quad_order=probe_q,
            precision=:bigfloat_accumulation,
            endpoint_correction=:both,
        )
        "N=$probe_N q=$probe_q float64=$( @sprintf("%.3e", i64.validation_error) ) bigfloat=$( @sprintf("%.3e", ibig.validation_error) ) coeff_delta=$( @sprintf("%.3e", norm(c64 - cbig, Inf)) )"
    catch err
        "failed: $(sprint(showerror, err))"
    end
    PIECEWISE_BIGFLOAT_PROBE_CACHE[] = result
    return result
end

function piecewise_debug_one_N(N::Int, s_test, u_truth, rhs_truth, u_scale, rhs_scale)
    q = piecewise_debug_quad_order(N)
    endpoint_correction = Symbol(get(ENV, "EX1_PIECEWISE_ENDPOINT_CORRECTION", "both"))

    local c_u_cheb, i_u_cheb, b_cheb, i_rhs_cheb
    t_cheb = @elapsed begin
        c_u_cheb, i_u_cheb = ex1_project_go_rhs_cheb2jac_normalized(ex1_u_go, N)
        b_cheb, i_rhs_cheb = ex1_project_go_rhs_cheb2jac_normalized(ex1_rhs_go, N)
    end

    local c_u_piece, i_u_piece, b_piece, i_rhs_piece
    t_piece = @elapsed begin
        c_u_piece, i_u_piece = ex1_project_go_rhs_piecewise_direct(
            ex1_u_go,
            N;
            quad_order=q,
            endpoint_correction=endpoint_correction,
        )
        b_piece, i_rhs_piece = ex1_project_go_rhs_piecewise_direct(
            ex1_rhs_go,
            N;
            quad_order=q,
            endpoint_correction=endpoint_correction,
        )
    end

    A, t_A = piecewise_build_go_system(N)
    go_piece_err, t_solve_piece = piecewise_solve_error(A, b_piece, s_test, u_truth)
    go_cheb_err, t_solve_cheb = piecewise_solve_error(A, b_cheb, s_test, u_truth)

    u_piece_vals = ex1_go_values(c_u_piece, s_test)
    u_cheb_vals = ex1_go_values(c_u_cheb, s_test)
    rhs_piece_vals = ex1_go_values(b_piece, s_test)
    rhs_cheb_vals = ex1_go_values(b_cheb, s_test)

    u_piece_err = maximum(abs.(u_piece_vals .- u_truth))
    u_cheb_err = maximum(abs.(u_cheb_vals .- u_truth))
    rhs_piece_err = maximum(abs.(rhs_piece_vals .- rhs_truth))
    rhs_cheb_err = maximum(abs.(rhs_cheb_vals .- rhs_truth))

    exact_projection_note =
        "piecewise_u=$( @sprintf("%.3e", u_piece_err) ), piecewise_solve=$( @sprintf("%.3e", go_piece_err) ); " *
        "cheb2jac_u=$( @sprintf("%.3e", u_cheb_err) ), cheb2jac_solve=$( @sprintf("%.3e", go_cheb_err) )"

    improved = go_piece_err < go_cheb_err
    return (
        N=N,
        quad_order=q,
        endpoint_correction=string(endpoint_correction),
        cheb2jac_u_projection_error=u_cheb_err,
        piecewise_u_projection_error=u_piece_err,
        cheb2jac_rhs_projection_error=rhs_cheb_err,
        piecewise_rhs_projection_error=rhs_piece_err,
        go_error_piecewise_rhs=go_piece_err,
        go_error_cheb2jac_rhs=go_cheb_err,
        piecewise_projection_time=t_piece,
        cheb2jac_projection_time=t_cheb,
        solve_time=t_solve_piece + t_solve_cheb,
        total_time=t_piece + t_cheb + t_A + t_solve_piece + t_solve_cheb,
        quadrature_order_sweep=piecewise_order_sweep_u(N),
        bigfloat_probe=piecewise_bigfloat_probe(),
        exact_projection_vs_physical_note=exact_projection_note,
        status=improved ? "piecewise_improved" : "cheb2jac_still_better",
        notes="kerneldepth=$PIECEWISE_DEBUG_KERNELDEPTH; rhs_piece_status=$(i_rhs_piece.status); u_piece_status=$(i_u_piece.status); rhs_cheb_status=$(i_rhs_cheb.status); u_cheb_status=$(i_u_cheb.status)",
    )
end

function run_piecewise_debug()
    println("Piecewise direct Jacobi projection debug for ex1")
    println("N values: ", PIECEWISE_DEBUG_NS)
    println("output: ", PIECEWISE_DEBUG_CSV)
    println("Precomputing RHS truth on ex1 test grid...")
    local rhs_truth
    t_truth = @elapsed rhs_truth = ex1_rhs_go.(EX1_STEST)
    println("  RHS truth points = ", length(EX1_STEST), ", time = ", @sprintf("%.3f", t_truth), " s")

    rows = NamedTuple[]
    for N in PIECEWISE_DEBUG_NS
        println()
        @printf("N = %d, quad_order = %d\n", N, piecewise_debug_quad_order(N))
        row = try
            GC.gc()
            r = piecewise_debug_one_N(N, EX1_STEST, EX1_UTEST_EXACT, rhs_truth, EX1_UTEST_SCALE, maximum(abs.(rhs_truth)))
            @printf("  u cheb=%.3e piece=%.3e | rhs cheb=%.3e piece=%.3e | solve cheb=%.3e piece=%.3e | status=%s\n",
                r.cheb2jac_u_projection_error,
                r.piecewise_u_projection_error,
                r.cheb2jac_rhs_projection_error,
                r.piecewise_rhs_projection_error,
                r.go_error_cheb2jac_rhs,
                r.go_error_piecewise_rhs,
                r.status)
            r
        catch err
            @warn "Piecewise debug failed for N=$N" exception=(err, catch_backtrace())
            piecewise_failure_row(N, err)
        end
        push!(rows, row)
        piecewise_write_csv(PIECEWISE_DEBUG_CSV, rows)
    end

    println()
    println("Wrote ", PIECEWISE_DEBUG_CSV)
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_piecewise_debug()
end
