# test_GO_N23658_worldsafe.jl
# Put this file in examples/ex1_2ndVolterraEqn and run:
#   julia --project=. examples/ex1_2ndVolterraEqn/test_GO_N23658_worldsafe.jl
#
# Purpose:
#   Load definitions from ex1_GO.jl without running the full sweeps, then test
#   the single case N = 23658 using a fixed 20000-term RHS cache.
#
# Important:
#   The ex1_GO.jl definitions are loaded at top level before main() is defined.
#   This avoids Julia world-age errors such as:
#       method too new to be called from this world context

using Dates
using LinearAlgebra
using Printf

ex1_go_path = joinpath(@__DIR__, "ex1_GO.jl")
test_N = 23658
rhs_fixed_N_test = 20000

function load_ex1_go_definitions_toplevel(ex1_go_path::AbstractString)
    path = abspath(ex1_go_path)
    isfile(path) || error("ex1_GO.jl not found: $path")

    src = read(path, String)
    marker = "# Script entry."
    idx = findfirst(marker, src)
    idx === nothing && error("Could not find marker '$marker' in $path")

    defs = src[1:first(idx)-1]
    include_string(Main, defs, path)
    return path
end

# Load ex1_GO.jl definitions at top level to avoid world-age issues.
loaded_ex1_go_path = load_ex1_go_definitions_toplevel(ex1_go_path)

function print_value(label, value)
    println(rpad(label, 36), " = ", value)
end

function csv_escape(x)
    s = string(x)
    if occursin(",", s) || occursin("\"", s) || occursin("\n", s) || occursin("\r", s)
        return "\"" * replace(s, "\"" => "\"\"") * "\""
    end
    return s
end

function assert_vector_finite(label::AbstractString, v)
    all(isfinite, v) || error(label * " contains non-finite entries")
    return nothing
end

function max_abs_value(v)
    return isempty(v) ? NaN : maximum(abs, v)
end

function make_fixed_rhs_cache_test(rhs_fn, rhs_fixed_N::Int)
    rhs_fixed_N > 0 || throw(ArgumentError("rhs_fixed_N must be positive"))

    if isdefined(Main, :precompute_rhs_cache)
        cache = precompute_rhs_cache(rhs_fn; rhs_fixed_N=rhs_fixed_N)
        return Vector{Float64}(cache)
    end

    isdefined(Main, :build_rhs) || error("Neither precompute_rhs_cache nor build_rhs is defined after loading ex1_GO.jl")
    cache = build_rhs(rhs_fixed_N, rhs_fn)
    return Vector{Float64}(cache)
end

function rhs_from_fixed_cache_test(N::Int, rhs_cache::AbstractVector{<:Real})
    fixed_N = length(rhs_cache)
    if N <= fixed_N
        return Vector{Float64}(@view rhs_cache[1:N])
    end

    rhs = zeros(Float64, N)
    rhs[1:fixed_N] .= rhs_cache
    return rhs
end

function rhs_diagnostics_test(N::Int, rhs::AbstractVector, rhs_cache::AbstractVector, rhs_precompute_sec::Real)
    fixed_N = length(rhs_cache)
    effective_N = min(N, fixed_N)
    padding_N = max(N - fixed_N, 0)
    truncation_N = max(fixed_N - N, 0)

    return (
        rhs_fixed_N=fixed_N,
        rhs_effective_N=effective_N,
        rhs_padding_N=padding_N,
        rhs_truncation_N=truncation_N,
        rhs_precompute_sec=rhs_precompute_sec,
        rhs_cache_norm=norm(rhs_cache),
        rhs_cache_max_abs=max_abs_value(rhs_cache),
        rhs_cache_isfinite=all(isfinite, rhs_cache),
        rhs_norm=norm(rhs),
        rhs_max_abs=max_abs_value(rhs),
        rhs_isfinite=all(isfinite, rhs),
    )
end

function solve_single_N_with_fixed_rhs_test(N::Int, kerneldepth::Int, rhs_cache; rhs_precompute_sec::Real)
    println()
    println("Stage 1: build operator")
    GC.gc()
    t_vk = @timed build_operator(N, kerneldepth)
    VK = t_vk.value
    print_value("operator_build_sec", t_vk.time)
    print_value("operator_type", typeof(VK))
    print_value("operator_nnz", matrix_nnz(VK))

    println()
    println("Stage 2: build RHS from fixed cache")
    rhs = rhs_from_fixed_cache_test(N, rhs_cache)
    rhs_diag = rhs_diagnostics_test(N, rhs, rhs_cache, rhs_precompute_sec)
    assert_vector_finite("rhs", rhs)
    print_value("rhs_fixed_N", rhs_diag.rhs_fixed_N)
    print_value("rhs_effective_N", rhs_diag.rhs_effective_N)
    print_value("rhs_padding_N", rhs_diag.rhs_padding_N)
    print_value("rhs_truncation_N", rhs_diag.rhs_truncation_N)
    print_value("rhs_norm", rhs_diag.rhs_norm)
    print_value("rhs_max_abs", rhs_diag.rhs_max_abs)
    print_value("rhs_isfinite", rhs_diag.rhs_isfinite)

    println()
    println("Stage 3: build system")
    GC.gc()
    t_A = @timed build_system(VK, N)
    Aop = t_A.value
    print_value("system_build_sec", t_A.time)
    print_value("system_type", typeof(Aop))
    print_value("system_nnz", matrix_nnz(Aop))

    println()
    println("Stage 4: solve coefficient system")
    GC.gc()
    t_solve = @timed solve_system(Aop, rhs)
    uh_coeff = t_solve.value
    assert_vector_finite("uh_coeff", uh_coeff)
    print_value("solve_sec", t_solve.time)
    print_value("uh_coeff_norm", norm(uh_coeff))
    print_value("uh_coeff_max_abs", max_abs_value(uh_coeff))
    print_value("uh_coeff_isfinite", all(isfinite, uh_coeff))

    println()
    println("Stage 5: residual")
    residual = Aop * uh_coeff - rhs
    assert_vector_finite("residual", residual)
    residual_norm = norm(residual)
    relative_residual = residual_norm / max(norm(rhs), eps(Float64))
    print_value("residual_norm", residual_norm)
    print_value("relative_residual", relative_residual)

    println()
    println("Stage 6: small-grid function evaluation")
    p = ex1_params()
    go_space = Jacobi(0, 1, 0..1)

    # Keep this grid small to avoid crashing the REPL/process during diagnosis.
    # This is a stability test, not the final high-resolution error plot.
    stest = collect(range(0.0, 1.0; length=1001))
    utest_exact = u_go.(stest, Ref(p))

    GC.gc()
    t_eval = @timed Fun(go_space, uh_coeff).(stest)
    uh = t_eval.value
    assert_vector_finite("uh values on small test grid", uh)

    err = uh .- utest_exact
    assert_vector_finite("pointwise error", err)
    max_abs_error = max_abs_value(err)
    relative_error = norm(err) / max(norm(utest_exact), eps(Float64))

    print_value("eval_sec", t_eval.time)
    print_value("max_abs_error_small_grid", max_abs_error)
    print_value("relative_error_small_grid", relative_error)
    print_value("max_abs_uh_small_grid", max_abs_value(uh))
    print_value("max_abs_uexact_small_grid", max_abs_value(utest_exact))

    return (
        method=method,
        N=N,
        tol=tol,
        equation_sign=equation_sign,
        kerneldepth=kerneldepth,
        rhs_fixed_N=rhs_diag.rhs_fixed_N,
        rhs_effective_N=rhs_diag.rhs_effective_N,
        rhs_padding_N=rhs_diag.rhs_padding_N,
        rhs_truncation_N=rhs_diag.rhs_truncation_N,
        rhs_precompute_sec=rhs_diag.rhs_precompute_sec,
        rhs_cache_norm=rhs_diag.rhs_cache_norm,
        rhs_cache_max_abs=rhs_diag.rhs_cache_max_abs,
        rhs_cache_isfinite=rhs_diag.rhs_cache_isfinite,
        rhs_norm=rhs_diag.rhs_norm,
        rhs_max_abs=rhs_diag.rhs_max_abs,
        rhs_isfinite=rhs_diag.rhs_isfinite,
        operator_type=string(typeof(VK)),
        system_type=string(typeof(Aop)),
        operator_nnz=matrix_nnz(VK),
        system_nnz=matrix_nnz(Aop),
        operator_build_sec=t_vk.time,
        system_build_sec=t_A.time,
        solve_sec=t_solve.time,
        eval_sec=t_eval.time,
        residual_norm=residual_norm,
        relative_residual=relative_residual,
        max_abs_error=max_abs_error,
        relative_error=relative_error,
        max_abs_uh=max_abs_value(uh),
        max_abs_uexact=max_abs_value(utest_exact),
        uh_coeff_norm=norm(uh_coeff),
        uh_coeff_max_abs=max_abs_value(uh_coeff),
        uh_coeff_isfinite=all(isfinite, uh_coeff),
        status="ok",
        error_message="",
    )
end

function failure_row_test(N::Int, rhs_fixed_N::Int, err)
    return (
        method=isdefined(Main, :method) ? method : "GutlebOlver",
        N=N,
        tol=isdefined(Main, :tol) ? tol : NaN,
        equation_sign=isdefined(Main, :equation_sign) ? equation_sign : "u + V_K u = f",
        kerneldepth=isdefined(Main, :kerneldepth) ? kerneldepth : missing,
        rhs_fixed_N=rhs_fixed_N,
        rhs_effective_N=missing,
        rhs_padding_N=missing,
        rhs_truncation_N=missing,
        rhs_precompute_sec=NaN,
        rhs_cache_norm=NaN,
        rhs_cache_max_abs=NaN,
        rhs_cache_isfinite=false,
        rhs_norm=NaN,
        rhs_max_abs=NaN,
        rhs_isfinite=false,
        operator_type="",
        system_type="",
        operator_nnz=missing,
        system_nnz=missing,
        operator_build_sec=NaN,
        system_build_sec=NaN,
        solve_sec=NaN,
        eval_sec=NaN,
        residual_norm=NaN,
        relative_residual=NaN,
        max_abs_error=NaN,
        relative_error=NaN,
        max_abs_uh=NaN,
        max_abs_uexact=NaN,
        uh_coeff_norm=NaN,
        uh_coeff_max_abs=NaN,
        uh_coeff_isfinite=false,
        status="failed",
        error_message=sprint(showerror, err),
    )
end

function write_one_row_csv_test(path::AbstractString, row)
    cols = collect(propertynames(row))
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(string.(cols), ","))
        println(io, join((csv_escape(getproperty(row, c)) for c in cols), ","))
    end
    return cols
end

function main()
    println("test_GO_N23658_worldsafe")
    print_value("start", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    print_value("script", loaded_ex1_go_path)
    print_value("test_N", test_N)
    print_value("rhs_fixed_N_test", rhs_fixed_N_test)

    println()
    println("Loaded ex1_GO definitions")
    print_value("method", method)
    print_value("equation_sign", equation_sign)
    print_value("kerneldepth", kerneldepth)
    print_value("tol", tol)
    print_value("VoltMat_Cheb defined", isdefined(Main, :VoltMat_Cheb))
    print_value("GutlebOlver defined", isdefined(Main, :GutlebOlver))

    row = try
        p = ex1_params()
        gl_x, gl_w = gausslegendre_rule(p.rhs_q)
        rhs_fn = make_rhs_go(p, gl_x, gl_w)

        println()
        println("Precomputing fixed RHS cache")
        GC.gc()
        rhs_cache_timed = @timed make_fixed_rhs_cache_test(rhs_fn, rhs_fixed_N_test)
        rhs_cache = rhs_cache_timed.value
        rhs_precompute_sec = rhs_cache_timed.time

        print_value("rhs_precompute_sec", rhs_precompute_sec)
        print_value("length(rhs_cache)", length(rhs_cache))
        print_value("norm(rhs_cache)", norm(rhs_cache))
        print_value("maximum(abs, rhs_cache)", max_abs_value(rhs_cache))
        print_value("all finite rhs_cache", all(isfinite, rhs_cache))

        assert_vector_finite("rhs_cache", rhs_cache)

        solve_single_N_with_fixed_rhs_test(
            test_N,
            kerneldepth,
            rhs_cache;
            rhs_precompute_sec=rhs_precompute_sec,
        )
    catch err
        println()
        println("FAILED")
        println("error_message = ", sprint(showerror, err))
        failure_row_test(test_N, rhs_fixed_N_test, err)
    end

    println()
    println("Result")
    for name in propertynames(row)
        print_value(String(name), getproperty(row, name))
    end

    out_csv = joinpath(@__DIR__, "data", "test_GO_N23658_worldsafe.csv")
    write_one_row_csv_test(out_csv, row)
    println()
    print_value("wrote", out_csv)
    print_value("end", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))

    return row.status == "ok" ? 0 : 1
end

exit(main())
