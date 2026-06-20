# test_GO_N23658.jl
# Put this file in examples/ex1_2ndVolterraEqn and run:
#   julia --project=. examples/ex1_2ndVolterraEqn/test_GO_N23658.jl
#
# Purpose:
#   Load definitions from ex1_GO.jl without running its full sweeps, then test
#   the single case N = 23658 with the fixed 20000-term RHS cache.

using Dates
using Printf

function load_ex1_go_definitions(ex1_go_path::AbstractString)
    path = abspath(ex1_go_path)
    isfile(path) || error("ex1_GO.jl not found: $path")

    src = read(path, String)
    marker = "# Script entry."
    idx = findfirst(marker, src)
    idx === nothing && error("Could not find marker '$marker' in $path")

    # Evaluate only definitions/configuration before the script entry section.
    defs = src[1:first(idx)-1]
    include_string(Main, defs, path)
    return path
end

function print_value(label, value)
    println(rpad(label, 30), " = ", value)
end

function main()
    test_N = 23658
    ex1_go_path = joinpath(@__DIR__, "ex1_GO.jl")

    println("test_GO_N23658")
    print_value("start", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    print_value("script", ex1_go_path)

    load_ex1_go_definitions(ex1_go_path)

    println()
    println("Loaded ex1_GO definitions")
    print_value("method", method)
    print_value("equation_sign", equation_sign)
    print_value("kerneldepth", kerneldepth)
    print_value("rhs_fixed_N", rhs_fixed_N)
    print_value("tol", tol)
    print_value("VoltMat_Cheb defined", isdefined(Main, :VoltMat_Cheb))
    print_value("GutlebOlver defined", isdefined(Main, :GutlebOlver))

    p = ex1_params()
    gl_x, gl_w = gausslegendre_rule(p.rhs_q)
    rhs_fn = make_rhs_go(p, gl_x, gl_w)
    go_space = Jacobi(0, 1, 0..1)
    stest = make_test_grid(p)
    utest_exact = u_go.(stest, Ref(p))

    println()
    println("Precomputing fixed RHS cache")
    GC.gc()
    rhs_cache_timed = @timed precompute_rhs_cache(rhs_fn; rhs_fixed_N=rhs_fixed_N)
    rhs_cache = rhs_cache_timed.value
    rhs_precompute_sec = rhs_cache_timed.time

    print_value("rhs_precompute_sec", rhs_precompute_sec)
    print_value("length(rhs_cache)", length(rhs_cache))
    print_value("norm(rhs_cache)", norm(rhs_cache))
    print_value("maximum(abs, rhs_cache)", maximum(abs, rhs_cache))
    print_value("all finite rhs_cache", all(isfinite, rhs_cache))

    println()
    println("Solving single case")
    print_value("N", test_N)

    row = try
        solve_error_one_N(
            test_N,
            kerneldepth,
            rhs_cache,
            go_space,
            stest,
            utest_exact;
            rhs_precompute_sec=rhs_precompute_sec,
        )
    catch err
        println("FAILED")
        println("error_message = ", sprint(showerror, err))
        println("backtrace:")
        showerror(stdout, err, catch_backtrace())
        println()
        return 1
    end

    println()
    println("Result")
    fields_to_print = [
        :method,
        :N,
        :tol,
        :equation_sign,
        :grid_type,
        :kerneldepth,
        :rhs_projection_Nmax,
        :rhs_projection_N,
        :rhs_fixed_N,
        :rhs_effective_N,
        :rhs_padding_N,
        :rhs_truncation_N,
        :rhs_fixed_precompute_sec,
        :rhs_norm,
        :rhs_max_abs,
        :rhs_isfinite,
        :operator_type,
        :system_type,
        :operator_nnz,
        :system_nnz,
        :residual_norm,
        :relative_residual,
        :max_abs_error,
        :relative_error,
        :max_abs_uh,
        :max_abs_uexact,
        :uh_coeff_norm,
        :uh_coeff_max_abs,
        :uh_coeff_isfinite,
        :status,
        :error_message,
    ]

    for name in fields_to_print
        print_value(String(name), getproperty(row, name))
    end

    # Also write a one-row CSV for quick inspection.
    out_csv = joinpath(@__DIR__, "data", "test_GO_N23658.csv")
    cols = fields_to_print
    mkpath(dirname(out_csv))
    open(out_csv, "w") do io
        println(io, join(string.(cols), ","))
        println(io, join((csv_escape(getproperty(row, c)) for c in cols), ","))
    end
    println()
    print_value("wrote", out_csv)
    print_value("end", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))

    return row.status == "ok" ? 0 : 1
end

exit(main())
