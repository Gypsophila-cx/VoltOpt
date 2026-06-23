# ex2_run.jl
#
# Purpose:
#   Main runner for Example 2. It runs the VoltMat and Gutleb--Olver sweeps in
#   separate Julia processes, cleans stale CSV outputs when requested, and checks
#   that the expected diagnostic columns are present.
#
# Problem:
#   Second-kind Volterra equation for the aging Maxwell model,
#   u + V_K u = f, shared by ex2_VoltMat.jl and ex2_GO.jl.
#
# Method:
#   Orchestrates the VoltMat / Chebyshev and Gutleb--Olver / shifted Jacobi
#   implementations without changing their numerical grids or benchmark setup.
#
# Outputs:
#   data/ex2_VoltMat_error.csv
#   data/ex2_VoltMat_time.csv
#   data/ex2_GO_error.csv
#   data/ex2_GO_time.csv
#
# Run from the repository root:
#   julia --project=. examples/ex2_2ndVolterraEqn/ex2_run.jl
#
# Optional examples:
#   julia --project=. examples/ex2_2ndVolterraEqn/ex2_run.jl --only voltmat
#   julia --project=. examples/ex2_2ndVolterraEqn/ex2_run.jl --only go
#   julia --project=. examples/ex2_2ndVolterraEqn/ex2_run.jl --no-clean
#   julia --project=. examples/ex2_2ndVolterraEqn/ex2_run.jl --no-check
#   julia --project=. examples/ex2_2ndVolterraEqn/ex2_run.jl --continue
#   julia --project=. examples/ex2_2ndVolterraEqn/ex2_run.jl --no-exit
#
# REPL note:
#   If this file is executed by include("ex2_run.jl") inside a REPL, it will
#   never call exit(...). This prevents the REPL from being closed when a child
#   script fails. When executed as a standalone script, it returns a nonzero
#   exit code on failure unless --no-exit is supplied.

using Dates

function usage()
    println("""
Usage:
  julia --project=. examples/ex2_2ndVolterraEqn/ex2_run.jl [options]

Options:
  --only MODE        MODE = both | voltmat | go. Default: both.
  --voltmat PATH    Path to ex2_VoltMat.jl. Default: ex2_VoltMat.jl in this folder.
  --go PATH         Path to ex2_GO.jl. Default: ex2_GO.jl in this folder.
  --project PATH    Julia project used by child processes. Default: repository root.
  --continue        Continue to the next script if one script fails or output check fails.
  --no-clean        Do not remove old CSV files before running.
  --no-check        Do not check generated CSV files after running.
  --no-exit         Do not call exit(1) even when run as a standalone script.
  --help            Print this message.

Output files expected in data/:
  ex2_VoltMat_error.csv
  ex2_VoltMat_time.csv
  ex2_GO_error.csv
  ex2_GO_time.csv

This runner verifies that both methods output the fixed-RHS diagnostic columns:
  rhs_projection_Nmax, rhs_projection_N, rhs_fixed_N, rhs_effective_N,
  rhs_padding_N, rhs_truncation_N, rhs_fixed_precompute_sec,
  rhs_norm, rhs_max_abs, rhs_isfinite.

For error CSV files, it also verifies:
  uh_coeff_norm, uh_coeff_max_abs, uh_coeff_isfinite.
""")
end

function running_as_standalone_script()
    isempty(PROGRAM_FILE) && return false
    return abspath(PROGRAM_FILE) == abspath(@__FILE__)
end

function resolve_path(path::AbstractString; base::AbstractString=@__DIR__)
    if isabspath(path)
        return normpath(path)
    end
    return normpath(joinpath(base, path))
end

function parse_args(args)
    opts = Dict{String, Any}(
        "only" => "both",
        "voltmat" => joinpath(@__DIR__, "ex2_VoltMat.jl"),
        "go" => joinpath(@__DIR__, "ex2_GO.jl"),
        "project" => normpath(joinpath(@__DIR__, "..", "..")),
        "continue" => false,
        "clean" => true,
        "check" => true,
        "no_exit" => false,
        "help" => false,
    )

    i = 1
    while i <= length(args)
        arg = args[i]

        if arg == "--help" || arg == "-h"
            opts["help"] = true
            i += 1
        elseif arg == "--continue"
            opts["continue"] = true
            i += 1
        elseif arg == "--no-clean"
            opts["clean"] = false
            i += 1
        elseif arg == "--no-check"
            opts["check"] = false
            i += 1
        elseif arg == "--no-exit"
            opts["no_exit"] = true
            i += 1
        elseif arg in ("--only", "--voltmat", "--go", "--project")
            if i == length(args)
                error("Missing value after $(arg)")
            end
            key = replace(arg[3:end], "-" => "_")
            opts[key] = args[i + 1]
            i += 2
        else
            error("Unknown argument: $(arg). Use --help for usage.")
        end
    end

    only = lowercase(String(opts["only"]))
    if !(only in ("both", "voltmat", "go"))
        error("Invalid --only value: $(opts["only"]). Expected both, voltmat, or go.")
    end
    opts["only"] = only

    opts["voltmat"] = resolve_path(String(opts["voltmat"]))
    opts["go"] = resolve_path(String(opts["go"]))
    opts["project"] = resolve_path(String(opts["project"]))

    return opts
end

function julia_executable()
    return joinpath(Sys.BINDIR, Base.julia_exename())
end

function check_script(path::AbstractString)
    full = abspath(path)
    if !isfile(full)
        error("Script not found: $(full)")
    end
    return full
end

function file_mtime_string(path::AbstractString)
    if !isfile(path)
        return "missing"
    end

    try
        return Dates.format(unix2datetime(mtime(path)), "yyyy-mm-dd HH:MM:SS")
    catch
        return string(mtime(path))
    end
end

function method_outputs(label::AbstractString)
    data_dir = joinpath(@__DIR__, "data")

    if label == "VoltMat_Cheb"
        return [
            joinpath(data_dir, "ex2_VoltMat_error.csv"),
            joinpath(data_dir, "ex2_VoltMat_time.csv"),
        ]
    elseif label == "GutlebOlver"
        return [
            joinpath(data_dir, "ex2_GO_error.csv"),
            joinpath(data_dir, "ex2_GO_time.csv"),
        ]
    else
        error("Unknown method label: $(label)")
    end
end

function clean_outputs_for_tasks(tasks)
    println()
    println("Cleaning old CSV outputs")
    data_dir = joinpath(@__DIR__, "data")
    mkpath(data_dir)

    paths = String[]
    for (label, _) in tasks
        append!(paths, method_outputs(label))
    end

    for path in unique(paths)
        if isfile(path)
            rm(path; force=true)
            println("removed ", path)
        else
            println("not found ", path)
        end
    end

    flush(stdout)
end

function csv_header(path::AbstractString)
    open(path, "r") do io
        if eof(io)
            return String[]
        end
        return split(readline(io), ",")
    end
end

function count_csv_rows(path::AbstractString)
    if !isfile(path)
        return 0
    end

    n = 0
    open(path, "r") do io
        for _ in eachline(io)
            n += 1
        end
    end

    return max(n - 1, 0)
end

function count_failed_status(path::AbstractString)
    if !isfile(path)
        return missing
    end

    header = csv_header(path)
    idx = findfirst(==("status"), header)
    if idx === nothing
        return missing
    end

    failed = 0
    open(path, "r") do io
        first_line = true
        for line in eachline(io)
            if first_line
                first_line = false
                continue
            end

            # The status field itself is simple, so a lightweight split is enough here.
            parts = split(line, ",")
            if length(parts) >= idx && strip(parts[idx]) == "failed"
                failed += 1
            end
        end
    end

    return failed
end

function check_required_columns(csv_path::AbstractString, required_columns)
    if !isfile(csv_path)
        println("WARNING: missing CSV: ", csv_path)
        return false
    end

    header = csv_header(csv_path)
    missing_columns = String[]

    for col in required_columns
        if !(col in header)
            push!(missing_columns, col)
        end
    end

    rows = count_csv_rows(csv_path)
    failed_count = count_failed_status(csv_path)

    if isempty(missing_columns)
        println("checked ", basename(csv_path), ": ok, rows = ", rows,
            ", failed rows = ", failed_count)
        return true
    end

    println("WARNING: CSV is missing required columns")
    println("  file    = ", csv_path)
    println("  rows    = ", rows)
    println("  missing = ", missing_columns)
    return false
end

function check_outputs_for_method(label::AbstractString)
    rhs_required_columns = [
        "rhs_projection_Nmax",
        "rhs_projection_N",
        "rhs_fixed_N",
        "rhs_effective_N",
        "rhs_padding_N",
        "rhs_truncation_N",
        "rhs_fixed_precompute_sec",
        "rhs_norm",
        "rhs_max_abs",
        "rhs_isfinite",
    ]

    error_extra_columns = [
        "uh_coeff_norm",
        "uh_coeff_max_abs",
        "uh_coeff_isfinite",
    ]

    if label == "VoltMat_Cheb"
        error_csv = joinpath(@__DIR__, "data", "ex2_VoltMat_error.csv")
        time_csv = joinpath(@__DIR__, "data", "ex2_VoltMat_time.csv")
    elseif label == "GutlebOlver"
        error_csv = joinpath(@__DIR__, "data", "ex2_GO_error.csv")
        time_csv = joinpath(@__DIR__, "data", "ex2_GO_time.csv")
    else
        error("Unknown method label: $(label)")
    end

    println()
    println("Checking outputs for ", label)

    ok_error = check_required_columns(error_csv, vcat(rhs_required_columns, error_extra_columns))
    ok_time = check_required_columns(time_csv, rhs_required_columns)

    return ok_error && ok_time
end

function run_script(label::AbstractString, script_path::AbstractString, project_path::AbstractString)
    script = check_script(script_path)
    project = abspath(project_path)
    julia = julia_executable()

    println()
    println("============================================================")
    println("Running ", label)
    println("script  = ", script)
    println("mtime   = ", file_mtime_string(script))
    println("project = ", project)
    println("start   = ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    println("============================================================")
    flush(stdout)

    cmd = `$(julia) --project=$(project) $(script)`

    ok = false
    try
        ok = success(cmd)
    catch err
        println("Child process invocation threw an exception.")
        println("label = ", label)
        println("error = ", sprint(showerror, err))
        ok = false
    end

    println("============================================================")
    println(label, ok ? " finished successfully" : " failed")
    println("end     = ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    println("============================================================")
    flush(stdout)

    return ok
end

function main(args=ARGS)
    opts = parse_args(args)

    if Bool(opts["help"])
        usage()
        return true
    end

    only = String(opts["only"])
    project_path = String(opts["project"])
    continue_on_error = Bool(opts["continue"])
    clean_before_run = Bool(opts["clean"])
    check_after_run = Bool(opts["check"])

    tasks = Tuple{String, String}[]
    if only == "both" || only == "voltmat"
        push!(tasks, ("VoltMat_Cheb", String(opts["voltmat"])))
    end
    if only == "both" || only == "go"
        push!(tasks, ("GutlebOlver", String(opts["go"])))
    end

    println("ex2 runner")
    println("method isolation = separate Julia processes")
    println("selected mode    = ", only)
    println("continue on err  = ", continue_on_error)
    println("clean old CSV    = ", clean_before_run)
    println("check outputs    = ", check_after_run)
    println("standalone run   = ", running_as_standalone_script())
    println("project          = ", project_path)
    println("number of tasks  = ", length(tasks))
    flush(stdout)

    if clean_before_run
        clean_outputs_for_tasks(tasks)
    end

    results = Pair{String, Bool}[]
    output_results = Pair{String, Bool}[]

    for (label, script) in tasks
        ok = run_script(label, script, project_path)
        push!(results, label => ok)

        output_ok = true
        if ok && check_after_run
            output_ok = check_outputs_for_method(label)
            push!(output_results, label => output_ok)
        elseif check_after_run
            push!(output_results, label => false)
        end

        if (!ok || !output_ok) && !continue_on_error
            println()
            println("Stopping because ", label, " failed or output check failed.")
            println("Use --continue to run remaining scripts.")
            break
        end
    end

    println()
    println("Summary")
    for (label, ok) in results
        println(rpad(label, 16), ok ? "run ok" : "run failed")
    end

    if check_after_run
        println()
        println("Output checks")
        for (label, ok) in output_results
            println(rpad(label, 16), ok ? "outputs ok" : "outputs failed")
        end
    end

    all_runs_ok = all(last, results) && length(results) == length(tasks)
    all_outputs_ok = !check_after_run || (all(last, output_results) && length(output_results) == length(results))

    return all_runs_ok && all_outputs_ok
end

run_ok = main(ARGS)

if !run_ok
    if running_as_standalone_script() && !("--no-exit" in ARGS)
        exit(1)
    else
        println()
        println("ex2_run.jl completed with failure status, but exit(1) was not called.")
        println("This is intentional when included from a REPL or when --no-exit is supplied.")
    end
end
