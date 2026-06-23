# ex1_run.jl
#
# Purpose:
#   Main runner for the ex1 warmup comparison. It launches the error and timing
#   sweeps in separate Julia processes and checks the expected CSV outputs.
#
# Problem:
#   Low-order Volterra integro-differential warmup problem shared by
#   ex1_error.jl, ex1_timing.jl, and ex1_common.jl.
#
# Method:
#   Orchestrates the VoltMat / Chebyshev, VoltMat / Legendre, and
#   Gutleb--Olver / shifted Jacobi P^(1,0) comparison scripts without changing
#   their numerical settings.
#
# Inputs:
#   ex1_error.jl
#   ex1_timing.jl
#   ex1_common.jl
#
# Outputs:
#   data/ex1_error_N1_100.csv
#   data/ex1_timing_N10_1000_loggrid.csv
#
# Run from the repository root:
#   julia --project=. examples/ex1_warmup/ex1_run.jl
#
# Optional examples:
#   julia --project=. examples/ex1_warmup/ex1_run.jl --only error
#   julia --project=. examples/ex1_warmup/ex1_run.jl --only timing
#   julia --project=. examples/ex1_warmup/ex1_run.jl --no-clean
#   julia --project=. examples/ex1_warmup/ex1_run.jl --no-check
#   julia --project=. examples/ex1_warmup/ex1_run.jl --continue
#   julia --project=. examples/ex1_warmup/ex1_run.jl --no-exit

using Dates

function usage()
    println("""
Usage:
  julia --project=. examples/ex1_warmup/ex1_run.jl [options]

Options:
  --only MODE        MODE = both | error | timing. Default: both.
  --error PATH       Path to ex1_error.jl. Default: ex1_error.jl in this folder.
  --timing PATH      Path to ex1_timing.jl. Default: ex1_timing.jl in this folder.
  --project PATH     Julia project used by child processes. Default: repository root.
  --continue         Continue to the next script if one script fails or output check fails.
  --no-clean         Do not remove old CSV files before running.
  --no-check         Do not check generated CSV files after running.
  --no-exit          Do not call exit(1) even when run as a standalone script.
  --help             Print this message.

Output files expected in data/:
  ex1_error_N1_100.csv
  ex1_timing_N10_1000_loggrid.csv

Each sweep contains the warmup comparison methods:
  VoltMat / Chebyshev
  VoltMat / Legendre
  GutlebOlver / shifted Jacobi P^(1,0)
""")
end

function running_as_standalone_script()
    isempty(PROGRAM_FILE) && return false
    return abspath(PROGRAM_FILE) == abspath(@__FILE__)
end

function resolve_path(path::AbstractString; base::AbstractString=@__DIR__)
    isabspath(path) && return normpath(path)
    return normpath(joinpath(base, path))
end

function parse_args(args)
    opts = Dict{String, Any}(
        "only" => "both",
        "error" => joinpath(@__DIR__, "ex1_error.jl"),
        "timing" => joinpath(@__DIR__, "ex1_timing.jl"),
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
        elseif arg in ("--only", "--error", "--timing", "--project")
            i == length(args) && error("Missing value after $(arg)")
            key = replace(arg[3:end], "-" => "_")
            opts[key] = args[i + 1]
            i += 2
        else
            error("Unknown argument: $(arg). Use --help for usage.")
        end
    end

    only = lowercase(String(opts["only"]))
    if !(only in ("both", "error", "timing"))
        error("Invalid --only value: $(opts["only"]). Expected both, error, or timing.")
    end
    opts["only"] = only
    opts["error"] = resolve_path(String(opts["error"]))
    opts["timing"] = resolve_path(String(opts["timing"]))
    opts["project"] = resolve_path(String(opts["project"]))
    return opts
end

julia_executable() = joinpath(Sys.BINDIR, Base.julia_exename())

function check_script(path::AbstractString)
    full = abspath(path)
    isfile(full) || error("Script not found: $(full)")
    return full
end

function file_mtime_string(path::AbstractString)
    isfile(path) || return "missing"
    try
        return Dates.format(unix2datetime(mtime(path)), "yyyy-mm-dd HH:MM:SS")
    catch
        return string(mtime(path))
    end
end

function task_outputs(label::AbstractString)
    data_dir = joinpath(@__DIR__, "data")
    label == "error" && return [joinpath(data_dir, "ex1_error_N1_100.csv")]
    label == "timing" && return [joinpath(data_dir, "ex1_timing_N10_1000_loggrid.csv")]
    error("Unknown task label: $(label)")
end

function clean_outputs_for_tasks(tasks)
    println()
    println("Cleaning old CSV outputs")
    data_dir = joinpath(@__DIR__, "data")
    mkpath(data_dir)

    paths = String[]
    for (label, _) in tasks
        append!(paths, task_outputs(label))
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
        eof(io) && return String[]
        return csv_fields(readline(io))
    end
end

function csv_fields(line::AbstractString)
    fields = String[]
    buf = IOBuffer()
    in_quotes = false
    i = firstindex(line)
    while i <= lastindex(line)
        c = line[i]
        if c == '"'
            ni = nextind(line, i)
            if in_quotes && ni <= lastindex(line) && line[ni] == '"'
                print(buf, '"')
                i = nextind(line, ni)
                continue
            end
            in_quotes = !in_quotes
        elseif c == ',' && !in_quotes
            push!(fields, String(take!(buf)))
        else
            print(buf, c)
        end
        i = nextind(line, i)
    end
    push!(fields, String(take!(buf)))
    return fields
end

function count_csv_rows(path::AbstractString)
    isfile(path) || return 0
    n = 0
    open(path, "r") do io
        for _ in eachline(io)
            n += 1
        end
    end
    return max(n - 1, 0)
end

function status_counts(path::AbstractString)
    isfile(path) || return Dict{String, Int}()
    header = csv_header(path)
    idx = findfirst(==("status"), header)
    idx === nothing && return Dict{String, Int}()

    counts = Dict{String, Int}()
    open(path, "r") do io
        first_line = true
        for line in eachline(io)
            if first_line
                first_line = false
                continue
            end
            parts = csv_fields(line)
            status = length(parts) >= idx ? strip(parts[idx]) : ""
            counts[status] = get(counts, status, 0) + 1
        end
    end
    return counts
end

function check_required_columns(csv_path::AbstractString, required_columns)
    if !isfile(csv_path)
        println("WARNING: missing CSV: ", csv_path)
        return false
    end

    header = csv_header(csv_path)
    missing_columns = [col for col in required_columns if !(col in header)]
    rows = count_csv_rows(csv_path)
    counts = status_counts(csv_path)

    if isempty(missing_columns) && rows > 0
        println("checked ", basename(csv_path), ": ok, rows = ", rows,
            ", status counts = ", counts)
        return true
    end

    println("WARNING: CSV check failed")
    println("  file          = ", csv_path)
    println("  rows          = ", rows)
    println("  status counts = ", counts)
    println("  missing       = ", missing_columns)
    return false
end

function check_output_for_task(label::AbstractString)
    println()
    println("Checking outputs for ", label)
    if label == "error"
        return check_required_columns(
            first(task_outputs(label)),
            ["method", "basis", "N", "max_abs_error", "status", "error_message"],
        )
    elseif label == "timing"
        return check_required_columns(
            first(task_outputs(label)),
            ["method", "basis", "N", "median_elapsed_time", "status", "error_message"],
        )
    end
    error("Unknown task label: $(label)")
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
    ok = try
        success(cmd)
    catch err
        println("Child process invocation threw an exception.")
        println("label = ", label)
        println("error = ", sprint(showerror, err))
        false
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
    if only == "both" || only == "error"
        push!(tasks, ("error", String(opts["error"])))
    end
    if only == "both" || only == "timing"
        push!(tasks, ("timing", String(opts["timing"])))
    end

    println("ex1 warmup runner")
    println("task isolation  = separate Julia processes")
    println("selected mode   = ", only)
    println("continue on err = ", continue_on_error)
    println("clean old CSV   = ", clean_before_run)
    println("check outputs   = ", check_after_run)
    println("standalone run  = ", running_as_standalone_script())
    println("project         = ", project_path)
    println("number of tasks = ", length(tasks))
    flush(stdout)

    clean_before_run && clean_outputs_for_tasks(tasks)

    results = Pair{String, Bool}[]
    output_results = Pair{String, Bool}[]
    for (label, script) in tasks
        ok = run_script(label, script, project_path)
        push!(results, label => ok)

        output_ok = true
        if ok && check_after_run
            output_ok = check_output_for_task(label)
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
        println(rpad(label, 12), ok ? "run ok" : "run failed")
    end
    if check_after_run
        println()
        println("Output checks")
        for (label, ok) in output_results
            println(rpad(label, 12), ok ? "outputs ok" : "outputs failed")
        end
    end

    all_runs_ok = all(last, results) && length(results) == length(tasks)
    all_outputs_ok = !check_after_run ||
                     (all(last, output_results) && length(output_results) == length(results))
    return all_runs_ok && all_outputs_ok
end

run_ok = main(ARGS)

if !run_ok
    if running_as_standalone_script() && !("--no-exit" in ARGS)
        exit(1)
    else
        println()
        println("ex1_run.jl completed with failure status, but exit(1) was not called.")
        println("This is intentional when included from a REPL or when --no-exit is supplied.")
    end
end
