# ex4_run.jl
# Run ex4_VoltMat.jl and ex4_GO.jl in separate Julia processes.
#
# Examples:
#   julia --project=. examples/ex4_SPVIDE/ex4_run.jl --only voltmat
#   julia --project=. examples/ex4_SPVIDE/ex4_run.jl --only go
#   julia --project=. examples/ex4_SPVIDE/ex4_run.jl --only both

using Dates

script_dir = @__DIR__
include(joinpath(script_dir, "ex4_common.jl"))

data_dir = joinpath(script_dir, "data")

function usage()
    println("""
Usage:
  julia --project=. examples/ex4_SPVIDE/ex4_run.jl [options]

Options:
  --only MODE        MODE = both | voltmat | go. Default: both.
  --voltmat PATH    Path to ex4_VoltMat.jl. Default: ex4_VoltMat.jl in this folder.
  --go PATH         Path to ex4_GO.jl. Default: ex4_GO.jl in this folder.
  --project PATH    Julia project used by child processes. Default: repository root.
  --continue        Continue to the next method after a failure.
  --no-clean        Do not remove selected method result CSV files before running.
  --no-check        Do not check generated result CSV files after running.
  --no-exit         Do not call exit(1) when run as a standalone script.
  --help            Print this message.

Main outputs in data/:
  ex4_VoltMat_results.csv
  ex4_GO_results.csv
""")
end

function running_as_standalone_script()
    isempty(PROGRAM_FILE) && return false
    return abspath(PROGRAM_FILE) == abspath(@__FILE__)
end

function find_project_root(start_dir::AbstractString=script_dir)
    dir = abspath(start_dir)
    while true
        if isfile(joinpath(dir, "Project.toml")) && isdir(joinpath(dir, "src"))
            return dir
        end
        parent = dirname(dir)
        parent == dir && error("Could not find the VoltOpt project root from $(start_dir).")
        dir = parent
    end
end

function resolve_path(path::AbstractString; base::AbstractString=script_dir)
    isabspath(path) && return normpath(path)
    return normpath(joinpath(base, path))
end

function parse_args(args)
    opts = Dict{String, Any}(
        "only" => "both",
        "voltmat" => joinpath(script_dir, "ex4_VoltMat.jl"),
        "go" => joinpath(script_dir, "ex4_GO.jl"),
        "project" => find_project_root(script_dir),
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
        elseif arg == "--continue" || arg == "--continue-on-error"
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
            i == length(args) && error("Missing value after $(arg)")
            key = replace(arg[3:end], "-" => "_")
            opts[key] = args[i + 1]
            i += 2
        elseif startswith(arg, "--only=") || startswith(arg, "--project=")
            key, value = split(arg[3:end], "="; limit=2)
            opts[replace(key, "-" => "_")] = value
            i += 1
        else
            error("Unknown argument: $(arg). Use --help for usage.")
        end
    end

    only = lowercase(String(opts["only"]))
    only in ("both", "voltmat", "go") ||
        error("Invalid --only value: $(opts["only"]). Expected both, voltmat, or go.")
    opts["only"] = only

    opts["voltmat"] = resolve_path(String(opts["voltmat"]))
    opts["go"] = resolve_path(String(opts["go"]))
    opts["project"] = resolve_path(String(opts["project"]); base=pwd())
    return opts
end

function julia_executable()
    return joinpath(Sys.BINDIR, Base.julia_exename())
end

struct Ex4Task
    label::String
    script::String
    result_csv::String
end

function selected_tasks(opts)
    only = String(opts["only"])
    tasks = Ex4Task[]
    if only == "both" || only == "voltmat"
        push!(tasks, Ex4Task("VoltMat", String(opts["voltmat"]), joinpath(data_dir, "ex4_VoltMat_results.csv")))
    end
    if only == "both" || only == "go"
        push!(tasks, Ex4Task("GutlebOlver", String(opts["go"]), joinpath(data_dir, "ex4_GO_results.csv")))
    end
    return tasks
end

function clean_outputs(tasks)
    mkpath(data_dir)
    println()
    println("Cleaning selected Example 4 result CSV files")
    for task in tasks
        if isfile(task.result_csv)
            rm(task.result_csv; force=true)
            println("removed ", task.result_csv)
        else
            println("not found ", task.result_csv)
        end
    end
    flush(stdout)
end

function csv_header(path::AbstractString)
    open(path, "r") do io
        eof(io) && return String[]
        return split(readline(io), ",")
    end
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

function check_result_csv(task::Ex4Task)
    println()
    println("Checking output for ", task.label)

    if !isfile(task.result_csv)
        println("WARNING: missing result CSV: ", task.result_csv)
        return false
    end

    header = csv_header(task.result_csv)
    required = ["method", "N", "N_ref", "cauchy_coeff_error", "cauchy_grid_error", "total_time_sec", "status"]
    missing_columns = [col for col in required if !(col in header)]
    rows = count_csv_rows(task.result_csv)

    if isempty(missing_columns) && rows > 0
        println("checked ", basename(task.result_csv), ": ok, rows = ", rows)
        return true
    end

    println("WARNING: result CSV check failed")
    println("  file    = ", task.result_csv)
    println("  rows    = ", rows)
    println("  missing = ", missing_columns)
    return false
end

function run_task(task::Ex4Task, project_path::AbstractString)
    script = abspath(task.script)
    project = abspath(project_path)
    isfile(script) || error("Script not found for $(task.label): $(script)")
    isfile(joinpath(project, "Project.toml")) || error("Project.toml not found: $(project)")

    cmd = `$(julia_executable()) --project=$(project) $(script)`

    println()
    println("============================================================")
    println("Running ", task.label, " in an isolated Julia process")
    println("script   = ", script)
    println("project  = ", project)
    println("command  = ", cmd)
    println("start    = ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    println("============================================================")
    flush(stdout)

    exit_code = -1
    ok = false
    elapsed = @elapsed begin
        proc = run(pipeline(cmd; stdout=stdout, stderr=stderr); wait=false)
        wait(proc)
        exit_code = getfield(proc, :exitcode)
        ok = exit_code == 0
    end

    println("============================================================")
    println(task.label, ok ? " finished successfully" : " failed")
    println("exit code = ", exit_code)
    println("elapsed   = ", round(elapsed; digits=3), " sec")
    println("end       = ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    if !ok
        println("failed command:")
        println("  ", cmd)
        println("The child process wrote its error output above.")
    end
    println("============================================================")
    flush(stdout)

    return (; task, ok, exit_code, elapsed)
end

function main(args=ARGS)
    opts = parse_args(args)
    if Bool(opts["help"])
        usage()
        return true
    end

    mkpath(data_dir)
    tasks = selected_tasks(opts)
    continue_on_error = Bool(opts["continue"])
    check_after_run = Bool(opts["check"])

    println("Example 4 isolated runner")
    println("method isolation = separate Julia processes")
    println("selected mode    = ", opts["only"])
    println("continue on err  = ", continue_on_error)
    println("clean old CSV    = ", opts["clean"])
    println("check outputs    = ", check_after_run)
    println("project          = ", opts["project"])
    println("data dir         = ", data_dir)
    println("number of tasks  = ", length(tasks))
    flush(stdout)

    Bool(opts["clean"]) && clean_outputs(tasks)

    results = []
    output_results = Pair{String, Bool}[]
    for task in tasks
        result = run_task(task, String(opts["project"]))
        push!(results, result)

        output_ok = result.ok
        if result.ok && check_after_run
            output_ok = check_result_csv(task)
        end
        check_after_run && push!(output_results, task.label => output_ok)

        if (!result.ok || !output_ok) && !continue_on_error
            println()
            println("Stopping because ", task.label, " failed or output check failed.")
            println("Use --continue to run remaining methods.")
            break
        end
    end

    println()
    println("Summary")
    for result in results
        println(rpad(result.task.label, 16),
            result.ok ? "run ok" : "run failed",
            ", exit code = ", result.exit_code)
    end

    if check_after_run
        println()
        println("Output checks")
        for (label, ok) in output_results
            println(rpad(label, 16), ok ? "outputs ok" : "outputs failed")
        end
    end

    all_runs_ok = length(results) == length(tasks) && all(r -> r.ok, results)
    all_outputs_ok = !check_after_run || (length(output_results) == length(results) && all(last, output_results))
    return all_runs_ok && all_outputs_ok
end

run_ok = main(ARGS)

if !run_ok
    if running_as_standalone_script() && !("--no-exit" in ARGS)
        exit(1)
    else
        println()
        println("ex4_run.jl completed with failure status, but exit(1) was not called.")
    end
end
