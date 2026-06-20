# ex4_test.jl
# Isolated large-scale test for Example 4 at N = 30000.

using Dates

script_dir = @__DIR__
include(joinpath(script_dir, "ex4_common.jl"))

data_dir = joinpath(script_dir, "data")
test_results_csv = joinpath(data_dir, "ex4_test_results.csv")
shape_csv = joinpath(data_dir, "ex4_solution_shape.csv")

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

function julia_executable()
    return joinpath(Sys.BINDIR, Base.julia_exename())
end

struct TestTask
    label::String
    script::String
end

function run_test_task(task::TestTask, project_root::AbstractString, env_pairs)
    script = abspath(task.script)
    project = abspath(project_root)
    isfile(script) || error("Script not found for $(task.label): $(script)")

    base_cmd = `$(julia_executable()) --project=$(project) $(script)`
    cmd = addenv(base_cmd, env_pairs...)

    println()
    println("============================================================")
    println("Running Example 4 test task: ", task.label)
    println("script   = ", script)
    println("project  = ", project)
    println("command  = ", base_cmd)
    println("start    = ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    println("============================================================")
    flush(stdout)

    timeout_sec = ex4_env_int("EX4_TEST_TIMEOUT_SEC", 1200)
    exit_code = -1
    ok = false
    timed_out = false
    elapsed = @elapsed begin
        proc = run(pipeline(cmd; stdout=stdout, stderr=stderr); wait=false)
        deadline = time() + timeout_sec
        while process_running(proc) && time() < deadline
            sleep(1.0)
        end

        if process_running(proc)
            timed_out = true
            kill(proc)
        end

        wait(proc)
        exit_code = timed_out ? -1 : getfield(proc, :exitcode)
        ok = !timed_out && exit_code == 0
    end

    println("============================================================")
    println(task.label, ok ? " finished successfully" : " failed")
    println("exit code = ", exit_code)
    println("timeout   = ", timeout_sec, " sec")
    println("timed out = ", timed_out)
    println("elapsed   = ", round(elapsed; digits=3), " sec")
    println("end       = ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    if !ok
        println("failed command:")
        println("  ", base_cmd)
        println("The child process wrote its error output above.")
    end
    println("============================================================")
    flush(stdout)

    return (; task, ok, exit_code, elapsed, timed_out)
end

function summary_has_method(path::AbstractString, method::AbstractString)
    isfile(path) || return false
    open(path, "r") do io
        first_line = true
        for line in eachline(io)
            if first_line
                first_line = false
                continue
            end
            startswith(line, method * ",") && return true
        end
    end
    return false
end

function append_missing_failure_row(task::TestTask, n_test::Int, status::AbstractString)
    method = task.label == "VoltMat" ? "VoltMat" : "GutlebOlver"
    summary_has_method(test_results_csv, method) && return
    ex4_append_csv_row(test_results_csv, (
        method,
        n_test,
        ex4_reference_degree(n_test),
        NaN,
        NaN,
        NaN,
        status,
    ))
end

function print_csv_tail(path::AbstractString)
    println()
    println("Summary CSV: ", path)
    if !isfile(path)
        println("  missing")
        return
    end

    lines = readlines(path)
    for line in lines
        println(line)
    end
end

function main()
    mkpath(data_dir)
    ex4_write_csv_header(test_results_csv, ex4_test_result_columns)
    ex4_write_csv_header(shape_csv, ex4_shape_columns)

    omega = ex4_env_float("EX4_OMEGA", 1.0)
    grid_m = ex4_env_int("EX4_GRID_M", 4096)
    shape_points = ex4_env_int("EX4_SHAPE_POINTS", 1001)
    n_test = ex4_env_int("EX4_TEST_N", 30_000)

    env_pairs = [
        "EX4_N_LIST" => string(n_test),
        "EX4_N_FINAL" => string(n_test),
        "EX4_OMEGA" => string(omega),
        "EX4_GRID_M" => string(grid_m),
        "EX4_WRITE_SHAPE" => "1",
        "EX4_SHAPE_N" => string(n_test),
        "EX4_SHAPE_POINTS" => string(shape_points),
        "EX4_SUMMARY_CSV" => test_results_csv,
        "EX4_SHAPE_CSV" => shape_csv,
    ]

    project_root = find_project_root(script_dir)
    tasks = [
        TestTask("VoltMat", joinpath(script_dir, "ex4_VoltMat.jl")),
        TestTask("GutlebOlver", joinpath(script_dir, "ex4_GO.jl")),
    ]

    println("Example 4 large-scale isolated test")
    println("N                  = ", n_test)
    println("N_ref              = ", ex4_reference_degree(n_test))
    println("omega              = ", omega)
    println("diagnostic grid M  = ", grid_m)
    println("shape points       = ", shape_points)
    println("summary CSV        = ", test_results_csv)
    println("shape CSV          = ", shape_csv)
    flush(stdout)

    results = []
    for task in tasks
        result = run_test_task(task, project_root, env_pairs)
        push!(results, result)
        if !result.ok
            status = result.timed_out ?
                "failed:timeout_after_$(ex4_env_int("EX4_TEST_TIMEOUT_SEC", 1200))s" :
                "failed:child_exit_code_$(result.exit_code)"
            append_missing_failure_row(task, n_test, status)
        end
    end

    print_csv_tail(test_results_csv)
    println()
    println("Shape data CSV: ", shape_csv)

    all_ok = all(result -> result.ok, results)
    if !all_ok
        println()
        println("At least one child process failed. Completed CSV rows were preserved.")
        return 1
    end

    return 0
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    exit(main())
end
