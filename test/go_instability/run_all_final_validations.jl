# Run the final GO instability validations in separate Julia processes.

const SCRIPT_DIR = @__DIR__
const PROJECT_ROOT = normpath(joinpath(SCRIPT_DIR, "..", ".."))
const GO_PROJECT = joinpath(PROJECT_ROOT, "others", "GutlebOlver")
const JULIA_EXE = joinpath(Sys.BINDIR, Base.julia_exename())

const VALIDATION_SCRIPTS = (
    "run_padding_validation.jl",
    "run_construction_validation.jl",
    "run_nonnormality_validation.jl",
    "run_output_error_validation.jl",
    "run_layerwise_diagnostics.jl",
)

function run_child(script_name::AbstractString)
    script_path = joinpath(SCRIPT_DIR, script_name)
    cmd = `$(JULIA_EXE) --project=$(GO_PROJECT) $(script_path)`
    println("Running: ", cmd)
    ok = success(cmd)
    ok || error("validation failed: $script_name")
    return nothing
end

function generate_figures()
    matlab_script = joinpath(SCRIPT_DIR, "plot_go_instability.m")
    matlab_cmd = `matlab -batch "run('$matlab_script')"`
    println("Running: ", matlab_cmd)
    matlab_ok = try
        success(matlab_cmd)
    catch err
        println("MATLAB plotting unavailable: ", sprint(showerror, err))
        false
    end
    matlab_ok && return nothing

    fallback = joinpath(SCRIPT_DIR, "plot_go_instability.jl")
    fallback_cmd = `$(JULIA_EXE) --project=@v1.11 $(fallback)`
    println("MATLAB plotting failed; running Julia fallback: ", fallback_cmd)
    success(fallback_cmd) || error("Julia fallback plotting failed")
    return nothing
end

function main()
    for script in VALIDATION_SCRIPTS
        run_child(script)
    end
    generate_figures()
    println()
    println("Julia validations and figure generation complete.")
    return nothing
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
