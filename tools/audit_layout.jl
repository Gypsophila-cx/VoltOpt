#!/usr/bin/env julia

# Static repository layout audit. This script does not run examples,
# benchmarks, plotting code, or any data-generation step.

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))

mutable struct AuditState
    failures::Int
    warnings::Int
end

function pass(msg::AbstractString)
    println("PASS: ", msg)
end

function warn!(state::AuditState, msg::AbstractString)
    state.warnings += 1
    println("WARN: ", msg)
end

function fail!(state::AuditState, msg::AbstractString)
    state.failures += 1
    println("FAIL: ", msg)
end

repo_path(parts...) = joinpath(PROJECT_ROOT, parts...)
rel(path::AbstractString) = replace(relpath(path, PROJECT_ROOT), '\\' => '/')

function check_exists!(state::AuditState, paths, label::AbstractString)
    println("\n[", label, "]")
    for path in paths
        full = repo_path(split(path, '/')...)
        if ispath(full)
            pass(path)
        else
            fail!(state, "missing $(path)")
        end
    end
end

function check_root_artifacts!(state::AuditState, dirs)
    println("\n[root artifact scan]")
    artifact_exts = Set([".csv", ".pdf", ".png", ".mat"])
    found = String[]

    for dir in dirs
        full_dir = repo_path(split(dir, '/')...)
        isdir(full_dir) || continue
        for name in readdir(full_dir)
            full = joinpath(full_dir, name)
            isfile(full) || continue
            _, ext = splitext(name)
            lowercase(ext) in artifact_exts && push!(found, rel(full))
        end
    end

    if isempty(found)
        pass("no root-level CSV/PDF/PNG/MAT artifacts in audited directories")
    else
        for path in found
            fail!(state, "root-level generated artifact: $(path)")
        end
    end
end

function check_old_paths!(state::AuditState)
    println("\n[old path references]")
    forbidden = [
        "examples/ex1_2ndVolterraEqn",
        "examples/ex2_1stVolterraEqn",
        "examples/warmup",
        "examples/figures",
        "ex4_depth_build_time_N200",
    ]

    readme_path = repo_path("README.md")
    if isfile(readme_path)
        text = read(readme_path, String)
        hit = false
        for pattern in forbidden
            if occursin(pattern, text)
                fail!(state, "README.md contains old path reference: $(pattern)")
                hit = true
            end
        end
        !hit && pass("README.md has no audited old path references")
    else
        fail!(state, "missing README.md")
    end

    issue_path = repo_path("issue_list.md")
    if isfile(issue_path)
        text = read(issue_path, String)
        for pattern in forbidden
            occursin(pattern, text) &&
                warn!(state, "issue_list.md contains historical/staging reference: $(pattern)")
        end
    else
        warn!(state, "issue_list.md is missing")
    end
end

function main()
    state = AuditState(0, 0)

    if !(isfile(repo_path("Project.toml")) && isdir(repo_path("src")))
        fail!(state, "script is not running inside the VoltOpt repository")
    else
        pass("repository root: $(PROJECT_ROOT)")
    end

    key_dirs = [
        "examples/ex1_warmup",
        "examples/ex2_2ndVolterraEqn",
        "examples/ex3_pantograph",
        "examples/ex5_psespectra",
        "test/smooth_bench",
        "test/weaksingular_bench",
        "test/compvolt_adaptive",
    ]

    layout_dirs = [
        "examples/ex1_warmup/data",
        "examples/ex1_warmup/figures",
        "examples/ex2_2ndVolterraEqn/data",
        "examples/ex2_2ndVolterraEqn/figures",
        "examples/ex3_pantograph/data",
        "examples/ex3_pantograph/figures",
        "examples/ex5_psespectra/data",
        "examples/ex5_psespectra/figures",
        "test/smooth_bench/data",
        "test/smooth_bench/figures",
        "test/smooth_bench/legacy",
        "test/weaksingular_bench/data",
        "test/weaksingular_bench/figures",
        "test/compvolt_adaptive/data",
        "test/compvolt_adaptive/figures",
        "test/compvolt_adaptive/report",
    ]

    key_scripts = [
        "examples/ex1_warmup/ex1_run.jl",
        "examples/ex1_warmup/ex1_plot.m",
        "examples/ex2_2ndVolterraEqn/ex2_run.jl",
        "examples/ex2_2ndVolterraEqn/ex2_plot.m",
        "examples/ex3_pantograph/ex3_run.jl",
        "examples/ex3_pantograph/ex3_plot.m",
        "examples/ex5_psespectra/ex5_volterra_schur_besselj1.jl",
        "examples/ex5_psespectra/ex5_plot.m",
        "test/smooth_bench/plot_end2end_smooth_bench.m",
        "test/compvolt_adaptive/plot_nonlinear_bench.m",
        "test/compvolt_adaptive/plot_adaptive_oscillation.m",
    ]

    check_exists!(state, key_dirs, "key directories")
    check_exists!(state, layout_dirs, "data/figures/legacy directories")
    check_exists!(state, key_scripts, "key scripts")
    check_root_artifacts!(state, key_dirs)
    check_old_paths!(state)

    println("\n[summary]")
    println("warnings: ", state.warnings)
    println("failures: ", state.failures)

    if state.failures == 0
        println("PASS: layout audit completed")
        exit(0)
    else
        println("FAIL: layout audit found blocking issues")
        exit(1)
    end
end

main()
