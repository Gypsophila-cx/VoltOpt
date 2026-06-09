const BENCH_SCRIPTS = [
    joinpath(@__DIR__, "warmup", "bench_gutleb_olver_vs_voltmat.jl"),
    joinpath(@__DIR__, "ex1_2ndVolterraEqn", "bench_gutleb_olver_vs_voltmat.jl"),
    joinpath(@__DIR__, "ex2_1stVolterraEqn", "bench_gutleb_olver_vs_voltmat.jl"),
    joinpath(@__DIR__, "ex3_pantograph", "bench_gutleb_olver_vs_voltmat.jl"),
    joinpath(@__DIR__, "ex4_SPVIDE", "bench_gutleb_olver_vs_voltmat.jl"),
]

for script in BENCH_SCRIPTS
    println("============================================================")
    println("Running ", script)
    cmd = `$(Base.julia_cmd()) --project=$(dirname(@__DIR__)) $script`
    run(cmd)
end
