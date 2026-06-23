using Test
using VoltOpt

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))

repo_path(parts...) = joinpath(PROJECT_ROOT, parts...)

@testset "VoltOpt smoke test" begin
    @test isdefined(VoltOpt, :voltmat_cheb)
    @test isdefined(VoltOpt, :voltmat_leg)
    @test isdefined(VoltOpt, :compVolt_nonlinear_fast)
end

@testset "documented layout exists" begin
    for path in (
        "examples/ex1_warmup/ex1_run.jl",
        "examples/ex2_2ndVolterraEqn/ex2_run.jl",
        "examples/ex3_pantograph/ex3_run.jl",
        "examples/ex5_psespectra/ex5_volterra_schur_besselj1.jl",
        "test/smooth_bench/data",
        "test/smooth_bench/legacy",
        "test/weaksingular_bench/data",
        "test/nonlinear_bench/data",
        "tools/audit_layout.jl",
    )
        @test ispath(repo_path(split(path, '/')...))
    end
end
