using GutlebOlver
using Test

@testset "simple second-kind sparse solve" begin
    built = build_simple_secondkind_system(; N=32)
    row = solve_simple_secondkind_sparse(; N=32)

    @test row.raw_operator_type == "BandedMatrix"
    @test row.body_operator_type == "BandedMatrix"
    @test row.final_system_type == "BandedMatrix"
    @test safe_nnz(built.system) == row.final_system_nnz
    @test safe_bandwidth(built.system) == row.full_bandwidth
    @test safe_density(built.system) == row.density
    @test row.rhs_validation_error < 1e-13
    @test row.residual_relative < 1e-13
    @test row.max_abs_error < 1e-13
    @test row.status == "pass"
end
