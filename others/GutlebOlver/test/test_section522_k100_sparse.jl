using GutlebOlver
using Test
using TOML

const GOLDEN_SECTION522_K100 =
    TOML.parsefile(joinpath(@__DIR__, "golden_section522_k100.toml"))

golden_bandwidth(key) = Tuple(Int.(GOLDEN_SECTION522_K100[key]))
golden_float(key) = Float64(GOLDEN_SECTION522_K100[key])

function test_golden_float(actual, key)
    @test isapprox(actual, golden_float(key); atol=1e-10, rtol=1e-8)
end

@testset "Section 5.2.2 k=100 sparse solve" begin
    golden = GOLDEN_SECTION522_K100
    N = Int(golden["N"])
    kerneldepth = Int(golden["kerneldepth"])
    k = Float64(golden["k"])

    rhs = build_section522_rhs(; N=N, k=k)
    built = build_section522_operator(; N=N, kerneldepth=kerneldepth)
    row = solve_section522_k100_sparse(; N=N, kerneldepth=kerneldepth, k=k)

    @test rhs.validation_error <= 1e-5
    @test built.raw_stats.type == row.raw_operator_type
    @test built.body_stats.type == row.body_operator_type
    @test built.system_stats.type == row.final_system_type

    @test row.raw_operator_type == "BandedMatrix"
    @test row.body_operator_type == "BandedMatrix"
    @test row.final_system_type == "SparseMatrixCSC"
    @test row.final_system_nnz < row.N * row.N

    @test row.rhs_validation_error <= 1e-5
    @test row.body_residual_relative <= 1e-6
    @test row.max_abs_error <= 1e-6
    @test row.status == "pass"

    @test row.k == k
    @test row.N == N
    @test row.kerneldepth == kerneldepth
    test_golden_float(row.rhs_validation_error, "rhs_validation_error")
    test_golden_float(row.tau_residual, "tau_residual")
    test_golden_float(row.body_residual_relative, "body_residual_relative")
    test_golden_float(row.max_abs_error, "max_abs_error")
    test_golden_float(row.density, "density")

    @test row.raw_operator_type == golden["raw_operator_type"]
    @test row.raw_operator_nnz == Int(golden["raw_operator_nnz"])
    @test row.raw_operator_bandwidth == golden_bandwidth("raw_operator_bandwidth")

    @test row.body_operator_type == golden["body_operator_type"]
    @test row.body_operator_nnz == Int(golden["body_operator_nnz"])
    @test row.body_operator_bandwidth == golden_bandwidth("body_operator_bandwidth")

    @test row.final_system_type == golden["final_system_type"]
    @test row.final_system_nnz == Int(golden["final_system_nnz"])
    @test row.full_bandwidth == golden_bandwidth("final_system_bandwidth")
    @test row.final_system_bandwidth == golden_bandwidth("final_system_bandwidth")

    csv_path = tempname()
    write_results_csv(csv_path, (row,))
    append_result_csv(csv_path, row)
    @test length(readlines(csv_path)) == 3
end
