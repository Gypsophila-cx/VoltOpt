using GutlebOlver
using Test

function _first_grid_sample(plan, values)
    point_grid = GutlebOlver._triangular_kernel_point_grid(plan.grid)
    if point_grid isa Tuple
        xgrid, ygrid = point_grid
        idx = first(eachindex(xgrid))
        return xgrid[idx], ygrid[idx], values[idx]
    else
        idx = first(eachindex(point_grid))
        xy = point_grid[idx]
        return xy[1], xy[2], values[idx]
    end
end

@testset "triangular kernel projection" begin
    for depth in (1, 2, 4)
        plan = precompute_triangular_kernel_plan(depth, false)
        values = sample_triangular_kernel_values(plan, (x, y) -> 1.0)
        coeffs = project_triangular_kernel_coeffs_from_values(plan, values)
        @test coeffs.requested_kerneldepth == depth
        @test coeffs.kerneldepth >= 1
        @test coeffs.flip == false
    end

    plan = precompute_triangular_kernel_plan(3, false)
    values = sample_triangular_kernel_values(plan, (x, y) -> 1.0 + x + y)
    coeffs = triangular_kernel_coeffs_from_plan(plan, (x, y) -> 1.0 + x + y)
    point_grid = GutlebOlver._triangular_kernel_point_grid(plan.grid)
    expected_size = point_grid isa Tuple ? size(point_grid[1]) : size(point_grid)
    @test size(values) == expected_size
    @test coeffs.requested_kerneldepth == 3

    coeffs_exp = triangular_kernel_coeffs_from_plan(
        precompute_triangular_kernel_plan(5, true),
        (x, y) -> y * exp(x^2),
    )
    @test coeffs_exp.flip == true
    @test coeffs_exp.requested_kerneldepth == 5

    plan_no_flip = precompute_triangular_kernel_plan(3, false)
    vals_no_flip = sample_triangular_kernel_values(plan_no_flip, (x, y) -> x + 10y)
    x, y, value = _first_grid_sample(plan_no_flip, vals_no_flip)
    @test value ≈ x + 10y

    plan_flip = precompute_triangular_kernel_plan(3, true)
    vals_flip = sample_triangular_kernel_values(plan_flip, (x, y) -> x + 10y)
    xflip, yflip, value_flip = _first_grid_sample(plan_flip, vals_flip)
    @test value_flip ≈ (1 - xflip) + 10yflip

    c2 = triangular_kernel_coeffs_from_plan(precompute_triangular_kernel_plan(2, true), (x, y) -> y * exp(x^2))
    c5 = triangular_kernel_coeffs_from_plan(precompute_triangular_kernel_plan(5, true), (x, y) -> y * exp(x^2))
    @test c5.requested_kerneldepth > c2.requested_kerneldepth
    @test length(c5.cfs) > length(c2.cfs)
end
