#############################
## Triangular kernel sampling/projection.
##
## This file owns the reusable plan, sampling, and coefficient projection
## layer used by the strict-band Volterra construction.
#############################

struct TriangularKernelCoefficients{CFS,REC}
    cfs::CFS
    recurrence::REC
    kerneldepth::Int
    requested_kerneldepth::Int
    flip::Bool
end

struct TriangularKernelPlan{SP,BASIS,GRID,TRANSFORM,REC}
    space::SP
    basis::BASIS
    grid::GRID
    transform::TRANSFORM
    recurrence::REC
    kerneldepth::Int
    requested_kerneldepth::Int
    flip::Bool
end

function _triangular_kernel_blocked_array_type()
    return isdefined(BlockArrays, :BlockedArray) ?
        getfield(BlockArrays, :BlockedArray) :
        getfield(BlockArrays, :PseudoBlockArray)
end

function _triangular_kernel_blocked_coeffs(Kfun, kerneldepth::Integer)
    BlockedArrayCompat = _triangular_kernel_blocked_array_type()
    return BlockedArrayCompat(pad(Kfun, sum(1:Int(kerneldepth))), 1:Int(kerneldepth))
end

function precompute_triangular_kernel_plan(kerneldepth::Integer, flip::Bool)
    kerneldepth < 1 && throw(ArgumentError("kerneldepth must be positive"))

    sp = JacobiTriangle()
    basis = sp[:, Block.(1:kerneldepth)]
    grid, transform = MultivariateOrthogonalPolynomials.ContinuumArrays.plan_grid_transform(
        sp,
        (Block(kerneldepth),),
    )
    recurrence = ClenshawRecurrenceData(sp, Int(kerneldepth) + 2)

    return TriangularKernelPlan(
        sp,
        basis,
        grid,
        transform,
        recurrence,
        Int(kerneldepth),
        Int(kerneldepth),
        flip,
    )
end

function _triangular_kernel_point_grid(grid)
    if grid isa Tuple
        if length(grid) == 1
            return grid[1]
        elseif length(grid) == 2 && size(grid[1]) == size(grid[2])
            return grid
        end
    end
    return grid
end

function _sample_triangular_kernel_value(K::Function, xy, flip::Bool)
    x = xy[1]
    y = xy[2]
    return flip ? K(1 - x, y) : K(x, y)
end

function sample_triangular_kernel_values(plan::TriangularKernelPlan, K::Function)
    point_grid = _triangular_kernel_point_grid(plan.grid)
    if point_grid isa Tuple
        xgrid, ygrid = point_grid
        if plan.flip
            return map((x, y) -> K(1 - x, y), xgrid, ygrid)
        else
            return map(K, xgrid, ygrid)
        end
    else
        return map(xy -> _sample_triangular_kernel_value(K, xy, plan.flip), point_grid)
    end
end

function project_triangular_kernel_coeffs_from_values(plan::TriangularKernelPlan, values)
    point_grid = _triangular_kernel_point_grid(plan.grid)
    expected_size = point_grid isa Tuple ? size(point_grid[1]) : size(point_grid)
    size(values) == expected_size ||
        throw(DimensionMismatch("sample values must have the same size as the plan grid"))

    Kfun = plan.transform * values
    realized_kerneldepth = blocksize(Kfun)[1]
    cfs = _triangular_kernel_blocked_coeffs(Kfun, realized_kerneldepth)

    return TriangularKernelCoefficients(
        cfs,
        plan.recurrence,
        Int(realized_kerneldepth),
        plan.requested_kerneldepth,
        plan.flip,
    )
end

function triangular_kernel_coeffs_from_plan(plan::TriangularKernelPlan, K::Function)
    values = sample_triangular_kernel_values(plan, K)
    return project_triangular_kernel_coeffs_from_values(plan, values)
end

function precompute_triangular_kernel_coeffs(K::Function, kerneldepth::Integer, flip::Bool)
    kerneldepth < 1 && throw(ArgumentError("kerneldepth must be positive"))

    sp = JacobiTriangle()
    xy = axes(sp, 1)
    x, y = first.(xy), last.(xy)
    if flip
        Kfun = sp[:, Block.(1:kerneldepth)] \ (@. K(1 - x, y))
    else
        Kfun = sp[:, Block.(1:kerneldepth)] \ (@. K(x, y))
    end

    realized_kerneldepth = blocksize(Kfun)[1]
    cfs = _triangular_kernel_blocked_coeffs(Kfun, realized_kerneldepth)
    recurrence = ClenshawRecurrenceData(sp, realized_kerneldepth + 2)

    return TriangularKernelCoefficients(cfs, recurrence, Int(realized_kerneldepth),
                                        Int(kerneldepth), flip)
end

