module PantographPeriod

using LinearAlgebra
using Printf
using SparseArrays
using VoltOpt

export PantographOperator,
    PantographConvergenceError,
    MAX_SOLVE_DEGREE,
    ultraspherical_operators,
    build_scaled_pantograph_operator,
    solve_scaled_pantograph,
    solve_scaled_pantograph_adaptive,
    prepare_figure2_solver,
    solve_prepared_figure2,
    solve_figure2_pantograph,
    evaluate_pantograph,
    scaled_equation_residual,
    chebyshev_evaluate,
    chebyshev_derivative_coefficients,
    initial_condition_errors,
    coefficient_tail_estimate,
    restriction_derivative_crosscheck,
    default_initial_degree,
    validation_grid

const MAX_SOLVE_DEGREE = 5000

"""
Coefficient-space representation of

    D2 + (omega2*T^2/4)S1*S0
       - (gamma*T/(2q))S1*D1*R,

where `R = VoltOpt.restmat_cheb(-1, 2q-1, N)`.  The stored matrix is upper
triangular and complex from assembly onward.  RestMat currently returns the
restriction as a dense real matrix; it is used directly without an
additional dense copy.
"""
struct PantographOperator
    N::Int
    T::Float64
    omega2::Float64
    gamma::ComplexF64
    q::Float64
    delay_scale::ComplexF64
    matrix::UpperTriangular{ComplexF64,Matrix{ComplexF64}}
    restriction::Matrix{Float64}
    local_operator::SparseMatrixCSC{Float64,Int64}
    delay_operator::SparseMatrixCSC{Float64,Int64}
end

struct PantographConvergenceError <: Exception
    message::String
    result::NamedTuple
end

Base.showerror(io::IO, err::PantographConvergenceError) = print(io, err.message)

"""
    ultraspherical_operators(N)

Return sparse coefficient-space operators for degree `N`:

* `D1`: Chebyshev -> C^(1), with `(D1)[n,n+1] = n`;
* `D2`: Chebyshev -> C^(2), with `(D2)[n-1,n+1] = 2n`;
* `S0`: Chebyshev -> C^(1);
* `S1`: C^(1) -> C^(2).

All matrices have size `(N+1, N+1)` and use Julia's one-based indexing.
"""
function ultraspherical_operators(N::Integer)
    N >= 2 || throw(ArgumentError("N must be at least 2"))
    N <= MAX_SOLVE_DEGREE || throw(ArgumentError(
        "N=$N exceeds the configured hard limit N=$(MAX_SOLVE_DEGREE).",
    ))
    ndegree = Int(N)
    m = ndegree + 1

    D1 = spzeros(Float64, m, m)
    D2 = spzeros(Float64, m, m)
    S0 = spzeros(Float64, m, m)
    S1 = spzeros(Float64, m, m)

    @inbounds for n in 1:ndegree
        D1[n, n + 1] = Float64(n)
    end
    @inbounds for n in 2:ndegree
        D2[n - 1, n + 1] = 2.0 * n
    end

    S0[1, 1] = 1.0
    @inbounds for n in 1:ndegree
        S0[n + 1, n + 1] = 0.5
        if n >= 2
            S0[n - 1, n + 1] = -0.5
        end
    end

    @inbounds for n in 0:ndegree
        scale = 1.0 / (n + 1.0)
        S1[n + 1, n + 1] = scale
        if n >= 2
            S1[n - 1, n + 1] = -scale
        end
    end

    return (; D1, D2, S0, S1)
end

"""
    build_scaled_pantograph_operator(
        N, T; omega2, gamma, q, check_restriction=false
    )

Assemble the ultraspherical coefficient-space operator for

    y''(t) + omega2*y(t) = gamma*y'(q*t).

`VoltOpt.restmat_cheb(-1.0, 2q-1, N)` is called so that `R*c` represents
`u(q*x + q - 1)`.  All terms are assembled in `C^(2)` space as

    D2 + (omega2*T^2/4)S1*S0
       - (gamma*T/(2q))S1*D1*R.

No collocation differentiation matrix or time-stepping method is used.
"""
function build_scaled_pantograph_operator(
    N::Integer,
    T::Real;
    omega2::Real,
    gamma::Number,
    q::Real,
    check_restriction::Union{Bool,Real}=false,
)
    N >= 2 || throw(ArgumentError("N must be at least 2"))
    N <= MAX_SOLVE_DEGREE || throw(ArgumentError(
        "N=$N exceeds the configured hard limit N=$(MAX_SOLVE_DEGREE).",
    ))
    isfinite(T) && T > 0 || throw(ArgumentError("T must be finite and positive"))
    isfinite(omega2) || throw(ArgumentError("omega2 must be finite"))
    isfinite(real(gamma)) && isfinite(imag(gamma)) ||
        throw(ArgumentError("gamma must be finite"))
    isfinite(q) && 0 < q <= 1 ||
        throw(ArgumentError("q must satisfy 0 < q <= 1"))
    ndegree = Int(N)
    Tf = Float64(T)
    omega2f = Float64(omega2)
    gammaf = ComplexF64(gamma)
    qf = Float64(q)
    delay_scale = -gammaf * Tf / (2.0 * qf)
    m = ndegree + 1

    ops = ultraspherical_operators(ndegree)
    R = VoltOpt.restmat_cheb(
        -1.0,
        2.0qf - 1.0,
        ndegree;
        T=Float64,
        check=check_restriction,
    )

    # This is the only dense operator allocation.  mul! evaluates the exact
    # sparse coefficient-space product S1*D1*R directly into ComplexF64.
    L = zeros(ComplexF64, m, m)
    delay_operator = ops.S1 * ops.D1
    mul!(
        L,
        delay_operator,
        R,
        delay_scale,
        ComplexF64(0.0),
    )

    local_operator =
        ops.D2 + (omega2f * Tf^2 / 4.0) * (ops.S1 * ops.S0)
    rows, cols, vals = findnz(local_operator)
    @inbounds for k in eachindex(vals)
        L[rows[k], cols[k]] += ComplexF64(vals[k])
    end

    return PantographOperator(
        ndegree,
        Tf,
        omega2f,
        gammaf,
        qf,
        delay_scale,
        UpperTriangular(L),
        R,
        local_operator,
        delay_operator,
    )
end

function _left_value_row(N::Int)
    row = Vector{ComplexF64}(undef, N + 1)
    @inbounds for n in 0:N
        row[n + 1] = iseven(n) ? 1.0 + 0.0im : -1.0 + 0.0im
    end
    return row
end

function _left_derivative_row(N::Int)
    row = zeros(ComplexF64, N + 1)
    @inbounds for n in 1:N
        sign = isodd(n) ? 1.0 : -1.0
        row[n + 1] = ComplexF64(sign * n^2)
    end
    return row
end

"""
Solve with tau bordering.  The two complex boundary rows

    sum (-1)^n c_n = y0,
    sum n^2 (-1)^(n-1) c_n = (T/2)dy0

are stacked above the first `N-1` rows of the `C^(2)` equation.  The bordered
matrix has lower bandwidth two and a dense upper triangle.  A restricted
banded Givens QR solves that structure in `O(N^2)` work without copying it or
invoking a cubic-work general dense factorization.
"""
function _factor_lower_banded_qr!(
    A::Matrix{ComplexF64},
    lower_bandwidth::Int,
)
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("A must be square"))
    rotation_upper_rows = Int[]
    rotation_cosines = Float64[]
    rotation_sines = ComplexF64[]
    sizehint!(rotation_upper_rows, lower_bandwidth * n)
    sizehint!(rotation_cosines, lower_bandwidth * n)
    sizehint!(rotation_sines, lower_bandwidth * n)

    @inbounds for k in 1:(n - 1)
        for lower_row in min(n, k + lower_bandwidth):-1:(k + 1)
            upper_row = lower_row - 1
            b = A[lower_row, k]
            iszero(b) && continue
            a = A[upper_row, k]
            radius = hypot(abs(a), abs(b))
            isfinite(radius) && radius > 0.0 ||
                throw(LinearAlgebra.SingularException(k))
            if iszero(a)
                cosine = 0.0
                sine = conj(b) / abs(b)
            else
                cosine = abs(a) / radius
                sine = (a / abs(a)) * conj(b) / radius
            end

            for j in k:n
                upper_value = A[upper_row, j]
                lower_value = A[lower_row, j]
                A[upper_row, j] =
                    cosine * upper_value + sine * lower_value
                A[lower_row, j] =
                    -conj(sine) * upper_value + cosine * lower_value
            end
            A[lower_row, k] = 0.0 + 0.0im
            push!(rotation_upper_rows, upper_row)
            push!(rotation_cosines, cosine)
            push!(rotation_sines, sine)
        end
    end

    for i in 1:n
        diagonal = abs(A[i, i])
        isfinite(diagonal) && diagonal > 0.0 ||
            throw(LinearAlgebra.SingularException(i))
    end
    return (
        upper_rows=rotation_upper_rows,
        cosines=rotation_cosines,
        sines=rotation_sines,
    )
end

function _solve_lower_banded_qr!(
    A::Matrix{ComplexF64},
    rotations,
    rhs::Vector{ComplexF64},
)
    n = size(A, 1)
    length(rhs) == n || throw(DimensionMismatch("rhs length must match A"))
    @inbounds for k in eachindex(rotations.upper_rows)
        upper_row = rotations.upper_rows[k]
        lower_row = upper_row + 1
        cosine = rotations.cosines[k]
        sine = rotations.sines[k]
        upper_value = rhs[upper_row]
        lower_value = rhs[lower_row]
        rhs[upper_row] = cosine * upper_value + sine * lower_value
        rhs[lower_row] =
            -conj(sine) * upper_value + cosine * lower_value
    end

    @inbounds for i in n:-1:1
        value = rhs[i]
        for j in (i + 1):n
            value -= A[i, j] * rhs[j]
        end
        rhs[i] = value / A[i, i]
    end
    return rhs
end

function _compensated_dot(
    row::AbstractVector{ComplexF64},
    coefficients::AbstractVector{ComplexF64},
)
    total = 0.0 + 0.0im
    correction = 0.0 + 0.0im
    @inbounds for i in eachindex(row, coefficients)
        term = row[i] * coefficients[i] - correction
        updated = total + term
        correction = (updated - total) - term
        total = updated
    end
    return total
end

function _tau_residual!(
    residual::Vector{ComplexF64},
    operator::PantographOperator,
    coefficients::Vector{ComplexF64},
    value_row::Vector{ComplexF64},
    derivative_row::Vector{ComplexF64},
    boundary_rhs::Vector{ComplexF64},
)
    residual[1] =
        boundary_rhs[1] - _compensated_dot(value_row, coefficients)
    residual[2] =
        boundary_rhs[2] - _compensated_dot(derivative_row, coefficients)

    restricted = operator.restriction * coefficients
    equation_modes =
        operator.local_operator * coefficients +
        operator.delay_scale *
        (operator.delay_operator * restricted)
    residual[3:end] .= -(@view equation_modes[1:(length(coefficients) - 2)])
    return residual
end

function _prepare_tau_factorization!(operator::PantographOperator)
    ndegree = operator.N
    m = ndegree + 1
    L = parent(operator.matrix)
    value_row = _left_value_row(ndegree)
    derivative_row = _left_derivative_row(ndegree)

    # Move retained equation rows down in-place, iterating backwards so no
    # source row is overwritten before it is copied.
    @inbounds for destination in m:-1:3
        source = destination - 2
        for j in 1:m
            L[destination, j] = L[source, j]
        end
    end
    L[1, :] .= value_row
    L[2, :] .= derivative_row
    rotations = _factor_lower_banded_qr!(L, 2)
    return (; L, rotations, value_row, derivative_row)
end

function _solve_tau_factored(
    operator::PantographOperator,
    factorization,
    y0::Number,
    dy0::Number,
)
    m = operator.N + 1
    boundary_rhs = ComplexF64[
        ComplexF64(y0),
        ComplexF64(operator.T / 2.0) * ComplexF64(dy0),
    ]
    rhs = zeros(ComplexF64, m)
    rhs[1:2] .= boundary_rhs
    coefficients = _solve_lower_banded_qr!(
        factorization.L,
        factorization.rotations,
        rhs,
    )

    # Same-precision refinement evaluates the residual from RestMat and the
    # sparse ultraspherical factors, not from overwritten QR storage.
    tau_residual = zeros(ComplexF64, m)
    previous_residual = Inf
    refinement_steps = 0
    for step in 1:3
        _tau_residual!(
            tau_residual,
            operator,
            coefficients,
            factorization.value_row,
            factorization.derivative_row,
            boundary_rhs,
        )
        residual_norm = maximum(abs, tau_residual; init=0.0)
        if !isfinite(residual_norm) || residual_norm >= previous_residual
            break
        end
        correction = _solve_lower_banded_qr!(
            factorization.L,
            factorization.rotations,
            copy(tau_residual),
        )
        coefficients .+= correction
        previous_residual = residual_norm
        refinement_steps = step
    end
    _tau_residual!(
        tau_residual,
        operator,
        coefficients,
        factorization.value_row,
        factorization.derivative_row,
        boundary_rhs,
    )

    value_defect =
        _compensated_dot(factorization.value_row, coefficients) -
        boundary_rhs[1]
    derivative_defect =
        _compensated_dot(factorization.derivative_row, coefficients) -
        boundary_rhs[2]
    return (
        coefficients=coefficients,
        y0=ComplexF64(y0),
        dy0=ComplexF64(dy0),
        boundary_residual=max(abs(value_defect), abs(derivative_defect)),
        tau_residual=maximum(abs, tau_residual; init=0.0),
        iterative_refinement_steps=refinement_steps,
    )
end

function _solve_scaled_pantograph_fixed(
    N::Integer,
    T::Real;
    y0::Number,
    dy0::Number,
    omega2::Real,
    gamma::Number,
    q::Real,
    return_info::Bool=false,
    check_restriction::Union{Bool,Real}=false,
)
    started = time_ns()
    assembly_started = time_ns()
    operator = build_scaled_pantograph_operator(
        N,
        T;
        omega2=omega2,
        gamma=gamma,
        q=q,
        check_restriction=check_restriction,
    )
    assembly_time = (time_ns() - assembly_started) * 1.0e-9

    solve_started = time_ns()
    factorization = _prepare_tau_factorization!(operator)
    solution = _solve_tau_factored(operator, factorization, y0, dy0)
    linear_solve_time = (time_ns() - solve_started) * 1.0e-9
    total_time = (time_ns() - started) * 1.0e-9

    info = (
        coefficients=solution.coefficients,
        N=operator.N,
        T=Float64(T),
        omega2=operator.omega2,
        gamma=operator.gamma,
        q=operator.q,
        y0=solution.y0,
        dy0=solution.dy0,
        assembly_time=assembly_time,
        linear_solve_time=linear_solve_time,
        solve_time=total_time,
        boundary_residual=solution.boundary_residual,
        tau_residual=solution.tau_residual,
        iterative_refinement_steps=solution.iterative_refinement_steps,
        operator_storage=:dense_upper_triangular,
        operator_eltype=eltype(factorization.L),
    )
    return return_info ? info : solution.coefficients
end

function _solve_scaled_pantograph_pair(
    N::Integer,
    T::Real;
    omega2::Real,
    gamma::Number,
    q::Real,
    check_restriction::Union{Bool,Real}=false,
)
    started = time_ns()
    assembly_started = time_ns()
    operator = build_scaled_pantograph_operator(
        N,
        T;
        omega2=omega2,
        gamma=gamma,
        q=q,
        check_restriction=check_restriction,
    )
    assembly_time = (time_ns() - assembly_started) * 1.0e-9

    factor_started = time_ns()
    factorization = _prepare_tau_factorization!(operator)
    factorization_time = (time_ns() - factor_started) * 1.0e-9
    rhs_started = time_ns()
    phi0 = _solve_tau_factored(operator, factorization, 1.0, 0.0)
    phi1 = _solve_tau_factored(operator, factorization, 0.0, 1.0)
    rhs_solve_time = (time_ns() - rhs_started) * 1.0e-9
    total_time = (time_ns() - started) * 1.0e-9

    info = (
        c_phi0=phi0.coefficients,
        c_phi1=phi1.coefficients,
        N=operator.N,
        T=operator.T,
        omega2=operator.omega2,
        gamma=operator.gamma,
        q=operator.q,
        phi0=phi0,
        phi1=phi1,
        assembly_time=assembly_time,
        factorization_time=factorization_time,
        rhs_solve_time=rhs_solve_time,
        solve_time=total_time,
        factorization_count=1,
        rhs_count=2,
        operator_storage=:dense_upper_triangular,
        operator_eltype=eltype(factorization.L),
    )
    return info
end

"""
Evaluate a Chebyshev series by Clenshaw recurrence.
"""
function chebyshev_evaluate(coefficients::AbstractVector, x::Real)
    isempty(coefficients) && return zero(eltype(coefficients))
    length(coefficients) == 1 && return coefficients[1]
    Tresult = promote_type(eltype(coefficients), typeof(x))
    b1 = zero(Tresult)
    b2 = zero(Tresult)
    two_x = 2.0 * x
    @inbounds for k in length(coefficients):-1:2
        b0 = coefficients[k] + two_x * b1 - b2
        b2 = b1
        b1 = b0
    end
    return coefficients[1] + x * b1 - b2
end

"""
Return same-length Chebyshev coefficients of the derivative with respect to
the standard variable `x`.  The final coefficient is exactly zero.
"""
function chebyshev_derivative_coefficients(coefficients::AbstractVector)
    m = length(coefficients)
    Tcoeff = eltype(coefficients)
    derivative = zeros(Tcoeff, m)
    m <= 1 && return derivative

    N = m - 1
    derivative[N] = 2.0 * N * coefficients[N + 1]
    if N >= 2
        derivative[N - 1] = 2.0 * (N - 1) * coefficients[N]
    end
    @inbounds for k in (N - 3):-1:0
        derivative[k + 1] =
            derivative[k + 3] + 2.0 * (k + 1) * coefficients[k + 2]
    end
    derivative[1] *= 0.5
    return derivative
end

function evaluate_pantograph(
    coefficients::AbstractVector,
    T::Real,
    tgrid::AbstractVector{<:Real},
)
    Tf = Float64(T)
    Tf > 0 || throw(ArgumentError("T must be positive"))
    values = Vector{ComplexF64}(undef, length(tgrid))
    @inbounds for j in eachindex(tgrid)
        x = 2.0 * Float64(tgrid[j]) / Tf - 1.0
        values[j] = ComplexF64(chebyshev_evaluate(coefficients, x))
    end
    return values
end

"""
    scaled_equation_residual(c, T, tgrid; omega2, gamma, q,
                             return_details=false)

Independently evaluate the physical residual

    y''(t) + omega2*y(t) - gamma*y'(q*t)

from differentiated Chebyshev coefficients.  The evaluation points are
independent of the coefficient-space solve.
"""
function scaled_equation_residual(
    coefficients::AbstractVector,
    T::Real,
    tgrid::AbstractVector{<:Real};
    omega2::Real,
    gamma::Number,
    q::Real,
    return_details::Bool=false,
)
    Tf = Float64(T)
    omega2f = Float64(omega2)
    gammaf = ComplexF64(gamma)
    qf = Float64(q)
    isfinite(Tf) && Tf > 0 || throw(ArgumentError("T must be positive"))
    isfinite(qf) && 0 < qf <= 1 ||
        throw(ArgumentError("q must satisfy 0 < q <= 1"))
    dx = chebyshev_derivative_coefficients(coefficients)
    dxx = chebyshev_derivative_coefficients(dx)
    residual = Vector{ComplexF64}(undef, length(tgrid))
    delayed_derivative = Vector{ComplexF64}(undef, length(tgrid))
    max_ytt = 0.0
    max_y = 0.0
    max_delayed_derivative = 0.0

    @inbounds for j in eachindex(tgrid)
        x = 2.0 * Float64(tgrid[j]) / Tf - 1.0
        xq = qf * x + (qf - 1.0)
        y = chebyshev_evaluate(coefficients, x)
        ytt = (4.0 / Tf^2) * chebyshev_evaluate(dxx, x)
        delayed =
            (2.0 / Tf) * chebyshev_evaluate(dx, xq)
        delayed_derivative[j] = ComplexF64(delayed)
        residual[j] =
            ComplexF64(ytt + omega2f * y - gammaf * delayed)
        max_ytt = max(max_ytt, abs(ytt))
        max_y = max(max_y, abs(y))
        max_delayed_derivative =
            max(max_delayed_derivative, abs(delayed))
    end

    if return_details
        max_residual = maximum(abs, residual; init=0.0)
        residual_scale = max(
            1.0,
            max_ytt +
            abs(omega2f) * max_y +
            abs(gammaf) * max_delayed_derivative,
        )
        return (
            residual=residual,
            delayed_derivative=delayed_derivative,
            max_residual=max_residual,
            absolute_residual=max_residual,
            relative_residual=max_residual / residual_scale,
            residual_scale=residual_scale,
            max_ytt=max_ytt,
            max_y=max_y,
            max_delayed_derivative=max_delayed_derivative,
        )
    end
    return residual
end

function initial_condition_errors(
    coefficients::AbstractVector,
    T::Real;
    y0::Number=1.0 + 0.0im,
    dy0::Number=0.0 + 1.0im,
)
    Tf = Float64(T)
    complex_coefficients =
        coefficients isa Vector{ComplexF64} ?
        coefficients :
        ComplexF64.(coefficients)
    N = length(complex_coefficients) - 1
    y_at_zero = _compensated_dot(
        _left_value_row(N),
        complex_coefficients,
    )
    dy_at_zero = (2.0 / Tf) * _compensated_dot(
        _left_derivative_row(N),
        complex_coefficients,
    )
    return (
        value=abs(y_at_zero - ComplexF64(y0)),
        derivative=abs(dy_at_zero - ComplexF64(dy0)),
        y_at_zero=ComplexF64(y_at_zero),
        dy_at_zero=ComplexF64(dy_at_zero),
    )
end

function coefficient_tail_estimate(
    coefficients::AbstractVector;
    tail_length::Integer=32,
)
    isempty(coefficients) && return 0.0
    count = min(length(coefficients), max(1, Int(tail_length)))
    first_tail = length(coefficients) - count + 1
    return maximum(abs, @view coefficients[first_tail:end]; init=0.0)
end

"""
Cross-check `u'(q*x+q-1)` by direct differentiation and by differentiating
the RestMat-restricted Chebyshev series.
"""
function restriction_derivative_crosscheck(
    coefficients::AbstractVector,
    xgrid::AbstractVector{<:Real};
    q::Real=1.0 / 6.0,
)
    qf = Float64(q)
    0.0 < qf <= 1.0 || throw(ArgumentError("q must satisfy 0 < q <= 1"))
    N = length(coefficients) - 1
    R = VoltOpt.restmat_cheb(-1.0, 2.0qf - 1.0, N; T=Float64)
    restricted = R * coefficients
    derivative = chebyshev_derivative_coefficients(coefficients)
    restricted_derivative =
        chebyshev_derivative_coefficients(restricted)
    error_max = 0.0
    @inbounds for x in xgrid
        xq = qf * Float64(x) + qf - 1.0
        direct = chebyshev_evaluate(derivative, xq)
        via_restriction =
            chebyshev_evaluate(restricted_derivative, Float64(x)) / qf
        error_max = max(error_max, abs(direct - via_restriction))
    end
    return error_max
end

function validation_grid(T::Real; points_per_period::Integer=32)
    points_per_period >= 8 ||
        throw(ArgumentError("points_per_period must be at least 8"))
    periods = max(1, ceil(Int, Float64(T) / (2pi)))
    count = Int(points_per_period) * periods + 1
    return collect(range(0.0, Float64(T); length=count))
end

_round_degree(N::Integer; multiple::Integer=16) =
    max(2, Int(multiple) * cld(Int(N), Int(multiple)))

function default_initial_degree(T::Real)
    halfspan = Float64(T) / 2.0
    estimate =
        ceil(Int, halfspan + 12.0 * cbrt(max(halfspan, 1.0)) + 32.0)
    return _round_degree(estimate)
end

function _cauchy_error(
    coarse_coefficients::AbstractVector,
    fine_coefficients::AbstractVector,
    T::Real,
    grid::AbstractVector{<:Real},
)
    error_max = 0.0
    @inbounds for t in grid
        x = 2.0 * Float64(t) / Float64(T) - 1.0
        coarse = chebyshev_evaluate(coarse_coefficients, x)
        fine = chebyshev_evaluate(fine_coefficients, x)
        error_max = max(error_max, abs(coarse - fine))
    end
    return error_max
end

function _scaled_solution_diagnostics(
    coefficients::AbstractVector,
    T::Real,
    y0::Number,
    dy0::Number,
    grid::AbstractVector{<:Real};
    omega2::Real,
    gamma::Number,
    q::Real,
    cauchy_error,
    cauchy_tol::Real,
    residual_tol::Real,
    relative_residual_tol::Real,
    initial_tol::Real,
    tail_tol::Real,
)
    residual = scaled_equation_residual(
        coefficients,
        T,
        grid;
        omega2=omega2,
        gamma=gamma,
        q=q,
        return_details=true,
    )
    initial = initial_condition_errors(
        coefficients,
        T;
        y0=y0,
        dy0=dy0,
    )
    tail = coefficient_tail_estimate(coefficients)
    cauchy_ok = cauchy_error === nothing || cauchy_error <= cauchy_tol
    converged =
        cauchy_ok &&
        residual.absolute_residual <= residual_tol &&
        residual.relative_residual <= relative_residual_tol &&
        initial.value <= initial_tol &&
        initial.derivative <= initial_tol &&
        tail <= tail_tol
    return (
        converged=converged,
        equation_residual=residual.absolute_residual,
        absolute_residual=residual.absolute_residual,
        relative_residual=residual.relative_residual,
        initial_value_error=initial.value,
        initial_derivative_error=initial.derivative,
        coefficient_tail=tail,
        tail_estimate=tail,
        cauchy_error=cauchy_error,
    )
end

function _adaptive_initial_degree(
    T::Real,
    N0::Union{Nothing,Integer},
    Nmax::Integer,
)
    Nmax <= MAX_SOLVE_DEGREE || throw(ArgumentError(
        "Nmax=$Nmax exceeds the hard solve limit N=$(MAX_SOLVE_DEGREE).",
    ))
    Nmax >= 4 || throw(ArgumentError("Nmax must be at least 4"))
    requested =
        N0 === nothing ? default_initial_degree(T) : _round_degree(Int(N0))
    largest_doubled_coarse =
        max(2, 16 * fld(div(Int(Nmax), 2), 16))
    return min(requested, largest_doubled_coarse)
end

function _prepare_scaled_pantograph_pair_adaptive(
    T::Real;
    omega2::Real,
    gamma::Number,
    q::Real,
    N0::Union{Nothing,Integer}=nothing,
    Nmax::Integer=MAX_SOLVE_DEGREE,
    cauchy_tol::Real=1.0e-10,
    residual_tol::Real=1.0e-9,
    relative_residual_tol::Real=Inf,
    initial_tol::Real=1.0e-11,
    tail_tol::Real=1.0e-12,
    validation_points_per_period::Integer=32,
    max_refinements::Integer=4,
    verbose::Bool=true,
    throw_on_failure::Bool=true,
)
    Tf = Float64(T)
    coarse_N = _adaptive_initial_degree(Tf, N0, Nmax)
    max_refinements >= 1 ||
        throw(ArgumentError("max_refinements must be positive"))
    grid = validation_grid(
        Tf;
        points_per_period=validation_points_per_period,
    )
    coarse = _solve_scaled_pantograph_pair(
        coarse_N,
        Tf;
        omega2=omega2,
        gamma=gamma,
        q=q,
    )
    total_solve_time = coarse.solve_time
    final_result = nothing

    for refinement in 1:Int(max_refinements)
        fine_N = min(2 * coarse_N, Int(Nmax))
        fine_N > coarse_N || break
        verbose && @printf(
            "T=%.16g fundamentals: comparing N=%d with N=%d\n",
            Tf,
            coarse_N,
            fine_N,
        )
        fine = _solve_scaled_pantograph_pair(
            fine_N,
            Tf;
            omega2=omega2,
            gamma=gamma,
            q=q,
        )
        total_solve_time += fine.solve_time
        phi0_cauchy =
            _cauchy_error(coarse.c_phi0, fine.c_phi0, Tf, grid)
        phi1_cauchy =
            _cauchy_error(coarse.c_phi1, fine.c_phi1, Tf, grid)
        phi0 = _scaled_solution_diagnostics(
            fine.c_phi0,
            Tf,
            1.0,
            0.0,
            grid;
            omega2=omega2,
            gamma=gamma,
            q=q,
            cauchy_error=phi0_cauchy,
            cauchy_tol=cauchy_tol,
            residual_tol=residual_tol,
            relative_residual_tol=relative_residual_tol,
            initial_tol=initial_tol,
            tail_tol=tail_tol,
        )
        phi1 = _scaled_solution_diagnostics(
            fine.c_phi1,
            Tf,
            0.0,
            1.0,
            grid;
            omega2=omega2,
            gamma=gamma,
            q=q,
            cauchy_error=phi1_cauchy,
            cauchy_tol=cauchy_tol,
            residual_tol=residual_tol,
            relative_residual_tol=relative_residual_tol,
            initial_tol=initial_tol,
            tail_tol=tail_tol,
        )
        final_result = (
            c_phi0=fine.c_phi0,
            c_phi1=fine.c_phi1,
            coefficients_phi0=fine.c_phi0,
            coefficients_phi1=fine.c_phi1,
            coarse_coefficients_phi0=coarse.c_phi0,
            coarse_coefficients_phi1=coarse.c_phi1,
            converged=phi0.converged && phi1.converged,
            T=Tf,
            omega2=Float64(omega2),
            gamma=ComplexF64(gamma),
            q=Float64(q),
            N=fine_N,
            coarse_N=coarse_N,
            refinement=refinement,
            validation_grid=grid,
            number_of_validation_points=length(grid),
            phi0=phi0,
            phi1=phi1,
            solve_time=total_solve_time,
            final_level_solve_time=fine.solve_time,
            final_assembly_time=fine.assembly_time,
            final_factorization_time=fine.factorization_time,
            final_rhs_solve_time=fine.rhs_solve_time,
            factorization_count_per_level=1,
            operator_storage=fine.operator_storage,
            operator_eltype=fine.operator_eltype,
            tolerances=(
                cauchy=Float64(cauchy_tol),
                residual=Float64(residual_tol),
                relative_residual=Float64(relative_residual_tol),
                initial=Float64(initial_tol),
                tail=Float64(tail_tol),
            ),
        )
        verbose && @printf(
            "  phi0 residual=%.3e relative=%.3e Cauchy=%.3e; phi1 residual=%.3e relative=%.3e Cauchy=%.3e; converged=%s\n",
            phi0.absolute_residual,
            phi0.relative_residual,
            phi0.cauchy_error,
            phi1.absolute_residual,
            phi1.relative_residual,
            phi1.cauchy_error,
            string(final_result.converged),
        )
        final_result.converged && return final_result
        fine_N == Nmax && break
        coarse_N = fine_N
        coarse = fine
        GC.gc()
    end

    final_result === nothing && error(
        "Adaptive fundamental solve could not form a fine comparison.",
    )
    message = @sprintf(
        "Fundamental solutions failed at T=%.16g, N=%d/%d: phi0 residual=%.3e relative=%.3e Cauchy=%.3e; phi1 residual=%.3e relative=%.3e Cauchy=%.3e; Nmax=%d.",
        Tf,
        final_result.coarse_N,
        final_result.N,
        final_result.phi0.absolute_residual,
        final_result.phi0.relative_residual,
        final_result.phi0.cauchy_error,
        final_result.phi1.absolute_residual,
        final_result.phi1.relative_residual,
        final_result.phi1.cauchy_error,
        Nmax,
    )
    throw_on_failure && throw(PantographConvergenceError(message, final_result))
    return merge(final_result, (failure_message=message,))
end

"""
    solve_scaled_pantograph_adaptive(y0, dy0;
        omega2, gamma, q, T=100.0, ...)

Adaptively solve `y''+omega2*y=gamma*y'(q*t)` in Chebyshev/ultraspherical
coefficient space.  Degree pairs are accepted only after Cauchy, independent
absolute and relative residual, initial-condition, and coefficient-tail
checks.
"""
function solve_scaled_pantograph_adaptive(
    y0::Number,
    dy0::Number;
    omega2::Real,
    gamma::Number,
    q::Real,
    T::Real=100.0,
    N0::Union{Nothing,Integer}=nothing,
    Nmax::Integer=MAX_SOLVE_DEGREE,
    cauchy_tol::Real=1.0e-10,
    residual_tol::Real=1.0e-9,
    relative_residual_tol::Real=Inf,
    initial_tol::Real=1.0e-11,
    tail_tol::Real=1.0e-12,
    validation_points_per_period::Integer=32,
    max_refinements::Integer=4,
    verbose::Bool=true,
    throw_on_failure::Bool=true,
)
    Tf = Float64(T)
    coarse_N = _adaptive_initial_degree(Tf, N0, Nmax)
    grid = validation_grid(
        Tf;
        points_per_period=validation_points_per_period,
    )
    coarse = _solve_scaled_pantograph_fixed(
        coarse_N,
        Tf;
        y0=y0,
        dy0=dy0,
        omega2=omega2,
        gamma=gamma,
        q=q,
        return_info=true,
    )
    total_solve_time = coarse.solve_time
    final_result = nothing
    for refinement in 1:Int(max_refinements)
        fine_N = min(2 * coarse_N, Int(Nmax))
        fine_N > coarse_N || break
        verbose && @printf(
            "T=%.16g: comparing N=%d with N=%d\n",
            Tf,
            coarse_N,
            fine_N,
        )
        fine = _solve_scaled_pantograph_fixed(
            fine_N,
            Tf;
            y0=y0,
            dy0=dy0,
            omega2=omega2,
            gamma=gamma,
            q=q,
            return_info=true,
        )
        total_solve_time += fine.solve_time
        cauchy = _cauchy_error(
            coarse.coefficients,
            fine.coefficients,
            Tf,
            grid,
        )
        diagnostics = _scaled_solution_diagnostics(
            fine.coefficients,
            Tf,
            y0,
            dy0,
            grid;
            omega2=omega2,
            gamma=gamma,
            q=q,
            cauchy_error=cauchy,
            cauchy_tol=cauchy_tol,
            residual_tol=residual_tol,
            relative_residual_tol=relative_residual_tol,
            initial_tol=initial_tol,
            tail_tol=tail_tol,
        )
        final_result = merge(
            diagnostics,
            (
                coefficients=fine.coefficients,
                T=Tf,
                omega2=Float64(omega2),
                gamma=ComplexF64(gamma),
                q=Float64(q),
                y0=ComplexF64(y0),
                dy0=ComplexF64(dy0),
                N=fine_N,
                coarse_N=coarse_N,
                refinement=refinement,
                number_of_validation_points=length(grid),
                solve_time=total_solve_time,
                final_level_solve_time=fine.solve_time,
                final_assembly_time=fine.assembly_time,
                final_linear_solve_time=fine.linear_solve_time,
                boundary_residual=fine.boundary_residual,
                operator_storage=fine.operator_storage,
                operator_eltype=fine.operator_eltype,
                tolerances=(
                    cauchy=Float64(cauchy_tol),
                    residual=Float64(residual_tol),
                    relative_residual=Float64(relative_residual_tol),
                    initial=Float64(initial_tol),
                    tail=Float64(tail_tol),
                ),
            ),
        )
        verbose && @printf(
            "  residual=%.3e relative=%.3e IC=(%.3e, %.3e) tail=%.3e Cauchy=%.3e converged=%s\n",
            diagnostics.absolute_residual,
            diagnostics.relative_residual,
            diagnostics.initial_value_error,
            diagnostics.initial_derivative_error,
            diagnostics.coefficient_tail,
            diagnostics.cauchy_error,
            string(diagnostics.converged),
        )
        diagnostics.converged && return final_result
        fine_N == Nmax && break
        coarse_N = fine_N
        coarse = fine
        GC.gc()
    end
    final_result === nothing && error(
        "Adaptive solve could not form a fine comparison.",
    )
    message = @sprintf(
        "Proportional-delay solve failed at T=%.16g, N=%d/%d: residual=%.3e relative=%.3e IC=(%.3e, %.3e) tail=%.3e Cauchy=%.3e; Nmax=%d.",
        Tf,
        final_result.coarse_N,
        final_result.N,
        final_result.absolute_residual,
        final_result.relative_residual,
        final_result.initial_value_error,
        final_result.initial_derivative_error,
        final_result.coefficient_tail,
        final_result.cauchy_error,
        Nmax,
    )
    throw_on_failure && throw(PantographConvergenceError(message, final_result))
    return merge(final_result, (failure_message=message,))
end

"""
    solve_scaled_pantograph(y0, dy0;
        omega2, gamma, q, T=100.0, N=nothing, adaptive=true, ...)

Solve `y''(t)+omega2*y(t)=gamma*y'(q*t)` with `y(0)=y0` and
`y'(0)=dy0`.  RestMat supplies the proportional Chebyshev restriction and
all differential terms are assembled in ultraspherical coefficient space.
Supplying `N` selects a fixed-degree solve; otherwise `adaptive=true`
compares successive truncations.  The diagnostic result contains the
coefficients, independent residuals, initial errors, tail and Cauchy error,
degree, timing, and convergence status.
"""
function solve_scaled_pantograph(
    y0::Number,
    dy0::Number;
    omega2::Real,
    gamma::Number,
    q::Real,
    T::Real=100.0,
    N::Union{Nothing,Integer}=nothing,
    adaptive::Bool=true,
    N0::Union{Nothing,Integer}=nothing,
    Nmax::Integer=MAX_SOLVE_DEGREE,
    tol::Real=1.0e-10,
    residual_tol::Real=1.0e-9,
    relative_residual_tol::Real=Inf,
    initial_tol::Real=1.0e-11,
    tail_tol::Real=1.0e-12,
    validation_points_per_period::Integer=32,
    max_refinements::Integer=4,
    verbose::Bool=false,
    throw_on_failure::Bool=true,
    return_info::Bool=true,
)
    y0c = ComplexF64(y0)
    dy0c = ComplexF64(dy0)
    if N === nothing && adaptive
        result = solve_scaled_pantograph_adaptive(
            y0c,
            dy0c;
            omega2=omega2,
            gamma=gamma,
            q=q,
            T=T,
            N0=N0,
            Nmax=Nmax,
            cauchy_tol=tol,
            residual_tol=residual_tol,
            relative_residual_tol=relative_residual_tol,
            initial_tol=initial_tol,
            tail_tol=tail_tol,
            validation_points_per_period=validation_points_per_period,
            max_refinements=max_refinements,
            verbose=verbose,
            throw_on_failure=throw_on_failure,
        )
        return return_info ? result : result.coefficients
    end

    fixed_N =
        N === nothing ?
        (N0 === nothing ? default_initial_degree(T) : Int(N0)) :
        Int(N)
    fixed_N <= min(Int(Nmax), MAX_SOLVE_DEGREE) || throw(ArgumentError(
        "fixed degree N=$fixed_N exceeds the configured limit",
    ))
    fixed = _solve_scaled_pantograph_fixed(
        fixed_N,
        T;
        y0=y0c,
        dy0=dy0c,
        omega2=omega2,
        gamma=gamma,
        q=q,
        return_info=true,
    )
    grid = validation_grid(
        T;
        points_per_period=validation_points_per_period,
    )
    diagnostics = _scaled_solution_diagnostics(
        fixed.coefficients,
        T,
        y0c,
        dy0c,
        grid;
        omega2=omega2,
        gamma=gamma,
        q=q,
        cauchy_error=nothing,
        cauchy_tol=tol,
        residual_tol=residual_tol,
        relative_residual_tol=relative_residual_tol,
        initial_tol=initial_tol,
        tail_tol=tail_tol,
    )
    result = merge(
        fixed,
        diagnostics,
        (
            adaptive=false,
            number_of_validation_points=length(grid),
            tolerances=(
                cauchy=Float64(tol),
                residual=Float64(residual_tol),
                relative_residual=Float64(relative_residual_tol),
                initial=Float64(initial_tol),
                tail=Float64(tail_tol),
            ),
        ),
    )
    if !result.converged && throw_on_failure
        message = @sprintf(
            "Fixed proportional-delay solve failed at N=%d: residual=%.3e relative=%.3e IC=(%.3e, %.3e) tail=%.3e.",
            fixed_N,
            result.absolute_residual,
            result.relative_residual,
            result.initial_value_error,
            result.initial_derivative_error,
            result.coefficient_tail,
        )
        throw(PantographConvergenceError(message, result))
    end
    return return_info ? result : result.coefficients
end

function _figure2_diagnostics(
    coefficients::AbstractVector,
    T::Real,
    y0::Number,
    dy0::Number,
    grid::AbstractVector{<:Real};
    cauchy_error,
    cauchy_tol::Real,
    residual_tol::Real,
    relative_residual_tol::Real,
    initial_tol::Real,
    tail_tol::Real,
)
    residual = scaled_equation_residual(
        coefficients,
        T,
        grid;
        omega2=2.0,
        gamma=5.0im,
        q=1.0 / 6.0,
        return_details=true,
    )
    initial = initial_condition_errors(
        coefficients,
        T;
        y0=y0,
        dy0=dy0,
    )
    tail = coefficient_tail_estimate(coefficients)
    cauchy_ok = cauchy_error === nothing || cauchy_error <= cauchy_tol
    converged =
        cauchy_ok &&
        residual.absolute_residual <= residual_tol &&
        residual.relative_residual <= relative_residual_tol &&
        initial.value <= initial_tol &&
        initial.derivative <= initial_tol &&
        tail <= tail_tol
    return (
        converged=converged,
        equation_residual=residual.absolute_residual,
        absolute_residual=residual.absolute_residual,
        relative_residual=residual.relative_residual,
        initial_value_error=initial.value,
        initial_derivative_error=initial.derivative,
        coefficient_tail=tail,
        tail_estimate=tail,
        cauchy_error=cauchy_error,
    )
end

"""
    prepare_figure2_solver(; T=1000.0, N=nothing, adaptive=true, ...)

Prepare reusable fundamental solutions for

    y''(t) + 2y(t) = 5im*y'(t/6),  0 <= t <= T.

At each tested degree, RestMat constructs the `[-1,-2/3]` Chebyshev
restriction once and the ultraspherical tau matrix is factored once with
complex unitary Givens rotations.  The two right-hand sides define
`phi0(0)=1, phi0'(0)=0` and `phi1(0)=0, phi1'(0)=1`.  The default `T=1000`
adaptive calculation checks Cauchy error, absolute and relative independent
residuals, coefficient tails, and both initial conditions.

The returned `NamedTuple` contains `coefficients_phi0`,
`coefficients_phi1`, their diagnostics, common `N` and `T`, timing, and
factorization metadata.  Pass it to [`solve_prepared_figure2`](@ref) to try
many complex initial values without rebuilding RestMat or refactoring the
matrix.
"""
function prepare_figure2_solver(
    ;
    T::Real=1000.0,
    N::Union{Nothing,Integer}=nothing,
    adaptive::Bool=true,
    N0::Integer=512,
    Nmax::Integer=MAX_SOLVE_DEGREE,
    tol::Real=1.0e-10,
    residual_tol::Real=1.0e-9,
    relative_residual_tol::Real=1.0e-11,
    initial_tol::Real=1.0e-11,
    tail_tol::Real=1.0e-12,
    validation_points_per_period::Integer=32,
    max_refinements::Integer=4,
    verbose::Bool=false,
    throw_on_failure::Bool=true,
)
    if N === nothing && adaptive
        prepared = _prepare_scaled_pantograph_pair_adaptive(
            T;
            omega2=2.0,
            gamma=5.0im,
            q=1.0 / 6.0,
            N0=N0,
            Nmax=Nmax,
            cauchy_tol=tol,
            residual_tol=residual_tol,
            relative_residual_tol=relative_residual_tol,
            initial_tol=initial_tol,
            tail_tol=tail_tol,
            validation_points_per_period=validation_points_per_period,
            max_refinements=max_refinements,
            verbose=verbose,
            throw_on_failure=throw_on_failure,
        )
        return merge(
            prepared,
            (
                adaptive=true,
                factorization_reused=true,
            ),
        )
    end

    fixed_N = N === nothing ? Int(N0) : Int(N)
    fixed_N <= Nmax || throw(ArgumentError(
        "fixed degree N=$fixed_N exceeds Nmax=$Nmax",
    ))
    pair = _solve_scaled_pantograph_pair(
        fixed_N,
        T;
        omega2=2.0,
        gamma=5.0im,
        q=1.0 / 6.0,
    )
    grid = validation_grid(
        T;
        points_per_period=validation_points_per_period,
    )
    phi0 = _figure2_diagnostics(
        pair.c_phi0,
        T,
        1.0,
        0.0,
        grid;
        cauchy_error=nothing,
        cauchy_tol=tol,
        residual_tol=residual_tol,
        relative_residual_tol=relative_residual_tol,
        initial_tol=initial_tol,
        tail_tol=tail_tol,
    )
    phi1 = _figure2_diagnostics(
        pair.c_phi1,
        T,
        0.0,
        1.0,
        grid;
        cauchy_error=nothing,
        cauchy_tol=tol,
        residual_tol=residual_tol,
        relative_residual_tol=relative_residual_tol,
        initial_tol=initial_tol,
        tail_tol=tail_tol,
    )
    result = merge(
        pair,
        (
            coefficients_phi0=pair.c_phi0,
            coefficients_phi1=pair.c_phi1,
            converged=phi0.converged && phi1.converged,
            adaptive=false,
            phi0=phi0,
            phi1=phi1,
            validation_grid=grid,
            number_of_validation_points=length(grid),
            factorization_count_per_level=1,
            factorization_reused=true,
            tolerances=(
                cauchy=Float64(tol),
                residual=Float64(residual_tol),
                relative_residual=Float64(relative_residual_tol),
                initial=Float64(initial_tol),
                tail=Float64(tail_tol),
            ),
        ),
    )
    if !result.converged && throw_on_failure
        message = @sprintf(
            "Fixed Figure 2 preparation failed at T=%.4g, N=%d: phi0 residual=%.3e, phi1 residual=%.3e.",
            Float64(T),
            fixed_N,
            phi0.absolute_residual,
            phi1.absolute_residual,
        )
        throw(PantographConvergenceError(message, result))
    end
    return result
end

"""
    solve_prepared_figure2(prepared, y0, dy0; ...)

Construct the Figure 2 solution with `y(0)=y0`, `y'(0)=dy0` by the complex
linear combination

    c = y0*prepared.coefficients_phi0 +
        dy0*prepared.coefficients_phi1.

No RestMat assembly or matrix factorization is repeated.  The combined
coefficients are nevertheless checked independently for
`y''+2y-5im*y'(t/6)`, both initial conditions, coefficient tail, and (when
adaptive coarse coefficients are available) their own Cauchy error.
The returned diagnostic `NamedTuple` contains the combined coefficients,
errors, common `N`/`T`, preparation time, and validation time.
"""
function solve_prepared_figure2(
    prepared::NamedTuple,
    y0::Number,
    dy0::Number;
    validation_points_per_period::Integer=32,
    cauchy_tol::Real=prepared.tolerances.cauchy,
    residual_tol::Real=prepared.tolerances.residual,
    relative_residual_tol::Real=prepared.tolerances.relative_residual,
    initial_tol::Real=prepared.tolerances.initial,
    tail_tol::Real=prepared.tolerances.tail,
    throw_on_failure::Bool=true,
)
    prepared.converged || throw(ArgumentError(
        "prepared solver has not passed its validation criteria",
    ))
    y0c = ComplexF64(y0)
    dy0c = ComplexF64(dy0)
    started = time_ns()
    coefficients = ComplexF64.(
        y0c .* prepared.coefficients_phi0 .+
        dy0c .* prepared.coefficients_phi1
    )
    grid = validation_grid(
        prepared.T;
        points_per_period=validation_points_per_period,
    )
    cauchy_error = if hasproperty(prepared, :coarse_coefficients_phi0)
        coarse = ComplexF64.(
            y0c .* prepared.coarse_coefficients_phi0 .+
            dy0c .* prepared.coarse_coefficients_phi1
        )
        _cauchy_error(coarse, coefficients, prepared.T, grid)
    else
        nothing
    end
    diagnostics = _figure2_diagnostics(
        coefficients,
        prepared.T,
        y0c,
        dy0c,
        grid;
        cauchy_error=cauchy_error,
        cauchy_tol=cauchy_tol,
        residual_tol=residual_tol,
        relative_residual_tol=relative_residual_tol,
        initial_tol=initial_tol,
        tail_tol=tail_tol,
    )
    validation_time = (time_ns() - started) * 1.0e-9
    result = merge(
        diagnostics,
        (
            coefficients=coefficients,
            T=prepared.T,
            N=prepared.N,
            omega2=2.0,
            gamma=5.0im,
            q=1.0 / 6.0,
            y0=y0c,
            dy0=dy0c,
            adaptive=prepared.adaptive,
            solve_time=validation_time,
            validation_time=validation_time,
            preparation_time=prepared.solve_time,
            operator_storage=prepared.operator_storage,
            operator_eltype=prepared.operator_eltype,
            tolerances=(
                cauchy=Float64(cauchy_tol),
                residual=Float64(residual_tol),
                relative_residual=Float64(relative_residual_tol),
                initial=Float64(initial_tol),
                tail=Float64(tail_tol),
            ),
        ),
    )
    if !result.converged && throw_on_failure
        message = @sprintf(
            "Prepared Figure 2 solution failed at T=%.4g, N=%d: residual=%.3e, relative=%.3e, IC=(%.3e, %.3e), Cauchy=%s.",
            prepared.T,
            prepared.N,
            result.absolute_residual,
            result.relative_residual,
            result.initial_value_error,
            result.initial_derivative_error,
            string(result.cauchy_error),
        )
        throw(PantographConvergenceError(message, result))
    end
    return result
end

"""
    solve_figure2_pantograph(y0, dy0;
        T=1000.0, N=nothing, adaptive=true, N0=512, Nmax=5000, ...)

Solve

    y''(t) + 2y(t) = 5im*y'(t/6),  0 <= t <= T,

with `y(0)=y0` and `y'(0)=dy0`.  The Figure 2 wrapper defaults to `T=1000`.
Real or complex inputs are promoted to `ComplexF64`.  RestMat supplies the
Chebyshev restriction to `[-1,-2/3]`; differentiation, basis conversion,
tau boundary rows, and the complex unitary Givens solve all remain in
ultraspherical coefficient space.

The returned diagnostics include the coefficients, absolute and relative
independent residuals, both initial errors, tail estimate, Cauchy error,
degree, timing, and convergence status.  Use [`prepare_figure2_solver`](@ref)
when solving repeatedly for different initial values.
"""
function solve_figure2_pantograph(
    y0::Number,
    dy0::Number;
    T::Real=1000.0,
    N::Union{Nothing,Integer}=nothing,
    adaptive::Bool=true,
    N0::Integer=512,
    Nmax::Integer=MAX_SOLVE_DEGREE,
    tol::Real=1.0e-10,
    residual_tol::Real=1.0e-9,
    relative_residual_tol::Real=1.0e-11,
    return_info::Bool=true,
    verbose::Bool=false,
    throw_on_failure::Bool=true,
)
    return solve_scaled_pantograph(
        ComplexF64(y0),
        ComplexF64(dy0);
        omega2=2.0,
        gamma=5.0im,
        q=1.0 / 6.0,
        T=T,
        N=N,
        adaptive=adaptive,
        N0=N0,
        Nmax=Nmax,
        tol=tol,
        residual_tol=residual_tol,
        relative_residual_tol=relative_residual_tol,
        initial_tol=1.0e-11,
        tail_tol=1.0e-12,
        verbose=verbose,
        throw_on_failure=throw_on_failure,
        return_info=return_info,
    )
end

end # module PantographPeriod
