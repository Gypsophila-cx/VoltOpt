function pad_to_length(c, n::Integer)
    n < 0 && throw(ArgumentError("target length must be non-negative"))
    v = zeros(Float64, Int(n))
    m = min(length(c), Int(n))
    if m > 0
        v[1:m] .= c[1:m]
    end
    return v
end

function _rhs_callable(f)
    if f isa Function
        return f
    elseif f isa AbstractVector
        coeffs = f
        return x -> begin
            value = zero(float(x))
            power = one(float(x))
            for c in coeffs
                value += c * power
                power *= x
            end
            value
        end
    end
    throw(ArgumentError("RHS input must be a callable object or polynomial coefficients"))
end

function _validation_grid(domain, validation_grid)
    validation_grid !== nothing && return validation_grid
    a, b = extrema(domain)
    return range(float(a), float(b); length=1000)
end

function _rhs_basis_label(alpha, beta, domain)
    a, b = extrema(domain)
    return "Jacobi($alpha,$beta,$a..$b)"
end

"""
    validate_rhs_projection(rhs_coeff, f; alpha, beta, domain=0..1, validation_grid=nothing)

Evaluate shifted Jacobi RHS coefficients back on a physical grid and return
the maximum absolute error against `f`.
"""
function validate_rhs_projection(
    rhs_coeff,
    f;
    alpha,
    beta,
    domain=0..1,
    validation_grid=nothing,
)
    fcall = _rhs_callable(f)
    space = Jacobi(alpha, beta, domain)
    approx = Fun(space, rhs_coeff)
    grid = _validation_grid(domain, validation_grid)
    return maximum(abs(approx(x) - fcall(x)) for x in grid)
end

"""
    build_go_rhs_coefficients(f, N; alpha, beta, domain=0..1,
        method=:fixed_gauss_jacobi, projection_tol=1e-12, validate=true,
        validation_grid=nothing)

Build a fixed-length shifted Jacobi RHS coefficient vector and projection
metadata.  The default path uses the package's fixed-`N` ApproxFun/Jacobi
coefficient construction, which is the numerical path used by the golden
Section 5.2.2 workflow.  `method=:exact_polynomial` accepts a vector of
monomial coefficients `[a0, a1, ...]`; `method=:adaptive_approxfun` is exposed
for verification only and records that status in `notes`.
"""
function build_go_rhs_coefficients(
    f,
    N::Integer;
    alpha,
    beta,
    domain=0..1,
    method=:fixed_gauss_jacobi,
    projection_tol=1e-12,
    validate::Bool=true,
    validation_grid=nothing,
)
    N < 1 && throw(ArgumentError("N must be positive"))
    fcall = _rhs_callable(f)
    space = Jacobi(alpha, beta, domain)

    raw_fun =
        method == :fixed_gauss_jacobi ? Fun(fcall, space, Int(N)) :
        method == :exact_polynomial ? Fun(fcall, space, Int(N)) :
        method == :adaptive_approxfun ? Fun(fcall, space) :
        throw(ArgumentError("unsupported RHS projection method $(method)"))

    rhs_coeff = pad_to_length(raw_fun.coefficients, Int(N))
    tail_start = max(1, length(rhs_coeff) - 9)
    validation_error = validate ?
        validate_rhs_projection(
            rhs_coeff,
            fcall;
            alpha=alpha,
            beta=beta,
            domain=domain,
            validation_grid=validation_grid,
        ) :
        missing
    status = validation_error === missing || validation_error <= max(projection_tol, 1e-8) ?
        "pass" :
        "check"
    notes = method == :adaptive_approxfun ?
        ("Adaptive ApproxFun projection used for verification; fixed-N projection is preferred for formal workflows.",) :
        ("Fixed-length shifted Jacobi RHS projection.",)

    info = (
        method=method,
        basis=_rhs_basis_label(alpha, beta, domain),
        coefficient_length=length(raw_fun.coefficients),
        tail_norm=norm(view(rhs_coeff, tail_start:length(rhs_coeff))),
        validation_error=validation_error,
        status=status,
        notes=notes,
    )
    return rhs_coeff, info
end
