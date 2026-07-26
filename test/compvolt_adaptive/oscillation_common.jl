module CompVoltAdaptiveOscillation

using VoltOpt

const INPUT_DEGREE = 16
const INPUT_LENGTH = INPUT_DEGREE + 1
const INITIAL_DEGREE = INPUT_DEGREE
const TOLERANCE = 1.0e-10
const MAX_DEGREE = 1024
const MAX_REFINEMENTS = 7
const REFERENCE_FLOOR = 256
const EVALUATION_DEGREE = 1024
const OMEGAS = [0, 5, 10, 20, 40, 80, 160]

input_function(x) = exp(x)

function input_coefficients()
    x = [-cospi(i / INPUT_DEGREE) for i in 0:INPUT_DEGREE]
    coefficients = VoltOpt.vals2cheb(input_function.(x))
    @inbounds for i in 2:2:length(coefficients)
        coefficients[i] = -coefficients[i]
    end
    return Vector{Float64}(coefficients)
end

oscillatory_kernel(omega::Real) = (x, t, z) -> cos(omega * (x - t)) * z

function adaptive_solution(omega::Real, cu::AbstractVector=input_coefficients())
    return VoltOpt.compVolt_nonlinear_fast(
        oscillatory_kernel(omega),
        cu;
        M0=INITIAL_DEGREE,
        tol=TOLERANCE,
        Mmax=MAX_DEGREE,
        max_refinements=MAX_REFINEMENTS,
        return_info=true,
    )
end

function fixed_reference(
    omega::Real,
    accepted_M::Int,
    cu::AbstractVector=input_coefficients(),
)
    Mref = max(2 * accepted_M, REFERENCE_FLOOR)
    Mref <= MAX_DEGREE || error(
        "Reference degree Mref=$Mref exceeds configured maximum $MAX_DEGREE.",
    )
    coefficients = VoltOpt.compVolt_nonlinear_fast(
        oscillatory_kernel(omega), cu; M=Mref,
    )
    return Mref, coefficients
end

function chebyshev_evaluate(coefficients::AbstractVector, x::Real)
    isempty(coefficients) && return zero(x)
    length(coefficients) == 1 && return coefficients[1]

    T = promote_type(eltype(coefficients), typeof(x))
    b1 = zero(T)
    b2 = zero(T)
    @inbounds for k in length(coefficients):-1:2
        b0 = muladd(2 * x, b1, coefficients[k] - b2)
        b2 = b1
        b1 = b0
    end
    return muladd(x, b1, coefficients[1] - b2)
end

function comparison_error(
    adaptive_coefficients::AbstractVector,
    reference_coefficients::AbstractVector;
    evaluation_degree::Int=EVALUATION_DEGREE,
)
    evaluation_degree >= 1 || throw(ArgumentError("evaluation_degree must be positive."))
    error_inf = 0.0
    reference_norm = 0.0
    @inbounds for i in 0:evaluation_degree
        x = -cospi(i / evaluation_degree)
        adaptive_value = chebyshev_evaluate(adaptive_coefficients, x)
        reference_value = chebyshev_evaluate(reference_coefficients, x)
        error_inf = max(error_inf, abs(adaptive_value - reference_value))
        reference_norm = max(reference_norm, abs(reference_value))
    end
    relative_error = error_inf / max(reference_norm, eps(Float64))
    return error_inf, relative_error
end

end # module CompVoltAdaptiveOscillation
