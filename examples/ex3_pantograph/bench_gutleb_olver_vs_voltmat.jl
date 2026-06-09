# problem: ex3
# original script: examples/ex3_pantograph/ex3_pantograph_firstkind.jl
# N source: collect(2:1:100)
# error_type: max_abs, from sweep.errors/err_inf in the original script
# original integral form: differentiated pantograph Volterra terms on [0,x] after x=(X+1)/2
# GO integral form after transform: int_0^s plus shifted argument u(qs)
# GO output basis: not selected; pantograph/shifted-argument sparse GO wrapper is not certified
# voltmat basis: Chebyshev
# error grid: physical [0,1] s grid

include(joinpath(dirname(@__DIR__), "bench_common.jl"))
using SpecialFunctions

const q_default = 0.5

u_go(s) = exp(s) * cos(20s)

function rhs_go(s)
    lambda = 1.0 + 20.0im
    z = sqrt(pi) / 2 * lambda * exp(lambda * s + lambda^2 / 4) *
        (erf((1 - q_default) * s + lambda / 2) - erf(lambda / 2))
    endpoint = (1 - q_default) * exp(q_default * s - (1 - q_default)^2 * s^2) *
        cos(20 * q_default * s)
    return real(z) + endpoint
end

K1_go(s, r) = -2.0 * (s - r) * exp(-(s - r)^2)
K2_go(s, r) = -2.0 * (s - q_default * r) * exp(-(s - q_default * r)^2)
delay_coeff_go(s) = -q_default * exp(-((1 - q_default) * s)^2)

prob = BenchProblem(
    name = "ex3",
    Ns = collect(2:1:100),
    ncoeff_from_N = N -> N + 1,
    rhs = rhs_go,
    exact = u_go,
    local_coeff = s -> 1.0,
    delay_coeff = delay_coeff_go,
    delay_scale = q_default,
    volterra_terms = [
        (sign = 1.0, kernel = K1_go, input_scale = 1.0),
        (sign = -q_default, kernel = K2_go, input_scale = q_default),
    ],
    error_type = "max_abs",
    primary_error = :exact,
    original_script = "examples/ex3_pantograph/ex3_pantograph_firstkind.jl",
    n_source = "collect(2:1:100)",
    error_source = "sweep.errors/err_inf on physical grid",
    integral_form = "int_0^s terms with pantograph argument q*s",
    go_integral_form = "int_0^s plus shifted argument; GO skipped",
    variable_transform = "original script already uses s=(X+1)/2 physical variable",
    go_output_basis = "not selected because shifted-argument GO sparse system assembly is not certified",
    notes = "pantograph equation; GO skipped rather than mixing basis or shifted-argument operators",
)

run_benchmark(prob)
