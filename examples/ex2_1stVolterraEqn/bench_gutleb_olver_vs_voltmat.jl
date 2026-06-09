# problem: ex2
# original script: examples/ex2_1stVolterraEqn/ex2_firstkind_direct_vs_differentiated.jl
# N source: [16, 24, ..., 1024]
# error_type: max_abs, from max_error_on_grid in the original script
# original integral form: first-kind int_{-1}^x; benchmark uses original differentiated second-kind path
# GO integral form after transform: int_0^s for the differentiated Kx equation
# kernel transform: Kx_go(s,r) = 2*Kx_original(2s-1,2r-1)
# GO output basis: shifted Jacobi P^(0,1) for int_0^s if a certified sparse wrapper is added
# voltmat basis: Chebyshev
# error grid: physical [0,1] s grid, mapped from original x=2s-1

include(joinpath(dirname(@__DIR__), "bench_common.jl"))

u_original(x) = 1.0 / (25x^2 + 1.0)
u_go(s) = u_original(chebX(s))
K1x_original(x, y) = begin
    d = x - y
    -40d * exp(-20d^2)
end
K1x_go(s, r) = 2.0 * K1x_original(chebX(s), chebX(r))

function gausslegendre_rule(n::Int)
    beta = [k / sqrt(4.0 * k^2 - 1.0) for k in 1:n-1]
    E = eigen(SymTridiagonal(zeros(n), beta))
    p = sortperm(E.values)
    return E.values[p], (2 .* abs2.(E.vectors[1, :]))[p]
end

const GLX_EX2, GLW_EX2 = gausslegendre_rule(300)

function int_0_s_ex2(f, s)
    s <= 0 && return 0.0
    h = 0.5 * s
    c = h
    acc = 0.0
    @inbounds for i in eachindex(GLX_EX2)
        acc += GLW_EX2[i] * f(c + h * GLX_EX2[i])
    end
    return h * acc
end

rhs_go(s) = u_go(s) + int_0_s_ex2(r -> K1x_go(s, r) * u_go(r), s)

prob = BenchProblem(
    name = "ex2",
    Ns = [16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024],
    ncoeff_from_N = N -> N + 1,
    rhs = rhs_go,
    exact = u_go,
    local_coeff = s -> 1.0,
    volterra_terms = [(sign = 1.0, kernel = K1x_go, input_scale = 1.0)],
    error_type = "max_abs",
    primary_error = :exact,
    original_script = "examples/ex2_1stVolterraEqn/ex2_firstkind_direct_vs_differentiated.jl",
    n_source = "[16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]",
    error_source = "max_error_on_grid; differentiated second-kind curve is used as the benchmark path",
    integral_form = "differentiated second-kind int_{-1}^x, transformed to int_0^s",
    go_integral_form = "int_0^s",
    variable_transform = "x_original = 2s - 1; Kx_go(s,r)=2*Kx_original(2s-1,2r-1)",
    go_output_basis = "shifted Jacobi P^(0,1), not used because sparse GO differentiated wrapper is not certified",
    notes = "K1 differentiated second-kind path from the original comparison",
)

run_benchmark(prob)
