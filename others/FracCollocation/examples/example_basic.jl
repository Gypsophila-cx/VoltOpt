using FracCollocation

prob = FracVIEProblem(
    mu = 0.5,
    lambda = 0.5,
    alpha = -0.5,
    beta = -0.5,
    K = (x, s) -> exp(x - s),
    g = x -> 1.0,
)

sol = solve(prob, 40)
println(maximum(abs.(sol.values)))
