module Ex7Common

export composite_test_grid, error_degrees

"""
    error_degrees(nmax; nmin=10, step=10)

Return the shared formal Example 7 error degrees.  The default settings with
`nmax=2000` produce the single uniform sequence `10:10:2000`.
"""
function error_degrees(
    nmax::Int;
    nmin::Int = 10,
    step::Int = 10,
)
    nmin >= 0 || throw(ArgumentError("error nmin must be nonnegative"))
    nmax >= nmin || throw(ArgumentError("error nmax must be at least nmin"))
    step > 0 || throw(ArgumentError("error step must be positive"))
    (nmax - nmin) % step == 0 ||
        throw(ArgumentError("error nmax must lie on the uniform nmin:step grid"))
    return collect(nmin:step:nmax)
end

"""
    composite_test_grid(uniform_count=10001, fractional_count=5001)

Build the common physical comparison grid from a uniform `x` grid and a
uniform `z=s^(1/6)` grid mapped by `s=z^6`, `x=2s-1`.  Exact duplicate
Float64 physical `x` values are removed after sorting.  The corresponding
`s` coordinates are retained so both methods can evaluate endpoint factors
without an unnecessary `s -> x -> s` round trip.
"""
function composite_test_grid(
    uniform_count::Int = 10001,
    fractional_count::Int = 5001,
)
    uniform_count >= 2 || throw(ArgumentError("uniform_count must be at least 2"))
    fractional_count >= 2 || throw(ArgumentError("fractional_count must be at least 2"))

    x_uniform = collect(range(-1.0, 1.0; length = uniform_count))
    s_uniform = @. (x_uniform + 1.0) / 2.0

    z_fractional = collect(range(0.0, 1.0; length = fractional_count))
    s_fractional = z_fractional .^ 6
    x_fractional = @. 2.0 * s_fractional - 1.0

    x_all = vcat(x_uniform, x_fractional)
    s_all = vcat(s_uniform, s_fractional)
    order = sortperm(x_all; alg = MergeSort)

    x = Float64[]
    s = Float64[]
    sizehint!(x, length(x_all))
    sizehint!(s, length(s_all))

    @inbounds for idx in order
        xi = x_all[idx]
        if isempty(x) || xi != x[end]
            push!(x, xi)
            push!(s, s_all[idx])
        elseif xi == -1.0 && s_all[idx] == 0.0
            # Make the exact endpoint representative deterministic when tiny
            # positive z^6 values round to the same physical Float64 x=-1.
            s[end] = 0.0
        end
    end

    return (
        x = x,
        s = s,
        uniform_count = uniform_count,
        fractional_count = fractional_count,
        total_count = length(x),
    )
end

end # module Ex7Common
