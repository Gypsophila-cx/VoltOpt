"""
    assemble_semibanded_system(body_operator, N; evaluation_row=build_initial_evaluation_row(N))

Assemble a semi-banded sparse system from one dense evaluation/tau row and the
first `N-1` rows of a banded body operator.  The evaluation row is inserted by
triplets, and the body rows are scanned only over the declared band.
"""
function assemble_semibanded_system(
    body_operator,
    N::Integer;
    evaluation_row=build_initial_evaluation_row(N),
)
    N < 2 && throw(ArgumentError("N must be at least 2"))
    size(body_operator, 1) == N && size(body_operator, 2) == N ||
        throw(DimensionMismatch("body operator must be N by N"))
    length(evaluation_row) == N ||
        throw(DimensionMismatch("evaluation row length must match N"))

    rowidx = Int[]
    colidx = Int[]
    vals = Float64[]

    @inbounds for j = 1:N
        val = evaluation_row[j]
        if !iszero(val)
            push!(rowidx, 1)
            push!(colidx, j)
            push!(vals, val)
        end
    end

    lower_bw, upper_bw = BandedMatrices.bandwidths(body_operator)
    lower_bw = Int(lower_bw)
    upper_bw = Int(upper_bw)

    @inbounds for source_row = 1:(N - 1)
        jlo = max(1, source_row - lower_bw)
        jhi = min(N, source_row + upper_bw)
        for j = jlo:jhi
            val = body_operator[source_row, j]
            if !iszero(val)
                push!(rowidx, source_row + 1)
                push!(colidx, j)
                push!(vals, val)
            end
        end
    end

    return sparse(rowidx, colidx, vals, N, N)
end

"""
    assemble_section522_semibanded_system(body_operator, N)

Assemble the Section 5.2.2 VIDE system with the first row enforcing `u(0)`.
The body rows are the first `N-1` rows of `D - S_00_to_12 * R * L * Vraw`.
"""
assemble_section522_semibanded_system(body_operator, N::Integer) =
    assemble_semibanded_system(body_operator, N; evaluation_row=build_initial_evaluation_row(N))

"""
    build_secondkind_equation_system(Vraw, N; convention=:plus)

Assemble the strict-band second-kind Volterra system.  `convention=:plus`
builds `A = I + Vphys` for `u + int K u = f`; `convention=:original` builds
`A = I - Vphys` for the original repository convention `u = g + int K u`.
"""
function build_secondkind_equation_system(Vraw, N::Integer; convention=:plus, check_band::Bool=true)
    sign =
        convention == :plus ? 1.0 :
        convention == :original ? -1.0 :
        throw(ArgumentError("convention must be :plus or :original"))
    return build_secondkind_system_strictband(Vraw, N; sign=sign, check_band=check_band)
end
