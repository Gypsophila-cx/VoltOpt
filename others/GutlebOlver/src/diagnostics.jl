storage_label(A) =
    A isa BandedMatrix ? "BandedMatrix" :
    A isa SparseMatrixCSC ? "SparseMatrixCSC" :
    string(typeof(A))

"""
    assert_no_dense_matrix(A)

Reject dense operator inputs on the formal sparse workflow.  `BandedMatrix` and
`SparseMatrixCSC` are accepted; generic dense matrix-like inputs are rejected.
"""
function assert_no_dense_matrix(A)
    if A isa BandedMatrix || A isa SparseMatrixCSC
        return A
    end
    throw(ArgumentError("formal Gutleb--Olver sparse workflow only accepts BandedMatrix or SparseMatrixCSC operators, got $(typeof(A))"))
end

"""
    safe_bandwidth(A) -> (lower, upper)

Return the actual nonzero lower/upper bandwidth of a `BandedMatrix` or
`SparseMatrixCSC` by scanning only the sparse/banded storage pattern. This
diagnostic does not materialize dense storage.
"""
function safe_bandwidth(A::BandedMatrix)
    rows, cols = size(A)
    structural_l, structural_u = BandedMatrices.bandwidths(A)
    structural_l = Int(structural_l)
    structural_u = Int(structural_u)

    actual_l = 0
    actual_u = 0
    @inbounds for j = 1:cols
        ilo = max(1, j - structural_u)
        ihi = min(rows, j + structural_l)
        for i = ilo:ihi
            if !iszero(A[i, j])
                actual_l = max(actual_l, i - j)
                actual_u = max(actual_u, j - i)
            end
        end
    end

    return (actual_l, actual_u)
end

function safe_bandwidth(A::SparseMatrixCSC)
    lower_bw = 0
    upper_bw = 0
    @inbounds for j = 1:size(A, 2)
        for p = A.colptr[j]:(A.colptr[j + 1] - 1)
            val = A.nzval[p]
            if !iszero(val)
                i = A.rowval[p]
                lower_bw = max(lower_bw, i - j)
                upper_bw = max(upper_bw, j - i)
            end
        end
    end
    return (lower_bw, upper_bw)
end

safe_bandwidth(A) =
    throw(ArgumentError("safe_bandwidth is only defined for BandedMatrix and SparseMatrixCSC inputs"))

"""
    safe_nnz(A) -> Integer

Count actual stored nonzero values in a `BandedMatrix` or `SparseMatrixCSC`
without converting the operator to dense storage.
"""
function safe_nnz(A::BandedMatrix)
    rows, cols = size(A)
    structural_l, structural_u = BandedMatrices.bandwidths(A)
    structural_l = Int(structural_l)
    structural_u = Int(structural_u)

    nz = 0
    @inbounds for j = 1:cols
        ilo = max(1, j - structural_u)
        ihi = min(rows, j + structural_l)
        for i = ilo:ihi
            if !iszero(A[i, j])
                nz += 1
            end
        end
    end

    return nz
end

function safe_nnz(A::SparseMatrixCSC)
    nz = 0
    @inbounds for p in eachindex(A.nzval)
        nz += !iszero(A.nzval[p])
    end
    return nz
end

safe_nnz(A) =
    throw(ArgumentError("safe_nnz is only defined for BandedMatrix and SparseMatrixCSC inputs"))

"""
    safe_density(A) -> Float64

Return `safe_nnz(A) / prod(size(A))` for sparse or banded operators without any
dense fallback.
"""
function safe_density(A)
    rows, cols = size(A)
    return safe_nnz(A) / (rows * cols)
end

function operator_summary(A)
    assert_no_dense_matrix(A)
    rows, cols = size(A)
    lower_bw, upper_bw = safe_bandwidth(A)
    nz = safe_nnz(A)
    return (
        type=storage_label(A),
        rows=rows,
        cols=cols,
        nnz=nz,
        lower_bandwidth=lower_bw,
        upper_bandwidth=upper_bw,
        bandwidth=(lower_bw, upper_bw),
        density=nz / (rows * cols),
    )
end

operator_stats(A) = operator_summary(A)
banded_stats(A::BandedMatrix) = operator_summary(A)
sparse_stats(A::SparseMatrixCSC) = operator_summary(A)

function result_summary(result::NamedTuple)
    return (
        status=result.status,
        N=result.N,
        kerneldepth=result.kerneldepth,
        rhs_validation_error=result.rhs_validation_error,
        body_residual_relative=result.body_residual_relative,
        max_abs_error=result.max_abs_error,
        raw_operator=(result.raw_operator_type, result.raw_operator_nnz, result.raw_operator_bandwidth),
        body_operator=(result.body_operator_type, result.body_operator_nnz, result.body_operator_bandwidth),
        final_system=(result.final_system_type, result.final_system_nnz, result.full_bandwidth),
    )
end

const RESULT_CSV_COLUMNS = (
    :k,
    :N,
    :kerneldepth,
    :rhs_validation_error,
    :tau_residual,
    :body_residual_relative,
    :max_abs_error,
    :raw_operator_type,
    :raw_operator_nnz,
    :raw_operator_bandwidth,
    :body_operator_type,
    :body_operator_nnz,
    :body_operator_bandwidth,
    :final_system_type,
    :final_system_nnz,
    :full_bandwidth,
    :body_bandwidth,
    :density,
    :rhs_time,
    :kernel_coefficient_time,
    :volterra_construction_time,
    :system_assembly_time,
    :solve_time,
    :evaluation_time,
    :exact_projection_time,
    :total_time,
    :status,
    :notes,
)

_csv_results(result::NamedTuple) = (result,)
_csv_results(results) = results

function _result_value(result::NamedTuple, key::Symbol)
    if key in keys(result)
        return getproperty(result, key)
    elseif key == :rhs_time && :timings in keys(result)
        return result.timings.rhs
    elseif key == :kernel_coefficient_time && :timings in keys(result)
        return result.timings.kernel_coefficients
    elseif key == :volterra_construction_time && :timings in keys(result)
        return result.timings.volterra_construction
    elseif key == :system_assembly_time && :timings in keys(result)
        return result.timings.system_assembly
    elseif key == :solve_time && :timings in keys(result)
        return result.timings.solve
    elseif key == :evaluation_time && :timings in keys(result)
        return result.timings.evaluation
    elseif key == :exact_projection_time && :timings in keys(result)
        return result.timings.exact_projection
    elseif key == :total_time && :timings in keys(result)
        return result.timings.total
    else
        return missing
    end
end

function _csv_escape(value)
    text = value === missing ? "" : string(value)
    if occursin(',', text) || occursin('"', text) || occursin('\n', text) || occursin('\r', text)
        return "\"" * replace(text, "\"" => "\"\"") * "\""
    end
    return text
end

function _write_result_csv_header(io)
    println(io, join(string.(RESULT_CSV_COLUMNS), ","))
end

function _write_result_csv_row(io, result::NamedTuple)
    values = (_csv_escape(_result_value(result, col)) for col in RESULT_CSV_COLUMNS)
    println(io, join(values, ","))
end

"""
    write_results_csv(path, results)

Write one or more workflow result `NamedTuple`s to `path` as CSV using the
stable public result fields.
"""
function write_results_csv(path::AbstractString, results)
    open(path, "w") do io
        _write_result_csv_header(io)
        for result in _csv_results(results)
            _write_result_csv_row(io, result)
        end
    end
    return path
end

"""
    append_result_csv(path, result)

Append one workflow result `NamedTuple` to a CSV file, writing the header first
when the file is new or empty.
"""
function append_result_csv(path::AbstractString, result::NamedTuple)
    needs_header = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        needs_header && _write_result_csv_header(io)
        _write_result_csv_row(io, result)
    end
    return path
end
