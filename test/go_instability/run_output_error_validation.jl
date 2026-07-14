# Final output-error validation on padding-converged systems.

include(joinpath(@__DIR__, "go_instability_core.jl"))
using .GOInstabilityCore
using LinearAlgebra
using Printf

const SCRIPT_DIR = @__DIR__
const DATA_DIR = joinpath(SCRIPT_DIR, "data")
const PADDING_CSV = joinpath(DATA_DIR, "go_padding_validation.csv")
const OUTPUT_CSV = joinpath(DATA_DIR, "go_output_error_validation.csv")
const DEPTH_CSV = joinpath(DATA_DIR, "go_depth_validation.csv")
const MEMORY_BUDGET_BYTES = default_memory_budget_bytes()
const BIGFLOAT_BITS = 128
const NOUT = 72
const PRIMARY_DEPTH = 36
const DEPTHS = (8, 16, 24, 32, 36)
const KERNEL = "mixed_cos20xy"

const OUTPUT_COLUMNS = (
    :kernel,
    :validation_mode,
    :depth,
    :Nout,
    :M,
    :precision_bits,
    :norm_b,
    :state_norm,
    :reference_state_norm,
    :max_state_block_norm,
    :residual_inf,
    :residual_fro,
    :componentwise_backward_error,
    :backward_error_over_eps,
    :final_operator_norm,
    :float64_operator_norm,
    :Float64_to_128_operator_error,
    :relative_operator_error,
    :identity_relative_error,
    :identity_absolute_error,
    :norm_DeltaV_true,
    :norm_DeltaV_sum,
    :norm_DeltaV_T,
    :norm_DeltaV_solve,
    :norm_DeltaV_b,
    :norm_DeltaV_P,
    :relative_DeltaV_T_to_true,
    :relative_DeltaV_solve_to_true,
    :relative_DeltaV_b_to_true,
    :relative_DeltaV_P_to_true,
    :relative_DeltaV_T_to_Vstar,
    :relative_DeltaV_solve_to_Vstar,
    :relative_DeltaV_b_to_Vstar,
    :relative_DeltaV_P_to_Vstar,
    :aligned_DeltaV_T,
    :aligned_DeltaV_solve,
    :aligned_DeltaV_b,
    :aligned_DeltaV_P,
    :decomposition_relative_error,
    :dominant_error_source,
    :actual_output_gain,
    :padding_error,
    :estimated_peak_bytes,
    :memory_budget_bytes,
    :status,
)

const DEPTH_COLUMNS = (
    :kernel,
    :depth,
    :Nout,
    :M,
    :precision_bits,
    :kernel_tail,
    :state_norm,
    :reference_state_norm,
    :residual_inf,
    :residual_fro,
    :componentwise_backward_error,
    :backward_error_over_eps,
    :final_operator_norm,
    :final_operator_error,
    :relative_operator_error,
    :padding_error,
    :estimated_peak_bytes,
    :memory_budget_bytes,
    :status,
)

function parse_simple_csv(path::AbstractString)
    isfile(path) || return Vector{Dict{String,String}}()
    lines = readlines(path)
    isempty(lines) && return Vector{Dict{String,String}}()
    header = split(lines[1], ',')
    rows = Dict{String,String}[]
    for line in lines[2:end]
        isempty(strip(line)) && continue
        cells = split(line, ',')
        row = Dict{String,String}()
        for (i, key) in pairs(header)
            row[String(key)] = i <= length(cells) ? String(cells[i]) : ""
        end
        push!(rows, row)
    end
    return rows
end

function maybe_parse_int(text::AbstractString)
    isempty(strip(text)) && return missing
    return parse(Int, text)
end

function maybe_parse_float(text::AbstractString)
    isempty(strip(text)) && return missing
    return parse(Float64, text)
end

function padding_rows_for(depth::Integer)
    rows = parse_simple_csv(PADDING_CSV)
    return filter(rows) do row
        get(row, "kernel", "") == KERNEL &&
        maybe_parse_int(get(row, "depth", "")) == Int(depth) &&
        maybe_parse_int(get(row, "Nout", "")) == NOUT
    end
end

function selected_M_from_padding(depth::Integer)
    rows = padding_rows_for(depth)
    isempty(rows) && return recommended_internal_M(NOUT, depth; safety=8)
    selected = maybe_parse_int(get(rows[1], "selected_minimal_M", ""))
    return selected === missing ? recommended_internal_M(NOUT, depth; safety=8) :
                                  selected
end

function padding_error_for(depth::Integer, M::Integer)
    rows = padding_rows_for(depth)
    for row in rows
        maybe_parse_int(get(row, "M", "")) == Int(M) || continue
        return maybe_parse_float(get(row, "padding_error_to_max", ""))
    end
    return missing
end

function coeffs_for_mixed(depth::Integer)
    spec = default_kernel_specs()[1]
    return spec, make_coeffs(spec, depth; flip=true),
           "native_float64_projection_fixed_for_b_isolation"
end

function production_peak_bytes(depth::Integer, Nout::Integer, M::Integer;
                               identity::Bool=false)
    big_states = identity ? 6 : 2
    float_states = 2
    big_peak = estimate_peak_bytes(
        depth,
        M,
        BigFloat;
        bits=BIGFLOAT_BITS,
        complete_states=big_states,
        work_state_equivalents=2,
        dense_output_matrices=identity ? 6 : 3,
        Nout=Nout,
    )
    float_peak = estimate_peak_bytes(
        depth,
        M,
        Float64;
        complete_states=float_states,
        work_state_equivalents=3,
        dense_output_matrices=2,
        Nout=Nout,
    )
    return big_peak + float_peak
end

function zero_output_matrix(::Type{T}, n::Integer) where {T}
    return zeros(T, Int(n), Int(n))
end

function matrix_inner(A::AbstractMatrix, B::AbstractMatrix)
    size(A) == size(B) || throw(DimensionMismatch("matrix inner product"))
    T = promote_type(eltype(A), eltype(B))
    s = zero(T)
    @inbounds for j = 1:size(A, 2)
        for i = 1:size(A, 1)
            s += T(A[i, j]) * T(B[i, j])
        end
    end
    return s
end

function dominant_source(norms)
    best_name = ""
    best_val = BigFloat(-1)
    for (name, value) in norms
        v = BigFloat(value)
        if v > best_val
            best_name = name
            best_val = v
        end
    end
    return best_name
end

function compare_float64_to_big128(depth::Integer, Nout::Integer, M::Integer;
                                   identity::Bool=false)
    peak = production_peak_bytes(depth, Nout, M; identity=identity)
    if peak > MEMORY_BUDGET_BYTES
        return (status="skipped_memory_budget", estimated_peak_bytes=peak)
    end

    spec, coeffs, source = coeffs_for_mixed(depth)
    sys64 = build_transfer_system_independent(
        coeffs;
        depth=depth,
        Nout=Nout,
        M=M,
        arithmetic_type=Float64,
        kernel_name=spec.name,
        kernel_definition=spec.definition,
        coefficient_source=source,
        flip=true,
    )
    z64 = solve_T_rhs(sys64)
    b64 = rhs_state(sys64)
    V64 = apply_P(sys64, z64; output=:final)
    stats64 = streaming_residual_stats(sys64, z64)
    state_norm64 = state_fro_norm(z64)
    max_block64 = max_state_block_norm(z64)
    operator_norm64 = matrix_fro_norm(V64)

    big_result = setprecision(BigFloat, BIGFLOAT_BITS) do
        sysbig = build_transfer_system_independent(
            coeffs;
            depth=depth,
            Nout=Nout,
            M=M,
            arithmetic_type=BigFloat,
            kernel_name=spec.name,
            kernel_definition=spec.definition,
            coefficient_source=source,
            flip=true,
        )
        sys64lift = lift_transfer_system(BigFloat, sys64)
        bbig = convert_state(BigFloat, b64)
        zref = solve_T(sysbig, bbig)
        Vref = apply_P(sysbig, zref; output=:final)
        V64big = BigFloat.(V64)
        DeltaV = matrix_difference(V64big, Vref)
        err = matrix_fro_norm(DeltaV)
        normV = matrix_fro_norm(Vref)
        result = Dict{Symbol,Any}(
            :norm_b => state_fro_norm(bbig),
            :reference_state_norm => state_fro_norm(zref),
            :final_operator_norm => normV,
            :operator_error => err,
            :relative_operator_error => err / max(normV, eps(BigFloat)),
            :identity_relative_error => missing,
            :identity_absolute_error => missing,
            :norm_DeltaV_true => missing,
            :norm_DeltaV_sum => missing,
            :norm_DeltaV_T => missing,
            :norm_DeltaV_solve => missing,
            :norm_DeltaV_b => missing,
            :norm_DeltaV_P => missing,
            :relative_DeltaV_T_to_true => missing,
            :relative_DeltaV_solve_to_true => missing,
            :relative_DeltaV_b_to_true => missing,
            :relative_DeltaV_P_to_true => missing,
            :relative_DeltaV_T_to_Vstar => missing,
            :relative_DeltaV_solve_to_Vstar => missing,
            :relative_DeltaV_b_to_Vstar => missing,
            :relative_DeltaV_P_to_Vstar => missing,
            :aligned_DeltaV_T => missing,
            :aligned_DeltaV_solve => missing,
            :aligned_DeltaV_b => missing,
            :aligned_DeltaV_P => missing,
            :decomposition_relative_error => missing,
            :dominant_error_source => missing,
            :actual_output_gain => missing,
        )
        if identity
            zhat = convert_state(BigFloat, z64)
            That_zhat = apply_T(sys64lift, zhat)
            Tstar_zhat = apply_T(sysbig, zhat)
            minus_DeltaT_zhat = state_difference(Tstar_zhat, That_zhat)
            rsolve = state_difference(That_zhat, bbig)

            DeltaV_T = apply_P(sysbig, solve_T(sysbig, minus_DeltaT_zhat);
                               output=:final)
            DeltaV_solve = apply_P(sysbig, solve_T(sysbig, rsolve);
                                   output=:final)
            DeltaV_b = zero_output_matrix(BigFloat, Nout)
            Phat_zhat = apply_P(sys64lift, zhat; output=:final)
            Pstar_zhat = apply_P(sysbig, zhat; output=:final)
            DeltaV_P = matrix_difference(Phat_zhat, Pstar_zhat)

            DeltaV_sum = matrix_add(matrix_add(DeltaV_T, DeltaV_solve),
                                    matrix_add(DeltaV_b, DeltaV_P))
            DeltaV_true = matrix_difference(Phat_zhat, Vref)
            identity_error = matrix_difference(DeltaV_true, DeltaV_sum)
            norm_true = matrix_fro_norm(DeltaV_true)
            normV = matrix_fro_norm(Vref)
            norm_T = matrix_fro_norm(DeltaV_T)
            norm_solve = matrix_fro_norm(DeltaV_solve)
            norm_b = matrix_fro_norm(DeltaV_b)
            norm_P = matrix_fro_norm(DeltaV_P)
            norm_sum = matrix_fro_norm(DeltaV_sum)
            norm_rsolve = state_fro_norm(rsolve)
            result[:identity_absolute_error] = matrix_fro_norm(identity_error)
            result[:identity_relative_error] =
                matrix_fro_norm(identity_error) / max(norm_true, eps(BigFloat))
            result[:norm_DeltaV_true] = norm_true
            result[:norm_DeltaV_sum] = norm_sum
            result[:norm_DeltaV_T] = norm_T
            result[:norm_DeltaV_solve] = norm_solve
            result[:norm_DeltaV_b] = norm_b
            result[:norm_DeltaV_P] = norm_P
            result[:relative_DeltaV_T_to_true] = norm_T / max(norm_true, eps(BigFloat))
            result[:relative_DeltaV_solve_to_true] = norm_solve / max(norm_true, eps(BigFloat))
            result[:relative_DeltaV_b_to_true] = norm_b / max(norm_true, eps(BigFloat))
            result[:relative_DeltaV_P_to_true] = norm_P / max(norm_true, eps(BigFloat))
            result[:relative_DeltaV_T_to_Vstar] = norm_T / max(normV, eps(BigFloat))
            result[:relative_DeltaV_solve_to_Vstar] = norm_solve / max(normV, eps(BigFloat))
            result[:relative_DeltaV_b_to_Vstar] = norm_b / max(normV, eps(BigFloat))
            result[:relative_DeltaV_P_to_Vstar] = norm_P / max(normV, eps(BigFloat))
            result[:aligned_DeltaV_T] =
                matrix_inner(DeltaV_T, DeltaV_true) / max(norm_true, eps(BigFloat))
            result[:aligned_DeltaV_solve] =
                matrix_inner(DeltaV_solve, DeltaV_true) / max(norm_true, eps(BigFloat))
            result[:aligned_DeltaV_b] = zero(BigFloat)
            result[:aligned_DeltaV_P] =
                matrix_inner(DeltaV_P, DeltaV_true) / max(norm_true, eps(BigFloat))
            result[:decomposition_relative_error] = result[:identity_relative_error]
            result[:dominant_error_source] = dominant_source(Dict(
                "T_construction" => norm_T,
                "solve_rounding" => norm_solve,
                "b_construction" => norm_b,
                "P_construction" => norm_P,
            ))
            result[:actual_output_gain] =
                norm_solve / max(norm_rsolve, eps(BigFloat))
        end
        result
    end

    return (
        status="completed",
        estimated_peak_bytes=peak,
        kernel_tail=coeff_tail_maxabs(coeffs; tail_blocks=3),
        norm_b=big_result[:norm_b],
        state_norm=state_norm64,
        reference_state_norm=big_result[:reference_state_norm],
        max_state_block_norm=max_block64,
        residual_inf=stats64.norm_rho_inf,
        residual_fro=stats64.norm_rho_fro,
        componentwise_backward_error=stats64.beta_componentwise,
        backward_error_over_eps=stats64.beta_componentwise / eps(Float64),
        final_operator_norm=big_result[:final_operator_norm],
        float64_operator_norm=operator_norm64,
        operator_error=big_result[:operator_error],
        relative_operator_error=big_result[:relative_operator_error],
        identity_relative_error=big_result[:identity_relative_error],
        identity_absolute_error=big_result[:identity_absolute_error],
        norm_DeltaV_true=big_result[:norm_DeltaV_true],
        norm_DeltaV_sum=big_result[:norm_DeltaV_sum],
        norm_DeltaV_T=big_result[:norm_DeltaV_T],
        norm_DeltaV_solve=big_result[:norm_DeltaV_solve],
        norm_DeltaV_b=big_result[:norm_DeltaV_b],
        norm_DeltaV_P=big_result[:norm_DeltaV_P],
        relative_DeltaV_T_to_true=big_result[:relative_DeltaV_T_to_true],
        relative_DeltaV_solve_to_true=big_result[:relative_DeltaV_solve_to_true],
        relative_DeltaV_b_to_true=big_result[:relative_DeltaV_b_to_true],
        relative_DeltaV_P_to_true=big_result[:relative_DeltaV_P_to_true],
        relative_DeltaV_T_to_Vstar=big_result[:relative_DeltaV_T_to_Vstar],
        relative_DeltaV_solve_to_Vstar=big_result[:relative_DeltaV_solve_to_Vstar],
        relative_DeltaV_b_to_Vstar=big_result[:relative_DeltaV_b_to_Vstar],
        relative_DeltaV_P_to_Vstar=big_result[:relative_DeltaV_P_to_Vstar],
        aligned_DeltaV_T=big_result[:aligned_DeltaV_T],
        aligned_DeltaV_solve=big_result[:aligned_DeltaV_solve],
        aligned_DeltaV_b=big_result[:aligned_DeltaV_b],
        aligned_DeltaV_P=big_result[:aligned_DeltaV_P],
        decomposition_relative_error=big_result[:decomposition_relative_error],
        dominant_error_source=big_result[:dominant_error_source],
        actual_output_gain=big_result[:actual_output_gain],
    )
end

function primary_row()
    M = selected_M_from_padding(PRIMARY_DEPTH)
    result = compare_float64_to_big128(PRIMARY_DEPTH, NOUT, M; identity=true)
    if result.status != "completed"
        return (
            kernel=KERNEL,
            validation_mode="fixed_b_isolation",
            depth=PRIMARY_DEPTH,
            Nout=NOUT,
            M=M,
            precision_bits=BIGFLOAT_BITS,
            norm_b=missing,
            state_norm=missing,
            reference_state_norm=missing,
            max_state_block_norm=missing,
            residual_inf=missing,
            residual_fro=missing,
            componentwise_backward_error=missing,
            backward_error_over_eps=missing,
            final_operator_norm=missing,
            float64_operator_norm=missing,
            Float64_to_128_operator_error=missing,
            relative_operator_error=missing,
            identity_relative_error=missing,
            identity_absolute_error=missing,
            norm_DeltaV_true=missing,
            norm_DeltaV_sum=missing,
            norm_DeltaV_T=missing,
            norm_DeltaV_solve=missing,
            norm_DeltaV_b=missing,
            norm_DeltaV_P=missing,
            relative_DeltaV_T_to_true=missing,
            relative_DeltaV_solve_to_true=missing,
            relative_DeltaV_b_to_true=missing,
            relative_DeltaV_P_to_true=missing,
            relative_DeltaV_T_to_Vstar=missing,
            relative_DeltaV_solve_to_Vstar=missing,
            relative_DeltaV_b_to_Vstar=missing,
            relative_DeltaV_P_to_Vstar=missing,
            aligned_DeltaV_T=missing,
            aligned_DeltaV_solve=missing,
            aligned_DeltaV_b=missing,
            aligned_DeltaV_P=missing,
            decomposition_relative_error=missing,
            dominant_error_source=missing,
            actual_output_gain=missing,
            padding_error=padding_error_for(PRIMARY_DEPTH, M),
            estimated_peak_bytes=result.estimated_peak_bytes,
            memory_budget_bytes=MEMORY_BUDGET_BYTES,
            status=result.status,
        )
    end
    return (
        kernel=KERNEL,
        validation_mode="fixed_b_isolation",
        depth=PRIMARY_DEPTH,
        Nout=NOUT,
        M=M,
        precision_bits=BIGFLOAT_BITS,
        norm_b=result.norm_b,
        state_norm=result.state_norm,
        reference_state_norm=result.reference_state_norm,
        max_state_block_norm=result.max_state_block_norm,
        residual_inf=result.residual_inf,
        residual_fro=result.residual_fro,
        componentwise_backward_error=result.componentwise_backward_error,
        backward_error_over_eps=result.backward_error_over_eps,
        final_operator_norm=result.final_operator_norm,
        float64_operator_norm=result.float64_operator_norm,
        Float64_to_128_operator_error=result.operator_error,
        relative_operator_error=result.relative_operator_error,
        identity_relative_error=result.identity_relative_error,
        identity_absolute_error=result.identity_absolute_error,
        norm_DeltaV_true=result.norm_DeltaV_true,
        norm_DeltaV_sum=result.norm_DeltaV_sum,
        norm_DeltaV_T=result.norm_DeltaV_T,
        norm_DeltaV_solve=result.norm_DeltaV_solve,
        norm_DeltaV_b=result.norm_DeltaV_b,
        norm_DeltaV_P=result.norm_DeltaV_P,
        relative_DeltaV_T_to_true=result.relative_DeltaV_T_to_true,
        relative_DeltaV_solve_to_true=result.relative_DeltaV_solve_to_true,
        relative_DeltaV_b_to_true=result.relative_DeltaV_b_to_true,
        relative_DeltaV_P_to_true=result.relative_DeltaV_P_to_true,
        relative_DeltaV_T_to_Vstar=result.relative_DeltaV_T_to_Vstar,
        relative_DeltaV_solve_to_Vstar=result.relative_DeltaV_solve_to_Vstar,
        relative_DeltaV_b_to_Vstar=result.relative_DeltaV_b_to_Vstar,
        relative_DeltaV_P_to_Vstar=result.relative_DeltaV_P_to_Vstar,
        aligned_DeltaV_T=result.aligned_DeltaV_T,
        aligned_DeltaV_solve=result.aligned_DeltaV_solve,
        aligned_DeltaV_b=result.aligned_DeltaV_b,
        aligned_DeltaV_P=result.aligned_DeltaV_P,
        decomposition_relative_error=result.decomposition_relative_error,
        dominant_error_source=result.dominant_error_source,
        actual_output_gain=result.actual_output_gain,
        padding_error=padding_error_for(PRIMARY_DEPTH, M),
        estimated_peak_bytes=result.estimated_peak_bytes,
        memory_budget_bytes=MEMORY_BUDGET_BYTES,
        status="completed",
    )
end

function full_pipeline_unavailable_row()
    M = selected_M_from_padding(PRIMARY_DEPTH)
    return (
        kernel=KERNEL,
        validation_mode="full_pipeline",
        depth=PRIMARY_DEPTH,
        Nout=NOUT,
        M=M,
        precision_bits=BIGFLOAT_BITS,
        norm_b=missing,
        state_norm=missing,
        reference_state_norm=missing,
        max_state_block_norm=missing,
        residual_inf=missing,
        residual_fro=missing,
        componentwise_backward_error=missing,
        backward_error_over_eps=missing,
        final_operator_norm=missing,
        float64_operator_norm=missing,
        Float64_to_128_operator_error=missing,
        relative_operator_error=missing,
        identity_relative_error=missing,
        identity_absolute_error=missing,
        norm_DeltaV_true=missing,
        norm_DeltaV_sum=missing,
        norm_DeltaV_T=missing,
        norm_DeltaV_solve=missing,
        norm_DeltaV_b=missing,
        norm_DeltaV_P=missing,
        relative_DeltaV_T_to_true=missing,
        relative_DeltaV_solve_to_true=missing,
        relative_DeltaV_b_to_true=missing,
        relative_DeltaV_P_to_true=missing,
        relative_DeltaV_T_to_Vstar=missing,
        relative_DeltaV_solve_to_Vstar=missing,
        relative_DeltaV_b_to_Vstar=missing,
        relative_DeltaV_P_to_Vstar=missing,
        aligned_DeltaV_T=missing,
        aligned_DeltaV_solve=missing,
        aligned_DeltaV_b=missing,
        aligned_DeltaV_P=missing,
        decomposition_relative_error=missing,
        dominant_error_source=missing,
        actual_output_gain=missing,
        padding_error=padding_error_for(PRIMARY_DEPTH, M),
        estimated_peak_bytes=missing,
        memory_budget_bytes=MEMORY_BUDGET_BYTES,
        status="high_precision_projection_unavailable",
    )
end

function depth_row(depth::Integer)
    M = recommended_internal_M(NOUT, depth; safety=8)
    result = compare_float64_to_big128(depth, NOUT, M; identity=false)
    if result.status != "completed"
        return (
            kernel=KERNEL,
            depth=Int(depth),
            Nout=NOUT,
            M=M,
            precision_bits=BIGFLOAT_BITS,
            kernel_tail=missing,
            state_norm=missing,
            reference_state_norm=missing,
            residual_inf=missing,
            residual_fro=missing,
            componentwise_backward_error=missing,
            backward_error_over_eps=missing,
            final_operator_norm=missing,
            final_operator_error=missing,
            relative_operator_error=missing,
            padding_error=padding_error_for(depth, M),
            estimated_peak_bytes=result.estimated_peak_bytes,
            memory_budget_bytes=MEMORY_BUDGET_BYTES,
            status=result.status,
        )
    end
    return (
        kernel=KERNEL,
        depth=Int(depth),
        Nout=NOUT,
        M=M,
        precision_bits=BIGFLOAT_BITS,
        kernel_tail=result.kernel_tail,
        state_norm=result.state_norm,
        reference_state_norm=result.reference_state_norm,
        residual_inf=result.residual_inf,
        residual_fro=result.residual_fro,
        componentwise_backward_error=result.componentwise_backward_error,
        backward_error_over_eps=result.backward_error_over_eps,
        final_operator_norm=result.final_operator_norm,
        final_operator_error=result.operator_error,
        relative_operator_error=result.relative_operator_error,
        padding_error=padding_error_for(depth, M),
        estimated_peak_bytes=result.estimated_peak_bytes,
        memory_budget_bytes=MEMORY_BUDGET_BYTES,
        status="completed",
    )
end

function main()
    mkpath(DATA_DIR)
    row = primary_row()
    full_row = full_pipeline_unavailable_row()
    write_csv(OUTPUT_CSV, OUTPUT_COLUMNS, [row, full_row])
    @printf("output d=%d Nout=%d M=%d err=%s rel=%s beta/eps=%s decomp=%s dominant=%s\n",
            row.depth, row.Nout, row.M,
            sci(row.Float64_to_128_operator_error),
            sci(row.relative_operator_error),
            sci(row.backward_error_over_eps),
            sci(row.decomposition_relative_error),
            string(row.dominant_error_source))
    flush(stdout)
    GC.gc(true)

    depth_rows = NamedTuple[]
    for depth in DEPTHS
        drow = depth_row(depth)
        push!(depth_rows, drow)
        write_csv(DEPTH_CSV, DEPTH_COLUMNS, depth_rows)
        @printf("depth d=%d M=%d err=%s rel=%s beta/eps=%s\n",
                drow.depth, drow.M, sci(drow.final_operator_error),
                sci(drow.relative_operator_error),
                sci(drow.backward_error_over_eps))
        flush(stdout)
        GC.gc(true)
    end
    write_csv(DEPTH_CSV, DEPTH_COLUMNS, depth_rows)
    println("Wrote CSV: ", OUTPUT_CSV)
    println("Wrote CSV: ", DEPTH_CSV)
    return row, depth_rows
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
