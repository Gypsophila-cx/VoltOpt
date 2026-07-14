# Independent construction validation for T_{d,M} and P_{Nout,M}.

include(joinpath(@__DIR__, "go_instability_core.jl"))
using .GOInstabilityCore
using LinearAlgebra
using Printf
using Random
using Statistics

const SCRIPT_DIR = @__DIR__
const DATA_DIR = joinpath(SCRIPT_DIR, "data")
const OUTPUT_CSV = joinpath(DATA_DIR, "go_construction_validation.csv")
const BIGFLOAT_BITS = 128
const MEMORY_BUDGET_BYTES = default_memory_budget_bytes()
const EPS64 = eps(Float64)
const TINY128 = BigFloat(2.0)^(-120)
const PRODUCTION_NOUT = 72
const PRODUCTION_DEPTHS = (8, 16, 24, 32, 36)
const EXPLICIT_DEPTHS = (2, 3, 4, 5)
const P_STAGES = (:extraction, :lowering, :reflection, :conversion, :crop, :full)

const COLUMNS = (
    :record_type,
    :object_name,
    :precision_reference,
    :depth,
    :level,
    :Nout,
    :M,
    :probe_name,
    :rows,
    :cols,
    :norm_64,
    :norm_128,
    :absolute_construction_error,
    :relative_construction_error,
    :construction_error_over_eps,
    :componentwise_construction_error,
    :componentwise_error_over_eps,
    :application_rounding_error,
    :application_rounding_error_over_eps,
    :total_action_error,
    :total_action_error_over_eps,
    :sparsity_pattern_mismatch_count,
    :unexpected_nonzero_count,
    :diagonal_unit_block_error,
    :max_absolute_zero_pattern_error,
    :left_inverse_defect_64,
    :left_inverse_defect_64_recomputed_128,
    :left_inverse_defect_128,
    :status,
)

function row(; record_type, object_name, depth, Nout, M,
             precision_reference="independent_128bit_analytic",
             level=missing, probe_name="", rows=missing, cols=missing,
             norm_64=missing, norm_128=missing,
             absolute_construction_error=missing,
             relative_construction_error=missing,
             construction_error_over_eps=missing,
             componentwise_construction_error=missing,
             componentwise_error_over_eps=missing,
             application_rounding_error=missing,
             application_rounding_error_over_eps=missing,
             total_action_error=missing,
             total_action_error_over_eps=missing,
             sparsity_pattern_mismatch_count=missing,
             unexpected_nonzero_count=missing,
             diagonal_unit_block_error=missing,
             max_absolute_zero_pattern_error=missing,
             left_inverse_defect_64=missing,
             left_inverse_defect_64_recomputed_128=missing,
             left_inverse_defect_128=missing,
             status="completed")
    return (
        record_type=String(record_type),
        object_name=String(object_name),
        precision_reference=String(precision_reference),
        depth=Int(depth),
        level=level,
        Nout=Int(Nout),
        M=Int(M),
        probe_name=String(probe_name),
        rows=rows,
        cols=cols,
        norm_64=norm_64,
        norm_128=norm_128,
        absolute_construction_error=absolute_construction_error,
        relative_construction_error=relative_construction_error,
        construction_error_over_eps=construction_error_over_eps,
        componentwise_construction_error=componentwise_construction_error,
        componentwise_error_over_eps=componentwise_error_over_eps,
        application_rounding_error=application_rounding_error,
        application_rounding_error_over_eps=application_rounding_error_over_eps,
        total_action_error=total_action_error,
        total_action_error_over_eps=total_action_error_over_eps,
        sparsity_pattern_mismatch_count=sparsity_pattern_mismatch_count,
        unexpected_nonzero_count=unexpected_nonzero_count,
        diagonal_unit_block_error=diagonal_unit_block_error,
        max_absolute_zero_pattern_error=max_absolute_zero_pattern_error,
        left_inverse_defect_64=left_inverse_defect_64,
        left_inverse_defect_64_recomputed_128=left_inverse_defect_64_recomputed_128,
        left_inverse_defect_128=left_inverse_defect_128,
        status=String(status),
    )
end

function coeffs_for_mixed(depth::Integer)
    spec = default_kernel_specs()[1]
    return spec, make_coeffs(spec, depth; flip=true),
           "native_float64_projection_fixed_for_b_isolation"
end

function independent_systems(depth::Integer, Nout::Integer, M::Integer)
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
    sys128 = build_transfer_system_independent(
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
    return sys64, sys128, sys64lift
end

function dense_big(A)
    return BigFloat.(Matrix(A))
end

function fro_dense(A)
    s = zero(eltype(A) <: BigFloat ? BigFloat : Float64)
    @inbounds for v in A
        s += abs2(v)
    end
    return sqrt(s)
end

function matrix_metrics(A64, A128; diagonal_unit::Bool=false)
    A64b = dense_big(A64)
    A128m = Matrix(A128)
    rows, cols = size(A128m)
    diff = A64b - A128m
    abs_err = fro_dense(diff)
    norm64 = fro_dense(A64b)
    norm128 = fro_dense(A128m)
    rel_err = abs_err / max(norm128, TINY128)

    row_scale = [maximum(abs.(A128m[i, :])) for i = 1:rows]
    comp = zero(BigFloat)
    mismatch = 0
    unexpected = 0
    max_zero = zero(BigFloat)
    @inbounds for j = 1:cols
        for i = 1:rows
            ztol = max(row_scale[i], one(BigFloat)) *
                   eps(BigFloat) * BigFloat(4096)
            ref_zero = abs(A128m[i, j]) <= ztol
            got_zero = abs(A64b[i, j]) <= ztol
            mismatch += ref_zero != got_zero ? 1 : 0
            if ref_zero
                if !got_zero
                    unexpected += 1
                    max_zero = max(max_zero, abs(A64b[i, j]))
                end
            else
                denom = max(abs(A128m[i, j]), row_scale[i] * eps(BigFloat))
                comp = max(comp, abs(diff[i, j]) / denom)
            end
        end
    end
    diagerr = missing
    if diagonal_unit
        diagerr = zero(BigFloat)
        @inbounds for i = 1:min(rows, cols)
            diagerr = max(diagerr, abs(A64b[i, i] - one(BigFloat)))
        end
    end
    return (
        rows=rows,
        cols=cols,
        norm_64=norm64,
        norm_128=norm128,
        abs_err=abs_err,
        rel_err=rel_err,
        comp_err=comp,
        mismatch=mismatch,
        unexpected=unexpected,
        diagerr=diagerr,
        max_zero=max_zero,
    )
end

function matrix_compare_row(record_type, object_name, depth, Nout, M,
                            A64, A128; level=missing,
                            probe_name="", diagonal_unit=false,
                            left_defects=(missing, missing, missing))
    m = matrix_metrics(A64, A128; diagonal_unit=diagonal_unit)
    return row(
        record_type=record_type,
        object_name=object_name,
        depth=depth,
        level=level,
        Nout=Nout,
        M=M,
        probe_name=probe_name,
        rows=m.rows,
        cols=m.cols,
        norm_64=m.norm_64,
        norm_128=m.norm_128,
        absolute_construction_error=m.abs_err,
        relative_construction_error=m.rel_err,
        construction_error_over_eps=m.rel_err / EPS64,
        componentwise_construction_error=m.comp_err,
        componentwise_error_over_eps=m.comp_err / EPS64,
        sparsity_pattern_mismatch_count=m.mismatch,
        unexpected_nonzero_count=m.unexpected,
        diagonal_unit_block_error=m.diagerr,
        max_absolute_zero_pattern_error=m.max_zero,
        left_inverse_defect_64=left_defects[1],
        left_inverse_defect_64_recomputed_128=left_defects[2],
        left_inverse_defect_128=left_defects[3],
    )
end

function state_action_metrics(sys64, sys128, sys64lift, x64)
    x128 = convert_state(BigFloat, x64)
    y_ref = apply_T(sys128, x128)
    y_construct64 = apply_T(sys64lift, x128)
    y_native64 = convert_state(BigFloat, apply_T(sys64, x64))
    ref_norm = max(state_fro_norm(y_ref), TINY128)
    construct = state_fro_norm(state_difference(y_construct64, y_ref)) / ref_norm
    rounding = state_fro_norm(state_difference(y_native64, y_construct64)) / ref_norm
    total = state_fro_norm(state_difference(y_native64, y_ref)) / ref_norm
    return construct, rounding, total
end

function P_action_metrics(sys64, sys128, sys64lift, x64)
    x128 = convert_state(BigFloat, x64)
    v_ref = apply_P(sys128, x128; output=:final)
    v_construct64 = apply_P(sys64lift, x128; output=:final)
    v_native64 = BigFloat.(apply_P(sys64, x64; output=:final))
    ref_norm = max(matrix_fro_norm(v_ref), TINY128)
    construct = matrix_fro_norm(matrix_difference(v_construct64, v_ref)) / ref_norm
    rounding = matrix_fro_norm(matrix_difference(v_native64, v_construct64)) / ref_norm
    total = matrix_fro_norm(matrix_difference(v_native64, v_ref)) / ref_norm
    return construct, rounding, total
end

function action_row(record_type, object_name, depth, Nout, M, probe_name,
                    construction, rounding, total)
    return row(
        record_type=record_type,
        object_name=object_name,
        depth=depth,
        Nout=Nout,
        M=M,
        probe_name=probe_name,
        relative_construction_error=construction,
        construction_error_over_eps=construction / EPS64,
        application_rounding_error=rounding,
        application_rounding_error_over_eps=rounding / EPS64,
        total_action_error=total,
        total_action_error_over_eps=total / EPS64,
    )
end

function jacobi_banded_from_sys(sys::GOTransferSystem{T}) where {T}
    J = GOInstabilityCore.banded_zero(T, sys.M, 1, 1)
    @inbounds for i = 1:sys.M
        J[i, i] = sys.diag[i]
        if i < sys.M
            J[i + 1, i] = sys.subdiag[i]
            J[i, i + 1] = sys.supdiag[i]
        end
    end
    return J
end

function left_inverse_defects(::Type{T}, k::Integer) where {T}
    kk = Int(k)
    raw_x = analytic_triangle_jacobi_block(T, :x, kk + 1, kk)
    raw_y = analytic_triangle_jacobi_block(T, :y, kk + 1, kk)
    Bx_all, By_all, _, _ = analytic_clenshaw_recurrence_blocks(T, kk)
    Bplus = hcat(Matrix(Bx_all[kk])', Matrix(By_all[kk])')
    Braw = vcat(raw_x', raw_y')
    defect = Bplus * Braw - Matrix{T}(I, kk + 1, kk + 1)
    return fro_dense(defect)
end

function left_inverse_defects_for_level(k::Integer)
    d64 = left_inverse_defects(Float64, k)
    raw_x64 = BigFloat.(analytic_triangle_jacobi_block(Float64, :x, k + 1, k))
    raw_y64 = BigFloat.(analytic_triangle_jacobi_block(Float64, :y, k + 1, k))
    Bx64, By64, _, _ = analytic_clenshaw_recurrence_blocks(Float64, k)
    Bplus64 = hcat(BigFloat.(Matrix(Bx64[k]))',
                   BigFloat.(Matrix(By64[k]))')
    Braw64 = vcat(raw_x64', raw_y64')
    d64re = fro_dense(Bplus64 * Braw64 -
                      Matrix{BigFloat}(I, k + 1, k + 1))
    d128 = left_inverse_defects(BigFloat, k)
    return d64, d64re, d128
end

function run_explicit_global!(rows)
    for depth in EXPLICIT_DEPTHS
        Nout = depth + 2
        M = Nout + depth + 8
        sys64, sys128, _ = independent_systems(depth, Nout, M)
        T64 = explicit_T_matrix(sys64)
        T128 = explicit_T_matrix(sys128)
        push!(rows, matrix_compare_row(
            "explicit_global_T", "T_explicit_Frobenius", depth, Nout, M,
            T64, T128; diagonal_unit=true,
        ))
        P64 = explicit_P_matrix(sys64; output=:final)
        P128 = explicit_P_matrix(sys128; output=:final)
        push!(rows, matrix_compare_row(
            "explicit_global_P", "P_explicit_Frobenius", depth, Nout, M,
            P64, P128,
        ))
        @printf("explicit d=%d dim=%d relT=%s relP=%s\n",
                depth, size(T64, 1),
                sci(rows[end - 1].relative_construction_error),
                sci(rows[end].relative_construction_error))
        flush(stdout)
        GC.gc(true)
    end
end

function run_local_blocks!(rows)
    for depth in PRODUCTION_DEPTHS
        Nout = PRODUCTION_NOUT
        M = recommended_internal_M(Nout, depth; safety=8)
        sys64, sys128, _ = independent_systems(depth, Nout, M)
        for k = 1:(depth - 1)
            defects = left_inverse_defects_for_level(k)
            raw_x64 = analytic_triangle_jacobi_block(Float64, :x, k + 1, k)
            raw_x128 = analytic_triangle_jacobi_block(BigFloat, :x, k + 1, k)
            raw_y64 = analytic_triangle_jacobi_block(Float64, :y, k + 1, k)
            raw_y128 = analytic_triangle_jacobi_block(BigFloat, :y, k + 1, k)
            push!(rows, matrix_compare_row("local_T_block", "B_n_raw_x", depth,
                                           Nout, M, raw_x64, raw_x128;
                                           level=k, left_defects=defects))
            push!(rows, matrix_compare_row("local_T_block", "B_n_raw_y", depth,
                                           Nout, M, raw_y64, raw_y128;
                                           level=k, left_defects=defects))
            push!(rows, matrix_compare_row("local_T_block", "B_n_plus_x", depth,
                                           Nout, M, sys64.Bx[k], sys128.Bx[k];
                                           level=k, left_defects=defects))
            push!(rows, matrix_compare_row("local_T_block", "B_n_plus_y", depth,
                                           Nout, M, sys64.By[k], sys128.By[k];
                                           level=k, left_defects=defects))
            push!(rows, matrix_compare_row("local_T_block",
                                           "preconditioned_first_coupling_B",
                                           depth, Nout, M, sys64.B[k],
                                           sys128.B[k]; level=k,
                                           left_defects=defects))
            if k <= length(sys64.Ctail)
                push!(rows, matrix_compare_row("local_T_block",
                                               "preconditioned_second_coupling_C",
                                               depth, Nout, M, sys64.Ctail[k],
                                               sys128.Ctail[k]; level=k,
                                               left_defects=defects))
            end
            identity_norm = sqrt(BigFloat(k + 1))
            defect_rel = defects[2] / identity_norm
            push!(rows, row(
                record_type="local_T_block",
                object_name="B_n_plus_B_n_minus_I",
                depth=depth,
                level=k,
                Nout=Nout,
                M=M,
                rows=k + 1,
                cols=k + 1,
                norm_64=defects[1],
                norm_128=defects[3],
                absolute_construction_error=defects[2],
                relative_construction_error=defect_rel,
                construction_error_over_eps=defect_rel / EPS64,
                left_inverse_defect_64=defects[1],
                left_inverse_defect_64_recomputed_128=defects[2],
                left_inverse_defect_128=defects[3],
            ))
        end
        J64 = jacobi_banded_from_sys(sys64)
        J128 = jacobi_banded_from_sys(sys128)
        push!(rows, matrix_compare_row("local_T_block", "left_Jacobi_block",
                                       depth, Nout, M, J64, J128; level=0))
        push!(rows, matrix_compare_row("local_T_block", "right_Jacobi_block",
                                       depth, Nout, M, J64, J128; level=0))
        @printf("local blocks d=%d rows=%d\n", depth, length(rows))
        flush(stdout)
        GC.gc(true)
    end
end

function normalized_values(vals::Vector{Float64})
    nrm = norm(vals)
    nrm == 0 && return vals
    return vals ./ nrm
end

function probe_vector(sys, name::Symbol)
    dim = state_dimension(sys)
    vals = zeros(Float64, dim)
    if name == :low_basis
        vals[1] = 1.0
    elseif name == :highest_level_basis
        layout = state_layout_rows(sys)
        last_level = layout[end]
        idx = last_level.first_state_index +
              min(last_level.entries_per_block - 1, max(0, sys.M ÷ 2))
        vals[idx] = 1.0
    elseif name == :alternating
        vals = [isodd(i) ? 1.0 : -1.0 for i = 1:dim]
        vals = normalized_values(vals)
    elseif name == :smooth_ramp
        vals = collect(range(0.0, 1.0; length=dim))
        vals = normalized_values(vals)
    elseif name == :fixed_random
        rng = MersenneTwister(20260710)
        vals = randn(rng, dim)
        vals = normalized_values(vals)
    else
        throw(ArgumentError("unknown probe $name"))
    end
    return unflatten_state(sys, vals)
end

function summary_action_rows(record_type, object_name, depth, Nout, M, triples)
    constructs = [t[1] for t in triples]
    rounds = [t[2] for t in triples]
    totals = [t[3] for t in triples]
    return (
        action_row(record_type, object_name, depth, Nout, M, "max_over_probes",
                   maximum(constructs), maximum(rounds), maximum(totals)),
        action_row(record_type, object_name, depth, Nout, M, "median_over_probes",
                   median(constructs), median(rounds), median(totals)),
    )
end

function run_matrix_free_probes!(rows)
    probe_names = (:low_basis, :highest_level_basis, :alternating,
                   :smooth_ramp, :fixed_random)
    for depth in PRODUCTION_DEPTHS
        Nout = PRODUCTION_NOUT
        M = recommended_internal_M(Nout, depth; safety=8)
        peak = estimate_peak_bytes(depth, M, BigFloat; bits=BIGFLOAT_BITS,
                                   complete_states=4,
                                   work_state_equivalents=4,
                                   dense_output_matrices=4,
                                   Nout=Nout)
        if peak > MEMORY_BUDGET_BYTES
            push!(rows, row(record_type="matrix_free_T_probe",
                            object_name="T_action", depth=depth,
                            Nout=Nout, M=M, status="skipped_memory_budget"))
            push!(rows, row(record_type="matrix_free_P_probe",
                            object_name="P_action", depth=depth,
                            Nout=Nout, M=M, status="skipped_memory_budget"))
            continue
        end
        sys64, sys128, sys64lift = independent_systems(depth, Nout, M)
        Ttriples = Tuple{BigFloat,BigFloat,BigFloat}[]
        Ptriples = Tuple{BigFloat,BigFloat,BigFloat}[]
        for pname in probe_names
            x64 = probe_vector(sys64, pname)
            Tm = state_action_metrics(sys64, sys128, sys64lift, x64)
            Pm = P_action_metrics(sys64, sys128, sys64lift, x64)
            push!(Ttriples, Tm)
            push!(Ptriples, Pm)
            push!(rows, action_row("matrix_free_T_probe", "T_action",
                                   depth, Nout, M, String(pname), Tm...))
            push!(rows, action_row("matrix_free_P_probe", "P_action",
                                   depth, Nout, M, String(pname), Pm...))
            GC.gc(true)
        end
        append!(rows, summary_action_rows("matrix_free_T_probe", "T_action",
                                          depth, Nout, M, Ttriples))
        append!(rows, summary_action_rows("matrix_free_P_probe", "P_action",
                                          depth, Nout, M, Ptriples))
        @printf("probes d=%d maxT=%s maxP=%s\n", depth,
                sci(maximum(t[3] for t in Ttriples)),
                sci(maximum(t[3] for t in Ptriples)))
        flush(stdout)
        GC.gc(true)
    end
end

function stage_metrics(sys64, sys128, sys64lift, x64, stage)
    x128 = convert_state(BigFloat, x64)
    ref = apply_P_stage(sys128, x128, stage)
    construct = apply_P_stage(sys64lift, x128, stage)
    native = BigFloat.(Matrix(apply_P_stage(sys64, x64, stage)))
    refm = Matrix(ref)
    constructm = Matrix(construct)
    ref_norm = max(fro_dense(refm), TINY128)
    c = fro_dense(constructm - refm) / ref_norm
    r = fro_dense(native - constructm) / ref_norm
    t = fro_dense(native - refm) / ref_norm
    return c, r, t, matrix_metrics(apply_P_stage(sys64, x64, stage), ref)
end

function run_P_stages!(rows)
    depth = 36
    Nout = PRODUCTION_NOUT
    M = recommended_internal_M(Nout, depth; safety=8)
    sys64, sys128, sys64lift = independent_systems(depth, Nout, M)
    x64 = probe_vector(sys64, :smooth_ramp)
    for stage in P_STAGES
        c, r, t, mm = stage_metrics(sys64, sys128, sys64lift, x64, stage)
        push!(rows, row(
            record_type="local_P_stage",
            object_name=String(stage),
            depth=depth,
            level=missing,
            Nout=Nout,
            M=M,
            probe_name="smooth_ramp",
            rows=mm.rows,
            cols=mm.cols,
            norm_64=mm.norm_64,
            norm_128=mm.norm_128,
            absolute_construction_error=mm.abs_err,
            relative_construction_error=c,
            construction_error_over_eps=c / EPS64,
            componentwise_construction_error=mm.comp_err,
            componentwise_error_over_eps=mm.comp_err / EPS64,
            application_rounding_error=r,
            application_rounding_error_over_eps=r / EPS64,
            total_action_error=t,
            total_action_error_over_eps=t / EPS64,
            sparsity_pattern_mismatch_count=mm.mismatch,
            unexpected_nonzero_count=mm.unexpected,
            max_absolute_zero_pattern_error=mm.max_zero,
        ))
    end
    @printf("P stages d=%d M=%d\n", depth, M)
    flush(stdout)
    GC.gc(true)
end

function main()
    mkpath(DATA_DIR)
    rows = NamedTuple[]
    setprecision(BigFloat, BIGFLOAT_BITS) do
        run_explicit_global!(rows)
        write_csv(OUTPUT_CSV, COLUMNS, rows)
        run_local_blocks!(rows)
        write_csv(OUTPUT_CSV, COLUMNS, rows)
        run_matrix_free_probes!(rows)
        write_csv(OUTPUT_CSV, COLUMNS, rows)
        run_P_stages!(rows)
    end
    write_csv(OUTPUT_CSV, COLUMNS, rows)
    println("Wrote CSV: ", OUTPUT_CSV)
    return rows
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
