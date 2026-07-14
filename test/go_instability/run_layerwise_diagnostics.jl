# Layerwise generation and propagation diagnostics for the GO Clenshaw solve.

include(joinpath(@__DIR__, "go_instability_core.jl"))
using .GOInstabilityCore
using LinearAlgebra
using Printf
using Statistics

const GOCore = GOInstabilityCore
const SCRIPT_DIR = @__DIR__
const DATA_DIR = joinpath(SCRIPT_DIR, "data")
const OUTPUT_CSV = joinpath(DATA_DIR, "go_layerwise_diagnostics.csv")
const MEMORY_BUDGET_BYTES = default_memory_budget_bytes()
const BIGFLOAT_BITS = 128
const NOUT = 72
const DEPTHS = (16, 24, 32, 36)
const PRIMARY_DEPTH = 36
const SMALL_DEPTHS = (2, 3, 4, 5)
const KERNEL = "mixed_cos20xy"
const JULIA_EXE = joinpath(Sys.BINDIR, Base.julia_exename())
const GO_PROJECT = joinpath(normpath(joinpath(SCRIPT_DIR, "..", "..")),
                            "others", "GutlebOlver")

const COLUMNS = (
    :record_type, :kernel, :depth, :Nout, :M, :precision_bits,
    :level_index, :code_index, :total_degree, :distance_to_output,
    :distance_to_terminal, :propagation_distance, :probe_or_mode,
    :layer_state_norm, :layer_state_maxabs, :rhs_norm, :A_term_norm,
    :C_term_norm, :local_residual_norm, :local_residual_maxabs,
    :local_normwise_backward_error, :local_componentwise_backward_error,
    :local_backward_error_over_eps, :output_contribution_norm,
    :actual_output_gain, :worst_case_layer_gain, :actual_to_worst_ratio,
    :relative_contribution_to_total, :alignment_with_total,
    :signed_projection_fraction, :local_injection_norm, :A_inherited_norm,
    :C_inherited_norm, :total_layer_error_norm, :combination_ratio,
    :A_fraction, :C_fraction, :cumulative_layers_added,
    :cumulative_output_error_norm, :cumulative_relative_to_total,
    :residual_block_reconstruction_error,
    :layer_contribution_reconstruction_error, :signed_projection_sum_error,
    :dominant_residual_layer, :dominant_residual_total_degree,
    :dominant_contribution_layer, :dominant_contribution_total_degree,
    :max_actual_output_gain, :median_actual_output_gain,
    :deepest_quartile_contribution_norm,
    :shallowest_quartile_contribution_norm, :layers_for_50_percent,
    :layers_for_90_percent, :layers_for_99_percent,
    :sum_A_inherited_norm, :sum_C_inherited_norm,
    :A_channel_fraction, :C_channel_fraction, :total_reconstruction_error,
    :estimated_peak_bytes, :memory_budget_bytes, :reconstruction_error,
    :status,
)

function empty_record()
    return NamedTuple{COLUMNS}(ntuple(_ -> missing, length(COLUMNS)))
end

record(; kwargs...) = merge(empty_record(), (; kwargs...))

scalar_type(sys::GOTransferSystem{T}) where {T} = T

function copy_level(level)
    return [GOCore.copy_banded(A) for A in level]
end

function convert_level(::Type{T}, level) where {T}
    return [convert_banded(T, A) for A in level]
end

function level_fro_norm(level)
    isempty(level) && return 0.0
    T = eltype(first(level))
    s = zero(T <: BigFloat ? BigFloat : Float64)
    for A in level
        l, u = GOCore.bandwidths_int(A)
        @inbounds for j = 1:size(A, 2)
            for i = max(1, j - u):min(size(A, 1), j + l)
                s += abs2(A[i, j])
            end
        end
    end
    return sqrt(s)
end

function level_maxabs(level)
    isempty(level) && return 0.0
    T = eltype(first(level))
    best = zero(T <: BigFloat ? BigFloat : Float64)
    for A in level
        l, u = GOCore.bandwidths_int(A)
        @inbounds for j = 1:size(A, 2)
            for i = max(1, j - u):min(size(A, 1), j + l)
                best = max(best, abs(A[i, j]))
            end
        end
    end
    return best
end

function level_difference_sums(X, Y)
    diff2 = zero(BigFloat)
    ynorm2 = zero(BigFloat)
    for block = 1:length(X)
        A = X[block]
        B = Y[block]
        l, u = GOCore.bandwidths_int(A)
        @inbounds for j = 1:size(A, 2)
            for i = max(1, j - u):min(size(A, 1), j + l)
                diff2 += abs2(A[i, j] - B[i, j])
                ynorm2 += abs2(B[i, j])
            end
        end
    end
    return diff2, ynorm2
end

function matrix_axpy!(Y, alpha, X)
    size(Y) == size(X) || throw(DimensionMismatch("matrix axpy"))
    @inbounds for j = 1:size(Y, 2), i = 1:size(Y, 1)
        Y[i, j] += alpha * X[i, j]
    end
    return Y
end

function matrix_inner(A, B)
    size(A) == size(B) || throw(DimensionMismatch("matrix inner product"))
    T = promote_type(eltype(A), eltype(B))
    s = zero(T)
    @inbounds for j = 1:size(A, 2), i = 1:size(A, 1)
        s += conj(T(A[i, j])) * T(B[i, j])
    end
    return s
end

function output_from_level(sys::GOTransferSystem{T}, level) where {T}
    state = GOClenshawState{T}([level])
    return apply_P(sys, state; output=:final)
end

# add_forward_couplings! returns -A_k next - C_k nextnext.  Split those
# channels so their inherited directions can be measured independently.
function forward_parts(sys::GOTransferSystem{T}, k::Integer, next, nextnext) where {T}
    A_part = GOCore.zero_level(sys, k)
    C_part = GOCore.zero_level(sys, k)
    if next !== nothing
        GOCore.add_forward_couplings!(sys, A_part, k, next, nothing)
    end
    if nextnext !== nothing && k <= length(sys.Ctail)
        target = GOCore.level_bandwidth(sys, k)
        cterms = GOCore.mat_state_mul(sys.Ctail[k], nextnext, target, target)
        GOCore.add_level!(C_part, cterms, -one(T))
    end
    return A_part, C_part
end

# Reproduce apply_T's operation order at one level.  This independently
# reevaluates the lifted Float64 recurrence in 128-bit arithmetic.
function local_residual_level(sys::GOTransferSystem{T}, zlevel, next,
                              nextnext, blevel, k::Integer) where {T}
    rho = copy_level(zlevel)
    if k < sys.depth
        raw_target = max(sys.depth - k, 0)
        target = GOCore.banded_clamp(sys.n, raw_target)
        pretarget = GOCore.banded_clamp(sys.n, max(raw_target - 1, 0))
        xstate = GOCore.mat_state_mul(sys.Bx[k], next, pretarget, pretarget)
        xterms = [GOCore.leftmul_tridiag_banded(
                      sys.subdiag, sys.diag, sys.supdiag, xstate[i],
                      target, target) for i = 1:k]
        GOCore.add_level!(rho, xterms, -one(T))

        ystate = GOCore.mat_state_mul(sys.By[k], next, pretarget, pretarget)
        yterms = [GOCore.rightmul_tridiag_banded(
                      ystate[i], sys.subdiag, sys.diag, sys.supdiag,
                      target, target) for i = 1:k]
        GOCore.add_level!(rho, yterms, -one(T))

        bterms = GOCore.mat_state_mul(sys.B[k], next, target, target)
        GOCore.add_level!(rho, bterms, one(T))
        if k + 2 <= sys.depth && k <= length(sys.Ctail)
            cterms = GOCore.mat_state_mul(sys.Ctail[k], nextnext,
                                        target, target)
            GOCore.add_level!(rho, cterms, one(T))
        end
    end
    GOCore.add_level!(rho, blevel, -one(T))
    return rho
end

function lifted_local_inputs(z64, b64, k::Integer)
    d = length(z64.levels)
    zlevel = convert_level(BigFloat, z64.levels[k])
    next = k < d ? convert_level(BigFloat, z64.levels[k + 1]) : nothing
    nextnext = k + 2 <= d ? convert_level(BigFloat, z64.levels[k + 2]) : nothing
    blevel = convert_level(BigFloat, b64.levels[k])
    return zlevel, next, nextnext, blevel
end

function reevaluate_rho(sys64lift, z64, b64, k::Integer)
    zlevel, next, nextnext, blevel = lifted_local_inputs(z64, b64, k)
    return local_residual_level(sys64lift, zlevel, next, nextnext, blevel, k)
end

function componentwise_backward_error(sys, zlevel, next, nextnext, blevel,
                                      rho, k, tiny)
    denominator = GOCore.abs_level(zlevel)
    GOCore.add_level!(denominator, GOCore.abs_level(blevel),
                     one(scalar_type(sys)))
    if k < sys.depth
        next_abs = GOCore.abs_level(next)
        nextnext_abs = nextnext === nothing ? nothing : GOCore.abs_level(nextnext)
        GOCore.add_abs_forward_couplings!(sys, denominator, k, next_abs,
                                        nextnext_abs)
    end

    best = zero(scalar_type(sys))
    for block = 1:length(rho)
        R = rho[block]
        D = denominator[block]
        l, u = GOCore.bandwidths_int(R)
        @inbounds for j = 1:size(R, 2)
            for i = max(1, j - u):min(size(R, 1), j + l)
                best = max(best, abs(R[i, j]) / max(abs(D[i, j]), tiny))
            end
        end
    end
    return best
end

function layer_coordinates(k::Integer, depth::Integer)
    return (
        level_index=Int(k),
        code_index=Int(k),
        total_degree=Int(k) - 1,
        distance_to_output=Int(k) - 1,
        distance_to_terminal=Int(depth) - Int(k),
        propagation_distance=Int(k) - 1,
    )
end

function production_peak_bytes(depth::Integer, M::Integer)
    return estimate_peak_bytes(
        depth, M, BigFloat;
        bits=BIGFLOAT_BITS,
        complete_states=4,
        work_state_equivalents=3,
        dense_output_matrices=8,
        Nout=NOUT,
    ) + estimate_peak_bytes(
        depth, M, Float64;
        complete_states=2,
        work_state_equivalents=2,
        dense_output_matrices=2,
        Nout=NOUT,
    )
end

function full_error_recurrence(sys, rho_provider, d::Integer; emit_rows::Bool)
    next = nothing
    nextnext = nothing
    rows = NamedTuple[]
    sum_A = zero(BigFloat)
    sum_C = zero(BigFloat)

    for k = d:-1:1
        rho = rho_provider(k)
        A_part, C_part = forward_parts(sys, k, next, nextnext)
        current = rho
        GOCore.add_level!(current, A_part, one(BigFloat))
        GOCore.add_level!(current, C_part, one(BigFloat))
        local_norm = level_fro_norm(rho)
        A_norm = level_fro_norm(A_part)
        C_norm = level_fro_norm(C_part)
        total_norm = level_fro_norm(current)
        denom = local_norm + A_norm + C_norm
        inherited = A_norm + C_norm
        sum_A += A_norm
        sum_C += C_norm
        if emit_rows
            coords = layer_coordinates(k, d)
            push!(rows, record(;
                record_type="total_error_recurrence_step", kernel=KERNEL,
                depth=d, Nout=sys.Nout, M=sys.M,
                precision_bits=BIGFLOAT_BITS, coords...,
                probe_or_mode="highest_degree_to_output",
                local_injection_norm=local_norm,
                A_inherited_norm=A_norm, C_inherited_norm=C_norm,
                total_layer_error_norm=total_norm,
                combination_ratio=denom == 0 ? missing : total_norm / denom,
                A_fraction=inherited == 0 ? missing : A_norm / inherited,
                C_fraction=inherited == 0 ? missing : C_norm / inherited,
                memory_budget_bytes=MEMORY_BUDGET_BYTES, status="completed",
            ))
        end
        nextnext = next
        next = current
    end
    return output_from_level(sys, next), rows, sum_A, sum_C
end

function propagate_one_source(sys, rho_level, source_level::Integer)
    next = copy_level(rho_level)
    nextnext = nothing
    for k = (Int(source_level) - 1):-1:1
        A_part, C_part = forward_parts(sys, k, next, nextnext)
        current = A_part
        GOCore.add_level!(current, C_part, one(BigFloat))
        nextnext = next
        next = current
    end
    return output_from_level(sys, next)
end

function analyze_depth(depth::Integer; emit_detail::Bool=false)
    d = Int(depth)
    M = recommended_internal_M(NOUT, d; safety=8)
    peak = production_peak_bytes(d, M)
    @printf("layerwise allocation estimate d=%d Nout=%d M=%d: %.3f GiB (budget %.3f GiB)\n",
            d, NOUT, M, peak / 1024.0^3, MEMORY_BUDGET_BYTES / 1024.0^3)
    flush(stdout)
    peak <= MEMORY_BUDGET_BYTES || error("estimated layerwise peak exceeds 1.0 GiB")

    spec = default_kernel_specs()[1]
    coeffs = make_coeffs(spec, d; flip=true)
    source = "native_float64_projection_fixed_for_b_isolation"
    sys64 = build_transfer_system_independent(
        coeffs; depth=d, Nout=NOUT, M=M, arithmetic_type=Float64,
        kernel_name=spec.name, kernel_definition=spec.definition,
        coefficient_source=source, flip=true)
    b64 = rhs_state(sys64)
    z64 = solve_T_rhs(sys64)

    return setprecision(BigFloat, BIGFLOAT_BITS) do
        tiny = BigFloat(2)^(-120)
        sys128 = build_transfer_system_independent(
            coeffs; depth=d, Nout=NOUT, M=M, arithmetic_type=BigFloat,
            kernel_name=spec.name, kernel_definition=spec.definition,
            coefficient_source=source, flip=true)
        sys64lift = lift_transfer_system(BigFloat, sys64)
        metrics = Vector{NamedTuple}(undef, d)
        eps64 = BigFloat(eps(Float64))
        residual_diff2 = zero(BigFloat)
        residual_reference2 = zero(BigFloat)

        for k = 1:d
            zlevel, next, nextnext, blevel = lifted_local_inputs(z64, b64, k)
            rho = local_residual_level(sys64lift, zlevel, next, nextnext,
                                       blevel, k)
            # This second block evaluation is the matching block of rsolve.
            rsolve_level = local_residual_level(sys64lift, zlevel, next,
                                                nextnext, blevel, k)
            d2, r2 = level_difference_sums(rho, rsolve_level)
            residual_diff2 += d2
            residual_reference2 += r2
            A_part, C_part = forward_parts(sys64lift, k, next, nextnext)
            znorm = level_fro_norm(zlevel)
            bnorm = level_fro_norm(blevel)
            Anorm = level_fro_norm(A_part)
            Cnorm = level_fro_norm(C_part)
            rnorm = level_fro_norm(rho)
            denom = znorm + Anorm + Cnorm + bnorm
            comp = componentwise_backward_error(
                sys64lift, zlevel, next, nextnext, blevel, rho, k, tiny)
            metrics[k] = (
                layer_state_norm=znorm,
                layer_state_maxabs=level_maxabs(zlevel),
                rhs_norm=bnorm,
                A_term_norm=Anorm,
                C_term_norm=Cnorm,
                local_residual_norm=rnorm,
                local_residual_maxabs=level_maxabs(rho),
                local_normwise_backward_error=denom == 0 ? missing : rnorm / denom,
                local_componentwise_backward_error=comp,
                local_backward_error_over_eps=comp / eps64,
            )
            k % 4 == 0 && GC.gc(true)
        end

        residual_reconstruction = sqrt(residual_diff2) /
                                  max(sqrt(residual_reference2), tiny)
        GC.gc(true)

        rho_provider = k -> reevaluate_rho(sys64lift, z64, b64, k)
        DeltaV_solve, recurrence_rows, sum_A, sum_C =
            full_error_recurrence(sys128, rho_provider, d; emit_rows=emit_detail)
        total_norm = matrix_fro_norm(DeltaV_solve)
        DeltaV_sum = zeros(BigFloat, NOUT, NOUT)
        cumulative = zeros(BigFloat, NOUT, NOUT)
        shallow_sum = zeros(BigFloat, NOUT, NOUT)
        deep_sum = zeros(BigFloat, NOUT, NOUT)
        contribution_norms = fill(zero(BigFloat), d)
        gains = Union{Missing,BigFloat}[missing for _ = 1:d]
        signed_sum = zero(BigFloat)
        layer_rows = NamedTuple[]
        q = cld(d, 4)
        thresholds = Dict{Float64,Union{Missing,Int}}(
            0.50 => missing, 0.90 => missing, 0.99 => missing)

        for (added, k) in enumerate(d:-1:1)
            rho_k = rho_provider(k)
            DeltaV_k = propagate_one_source(sys128, rho_k, k)
            cnorm = matrix_fro_norm(DeltaV_k)
            rnorm = metrics[k].local_residual_norm
            gain = rnorm == 0 ? missing : cnorm / rnorm
            inner = matrix_inner(DeltaV_k, DeltaV_solve)
            alignment = cnorm == 0 || total_norm == 0 ? missing :
                        real(inner) / (cnorm * total_norm)
            signed = total_norm == 0 ? missing : real(inner) / total_norm^2
            relative = total_norm == 0 ? missing : cnorm / total_norm
            contribution_norms[k] = cnorm
            gains[k] = gain
            signed !== missing && (signed_sum += signed)
            matrix_axpy!(DeltaV_sum, one(BigFloat), DeltaV_k)
            matrix_axpy!(cumulative, one(BigFloat), DeltaV_k)
            k <= q && matrix_axpy!(shallow_sum, one(BigFloat), DeltaV_k)
            if added == q
                copyto!(deep_sum, cumulative)
            end
            cumulative_norm = matrix_fro_norm(cumulative)
            cumulative_relative = total_norm == 0 ? missing :
                                  cumulative_norm / total_norm
            if cumulative_relative !== missing
                for threshold in keys(thresholds)
                    if thresholds[threshold] === missing &&
                       cumulative_relative >= BigFloat(threshold)
                        thresholds[threshold] = added
                    end
                end
            end

            if emit_detail
                coords = layer_coordinates(k, d)
                push!(layer_rows, record(;
                    record_type="layer_source_contribution", kernel=KERNEL,
                    depth=d, Nout=NOUT, M=M, precision_bits=BIGFLOAT_BITS,
                    coords..., probe_or_mode="actual_float64_local_residual",
                    metrics[k]..., output_contribution_norm=cnorm,
                    actual_output_gain=gain,
                    relative_contribution_to_total=relative,
                    alignment_with_total=alignment,
                    signed_projection_fraction=signed,
                    cumulative_layers_added=added,
                    cumulative_output_error_norm=cumulative_norm,
                    cumulative_relative_to_total=cumulative_relative,
                    estimated_peak_bytes=peak,
                    memory_budget_bytes=MEMORY_BUDGET_BYTES,
                    status="completed",
                ))
            end
            DeltaV_k = nothing
            GC.gc(true)
        end

        layer_reconstruction = matrix_fro_norm(
            matrix_difference(DeltaV_sum, DeltaV_solve)) / max(total_norm, tiny)
        signed_error = abs(signed_sum - one(BigFloat))
        dominant_residual = argmax([m.local_residual_norm for m in metrics])
        dominant_contribution = argmax(contribution_norms)
        finite_gains = BigFloat[g for g in gains if g !== missing]
        inherited_total = sum_A + sum_C
        summary = record(;
            record_type="depth_summary", kernel=KERNEL, depth=d,
            Nout=NOUT, M=M, precision_bits=BIGFLOAT_BITS,
            probe_or_mode="padding_converged_M_equals_Nout_plus_d_plus_8",
            dominant_residual_layer=dominant_residual,
            dominant_residual_total_degree=dominant_residual - 1,
            dominant_contribution_layer=dominant_contribution,
            dominant_contribution_total_degree=dominant_contribution - 1,
            max_actual_output_gain=isempty(finite_gains) ? missing : maximum(finite_gains),
            median_actual_output_gain=isempty(finite_gains) ? missing : median(finite_gains),
            deepest_quartile_contribution_norm=matrix_fro_norm(deep_sum),
            shallowest_quartile_contribution_norm=matrix_fro_norm(shallow_sum),
            layers_for_50_percent=thresholds[0.50],
            layers_for_90_percent=thresholds[0.90],
            layers_for_99_percent=thresholds[0.99],
            residual_block_reconstruction_error=residual_reconstruction,
            layer_contribution_reconstruction_error=layer_reconstruction,
            signed_projection_sum_error=signed_error,
            sum_A_inherited_norm=sum_A, sum_C_inherited_norm=sum_C,
            A_channel_fraction=inherited_total == 0 ? missing : sum_A / inherited_total,
            C_channel_fraction=inherited_total == 0 ? missing : sum_C / inherited_total,
            total_reconstruction_error=layer_reconstruction,
            estimated_peak_bytes=peak, memory_budget_bytes=MEMORY_BUDGET_BYTES,
            status="completed",
        )

        checks = NamedTuple[
            record(; record_type="reconstruction_check", kernel=KERNEL,
                   depth=d, Nout=NOUT, M=M, precision_bits=BIGFLOAT_BITS,
                   probe_or_mode="residual_block_reconstruction_error",
                   residual_block_reconstruction_error=residual_reconstruction,
                   reconstruction_error=residual_reconstruction,
                   status="completed"),
            record(; record_type="reconstruction_check", kernel=KERNEL,
                   depth=d, Nout=NOUT, M=M, precision_bits=BIGFLOAT_BITS,
                   probe_or_mode="layer_contribution_reconstruction_error",
                   layer_contribution_reconstruction_error=layer_reconstruction,
                   reconstruction_error=layer_reconstruction,
                   status="completed"),
            record(; record_type="reconstruction_check", kernel=KERNEL,
                   depth=d, Nout=NOUT, M=M, precision_bits=BIGFLOAT_BITS,
                   probe_or_mode="signed_projection_sum_error",
                   signed_projection_sum_error=signed_error,
                   reconstruction_error=signed_error,
                   status="completed"),
        ]

        @printf("layerwise d=%d residual-recon=%s layer-recon=%s signed-sum=%s\n",
                d, sci(residual_reconstruction), sci(layer_reconstruction),
                sci(signed_error))
        flush(stdout)
        return (summary=summary, layer_rows=layer_rows,
                recurrence_rows=recurrence_rows, checks=checks)
    end
end

function explicit_peak_bytes(depth, Nout, M, dim)
    tbytes = Int(dim)^2 * sizeof(Float64)
    pbytes = Int(Nout)^2 * Int(dim) * sizeof(Float64)
    return 4tbytes + 4pbytes + estimate_peak_bytes(
        depth, M, Float64; complete_states=4, work_state_equivalents=2,
        dense_output_matrices=2, Nout=Nout)
end

function explicit_layer_rows(depth::Integer)
    d = Int(depth)
    Nout = d + 2
    M = recommended_internal_M(Nout, d; safety=8)
    dim = estimate_state_dimension(M, d)
    peak = explicit_peak_bytes(d, Nout, M, dim)
    @printf("explicit layer-gain allocation estimate d=%d dim=%d: %.3f GiB\n",
            d, dim, peak / 1024.0^3)
    flush(stdout)
    peak <= MEMORY_BUDGET_BYTES || error("explicit layer gain exceeds memory budget")

    spec = default_kernel_specs()[1]
    coeffs = make_coeffs(spec, d; flip=true)
    source = "native_float64_projection_fixed_for_b_isolation"
    sys64 = build_transfer_system_independent(
        coeffs; depth=d, Nout=Nout, M=M, arithmetic_type=Float64,
        kernel_name=spec.name, kernel_definition=spec.definition,
        coefficient_source=source, flip=true)
    b64 = rhs_state(sys64)
    z64 = solve_T_rhs(sys64)
    T64 = explicit_T_matrix(sys64)
    P64 = explicit_P_matrix(sys64; output=:final)
    G = transpose(transpose(T64) \ transpose(P64))
    layout = state_layout_rows(sys64)

    rho_levels = setprecision(BigFloat, BIGFLOAT_BITS) do
        sys64lift = lift_transfer_system(BigFloat, sys64)
        [begin
             zlevel = convert_level(BigFloat, z64.levels[k])
             next = k < d ? convert_level(BigFloat, z64.levels[k + 1]) : nothing
             nextnext = k + 2 <= d ? convert_level(BigFloat, z64.levels[k + 2]) : nothing
             blevel = convert_level(BigFloat, b64.levels[k])
             local_residual_level(sys64lift, zlevel, next, nextnext, blevel, k)
         end for k = 1:d]
    end

    rows = NamedTuple[]
    for k = 1:d
        idx = layout[k].first_state_index:layout[k].last_state_index
        Gk = G[:, idx]
        worst = opnorm(Gk, 2)
        rho = Float64.(flatten_state(GOClenshawState{BigFloat}([rho_levels[k]])))
        # flatten_state accepts a one-level state and preserves the level block order.
        rnorm = norm(rho)
        actual = rnorm == 0 ? missing : norm(Gk * rho) / rnorm
        ratio = actual === missing || worst == 0 ? missing : actual / worst
        coords = layer_coordinates(k, d)
        push!(rows, record(;
            record_type="explicit_layer_gain", kernel=KERNEL, depth=d,
            Nout=Nout, M=M, precision_bits=BIGFLOAT_BITS, coords...,
            probe_or_mode="explicit_Float64_Gk_with_128bit_reevaluated_residual",
            local_residual_norm=level_fro_norm(rho_levels[k]),
            actual_output_gain=actual, worst_case_layer_gain=worst,
            actual_to_worst_ratio=ratio, estimated_peak_bytes=peak,
            memory_budget_bytes=MEMORY_BUDGET_BYTES, status="completed",
        ))
    end
    return rows
end

function append_csv(path, rows)
    open(path, "a") do io
        for row in rows
            println(io, join((csv_cell(getproperty(row, col)) for col in COLUMNS), ","))
        end
    end
end

function run_isolated(argument)
    cmd = `$(JULIA_EXE) --project=$(GO_PROJECT) $(@__FILE__) $(argument)`
    println("Running isolated layerwise case: ", cmd)
    success(cmd) || error("isolated layerwise case failed: $argument")
end

function child_main(argument)
    if startswith(argument, "--production-depth=")
        depth = parse(Int, split(argument, "=", limit=2)[2])
        result = analyze_depth(depth; emit_detail=depth == PRIMARY_DEPTH)
        rows = NamedTuple[]
        append!(rows, result.layer_rows)
        append!(rows, result.recurrence_rows)
        push!(rows, result.summary)
        append!(rows, result.checks)
        append_csv(OUTPUT_CSV, rows)
    elseif startswith(argument, "--explicit-depth=")
        depth = parse(Int, split(argument, "=", limit=2)[2])
        append_csv(OUTPUT_CSV, explicit_layer_rows(depth))
    else
        error("unknown layerwise argument: $argument")
    end
end

function main()
    mkpath(DATA_DIR)
    if !isempty(ARGS)
        child_main(only(ARGS))
        return nothing
    end
    write_csv(OUTPUT_CSV, COLUMNS, NamedTuple[])
    for depth in DEPTHS
        run_isolated("--production-depth=$depth")
    end
    for depth in SMALL_DEPTHS
        run_isolated("--explicit-depth=$depth")
    end
    println("Wrote CSV: ", OUTPUT_CSV)
    return nothing
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
