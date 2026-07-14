# Final small-system nonnormality validation for T_{d,M}.

include(joinpath(@__DIR__, "go_instability_core.jl"))
using .GOInstabilityCore
using LinearAlgebra
using Printf

const SCRIPT_DIR = @__DIR__
const DATA_DIR = joinpath(SCRIPT_DIR, "data")
const OUTPUT_CSV = joinpath(DATA_DIR, "go_nonnormality_diagnostics.csv")
const MEMORY_BUDGET_BYTES = default_memory_budget_bytes()
const DEPTHS = (2, 3, 4, 5)
const MAX_EXPLICIT_DIM = 1400

const COLUMNS = (
    :kernel,
    :depth,
    :Nout,
    :M,
    :state_dimension,
    :estimated_peak_bytes,
    :memory_budget_bytes,
    :eigenvalue_distance_from_one,
    :departure_from_normality,
    :sigma_min,
    :sigma_max,
    :cond_2,
    :norm_Tinv_2,
    :norm_P_Tinv_2,
    :output_screening_ratio,
    :normalized_backward_error,
    :normalized_backward_error_over_eps,
    :residual_norm,
    :status,
)

function explicit_peak_bytes(depth::Integer, Nout::Integer, M::Integer,
                             dim::Integer)
    t_bytes = Int(dim)^2 * sizeof(Float64)
    p_bytes = Int(Nout)^2 * Int(dim) * sizeof(Float64)
    work_bytes = 4 * t_bytes + 2 * p_bytes
    state_bytes = estimate_peak_bytes(
        depth,
        M,
        Float64;
        complete_states=3,
        work_state_equivalents=2,
        dense_output_matrices=1,
        Nout=Nout,
    )
    return state_bytes + work_bytes
end

function skipped_row(depth::Integer, Nout::Integer, M::Integer,
                     dim::Integer, peak::Integer, status::AbstractString)
    return (
        kernel="mixed_cos20xy",
        depth=Int(depth),
        Nout=Int(Nout),
        M=Int(M),
        state_dimension=Int(dim),
        estimated_peak_bytes=Int(peak),
        memory_budget_bytes=MEMORY_BUDGET_BYTES,
        eigenvalue_distance_from_one=missing,
        departure_from_normality=missing,
        sigma_min=missing,
        sigma_max=missing,
        cond_2=missing,
        norm_Tinv_2=missing,
        norm_P_Tinv_2=missing,
        output_screening_ratio=missing,
        normalized_backward_error=missing,
        normalized_backward_error_over_eps=missing,
        residual_norm=missing,
        status=String(status),
    )
end

function run_case(depth::Integer)
    spec = default_kernel_specs()[1]
    Nout = Int(depth) + 2
    M = recommended_internal_M(Nout, depth; safety=8)
    dim = estimate_state_dimension(M, depth)
    peak = explicit_peak_bytes(depth, Nout, M, dim)
    if Nout <= depth || M <= depth
        return skipped_row(depth, Nout, M, dim, peak, "invalid_padding_logic")
    elseif dim > MAX_EXPLICIT_DIM
        return skipped_row(depth, Nout, M, dim, peak, "skipped_explicit_dim_limit")
    elseif peak > MEMORY_BUDGET_BYTES
        return skipped_row(depth, Nout, M, dim, peak, "skipped_memory_budget")
    end

    coeffs = make_coeffs(spec, depth; flip=true)
    sys = build_transfer_system_explicit(
        coeffs;
        depth=depth,
        Nout=Nout,
        M=M,
        arithmetic_type=Float64,
        kernel_name=spec.name,
        kernel_definition=spec.definition,
        coefficient_source="native_float64_projection",
        flip=true,
    )
    Tmat = explicit_T_matrix(sys)
    Pmat = explicit_P_matrix(sys; output=:final)
    eigdist = maximum(abs.(eigvals(Tmat) .- 1.0))
    departure = norm(Tmat' * Tmat - Tmat * Tmat') /
                max(norm(Tmat)^2, eps(Float64))
    sv = svdvals(Tmat)
    sigma_min = minimum(sv)
    sigma_max = maximum(sv)
    cond2 = sigma_max / sigma_min
    norm_Tinv = inv(sigma_min)
    PTinv = Pmat / Tmat
    norm_P_Tinv = opnorm(PTinv, 2)
    b = rhs_state(sys)
    z = solve_T(sys, b)
    rho = recurrence_residual(sys, z)
    rho_vec = flatten_state(rho)
    z_vec = flatten_state(z)
    b_vec = flatten_state(b)
    beta = norm(rho_vec) /
           max(opnorm(Tmat, 2) * norm(z_vec) + norm(b_vec), eps(Float64))

    return (
        kernel="mixed_cos20xy",
        depth=Int(depth),
        Nout=Int(Nout),
        M=Int(M),
        state_dimension=state_dimension(sys),
        estimated_peak_bytes=Int(peak),
        memory_budget_bytes=MEMORY_BUDGET_BYTES,
        eigenvalue_distance_from_one=eigdist,
        departure_from_normality=departure,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        cond_2=cond2,
        norm_Tinv_2=norm_Tinv,
        norm_P_Tinv_2=norm_P_Tinv,
        output_screening_ratio=norm_P_Tinv / max(norm_Tinv, eps(Float64)),
        normalized_backward_error=beta,
        normalized_backward_error_over_eps=beta / eps(Float64),
        residual_norm=norm(rho_vec),
        status="completed",
    )
end

function main()
    mkpath(DATA_DIR)
    rows = NamedTuple[]
    for depth in DEPTHS
        row = run_case(depth)
        push!(rows, row)
        write_csv(OUTPUT_CSV, COLUMNS, rows)
        if row.status == "completed"
            @printf("nonnormal d=%d dim=%d sigma_min=%.3e cond=%.3e PTinv=%.3e beta/eps=%.3e\n",
                    row.depth, row.state_dimension, row.sigma_min,
                    row.cond_2, row.norm_P_Tinv_2,
                    row.normalized_backward_error_over_eps)
        else
            @printf("nonnormal d=%d dim=%d status=%s\n",
                    row.depth, row.state_dimension, row.status)
        end
        flush(stdout)
        GC.gc(true)
    end
    write_csv(OUTPUT_CSV, COLUMNS, rows)
    println("Wrote CSV: ", OUTPUT_CSV)
    return rows
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
