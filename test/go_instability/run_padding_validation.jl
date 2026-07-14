# Final internal-padding validation for the Gutleb-Olver Clenshaw system.

include(joinpath(@__DIR__, "go_instability_core.jl"))
using .GOInstabilityCore
using Printf

const SCRIPT_DIR = @__DIR__
const DATA_DIR = joinpath(SCRIPT_DIR, "data")
const OUTPUT_CSV = joinpath(DATA_DIR, "go_padding_validation.csv")
const MEMORY_BUDGET_BYTES = default_memory_budget_bytes()
const PADDING_TOL = 1.0e-12
const NOUT = 72
const DEPTHS = (8, 16, 24, 32, 36)
const KERNELS = ("mixed_cos20xy", "constant_one_exact")

const COLUMNS = (
    :kernel,
    :depth,
    :Nout,
    :M,
    :requested_M,
    :effective_M,
    :padding,
    :state_dimension,
    :estimated_peak_bytes,
    :memory_budget_bytes,
    :level1_block_rows,
    :level1_block_cols,
    :arithmetic,
    :precision_bits,
    :coefficient_source,
    :operator_norm,
    :padding_error_to_max,
    :padding_converged,
    :selected_minimal_M,
    :status,
)

function coeffs_for_case(kernel_name::AbstractString, depth::Integer)
    if kernel_name == "mixed_cos20xy"
        spec = default_kernel_specs()[1]
        return spec, make_coeffs(spec, depth; flip=true),
               "native_float64_projection"
    elseif kernel_name == "constant_one_exact"
        spec = default_kernel_specs()[2]
        return spec, make_exact_constant_coeffs(depth; flip=true),
               "exact_constant_coefficients"
    end
    throw(ArgumentError("unknown kernel: $kernel_name"))
end

function plan_peak_bytes(depth::Integer, Nout::Integer, M::Integer)
    return estimate_peak_bytes(
        depth,
        M,
        Float64;
        complete_states=1,
        work_state_equivalents=3,
        dense_output_matrices=2,
        Nout=Nout,
    )
end

function solve_output(spec, coeffs, source, depth::Integer, Nout::Integer,
                      M::Integer)
    requested_M = Int(M)
    peak = plan_peak_bytes(depth, Nout, requested_M)
    if peak > MEMORY_BUDGET_BYTES
        return (
            V=missing,
            requested_M=requested_M,
            effective_M=requested_M,
            state_dimension=estimate_state_dimension(requested_M, depth),
            estimated_peak_bytes=peak,
            level1_block_rows=missing,
            level1_block_cols=missing,
            status="skipped_memory_budget",
        )
    end

    sys = build_transfer_system_explicit(
        coeffs;
        depth=depth,
        Nout=Nout,
        M=requested_M,
        arithmetic_type=Float64,
        kernel_name=spec.name,
        kernel_definition=spec.definition,
        coefficient_source=source,
        flip=true,
    )
    z = solve_T_rhs(sys)
    level1_block = z.levels[1][1]
    effective_M = sys.M
    status = if effective_M != requested_M
        "failed_effective_M_mismatch"
    elseif size(level1_block) != (requested_M, requested_M)
        "failed_level1_block_size"
    else
        "computed"
    end
    V = status == "computed" ? apply_P(sys, z; output=:final) : missing
    return (
        V=V,
        requested_M=requested_M,
        effective_M=effective_M,
        state_dimension=state_dimension(sys),
        estimated_peak_bytes=peak,
        level1_block_rows=size(level1_block, 1),
        level1_block_cols=size(level1_block, 2),
        status=status,
    )
end

function selected_minimal_M(Ms, rel_to_max)
    for (M, err) in zip(Ms, rel_to_max)
        if err !== missing && err <= PADDING_TOL
            return Int(M)
        end
    end
    return missing
end

function run_group(kernel_name::AbstractString, depth::Integer, Nout::Integer)
    spec, coeffs, source = coeffs_for_case(kernel_name, depth)
    Ms = collect(padding_sequence(Nout, depth; safety=8))
    results = [solve_output(spec, coeffs, source, depth, Nout, M) for M in Ms]

    all_computed = all(r.status == "computed" for r in results)
    max_output = all_computed ? results[end].V : missing
    max_norm = all_computed ? matrix_fro_norm(max_output) : missing
    rel_to_max = Any[]

    if all_computed
        for result in results
            diff = matrix_fro_norm(matrix_difference(result.V, max_output))
            push!(rel_to_max, diff / max(max_norm, eps(Float64)))
        end
    else
        append!(rel_to_max, fill(missing, length(results)))
    end

    M3 = Int(Nout + depth + 8)
    M4 = Int(Nout + 2 * depth + 8)
    idx_M3 = findfirst(==(M3), Ms)
    idx_M4 = findfirst(==(M4), Ms)
    m3_converged = all_computed &&
                   idx_M3 !== nothing &&
                   idx_M4 !== nothing &&
                   rel_to_max[idx_M3] <= PADDING_TOL &&
                   rel_to_max[idx_M4] <= PADDING_TOL
    group_selected_M = m3_converged ? M3 : selected_minimal_M(Ms, rel_to_max)

    rows = NamedTuple[]
    for i in eachindex(Ms)
        result = results[i]
        rel = rel_to_max[i]
        row_converged = all_computed && rel <= PADDING_TOL
        status = if result.status != "computed"
            result.status
        elseif row_converged
            Ms[i] == group_selected_M ? "selected_padding_converged" :
                                        "padding_converged"
        else
            "not_converged_to_max_M"
        end
        push!(rows, (
            kernel=String(kernel_name),
            depth=Int(depth),
            Nout=Int(Nout),
            M=Int(Ms[i]),
            requested_M=result.requested_M,
            effective_M=result.effective_M,
            padding=Int(Ms[i] - Nout),
            state_dimension=result.state_dimension,
            estimated_peak_bytes=result.estimated_peak_bytes,
            memory_budget_bytes=MEMORY_BUDGET_BYTES,
            level1_block_rows=result.level1_block_rows,
            level1_block_cols=result.level1_block_cols,
            arithmetic="Float64",
            precision_bits=53,
            coefficient_source=source,
            operator_norm=all_computed ? matrix_fro_norm(result.V) : missing,
            padding_error_to_max=rel,
            padding_converged=row_converged,
            selected_minimal_M=group_selected_M,
            status=status,
        ))
    end
    return rows
end

function main()
    mkpath(DATA_DIR)
    rows = NamedTuple[]
    for kernel in KERNELS, depth in DEPTHS
        group_rows = run_group(kernel, depth, NOUT)
        append!(rows, group_rows)
        write_csv(OUTPUT_CSV, COLUMNS, rows)
        selected = group_rows[1].selected_minimal_M
        m0 = group_rows[1]
        m3 = group_rows[findfirst(r -> r.M == NOUT + depth + 8, group_rows)]
        @printf("padding kernel=%s d=%d Nout=%d M0err=%.3e M(d)=%.0f err=%.3e selected=%s\n",
                kernel, depth, NOUT, Float64(m0.padding_error_to_max),
                Float64(m3.M), Float64(m3.padding_error_to_max),
                string(selected))
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
