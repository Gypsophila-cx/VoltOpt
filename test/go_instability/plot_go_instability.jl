# Reproducible fallback plotting when MATLAB is unavailable.

using DelimitedFiles
using Plots

const SCRIPT_DIR = @__DIR__
const DATA_DIR = joinpath(SCRIPT_DIR, "data")
const FIG_DIR = joinpath(SCRIPT_DIR, "figures")

function read_table(name)
    data, header = readdlm(joinpath(DATA_DIR, name), ',', Any, '\n'; header=true)
    names = vec(String.(header))
    return (data=data, names=names)
end

column(tbl, name) = tbl.data[:, only(findall(==(String(name)), tbl.names))]
strings(tbl, name) = string.(column(tbl, name))

function numbers(tbl, name)
    return map(column(tbl, name)) do value
        value isa Number && return Float64(value)
        text = strip(string(value))
        isempty(text) ? NaN : parse(Float64, text)
    end
end

function subset(tbl, mask)
    return (data=tbl.data[mask, :], names=tbl.names)
end

positive(values) = map(x -> isfinite(x) && x > 0 ? x : NaN, values)

function save_both(p, name)
    savefig(p, joinpath(FIG_DIR, name * ".pdf"))
    savefig(p, joinpath(FIG_DIR, name * ".png"))
end

function line_plot(x, y, xlabel_text, ylabel_text; logy=true, label="")
    plot_y = logy ? positive(y) : y
    return plot(x, plot_y; marker=:circle, linewidth=1.5, label=label,
                xlabel=xlabel_text, ylabel=ylabel_text,
                yscale=logy ? :log10 : :identity)
end

function construction_plot(tbl, record_type, ylabel_text)
    mask = (strings(tbl, "record_type") .== record_type) .&
           (strings(tbl, "probe_name") .== "max_over_probes")
    sub = subset(tbl, mask)
    p = line_plot(numbers(sub, "depth"),
                  numbers(sub, "relative_construction_error"),
                  "depth", ylabel_text; label="construction")
    plot!(p, numbers(sub, "depth"),
          positive(numbers(sub, "application_rounding_error")); marker=:square,
          linewidth=1.5, label="application rounding")
    return p
end

function main()
    mkpath(FIG_DIR)
    default(fontfamily="Times Roman", guidefontsize=12, tickfontsize=12,
            legendfontsize=10, linewidth=1.5, framestyle=:box, grid=true,
            size=(1620, 1080), dpi=300, left_margin=10 * Plots.mm,
            bottom_margin=7 * Plots.mm)

    padding = read_table("go_padding_validation.csv")
    nonnormal = read_table("go_nonnormality_diagnostics.csv")
    output_error = read_table("go_output_error_validation.csv")
    depth = read_table("go_depth_validation.csv")
    construction = read_table("go_construction_validation.csv")
    layerwise = read_table("go_layerwise_diagnostics.csv")

    pmask = strings(padding, "kernel") .== "mixed_cos20xy"
    psub = subset(padding, pmask)
    pp = plot(; xlabel="padding = M - Nout", ylabel="padding error to max",
              yscale=:log10)
    for d in unique(Int.(numbers(psub, "depth")))
        mask = Int.(numbers(psub, "depth")) .== d
        plot!(pp, numbers(psub, "padding")[mask],
              positive(numbers(psub, "padding_error_to_max")[mask]); marker=:circle,
              linewidth=1.5, label="d=$d")
    end
    save_both(pp, "go_padding_convergence")

    for (field, ylabel_text, name) in (
        ("sigma_min", "sigma_min(T)", "go_sigma_min_vs_depth"),
        ("cond_2", "cond_2(T)", "go_condition_vs_depth"),
        ("norm_P_Tinv_2", "||P T^-1||_2", "go_output_gain_vs_depth"),
    )
        save_both(line_plot(numbers(nonnormal, "depth"),
                            numbers(nonnormal, field), "depth", ylabel_text), name)
    end
    save_both(line_plot(numbers(depth, "depth"),
                        numbers(depth, "final_operator_error"), "depth",
                        "||V_Float64 - V_128||_F"),
              "go_operator_error_vs_depth")
    save_both(construction_plot(construction, "matrix_free_T_probe",
                                "T construction/action error"),
              "go_T_construction_error_vs_depth")
    save_both(construction_plot(construction, "matrix_free_P_probe",
                                "P construction/action error"),
              "go_P_construction_error_vs_depth")

    omask = strings(output_error, "validation_mode") .== "fixed_b_isolation"
    osub = subset(output_error, omask)
    values = [numbers(osub, field)[1] for field in
              ("norm_DeltaV_T", "norm_DeltaV_solve", "norm_DeltaV_b",
               "norm_DeltaV_P")]
    labels = ["T construction", "solve rounding", "b construction",
              "P construction"]
    keep = values .> 0
    bp = bar(labels[keep], values[keep]; yscale=:log10, legend=false,
             ylabel="contribution norm", xrotation=25)
    save_both(bp, "go_error_contribution_breakdown")

    lmask = (strings(layerwise, "record_type") .==
             "layer_source_contribution") .&
            (Int.(numbers(layerwise, "depth")) .== 36) .&
            (Int.(numbers(layerwise, "Nout")) .== 72) .&
            (Int.(numbers(layerwise, "M")) .== 116)
    layers = subset(layerwise, lmask)
    degree = numbers(layers, "total_degree")
    order = sortperm(degree)
    p1 = line_plot(degree[order], numbers(layers, "local_residual_norm")[order],
                   "total degree", "Frobenius norm"; label="local residual")
    plot!(p1, degree[order],
          positive(numbers(layers, "output_contribution_norm")[order]);
          marker=:square, linewidth=1.5, label="output contribution")
    save_both(p1, "go_layer_residual_contribution")

    p2 = line_plot(degree[order], numbers(layers, "actual_output_gain")[order],
                   "total degree", "actual output gain")
    save_both(p2, "go_layer_output_gain")

    added = numbers(layers, "cumulative_layers_added")
    cumulative_order = sortperm(added)
    p3 = line_plot(added[cumulative_order],
                   numbers(layers, "cumulative_relative_to_total")[cumulative_order],
                   "deepest layers included", "cumulative error / total error";
                   logy=false)
    hline!(p3, [0.5, 0.9]; linestyle=:dash, color=:gray, label="")
    hline!(p3, [1.0]; linestyle=:dot, color=:black, label="")
    save_both(p3, "go_layer_cumulative_error")

    println("Wrote figures to ", FIG_DIR)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
