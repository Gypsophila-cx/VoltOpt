% Final figures for the Gutleb-Olver instability validation.

script_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(fileparts(fileparts(script_dir)), 'tools'));
data_dir = fullfile(script_dir, 'data');
fig_dir = fullfile(script_dir, 'figures');
if ~exist(fig_dir, 'dir')
    mkdir(fig_dir);
end

padding = readtable(fullfile(data_dir, 'go_padding_validation.csv'));
nonnormal = readtable(fullfile(data_dir, 'go_nonnormality_diagnostics.csv'));
output_error = readtable(fullfile(data_dir, 'go_output_error_validation.csv'));
depth = readtable(fullfile(data_dir, 'go_depth_validation.csv'));
construction = readtable(fullfile(data_dir, 'go_construction_validation.csv'));
layerwise = readtable(fullfile(data_dir, 'go_layerwise_diagnostics.csv'));

plot_padding_convergence(padding, fig_dir);
plot_log_xy(nonnormal.depth, zero_to_nan(nonnormal.sigma_min), ...
    'depth', '\sigma_{min}(T)', ...
    fullfile(fig_dir, 'go_sigma_min_vs_depth'));
plot_log_xy(nonnormal.depth, zero_to_nan(nonnormal.cond_2), ...
    'depth', 'cond_2(T)', ...
    fullfile(fig_dir, 'go_condition_vs_depth'));
plot_log_xy(nonnormal.depth, zero_to_nan(nonnormal.norm_P_Tinv_2), ...
    'depth', '||P T^{-1}||_2', ...
    fullfile(fig_dir, 'go_output_gain_vs_depth'));
plot_log_xy(depth.depth, zero_to_nan(depth.final_operator_error), ...
    'depth', '||V_{Float64} - V_{128}||_F', ...
    fullfile(fig_dir, 'go_operator_error_vs_depth'));
plot_construction_action(construction, 'matrix_free_T_probe', ...
    'T construction/action error', fullfile(fig_dir, 'go_T_construction_error_vs_depth'));
plot_construction_action(construction, 'matrix_free_P_probe', ...
    'P construction/action error', fullfile(fig_dir, 'go_P_construction_error_vs_depth'));
plot_error_contributions(output_error, fig_dir);
plot_layer_residual_contribution(layerwise, fig_dir);
plot_layer_output_gain(layerwise, fig_dir);
plot_layer_cumulative_error(layerwise, fig_dir);

fprintf('Wrote figures to %s\n', fig_dir);
disp(output_error(:, {'validation_mode', 'depth', 'Nout', 'M', ...
    'Float64_to_128_operator_error', 'relative_operator_error', ...
    'decomposition_relative_error', 'dominant_error_source', 'status'}));

function plot_padding_convergence(tbl, fig_dir)
    mask = strcmp(string(tbl.kernel), 'mixed_cos20xy');
    tbl = tbl(mask, :);
    y = zero_to_nan(tbl.padding_error_to_max);
    depths = unique(tbl.depth);
    fig = new_figure();
    hold on;
    for k = 1:numel(depths)
        d = depths(k);
        rows = tbl.depth == d;
        plot(tbl.padding(rows), y(rows), '-o', ...
            'LineWidth', 1.5, 'DisplayName', sprintf('d=%d', d));
    end
    hold off;
    xlabel('padding = M - Nout');
    ylabel('padding error to max');
    set(gca, 'YScale', 'log');
    legend('Location', 'best');
    style_axes();
    save_both(fig, fullfile(fig_dir, 'go_padding_convergence'));
end

function plot_layer_residual_contribution(tbl, fig_dir)
    sub = primary_layer_rows(tbl);
    sub = sortrows(sub, 'total_degree');
    fig = new_figure();
    semilogy(sub.total_degree, zero_to_nan(sub.local_residual_norm), ...
        '-o', 'LineWidth', 1.5, 'DisplayName', 'local residual');
    hold on;
    semilogy(sub.total_degree, zero_to_nan(sub.output_contribution_norm), ...
        '-s', 'LineWidth', 1.5, 'DisplayName', 'output contribution');
    hold off;
    xlabel('total degree');
    ylabel('Frobenius norm');
    legend('Location', 'best');
    style_axes();
    save_both(fig, fullfile(fig_dir, 'go_layer_residual_contribution'));
end

function plot_layer_output_gain(tbl, fig_dir)
    sub = primary_layer_rows(tbl);
    sub = sortrows(sub, 'total_degree');
    fig = new_figure();
    semilogy(sub.total_degree, zero_to_nan(sub.actual_output_gain), ...
        '-o', 'LineWidth', 1.5);
    xlabel('total degree');
    ylabel('actual output gain');
    style_axes();
    save_both(fig, fullfile(fig_dir, 'go_layer_output_gain'));
end

function plot_layer_cumulative_error(tbl, fig_dir)
    sub = primary_layer_rows(tbl);
    sub = sortrows(sub, 'cumulative_layers_added');
    fig = new_figure();
    plot(sub.cumulative_layers_added, sub.cumulative_relative_to_total, ...
        '-o', 'LineWidth', 1.5);
    hold on;
    yline(0.5, '--', 'LineWidth', 1.0);
    yline(0.9, '--', 'LineWidth', 1.0);
    yline(1.0, ':', 'LineWidth', 1.0);
    hold off;
    xlabel('deepest layers included');
    ylabel('cumulative error / total error');
    style_axes();
    save_both(fig, fullfile(fig_dir, 'go_layer_cumulative_error'));
end

function sub = primary_layer_rows(tbl)
    mask = strcmp(string(tbl.record_type), 'layer_source_contribution') & ...
        tbl.depth == 36 & tbl.Nout == 72 & tbl.M == 116;
    sub = tbl(mask, :);
end

function plot_log_xy(x, y, xlab, ylab, output_base)
    fig = new_figure();
    plot(x, y, '-o', 'LineWidth', 1.5);
    xlabel(xlab);
    ylabel(ylab);
    set(gca, 'YScale', 'log');
    style_axes();
    save_both(fig, output_base);
end

function plot_construction_action(tbl, record_type, ylab, output_base)
    mask = strcmp(string(tbl.record_type), record_type) & ...
        strcmp(string(tbl.probe_name), 'max_over_probes');
    sub = tbl(mask, :);
    fig = new_figure();
    semilogy(sub.depth, zero_to_nan(sub.relative_construction_error), ...
        '-o', 'LineWidth', 1.5, 'DisplayName', 'construction');
    hold on;
    semilogy(sub.depth, zero_to_nan(sub.application_rounding_error), ...
        '-s', 'LineWidth', 1.5, 'DisplayName', 'application rounding');
    hold off;
    xlabel('depth');
    ylabel(ylab);
    legend('Location', 'best');
    style_axes();
    save_both(fig, output_base);
end

function plot_error_contributions(output_error, fig_dir)
    mask = strcmp(string(output_error.validation_mode), 'fixed_b_isolation');
    row = output_error(mask, :);
    if isempty(row)
        return;
    end
    labels = {'T construction', 'solve rounding', 'b construction', 'P construction'};
    values = [row.norm_DeltaV_T(1), row.norm_DeltaV_solve(1), ...
        row.norm_DeltaV_b(1), row.norm_DeltaV_P(1)];
    keep = values > 0;
    labels = labels(keep);
    values = values(keep);
    fig = new_figure();
    bar(values, 'LineWidth', 1.5);
    set(gca, 'YScale', 'log');
    set(gca, 'XTick', 1:numel(labels), 'XTickLabel', labels);
    xtickangle(25);
    ylabel('contribution norm');
    style_axes();
    save_both(fig, fullfile(fig_dir, 'go_error_contribution_breakdown'));
end

function y = zero_to_nan(x)
    y = x;
    y(y == 0) = NaN;
end

function fig = new_figure()
    paperStyle = voltopt_plot_style();
    fig = figure('Color', 'w', 'Units', 'centimeters', ...
        'Position', [2, 2, paperStyle.FigureSize]);
end

function style_axes()
    paperStyle = voltopt_plot_style();
    set(gca, 'FontName', paperStyle.FontName, ...
        'FontSize', paperStyle.TickFontSize, ...
        'LineWidth', paperStyle.AxesLineWidth, ...
        'Box', 'on', ...
        'XGrid', 'on', ...
        'YGrid', 'on');
end

function save_both(fig, output_base)
    pdf_path = [output_base, '.pdf'];
    png_path = [output_base, '.png'];
    axesList = findall(fig, 'Type', 'axes');
    voltopt_export_figure(fig, axesList, pdf_path, png_path);
    close(fig);
end
