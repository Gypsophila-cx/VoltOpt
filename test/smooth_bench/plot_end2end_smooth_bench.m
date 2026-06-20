% plot_end2end_smooth_bench_loglog.m
% Log-log plots for end-to-end construction time:
%   VoltMat-Cheb vs Gutleb--Olver
%
% Features:
%   - y-axis in seconds
%   - log-log plots
%   - no markers
%   - grid on
%   - black dashed reference lines on the tail
%   - reference lines are NOT anchored at the last data point
%   - complexity labels are placed beside dashed lines
%   - dashed reference lines are hidden from legend

clear; clc; close all;

%% Folder

bench_dir = 'D:\Projects\VoltOpt\test\smooth_bench';

csv_fixedC = fullfile(bench_dir, 'bench_end2end_fixedC_varyN.csv');
csv_fixedN = fullfile(bench_dir, 'bench_end2end_fixedN_varyC.csv');

%% Global style

set(groot, 'defaultAxesFontName', 'Times New Roman');
set(groot, 'defaultTextFontName', 'Times New Roman');
set(groot, 'defaultAxesFontSize', 15);
set(groot, 'defaultLineLineWidth', 2.0);

tail_points_fixedC = 4;
tail_points_fixedN = 3;

%% ============================================================
%  Figure 1: fixed coefficient size C, varying N
% ============================================================

if isfile(csv_fixedC)

    T = readtable(csv_fixedC);

    N = get_numeric_col(T, 'N');

    % Convert ms to sec
    cheb_time = get_numeric_col(T, 'cheb_median_ms') / 1000.0;
    go_time   = get_numeric_col(T, 'gutleb_median_ms') / 1000.0;

    valid_cheb = isfinite(N) & isfinite(cheb_time) ...
        & N > 0 & cheb_time > 0;

    valid_go = isfinite(N) & isfinite(go_time) ...
        & N > 0 & go_time > 0;

    figure('Color', 'w', 'Position', [100, 100, 760, 560]);
    hold on; box on; grid on;

    loglog(N(valid_cheb), cheb_time(valid_cheb), '-', ...
        'DisplayName', 'Duffy--Cheb');

    loglog(N(valid_go), go_time(valid_go), '-', ...
        'DisplayName', 'Gutleb--Olver');

    % Reference lines.
    % line_factor < 1 moves the dashed line downward.
    % line_factor > 1 moves the dashed line upward.
    add_tail_reference_power( ...
        N(valid_cheb), cheb_time(valid_cheb), ...
        tail_points_fixedC, 1.0, ...
        '$O(N)$', ...
        0.58, 0.8, 0.8);

    add_tail_reference_power( ...
        N(valid_go), go_time(valid_go), ...
        tail_points_fixedC, 1.0, ...
        '$O(N)$', ...
        0.1, 1.2, 1.7);

    xlabel('truncation dimension $N$', 'Interpreter', 'latex');
    ylabel('time (sec)', ...
        'Interpreter', 'latex');

    legend('Location', 'northwest', 'Interpreter', 'latex');

    ax = gca;
    ax.XScale = 'log';
    ax.YScale = 'log';
    ax.XGrid = 'on';
    ax.YGrid = 'on';
    ax.XMinorGrid = 'on';
    ax.YMinorGrid = 'on';
    ax.MinorGridLineStyle = ':';

    out_pdf = fullfile(bench_dir, 'end2end_fixedC_varyN_time_loglog.pdf');
    out_png = fullfile(bench_dir, 'end2end_fixedC_varyN_time_loglog.png');

    exportgraphics(gcf, out_pdf, 'ContentType', 'vector');
    exportgraphics(gcf, out_png, 'Resolution', 300);

    fprintf('Saved: %s\n', out_pdf);
    fprintf('Saved: %s\n', out_png);

else
    warning('CSV not found: %s', csv_fixedC);
end

%% ============================================================
%  Figure 2: fixed N, varying input coefficient size C
% ============================================================

if isfile(csv_fixedN)

    T = readtable(csv_fixedN);

    % Use original benchmark sizes as x-axis.
    % In the CSV this is stored as C_rows.
    sizes = get_numeric_col(T, 'C_rows');

    % Convert ms to sec
    cheb_time = get_numeric_col(T, 'cheb_median_ms') / 1000.0;
    go_time   = get_numeric_col(T, 'gutleb_median_ms') / 1000.0;

    valid_cheb = isfinite(sizes) & isfinite(cheb_time) ...
        & sizes > 0 & cheb_time > 0;

    valid_go = isfinite(sizes) & isfinite(go_time) ...
        & sizes > 0 & go_time > 0;

    figure('Color', 'w', 'Position', [150, 150, 760, 560]);
    hold on; box on; grid on;

    loglog(sizes(valid_cheb), cheb_time(valid_cheb), '-', ...
        'DisplayName', 'Duffy--Cheb');

    loglog(sizes(valid_go), go_time(valid_go), '-', ...
        'DisplayName', 'Gutleb--Olver');

    % Cheb reference: O(n^2 log n)
    add_tail_reference_n2logn( ...
        sizes(valid_cheb), cheb_time(valid_cheb), ...
        tail_points_fixedN, ...
        '$O(n^2\log n)$', ...
        0.55, 0.70, 0.8);

    % Gutleb--Olver reference: O(n^3)
    add_tail_reference_power( ...
        sizes(valid_go), go_time(valid_go), ...
        tail_points_fixedN, 3.0, ...
        '$O(n^3)$', ...
        0.5, 1.5, 3);

    xlabel('bandwitdth $n$', 'Interpreter', 'latex');
    ylabel('time (sec)', ...
        'Interpreter', 'latex');

    legend('Location', 'northwest', 'Interpreter', 'latex');

    ax = gca;
    ax.XScale = 'log';
    ax.YScale = 'log';
    ax.XGrid = 'on';
    ax.YGrid = 'on';
    ax.XMinorGrid = 'on';
    ax.YMinorGrid = 'on';
    ax.MinorGridLineStyle = ':';

    out_pdf = fullfile(bench_dir, 'end2end_fixedN_varyC_time_loglog.pdf');
    out_png = fullfile(bench_dir, 'end2end_fixedN_varyC_time_loglog.png');

    exportgraphics(gcf, out_pdf, 'ContentType', 'vector');
    exportgraphics(gcf, out_png, 'Resolution', 300);

    fprintf('Saved: %s\n', out_pdf);
    fprintf('Saved: %s\n', out_png);

else
    warning('CSV not found: %s', csv_fixedN);
end

%% ============================================================
%  Helper: read numeric column robustly
% ============================================================

function x = get_numeric_col(T, name)

    if ~ismember(name, T.Properties.VariableNames)
        error('Column "%s" not found in table.', name);
    end

    v = T.(name);

    if isnumeric(v)
        x = double(v);
    elseif islogical(v)
        x = double(v);
    elseif iscell(v)
        x = str2double(string(v));
    elseif isstring(v)
        x = str2double(v);
    elseif ischar(v)
        x = str2double(string(v));
    elseif iscategorical(v)
        x = str2double(string(v));
    else
        try
            x = double(v);
        catch
            x = str2double(string(v));
        end
    end
end

%% ============================================================
%  Helper: black dashed reference line y ~ x^p
%
%  Important:
%  The reference line is NOT anchored at the last point.
%  Its vertical position is determined by the median of y / x^p
%  over the tail region, then multiplied by line_factor.
% ============================================================

function add_tail_reference_power(x, y, n_tail, p, label_text, label_pos, line_factor, label_factor)

    x = x(:);
    y = y(:);

    valid = isfinite(x) & isfinite(y) & x > 0 & y > 0;
    x = x(valid);
    y = y(valid);

    if numel(x) < 2
        return;
    end

    [x, idx] = sort(x);
    y = y(idx);

    n_tail = min(n_tail, numel(x));
    xt = x(end - n_tail + 1:end);
    yt = y(end - n_tail + 1:end);

    xref = logspace(log10(xt(1)), log10(xt(end)), 100);

    % Determine reference-line scale from the whole tail,
    % not from the last point.
    f_tail = xt.^p;
    scale = median(yt ./ f_tail, 'omitnan');

    % Move dashed line away from the solid curve.
    scale = scale * line_factor;

    yref = scale * xref.^p;

    loglog(xref, yref, 'k--', ...
        'LineWidth', 1.5, ...
        'HandleVisibility', 'off');

    % Label near the dashed line.
    label_pos = min(max(label_pos, 0), 1);
    x_label = exp((1 - label_pos) * log(xt(1)) + label_pos * log(xt(end)));
    y_label = scale * x_label.^p * label_factor;

    text(x_label, y_label, label_text, ...
        'Interpreter', 'latex', ...
        'FontSize', 14, ...
        'Color', 'k', ...
        'HorizontalAlignment', 'left', ...
        'VerticalAlignment', 'middle', ...
        'HandleVisibility', 'off');
end

%% ============================================================
%  Helper: black dashed reference line y ~ n^2 log n
%
%  Important:
%  The reference line is NOT anchored at the last point.
%  Its vertical position is determined by the median of
%  y / (n^2 log n) over the tail region, then multiplied
%  by line_factor.
% ============================================================

function add_tail_reference_n2logn(x, y, n_tail, label_text, label_pos, line_factor, label_factor)

    x = x(:);
    y = y(:);

    valid = isfinite(x) & isfinite(y) & x > 1 & y > 0;
    x = x(valid);
    y = y(valid);

    if numel(x) < 2
        return;
    end

    [x, idx] = sort(x);
    y = y(idx);

    n_tail = min(n_tail, numel(x));
    xt = x(end - n_tail + 1:end);
    yt = y(end - n_tail + 1:end);

    xref = logspace(log10(xt(1)), log10(xt(end)), 100);

    % Determine reference-line scale from the whole tail,
    % not from the last point.
    f_tail = xt.^2 .* log(xt);
    scale = median(yt ./ f_tail, 'omitnan');

    % Move dashed line away from the solid curve.
    scale = scale * line_factor;

    fref = xref.^2 .* log(xref);
    yref = scale * fref;

    loglog(xref, yref, 'k--', ...
        'LineWidth', 1.5, ...
        'HandleVisibility', 'off');

    % Label near the dashed line.
    label_pos = min(max(label_pos, 0), 1);
    x_label = exp((1 - label_pos) * log(xt(1)) + label_pos * log(xt(end)));
    y_label = scale * x_label.^2 * log(x_label) * label_factor;

    text(x_label, y_label, label_text, ...
        'Interpreter', 'latex', ...
        'FontSize', 14, ...
        'Color', 'k', ...
        'HorizontalAlignment', 'left', ...
        'VerticalAlignment', 'middle', ...
        'HandleVisibility', 'off');
end