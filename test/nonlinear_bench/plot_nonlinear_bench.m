% Plot nonlinear benchmark error and timing curves.
%
% Input file:
%   data/bench_nonlinear_volterra_clenshaw_vs_compvolt.csv
%
% Output files:
%   figures/nonlinear_check_error.pdf
%   figures/nonlinear_check_error.png
%   figures/nonlinear_timing.pdf
%   figures/nonlinear_timing.png

clear; clc; close all;

%% Paths
scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir)
    scriptDir = pwd;
end

dataDir = fullfile(scriptDir, 'data');
figDir = fullfile(scriptDir, 'figures');

if ~exist(figDir, 'dir')
    mkdir(figDir);
end

benchFile = fullfile(dataDir, 'bench_nonlinear_volterra_clenshaw_vs_compvolt.csv');

if ~isfile(benchFile)
    error('Cannot find nonlinear benchmark CSV: %s', benchFile);
end

T = readtable(benchFile);
require_columns(T, {'method', 'N', 'time', 'time_median', 'check_norm'}, benchFile);

%% Global style
set(groot, 'defaultAxesFontName', 'Times New Roman');
set(groot, 'defaultTextFontName', 'Times New Roman');
set(groot, 'defaultLegendFontName', 'Times New Roman');
set(groot, 'defaultAxesFontSize', 15);
set(groot, 'defaultLineLineWidth', 2.0);

%% Figure 1: consistency error on overlapping N values
plot_error_curve( ...
    T, ...
    fullfile(figDir, 'nonlinear_check_error.pdf'), ...
    fullfile(figDir, 'nonlinear_check_error.png'));

%% Figure 2: timing by method
plot_timing_curves( ...
    T, ...
    fullfile(figDir, 'nonlinear_timing.pdf'), ...
    fullfile(figDir, 'nonlinear_timing.png'));

fprintf('Saved figures to: %s\n', figDir);

%% Local helper functions

function require_columns(T, names, fileName)
    missingNames = setdiff(names, T.Properties.VariableNames);
    if ~isempty(missingNames)
        error('Missing required columns in %s: %s', fileName, strjoin(missingNames, ', '));
    end
end

function plot_error_curve(T, outPdf, outPng)
    % check_norm is already the difference between
    % Clenshaw value matrix and compVolt, so only one curve is drawn.

    methodStrings = string(T.method);

    N_all = get_numeric_col(T, 'N');
    err_all = get_numeric_col(T, 'check_norm');

    N_compVolt = unique(N_all(methodStrings == "compVolt"));
    N_clenshaw = unique(N_all(methodStrings == "clenshaw_valuemat"));

    % Only compare on overlapping N values.
    N_overlap = intersect(N_compVolt, N_clenshaw);
    N_overlap = N_overlap(isfinite(N_overlap) & N_overlap > 0);
    N_overlap = sort(N_overlap);

    x = [];
    y = [];

    for j = 1:numel(N_overlap)
        Nj = N_overlap(j);

        rows = N_all == Nj & isfinite(err_all);
        errVals = err_all(rows);

        if isempty(errVals)
            continue;
        end

        % Use one representative value for this N.
        x(end + 1, 1) = Nj; %#ok<AGROW>
        y(end + 1, 1) = median(errVals); %#ok<AGROW>
    end

    valid = isfinite(x) & isfinite(y) & x > 0;
    x = x(valid);
    y = y(valid);

    if isempty(x)
        error('No finite overlapping consistency-error data are available for %s.', outPdf);
    end

    [x, idx] = sort(x);
    y = y(idx);

    fig = figure('Color', 'w', 'Position', [100, 100, 760, 560]);
    hold on; box on; grid on;

    plot(x, y);

    xlabel('truncation dimension $N$', 'Interpreter', 'latex');
    ylabel('$\|F_{\mathrm{Clenshaw}} - F_{\mathrm{compVolt}}\|_{\infty}$', ...
        'Interpreter', 'latex');

    ax = gca;
    ax.FontName = 'Times New Roman';
    ax.XScale = 'linear';
    ax.YScale = 'linear';
    ax.XGrid = 'on';
    ax.YGrid = 'on';
    ax.YMinorGrid = 'on';
    ax.MinorGridLineStyle = ':';

    ylim([0, 3e-16]);

    exportgraphics(fig, outPdf, 'ContentType', 'vector');
    exportgraphics(fig, outPng, 'Resolution', 300);

    fprintf('Saved: %s\n', outPdf);
    fprintf('Saved: %s\n', outPng);
end

function plot_timing_curves(T, outPdf, outPng)
    % Required drawing order: compVolt first, Clenshaw second.
    methodOrder = ["compVolt", "clenshaw_valuemat"];
    allMethods = unique(string(T.method), 'stable');
    methods = methodOrder(ismember(methodOrder, allMethods));
    extraMethods = setdiff(allMethods, methods, 'stable');
    methods = [methods(:); extraMethods(:)];

    %% Auxiliary-line controls
    % Each auxiliary line is drawn only on the tail of its own method curve.
    refTailFraction = 0.40;

    % Reference-line vertical shifts.
    % Values smaller than 1 move the dashed reference lines downward.
    lineCompVoltN2logN_yScale = 0.65;
    lineClenshawN3_yScale     = 0.65;

    % compVolt auxiliary line: O(N^2 log N).
    % On log-log axes, xShift and yShift are multiplicative factors.
    labelCompVoltN2logN_xShift   = 1.00;  % >1 moves right, <1 moves left
    labelCompVoltN2logN_yShift   = 0.25;  % >1 moves up,    <1 moves down
    labelCompVoltN2logN_fontSize = 12;

    % Clenshaw auxiliary line: O(N^3).
    % On log-log axes, xShift and yShift are multiplicative factors.
    labelClenshawN3_xShift   = 1.25;      % >1 moves right, <1 moves left
    labelClenshawN3_yShift   = 0.75;      % >1 moves up,    <1 moves down
    labelClenshawN3_fontSize = 12;

    fig = figure('Color', 'w', 'Position', [100, 100, 760, 560]);
    hold on; box on; grid on;

    plotted = 0;
    dataByMethod = struct();

    for k = 1:numel(methods)
        method = methods(k);
        rows = string(T.method) == method;
        Tk = T(rows, :);

        x = get_numeric_col(Tk, 'N');
        y = get_numeric_col(Tk, 'time_median');

        valid = isfinite(x) & isfinite(y) & x > 0 & y > 0;
        x = x(valid);
        y = y(valid);

        if isempty(x)
            continue;
        end

        [x, idx] = sort(x);
        y = y(idx);

        loglog(x, y, ...
            'DisplayName', method_label(method));

        plotted = plotted + 1;

        fieldName = matlab.lang.makeValidName(char(method));
        dataByMethod.(fieldName).x = x;
        dataByMethod.(fieldName).y = y;
    end

    if plotted == 0
        error('No positive finite timing data are available for %s.', outPdf);
    end

    add_tail_reference_lines( ...
        dataByMethod, refTailFraction, ...
        lineCompVoltN2logN_yScale, lineClenshawN3_yScale, ...
        labelCompVoltN2logN_xShift, labelCompVoltN2logN_yShift, labelCompVoltN2logN_fontSize, ...
        labelClenshawN3_xShift, labelClenshawN3_yShift, labelClenshawN3_fontSize);

    xlabel('$N$', 'Interpreter', 'latex');
    ylabel('time (sec)', 'Interpreter', 'latex');
    legend('Location', 'northwest', 'Interpreter', 'latex');

    ax = gca;
    ax.FontName = 'Times New Roman';
    ax.XScale = 'log';
    ax.YScale = 'log';
    ax.XGrid = 'on';
    ax.YGrid = 'on';
    ax.YMinorGrid = 'on';
    ax.MinorGridLineStyle = ':';

    exportgraphics(fig, outPdf, 'ContentType', 'vector');
    exportgraphics(fig, outPng, 'Resolution', 300);

    fprintf('Saved: %s\n', outPdf);
    fprintf('Saved: %s\n', outPng);
end

function add_tail_reference_lines( ...
    dataByMethod, refTailFraction, ...
    lineCompVoltN2logN_yScale, lineClenshawN3_yScale, ...
    labelCompVoltN2logN_xShift, labelCompVoltN2logN_yShift, labelCompVoltN2logN_fontSize, ...
    labelClenshawN3_xShift, labelClenshawN3_yShift, labelClenshawN3_fontSize)

    % compVolt: O(N^2 log N), drawn near the tail of the compVolt data.
    if isfield(dataByMethod, matlab.lang.makeValidName('compVolt'))
        [xRef, yRef] = make_tail_reference( ...
            dataByMethod, 'compVolt', refTailFraction, 'N2logN');

        if ~isempty(xRef)
            yRef = lineCompVoltN2logN_yScale * yRef;

            loglog(xRef, yRef, 'k--', ...
                'LineWidth', 1.5, ...
                'HandleVisibility', 'off');

            labelIdx = max(1, round(0.68 * numel(xRef)));
            text( ...
                xRef(labelIdx) * labelCompVoltN2logN_xShift, ...
                yRef(labelIdx) * labelCompVoltN2logN_yShift, ...
                '$\mathcal{O}(N^2 \log N)$', ...
                'Interpreter', 'latex', ...
                'FontName', 'Times New Roman', ...
                'FontSize', labelCompVoltN2logN_fontSize, ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'middle');
        end
    end

    % Clenshaw: O(N^3), drawn near the tail of the Clenshaw data.
    if isfield(dataByMethod, matlab.lang.makeValidName('clenshaw_valuemat'))
        [xRef, yRef] = make_tail_reference( ...
            dataByMethod, 'clenshaw_valuemat', refTailFraction, 'N3');

        if ~isempty(xRef)
            yRef = lineClenshawN3_yScale * yRef;

            loglog(xRef, yRef, 'k--', ...
                'LineWidth', 1.5, ...
                'HandleVisibility', 'off');

            labelIdx = max(1, round(0.68 * numel(xRef)));
            text( ...
                xRef(labelIdx) * labelClenshawN3_xShift, ...
                yRef(labelIdx) * labelClenshawN3_yShift, ...
                '$\mathcal{O}(N^3)$', ...
                'Interpreter', 'latex', ...
                'FontName', 'Times New Roman', ...
                'FontSize', labelClenshawN3_fontSize, ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'middle');
        end
    end
end

function [xRef, yRef] = make_tail_reference(dataByMethod, methodName, refTailFraction, modelName)
    fieldName = matlab.lang.makeValidName(methodName);

    xRef = [];
    yRef = [];

    if ~isfield(dataByMethod, fieldName)
        return;
    end

    x = dataByMethod.(fieldName).x;
    y = dataByMethod.(fieldName).y;

    valid = isfinite(x) & isfinite(y) & x > 0 & y > 0;
    x = x(valid);
    y = y(valid);

    if numel(x) < 2
        return;
    end

    [x, idx] = sort(x);
    y = y(idx);

    xMin = min(x);
    xMax = max(x);

    if xMax <= xMin
        return;
    end

    logXMin = log10(xMin);
    logXMax = log10(xMax);
    logXStart = logXMax - refTailFraction * (logXMax - logXMin);

    xRef = logspace(logXStart, logXMax, 100).';

    xAnchor = x(end);
    yAnchor = y(end);

    switch modelName
        case 'N2logN'
            yRef = yAnchor .* reference_N2logN(xRef, xAnchor);
        case 'N3'
            yRef = yAnchor .* (xRef ./ xAnchor).^3;
        otherwise
            error('Unknown reference model: %s', modelName);
    end
end

function scale = reference_N2logN(x, xAnchor)
    x = double(x);
    xAnchor = double(xAnchor);

    logFactor = log(max(x, exp(1)));
    logAnchor = log(max(xAnchor, exp(1)));

    scale = (x ./ xAnchor).^2 .* (logFactor ./ logAnchor);
end

function label = method_label(method)
    method = char(method);
    switch method
        case 'clenshaw_valuemat'
            label = 'Clenshaw value matrix';
        case 'compVolt'
            label = 'compVolt';
        otherwise
            label = strrep(method, '_', '\_');
    end
end

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