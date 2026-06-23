% Plot smooth benchmark timing curves.
%
% Input files:
%   data/bench_smooth_fixedC_varyN.csv
%   data/bench_smooth_fixedN_varyC.csv
%
% Output files:
%   figures/smooth_fixedC_varyN_timing.pdf
%   figures/smooth_fixedC_varyN_timing.png
%   figures/smooth_fixedN_varyC_timing.pdf
%   figures/smooth_fixedN_varyC_timing.png

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

fixedCFile = fullfile(dataDir, 'bench_smooth_fixedC_varyN.csv');
fixedNFile = fullfile(dataDir, 'bench_smooth_fixedN_varyC.csv');

if ~isfile(fixedCFile)
    error('Cannot find smooth fixed-C benchmark CSV: %s', fixedCFile);
end

if ~isfile(fixedNFile)
    error('Cannot find smooth fixed-N benchmark CSV: %s', fixedNFile);
end

T_fixedC = readtable(fixedCFile);
T_fixedN = readtable(fixedNFile);

require_columns(T_fixedC, {'N', 'method', 't_total', 'status'}, fixedCFile);
require_columns(T_fixedN, {'C_rows', 'method', 't_total', 'status'}, fixedNFile);

warn_non_success(T_fixedC, fixedCFile);
warn_non_success(T_fixedN, fixedNFile);

T_fixedC = keep_success_rows(T_fixedC);
T_fixedN = keep_success_rows(T_fixedN);

%% Global style
set(groot, 'defaultAxesFontName', 'Times New Roman');
set(groot, 'defaultTextFontName', 'Times New Roman');
set(groot, 'defaultLegendFontName', 'Times New Roman');
set(groot, 'defaultAxesFontSize', 15);
set(groot, 'defaultLineLineWidth', 2.0);

%% Figure 1: fixed coefficient size C, varying N
plot_timing_curves( ...
    T_fixedC, ...
    'N', ...
    'truncation dimension $N$', ...
    fullfile(figDir, 'smooth_fixedC_varyN_timing.pdf'), ...
    fullfile(figDir, 'smooth_fixedC_varyN_timing.png'));

%% Figure 2: fixed N, varying coefficient size C
plot_timing_curves( ...
    T_fixedN, ...
    'C_rows', ...
    'kernel depth $n$', ...
    fullfile(figDir, 'smooth_fixedN_varyC_timing.pdf'), ...
    fullfile(figDir, 'smooth_fixedN_varyC_timing.png'));

fprintf('Saved figures to: %s\n', figDir);

%% Local helper functions

function require_columns(T, names, fileName)
    missingNames = setdiff(names, T.Properties.VariableNames);
    if ~isempty(missingNames)
        error('Missing required columns in %s: %s', fileName, strjoin(missingNames, ', '));
    end
end

function warn_non_success(T, fileName)
    if ~ismember('status', T.Properties.VariableNames)
        return;
    end

    status = string(T.status);
    bad = status ~= "success";
    if any(bad)
        warning('%s contains %d rows with status != success.', fileName, nnz(bad));
    end
end

function T = keep_success_rows(T)
    keep = true(height(T), 1);

    if ismember('status', T.Properties.VariableNames)
        keep = keep & string(T.status) == "success";
    end

    if ismember('method_status', T.Properties.VariableNames)
        keep = keep & string(T.method_status) ~= "failed";
    end

    T = T(keep, :);
end

function plot_timing_curves(T, xCol, xLabelText, outPdf, outPng)
    if height(T) == 0
        error('No successful timing rows are available for %s.', outPdf);
    end

    methodOrder = ["smooth_cheb_fft_convolution", "gutleb_olver_strictband"];
    allMethods = unique(string(T.method), 'stable');
    methods = methodOrder(ismember(methodOrder, allMethods));
    extraMethods = setdiff(allMethods, methods, 'stable');
    methods = [methods(:); extraMethods(:)];

    colors = lines(max(numel(methods), 1));

    fig = figure('Color', 'w', 'Position', [100, 100, 760, 560]);
    hold on; box on; grid on;

    plotted = 0;
    dataByMethod = struct();

    for k = 1:numel(methods)
        method = methods(k);
        rows = string(T.method) == method;
        Tk = T(rows, :);

        x = get_numeric_col(Tk, xCol);
        y = get_numeric_col(Tk, 't_total');

        valid = isfinite(x) & isfinite(y) & x > 0 & y > 0;
        x = x(valid);
        y = y(valid);

        if isempty(x)
            continue;
        end

        [x, idx] = sort(x);
        y = y(idx);

        loglog(x, y, '-', ...
            'Color', colors(k, :), ...
            'DisplayName', method_label(method));
        plotted = plotted + 1;

        fieldName = matlab.lang.makeValidName(char(method));
        dataByMethod.(fieldName).x = x;
        dataByMethod.(fieldName).y = y;
    end

    if plotted == 0
        error('No positive finite timing data are available for %s.', outPdf);
    end

    add_auxiliary_reference_lines(dataByMethod, xCol);

    xlabel(xLabelText, 'Interpreter', 'latex');
    ylabel('time (sec)', 'Interpreter', 'latex');
    legend('Location', 'northwest', 'Interpreter', 'latex');

    ax = gca;
    ax.FontName = 'Times New Roman';
    ax.XScale = 'log';
    ax.YScale = 'log';
    ax.XGrid = 'on';
    ax.YGrid = 'on';
    ax.XMinorGrid = 'on';
    ax.YMinorGrid = 'on';
    ax.MinorGridLineStyle = ':';

    exportgraphics(fig, outPdf, 'ContentType', 'vector');
    exportgraphics(fig, outPng, 'Resolution', 300);

    fprintf('Saved: %s\n', outPdf);
    fprintf('Saved: %s\n', outPng);
end

function add_auxiliary_reference_lines(dataByMethod, xCol)
    %% Auxiliary-line controls

    % Common line style.
    auxLineStyle = 'k--';
    auxLineWidth = 1.5;

    if strcmp(xCol, 'N')
        % Figure 1: fixed coefficient size C, varying N.

        % Duffy--Cheb auxiliary line: O(N).
        duffy_tailFraction      = 0.40;
        duffy_line_xScale       = 1.00;   % >1 moves line right, <1 moves line left
        duffy_line_yScale       = 0.90;   % >1 moves line up,    <1 moves line down
        duffy_label_posFraction = 0.70;   % label position along the dashed line
        duffy_label_xShift      = 1.00;   % >1 moves label right, <1 moves label left
        duffy_label_yShift      = 0.7;    % >1 moves label up,    <1 moves label down
        duffy_label_fontSize    = 12;

        % Gutleb--Olver auxiliary line: O(N).
        go_tailFraction      = 0.40;
        go_line_xScale       = 1.00;      % >1 moves line right, <1 moves line left
        go_line_yScale       = 0.65;      % >1 moves line up,    <1 moves line down
        go_label_posFraction = 0.70;      % label position along the dashed line
        go_label_xShift      = 1.00;      % >1 moves label right, <1 moves label left
        go_label_yShift      = 0.75;      % >1 moves label up,    <1 moves label down
        go_label_fontSize    = 12;

        draw_reference_line( ...
            dataByMethod, ...
            'smooth_cheb_fft_convolution', ...
            'N', ...
            '$\mathcal{O}(N)$', ...
            duffy_tailFraction, ...
            duffy_line_xScale, ...
            duffy_line_yScale, ...
            duffy_label_posFraction, ...
            duffy_label_xShift, ...
            duffy_label_yShift, ...
            duffy_label_fontSize, ...
            auxLineStyle, ...
            auxLineWidth);

        draw_reference_line( ...
            dataByMethod, ...
            'gutleb_olver_strictband', ...
            'N', ...
            '$\mathcal{O}(N)$', ...
            go_tailFraction, ...
            go_line_xScale, ...
            go_line_yScale, ...
            go_label_posFraction, ...
            go_label_xShift, ...
            go_label_yShift, ...
            go_label_fontSize, ...
            auxLineStyle, ...
            auxLineWidth);

    elseif strcmp(xCol, 'C_rows')
        % Figure 2: fixed N, varying coefficient size n.

        % Duffy--Cheb auxiliary line: O(n^2 log n).
        duffy_tailFraction      = 0.40;
        duffy_line_xScale       = 1.00;   % >1 moves line right, <1 moves line left
        duffy_line_yScale       = 0.80;   % >1 moves line up,    <1 moves line down
        duffy_label_posFraction = 0.70;   % label position along the dashed line
        duffy_label_xShift      = 1.20;   % >1 moves label right, <1 moves label left
        duffy_label_yShift      = 0.60;   % >1 moves label up,    <1 moves label down
        duffy_label_fontSize    = 12;

        % Gutleb--Olver auxiliary line: O(n^3).
        go_tailFraction      = 0.40;
        go_line_xScale       = 1.00;      % >1 moves line right, <1 moves line left
        go_line_yScale       = 0.8;       % >1 moves line up,    <1 moves line down
        go_label_posFraction = 0.70;      % label position along the dashed line
        go_label_xShift      = 1.20;      % >1 moves label right, <1 moves label left
        go_label_yShift      = 0.80;      % >1 moves label up,    <1 moves label down
        go_label_fontSize    = 12;

        draw_reference_line( ...
            dataByMethod, ...
            'smooth_cheb_fft_convolution', ...
            'C2logC', ...
            '$\mathcal{O}(n^2 \log n)$', ...
            duffy_tailFraction, ...
            duffy_line_xScale, ...
            duffy_line_yScale, ...
            duffy_label_posFraction, ...
            duffy_label_xShift, ...
            duffy_label_yShift, ...
            duffy_label_fontSize, ...
            auxLineStyle, ...
            auxLineWidth);

        draw_reference_line( ...
            dataByMethod, ...
            'gutleb_olver_strictband', ...
            'C3', ...
            '$\mathcal{O}(n^3)$', ...
            go_tailFraction, ...
            go_line_xScale, ...
            go_line_yScale, ...
            go_label_posFraction, ...
            go_label_xShift, ...
            go_label_yShift, ...
            go_label_fontSize, ...
            auxLineStyle, ...
            auxLineWidth);
    end
end

function draw_reference_line( ...
    dataByMethod, methodName, modelName, labelText, ...
    tailFraction, lineXScale, lineYScale, ...
    labelPosFraction, labelXShift, labelYShift, labelFontSize, ...
    auxLineStyle, auxLineWidth)

    [xRef, yRef] = make_tail_reference( ...
        dataByMethod, methodName, tailFraction, modelName, lineXScale, lineYScale);

    if isempty(xRef)
        return;
    end

    loglog(xRef, yRef, auxLineStyle, ...
        'LineWidth', auxLineWidth, ...
        'HandleVisibility', 'off');

    labelIdx = max(1, min(numel(xRef), round(labelPosFraction * numel(xRef))));

    text( ...
        xRef(labelIdx) * labelXShift, ...
        yRef(labelIdx) * labelYShift, ...
        labelText, ...
        'Interpreter', 'latex', ...
        'FontName', 'Times New Roman', ...
        'FontSize', labelFontSize, ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'middle');
end

function [xRef, yRef] = make_tail_reference( ...
    dataByMethod, methodName, tailFraction, modelName, lineXScale, lineYScale)

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
    logXStart = logXMax - tailFraction * (logXMax - logXMin);

    xRef = logspace(logXStart, logXMax, 100).';
    xRef = lineXScale * xRef;

    xRef = xRef(isfinite(xRef) & xRef > 0);

    if isempty(xRef)
        return;
    end

    xAnchor = x(end);
    yAnchor = lineYScale * y(end);

    yRef = yAnchor .* reference_scale(xRef, xAnchor, modelName);

    validRef = isfinite(xRef) & isfinite(yRef) & xRef > 0 & yRef > 0;
    xRef = xRef(validRef);
    yRef = yRef(validRef);
end

function scale = reference_scale(x, xAnchor, modelName)
    x = double(x);
    xAnchor = double(xAnchor);

    switch modelName
        case 'N'
            scale = x ./ xAnchor;

        case 'NlogN'
            logFactor = log(max(x, exp(1)));
            logAnchor = log(max(xAnchor, exp(1)));
            scale = (x ./ xAnchor) .* (logFactor ./ logAnchor);

        case 'N2'
            scale = (x ./ xAnchor).^2;

        case 'N3'
            scale = (x ./ xAnchor).^3;

        case 'C'
            scale = x ./ xAnchor;

        case 'ClogC'
            logFactor = log(max(x, exp(1)));
            logAnchor = log(max(xAnchor, exp(1)));
            scale = (x ./ xAnchor) .* (logFactor ./ logAnchor);

        case 'C2'
            scale = (x ./ xAnchor).^2;

        case 'C2logC'
            logFactor = log(max(x, exp(1)));
            logAnchor = log(max(xAnchor, exp(1)));
            scale = (x ./ xAnchor).^2 .* (logFactor ./ logAnchor);

        case 'C3'
            scale = (x ./ xAnchor).^3;

        otherwise
            error('Unknown reference model: %s', modelName);
    end
end

function label = method_label(method)
    method = char(method);
    switch method
        case 'smooth_cheb_fft_convolution'
            label = 'Duffy--Cheb';
        case 'gutleb_olver_strictband'
            label = 'Gutleb--Olver';
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