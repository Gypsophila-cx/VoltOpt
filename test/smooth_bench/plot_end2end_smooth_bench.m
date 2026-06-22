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
    'coefficient size $C$', ...
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
    end

    if plotted == 0
        error('No positive finite timing data are available for %s.', outPdf);
    end

    xlabel(xLabelText, 'Interpreter', 'latex');
    ylabel('time (sec)', 'Interpreter', 'latex');
    legend('Location', 'northwest', 'Interpreter', 'latex');

    ax = gca;
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
