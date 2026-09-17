% ex4_plot.m
%
% Purpose:
%   Plot error and timing curves for the Example 4 GO and VoltMat depth sweeps.
%
% Problem:
%   Oscillatory second-kind Volterra equation with N = 200 and variable kernel
%   depth.
%
% Method:
%   Reads precomputed CSV files from ex4_GO.jl and ex4_VoltMat.jl, filters
%   successful rows, and draws the original comparison style. The timing plot
%   includes tail reference lines used only as visual complexity guides.
%
% Expected input files:
%   data/ex4_GO_depth_sweep_N200.csv
%   data/ex4_VoltMat_depth_sweep_N200.csv
%
% Output files:
%   figures/ex4_depth_error_N200.pdf
%   figures/ex4_depth_error_N200.png
%   figures/ex4_depth_time_N200.pdf
%   figures/ex4_depth_time_N200.png
%
% Run:
%   matlab -batch "run('examples/ex4_oscillatory/ex4_plot.m')"

clear; clc; close all;

%% Plot options
% timePart:
%   "build" : only operator construction time
%   "total" : operator construction + solve time
%
% timeStat:
%   "min"    : less affected by OS jitter
%   "q25"    : lower quartile
%   "median" : BenchmarkTools median
timePart = "total";
timeStat = "min";

%% Reference-label options
% xShiftDecade:
%   > 0 moves the string to the right on log-log axes
%   < 0 moves the string to the left
%
% yShiftDecade:
%   > 0 moves the string upward on log-log axes
%   < 0 moves the string downward
%
% fontSize:
%   font size of the corresponding string
%
% xIndexFromEnd and yIndexFromEnd:
%   choose the base anchor point from the tail curve.
%   For example, 8 means the 8-th point from the end.

vmRefText = struct();
vmRefText.xShiftDecade = -0.1;
vmRefText.yShiftDecade = -0.5;
vmRefText.fontSize     = 12;
vmRefText.xIndexFromEnd = 8;
vmRefText.yIndexFromEnd = 12;

goRefText = struct();
goRefText.xShiftDecade = -0.2;
goRefText.yShiftDecade = 0.3;
goRefText.fontSize     = 12;
goRefText.xIndexFromEnd = 8;
goRefText.yIndexFromEnd = 12;

%% Paths
scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir)
    scriptDir = pwd;
end
addpath(fullfile(fileparts(fileparts(scriptDir)), 'tools'));
paperStyle = voltopt_plot_style();

dataDir = fullfile(scriptDir, 'data');
figDir  = fullfile(scriptDir, 'figures');

if ~exist(figDir, 'dir')
    mkdir(figDir);
end

goFile      = fullfile(dataDir, 'ex4_GO_depth_sweep_N200.csv');
voltmatFile = fullfile(dataDir, 'ex4_VoltMat_depth_sweep_N200.csv');

if ~isfile(goFile)
    error('Cannot find file: %s', goFile);
end

if ~isfile(voltmatFile)
    error('Cannot find file: %s', voltmatFile);
end

%% Read data
Tgo = readtable(goFile);
Tvm = readtable(voltmatFile);

Tgo = keep_success_rows(Tgo);
Tvm = keep_success_rows(Tvm);

Tgo = sortrows(Tgo, 'depth');
Tvm = sortrows(Tvm, 'depth');

%% Error data
TgoErr = Tgo(isfinite(Tgo.depth) & isfinite(Tgo.max_error) & Tgo.max_error > 0, :);
TvmErr = Tvm(isfinite(Tvm.depth) & isfinite(Tvm.max_error) & Tvm.max_error > 0, :);

%% Timing data
goTime = compute_time(Tgo, timePart, timeStat);
vmTime = compute_time(Tvm, timePart, timeStat);

validGOTime = isfinite(Tgo.depth) & isfinite(goTime) & goTime > 0;
validVMTime = isfinite(Tvm.depth) & isfinite(vmTime) & vmTime > 0;

goDepthTime = Tgo.depth(validGOTime);
vmDepthTime = Tvm.depth(validVMTime);
goTime      = goTime(validGOTime);
vmTime      = vmTime(validVMTime);

%% ============================================================
%  Figure 1: error curve
% =============================================================

fig1 = figure('Color', 'w', 'Units', 'centimeters', ...
    'Position', [2, 2, paperStyle.FigureSize]);
hold on;

semilogy(TvmErr.depth, TvmErr.max_error, ...
    'LineWidth', paperStyle.PlotLineWidth);

semilogy(TgoErr.depth, TgoErr.max_error,  ...
    'LineWidth', paperStyle.PlotLineWidth);

xlabel('kernel approximation order $n$', 'Interpreter', 'latex');
ylabel('error', 'Interpreter', 'latex');

legend( ...
    'Duffy--Chebyshev', ...
    'Gulteb--Olver', ...
    'Location', 'northwest', ...
    'Interpreter', 'latex' ...
);

ax1 = gca;
set(ax1, ...
    'FontName', paperStyle.FontName, ...
    'FontSize', 12, ...
    'TickLabelInterpreter', 'latex', ...
    'XScale', 'linear', ...
    'YScale', 'log' ...
);

box on;

pdfFile1 = fullfile(figDir, 'ex4_depth_error_N200.pdf');
pngFile1 = fullfile(figDir, 'ex4_depth_error_N200.png');

voltopt_export_figure(fig1, ax1, pdfFile1, pngFile1);

%% ============================================================
%  Figure 2: timing curve, log-log
% =============================================================

fig2 = figure('Color', 'w', 'Units', 'centimeters', ...
    'Position', [2, 2, paperStyle.FigureSize]);
hold on;

loglog(vmDepthTime, vmTime,  ...
    'LineWidth', paperStyle.PlotLineWidth);

loglog(goDepthTime, goTime,  ...
    'LineWidth', paperStyle.PlotLineWidth);

% Reference complexity curves on the tail.
tailCount = 40;

add_reference_curve(vmDepthTime, vmTime, tailCount, ...
    @(n) n.^2 .* log(n), '$\mathcal{O}(n^2\log n)$', 1.5, vmRefText);

add_reference_curve(goDepthTime, goTime, tailCount, ...
    @(n) n.^3, '$\mathcal{O}(n^3)$', 0.8, goRefText);

xlabel('kernel approximation order $n$', 'Interpreter', 'latex');

if timePart == "build"
    ylabel('construction time (sec)', 'Interpreter', 'latex');
    titleText = sprintf('construction time, %s statistic', timeStat);
else
    ylabel('time (sec)', 'Interpreter', 'latex');
    titleText = sprintf('time, %s statistic', timeStat);
end

legend( ...
    'Duffy--Chebyshev', ...
    'Gulteb--Olver', ...
    'Location', 'northwest', ...
    'Interpreter', 'latex' ...
);

ax2 = gca;
set(ax2, ...
    'FontName', paperStyle.FontName, ...
    'FontSize', 12, ...
    'TickLabelInterpreter', 'latex', ...
    'XScale', 'log', ...
    'YScale', 'log' ...
);

%grid on;
box on;

pdfFile2 = fullfile(figDir, 'ex4_depth_time_N200.pdf');
pngFile2 = fullfile(figDir, 'ex4_depth_time_N200.png');

voltopt_export_figure(fig2, ax2, pdfFile2, pngFile2);

%% Print summary
fprintf('Saved figures:\n');
fprintf('  %s\n', pdfFile1);
fprintf('  %s\n', pngFile1);
fprintf('  %s\n', pdfFile2);
fprintf('  %s\n', pngFile2);

[goBestErr, goIdx] = min(TgoErr.max_error);
[vmBestErr, vmIdx] = min(TvmErr.max_error);

fprintf('\nBest errors:\n');
fprintf('  GO sparse Volterra, N=200: depth = %d, max_error = %.6e\n', ...
    TgoErr.depth(goIdx), goBestErr);
fprintf('  Duffy_Chebyshev,       N=200: depth = %d, max_error = %.6e\n', ...
    TvmErr.depth(vmIdx), vmBestErr);

[goBestTime, goTimeIdx] = min(goTime);
[vmBestTime, vmTimeIdx] = min(vmTime);

fprintf('\nBest times using %s %s statistic:\n', timePart, timeStat);
fprintf('  GO sparse Volterra, N=200: depth = %d, time = %.6e sec\n', ...
    goDepthTime(goTimeIdx), goBestTime);
fprintf('  Duffy_Chebyshev,       N=200: depth = %d, time = %.6e sec\n', ...
    vmDepthTime(vmTimeIdx), vmBestTime);

%% Local helper functions

function T = keep_success_rows(T)
    if ismember('status', T.Properties.VariableNames)
        status = lower(string(T.status));
        T = T(status == "ok" | status == "success", :);
    end
end

function t = compute_time(T, timePart, timeStat)
    n = height(T);

    buildCol = benchmark_column_name("build", timeStat);
    solveCol = benchmark_column_name("solve", timeStat);

    if ismember(buildCol, T.Properties.VariableNames)
        buildTime = T.(buildCol);
    elseif ismember('benchmark_build_median_time', T.Properties.VariableNames)
        buildTime = T.benchmark_build_median_time;
    elseif ismember('operator_build_time', T.Properties.VariableNames)
        buildTime = T.operator_build_time;
    else
        error('No operator build-time column found.');
    end

    if ismember(solveCol, T.Properties.VariableNames)
        solveTime = T.(solveCol);
    elseif ismember('benchmark_solve_median_time', T.Properties.VariableNames)
        solveTime = T.benchmark_solve_median_time;
    elseif ismember('solve_time', T.Properties.VariableNames)
        solveTime = T.solve_time;
    else
        solveTime = zeros(n, 1);
    end

    if timePart == "build"
        t = buildTime;
    elseif timePart == "total"
        t = buildTime + solveTime;
    else
        error('Unknown timePart: %s', timePart);
    end
end

function col = benchmark_column_name(part, stat)
    if stat == "min"
        col = "benchmark_" + part + "_min_time";
    elseif stat == "q25"
        col = "benchmark_" + part + "_q25_time";
    elseif stat == "median"
        col = "benchmark_" + part + "_median_time";
    elseif stat == "q75"
        col = "benchmark_" + part + "_q75_time";
    else
        error('Unknown timeStat: %s', stat);
    end
    col = char(col);
end

function add_reference_curve(depth, timeValue, tailCount, rateFun, labelText, offsetFactor, textOpt)
    depth = depth(:);
    timeValue = timeValue(:);

    valid = isfinite(depth) & isfinite(timeValue) & depth > 1 & timeValue > 0;
    depth = depth(valid);
    timeValue = timeValue(valid);

    if numel(depth) < 3
        return;
    end

    [depth, idx] = sort(depth);
    timeValue = timeValue(idx);

    m = min(tailCount, numel(depth));
    tailDepth = depth(end-m+1:end);
    tailTime  = timeValue(end-m+1:end);

    refShape = rateFun(tailDepth);
    refShape = refShape(:);

    scale = median(tailTime ./ refShape, 'omitnan');
    refTime = offsetFactor * scale * refShape;

    loglog(tailDepth, refTime, 'k--', ...
        'LineWidth', 1.2, ...
        'HandleVisibility', 'off');

    if ~isfield(textOpt, 'xShiftDecade')
        textOpt.xShiftDecade = 0.0;
    end
    if ~isfield(textOpt, 'yShiftDecade')
        textOpt.yShiftDecade = 0.0;
    end
    if ~isfield(textOpt, 'fontSize')
        textOpt.fontSize = 10;
    end
    if ~isfield(textOpt, 'xIndexFromEnd')
        textOpt.xIndexFromEnd = 8;
    end
    if ~isfield(textOpt, 'yIndexFromEnd')
        textOpt.yIndexFromEnd = 10;
    end

    xIdx = max(1, numel(tailDepth) - textOpt.xIndexFromEnd + 1);
    yIdx = max(1, numel(refTime)   - textOpt.yIndexFromEnd + 1);

    xText = tailDepth(xIdx) * 10^(textOpt.xShiftDecade);
    yText = refTime(yIdx)   * 10^(textOpt.yShiftDecade);

    text(xText, yText, ['  ', labelText], ...
        'Interpreter', 'latex', ...
        'FontSize', textOpt.fontSize, ...
        'HorizontalAlignment', 'left', ...
        'VerticalAlignment', 'middle');
end
