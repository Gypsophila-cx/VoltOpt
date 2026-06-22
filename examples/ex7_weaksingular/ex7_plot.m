% ex7_plot.m
%
% Plot Cauchy error and timing curves for Example 7 weakly singular
% Volterra equation.
%
% Input files:
%   data/ex7_FracCollocation_error.csv
%   data/ex7_VoltMat_error.csv
%   data/ex7_FracCollocation_timing.csv
%   data/ex7_VoltMat_timing.csv
%
% Output files:
%   figures/ex7_cauchy_error_grid.pdf
%   figures/ex7_cauchy_error_grid.png
%   figures/ex7_total_time.pdf
%   figures/ex7_total_time.png

clear; clc; close all;

%% Paths
scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir)
    scriptDir = pwd;
end

dataDir = fullfile(scriptDir, 'data');
figDir  = fullfile(scriptDir, 'figures');

if ~exist(figDir, 'dir')
    mkdir(figDir);
end

fracErrFile  = fullfile(dataDir, 'ex7_FracCollocation_error.csv');
voltErrFile  = fullfile(dataDir, 'ex7_VoltMat_error.csv');

fracTimeFile = fullfile(dataDir, 'ex7_FracCollocation_timing.csv');
voltTimeFile = fullfile(dataDir, 'ex7_VoltMat_timing.csv');

requiredFiles = {fracErrFile, voltErrFile, fracTimeFile, voltTimeFile};
for k = 1:numel(requiredFiles)
    if ~isfile(requiredFiles{k})
        error('Missing file: %s', requiredFiles{k});
    end
end

%% Read and filter error data
TfracErr = readtable(fracErrFile);
TvoltErr = readtable(voltErrFile);

TfracErr = filter_success_finite(TfracErr, 'err_grid_inf');
TvoltErr = filter_success_finite(TvoltErr, 'err_grid_inf');

TfracErr = sortrows(TfracErr, 'N');
TvoltErr = sortrows(TvoltErr, 'N');

%% Read and filter timing data
TfracTime = readtable(fracTimeFile);
TvoltTime = readtable(voltTimeFile);

TfracTime = filter_success_finite(TfracTime, 'total_time_median');
TvoltTime = filter_success_finite(TvoltTime, 'total_time_median');

TfracTime = sortrows(TfracTime, 'N');
TvoltTime = sortrows(TvoltTime, 'N');

%% Plot options
set(groot, 'defaultAxesFontName', 'Times New Roman');
set(groot, 'defaultTextFontName', 'Times New Roman');
set(groot, 'defaultLineLineWidth', 2.0);
set(groot, 'defaultAxesFontSize', 16);
set(groot, 'defaultTextFontSize', 16);

%% Figure 1: Cauchy error curve
figErr = figure('Color', 'w');
hold on;

semilogy(TvoltErr.N, TvoltErr.err_grid_inf, '-', ...
    'DisplayName', 'Weak VoltMat-Cheb');

semilogy(TfracErr.N, TfracErr.err_grid_inf, '-', ...
    'DisplayName', 'Fractional collocation');

hold off;

set(gca, 'XScale', 'linear');
set(gca, 'YScale', 'log');

box on;

xlabel('$N$', 'Interpreter', 'latex');
ylabel('Cauchy error', 'Interpreter', 'latex');

legend('Location', 'northeast', 'Interpreter', 'latex');

allNErr = [TfracErr.N; TvoltErr.N];
allNErr = allNErr(isfinite(allNErr));
if ~isempty(allNErr)
    xlim([min(allNErr), max(allNErr)]);
end

allErr = [TfracErr.err_grid_inf; TvoltErr.err_grid_inf];
allErr = allErr(isfinite(allErr) & allErr > 0);
if ~isempty(allErr)
    ymin = min(allErr);
    ymax = max(allErr);
    ylim([10^(floor(log10(ymin)) - 0.2), ...
          10^(ceil(log10(ymax)) + 0.2)]);
end

pdfErrFile = fullfile(figDir, 'ex7_cauchy_error_grid.pdf');
pngErrFile = fullfile(figDir, 'ex7_cauchy_error_grid.png');

exportgraphics(figErr, pdfErrFile, 'ContentType', 'vector');
exportgraphics(figErr, pngErrFile, 'Resolution', 300);

%% Figure 2: Total timing curve
figTime = figure('Color', 'w');
axTime = axes(figTime);
hold(axTime, 'on');

loglog(axTime, TvoltTime.N, TvoltTime.total_time_median, '-', ...
    'DisplayName', 'Weak VoltMat-Cheb');

loglog(axTime, TfracTime.N, TfracTime.total_time_median, '-', ...
    'DisplayName', 'Fractional collocation');

set(axTime, 'XScale', 'log');
set(axTime, 'YScale', 'log');

grid on;
box(axTime, 'on');

xlabel(axTime, '$N$', 'Interpreter', 'latex');
ylabel(axTime, 'time (sec)', 'Interpreter', 'latex');

allNTime = [TfracTime.N; TvoltTime.N];
allNTime = allNTime(isfinite(allNTime) & allNTime > 0);
if ~isempty(allNTime)
    xlim(axTime, [min(allNTime), max(allNTime)]);
end

allTime = [TfracTime.total_time_median; TvoltTime.total_time_median];
allTime = allTime(isfinite(allTime) & allTime > 0);
if ~isempty(allTime)
    ymin = min(allTime);
    ymax = max(allTime);
    ylim(axTime, [10^(floor(log10(ymin)) - 0.35), ...
                  10^(ceil(log10(ymax)) + 0.35)]);
end

%% Guide-line label position parameters
% labelPos:
%   Position along the dashed segment. Smaller means left, larger means right.
%
% labelDxLog:
%   Extra horizontal displacement in log10(N).
%   Positive: move right. Negative: move left.
%
% labelDyLog:
%   Extra vertical displacement in log10(time).
%   Positive: move up. Negative: move down.

fracLabelPos   = 0.40;
fracLabelDxLog = -0.2;
fracLabelDyLog =  0.08;

voltLabelPos   = 0.40;
voltLabelDxLog = -0.02;
voltLabelDyLog = -0.08;

add_tail_power_guide(axTime, TfracTime.N, TfracTime.total_time_median, ...
    3, '$\mathcal{O}(N^3)$', 'above', ...
    fracLabelPos, fracLabelDxLog, fracLabelDyLog);

add_tail_power_guide(axTime, TvoltTime.N, TvoltTime.total_time_median, ...
    2, '$\mathcal{O}(N^2)$', 'below', ...
    voltLabelPos, voltLabelDxLog, voltLabelDyLog);

legend(axTime, 'Location', 'northwest', 'Interpreter', 'latex');

hold(axTime, 'off');

pdfTimeFile = fullfile(figDir, 'ex7_total_time.pdf');
pngTimeFile = fullfile(figDir, 'ex7_total_time.png');

exportgraphics(figTime, pdfTimeFile, 'ContentType', 'vector');
exportgraphics(figTime, pngTimeFile, 'Resolution', 300);

%% Print output paths
fprintf('Saved figures:\n');
fprintf('  %s\n', pdfErrFile);
fprintf('  %s\n', pngErrFile);
fprintf('  %s\n', pdfTimeFile);
fprintf('  %s\n', pngTimeFile);

%% Local helper: filter table rows
function T = filter_success_finite(T, colName)
    if ~ismember('N', T.Properties.VariableNames)
        error('Input table does not contain column N.');
    end

    if ~ismember(colName, T.Properties.VariableNames)
        error('Input table does not contain column %s.', colName);
    end

    ok = true(height(T), 1);

    if ismember('status', T.Properties.VariableNames)
        status = string(T.status);
        ok = ok & strcmpi(strtrim(status), "success");
    end

    vals = T.(colName);

    ok = ok & isfinite(T.N) & T.N > 0;
    ok = ok & isfinite(vals) & vals > 0;

    T = T(ok, :);

    if isempty(T)
        error('No valid successful rows remain after filtering column %s.', colName);
    end
end

%% Local helper: add tail power-law guide line
function add_tail_power_guide(ax, N, y, power, labelText, labelSide, labelPos, labelDxLog, labelDyLog)
    N = N(:);
    y = y(:);

    ok = isfinite(N) & isfinite(y) & N > 0 & y > 0;
    N = N(ok);
    y = y(ok);

    if numel(N) < 4
        return;
    end

    if nargin < 7 || isempty(labelPos)
        labelPos = 0.68;
    end
    if nargin < 8 || isempty(labelDxLog)
        labelDxLog = 0.0;
    end
    if nargin < 9 || isempty(labelDyLog)
        labelDyLog = 0.0;
    end

    labelPos = min(max(labelPos, 0.05), 0.95);

    [N, idx] = sort(N);
    y = y(idx);

    m = numel(N);

    tailCount = max(4, ceil(0.22 * m));
    tailIdx = (m - tailCount + 1):m;

    Nt = N(tailIdx);
    yt = y(tailIdx);

    x1 = Nt(1);
    x2 = Nt(end);
    xGuide = logspace(log10(x1), log10(x2), 80).';

    c = mean(log(yt) - power * log(Nt));

    switch lower(labelSide)
        case 'above'
            guideShift = 1.55;
            vAlign = 'bottom';
        case 'below'
            guideShift = 0.62;
            vAlign = 'top';
        otherwise
            guideShift = 1.0;
            vAlign = 'bottom';
    end

    yGuide = exp(c) .* xGuide.^power .* guideShift;

    yLim = ylim(ax);
    yGuide = min(max(yGuide, yLim(1) * 1.02), yLim(2) / 1.02);

    loglog(ax, xGuide, yGuide, 'k--', ...
        'LineWidth', 1.4, ...
        'HandleVisibility', 'off');

    labelIdx = round(labelPos * numel(xGuide));
    labelIdx = min(max(labelIdx, 1), numel(xGuide));

    xLabel = xGuide(labelIdx);
    yLabel = yGuide(labelIdx);

    xLabel = 10^(log10(xLabel) + labelDxLog);
    yLabel = 10^(log10(yLabel) + labelDyLog);

    xLim = xlim(ax);
    yLim = ylim(ax);

    logXMin = log10(xLim(1));
    logXMax = log10(xLim(2));
    logYMin = log10(yLim(1));
    logYMax = log10(yLim(2));

    xMargin = 0.04 * (logXMax - logXMin);
    yMargin = 0.06 * (logYMax - logYMin);

    xLabel = 10^min(max(log10(xLabel), logXMin + xMargin), logXMax - xMargin);
    yLabel = 10^min(max(log10(yLabel), logYMin + yMargin), logYMax - yMargin);

    text(ax, xLabel, yLabel, labelText, ...
        'Interpreter', 'latex', ...
        'FontSize', 15, ...
        'HorizontalAlignment', 'left', ...
        'VerticalAlignment', vAlign, ...
        'Color', 'k', ...
        'Clipping', 'on', ...
        'HandleVisibility', 'off');
end