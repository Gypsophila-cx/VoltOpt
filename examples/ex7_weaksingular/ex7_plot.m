% ex7_plot.m
%
% Purpose:
%   Plot the formal two-method Cauchy error and timing curves for Example 7.
%
% Problem:
%   Weakly singular Volterra equation solved by Weak Duffy--CFP and
%   Muntz--collocation (the standard Hou comparison method).
%
% Method:
%   Reads the four formal CSV files, keeps valid rows, and draws the
%   two-method error and total-time comparison figures.
%   The error sweep extends to N=2000; timing extends only to N=1500.
%   Existing auxiliary dashed complexity guides are retained as explicit
%   line objects.  Only the timing axes grid is disabled.
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
%
% Run:
%   matlab -batch "run('examples/ex7_weaksingular/ex7_plot.m')"

clear; clc; close all;

% Per-run figure switches.  This error-only refresh intentionally leaves the
% existing timing CSV files and timing figure files untouched.
plotErrorFigure  = true;
plotTimingFigure = true;

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

requiredFiles = {};
if plotErrorFigure
    requiredFiles = [requiredFiles, {fracErrFile, voltErrFile}];
end
if plotTimingFigure
    requiredFiles = [requiredFiles, {fracTimeFile, voltTimeFile}];
end
for k = 1:numel(requiredFiles)
    if ~isfile(requiredFiles{k})
        error('Missing file: %s', requiredFiles{k});
    end
end

%% Read and filter error data
if plotErrorFigure
    TfracErr = readtable(fracErrFile);
    TvoltErr = readtable(voltErrFile);

    TfracErr = filter_success_finite(TfracErr, 'cauchy_error');
    TvoltErr = filter_success_finite(TvoltErr, 'cauchy_error');

    TfracErr = sortrows(TfracErr, 'N');
    TvoltErr = sortrows(TvoltErr, 'N');
end

%% Read and filter timing data
if plotTimingFigure
    TfracTime = readtable(fracTimeFile);
    TvoltTime = readtable(voltTimeFile);

    TfracTime = filter_success_finite(TfracTime, 'total_median');
    TvoltTime = filter_success_finite(TvoltTime, 'total_median');

    TfracTime = sortrows(TfracTime, 'N');
    TvoltTime = sortrows(TvoltTime, 'N');
end

%% Plot options
set(groot, 'defaultAxesFontName', 'Times New Roman');
set(groot, 'defaultTextFontName', 'Times New Roman');
set(groot, 'defaultLineLineWidth', 1.5);
set(groot, 'defaultAxesFontSize', 12);
set(groot, 'defaultTextFontSize', 12);

%% Figure 1: Cauchy error curve
if plotErrorFigure
figErr = figure('Color', 'w');
hold on;

semilogy(TvoltErr.N, TvoltErr.cauchy_error, '-', ...
    'DisplayName', 'Weak Duffy-CFP');

semilogy(TfracErr.N, TfracErr.cauchy_error, '-', ...
    'DisplayName', 'Müntz-collocation');

hold off;

set(gca, 'XScale', 'linear');
set(gca, 'YScale', 'log');

box on;

xlabel('$N$', 'Interpreter', 'latex');
ylabel('Cauchy error', 'Interpreter', 'none');

legend('Location', 'northeast', 'Interpreter', 'none');

axError = gca;
xlim(axError, [0, 2000]);

allErr = [TfracErr.cauchy_error; TvoltErr.cauchy_error];
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
end

%% Figure 2: Total timing curve
if plotTimingFigure
figTime = figure('Color', 'w');
axTime = axes(figTime);
hold(axTime, 'on');

loglog(axTime, TvoltTime.N, TvoltTime.total_median, '-', ...
    'DisplayName', 'Weak Duffy-CFP');

loglog(axTime, TfracTime.N, TfracTime.total_median, '-', ...
    'DisplayName', 'Müntz-collocation');

set(axTime, 'XScale', 'log');
set(axTime, 'YScale', 'log');

grid(axTime, 'off');
box(axTime, 'on');

xlabel(axTime, '$N$', 'Interpreter', 'latex');
ylabel(axTime, 'time (sec)', 'Interpreter', 'latex');

allNTime = [TfracTime.N; TvoltTime.N];
allNTime = allNTime(isfinite(allNTime) & allNTime > 0);
if ~isempty(allNTime)
    xlim(axTime, [min(allNTime), max(allNTime)]);
end

allTime = [TfracTime.total_median; TvoltTime.total_median];
allTime = allTime(isfinite(allTime) & allTime > 0);
if ~isempty(allTime)
    ymin = min(allTime);
    ymax = max(allTime);
    ylim(axTime, [10^(floor(log10(ymin)) - 0.35), ...
                  10^(ceil(log10(ymax)) + 0.35)]);
end

%% Existing auxiliary dashed complexity guides
% These are explicit line objects, not axes-grid lines.  Keep their slope,
% style, placement logic, labels, and legend exclusion unchanged.
fracLabelPos   = 0.40;
fracLabelDxLog = -0.2;
fracLabelDyLog =  0.08;

voltLabelPos   = 0.40;
voltLabelDxLog = -0.02;
voltLabelDyLog = -0.08;

add_tail_power_guide(axTime, TfracTime.N, TfracTime.total_median, ...
    3, '$\mathcal{O}(N^3)$', 'above', ...
    fracLabelPos, fracLabelDxLog, fracLabelDyLog);

add_tail_power_guide(axTime, TvoltTime.N, TvoltTime.total_median, ...
    2, '$\mathcal{O}(N^2)$', 'below', ...
    voltLabelPos, voltLabelDxLog, voltLabelDyLog);

legend(axTime, 'Location', 'northwest', 'Interpreter', 'none');

hold(axTime, 'off');

pdfTimeFile = fullfile(figDir, 'ex7_total_time.pdf');
pngTimeFile = fullfile(figDir, 'ex7_total_time.png');

exportgraphics(figTime, pdfTimeFile, 'ContentType', 'vector');
exportgraphics(figTime, pngTimeFile, 'Resolution', 300);
end

%% Print output paths
fprintf('Saved figures:\n');
if plotErrorFigure
    fprintf('  %s\n', pdfErrFile);
    fprintf('  %s\n', pngErrFile);
end
if plotTimingFigure
    fprintf('  %s\n', pdfTimeFile);
    fprintf('  %s\n', pngTimeFile);
end

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

%% Local helper: add an existing tail power-law guide line
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
    xGuide = logspace(log10(Nt(1)), log10(Nt(end)), 80).';
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

    labelIdx = min(max(round(labelPos * numel(xGuide)), 1), numel(xGuide));
    xLabel = 10^(log10(xGuide(labelIdx)) + labelDxLog);
    yLabel = 10^(log10(yGuide(labelIdx)) + labelDyLog);

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
        'FontSize', 12, ...
        'HorizontalAlignment', 'left', ...
        'VerticalAlignment', vAlign, ...
        'Color', 'k', ...
        'Clipping', 'on', ...
        'HandleVisibility', 'off');
end
