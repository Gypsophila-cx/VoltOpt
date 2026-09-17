% ex6_plot.m
%
% Purpose:
%   Plot adaptive Picard equidistant-grid error and working-degree history.
%
% Problem:
%   Nonlinear saturated-memory Volterra--Urysohn equation solved by the Example
%   6 Picard scripts.
%
% Input file:
%   data/ex6_nonlinear_picard_results.csv
%
% Output files:
%   figures/ex6_picard_eqgrid_error_history.pdf
%   figures/ex6_picard_eqgrid_error_history.png
%
% Run:
%   matlab -batch "run('examples/ex6_nonlinear/ex6_plot.m')"

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

picardFile = fullfile(dataDir, 'ex6_nonlinear_picard_results.csv');

if ~isfile(picardFile)
    error('Cannot find file: %s', picardFile);
end

%% Plot options
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');

lineWidth  = 1.5;
markerSize = 6;
fontSize   = 12;

%% Read data
T_picard = readtable(picardFile);
requiredPicardCols = {'intermediate_iter', 'iter', 'eqgrid_err', 'N'};
checkColumns(T_picard, requiredPicardCols, picardFile);
T_picard = sortrows(T_picard, 'intermediate_iter');

%% Adaptive Picard intermediate-attempt history
intermediateIter = double(T_picard.intermediate_iter);
outerIter = double(T_picard.iter);
eqgridErrPicard = positiveOnly(T_picard.eqgrid_err);
workingDegree = double(T_picard.N);
newOuterIteration = [true; diff(outerIter) ~= 0];

fig2 = figure('Color', 'w', 'Position', [100, 100, 800, 560]);
ax2 = axes(fig2);
hold(ax2, 'on');

yyaxis(ax2, 'left');
hError = plot(ax2, intermediateIter, eqgridErrPicard, ...
    'LineWidth', lineWidth, 'DisplayName', 'error');
hStarts = plot(ax2, intermediateIter(newOuterIteration), ...
    eqgridErrPicard(newOuterIteration), 'o', ...
    'LineStyle', 'none', 'MarkerSize', markerSize, ...
    'MarkerFaceColor', hError.Color, 'MarkerEdgeColor', hError.Color, ...
    'DisplayName', 'new Picard iteration');
set(ax2, 'YScale', 'log');
ylabel(ax2, 'error');

yyaxis(ax2, 'right');
hDegree = stairs(ax2, intermediateIter, workingDegree, ...
    'LineWidth', lineWidth, 'DisplayName', 'working degree $N$');
set(ax2, 'YScale', 'linear');
ylabel(ax2, 'working degree $N$');

hold(ax2, 'off');
set(ax2, 'XScale', 'linear');
xlabel(ax2, 'intermediate iterations');
box(ax2, 'on');
set(ax2, 'FontSize', fontSize);
xlim(ax2, [min(intermediateIter), max(intermediateIter)]);
if numel(intermediateIter) <= 25
    xticks(ax2, intermediateIter);
end
legend(ax2, [hError, hStarts, hDegree], 'Location', 'best');
set(ax2, 'LooseInset', max(get(ax2, 'TightInset'), 0.04));

saveFigureStable(fig2, figDir, 'ex6_picard_eqgrid_error_history');

fprintf('Figures saved to:\n');
fprintf('  %s\n', fullfile(figDir, 'ex6_picard_eqgrid_error_history.pdf'));

%% Local helper functions
function checkColumns(T, requiredCols, fileName)
    names = T.Properties.VariableNames;
    for k = 1:numel(requiredCols)
        if ~ismember(requiredCols{k}, names)
            error('File %s does not contain required column: %s', ...
                fileName, requiredCols{k});
        end
    end
end

function y = positiveOnly(y)
    y = double(y);
    y(~isfinite(y) | y <= 0) = NaN;
end

function saveFigureStable(fig, figDir, fileBase)
    drawnow;

    pdfFile = fullfile(figDir, [fileBase, '.pdf']);
    pngFile = fullfile(figDir, [fileBase, '.png']);

    set(fig, 'Renderer', 'painters');

    oldUnits = get(fig, 'Units');
    set(fig, 'Units', 'inches');
    figPos = get(fig, 'Position');
    figWidth  = figPos(3);
    figHeight = figPos(4);

    set(fig, 'PaperUnits', 'inches');
    set(fig, 'PaperPositionMode', 'manual');
    set(fig, 'PaperPosition', [0, 0, figWidth, figHeight]);
    set(fig, 'PaperSize', [figWidth, figHeight]);

    try
        print(fig, pdfFile, '-dpdf', '-painters');
    catch
        warning('Painters PDF export failed. Trying OpenGL PDF export instead.');
        set(fig, 'Renderer', 'opengl');
        print(fig, pdfFile, '-dpdf');
    end

    print(fig, pngFile, '-dpng', '-r300');

    set(fig, 'Units', oldUnits);
end
