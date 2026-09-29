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
addpath(fullfile(fileparts(fileparts(scriptDir)), 'tools'));
paperStyle = voltopt_plot_style();

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

lineWidth  = paperStyle.PlotLineWidth;
markerSize = 6;
fontSize   = paperStyle.TickFontSize;

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

fig2 = figure('Color', 'w', 'Units', 'centimeters', ...
    'Position', [2, 2, paperStyle.FigureSize]);
ax2 = axes(fig2);
hold(ax2, 'on');

yyaxis(ax2, 'left');
hError = plot(ax2, intermediateIter, eqgridErrPicard, ...
    'LineWidth', lineWidth, 'DisplayName', 'error');
hStarts = plot(ax2, intermediateIter(newOuterIteration), ...
    eqgridErrPicard(newOuterIteration), 'o', ...
    'LineStyle', 'none', 'MarkerSize', markerSize, ...
    'MarkerFaceColor', hError.Color, 'MarkerEdgeColor', hError.Color, ...
    'DisplayName', 'Picard iteration');
set(ax2, 'YScale', 'log');
ylabel(ax2, 'error');

yyaxis(ax2, 'right');
hDegree = stairs(ax2, intermediateIter, workingDegree, ...
    'LineWidth', lineWidth, 'DisplayName', 'degree $N$');
set(ax2, 'YScale', 'linear');
ylabel(ax2, '$N$');

ax2.XColor = 'k';
ax2.YAxis(1).Color = 'k';
ax2.YAxis(2).Color = 'k';
ax2.YAxis(1).Label.Color = 'k';
ax2.YAxis(2).Label.Color = 'k';

hold(ax2, 'off');
set(ax2, 'XScale', 'linear');
xlabel(ax2, 'intermediate iterations');
box(ax2, 'on');
set(ax2, 'FontSize', fontSize);
xlim(ax2, [min(intermediateIter), max(intermediateIter)]);
xTickStart = ceil(min(intermediateIter) / 2) * 2;
xTickEnd = floor(max(intermediateIter) / 2) * 2;
xticks(ax2, xTickStart:2:xTickEnd);
lgd = legend(ax2, [hStarts, hError, hDegree], ...
    'Location', 'northeast', 'Orientation', 'vertical');
lgd.Box = 'on';
lgd.FontName = paperStyle.FontName;
lgd.FontSize = paperStyle.LegendFontSize;
drawnow;
lgd.Units = 'normalized';
legendPos = lgd.Position;
legendPos(1) = legendPos(1) - 0.02;
legendPos(2) = legendPos(2) - 0.19;
lgd.Position = legendPos;

% Keep the manually positioned legend strictly inside the final axes box.
% Use physical units so the clearance is independent of screen resolution.
oldAxUnits = ax2.Units;
oldLegUnits = lgd.Units;
ax2.Units = 'centimeters';
lgd.Units = 'centimeters';
ax2.PositionConstraint = 'innerposition';
ax2.InnerPosition = paperStyle.AxesInnerPosition;
drawnow;

axBox = ax2.InnerPosition;
legBox = lgd.Position;
marginX = 0.12;
marginY = 0.12;
rightLimit = axBox(1) + axBox(3) - marginX;
topLimit = axBox(2) + axBox(4) - marginY;

if legBox(1) + legBox(3) > rightLimit
    legBox(1) = rightLimit - legBox(3);
end
if legBox(2) + legBox(4) > topLimit
    legBox(2) = topLimit - legBox(4);
end

lgd.Position = legBox;
ax2.Units = oldAxUnits;
lgd.Units = oldLegUnits;
drawnow;

voltopt_export_figure(fig2, ax2, ...
    fullfile(figDir, 'ex6_picard_eqgrid_error_history.pdf'), ...
    fullfile(figDir, 'ex6_picard_eqgrid_error_history.png'));

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
