% ex6_plot.m
%
% Purpose:
%   Plot the Example 6 equidistant-grid error curves from the Picard output
%   tables.
%
% Problem:
%   Nonlinear saturated-memory Volterra--Urysohn equation solved by the Example
%   6 Picard scripts.
%
% Input files:
%   data/ex6_fixed10_varyN.csv
%   data/ex6_nonlinear_picard_results.csv
%
% Output files:
%   figures/ex6_fixed10_varyN_eqgrid_error.pdf
%   figures/ex6_fixed10_varyN_eqgrid_error.png
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

fixedFile  = fullfile(dataDir, 'ex6_fixed10_varyN.csv');
picardFile = fullfile(dataDir, 'ex6_nonlinear_picard_results.csv');

if ~isfile(fixedFile)
    error('Cannot find file: %s', fixedFile);
end

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
T_fixed  = readtable(fixedFile);
T_picard = readtable(picardFile);

requiredFixedCols = {'N', 'final_eqgrid_err'};
requiredPicardCols = {'iter', 'eqgrid_err'};

checkColumns(T_fixed, requiredFixedCols, fixedFile);
checkColumns(T_picard, requiredPicardCols, picardFile);

T_fixed  = sortrows(T_fixed, 'N');
T_picard = sortrows(T_picard, 'iter');

%% Figure 1: fixed 10 iterations, varying N
N = T_fixed.N;
eqgridErrFixed = positiveOnly(T_fixed.final_eqgrid_err);

fig1 = figure('Color', 'w');
ax1 = axes(fig1);
hold(ax1, 'on');

semilogy(ax1, N, eqgridErrFixed, ...
    'LineWidth', lineWidth);

hold(ax1, 'off');

set(ax1, 'XScale', 'linear');
set(ax1, 'YScale', 'log');

xlabel(ax1, '$N$');
ylabel(ax1, 'error');
title(ax1, 'Fixed 10 Picard iterations: varying $N$');

box(ax1, 'on');
set(ax1, 'FontSize', fontSize);

xlim(ax1, [min(N), max(N)]);

saveFigureStable(fig1, figDir, 'ex6_fixed10_varyN_eqgrid_error');

%% Figure 2: Picard iteration error history
iter = T_picard.iter;
eqgridErrPicard = positiveOnly(T_picard.eqgrid_err);

fig2 = figure('Color', 'w');
ax2 = axes(fig2);
hold(ax2, 'on');

semilogy(ax2, iter, eqgridErrPicard, ...
    'LineWidth', lineWidth);

hold(ax2, 'off');

set(ax2, 'XScale', 'linear');
set(ax2, 'YScale', 'log');

xlabel(ax2, 'iteration');
ylabel(ax2, 'error');
title(ax2, 'Picard iteration error history');

box(ax2, 'on');
set(ax2, 'FontSize', fontSize);

xlim(ax2, [min(iter), max(iter)]);
xticks(ax2, iter);

saveFigureStable(fig2, figDir, 'ex6_picard_eqgrid_error_history');

fprintf('Figures saved to:\n');
fprintf('  %s\n', fullfile(figDir, 'ex6_fixed10_varyN_eqgrid_error.pdf'));
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

    try
        print(fig, pdfFile, '-dpdf', '-vector', '-bestfit');
    catch
        warning('Painters PDF export failed. Trying OpenGL PDF export instead.');
        set(fig, 'Renderer', 'opengl');
        print(fig, pdfFile, '-dpdf', '-bestfit');
    end

    print(fig, pngFile, '-dpng', '-r300');
end
