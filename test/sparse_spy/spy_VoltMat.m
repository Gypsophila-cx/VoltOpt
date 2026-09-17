%% spy_VoltMat.m
%
% Read data/spy_VoltMat.mat and draw colored and black-and-white spy plots
% for Chebyshev, Legendre, and Jacobi Volterra coefficient matrices.
%
% Colors represent matrix magnitudes after adding machine-epsilon perturbations
% to the same top-dense/banded plotting structure used by the legacy Julia plot.
%
% Output files are written to figures/.
%
% Recommended run from the repository root:
%   matlab -batch "run('test/sparse_spy/spy_VoltMat.m')"

clear; clc; close all;

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

matPath = fullfile(dataDir, 'spy_VoltMat.mat');
if ~isfile(matPath)
    error('Cannot find MAT file: %s', matPath);
end

S = load(matPath);
require_fields(S, {'A_cheb', 'A_leg', 'A_jac'}, matPath);

if isfield(S, 'Nplot')
    nPlot = round(double(S.Nplot(1)));
else
    nPlot = min([size(S.A_cheb, 1), size(S.A_cheb, 2)]);
end

set(groot, 'defaultAxesFontName', 'Times New Roman');
set(groot, 'defaultTextFontName', 'Times New Roman');
set(groot, 'defaultAxesFontSize', paperStyle.TickFontSize);

markerSize = 8;

plot_one_spy( ...
    S.A_cheb, ...
    min(nPlot, min(size(S.A_cheb))), ...
    '$T_n(x)$', ...
    fullfile(figDir, 'spy_VoltMat_chebyshev.pdf'), ...
    fullfile(figDir, 'spy_VoltMat_chebyshev.png'), ...
    markerSize, ...
    true, ...
    2345);

plot_one_spy( ...
    S.A_leg, ...
    min(nPlot, min(size(S.A_leg))), ...
    '$P_n(x)$', ...
    fullfile(figDir, 'spy_VoltMat_legendre.pdf'), ...
    fullfile(figDir, 'spy_VoltMat_legendre.png'), ...
    markerSize, ...
    false, ...
    3456);

plot_one_spy( ...
    S.A_jac, ...
    min(nPlot, min(size(S.A_jac))), ...
    '$P_n^{(0.3,0.2)}(x)$', ...
    fullfile(figDir, 'spy_VoltMat_jacobi.pdf'), ...
    fullfile(figDir, 'spy_VoltMat_jacobi.png'), ...
    markerSize, ...
    true, ...
    1234);

figAll = figure('Color', 'w', 'Units', 'centimeters', ...
    'Position', [2, 2, paperStyle.MultiPanelFigureSize]);
tiledlayout(figAll, 1, 3, 'Padding', 'compact', 'TileSpacing', 'compact');

ax = nexttile;
draw_colored_spy(ax, S.A_cheb, min(nPlot, min(size(S.A_cheb))), ...
                 '$T_n(x)$', markerSize, true, 2345);

ax = nexttile;
draw_colored_spy(ax, S.A_leg, min(nPlot, min(size(S.A_leg))), ...
                 '$P_n(x)$', markerSize, false, 3456);

ax = nexttile;
draw_colored_spy(ax, S.A_jac, min(nPlot, min(size(S.A_jac))), ...
                 '$P_n^{(0.3,0.2)}(x)$', markerSize, true, 1234);

voltopt_export_figure(figAll, findall(figAll, 'Type', 'axes'), ...
    fullfile(figDir, 'spy_VoltMat_all.pdf'), ...
    fullfile(figDir, 'spy_VoltMat_all.png'), 'multipanel');

structurePlots = { ...
    S.A_cheb, 'Chebyshev basis', 'spy_VoltMat_Chebyshev_structure', true,  2345; ...
    S.A_leg,  'Legendre basis',  'spy_VoltMat_Legendre_structure',  false, 3456; ...
    S.A_jac,  'Jacobi basis',    'spy_VoltMat_Jacobi_structure',    true,  1234};

structureNPlot = min([nPlot; cellfun(@(A) min(size(A)), ...
                                     structurePlots(:, 1))]);
for k = 1:size(structurePlots, 1)
    outputStem = structurePlots{k, 3};
    plot_one_structure_spy( ...
        structurePlots{k, 1}, ...
        structureNPlot, ...
        structurePlots{k, 2}, ...
        fullfile(figDir, [outputStem, '.pdf']), ...
        fullfile(figDir, [outputStem, '.png']), ...
        markerSize, ...
        structurePlots{k, 4}, ...
        structurePlots{k, 5});
end

fprintf('Figures saved to:\n  %s\n', figDir);

%% Local helper functions

function require_fields(S, names, fileName)
    missing = {};
    for k = 1:numel(names)
        if ~isfield(S, names{k})
            missing{end + 1} = names{k}; %#ok<AGROW>
        end
    end

    if ~isempty(missing)
        error('Missing required variables in %s: %s', fileName, strjoin(missing, ', '));
    end
end

function plot_one_spy(A, nPlot, titleText, outPdf, outPng, markerSize, addTopDenseRows, seed)
    paperStyle = voltopt_plot_style();
    fig = figure('Color', 'w', 'Units', 'centimeters', ...
        'Position', [2, 2, paperStyle.FigureSize]);
    ax = axes(fig);

    draw_colored_spy(ax, A, nPlot, titleText, markerSize, addTopDenseRows, seed);

    voltopt_export_figure(fig, ax, outPdf, outPng, 'square');

    fprintf('Saved: %s\n', outPdf);
    fprintf('Saved: %s\n', outPng);
end

function plot_one_structure_spy(A, nPlot, titleText, outPdf, outPng, markerSize, addTopDenseRows, seed)
    paperStyle = voltopt_plot_style();
    fig = figure('Color', 'w', 'Units', 'centimeters', ...
        'Position', [2, 2, paperStyle.FigureSize]);
    ax = axes(fig);

    A = A(1:nPlot, 1:nPlot);
    [bandwidth, ~] = detect_bandwidth(A, addTopDenseRows);
    perturbation = add_machine_perturbation(A, bandwidth, addTopDenseRows, seed);
    [row, col] = find(abs(sparse(A)) + perturbation);
    if isempty(row)
        error('The plotted matrix block for %s has no nonzero entries.', titleText);
    end

    scatter(ax, col, row, markerSize, [0, 0, 0], 's', 'filled', ...
            'MarkerEdgeColor', 'none');

    xlabel(ax, 'column index', 'Interpreter', 'latex');
    ylabel(ax, 'row index', 'Interpreter', 'latex');
    title(ax, titleText, 'Interpreter', 'latex');

    xlim(ax, [0.5, nPlot + 0.5]);
    ylim(ax, [0.5, nPlot + 0.5]);
    axis(ax, 'image');

    set(ax, ...
        'Color', 'w', ...
        'XColor', 'k', ...
        'YColor', 'k', ...
        'YDir', 'reverse', ...
        'Box', 'on', ...
        'Layer', 'top', ...
        'FontName', paperStyle.FontName, ...
        'FontSize', paperStyle.TickFontSize, ...
        'LineWidth', paperStyle.AxesLineWidth, ...
        'XGrid', 'off', ...
        'YGrid', 'off');

    voltopt_export_figure(fig, ax, outPdf, outPng, 'square');

    fprintf('Saved: %s\n', outPdf);
    fprintf('Saved: %s\n', outPng);
end

function draw_colored_spy(ax, A, nPlot, titleText, markerSize, addTopDenseRows, seed)
    paperStyle = voltopt_plot_style();
    A = A(1:nPlot, 1:nPlot);

    [bandwidth, ~] = detect_bandwidth(A, addTopDenseRows);
    perturbation = add_machine_perturbation(A, bandwidth, addTopDenseRows, seed);
    Aabs = abs(sparse(A)) + perturbation;

    [row, col, val] = find(Aabs);
    if isempty(val)
        error('The plotted matrix block for %s has no nonzero entries.', titleText);
    end

    floorVal = eps;
    colorVal = log10(max(val, floorVal));
    colorFloor = log10(floorVal);
    colorTop = max(colorVal);
    if colorTop <= colorFloor
        colorTop = colorFloor + 1;
    end

    scatter(ax, col, row, markerSize, colorVal, 's', 'filled', ...
            'MarkerEdgeColor', 'none');

    colormap(ax, parula);
    caxis(ax, [colorFloor, colorTop]);

    cb = colorbar(ax);
    cb.Label.String = '';
    cb.FontName = paperStyle.FontName;
    cb.FontSize = paperStyle.TickFontSize;
    set_colorbar_power_ticks(cb, colorFloor, colorTop);

    xlabel(ax, 'column index', 'Interpreter', 'latex');
    ylabel(ax, 'row index', 'Interpreter', 'latex');
    title(ax, titleText, 'Interpreter', 'latex');

    xlim(ax, [0.5, nPlot + 0.5]);
    ylim(ax, [0.5, nPlot + 0.5]);
    axis(ax, 'image');

    set(ax, ...
        'YDir', 'reverse', ...
        'Box', 'on', ...
        'Layer', 'top', ...
        'FontName', paperStyle.FontName, ...
        'FontSize', paperStyle.TickFontSize, ...
        'LineWidth', paperStyle.AxesLineWidth, ...
        'XGrid', 'off', ...
        'YGrid', 'off');
end

function topRows = detect_raw_top_dense_rows(A)
    densityThreshold = 0.8;
    [nrow, ncol] = size(A);
    topRows = 0;

    for i = 1:nrow
        rowNnz = nnz(A(i, :));
        if rowNnz >= densityThreshold * ncol
            topRows = topRows + 1;
        else
            break;
        end
    end
end

function [bandwidth, rawTopRows] = detect_bandwidth(A, excludeDenseTop)
    defaultBandwidth = 10;
    [nrow, ncol] = size(A);

    if excludeDenseTop
        rawTopRows = detect_raw_top_dense_rows(A);
    else
        rawTopRows = 0;
    end

    [rows, cols] = find(A);
    keep = rows > rawTopRows;
    rows = rows(keep);
    cols = cols(keep);

    if isempty(rows)
        warning('No nonzeros found after excluding raw dense top rows. Use default bandwidth.');
        bandwidth = defaultBandwidth;
        return;
    end

    bandwidth = max(abs(rows - cols));
    bandwidth = max(0, min(round(bandwidth), min(nrow, ncol) - 1));
end

function perturbation = add_machine_perturbation(A, bandwidth, addTopDenseRows, seed)
    [nrow, ncol] = size(A);
    topRows = 0;
    if addTopDenseRows
        topRows = min(bandwidth, nrow);
    end

    maxEntries = topRows * ncol + ncol * (2 * bandwidth + 1);
    I = zeros(maxEntries, 1);
    J = zeros(maxEntries, 1);
    V = zeros(maxEntries, 1);
    count = 0;

    rng(seed, 'twister');

    for j = 1:ncol
        if addTopDenseRows
            for i = 1:topRows
                count = count + 1;
                I(count) = i;
                J(count) = j;
                V(count) = eps * (1.0 + rand());
            end
        end

        if addTopDenseRows
            iMin = max(topRows + 1, j - bandwidth);
        else
            iMin = max(1, j - bandwidth);
        end
        iMax = min(nrow, j + bandwidth);

        if iMin <= iMax
            for i = iMin:iMax
                count = count + 1;
                I(count) = i;
                J(count) = j;
                V(count) = eps * (1.0 + rand());
            end
        end
    end

    perturbation = sparse(I(1:count), J(1:count), V(1:count), nrow, ncol);
end

function set_colorbar_power_ticks(cb, colorFloor, colorTop)
    tickStart = ceil(colorFloor);
    tickStop = floor(colorTop);
    maxTicks = 6;

    if tickStart > tickStop
        ticks = [colorFloor, colorTop];
    else
        tickStep = max(1, ceil((tickStop - tickStart) / max(maxTicks - 1, 1)));
        ticks = tickStart:tickStep:tickStop;
        if ticks(end) ~= tickStop
            ticks = [ticks, tickStop];
        end
    end

    cb.Ticks = ticks;
    labels = cell(numel(ticks), 1);
    for k = 1:numel(ticks)
        labels{k} = sprintf('10^{%d}', round(ticks(k)));
    end
    cb.TickLabels = labels;
    cb.TickLabelInterpreter = 'tex';
end
