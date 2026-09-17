% Plot adaptive compVolt grid selection and reference error.
%
% Run from any working directory with:
%   matlab -batch "run('test/compvolt_adaptive/plot_adaptive_oscillation.m')"

clear; clc; close all;

scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir)
    scriptDir = pwd;
end
addpath(fullfile(fileparts(fileparts(scriptDir)), 'tools'));
paperStyle = voltopt_plot_style();
dataFile = fullfile(scriptDir, 'data', 'adaptive_oscillation.csv');
figureDir = fullfile(scriptDir, 'figures');
if ~isfile(dataFile)
    error('Cannot find adaptive oscillation CSV: %s', dataFile);
end
if ~exist(figureDir, 'dir')
    mkdir(figureDir);
end

T = readtable(dataFile);
required = {'omega', 'accepted_M', 'accepted_npts', 'error_inf'};
missing = setdiff(required, T.Properties.VariableNames);
if ~isempty(missing)
    error('Missing required CSV columns: %s', strjoin(missing, ', '));
end

omega = numeric_column(T, 'omega');
acceptedM = numeric_column(T, 'accepted_M');
acceptedNpts = numeric_column(T, 'accepted_npts');
errorInf = numeric_column(T, 'error_inf');

validGrid = isfinite(omega) & isfinite(acceptedM) & isfinite(acceptedNpts) & ...
    acceptedM > 0 & acceptedNpts == acceptedM + 1;
validError = isfinite(omega) & isfinite(errorInf) & errorInf > 0;
if ~all(validGrid)
    error('CSV contains invalid grid diagnostics.');
end
if ~all(validError)
    error('CSV contains nonpositive or non-finite error_inf values.');
end

set(groot, 'defaultAxesFontName', 'Times New Roman');
set(groot, 'defaultTextFontName', 'Times New Roman');
set(groot, 'defaultLegendFontName', 'Times New Roman');
set(groot, 'defaultAxesFontSize', paperStyle.TickFontSize);
set(groot, 'defaultLineLineWidth', paperStyle.PlotLineWidth);

%% Figure 1: accepted grid degree
fig1 = figure('Color', 'w', 'Units', 'centimeters', ...
    'Position', [2, 2, paperStyle.FigureSize]);
semilogy(omega, acceptedM, '-o', ...
    'LineWidth', paperStyle.PlotLineWidth, ...
    'MarkerSize', 6, ...
    'DisplayName', '$M$ (degree; $M+1$ points)');
box on; grid on;
xlabel('$\omega$', 'Interpreter', 'latex');
ylabel('accepted degree $M$', 'Interpreter', 'latex');
legend('Location', 'northwest', 'Interpreter', 'latex');
ax = gca;
ax.FontName = 'Times New Roman';
ax.FontSize = 12;
ax.YScale = 'log';
ax.YMinorGrid = 'on';
ax.MinorGridLineStyle = ':';
voltopt_export_figure(fig1, ax, ...
    fullfile(figureDir, 'adaptive_accepted_grid.pdf'), ...
    fullfile(figureDir, 'adaptive_accepted_grid.png'));

%% Figure 2: maximum error against the fixed high-resolution reference
fig2 = figure('Color', 'w', 'Units', 'centimeters', ...
    'Position', [2, 2, paperStyle.FigureSize]);
semilogy(omega, errorInf, '-s', ...
    'LineWidth', paperStyle.PlotLineWidth, ...
    'MarkerSize', 6, ...
    'DisplayName', '$\|F_{\mathrm{adaptive}}-F_{\mathrm{ref}}\|_{\infty}$');
box on; grid on;
xlabel('$\omega$', 'Interpreter', 'latex');
ylabel('maximum reference error', 'Interpreter', 'latex');
legend('Location', 'best', 'Interpreter', 'latex');
ax = gca;
ax.FontName = 'Times New Roman';
ax.FontSize = 12;
ax.YScale = 'log';
ax.YMinorGrid = 'on';
ax.MinorGridLineStyle = ':';
voltopt_export_figure(fig2, ax, ...
    fullfile(figureDir, 'adaptive_error_inf.pdf'), ...
    fullfile(figureDir, 'adaptive_error_inf.png'));

fprintf('Saved adaptive compVolt figures to: %s\n', figureDir);

function values = numeric_column(T, name)
    column = T.(name);
    if isnumeric(column)
        values = double(column);
    elseif islogical(column)
        values = double(column);
    else
        values = str2double(string(column));
    end
end
