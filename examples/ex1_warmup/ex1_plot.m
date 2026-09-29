% ex1_plot.m
%
% Purpose:
%   Plot error decay and end-to-end timing from warmup CSV files.
%
% Problem:
%   Low-order VIDE warmup comparison between VoltMat / Chebyshev,
%   VoltMat / Legendre, and Gutleb--Olver shifted Jacobi methods.
%
% Input files:
%   data/ex1_error_N1_100.csv
%   data/ex1_timing_N10_1000_loggrid.csv
%
% Output files:
%   figures/ex1_error_decay.pdf
%   figures/ex1_error_decay.png
%   figures/ex1_median_elapsed_time.pdf
%   figures/ex1_median_elapsed_time.png
%
% Notes:
%   The plots use status = success rows. N = 1 failed rows in the error CSV
%   are expected boundary diagnostics because the active solvers require
%   N >= 2.
%
% Run:
%   matlab -batch "run('examples/ex1_warmup/ex1_plot.m')"

clear; clc; close all;

this_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(fileparts(fileparts(this_dir)), 'tools'));
paperStyle = voltopt_plot_style();

data_dir = fullfile(this_dir, 'data');
fig_dir  = fullfile(this_dir, 'figures');

if ~exist(fig_dir, 'dir')
    mkdir(fig_dir);
end

error_csv  = fullfile(data_dir, 'ex1_error_N1_100.csv');
timing_csv = fullfile(data_dir, ...
    'ex1_timing_N10_1000_loggrid.csv');

%% Plot settings

% Set to true to plot only Duffy--Chebyshev.
% In this mode, no legend is displayed.
only_duffy_chebyshev = false;

% Set to false to plot only Duffy--Chebyshev and Gutleb--Olver.
% Set to true to also include the Duffy--Legendre curve.
% This option is ignored when only_duffy_chebyshev = true.
include_legendre = false;

all_method_keys = [
    "VoltMat / Chebyshev"
    "VoltMat / Legendre"
    "GutlebOlver / shifted Jacobi P^(1,0)"
];

all_method_names = {
    'Duffy--Chebyshev'
    'Duffy--Legendre'
    'Gutleb--Olver'
};

if only_duffy_chebyshev
    active_methods = 1;
elseif include_legendre
    active_methods = [1, 2, 3];
else
    active_methods = [1, 3];
end

method_keys  = all_method_keys(active_methods);
method_names = all_method_names(active_methods);

%% =========================
%  Figure 1: Error decay
% ==========================

T = readtable(error_csv);

% Keep successful rows only.
T = T(string(T.status) == "success", :);

labels = string(T.method) + " / " + string(T.basis);

fig1 = figure('Color', 'w', 'Units', 'centimeters', ...
    'Position', [2, 2, paperStyle.FigureSize]);
hold on;
box on;

for k = 1:numel(method_keys)
    idx = labels == method_keys(k);
    Tk = sortrows(T(idx, :), 'N');

    x = Tk.N;
    y = Tk.max_abs_error;

    % Avoid semilogy issues if an error is exactly zero.
    y(y <= 0) = realmin;

    semilogy(x, y, ...
        'LineWidth', paperStyle.PlotLineWidth);
end

xlabel('$N$', 'Interpreter', 'latex');
ylabel('error', 'Interpreter', 'latex');

ylim([1e-16, 1]);

if ~only_duffy_chebyshev
    legend(method_names, ...
        'Location', 'northeast', ...
        'Interpreter', 'latex');
end

ax1 = gca;
set(ax1, ...
    'FontName', 'Times New Roman', ...
    'FontSize', 12, ...
    'YScale', 'log');

voltopt_export_figure(fig1, ax1, ...
    fullfile(fig_dir, 'ex1_error_decay.pdf'), ...
    fullfile(fig_dir, 'ex1_error_decay.png'));

%% =========================
%  Figure 2: End-to-end timing
% ==========================

T = readtable(timing_csv);

% Keep successful rows only.
T = T(string(T.status) == "success", :);

labels = string(T.method) + " / " + string(T.basis);

fig2 = figure('Color', 'w', 'Units', 'centimeters', ...
    'Position', [2, 2, paperStyle.FigureSize]);
hold on;
box on;

for k = 1:numel(method_keys)
    idx = labels == method_keys(k);
    Tk = sortrows(T(idx, :), 'N');

    x = Tk.N;
    y = Tk.median_elapsed_time;

    % Avoid loglog issues if a timing value is nonpositive.
    y(y <= 0) = realmin;

    loglog(x, y, ...
        'LineWidth', paperStyle.PlotLineWidth);
end

xlabel('$N$', 'Interpreter', 'latex');
ylabel('time (sec)', 'Interpreter', 'latex');

if ~only_duffy_chebyshev
    legend(method_names, ...
        'Location', 'northwest', ...
        'Interpreter', 'latex');
end

ax2 = gca;
set(ax2, ...
    'FontName', 'Times New Roman', ...
    'FontSize', 12, ...
    'XScale', 'log', ...
    'YScale', 'log');

voltopt_export_figure(fig2, ax2, ...
    fullfile(fig_dir, 'ex1_median_elapsed_time.pdf'), ...
    fullfile(fig_dir, 'ex1_median_elapsed_time.png'));

fprintf('Saved figures:\n');
fprintf('  %s\n', ...
    fullfile(fig_dir, 'ex1_error_decay.pdf'));
fprintf('  %s\n', ...
    fullfile(fig_dir, 'ex1_error_decay.png'));
fprintf('  %s\n', ...
    fullfile(fig_dir, 'ex1_median_elapsed_time.pdf'));
fprintf('  %s\n', ...
    fullfile(fig_dir, 'ex1_median_elapsed_time.png'));
