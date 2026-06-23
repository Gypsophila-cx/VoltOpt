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
%   The plots use status = success rows. N = 1 failed rows in the error CSV are
%   expected boundary diagnostics because the active solvers require N >= 2.
%
% Run:
%   matlab -batch "run('examples/ex1_warmup/ex1_plot.m')"

clear; clc; close all;

this_dir = fileparts(mfilename('fullpath'));

data_dir = fullfile(this_dir, 'data');
fig_dir  = fullfile(this_dir, 'figures');

if ~exist(fig_dir, 'dir')
    mkdir(fig_dir);
end

error_csv  = fullfile(data_dir, 'ex1_error_N1_100.csv');
timing_csv = fullfile(data_dir, 'ex1_timing_N10_1000_loggrid.csv');

%% Plot settings
method_keys = [
    "VoltMat / Chebyshev"
    "VoltMat / Legendre"
    "GutlebOlver / shifted Jacobi P^(1,0)"
];

method_names = {
    'Duffy--Chebyshev'
    'Duffy--Legendre'
    'Gutleb--Olver'
};

%% =========================
%  Figure 1: Error decay
% ==========================

T = readtable(error_csv);

% Keep successful rows only.
T = T(string(T.status) == "success", :);

labels = string(T.method) + " / " + string(T.basis);

fig1 = figure('Color', 'w');
hold on; box on;

for k = 1:numel(method_keys)
    idx = labels == method_keys(k);
    Tk = sortrows(T(idx, :), 'N');

    x = Tk.N;
    y = Tk.max_abs_error;

    % Avoid semilogy issues if an error is exactly zero.
    y(y <= 0) = realmin;

    semilogy(x, y, ...
        'LineWidth', 1.5);
end

xlabel('$N$', 'Interpreter', 'latex');
ylabel('error', 'Interpreter', 'latex');
ylim([1e-16,1]);

legend(method_names, ...
    'Location', 'northeast', ...
    'Interpreter', 'latex');

set(gca, ...
    'FontName', 'Times New Roman', ...
    'FontSize', 12, ...
    'YScale', 'log');

set(fig1, 'Units', 'inches', 'Position', [1, 1, 6.2, 4.4]);

exportgraphics(fig1, fullfile(fig_dir, 'ex1_error_decay.pdf'), ...
    'ContentType', 'vector');
exportgraphics(fig1, fullfile(fig_dir, 'ex1_error_decay.png'), ...
    'Resolution', 300);

%% =========================
%  Figure 2: End-to-end timing
% ==========================

T = readtable(timing_csv);

% Keep successful rows only.
T = T(string(T.status) == "success", :);

labels = string(T.method) + " / " + string(T.basis);

fig2 = figure('Color', 'w');
hold on; box on; grid on;

for k = 1:numel(method_keys)
    idx = labels == method_keys(k);
    Tk = sortrows(T(idx, :), 'N');

    x = Tk.N;
    y = Tk.median_elapsed_time;

    y(y <= 0) = realmin;

    loglog(x, y, ...
        'LineWidth', 1.5);
end

xlabel('$N$', 'Interpreter', 'latex');
ylabel('time (sec)', 'Interpreter', 'latex');

legend(method_names, ...
    'Location', 'northwest', ...
    'Interpreter', 'latex');

set(gca, ...
    'FontName', 'Times New Roman', ...
    'FontSize', 12, ...
    'XScale', 'log', ...
    'YScale', 'log');

set(fig2, 'Units', 'inches', 'Position', [1, 1, 6.2, 4.4]);

exportgraphics(fig2, fullfile(fig_dir, 'ex1_median_elapsed_time.pdf'), ...
    'ContentType', 'vector');
exportgraphics(fig2, fullfile(fig_dir, 'ex1_median_elapsed_time.png'), ...
    'Resolution', 300);

fprintf('Saved figures:\n');
fprintf('  %s\n', fullfile(fig_dir, 'ex1_error_decay.pdf'));
fprintf('  %s\n', fullfile(fig_dir, 'ex1_error_decay.png'));
fprintf('  %s\n', fullfile(fig_dir, 'ex1_median_elapsed_time.pdf'));
fprintf('  %s\n', fullfile(fig_dir, 'ex1_median_elapsed_time.png'));
